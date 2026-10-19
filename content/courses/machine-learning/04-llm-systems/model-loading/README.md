# Model Loading: Hub to GPU

A model you can download is not yet a model you can serve. Between the files sitting in a Hugging Face Hub repository and a process answering requests on a GPU lies a chain of steps: resolving a revision, reading a byte format, matching thousands of tensor names to module parameters, slicing each tensor for the right GPU, and wiring up a tokenizer and chat template that reproduce what the model saw in training. When any link in that chain is wrong, the symptom is rarely a clean error. More often the model is slow to start, or it loads and then produces garbage, or it produces fluent text that is quietly worse than it should be.

This section takes that chain apart. Every "my custom fine-tune won't load, is slow to start, or gives garbage" ticket at an inference provider reduces to one of four layers: **the repo** (files and metadata), **the bytes** (format, dtype, layout), **the loader** (key mapping, sharding, quantization), or **the tokenizer** (template and special tokens). We work through each layer as math first and then as the code that checks it, ending with a triage playbook you can use on call.

You will get the most from this section if you have read the chapter opening, [LLM Systems & Inference](../README.md), especially its discussion of quantization and multi-GPU parallelism, and if transformer blocks and attention from [Deep Learning](../../02-deep-learning/README.md) are familiar. The quantized formats that appear in checkpoint headers are derived in [Quantization: Math to Code](../quantization/README.md); the runtimes that do the loading are mapped in [Inference Frameworks](../frameworks/README.md) and [LLM Serving Platforms](../serving-platforms.md). The runnable companions live in [`code/`](code/README.md).

> **Note:** This layer of the stack moves monthly. The details here were reviewed in October 2026 against transformers 5.19, huggingface_hub 2.1, safetensors 0.8, vLLM 0.31, and hf_xet 1.7. Pin versions in anything you ship.

> **Learning objectives**
> - Name the files in a Hub model repository and the fields in each that decide whether a model loads and behaves correctly.
> - Read a safetensors header by hand, locally or over HTTP, and verify dtypes, shapes, and sizes without loading any weights.
> - Explain what `from_pretrained` and serving-engine loaders do step by step, including key renaming, fused projections, and tensor-parallel slicing.
> - Estimate cold-start time from checkpoint size and the bandwidth of each link, and choose the technique that attacks the dominant term.
> - Verify a chat template by diffing token ids, and diagnose the common load failures from their symptoms.

## Loading as a matching problem

Before looking at any file, it helps to state what loading *is*. Loading is a **bijection problem under a byte budget**. A checkpoint is a dictionary `{name → (dtype, shape, bytes)}`; a model is a dictionary `{name → parameter slot}`. Loading succeeds when there is a renaming function `f` and a set of tensor operations (cast, transpose, concatenate, slice per rank, dequantize) that map one onto the other with nothing missing and nothing left over. It is fast when those bytes stream at the slowest link's bandwidth with no extra copies. It is correct only when the tokenizer feeds the model the same token ids it saw in training.

Everything in the rest of this section is a specialization of those three statements: the repo and byte format tell you what the left-hand dictionary contains, the loader is the renaming function and its tensor operations, cold-start math is the byte budget, and the tokenizer is the correctness condition that no amount of weight-checking can verify.

## Anatomy of a Hub repo

A Hub model repo is a git repository with large files stored out of band. The storage layer is [Xet](https://huggingface.co/docs/hub/xet/index), the successor to Git LFS, which splits files into content-defined chunks and deduplicates them across repos and revisions. A model, in other words, is **config plus weights plus tokenizer plus generation defaults**, and each of those lives in a different file. The table below lists the files you will meet and the fields a field engineer checks first in each.

| File | What it decides | Fields to check first |
|------|-----------------|---------------------------|
| `config.json` | Model class and shape | `architectures`, `model_type`, `hidden_size`, `num_attention_heads`, `num_key_value_heads`, `head_dim`, `rope_theta`, `rope_scaling`, `max_position_embeddings`, `tie_word_embeddings`, `vocab_size`, `torch_dtype`/`dtype`, `quantization_config`, `auto_map` |
| `generation_config.json` | Default decoding | `eos_token_id` (often a list), `bos_token_id`, `pad_token_id`, `temperature`, `top_p`, `do_sample` |
| `tokenizer.json` | The tokenizer itself (fast, `tokenizers` crate) | `model.type` (BPE/Unigram), `added_tokens`, `pre_tokenizer`, `post_processor` (adds BOS?) |
| `tokenizer_config.json` / `chat_template.jinja` | Special tokens and the chat template | `bos_token`, `eos_token`, `pad_token`, `added_tokens_decoder`, `chat_template` |
| `model.safetensors` or `model-0000k-of-0000n.safetensors` | Weights | dtype per tensor, presence of `lm_head.weight` |
| `model.safetensors.index.json` | Shard map | `metadata.total_size` (bytes), `weight_map` (tensor name → shard file) |
| `README.md` (model card) | License, base model, intended template | YAML front matter: `base_model`, `license`, `pipeline_tag` |
| `LICENSE`, gating | Legal access | Gated repos (`"gated": "manual"`) need an accepted license plus `HF_TOKEN` |
| `adapter_config.json` + `adapter_model.safetensors` | A LoRA adapter, not a full model | `base_model_name_or_path`, `r`, `lora_alpha`, `target_modules` |

Because these concerns are split across files, garbage output is more often a tokenizer or config bug than a weights bug. Four facts about the config deserve emphasis.

First, **`model_type` is the dispatch key.** `AutoModelForCausalLM` looks it up in a registry (`llama` maps to `LlamaForCausalLM`). If the installed library does not know the type, loading fails before a single byte of weights is read.

Second, **`auto_map` means custom code.** It points at `modeling_*.py` files in the repo, which only run with `trust_remote_code=True`. That executes arbitrary Python from the repo, so if you must use it, pin a commit `revision`.

Third, **revisions are commits.** `main` moves; a 40-character sha does not. A customer saying "it worked yesterday" on `main` often means the repo changed underneath them. Resolve the revision once and pin it: `revision="0e9e39f..."`.

Fourth, **the config dtype is a hint, not a guarantee.** `torch_dtype` (renamed `dtype` in transformers v5, though the old key is still read) records the training dtype; the safetensors header records what is actually stored. When they disagree, trust the header.

The real Llama 3.1 8B Instruct config (in [`code/samples/`](code/samples/)) shows three classic traps in eight lines. Its `eos_token_id` is a **list**, `[128001, 128008, 128009]`; its `rope_scaling.rope_type` is `"llama3"`, which older libraries reject; and `num_key_value_heads = 8` is less than `num_attention_heads = 32`, meaning grouped-query attention, which matters when you shard across GPUs with tensor parallelism.

## File formats and dtypes

The config tells you the shape of the model. The weight files tell you what is actually stored, and the format they use determines how fast, and how safely, those bytes can be read.

### The safetensors byte layout

A safetensors file is a JSON header plus a flat byte buffer, nothing more:

```
offset 0        8                    8+N                                   EOF
       ┌────────┬────────────────────┬──────────────────────────────────────┐
       │ N (u64 │ JSON header, UTF-8 │ byte buffer: tensors back to back,   │
       │  LE)   │ padded with ' '    │ row-major, little-endian, no holes   │
       └────────┴────────────────────┴──────────────────────────────────────┘
header = {"model.norm.weight": {"dtype":"BF16","shape":[576],"data_offsets":[b,e]},
          "__metadata__": {"format":"pt"}}          # values must be strings
tensor bytes live at file[8 + N + b : 8 + N + e], and e - b = prod(shape) * bits(dtype) / 8
```

Translating that layout into code takes four lines. This is the core of [`code/safetensors_inspect.py`](code/safetensors_inspect.py):

```python
(n,) = struct.unpack("<Q", f.read(8))       # header length
header = json.loads(f.read(n))              # starts with '{'
b, e = header[name]["data_offsets"]
f.seek(8 + n + b); raw = f.read(e - b)      # zero parsing of the payload
```

The consequence is that you can inspect any checkpoint, local or remote, by reading its first `8 + N` bytes. You never need to load a model to know its dtypes, shapes, and size. Several useful properties follow from the layout.

Because offsets are known up front, a loader can `mmap` the file and hand out tensor views with **zero copies** and no deserialization. Pages fault in lazily on first touch, which is why a "0.3 s load" on a warm page cache becomes 30 s on a cold node. When you benchmark, measure cold, with `sync; echo 3 > /proc/sys/vm/drop_caches`.

The same property enables **remote inspection**: two HTTP `Range` requests read the header of a 100 GB checkpoint. `safetensors_inspect.py --url` does exactly this, so you can verify a customer's dtypes and key names before pulling a byte of weights.

The format is also easy to **validate**. The spec forbids holes and duplicate keys, and `END - BEGIN` must equal `prod(shape) × bits / 8`. A mismatch means a corrupt or hand-edited file.

Finally, large checkpoints are **sharded**. `save_pretrained` writes shards plus `model.safetensors.index.json`. A missing shard is a common partial-upload failure; you catch it by comparing the index's `weight_map` against the shards actually present.

### Why pickle `.bin` is unsafe

The older format, `pytorch_model.bin`, is a zip of Python **pickles**. Unpickling executes the `__reduce__` callables the file names, so loading an untrusted `.bin` is remote code execution ([Python docs warning](https://docs.python.org/3/library/pickle.html)). PyTorch 2.6 made `torch.load(weights_only=True)` the default, which restricts unpickling to tensors and primitives.

> **Warning:** Even with `weights_only=True`, inference platforms should **refuse `.bin` files from customers** and convert them to safetensors in a sandbox. safetensors cannot execute code: it is JSON plus raw bytes.

### Other formats

safetensors is the common case, but you will meet other containers. They differ in structure and, more importantly, in how portable they are, as this comparison shows.

| Format | Structure | Who loads it | Portability |
|--------|-----------|--------------|-------------|
| **safetensors** | JSON header + flat buffer | transformers, vLLM, SGLang, TGI, candle, MLX | Any framework, any hardware |
| **PyTorch `.bin`/`.pt`** | Zip of pickles | Legacy PyTorch | Unsafe, Python-only |
| **GGUF** | Magic `GGUF`, version, KV metadata (arch, tokenizer, `tokenizer.chat_template`), tensor infos, aligned data | llama.cpp, Ollama, LM Studio, mistral.rs, candle | One file holds weights **and** tokenizer **and** template; k-quant block types |
| **ONNX** | Protobuf graph + initializers (external data > 2 GB) | ONNX Runtime, DirectML, edge | Graph is frozen; dynamic shapes need care |
| **TensorRT engine** | Serialized, compiled kernels | TensorRT / TensorRT-LLM | Tied to GPU arch, TRT version, and build-time max shapes; rebuild per SKU |
| **Orbax checkpoint** | Directory of array shards (TensorStore/OCDBT) | JAX, Flax NNX, MaxText | Sharding-aware restore onto a device mesh |
| **Keras `.weights.h5` / preset dir** | HDF5 or preset folder | Keras 3, KerasHub | Backend-agnostic (JAX, PyTorch, TF) |

### Dtypes you will see in a header

Each tensor in a safetensors header carries a dtype string. The bit layout of each dtype determines its range, and range is where many silent numerical bugs come from. Here are the ones you will see.

| safetensors dtype | Bits (sign/exp/mantissa) | Max finite | Where it appears |
|-------------------|--------------------------|------------|------------------|
| `F32` | 1/8/23 | 3.4e38 | Norms in some checkpoints, optimizer state |
| `F16` | 1/5/10 | **65504** | Older checkpoints, AWQ/GPTQ scales |
| `BF16` | 1/8/7 | 3.4e38 | Default for modern LLM weights |
| `F8_E4M3` | 1/4/3 | 448 | FP8 weights (DeepSeek-V3, `compressed-tensors`, ModelOpt) |
| `F8_E5M2` | 1/5/2 | 57344 | Gradients, some KV caches |
| `F8_E8M0` | 0/8/0 (power of two) | 2^127 | MX block scales |
| `F4` (E2M1) | 1/2/1 | 6 | MXFP4 / NVFP4 elements |
| `U8`, `I32` | containers | n/a | Packed low-bit weights: GPTQ/AWQ `qweight` (8 × int4 per I32), gpt-oss MXFP4 `*_blocks` (2 × fp4 per U8) |

Low-bit formats store values in blocks that share a scale. **MXFP4** (the OCP Microscaling format) groups 32 E2M1 values under one E8M0 scale, for about 4.25 bits per weight. **NVFP4** (Blackwell) groups 16 E2M1 values under an FP8 E4M3 scale, plus a per-tensor FP32 scale. In a header these appear as a `U8` or `F4` tensor plus a `*_scales` tensor, so for quantized checkpoints **element count is not parameter count**. DeepSeek-V3, for example, stores `F8_E4M3` weights with a `weight_scale_inv` tensor per 128×128 block. The quantizers themselves are derived in [Quantization: Math to Code](../quantization/README.md).

## Downloading and the cache

Knowing what the files contain, the next step is getting them onto a machine. The Python API offers a one-file download and a whole-revision snapshot; note the pinned commit and the `allow_patterns` filter, which skips `.bin`, GGUF, and `original/` files you do not need:

```python
from huggingface_hub import hf_hub_download, snapshot_download

cfg = hf_hub_download("Qwen/Qwen3-30B-A3B", "config.json")          # one file
path = snapshot_download(                                           # a whole revision
    "meta-llama/Llama-3.1-8B-Instruct",
    revision="0e9e39f249a16976918f6564b8830bc894c89659",           # pin a commit
    allow_patterns=["*.json", "*.safetensors", "tokenizer*"],       # skip .bin, GGUF, original/
)
```

The command line does the same. The CLI is **`hf`**; `huggingface-cli` is the deprecated name ([CLI guide](https://huggingface.co/docs/huggingface_hub/guides/cli)).

```bash
hf auth login                                   # or export HF_TOKEN=hf_...
hf download meta-llama/Llama-3.1-8B-Instruct --include "*.safetensors" --include "*.json" \
    --revision 0e9e39f249a16976918f6564b8830bc894c89659 --local-dir /mnt/nvme/llama31-8b
hf cache ls                                     # what is cached, how big
hf cache verify meta-llama/Llama-3.2-1B-Instruct   # checksum against the Hub
HF_HUB_OFFLINE=1 python serve.py                # never touch the network
```

Downloads land in a content-addressed cache under `$HF_HOME/hub` (default `~/.cache/huggingface/hub`, overridable with `HF_HUB_CACHE`). Its layout explains much of its behavior:

```
models--meta-llama--Llama-3.1-8B-Instruct/
├── blobs/<sha256 or etag>          # actual bytes, content-addressed
├── refs/main                       # text file: commit sha that "main" resolved to
├── snapshots/<commit-sha>/
│   ├── config.json -> ../../blobs/…
│   └── model-00001-of-00004.safetensors -> ../../blobs/…
└── trees/<commit-sha>.json         # cached file list for that commit
```

Snapshots are directories of **symlinks into `blobs/`**, so two revisions sharing a shard point at one blob. On filesystems without symlinks (some network mounts, Windows without developer mode) files are copied instead, and disk use multiplies. Recent `huggingface_hub` also shares Xet blobs **across repos**, so fine-tunes that reuse base shards cost no extra disk. When you want plain files instead of this layout, for instance when baking weights into an image or a volume, pass `--local-dir`.

For speed, `hf_xet` (installed with `huggingface_hub`) replaced `hf_transfer`. Set `HF_XET_HIGH_PERFORMANCE=1` on machines with big NICs to raise concurrency, and remember that requests without `HF_TOKEN` are rate-limited. For reproducibility, `HF_HUB_OFFLINE=1` makes every call resolve from cache; with a pinned sha, a fully cached load makes **zero** network calls, and an incomplete snapshot raises `IncompleteSnapshotError` rather than silently returning a partial folder.

All of this leads to a production rule: **the Hub is a distribution channel, not a serving dependency.** Mirror pinned revisions once into your own object store (`s3://models/<org>/<name>/<sha>/`), checksum them, and load from there or from local NVMe. The reasons are rate limits and outages, gated-token sprawl, revision drift, and egress cost across thousands of replicas.

## What `from_pretrained` actually does

With the files local, the loader takes over. `from_pretrained` looks like one call, but it runs seven distinct stages:

```
from_pretrained(repo_or_dir, dtype="auto", device_map="auto")
 1. resolve files       snapshot / local dir; read config.json (+ quantization_config)
 2. pick class          config.model_type → AutoModel mapping → LlamaForCausalLM
                        (auto_map + trust_remote_code → repo's modeling_*.py instead)
 3. skeleton on meta    build modules on the "meta" device: shapes, no memory
 4. plan placement      device_map="auto" (accelerate): fill GPU 0..n, then CPU, then disk
 5. stream state dict   for each shard: rename keys → convert (fuse/split/dequant) → shard (TP)
                        → cast to dtype → materialize on target device (4 threads by default)
 6. finalize            tie weights (lm_head ← embed_tokens if tie_word_embeddings),
                        init any missing params, report missing / unexpected / mismatched keys
 7. generation config   load generation_config.json → model.generation_config
```

In transformers v5 the argument is **`dtype`** (`torch_dtype` is the legacy name), and the default now follows the checkpoint's config dtype instead of upcasting to FP32. Step 5 is the [dynamic weight loader](https://huggingface.co/docs/transformers/main/en/weightconverter). As tensors stream in, `WeightRenaming` rules (for example `LayerNorm.gamma → LayerNorm.weight`) and `WeightConverter` rules (for example, stacking Mixtral's per-expert `w1`/`w3` into one `experts.gate_up_proj` with `MergeModulelist` plus `Concatenate`) transform them. These rules are **reversible**, so `save_pretrained` writes the original layout back. Peak memory is approximately the model size plus the largest merge, which is one MoE layer's experts.

Seen this way, `from_pretrained` is a key-matching problem: map checkpoint names to module parameters, then fill them. Prefix drift, tied weights, and fused projections cause most load failures. Three details of the stages matter in practice.

Step 3's **meta device** removes the "two copies" problem. Without it you would allocate random weights and then the checkpoint, doubling peak RAM.

Step 6's **load report** must be read. "Some weights were not initialized from the checkpoint" with `lm_head.weight` listed is not a warning to ignore; it means the model has a random output head.

Step 2's **`trust_remote_code`** path executes repo Python at load time. Inference platforms run it, if at all, in a sandbox at conversion time, never in the serving process. Note also that custom-code models skip transformers' built-in conversion mappings.

### Failure modes

The stages above fail in recognizable ways. This table maps each symptom to its root cause, the check that confirms it, and the fix; it is the backbone of the triage playbook at the end of the section.

| Symptom | Root cause | Check | Fix |
|---------|-----------|-------|-----|
| "weights not initialized" for every layer; unexpected keys start with `module.`, `_orig_mod.`, `base_model.model.` | Saved from a DDP / `torch.compile` / PEFT wrapper | Diff header keys against `model.state_dict().keys()` | Strip the prefix; save from the unwrapped model; `merge_and_unload()` for PEFT |
| Output is random text from token 1 | `lm_head.weight` missing and `tie_word_embeddings=false` (or present but tied flag wrong) | Is `lm_head.weight` in `weight_map`? | Set `tie_word_embeddings` to match the checkpoint |
| Fine at 4K tokens, degrades past 8K | `rope_scaling` dropped or rewritten by a fine-tune toolkit; or old library ignores `rope_type: llama3`/`yarn` | Diff `rope_scaling` and `rope_theta` against the base config | Restore the base values; upgrade the library |
| Fluent but worse than base, ignores system prompt | Chat template differs from training | Render both templates, diff token ids (see [Tokenizers and chat templates](#tokenizers-and-chat-templates)) | Ship the training template in `tokenizer_config.json` / `chat_template.jinja` |
| Never stops; prints `assistant` turns forever | EOS mismatch: instruct turn ends with `<\|eot_id\|>` (128009) but only `<\|end_of_text\|>` (128001) is an EOS | `generation_config.eos_token_id` vs template's turn terminator | Make `eos_token_id` a list containing every terminator |
| NaN / inf or garbage only in FP16 | BF16-trained activations exceed 65504 | Same prompt in BF16 is fine | Serve BF16 (or FP8 with scales), never FP16, for BF16-trained models |
| `size mismatch for embed_tokens: [128264, 4096] vs [128256, 4096]` | Tokens added in fine-tune, or vocab padded to a multiple of 64 | `len(tokenizer)` vs `config.vocab_size` vs header shape | `resize_token_embeddings(len(tok))` before saving; set `vocab_size` to the stored rows |
| `model type X not recognized` | Library older than the architecture, or custom code | `transformers.__version__`; `auto_map` in config | Upgrade, or pinned-revision `trust_remote_code` in a sandbox |
| Quality collapse on a quantized fine-tune | `quantization_config` stale (fine-tuned in BF16, config still says GPTQ), or `ignore` list missing `lm_head` | Header dtypes vs `quantization_config` | Requantize from the merged BF16 weights |

## How serving engines load

transformers is the reference loader, but production traffic goes through serving engines, and they load differently. vLLM and SGLang keep their own **model registries**, mapping an architecture string to an engine-native implementation with fused kernels, and their own loaders. They read the same `config.json` and safetensors files but never call transformers' `from_pretrained` for supported architectures. An unsupported `architectures` entry either falls back to the Transformers backend, which is slower, or fails.

### Name mapping and fused projections

Engines fuse `q_proj`, `k_proj`, and `v_proj` into one `qkv_proj` GEMM, and `gate_proj` and `up_proj` into one `gate_up_proj`, because one large matrix multiply is faster than three small ones. That means the checkpoint's names no longer match the engine's parameters one to one, and the engine must declare how to map them. vLLM's Llama implementation does it like this ([source](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama.py)):

```python
hf_to_vllm_mapper = WeightsMapper(
    orig_to_new_stacked={
        # weight_name: (param_name, shard_id)
        ".q_proj": (".qkv_proj", "q"),
        ".k_proj": (".qkv_proj", "k"),
        ".v_proj": (".qkv_proj", "v"),
        ".gate_proj": (".gate_up_proj", 0),
        ".up_proj": (".gate_up_proj", 1),
    }
)
```

Each parameter carries a `weight_loader(param, loaded_weight, shard_id)` that knows where in the fused tensor, and which rows for this rank, the incoming tensor goes. A customer checkpoint that **already** fused `qkv_proj`, which is common from custom training code, misses this mapping entirely. The fix is to split it back to the Hugging Face names, not to patch the engine.

### Tensor-parallel slicing math

The `weight_loader` also slices each tensor for its rank when the model runs with tensor parallelism. PyTorch stores `nn.Linear` weights as $W \in \mathbb{R}^{\text{out} \times \text{in}}$. With $t$ ranks, there are three slicing patterns:

```
ColumnParallel (q, k, v, gate, up):  W_r = W[r·out/t : (r+1)·out/t, :]      output split, no comm
RowParallel    (o_proj, down_proj):  W_r = W[:, r·in/t : (r+1)·in/t]        partial sums → all-reduce
Vocab-parallel embedding / lm_head:  rows r·V'/t … where V' = V padded up to a multiple of 64 (vLLM)
```

Attention adds a constraint. For grouped-query attention with $H$ query heads and $KV$ key/value heads, rank $r$ gets query heads $[rH/t, (r+1)H/t)$ and KV heads $[r \cdot KV/t, (r+1) \cdot KV/t)$. That requires $H \bmod t = 0$. If $KV < t$, each KV head is **replicated** across $t/KV$ ranks, which requires $t \bmod KV = 0$. Llama 3.1 8B ($H = 32$, $KV = 8$) therefore shards cleanly to $t \in \{1, 2, 4, 8\}$, while Qwen3-30B-A3B ($KV = 4$) at $t = 8$ replicates each KV head twice.

Fused tensors need extra care. The fused `gate_up_proj` $= [G; U]$ must be sliced **per sub-matrix**: rank $r$ gets $[G_r; U_r]$, not the contiguous $r$-th chunk of the fused tensor, which would hand rank 0 all of $G$. transformers' TP plan calls this `packed_colwise`.

> **Warning:** Slicing a fused tensor as one contiguous block loads without error and produces garbage. This is why checkpoints that arrive pre-fused are dangerous.

### Pre-quantized checkpoints

Quantized checkpoints add one more dispatch. Engines read `config.json → quantization_config.quant_method` and swap in the matching linear method and kernel. The table shows what each method stores and which vLLM kernel path it takes.

| `quant_method` | Stored tensors | Kernel path (vLLM) |
|----------------|----------------|--------------------|
| `awq` | `qweight` (I32, 8×int4), `qzeros`, `scales` (F16), group 128 | AWQ Marlin on Ampere+ |
| `gptq` | `qweight`, `qzeros`, `scales`, `g_idx` (act-order) | GPTQ Marlin / Machete |
| `fp8` | `F8_E4M3` weight + `weight_scale_inv` (block 128×128) or per-tensor scale | CUTLASS FP8 GEMM (Hopper+, Ada) |
| `compressed-tensors` | Per-group `config_groups` (W4A16, W8A8, FP8, NVFP4) + `ignore` list | Marlin / CUTLASS by scheme ([llm-compressor](https://github.com/vllm-project/llm-compressor) output) |
| `modelopt` | NVIDIA ModelOpt FP8 / NVFP4 | Blackwell FP4 tensor cores |
| `mxfp4` | gpt-oss experts as `*_blocks` (U8) + `*_scales` (E8M0) | MXFP4 MoE kernels |

The `--quantization` CLI flag overrides detection, but you usually should not pass it. The common ticket here is "my quantized fine-tune is slow", and the cause is usually that the GPU lacks the fast kernel (FP8 on Ampere falls back to weight-only Marlin), not that loading failed.

### LoRA adapters: merge or serve

A LoRA adapter is not a full model. It is `adapter_config.json` (`r`, `lora_alpha`, `target_modules`, `base_model_name_or_path`) plus `adapter_model.safetensors`, with keys like `base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight`. The math is a low-rank update to each adapted matrix:

$$W' = W + \frac{\alpha}{r} BA, \qquad A \in \mathbb{R}^{r \times \text{in}},\ B \in \mathbb{R}^{\text{out} \times r}$$

(rsLoRA scales by $\alpha / \sqrt{r}$ instead). Each adapted matrix adds $r(\text{in} + \text{out})$ parameters; for $r = 16$ on a $4096 \times 4096$ matrix that is $16 \times 8192 = 131{,}072$, or 0.8% of the matrix's 16.8 M.

You can either merge the update into the base weights or serve the adapter separately, and the choice is a real tradeoff:

| | **Merge** (`merge_and_unload()`, then serve as a full model) | **Serve unmerged** (`vllm serve base --enable-lora --lora-modules name=path`) |
|---|---|---|
| Latency | Zero overhead | Extra `2·r·(in+out)` FLOPs/token per adapted matrix; batched SGMV/Punica kernels |
| Memory | Full copy per fine-tune | One base, many adapters (multi-tenant) |
| Cold start | Full model load | Adapter is MBs: seconds |
| Gotchas | Merging into a quantized base needs dequant → merge → requant | `r` > `--max-lora-rank` (default 16) rejects; adapters that also train `embed_tokens`/`lm_head` (added tokens) may not be supported; base revision must match training |

The rule of thumb: serve unmerged for many low-traffic fine-tunes, and merge for one high-traffic fine-tune where every microsecond of time per output token matters.

## Cold-start math

So far we have asked whether a model loads correctly. The other half of the problem is how fast. Cold start is a bandwidth equation, and writing it out term by term shows where the time goes:

$$T_\text{cold} = T_\text{schedule} + T_\text{fetch} + T_\text{read} + T_\text{h2d} + T_\text{init}$$

Here $T_\text{fetch} \approx S / B_\text{net}$, paid only if weights are not already on the node. Reading from disk and copying host-to-device cost $T_\text{read} + T_\text{h2d} \approx S / \min(B_\text{disk}, B_\text{pcie})$ when the two are streamed and overlapped, and their sum otherwise. $T_\text{init}$ covers the CUDA context, KV-cache profiling, CUDA graph capture, and `torch.compile` if used. The size is $S = \text{params} \times \text{bytes/param}$, or $S/t$ per rank if checkpoints are pre-sharded. Ignoring scheduling and init, the core of it is $T \approx S / \min(B_\text{net}, B_\text{disk}, B_\text{pcie})$.

Plugging in real numbers makes the lesson concrete. The table uses BF16 sizes: the 8B model is 16.06 GB (from the Llama 3.1 index), and a 70B model with 70.55 B parameters is 141.1 GB.

| Path | Effective bandwidth | 8B (16.1 GB) | 70B (141 GB) |
|------|--------------------|--------------|--------------|
| Hub or S3, single HTTP stream | ~0.2 GB/s | 80 s | 12 min |
| Object store, parallel range reads, 25 GbE | ~3 GB/s | 5.4 s | 47 s |
| Object store, parallel, 100 GbE | ~10 GB/s | 1.6 s | 14 s |
| Local NVMe Gen4 (one drive) | ~6.5 GB/s | 2.5 s | 22 s |
| NVMe RAID-0 ×4 or page cache | ~25 GB/s | 0.6 s | 5.6 s |
| PCIe Gen5 x16 host → one GPU | ~50 GB/s | 0.3 s | 2.8 s (÷ t with TP, each GPU has its own link) |

Two lessons fall out. The network and disk terms dominate by 10 to 100 times, so **cache placement beats loader tuning**: production loads from local NVMe or parallel object-store reads, never from the Hub at request time. And once weights are local, $T_\text{init}$ (graph capture and compilation, often 20 to 60 s for large models) becomes the floor, which is why the last techniques below snapshot GPU state instead of reloading.

Each technique for faster loading attacks a specific term of the equation:

| Technique | Attacks | Mechanism |
|-----------|---------|-----------|
| safetensors `mmap` | Copies | Lazy page-in, no deserialization; worst case on network filesystems (random 4 KB faults) |
| [fastsafetensors](https://github.com/foundation-model-stack/fastsafetensors) (`--load-format fastsafetensors`) | `T_read + T_h2d` | Batched reads and GPUDirect Storage straight into GPU memory |
| [Run:ai Model Streamer](https://github.com/run-ai/runai-model-streamer) (`--load-format runai_streamer`) | `T_fetch` | Concurrent reads from S3/GCS/Azure/local into a CPU buffer, overlapped with H2D; `--model-loader-extra-config '{"concurrency":16}'` |
| [tensorizer](https://github.com/coreweave/tensorizer) (`--load-format tensorizer`) | `T_fetch` | Serialize once, stream from HTTP/S3 at line rate |
| Pre-sharded state (`sharded_state`) | `T_read` per rank | Save each TP rank's slice once; each rank reads `S/t` with no slicing |
| Local NVMe cache / warm pool | `T_fetch` | Daemonset pre-pulls pinned revisions; scale from warm nodes |
| GPU snapshotting ([cuda-checkpoint](https://github.com/NVIDIA/cuda-checkpoint) + CRIU, platform GPU memory snapshots) | `T_init` | Restore a process with weights and CUDA graphs already resident |

[`code/config_explain.py`](code/config_explain.py) prints this estimate for any `config.json`.

<a id="7-tokenizers-and-chat-templates"></a>

## Tokenizers and chat templates

A model can load correctly and start fast and still be wrong, because the last layer, the tokenizer, decides what the model actually sees. Four tokenizer families cover nearly every model you will serve, and each leaves a tell-tale sign in the repo:

| Family | Algorithm | Examples | Tell-tale |
|--------|-----------|----------|-----------|
| **BPE** ([Sennrich et al.](https://arxiv.org/abs/1508.07909)) | Greedy merges of frequent pairs from a merges table | GPT-2 lineage | `merges` in `tokenizer.json` |
| **Byte-level BPE** | BPE over bytes, so no unknown tokens | GPT-2, Llama 3 (128,256 vocab), Qwen | `Ġ` marks a leading space |
| **SentencePiece** ([repo](https://github.com/google/sentencepiece)) | Unigram LM or BPE over raw text, whitespace as `▁` | Llama 2, Gemma, T5 | `▁` prefix; `tokenizer.model` file |
| **tiktoken** ([repo](https://github.com/openai/tiktoken)) | Byte-level BPE with regex pre-split, fast Rust core | OpenAI `cl100k_base`, `o200k_base`; gpt-oss `o200k_harmony` | Ranks file, no merges table |

On top of the tokenizer sits the **chat template**: a Jinja program stored in `tokenizer_config.json` (as `chat_template`) or in `chat_template.jinja`. It turns a list of `{"role", "content"}` messages into the exact string the model was trained on. Two models render the same one-message conversation very differently:

```
SmolLM2:  <|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n
Llama 3 (simplified; 3.1 also injects a dated system header):
          <|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
```

Chat templates fail silently. With the wrong template the model still sees valid tokens, so its output is fluent. It is simply conditioned on an out-of-distribution prefix: system prompts get ignored, tool calls come out malformed, refusals spike, and evals drop a few points. Nothing errors. The only proof of correctness is to diff the token ids the serving path produces against the ones training saw. Do this on every custom-model onboarding:

```python
ids_serving = tok.apply_chat_template(msgs, add_generation_prompt=True)     # what the engine sends
ids_training = tok(training_formatter(msgs), add_special_tokens=False).input_ids  # what SFT saw
assert ids_serving == ids_training, first_divergence(ids_serving, ids_training)
```

The divergences this check catches are a familiar list. The most common is a **double BOS**: the template already emits `<|begin_of_text|>`, then the rendered string is tokenized again with `add_special_tokens=True`, and the post-processor adds a second one. Others are a missing `add_generation_prompt`, a fine-tune that added special tokens (`added_tokens`) the serving tokenizer lacks, trailing whitespace or newline differences, and a template that drops the system role. Keep in mind that vLLM uses the repo's template unless `--chat-template` overrides it, and that `--generation-config auto` pulls sampling defaults from `generation_config.json`, so both files are part of the deployed contract.

## Beyond PyTorch: JAX, Flax NNX, and Keras 3

Everything so far assumed PyTorch, but transformers v5 dropped its TensorFlow and Flax backends, so the JAX world loads through its own stack. The pattern is the same as the meta-device skeleton, expressed in Flax NNX and Orbax: build an abstract model with shapes only, then restore into it.

```python
# Flax NNX + Orbax: build an abstract model (shapes only), then restore into it
abstract = nnx.eval_shape(lambda: Transformer(cfg, rngs=nnx.Rngs(0)))
graphdef, abstract_state = nnx.split(abstract)
state = ocp.StandardCheckpointer().restore(ckpt_dir / "state", abstract_state)
model = nnx.merge(graphdef, state)
```

`nnx.eval_shape` plays the role of PyTorch's meta device, and Orbax restores each array **directly onto its target sharding** across a device mesh, with no host-side gather; this is how MaxText and TPU serving load. Converting Hugging Face safetensors into JAX means the same key renaming plus transposes, because `nn.Linear` stores `out × in` while Flax `Dense` kernels store `in × out`.

**Keras 3 and KerasHub** presets are backend-agnostic. `keras_hub.models.CausalLM.from_preset("hf://<org>/<repo>", dtype="bfloat16")` converts supported Hugging Face architectures (Llama 3, Gemma, Mistral, Mixtral, Qwen, gpt-oss, and others in [`utils/transformers/`](https://github.com/keras-team/keras-hub/tree/master/keras_hub/src/utils/transformers)) on load and runs them on JAX, PyTorch, or TensorFlow.

## A triage playbook

The four layers give you a diagnostic order. When a customer reports a problem, start from their words, run the first command, and check the most likely causes in order:

| Customer says | First command | Most likely cause |
|---------------|---------------|-------------------|
| "Won't load" | `safetensors_inspect.py <index.json>` and diff keys vs base model | Prefix drift, missing shard, fused or renamed projections, unknown `model_type` |
| "Loads but OOMs" | `config_explain.py config.json` | Weights + KV cache at max context exceed GPU; dtype upcast to FP32 |
| "Slow to start" | Time each term of the cold-start equation separately | Pulling from the Hub per replica; network filesystem with mmap; no warm cache |
| "Gives garbage" | Same prompt in transformers BF16 vs engine | Missing `lm_head`, FP16 overflow, TP slicing of fused tensors, quant config mismatch |
| "Worse than the base model" | Diff templated token ids vs training | Chat template, double BOS, EOS list, sampling defaults |
| "Never stops" | Print `generation_config.eos_token_id` | Turn terminator not in EOS list |

The best way to internalize this table is to break a load on purpose. Run [`code/safetensors_inspect.py`](code/safetensors_inspect.py) with `--url` against three Hub models (one with tied embeddings, one FP8, one MXFP4) and explain every tensor name you see. Then rename keys with a `module.` prefix, drop `lm_head.weight`, delete `rope_scaling`, and swap the chat template, recording the symptom of each. The table you produce is your on-call runbook. Finally, read one real loader end to end: vLLM's [`llama.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama.py), following `hf_to_vllm_mapper` and `AutoWeightsLoader`.

## Where this leads

Every tensor name in a checkpoint corresponds to a piece of the transformer block from [Deep Learning](../../02-deep-learning/README.md): attention projections, GQA heads, MoE experts, RoPE parameters. The FP8, MXFP4, NVFP4, AWQ, and GPTQ tensors you learned to recognize here are derived in [Quantization: Math to Code](../quantization/README.md). LoRA adapters and chat templates come from fine-tuning and SFT data formatting, covered in [Training & Post-Training](../../07-training-and-post-training/README.md). Getting weights onto nodes through image baking, daemonsets, and warm pools is the subject of [Cloud Native](../../../systems/03-cloud-native/README.md) in *Systems and Architecture*, and measuring each cold-start term alongside time to first token belongs to [Observability](../../../systems/04-observability/README.md). Once a model loads, [Serving & Load](../serving-and-load/README.md) takes up what happens when traffic arrives.

This is daily work at inference providers. At Fireworks, Together, and Baseten, field engineers debug customer checkpoints, LoRA multi-tenancy, and cold-start SLAs; Hugging Face owns the Hub storage, safetensors, and the transformers loader; NVIDIA ships ModelOpt FP8 and NVFP4 checkpoints, TensorRT-LLM engine builds, and GPUDirect Storage; Modal, Replicate, and RunPod compete on scale-to-zero, GPU memory snapshots, and weight caching; Anyscale and Databricks run vLLM-based serving from model registries and object-store streaming; and Google builds on Orbax, MaxText, KerasHub, and TPU serving.

### Further reading

- [safetensors format spec](https://github.com/huggingface/safetensors#format)
- [Transformers: Loading models](https://huggingface.co/docs/transformers/main/en/models) and [Dynamic weight loading](https://huggingface.co/docs/transformers/main/en/weightconverter)
- [huggingface_hub: Download files](https://huggingface.co/docs/huggingface_hub/guides/download) and [Understand caching](https://huggingface.co/docs/huggingface_hub/guides/manage-cache)
- [vLLM engine arguments](https://docs.vllm.ai/en/latest/configuration/engine_args.html)
- [Transformers: Chat templates](https://huggingface.co/docs/transformers/main/en/chat_templating)
- [GGUF spec](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md), [Hub Xet storage](https://huggingface.co/docs/hub/xet/index), [`hf` CLI guide](https://huggingface.co/docs/huggingface_hub/guides/cli)
- [vLLM Run:ai Model Streamer](https://docs.vllm.ai/en/latest/models/extensions/runai_model_streamer.html), [fastsafetensors](https://github.com/foundation-model-stack/fastsafetensors), [CoreWeave tensorizer](https://github.com/coreweave/tensorizer), [NVIDIA cuda-checkpoint](https://github.com/NVIDIA/cuda-checkpoint)
- [PEFT LoRA docs](https://huggingface.co/docs/peft) and the [LoRA paper](https://arxiv.org/abs/2106.09685)
- [FP8 formats paper](https://arxiv.org/abs/2209.05433) and the [OCP Microscaling (MX) spec](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [Flax NNX checkpointing](https://flax.readthedocs.io/en/latest/guides/checkpointing.html) and [KerasHub](https://keras.io/keras_hub/)

## Key terms

- **safetensors**: a weight format consisting of an 8-byte little-endian header length, a JSON header of tensor names, dtypes, shapes, and offsets, and a flat byte buffer.
- **revision**: a commit sha in a Hub repo; pinning one makes a load reproducible, while `main` can change.
- **meta device**: a PyTorch device that records tensor shapes without allocating memory, used to build a model skeleton before filling it from a checkpoint.
- **fused projection**: an engine-side parameter such as `qkv_proj` or `gate_up_proj` that concatenates several checkpoint tensors into one GEMM.
- **column- and row-parallel**: the two tensor-parallel slicing patterns, splitting a weight's output rows (no communication) or its input columns (partial sums followed by an all-reduce).
- **LoRA adapter**: a low-rank update $W' = W + (\alpha/r)BA$ stored separately from the base model.
- **cold start**: the time from scheduling a replica to serving its first request, dominated by fetching and reading weight bytes and then by engine initialization.
- **chat template**: a Jinja program that renders a list of chat messages into the exact token sequence the model was trained on.

## Summary

- A model is config plus weights plus tokenizer plus generation defaults, each in a different file; garbage output is more often a tokenizer or config bug than a weights bug.
- safetensors is a JSON header plus a flat byte buffer, so you can inspect any checkpoint, even remotely, by reading its first `8 + N` bytes.
- `from_pretrained` and engine loaders are key-matching problems; prefix drift, tied weights, fused projections, and tensor-parallel slicing of fused tensors cause most failures.
- Cold start is roughly $S / \min(B_\text{net}, B_\text{disk}, B_\text{pcie})$ plus engine init; cache placement beats loader tuning, and production never loads from the Hub at request time.
- Chat templates fail silently; only diffing templated token ids against training proves correctness.

## Review questions

1. A customer's checkpoint loads with "Some weights were not initialized" for every layer, and the unexpected keys all start with `module.`. What happened, and how do you fix it?
   <details><summary>Answer</summary>The checkpoint was saved from a DDP wrapper (similar prefixes come from `torch.compile`, `_orig_mod.`, and PEFT, `base_model.model.`), so every name carries a prefix the model's parameters lack and no key matches. Diff the header keys against `model.state_dict().keys()` to confirm, then strip the prefix or re-save from the unwrapped model (for PEFT, `merge_and_unload()` first).</details>
2. How can you learn the dtypes and shapes of every tensor in a 100 GB remote checkpoint without downloading it?
   <details><summary>Answer</summary>Read the safetensors header with HTTP `Range` requests: the first 8 bytes give the header length `N` as a little-endian u64, and the next `N` bytes are a JSON object listing each tensor's dtype, shape, and data offsets. Two range requests suffice; `safetensors_inspect.py --url` does exactly this.</details>
3. Hand-derive the parameter count of Llama 3.1 8B from its config (vocabulary 128,256; hidden size 4096; 32 layers; 32 query heads and 8 KV heads of dimension 128; MLP intermediate size 14,336; untied embeddings). Check it against the index file.
   <details><summary>Answer</summary>The embedding and the untied `lm_head` are each $128{,}256 \times 4096 = 525{,}336{,}576$. Per layer, attention has $q$ and $o$ at $4096 \times 4096 = 16{,}777{,}216$ each and $k$ and $v$ at $4096 \times 1024 = 4{,}194{,}304$ each, for 41,943,040; the MLP has three $4096 \times 14{,}336$ matrices, 176,160,768; and two RMSNorm vectors add 8,192. That is 218,112,000 per layer, or 6,979,584,000 for 32 layers. Adding both embedding matrices and the final 4,096-element norm gives 8,030,261,248, and at 2 bytes each in BF16 that is 16,060,522,496 bytes, exactly the index's `metadata.total_size`.</details>
4. Why must a fused `gate_up_proj` be sliced per sub-matrix under tensor parallelism, and why is getting it wrong so dangerous?
   <details><summary>Answer</summary>The fused tensor is $[G; U]$, and each rank needs its slice of both, $[G_r; U_r]$. Taking the contiguous $r$-th chunk of the fused tensor would give rank 0 all of $G$ and the last rank all of $U$. Because the shapes still match, this loads without error and silently produces garbage.</details>
5. Using the cold-start table, explain why "cache placement beats loader tuning", and what becomes the floor once weights are local.
   <details><summary>Answer</summary>A 70B BF16 model takes about 12 minutes over a single HTTP stream but about 22 s from one local NVMe drive and 2.8 s across PCIe Gen5; the network and disk terms dominate the others by 10 to 100 times, so where the bytes sit matters far more than how the loader reads them. Once weights are local, engine initialization (CUDA graph capture and compilation, often 20 to 60 s) is the floor, which GPU state snapshotting attacks.</details>
6. An instruct fine-tune never stops generating and keeps printing `assistant` turns. What do you check?
   <details><summary>Answer</summary>Compare `generation_config.eos_token_id` with the template's turn terminator. In Llama 3 an instruct turn ends with `<|eot_id|>` (128009); if only `<|end_of_text|>` (128001) is listed as EOS, generation runs past each turn. Make `eos_token_id` a list containing every terminator.</details>
7. Why are chat-template mismatches silent, and what is the one reliable test?
   <details><summary>Answer</summary>The model still receives valid tokens, so it produces fluent text; it is merely conditioned on an out-of-distribution prefix, which shows up as ignored system prompts, malformed tool calls, more refusals, and a few points lost on evals. The reliable test is to compare the token ids from `apply_chat_template(..., add_generation_prompt=True)` with the ids of the training formatter's output tokenized with `add_special_tokens=False`, and assert they are identical.</details>
