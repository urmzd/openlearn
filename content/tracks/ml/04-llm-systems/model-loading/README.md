# Model Loading: Hub to GPU

How a checkpoint becomes a running model, and how to debug it when it does not. Every "my custom fine-tune won't load / is slow to start / gives garbage" ticket at an inference cloud reduces to one of four layers: **the repo** (files and metadata), **the bytes** (format, dtype, layout), **the loader** (key mapping, sharding, quantization), or **the tokenizer** (template and special tokens). This chapter works through each layer as math first, then the code that checks it.

> Parent topic: [LLM Systems & Inference](../). Formats consumed here are explained in [Quantization: Math → Code](../quantization/); the runtimes that do the loading are mapped in [Inference Frameworks](../frameworks/) and [LLM Serving Platforms](../serving-platforms.md). Runnable examples: [`code/`](code/).

## Overview

- **Primary references** (all free): [safetensors format spec](https://github.com/huggingface/safetensors#format) (free), [Transformers: Loading models](https://huggingface.co/docs/transformers/main/en/models) (free), [Transformers: Dynamic weight loading](https://huggingface.co/docs/transformers/main/en/weightconverter) (free), [huggingface_hub: Download files](https://huggingface.co/docs/huggingface_hub/guides/download) and [Understand caching](https://huggingface.co/docs/huggingface_hub/guides/manage-cache) (free), [vLLM engine arguments](https://docs.vllm.ai/en/latest/configuration/engine_args.html) (free), [Transformers: Chat templates](https://huggingface.co/docs/transformers/main/en/chat_templating) (free)
- **Supplementary**: [GGUF spec](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) (free), [Hub Xet storage](https://huggingface.co/docs/hub/xet/index) (free), [`hf` CLI guide](https://huggingface.co/docs/huggingface_hub/guides/cli) (free), [vLLM Run:ai Model Streamer](https://docs.vllm.ai/en/latest/models/extensions/runai_model_streamer.html) (free), [fastsafetensors](https://github.com/foundation-model-stack/fastsafetensors) (free), [CoreWeave tensorizer](https://github.com/coreweave/tensorizer) (free), [NVIDIA cuda-checkpoint](https://github.com/NVIDIA/cuda-checkpoint) (free), [PEFT LoRA docs](https://huggingface.co/docs/peft) (free), [LoRA paper](https://arxiv.org/abs/2106.09685) (free), [FP8 formats paper](https://arxiv.org/abs/2209.05433) (free), [OCP Microscaling (MX) spec](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) (free), [Flax NNX checkpointing](https://flax.readthedocs.io/en/latest/guides/checkpointing.html) (free), [KerasHub](https://keras.io/keras_hub/) (free)
- **Prerequisites**: [LLM Systems & Inference](../) (esp. §6 quantization, §8 parallelism), [Deep Learning](../../02-deep-learning/) (transformer blocks, attention)
- **Estimated time**: 1-2 weeks at 8-10 hrs/week
- **Versions reviewed** (October 2026): transformers 5.19, huggingface_hub 2.1, safetensors 0.8, vLLM 0.31, hf_xet 1.7. Pin versions in anything you ship; this layer moves monthly

## Key Takeaways

- A model is **config + weights + tokenizer + generation defaults**, and each lives in a different file. Garbage output is more often a tokenizer or config bug than a weights bug
- **safetensors is a JSON header plus a flat byte buffer.** You can inspect any checkpoint, local or remote, by reading the first `8 + N` bytes; you never need to load it to know its dtypes, shapes, and size
- **`from_pretrained` is a key-matching problem.** Every load is "map checkpoint names to module parameters, then fill them"; prefix drift, tied weights, and fused projections cause most load failures
- **Cold start is a bandwidth equation**: `T ≈ bytes / min(B_net, B_disk, B_pcie)` plus engine init. Production loads from local NVMe or parallel object-store reads, never from the Hub at request time
- **Chat templates fail silently.** A wrong template still produces fluent text; only diffing token ids against training proves correctness

## How to Study

- Run [`code/safetensors_inspect.py`](code/safetensors_inspect.py) `--url` against three Hub models (one tied-embedding, one FP8, one MXFP4) and explain every tensor name you see
- Run [`code/config_explain.py`](code/config_explain.py) on the samples, then hand-derive the Llama 3.1 8B parameter count until you get 8,030,261,248 exactly (it matches `metadata.total_size / 2` in the official index)
- Break a load on purpose: rename keys with a `module.` prefix, drop `lm_head.weight`, delete `rope_scaling`, swap the chat template. Record the symptom of each; that table is your on-call runbook
- Read one real loader end to end: vLLM's [`llama.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama.py) `hf_to_vllm_mapper` and `AutoWeightsLoader`

---

# Concepts & Techniques

## Core Insight

Loading is a **bijection problem under a byte budget**. A checkpoint is a dictionary `{name → (dtype, shape, bytes)}`; a model is a dictionary `{name → parameter slot}`. Loading succeeds when there is a renaming function `f` and a set of tensor operations (cast, transpose, concatenate, slice per rank, dequantize) that map one onto the other with nothing missing and nothing left over. It is fast when those bytes stream at the slowest link's bandwidth with no extra copies. It is correct only when the tokenizer feeds the model the same token ids it saw in training. Everything below is a specialization of those three statements.

## 1. Anatomy of a Hub Repo

A Hub model repo is a git repository with large files stored out of band ([Xet](https://huggingface.co/docs/hub/xet/index), the successor to Git LFS: content-defined chunks, deduplicated across repos and revisions).

| File | What it decides | Fields an FDE checks first |
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

**Key ideas**:
- **`model_type` is the dispatch key.** `AutoModelForCausalLM` looks it up in a registry (`llama` → `LlamaForCausalLM`). If the installed library does not know the type, loading fails before a byte of weights is read
- **`auto_map` means custom code.** It points at `modeling_*.py` files in the repo, which only run with `trust_remote_code=True`. That executes arbitrary Python from the repo; pin a commit `revision` if you must use it
- **Revisions are commits.** `main` moves; a 40-char sha does not. A customer saying "it worked yesterday" on `main` often means the repo changed. Resolve once and pin: `revision="0e9e39f..."`
- **Config dtype is a hint, not a guarantee.** `torch_dtype` (renamed `dtype` in transformers v5, old key still read) records the training dtype; the safetensors header records what is actually stored. When they disagree, trust the header

The real Llama 3.1 8B Instruct config (in [`code/samples/`](code/samples/)) shows three classic traps in eight lines: `eos_token_id` is a **list** `[128001, 128008, 128009]`, `rope_scaling.rope_type = "llama3"` (older libraries reject it), and `num_key_value_heads = 8 < num_attention_heads = 32` (GQA, which matters for TP sharding).

## 2. File Formats and Dtypes

### 2.1 safetensors byte layout

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

Math → code ([`code/safetensors_inspect.py`](code/safetensors_inspect.py)):

```python
(n,) = struct.unpack("<Q", f.read(8))       # header length
header = json.loads(f.read(n))              # starts with '{'
b, e = header[name]["data_offsets"]
f.seek(8 + n + b); raw = f.read(e - b)      # zero parsing of the payload
```

**Key ideas**:
- **Zero-copy and mmap**: because offsets are known up front, a loader can `mmap` the file and hand out tensor views with no deserialization. Pages fault in lazily on first touch, which is why a "0.3 s load" on a warm page cache becomes 30 s on a cold node. Measure cold with `sync; echo 3 > /proc/sys/vm/drop_caches`
- **Remote inspection**: two HTTP `Range` requests read the header of a 100 GB checkpoint. `safetensors_inspect.py --url` does this, so you can verify a customer's dtypes and key names before pulling a byte of weights
- **Validation**: the spec forbids holes and duplicate keys; `END - BEGIN` must equal `prod(shape) × bits / 8`. A mismatch means a corrupt or hand-edited file
- **Sharding**: `save_pretrained` writes shards plus `model.safetensors.index.json`. A missing shard is a common partial-upload failure; compare `weight_map` against the shards present

### 2.2 Why pickle `.bin` is unsafe

`pytorch_model.bin` is a zip of Python **pickles**. Unpickling executes the `__reduce__` callables the file names, so loading an untrusted `.bin` is remote code execution ([Python docs warning](https://docs.python.org/3/library/pickle.html)). PyTorch 2.6 made `torch.load(weights_only=True)` the default, which restricts unpickling to tensors and primitives, but inference platforms should still **refuse `.bin` from customers** and convert to safetensors in a sandbox. safetensors cannot execute code: it is JSON plus raw bytes.

### 2.3 Other formats

| Format | Structure | Who loads it | Portability |
|--------|-----------|--------------|-------------|
| **safetensors** | JSON header + flat buffer | transformers, vLLM, SGLang, TGI, candle, MLX | Any framework, any hardware |
| **PyTorch `.bin`/`.pt`** | Zip of pickles | Legacy PyTorch | Unsafe, Python-only |
| **GGUF** | Magic `GGUF`, version, KV metadata (arch, tokenizer, `tokenizer.chat_template`), tensor infos, aligned data | llama.cpp, Ollama, LM Studio, mistral.rs, candle | One file holds weights **and** tokenizer **and** template; k-quant block types |
| **ONNX** | Protobuf graph + initializers (external data > 2 GB) | ONNX Runtime, DirectML, edge | Graph is frozen; dynamic shapes need care |
| **TensorRT engine** | Serialized, compiled kernels | TensorRT / TensorRT-LLM | Tied to GPU arch, TRT version, and build-time max shapes; rebuild per SKU |
| **Orbax checkpoint** | Directory of array shards (TensorStore/OCDBT) | JAX, Flax NNX, MaxText | Sharding-aware restore onto a device mesh |
| **Keras `.weights.h5` / preset dir** | HDF5 or preset folder | Keras 3, KerasHub | Backend-agnostic (JAX, PyTorch, TF) |

### 2.4 Dtypes you will see in a header

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

**MXFP4** (OCP MX): blocks of 32 E2M1 values share one E8M0 scale, about 4.25 bits per weight. **NVFP4** (Blackwell): blocks of 16 E2M1 values share an FP8 E4M3 scale plus a per-tensor FP32 scale. In a header these look like a `U8` or `F4` tensor plus a `*_scales` tensor, so **element count ≠ parameter count** for quantized checkpoints. DeepSeek-V3 stores `F8_E4M3` weights with a `weight_scale_inv` tensor per 128×128 block. The quantizers themselves are derived in [Quantization](../quantization/).

## 3. Downloading and the Cache

```python
from huggingface_hub import hf_hub_download, snapshot_download

cfg = hf_hub_download("Qwen/Qwen3-30B-A3B", "config.json")          # one file
path = snapshot_download(                                           # a whole revision
    "meta-llama/Llama-3.1-8B-Instruct",
    revision="0e9e39f249a16976918f6564b8830bc894c89659",           # pin a commit
    allow_patterns=["*.json", "*.safetensors", "tokenizer*"],       # skip .bin, GGUF, original/
)
```

```bash
hf auth login                                   # or export HF_TOKEN=hf_...
hf download meta-llama/Llama-3.1-8B-Instruct --include "*.safetensors" --include "*.json" \
    --revision 0e9e39f249a16976918f6564b8830bc894c89659 --local-dir /mnt/nvme/llama31-8b
hf cache ls                                     # what is cached, how big
hf cache verify meta-llama/Llama-3.2-1B-Instruct   # checksum against the Hub
HF_HUB_OFFLINE=1 python serve.py                # never touch the network
```

The CLI is **`hf`**; `huggingface-cli` is the deprecated name ([CLI guide](https://huggingface.co/docs/huggingface_hub/guides/cli)).

**Cache layout** (`$HF_HOME/hub`, default `~/.cache/huggingface/hub`, override with `HF_HUB_CACHE`):

```
models--meta-llama--Llama-3.1-8B-Instruct/
├── blobs/<sha256 or etag>          # actual bytes, content-addressed
├── refs/main                       # text file: commit sha that "main" resolved to
├── snapshots/<commit-sha>/
│   ├── config.json -> ../../blobs/…
│   └── model-00001-of-00004.safetensors -> ../../blobs/…
└── trees/<commit-sha>.json         # cached file list for that commit
```

**Key ideas**:
- **Symlinks dedupe revisions**: two revisions sharing a shard point at one blob. On filesystems without symlinks (some network mounts, Windows without developer mode) files are copied, and disk use multiplies. Recent `huggingface_hub` also shares Xet blobs **across repos**, so fine-tunes that reuse base shards cost no extra disk
- **`--local-dir`** writes plain files instead of the cache layout. Use it when baking weights into an image or a volume
- **Speed**: `hf_xet` (installed with `huggingface_hub`) replaced `hf_transfer`. Set `HF_XET_HIGH_PERFORMANCE=1` on big-NIC machines to raise concurrency. Without `HF_TOKEN`, requests are rate-limited
- **Offline**: `HF_HUB_OFFLINE=1` makes every call resolve from cache; with a pinned sha, a fully cached load makes **zero** network calls. An incomplete snapshot raises `IncompleteSnapshotError` instead of silently returning a partial folder
- **Production rule**: the Hub is a distribution channel, not a serving dependency. Mirror pinned revisions once into your own object store (`s3://models/<org>/<name>/<sha>/`), checksum them, and load from there or from local NVMe. Reasons: rate limits and outages, gated-token sprawl, revision drift, and egress cost across thousands of replicas

## 4. What `from_pretrained` Actually Does

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

In transformers v5 the argument is **`dtype`** (`torch_dtype` is the legacy name), and the default now follows the checkpoint's config dtype instead of upcasting to FP32. Step 5 is the [dynamic weight loader](https://huggingface.co/docs/transformers/main/en/weightconverter): `WeightRenaming` rules (e.g. `LayerNorm.gamma → LayerNorm.weight`) and `WeightConverter` rules (e.g. stack Mixtral's per-expert `w1`/`w3` into one `experts.gate_up_proj` with `MergeModulelist` + `Concatenate`) run as tensors stream in, and they are **reversible** so `save_pretrained` writes the original layout back. Peak memory ≈ model size plus the largest merge (one MoE layer's experts).

**Key ideas**:
- **Meta device** removes the "two copies" problem: without it you allocate random weights and then the checkpoint, doubling peak RAM
- **Read the load report.** "Some weights were not initialized from the checkpoint" with `lm_head.weight` listed is not a warning; it means a random output head
- **`trust_remote_code`** executes repo Python at load time. Inference platforms run it, if at all, in a sandbox at conversion time, never in the serving process. Note that custom-code models skip transformers' built-in conversion mappings

### Failure modes

| Symptom | Root cause | Check | Fix |
|---------|-----------|-------|-----|
| "weights not initialized" for every layer; unexpected keys start with `module.`, `_orig_mod.`, `base_model.model.` | Saved from a DDP / `torch.compile` / PEFT wrapper | Diff header keys against `model.state_dict().keys()` | Strip the prefix; save from the unwrapped model; `merge_and_unload()` for PEFT |
| Output is random text from token 1 | `lm_head.weight` missing and `tie_word_embeddings=false` (or present but tied flag wrong) | Is `lm_head.weight` in `weight_map`? | Set `tie_word_embeddings` to match the checkpoint |
| Fine at 4K tokens, degrades past 8K | `rope_scaling` dropped or rewritten by a fine-tune toolkit; or old library ignores `rope_type: llama3`/`yarn` | Diff `rope_scaling` and `rope_theta` against the base config | Restore the base values; upgrade the library |
| Fluent but worse than base, ignores system prompt | Chat template differs from training | Render both templates, diff token ids (§7) | Ship the training template in `tokenizer_config.json` / `chat_template.jinja` |
| Never stops; prints `assistant` turns forever | EOS mismatch: instruct turn ends with `<\|eot_id\|>` (128009) but only `<\|end_of_text\|>` (128001) is an EOS | `generation_config.eos_token_id` vs template's turn terminator | Make `eos_token_id` a list containing every terminator |
| NaN / inf or garbage only in FP16 | BF16-trained activations exceed 65504 | Same prompt in BF16 is fine | Serve BF16 (or FP8 with scales), never FP16, for BF16-trained models |
| `size mismatch for embed_tokens: [128264, 4096] vs [128256, 4096]` | Tokens added in fine-tune, or vocab padded to a multiple of 64 | `len(tokenizer)` vs `config.vocab_size` vs header shape | `resize_token_embeddings(len(tok))` before saving; set `vocab_size` to the stored rows |
| `model type X not recognized` | Library older than the architecture, or custom code | `transformers.__version__`; `auto_map` in config | Upgrade, or pinned-revision `trust_remote_code` in a sandbox |
| Quality collapse on a quantized fine-tune | `quantization_config` stale (fine-tuned in BF16, config still says GPTQ), or `ignore` list missing `lm_head` | Header dtypes vs `quantization_config` | Requantize from the merged BF16 weights |

## 5. How Serving Engines Load

vLLM and SGLang keep their own **model registries** (architecture string → engine-native implementation with fused kernels) and their own loaders; they read the same `config.json` and safetensors but never call transformers' `from_pretrained` for supported architectures. An unsupported `architectures` entry falls back to the Transformers backend (slower) or fails.

### 5.1 Name mapping and fused projections

Engines fuse `q_proj, k_proj, v_proj` into one `qkv_proj` GEMM and `gate_proj, up_proj` into one `gate_up_proj`. vLLM's Llama declares the mapping ([source](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama.py)):

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

Each parameter carries a `weight_loader(param, loaded_weight, shard_id)` that knows where in the fused tensor, and which rows for this rank, the incoming tensor goes. A customer checkpoint that **already** fused `qkv_proj` (common from custom training code) misses this mapping: the fix is to split it back to HF names, not to patch the engine.

### 5.2 Tensor-parallel slicing math

PyTorch stores `nn.Linear` weights as `W ∈ ℝ^{out × in}`. With `t` ranks:

```
ColumnParallel (q, k, v, gate, up):  W_r = W[r·out/t : (r+1)·out/t, :]      output split, no comm
RowParallel    (o_proj, down_proj):  W_r = W[:, r·in/t : (r+1)·in/t]        partial sums → all-reduce
Vocab-parallel embedding / lm_head:  rows r·V'/t … where V' = V padded up to a multiple of 64 (vLLM)
```

For GQA with `H` query heads and `KV` key/value heads, rank `r` gets query heads `[r·H/t, (r+1)·H/t)` and KV heads `[r·KV/t, (r+1)·KV/t)`. Requirements: `H mod t = 0`; if `KV < t`, each KV head is **replicated** across `t/KV` ranks (so `t mod KV = 0`). Llama 3.1 8B (`H=32, KV=8`) shards cleanly to `t ∈ {1,2,4,8}`; Qwen3-30B-A3B (`KV=4`) at `t=8` replicates each KV head twice.

The fused `gate_up_proj = [G; U]` must be sliced **per sub-matrix**: rank `r` gets `[G_r; U_r]`, not the contiguous `r`-th chunk of the fused tensor (which would hand rank 0 all of `G`). Transformers' TP plan calls this `packed_colwise`. Getting it wrong loads without error and produces garbage, which is why fused checkpoints are dangerous.

### 5.3 Pre-quantized checkpoints

Engines read `config.json → quantization_config.quant_method` and swap in the matching linear method and kernel:

| `quant_method` | Stored tensors | Kernel path (vLLM) |
|----------------|----------------|--------------------|
| `awq` | `qweight` (I32, 8×int4), `qzeros`, `scales` (F16), group 128 | AWQ Marlin on Ampere+ |
| `gptq` | `qweight`, `qzeros`, `scales`, `g_idx` (act-order) | GPTQ Marlin / Machete |
| `fp8` | `F8_E4M3` weight + `weight_scale_inv` (block 128×128) or per-tensor scale | CUTLASS FP8 GEMM (Hopper+, Ada) |
| `compressed-tensors` | Per-group `config_groups` (W4A16, W8A8, FP8, NVFP4) + `ignore` list | Marlin / CUTLASS by scheme ([llm-compressor](https://github.com/vllm-project/llm-compressor) output) |
| `modelopt` | NVIDIA ModelOpt FP8 / NVFP4 | Blackwell FP4 tensor cores |
| `mxfp4` | gpt-oss experts as `*_blocks` (U8) + `*_scales` (E8M0) | MXFP4 MoE kernels |

`--quantization` on the CLI overrides detection; usually you should not pass it. The common ticket: "quantized fine-tune is slow" because the GPU lacks the fast kernel (FP8 on Ampere falls back to weight-only Marlin), not because loading failed.

### 5.4 LoRA adapters: merge or serve

An adapter is `adapter_config.json` (`r`, `lora_alpha`, `target_modules`, `base_model_name_or_path`) plus `adapter_model.safetensors` with keys like `base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight`. The math:

```
W' = W + (α / r) · B A        A ∈ ℝ^{r × in}, B ∈ ℝ^{out × r}   (rsLoRA: α / √r)
extra params per adapted matrix = r · (in + out)    e.g. r=16, 4096×4096 → 131,072 (0.8 %)
```

| | **Merge** (`merge_and_unload()`, then serve as a full model) | **Serve unmerged** (`vllm serve base --enable-lora --lora-modules name=path`) |
|---|---|---|
| Latency | Zero overhead | Extra `2·r·(in+out)` FLOPs/token per adapted matrix; batched SGMV/Punica kernels |
| Memory | Full copy per fine-tune | One base, many adapters (multi-tenant) |
| Cold start | Full model load | Adapter is MBs: seconds |
| Gotchas | Merging into a quantized base needs dequant → merge → requant | `r` > `--max-lora-rank` (default 16) rejects; adapters that also train `embed_tokens`/`lm_head` (added tokens) may not be supported; base revision must match training |

Rule: serve unmerged for many low-traffic fine-tunes; merge for one high-traffic fine-tune where every microsecond of TPOT matters.

## 6. Cold Start Math

```
T_cold = T_schedule + T_fetch + T_read + T_h2d + T_init
T_fetch ≈ S / B_net           (only if weights are not already on the node)
T_read + T_h2d ≈ S / min(B_disk, B_pcie)   when streamed and overlapped, else their sum
T_init  = CUDA context + KV-cache profiling + CUDA graph capture (+ torch.compile)
S = params × bytes/param      (per rank: S / t if checkpoints are pre-sharded)
```

Worked examples (BF16; 8B = 16.06 GB from the Llama 3.1 index, 70B = 70.55 B params = 141.1 GB):

| Path | Effective bandwidth | 8B (16.1 GB) | 70B (141 GB) |
|------|--------------------|--------------|--------------|
| Hub or S3, single HTTP stream | ~0.2 GB/s | 80 s | 12 min |
| Object store, parallel range reads, 25 GbE | ~3 GB/s | 5.4 s | 47 s |
| Object store, parallel, 100 GbE | ~10 GB/s | 1.6 s | 14 s |
| Local NVMe Gen4 (one drive) | ~6.5 GB/s | 2.5 s | 22 s |
| NVMe RAID-0 ×4 or page cache | ~25 GB/s | 0.6 s | 5.6 s |
| PCIe Gen5 x16 host → one GPU | ~50 GB/s | 0.3 s | 2.8 s (÷ t with TP, each GPU has its own link) |

Two lessons fall out. The network and disk terms dominate by 10-100×, so **cache placement beats loader tuning**. And once weights are local, `T_init` (graph capture, compile, often 20-60 s for large models) becomes the floor, which is why the last tricks below snapshot GPU state instead of reloading.

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

## 7. Tokenizers and Chat Templates

| Family | Algorithm | Examples | Tell-tale |
|--------|-----------|----------|-----------|
| **BPE** ([Sennrich et al.](https://arxiv.org/abs/1508.07909)) | Greedy merges of frequent pairs from a merges table | GPT-2 lineage | `merges` in `tokenizer.json` |
| **Byte-level BPE** | BPE over bytes, so no unknown tokens | GPT-2, Llama 3 (128,256 vocab), Qwen | `Ġ` marks a leading space |
| **SentencePiece** ([repo](https://github.com/google/sentencepiece)) | Unigram LM or BPE over raw text, whitespace as `▁` | Llama 2, Gemma, T5 | `▁` prefix; `tokenizer.model` file |
| **tiktoken** ([repo](https://github.com/openai/tiktoken)) | Byte-level BPE with regex pre-split, fast Rust core | OpenAI `cl100k_base`, `o200k_base`; gpt-oss `o200k_harmony` | Ranks file, no merges table |

A **chat template** is a Jinja program stored in `tokenizer_config.json` (`chat_template`) or `chat_template.jinja`. It turns `[{"role","content"}]` into the exact string the model was trained on:

```
SmolLM2:  <|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n
Llama 3 (simplified; 3.1 also injects a dated system header):
          <|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
```

**Why mismatches are silent**: the model still sees valid tokens, so output is fluent. It is just conditioned on an out-of-distribution prefix: system prompts get ignored, tool calls malformed, refusals spike, evals drop a few points. Nothing errors.

**How to verify** (do this on every custom model onboarding):

```python
ids_serving = tok.apply_chat_template(msgs, add_generation_prompt=True)     # what the engine sends
ids_training = tok(training_formatter(msgs), add_special_tokens=False).input_ids  # what SFT saw
assert ids_serving == ids_training, first_divergence(ids_serving, ids_training)
```

Common divergences: **double BOS** (the template already emits `<|begin_of_text|>`, then the rendered string is tokenized again with `add_special_tokens=True`, so the post-processor adds a second one), missing `add_generation_prompt`, a fine-tune that added special tokens (`added_tokens`) the serving tokenizer lacks, trailing whitespace or newline differences, and a template that drops the system role. vLLM uses the repo template unless `--chat-template` overrides it, and `--generation-config auto` pulls sampling defaults from `generation_config.json`, so both files are part of the deployed contract.

## 8. Beyond PyTorch: JAX, Flax NNX, Keras 3

transformers v5 dropped its TensorFlow and Flax backends, so the JAX world loads through its own stack:

```python
# Flax NNX + Orbax: build an abstract model (shapes only), then restore into it
abstract = nnx.eval_shape(lambda: Transformer(cfg, rngs=nnx.Rngs(0)))
graphdef, abstract_state = nnx.split(abstract)
state = ocp.StandardCheckpointer().restore(ckpt_dir / "state", abstract_state)
model = nnx.merge(graphdef, state)
```

`nnx.eval_shape` plays the role of PyTorch's meta device, and Orbax restores each array **directly onto its target sharding** across a device mesh (no host-side gather), which is how MaxText and TPU serving load. Converting HF safetensors into JAX means the same key renaming plus transposes (`nn.Linear` stores `out × in`; Flax `Dense` kernels store `in × out`).

**Keras 3 / KerasHub** presets are backend-agnostic: `keras_hub.models.CausalLM.from_preset("hf://<org>/<repo>", dtype="bfloat16")` converts supported HF architectures (Llama 3, Gemma, Mistral, Mixtral, Qwen, gpt-oss, and others in [`utils/transformers/`](https://github.com/keras-team/keras-hub/tree/master/keras_hub/src/utils/transformers)) on load and runs them on JAX, PyTorch, or TensorFlow.

---

## Triage Playbook

| Customer says | First command | Most likely cause |
|---------------|---------------|-------------------|
| "Won't load" | `safetensors_inspect.py <index.json>` and diff keys vs base model | Prefix drift, missing shard, fused or renamed projections, unknown `model_type` |
| "Loads but OOMs" | `config_explain.py config.json` | Weights + KV cache at max context exceed GPU; dtype upcast to FP32 |
| "Slow to start" | Time each term of §6 separately | Pulling from the Hub per replica; network filesystem with mmap; no warm cache |
| "Gives garbage" | Same prompt in transformers BF16 vs engine | Missing `lm_head`, FP16 overflow, TP slicing of fused tensors, quant config mismatch |
| "Worse than the base model" | Diff templated token ids vs training | Chat template, double BOS, EOS list, sampling defaults |
| "Never stops" | Print `generation_config.eos_token_id` | Turn terminator not in EOS list |

## Connections to Other Tracks

| Concept | Connected Track | Application |
|---------|-----------------|-------------|
| Transformer blocks, GQA, MoE, RoPE | [Deep Learning](../../02-deep-learning/) | What each tensor name means |
| FP8/MXFP4/NVFP4, AWQ, GPTQ | [Quantization](../quantization/) | Reading pre-quantized checkpoints |
| Fine-tuning, LoRA, SFT data formatting | [Training & Post-Training](../../07-training-and-post-training/) | Where adapters and templates come from |
| Image baking, daemonsets, warm pools | [Cloud Native](../../../systems/03-cloud-native/) | Getting weights onto nodes |
| Load-time and TTFT metrics | [Observability](../../../systems/04-observability/) | Measuring each cold-start term |

## Company Relevance

| Company | How This Appears | Difficulty |
|---------|-----------------|------------|
| Fireworks / Together / Baseten | Custom-model onboarding, LoRA multi-tenancy, cold-start SLAs; FDEs debug customer checkpoints daily | Expert |
| Hugging Face | Hub storage (Xet), safetensors, transformers loader, TGI/Inference Endpoints | Expert |
| NVIDIA | ModelOpt FP8/NVFP4 checkpoints, TensorRT-LLM engine builds, GPUDirect Storage | Expert |
| Modal / Replicate / RunPod | Scale-to-zero, GPU memory snapshots, weight caching | Expert |
| Anyscale / Databricks | vLLM-based serving, model registries, object-store streaming | Expert |
| Google | Orbax, MaxText, KerasHub presets, TPU serving | Expert |
