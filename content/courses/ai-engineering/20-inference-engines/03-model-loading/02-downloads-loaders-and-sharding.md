# Downloads, loaders, and sharding

This part of [Model Loading: Hub to GPU](README.md) develops the mechanisms below. Work the numerical examples or trace the code before checking the answer.

> **Learning objectives**
> - Trace checkpoint bytes through caches, loaders, and device shards.

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
| Fluent but worse than base, ignores system prompt | Chat template differs from training | Render both templates, diff token ids (see [Tokenizers and chat templates](03-cold-start-and-tokenizers.md#tokenizers-and-chat-templates)) | Ship the training template in `tokenizer_config.json` / `chat_template.jinja` |
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

## Summary

Trace checkpoint bytes through caches, loaders, and device shards. State the assumptions before applying the result to a different workload.

## Review questions

1. Why does a cache hit not eliminate startup time?

   <details><summary>Answer</summary>

   Deserialization, validation, CPU-to-device transfer, layout conversion, and warmup still take time.

   </details>

[Chapter contents](README.md) · [Next: Cold starts and tokenizer equivalence](03-cold-start-and-tokenizers.md)
