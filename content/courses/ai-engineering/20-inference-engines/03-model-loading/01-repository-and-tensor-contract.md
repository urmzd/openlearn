# Repository and tensor contract

This part of [Model Loading: Hub to GPU](README.md) develops the mechanisms below. Work the numerical examples or trace the code before checking the answer.

> **Learning objectives**
> - Match model configuration and tensor metadata before allocating weights.

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

Low-bit formats store values in blocks that share a scale. **MXFP4** (the OCP Microscaling format) groups 32 E2M1 values under one E8M0 scale, for about 4.25 bits per weight. **NVFP4** (Blackwell) groups 16 E2M1 values under an FP8 E4M3 scale, plus a per-tensor FP32 scale. In a header these appear as a `U8` or `F4` tensor plus a `*_scales` tensor, so for quantized checkpoints **element count is not parameter count**. DeepSeek-V3, for example, stores `F8_E4M3` weights with a `weight_scale_inv` tensor per 128×128 block. The quantizers themselves are derived in [Quantization: Math to Code](../01-quantization/README.md).

## Summary

Match model configuration and tensor metadata before allocating weights. State the assumptions before applying the result to a different workload.

## Review questions

1. Why can a shape-correct checkpoint still be wrong?

   <details><summary>Answer</summary>

   Tensor names, permutations, architecture semantics, tokenizer, and chat template can differ while shapes still match.

   </details>

[Chapter contents](README.md) · [Next: Downloads, loaders, and sharding](02-downloads-loaders-and-sharding.md)
