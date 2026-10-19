# Part 8: Inference

Generating text fast and exactly. The sampler and logit processors in the course's fixed op order, the KV cache with incremental decode and an incremental UTF-8 detokenizer, the paged KV block pool in C (`rt.04`) and the Python paged cache over it, a radix prefix cache, quantization (int8, int4, fp8, KV), speculative decoding, and constrained decoding from regex and JSON schema to token masks.

**Course passes**: rt.04 and L8.1 to L8.7 belong to [Pass 6](../paths/course-p06-inference-and-kernels/milestone.md)

**Before you start**: the just-in-time math `M07.6`, `M09.3`, `M09.4` and the solve part `S-M09b`; the C structures `ds.01` to `ds.03` and the radix tree `ds.07` in [Systems Data Structures](../../../tracks/algorithms/16-systems-data-structures/); the Llama-family model of [Part 7](../p07-modern-block/).

## Key ideas

- **Decoding is memory-bound**: each new token reads every weight and the whole KV cache once, so bytes moved, not FLOPs, set the speed.
- **Paging** stores the KV cache in fixed-size blocks with a block table per sequence; shared prefixes share blocks by reference count and copy on write.
- **Sampling is specified to the bit** (`spec/sampling.md`): penalties, temperature, top-k, top-p, min-p, an f64 softmax, one uniform draw, inverse CDF, so Python and Rust emit identical tokens.
- **Speculative decoding** drafts several tokens cheaply and verifies them in one forward pass; with greedy acceptance the output is unchanged.

## Modules

| Module | Topic | Kind | Pass |
|---|---|---|---|
| `rt.04` | Optional standalone C paged KV block pool: refcount, copy on write, chained block hashes, prefix index, LRU, export and import (format v1); parity via fixtures | optional | 6 |
| `L8.1` | Sampling and logit processors (`spec/sampling.md`) | build | 6 |
| `L8.2` | KV cache, incremental decode, incremental UTF-8 detokenizer, `generate` | build | 6 |
| `L8.3` | Paged KV cache implemented in Python | build | 6 |
| `L8.4` | Radix prefix cache (block-granular, LRU leaf eviction, locks) | build | 6 |
| `L8.5` | Quantization: int8 per-channel, int4 group (packed), fp8, KV quant | build | 6 |
| `L8.6` | Speculative decoding: n-gram, prompt-lookup, and model drafts | build | 6 |
| `L8.7` | Constrained decoding: regex to DFA to token masks, JSON-schema subset | build | 6 |

## Chapters

| # | Module | Chapter | Kind | Pass |
|---|---|---|---|---|
| 1 | `L8.1` | [Sampling and logit processors](01-sampling-and-logit-processors.md) | build | 6 |
| 2 | `L8.2` | [KV cache, incremental decode, incremental UTF-8 detokenizer, generate](02-kv-cache-and-generate.md) | build | 6 |
| 3 | `L8.3` | [Paged KV cache in Python](03-paged-kv-cache.md) | side | 6 |
| 4 | `L8.4` | [Radix prefix cache (block-granular, LRU leaf eviction, locks)](04-radix-prefix-cache.md) | build | 6 |
| 5 | `L8.5` | [Quantization: int8, int4 group (packed), fp8, KV quant](05-quantization.md) | build | 6 |
| 6 | `L8.6` | [Speculative decoding: n-gram, prompt-lookup, and model drafts](06-speculative-decoding.md) | build | 6 |
| 7 | `L8.7` | [Constrained decoding: regex to DFA to token masks, JSON-schema subset](07-constrained-decoding.md) | build | 6 |
| 8 | `rt.04` | [Paged KV block pool (format v1, optional C)](08-paged-kv-block-pool.md) | side | 6 |

## Going further

- Kwon et al., [*Efficient Memory Management for LLM Serving with PagedAttention*](https://arxiv.org/abs/2309.06180) (vLLM block manager); Zheng et al., [SGLang](https://arxiv.org/abs/2312.07104) (RadixAttention); [llama.cpp](https://github.com/ggml-org/llama.cpp) (`llama_batch`).
- Leviathan, Kalman, and Matias, [*Fast Inference from Transformers via Speculative Decoding*](https://arxiv.org/abs/2211.17192); Willard and Louf, [*Efficient Guided Generation for LLMs*](https://arxiv.org/abs/2307.09702).
