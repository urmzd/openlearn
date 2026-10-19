# Inference

By the end of the previous chapter you have a Llama-family model that reproduces a real checkpoint's logits. This chapter is about turning those logits into text, fast and exactly. Fast, because decoding is memory-bound: each new token reads every weight and the whole KV cache once, so bytes moved, not FLOPs, set the speed. Exactly, because two engines (the Python reference and the Rust server) must emit the same tokens from the same request, and every optimization must leave the output unchanged or identically distributed.

The arc runs from the single token to the whole serving-ready toolkit. It starts with the sampler, specified to the bit: penalties, temperature, top-k, top-p, min-p, an f64 softmax, one uniform draw, and the inverse CDF, so Python and Rust agree. It then caches keys and values so each step processes one token, and streams byte-level text without half characters. The middle of the chapter is about memory: paging the KV cache into fixed-size blocks with a block table per sequence, sharing prefixes by reference count and copy on write, finding shared prefixes across requests with a radix tree, and shrinking weights and the cache with quantization. The chapter ends with two ways to change what each pass produces: speculative decoding, which drafts several tokens cheaply and verifies them in one pass without changing the output, and constrained decoding, which compiles a regex or JSON schema into token masks so invalid output is impossible.

> **Learning objectives**
> - Implement a bit-exact sampling pipeline and explain why its order, sums, and tie rules matter.
> - Build a KV cache whose output equals full recomputation, with rollback and an incremental UTF-8 detokenizer.
> - Explain paging, reference counting, copy on write, and prefix caching for the KV cache.
> - Quantize weights and KV to int8, int4, fp8, and MXFP4 with a derived error budget.
> - Prove that speculative decoding preserves the target distribution, and compile a regex or JSON schema subset into token masks.

## Before you begin

You should have the Llama-family model of [the modern block](../08-modern-block/README.md), in particular [GQA attention](../08-modern-block/05-gqa-attention.md) and [the Llama-family model](../08-modern-block/09-llama-family-model.md). Three sections of *Mathematics for Computing* are used just in time: [rejection sampling and residual distributions](../../mathematics-for-computing/08-probability-statistics/07-rejection-sampling-and-residual-distributions.md) for speculative decoding, [error analysis and tolerance budgets](../../mathematics-for-computing/10-numerical-methods-and-floating-point/03-error-analysis-condition-numbers-and-tolerance-budgets.md) and [FP8 and microscaling formats](../../mathematics-for-computing/10-numerical-methods-and-floating-point/04-fp8-and-microscaling-formats.md) for quantization, with [the floating-point problem set, part b](../../mathematics-for-computing/10-numerical-methods-and-floating-point/09-problem-set-b.md) for practice. The optional C sections rely on the structures in [systems data structures](../../algorithms/13-systems-data-structures/README.md) from *Algorithms*: [the growable array](../../algorithms/13-systems-data-structures/01-growable-array.md), [the Swiss table](../../algorithms/13-systems-data-structures/02-swiss-table.md), [the intrusive LRU](../../algorithms/13-systems-data-structures/03-intrusive-list-and-lru.md), and [the radix tree](../../algorithms/13-systems-data-structures/07-radix-tree.md).

## In this chapter

1. [Sampling and logit processors](01-sampling-and-logit-processors.md): the eleven-step sampling pipeline of `spec/sampling.md`, with penalties, filters, and a single uniform draw, written so Python and Rust emit the same token.
2. [The KV cache, incremental decoding, and generate](02-kv-cache-and-generate.md): an exact KV cache with committed length and rollback, the generate loop, an incremental UTF-8 detokenizer, and safe streaming around stop strings.
3. [A paged KV cache in Python](03-paged-kv-cache.md): an optional section that stores the cache in fixed-size blocks with block tables, reference counts, and copy on write, matching the contiguous cache exactly.
4. [A radix prefix cache](04-radix-prefix-cache.md): block-granular prefix matching over a radix tree in Rust, with locks that pin running requests and LRU leaf eviction.
5. [Quantization: int8, int4 groups, fp8, and the KV cache](05-quantization.md): symmetric quantization schemes in the kernel's byte layout, each with a derived error budget and a measured perplexity cost.
6. [Speculative decoding](06-speculative-decoding.md): n-gram, prompt-lookup, and model drafts, verified in one target pass with rules that keep greedy output identical and sampled output exactly distributed.
7. [Constrained decoding: from regex to token masks](07-constrained-decoding.md): regex to NFA to canonical DFA to token masks, and a JSON schema subset compiled to a regex for tool calls.
8. [A paged KV block pool in C](08-paged-kv-block-pool.md): an optional C block pool with free, used, and cached states, chained block hashes, an LRU, and a checksummed export format.

## Further reading

- Kwon et al., [*Efficient Memory Management for LLM Serving with PagedAttention*](https://arxiv.org/abs/2309.06180) (the vLLM block manager); Zheng et al., [SGLang](https://arxiv.org/abs/2312.07104) (RadixAttention); [llama.cpp](https://github.com/ggml-org/llama.cpp) (`llama_batch`).
- Leviathan, Kalman, and Matias, [*Fast Inference from Transformers via Speculative Decoding*](https://arxiv.org/abs/2211.17192); Willard and Louf, [*Efficient Guided Generation for LLMs*](https://arxiv.org/abs/2307.09702).
