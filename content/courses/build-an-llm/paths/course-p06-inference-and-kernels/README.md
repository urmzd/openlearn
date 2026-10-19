# Pass 6: Inference and Kernels

A model that matches Hugging Face is correct, but a correct model decoded naively is slow: without a cache, every new token recomputes attention over the whole prefix. This pass builds the Python inference stack that makes decoding fast without changing its output: samplers and logit processors, the KV cache with incremental decoding, a paged cache and a radix prefix cache, quantized weights, speculative decoding, and constrained decoding that guarantees every document parses and validates.

The rule that runs through the pass is that every optimization must be invisible in the output. Each one is checked against the no-cache greedy tokens, and that definition of correctness is what the Rust engine of Pass 7 inherits. A second half, entirely optional, writes the runtime pieces and kernels in standalone C: an arena allocator, data structures, a thread pool, a paged block pool, numerical approximations, a tiled matmul, softmax, flash and paged attention, and fused quantized matmul. Nothing later depends on them.

> **Learning objectives**
> - Explain how error analysis and tolerance budgets decide when two numerical results are equal enough.
> - Explain how the KV cache makes decoding linear in length and how the paged and radix caches share memory.
> - Explain how quantization formats trade accuracy for memory and how that trade is measured.
> - Explain why speculative decoding never changes the greedy output and how constrained decoding masks tokens.

## Before you begin

This pass needs [Pass 5](../course-p05-transformer/README.md); its Llama-family model is the model every section here runs on.

## In this pass

1. [Rejection sampling and residual distributions](../../../mathematics-for-computing/08-probability-statistics/07-rejection-sampling-and-residual-distributions.md). Afterwards you can explain why rejection sampling with a residual distribution yields exact samples from the target.
2. [Error analysis, condition numbers, and tolerance budgets](../../../mathematics-for-computing/10-numerical-methods-and-floating-point/03-error-analysis-condition-numbers-and-tolerance-budgets.md). Afterwards you can explain how condition numbers and error analysis set a tolerance budget.
3. [FP8 and microscaling formats](../../../mathematics-for-computing/10-numerical-methods-and-floating-point/04-fp8-and-microscaling-formats.md). Afterwards you can explain the FP8 E4M3 and E5M2 formats and how MXFP4 and MXFP8 share an E8M0 scale per block.
4. [Problem set B: condition numbers, error bounds, Newton, polynomial approximation, and low precision](../../../mathematics-for-computing/10-numerical-methods-and-floating-point/09-problem-set-b.md). Afterwards you can work every problem in the set and explain each answer.
5. [Sampling and logit processors](../../09-inference/01-sampling-and-logit-processors.md). Afterwards you can explain the order in which logit processors apply temperature, top-k, top-p, and penalties before sampling.
6. [The KV cache, incremental decoding, and generate](../../09-inference/02-kv-cache-and-generate.md). Afterwards you can explain how the KV cache makes decode incremental and how the UTF-8 detokenizer emits only complete characters.
7. [A paged KV cache in Python](../../09-inference/03-paged-kv-cache.md). Afterwards you can explain how a paged KV cache maps logical positions to physical blocks.
8. [A radix tree over token ids](../../../algorithms/13-systems-data-structures/07-radix-tree.md). Afterwards you can explain how the radix tree over token ids splits edges and how the index-linked LRU leaf list picks evictions.
9. [A radix prefix cache](../../09-inference/04-radix-prefix-cache.md). Afterwards you can explain how the radix prefix cache matches at block granularity, locks shared prefixes, and evicts LRU leaves.
10. [Quantization: int8, int4 groups, fp8, and the KV cache](../../09-inference/05-quantization.md). Afterwards you can explain how int8, packed int4 group, and fp8 quantization scale weights and how KV quantization works.
11. [Speculative decoding: n-gram, prompt-lookup, and model drafts](../../09-inference/06-speculative-decoding.md). Afterwards you can explain why speculative decoding with n-gram, prompt-lookup, and model drafts preserves the target distribution.
12. [Constrained decoding: from regex to token masks](../../09-inference/07-constrained-decoding.md). Afterwards you can trace how a regex becomes a DFA and then per-token masks, and how the JSON-schema subset compiles to one.
13. [Benchmarks and perf gates](../../../software-craftsmanship/03-testing-mentality/04-benchmarks-and-perf-gates.md). Afterwards you can explain how a benchmark controls noise and how a perf gate decides a regression.
14. [What the System Does After Pass 6](milestone.md). Afterwards you can explain why the Python inference stack matches no-cache output token for token with every optimization on.

## Optional depth: standalone C

These sections write runtime utilities, data structures, and kernels in C. Each compiles its own test binary, and numerical parity is checked against vectors emitted by the corresponding Python solution:

15. [C runtime support and an arena allocator with marks](../../10-kernels/01-arena-allocator-with-marks.md). Afterwards you can explain how the arena allocator hands out aligned memory and how marks free in bulk.
16. [A growable array](../../../algorithms/13-systems-data-structures/01-growable-array.md). Afterwards you can explain how tl_vec grows geometrically and why push is amortized O(1).
17. [A Swiss table](../../../algorithms/13-systems-data-structures/02-swiss-table.md). Afterwards you can explain how the Swiss table probes with control bytes and groups.
18. [An intrusive list and an LRU](../../../algorithms/13-systems-data-structures/03-intrusive-list-and-lru.md). Afterwards you can explain how an intrusive list links nodes without allocation and how it builds an LRU.
19. [Heap top-k in C](../../../algorithms/13-systems-data-structures/04-binary-heap-top-k.md). Afterwards you can explain how a size-k min-heap keeps the top k in O(n log k).
20. [Thread pool and tl_parallel_for](../../10-kernels/02-thread-pool-and-parallel-for.md). Afterwards you can explain how the thread pool splits a range in tl_parallel_for and waits for every chunk.
21. [A paged KV block pool in C](../../09-inference/08-paged-kv-block-pool.md). Afterwards you can explain the paged KV block pool's format v1 and how blocks are allocated, shared, and freed.
22. [Fixed-point iteration and fast inverse square root in C](../../../mathematics-for-computing/10-numerical-methods-and-floating-point/05-fixed-point-iteration-and-rsqrt.md). Afterwards you can explain why the fast inverse square root's magic constant gives a good first guess and how a Newton step refines it.
23. [Polynomial approximation and range reduction: expf in C](../../../mathematics-for-computing/10-numerical-methods-and-floating-point/06-polynomial-approximation-and-expf.md). Afterwards you can explain how expf reduces its range and approximates the remainder with a polynomial.
24. [Low-precision conversions in C](../../../mathematics-for-computing/10-numerical-methods-and-floating-point/07-low-precision-c-conversions.md). Afterwards you can explain how the C conversions round to bf16, fp16, and fp8 and handle NaN and overflow.
25. [Cache-blocked, packed, batch-invariant matmul in C](../../10-kernels/03-tiled-batch-invariant-matmul.md). Afterwards you can explain how cache blocking and packing speed up the C matmul and why a fixed reduction order makes it batch-invariant.
26. [Softmax in C: three-pass and online two-pass](../../10-kernels/04-softmax-three-pass-and-online.md). Afterwards you can explain how the online two-pass softmax rescales its running sum to save a pass.
27. [FlashAttention forward in C](../../10-kernels/05-flash-attention-forward.md). Afterwards you can explain how FlashAttention's forward tiles keys and values and keeps a running max and sum.
28. [Paged attention for decode in C](../../10-kernels/06-paged-attention-decode.md). Afterwards you can explain how paged attention for decode reads keys and values through the block table.
29. [Fused int4 and int8 dequantize-matmul in C](../../10-kernels/07-fused-quantized-matmul.md). Afterwards you can explain how the fused int4 and int8 dequantize-matmul unpacks and scales weights inside the inner loop.
30. [Elementwise kernels in C: RMSNorm, RoPE, SiLU-mul, embedding, add, argmax](../../10-kernels/08-elementwise-kernels.md). Afterwards you can explain each C elementwise kernel (RMSNorm, RoPE, SiLU-mul, embedding, add, argmax) and the numerics it must match.

The stages are also in this pass's manifest, [`path.tsv`](path.tsv). [Pass 7](../course-p07-serving-platform/README.md) moves inference into the Rust engine and puts a production platform around it.
