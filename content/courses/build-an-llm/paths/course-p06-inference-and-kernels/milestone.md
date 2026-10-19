# What the System Does After Pass 6

After Pass 6 the system has a Python inference stack (samplers, the KV cache, a paged cache, quantization, and speculative and constrained decoding) that matches no-cache output token for token. It rests on sampling statistics, floating-point error analysis, and property tests. A set of standalone C exercises is optional depth, and nothing later depends on it. Everything from earlier passes keeps working. This page gives the commands the finished solution runs, what each produces, and why each piece is needed.

The stack is the [Inference](../../p08-inference/README.md) chapter: [sampling and logit processors](../../p08-inference/01-sampling-and-logit-processors.md), [the KV cache and generate](../../p08-inference/02-kv-cache-and-generate.md), [the radix prefix cache](../../p08-inference/04-radix-prefix-cache.md) with its [radix tree](../../../algorithms/16-systems-data-structures/07-radix-tree.md), [quantization](../../p08-inference/05-quantization.md), [speculative decoding](../../p08-inference/06-speculative-decoding.md), and [constrained decoding](../../p08-inference/07-constrained-decoding.md), with the formats of [FP8 and microscaling](../../../mathematics-for-computing/09-numerical-methods-and-floating-point/04-fp8-and-microscaling-formats.md). Its numerics and tests are [rejection sampling](../../../mathematics-for-computing/07-probability-statistics/06-rejection-sampling-and-residual-distributions.md), [error analysis and tolerance budgets](../../../mathematics-for-computing/09-numerical-methods-and-floating-point/03-error-analysis-condition-numbers-and-tolerance-budgets.md), [the floating-point problem set](../../../mathematics-for-computing/09-numerical-methods-and-floating-point/91-problem-set-b.md), and [benchmarks and perf gates](../../../software-craftsmanship/03-testing-mentality/04-benchmarks-and-perf-gates.md).

## The Python inference stack matches no-cache output token for token

The stack runs through the system's own entry point on the Llama model of Pass 5. With no cache and with the contiguous cache (using float16 keys and values), it produces the same greedy tokens. Quantized weights, using the FP8 and microscaling conversions, keep the model's predictions. Speculative decoding changes the speed and never the greedy output. Constrained decoding produces documents that all parse and validate. A tiny random two-layer model shows all of this in seconds; the same commands run on SmolLM2-135M-Instruct, and `bench decode` measures the cache speedup.

Why this matters: every serving optimization must be invisible in the output. Comparing each one with the no-cache greedy tokens is the definition of "correct" that the Rust engine of Pass 7 inherits. The cache is what makes decoding linear instead of quadratic in length, quantization is what fits larger models in memory, speculation is what trades spare compute for latency, and constrained decoding is what lets tool calls be parsed reliably.

The solution defines the Pass 6 inference forms of the `tinyllm` role under "Verbs of later passes" in [`cli-roles.md`](../../../../../course/contracts/spec/cli-roles.md): `generate` with `--cache` and `--spec`, `eval ppl`, and `bench decode`. Every verb keeps the rules of that page: exit 2 on a usage error, and the last stdout line is one JSON object.

```text
{tinyllm} generate --model <llama dir> (--prompt <text> | --prompt-file <file>)
              [--max-tokens N] [--greedy | --temperature T] [--seed S]
              [--cache none|contiguous|paged] [--kv-dtype f32|f16]
              [--spec ngram|prompt-lookup --k K] [--json-schema <file>]
    Pass 5's generate through the KV-cache section's generate with the
    chosen cache ("paged": the optional pure-Python PagedKVCache, behind
    the same cache interface; not checked here). --spec runs
    speculative_generate with an NGramDraft fitted on the prompt's ids
    (order 3) or a PromptLookupDraft. --json-schema decodes each prompt
    under the constrained-decoding constraint over the byte vocabulary
    until no token is allowed. Final line: {"ids": [every prompt's
    generated ids, concatenated], "text", "per_prompt" (with
    --prompt-file), and with --spec {"drafted", "accepted",
    "acceptance_rate", "target_calls"}, with --json-schema {"documents",
    "valid", "valid_fraction"} (valid: the text json-parses and satisfies
    the schema)}.

{tinyllm} eval ppl --model <llama dir> --data <text file> --quant int8|q4_g32|fp8_e4m3|mxfp4 [--tokens N]
    Next-token NLL over the first N tokens (windows of up to 128), once
    with the float32 model and once after quantize_model, which uses the
    Python format conversions. Final line: {"tokens", "ppl_fp32", "ppl",
    "increase_pct", "top1_agree" (how often the quantized model's argmax
    equals the float32 model's), "nll_delta" (|mean NLL difference| in
    nats)}.

{tinyllm} bench decode --model <llama dir> [--tokens N] [--prompt <text>]
    Greedy decode of N tokens with cache "none" and "contiguous" (best of
    two each). Final line: {"tokens", "tok_s_none", "tok_s_contiguous",
    "cache_speedup"}.
```

The [paged KV cache in Python](../../p08-inference/03-paged-kv-cache.md) is optional and is not checked here. The radix prefix cache is written in Rust (its tests run with `cargo test`), and no CLI verb above calls it: the Rust engine of Pass 7 reuses it. The C [paged KV block pool](../../p08-inference/08-paged-kv-block-pool.md) is optional depth.

Because the tiny model is random, its perplexity says little. On it, quantization is judged by top-1 agreement with the float32 model and by the NLL shift; on SmolLM2, it is judged by the perplexity increase.

## Optional depth: C kernels, runtime utilities, and data structures

The standalone C exercises cover runtime utilities ([the arena allocator](../../p09-kernels/02-arena-allocator-with-marks.md), [the thread pool](../../p09-kernels/03-thread-pool-and-parallel-for.md), and the block pool), the data structures from [a growable array](../../../algorithms/16-systems-data-structures/01-growable-array.md) to [heap top-k](../../../algorithms/16-systems-data-structures/04-binary-heap-top-k.md), numerical routines from [fast inverse square root](../../../mathematics-for-computing/09-numerical-methods-and-floating-point/05-fixed-point-iteration-and-rsqrt.md) to [low-precision conversions](../../../mathematics-for-computing/09-numerical-methods-and-floating-point/07-low-precision-c-conversions.md), and the kernels from [the tiled matmul](../../p09-kernels/04-tiled-batch-invariant-matmul.md) to [the elementwise kernels](../../p09-kernels/09-elementwise-kernels.md), with the [C primer](../../../programming-languages/02-language-and-tool-primers/03-c.md) behind them. They are optional, and nothing in later passes depends on them. Each compiles its own test binary, and numerical parity uses vectors emitted by the corresponding Python solution.

[Pass 7](../course-p07-serving-platform/README.md) carries the same correctness rule into the Rust engine.
