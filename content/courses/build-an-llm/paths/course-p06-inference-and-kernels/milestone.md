# What the system does after Pass 6

**Pass result**: Python inference stack (sampler, KV, paged KV, quant, spec decoding, constrained decoding), plus optional standalone C exercises.

Each capability below is built from the listed modules. The sections that follow give the commands the finished solution runs, what each one produces, and why each piece is needed. Everything from earlier passes keeps working.

| What the system does | Built from |
|---|---|
| The Python inference stack matches no-cache output token for token. | `ds.07`, `L8.1`, `L8.2`, `L8.4`, `L8.5`, `L8.6`, `L8.7`, `M09.4` |
| The numerics and testing the stack rests on: sampling statistics, floating-point error, and property tests. | `M07.6`, `M09.3`, `S-M09b`, `craft.06` |
| Optional depth: C kernels, runtime utilities, and data structures. Nothing later depends on them. | `ds.01`, `ds.02`, `ds.03`, `ds.04`, `rt.02`, `rt.03`, `rt.04`, `M09.7`, `M09.5`, `M09.6`, `L9.1`, `L9.2`, `L9.3`, `L9.4`, `L9.5`, `L9.6`, `lang.03`, `craft.06` |


## Details

## The Python inference stack matches no-cache output token for token

The Python inference stack, run through the system's own entry point, on the Llama
model of Pass 5: the same greedy tokens with no cache, the contiguous cache
(L8.2, with float16 keys and values); quantized weights (L8.5, using M09.4's FP8 and
microscaling conversions) that keep the model's predictions; speculative
decoding (L8.6) that changes the speed and never
the greedy output; and constrained decoding (L8.7) whose every document
parses and validates. A tiny random two-layer model shows all of this in
seconds; the same commands run on SmolLM2-135M-Instruct, and `bench decode`
measures the cache speedup.

Why it is needed: every serving optimization must be invisible in the
output. Comparing each one with the no-cache greedy tokens is the
definition of "correct" that the Rust engine of Pass 7 inherits; the cache
is what makes decoding linear instead of quadratic in length, quantization
is what fits larger models in memory, speculation is what trades spare
compute for latency, and constrained decoding is what lets tool calls be
parsed reliably.

The solution defines the Pass 6 inference forms of the `tinyllm` role
([`cli-roles.md`](../../../../../course/contracts/spec/cli-roles.md), "Verbs of later passes": generate with --cache and
--spec, eval ppl, bench decode). Every verb keeps the rules of that page:
exit 2 on a usage error, the last stdout line is one JSON object.

  {tinyllm} generate --model <llama dir> (--prompt <text> | --prompt-file <file>)
                [--max-tokens N] [--greedy | --temperature T] [--seed S]
                [--cache none|contiguous|paged] [--kv-dtype f32|f16]
                [--spec ngram|prompt-lookup --k K] [--json-schema <file>]
      Pass 5's generate through L8.2's generate with the chosen cache
      ("paged": the optional L8.3's pure-Python PagedKVCache, behind L8.2's
      cache interface; not checked here). --spec runs L8.6's
      speculative_generate with an NGramDraft fitted on the prompt's ids
      (order 3) or a PromptLookupDraft. --json-schema decodes each prompt
      under L8.7's constraint over the byte vocabulary until no token is
      allowed. Final line: {"ids": [every prompt's generated ids,
      concatenated], "text", "per_prompt" (with --prompt-file), and with
      --spec {"drafted", "accepted", "acceptance_rate", "target_calls"},
      with --json-schema {"documents", "valid", "valid_fraction"} (valid: the
      text json-parses and satisfies the schema)}.

  {tinyllm} eval ppl --model <llama dir> --data <text file> --quant int8|q4_g32|fp8_e4m3|mxfp4 [--tokens N]
      Next-token NLL over the first N tokens (windows of up to 128), once
      with the float32 model and once after L8.5's quantize_model, which uses
      M09.4's Python format conversions. Final
      line: {"tokens", "ppl_fp32", "ppl", "increase_pct", "top1_agree" (how
      often the quantized model's argmax equals the float32 model's),
      "nll_delta" (|mean NLL difference| in nats)}.

  {tinyllm} bench decode --model <llama dir> [--tokens N] [--prompt <text>]
      Greedy decode of N tokens with cache "none" and "contiguous" (best of
      two each). Final line: {"tokens", "tok_s_none", "tok_s_contiguous",
      "cache_speedup"}.

L8.4's radix prefix cache is written in Rust (its tests run with
`cargo test`) and no CLI verb above calls it: the Rust engine of Pass 7
reuses it. The C paged-KV block pool (rt.04) is optional depth. The tiny model is random, so its perplexity says
little; on it, quantization is judged by top-1 agreement with the float32
model and the NLL shift, and on SmolLM2 by the perplexity increase.

## Optional depth: C kernels, runtime utilities, and data structures

The standalone C exercises (kernels, runtime utilities, data structures)
are optional and nothing in later passes depends on them. Each compiles its
own test binary, and numerical parity uses vectors emitted by the
corresponding Python solution.
