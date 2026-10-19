# What the system does after Pass 6

**Pass result**: Python inference stack (sampler, KV, paged KV, quant, spec decoding, constrained decoding), plus optional standalone C exercises.

Each capability below is built from the listed modules. The sections that follow give the commands the finished solution runs, what each one produces, and why each piece is needed. Everything from earlier passes keeps working.

| What the system does | Built from |
|---|---|
| The Python inference stack matches no-cache output token for token. | `ds.07`, `L8.1`, `L8.2`, `L8.4`, `L8.5`, `L8.6`, `L8.7`, `M09.4` |
| Inference: cached, quantized, speculative, and constrained decoding. | `M07.6`, `M09.3`, `S-M09b`, `craft.06` |
| Optional C kernels, runtime utilities, and data structures. | `ds.01`, `ds.02`, `ds.03`, `ds.04`, `rt.02`, `rt.03`, `rt.04`, `M09.7`, `M09.5`, `M09.6`, `L9.1`, `L9.2`, `L9.3`, `L9.4`, `L9.5`, `L9.6`, `lang.03`, `craft.06` |


## Details

## The Python inference stack matches no-cache output token for token

The Python inference stack, run through the system's own entry point, on the Llama
model of MS-L7: the same greedy tokens with no cache, the contiguous cache
(L8.2, with float16 keys and values); quantized weights (L8.5, using M09.4's FP8 and
microscaling conversions) that keep the model's predictions; speculative
decoding (L8.6) that changes the speed and never
the greedy output; and constrained decoding (L8.7) whose every document
parses and validates. PR CI runs it on the committed tiny model; the
nightly run uses SmolLM2-135M-Instruct pulled by MS-L7; the cache speedup
is a local perf step.

The solution defines the Pass 6 inference forms of the `tinyllm` role
([`cli-roles.md`](../../../../../course/contracts/spec/cli-roles.md), "Verbs of later passes": generate with --cache and
--spec, eval ppl, bench decode). Every verb keeps the rules of that page:
exit 2 on a usage error, the last stdout line is one JSON object.

  {tinyllm} generate --model <llama dir> (--prompt <text> | --prompt-file <file>)
                [--max-tokens N] [--greedy | --temperature T] [--seed S]
                [--cache none|contiguous|paged] [--kv-dtype f32|f16]
                [--spec ngram|prompt-lookup --k K] [--json-schema <file>]
      MS-L7's generate through L8.2's generate with the chosen cache
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

DESIGN 4.3 step (5), "the L8.4 course tests pass from cargo test", and the
rt.04 block pool are this milestone's `requires`: a fresh pass of L8.4 ran
its Rust course tests with cargo, as MS-L9 treats its sanitizer step
(DEVIATIONS B83-08). The tiny model is random, so its perplexity says
little: the PR quantization steps bound top-1 agreement and the NLL shift;
the nightly steps keep DESIGN 4.3's perplexity budgets on SmolLM2.

## Inference: cached, quantized, speculative, and constrained decoding

MS-L8 verifies the core Python inference stack. Standalone C exercises are
optional and deliberately do not gate this pass. Earlier pass smoke steps
rerun through the spiral invariant.

## Optional C kernels, runtime utilities, and data structures

requirement of MS-P6 or any production runtime milestone. Each required
module's C check compiles its own test binary, and numerical parity uses the
checked-in vectors emitted by the corresponding Python reference.