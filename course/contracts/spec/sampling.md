# Sampling: one op order in every language

<!-- modules: L8.1 (Python tinyllm/infer/sample.py), L10.1 (Rust tl-engine/src/sample.rs), L10.6 (disaggregated hand-off), L8.6 and L10.8 (speculative acceptance draws)
     conformance: parity/sampler (golden ids from shared fixture logits) -->

Given **the same logits and the same seed**, the Python sampler and the Rust sampler return the same token id and the same logprob (D11). That holds only if both apply the same operations in the same order with the same arithmetic, which this page fixes. End-to-end streams that come from different float paths (Python numpy logits vs Rust C-kernel logits) are compared greedily under the near-tie rule, never seeded.

Symbols: `V` the vocabulary size; `x` the model's next-token logits (f32, length `V`); `prompt` and `out` the prompt ids and the ids generated so far for this request; `rng` the request's generator.

## Parameters

Field names follow `SamplingParams` (L8.1) and the OpenAI request (openapi/openai-subset.v1.yaml); defaults turn each step off.

| Parameter | Default | Off when |
|---|---|---|
| `temperature` `T` | 1.0 | (0 means greedy) |
| `repetition_penalty` `r` | 1.0 | `r == 1` |
| `presence_penalty` `a_p` | 0.0 | `a_p == 0` |
| `frequency_penalty` `a_f` | 0.0 | `a_f == 0` |
| `top_k` `k` | 0 | `k == 0` or `k >= V` |
| `top_p` `p` | 1.0 | `p >= 1` |
| `min_p` `m` | 0.0 | `m == 0` |
| `seed` | engine-chosen | |

The generator is `rng = stream(seed, sample)` of spec/pcg32.md, created once per request.

## The steps

All arithmetic from step 1 on is IEEE f64. Every sum runs over ids in **ascending id order**.

1. **Widen.** `l[i] = f64(x[i])` for every id.
2. **Repetition penalty** (HF semantics). For each distinct id `i` in `prompt` and `out`: `l[i] = l[i] / r` if `l[i] > 0`, else `l[i] = l[i] * r`.
3. **Presence and frequency** (OpenAI semantics, generated ids only). For each id `i` that occurs `c > 0` times in `out`: `l[i] = l[i] - a_f * c - a_p`.
   The **logprob** reported for the chosen token is `log_softmax(l)[chosen]` over all `V` ids at this point, computed as in step 9, before temperature and filtering.
4. **Greedy.** If `T == 0`: return the id with the largest `l` (ties to the lowest id). No draw is taken; steps 5 to 11 are skipped.
5. **Temperature.** `l[i] = l[i] / T`.
6. **Top-k.** Order ids by `(l desc, id asc)` and keep the first `k`.
7. **Top-p.** Over the kept ids, compute `q` by step 9. Walk the kept ids in `(l desc, id asc)` order adding `q` to a running sum `s`; keep every id up to and including the first one where `s >= p` (the token that crosses `p` stays).
8. **Min-p.** Over the kept ids, compute `q` by step 9 and keep the ids with `q[i] >= m * max(q)`.
9. **Softmax** over the kept set `K`: `M = max over K of l`, `e[i] = exp(l[i] - M)`, `Z = sum over K of e` (ascending id order), `q[i] = e[i] / Z`. `exp` and `log` are the platform's f64 functions (Python `math.exp`, Rust `f64::exp`), which is why parity is checked on one machine.
10. **Draw** exactly one `u = rng.uniform_f64()` per sampled token.
11. **Inverse CDF.** Walk `K` in ascending id order adding `q[i]` to `c`; return the first id with `u < c`. If rounding leaves `c` below `u` after the last id, return the largest id in `K` with `q > 0`. Ids whose logit is `-inf` never enter `K`, so the fallback is always a token the processors kept.

Steps 7 and 8 recompute `q` on the current kept set, so `top_p` sees the distribution after `top_k`, and `min_p` the one after `top_p`.

## Draw accounting

Exactly one `uniform_f64` (two `next_u32`) per sampled token and none per greedy token, so the generator's position after `n` sampled tokens is known. Disaggregated serving relies on it: the prefill worker samples the first token and reports `rng_draws_consumed` (1 or 0) in `KvHandle`; the decode worker creates `stream(seed, sample)` and calls `uniform_f64()` that many times before its first draw (proto/tl/engine/v1/engine.proto). Speculative decoding's acceptance tests (M07.6) draw from the same `rng`, in the order L8.6 defines, and that order is part of the L8.6 and L10.8 parity contract.

## Worked example

`V = 5`, `x = [1.0, 3.0, 2.0, 3.0, -1.0]`, no history, `T = 1`, `top_k = 3`, `top_p = 0.8`, `seed = 0`.

- **Steps 1 to 3.** Nothing changes. Logprobs: `M = 3`, `Z = e^-2 + 1 + e^-1 + 1 + e^-4 = 2.52153`, so ids 1 and 3 each have logprob `-ln Z = -0.92487`.
- **Step 5.** `T = 1`: no change.
- **Step 6.** Order by `(l desc, id asc)`: `(3.0, id 1), (3.0, id 3), (2.0, id 2), (1.0, id 0), (-1.0, id 4)`. Keep ids 1, 3, 2.
- **Step 7.** `q` over `{1, 2, 3}` (ascending ids for `Z`): `Z = 1 + e^-1 + 1 = 2.3679`, `q[1] = q[3] = 0.42232`, `q[2] = 0.15536`. Walk 1, 3, 2: `s = 0.42232` (below 0.8), `s = 0.84464` (crosses 0.8 at id 3): keep `{1, 3}`.
- **Step 9.** `q[1] = q[3] = 0.5`.
- **Step 10.** `stream(0, sample)`: `child_seed(0, 4) = 0xF88BB8A8724C81EC`, and its first `uniform_f64()` is `u = 0.80209...`.
- **Step 11.** Walk 1, 3: `c = 0.5` (`u` not below it), `c = 1.0` (`u < 1.0`): the token is **id 3**, with logprob `-0.92487`.
