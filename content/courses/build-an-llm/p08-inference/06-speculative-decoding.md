<!-- ol:module L8.6 -->
# Speculative decoding: n-gram, prompt-lookup, and model drafts

## Overview

| | |
|---|---|
| **Module** | `L8.6` · build · Python · Pass 6 · 3 to 4 h |
| **You build** | `python/tinyllm/infer/spec.py`: `verify_draft`, `speculative_generate`, and three drafts: `NGramDraft`, `PromptLookupDraft`, `ModelDraft` (the `DraftModel` protocol is in the contract) |
| **Contract** | [`course/contracts/py/tinyllm/infer/spec.pyi`](../../../../course/contracts/py/tinyllm/infer/spec.pyi) |
| **Tests** | `course/tests/L8.6/test_spec.py` (and the toy models in `_toy.py`) (what they check: section 4) |
| **Needs** | [`M07.6` rejection sampling](../../../tracks/math/07-probability-statistics/06-rejection-sampling-and-residual-distributions.md) (`speculative_step`), [`L8.2` KV cache and generate](02-kv-cache-and-generate.md) (`KVCache.truncate`, `cache_dims`, `Generation`), [`L2.1` n-gram LM](../p02-statistical-lm/01-ngram-kneser-ney.md) (`NGramLM`), [`L8.1` sampling](01-sampling-and-logit-processors.md) (`sample`, `sampling_distribution`, `token_logprobs`, `request_rng`) (or `--ref-deps`). Reading: [`M11.1` entropy and KL](../../../tracks/math/11-information-theory/01-entropy-cross-entropy-and-kl.md) |
| **Used by** | `L10.8` speculative decoding in the Rust engine, held to this module on shared draft and target logits (joins `used_by` when registered) |
| **Milestone** | `MS-L8` (step `spec-ngram-greedy`: token-identical output, `acceptance_rate` reported) |
| **Optional depth** | Leviathan, Kalman, and Matias, *Fast Inference from Transformers via Speculative Decoding* (ICML 2023); Chen et al., *Accelerating Large Language Model Decoding with Speculative Sampling* (2023); Saxena, *Prompt Lookup Decoding* (2023) |

## Key Takeaways

- **Decoding one token costs one pass over the weights; scoring $k + 1$ tokens costs about the same**, so a draft that guesses $k$ tokens and a target that checks them in one pass can emit several tokens per pass (`test_stats_and_one_pass_per_round`).
- **Greedy verification never changes the output**: keep drafts while they equal the target's argmax, then add the target's own token (`test_speculative_greedy_equals_greedy`).
- **Sampled verification keeps the target's distribution exactly**: accept $x \sim q$ with probability $\min(1, p(x)/q(x))$, otherwise draw from $\max(0, p - q)$ normalized; enumerated exactly, the output is $p$ as fractions (`test_one_token_output_is_the_target_exactly`, `test_two_token_draft_is_exact_at_each_position`).
- **Rejected drafts must leave no trace in the KV cache**: truncate back to the kept length, then feed the replacement token first next round (`test_rejected_drafts_are_rolled_back`).
- **Cheap drafts are enough** for repetitive text: copying from the prompt (`test_prompt_lookup_hand_example`) or an n-gram model (`test_ngram_draft_follows_the_model`) cost nothing to run.

## How to work this chapter

```bash
ol start L8.6              # stubs spec.py into your repo, contract alongside
ol tests L8.6              # read the test catalog first
ol check L8.6              # exit code is the verdict
ol check L8.6 --ref-deps   # only if you skipped M07.6, L8.2, L2.1, or L8.1
ol diff  L8.6              # after passing: your code against the reference
```

You also write graded tests (rung R5) in `python/tests/l8-6-spec/`: they run against the reference with one planted bug at a time, and the share they catch is your grade.

---

## 1. Why now

Your `generate` (L8.2) produces one token per forward pass. Profile it on SmolLM2 and the pass is dominated by reading every weight from memory once; the arithmetic for one token is small. A pass over 5 tokens (with the KV cache, 5 new query rows) reads the same weights once and costs barely more. So decoding wastes most of what each pass could do. Speculative decoding spends that slack: something cheap guesses the next $k$ tokens, the target scores all of them in one pass, and every correct guess is a token you did not pay a pass for. The trap is correctness: a guess that is accepted when the target would not have produced it changes the output, and for sampling it changes the distribution. This module builds the verification rules that make the speedup free of either, plus three drafts, and Part 10's Rust engine (L10.8) is checked against it.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $V$ | vocabulary size | int |
| $k$ | the number of draft tokens per round | int $\ge 0$ |
| $x_1, \dots, x_m$ | the draft tokens of one round, $m \le k$ | ids |
| $p_i$ | the target's sampling distribution (L8.1) at draft position $i$, given everything before $x_i$ | float64 $[V]$ |
| $q_i$ | the draft's distribution that $x_i$ was drawn from (one-hot for a deterministic draft) | float64 $[V]$ |
| $u$ | a uniform draw in $[0, 1)$ from the request's PCG32 stream | float |
| $\alpha = \sum_v \min(p(v), q(v))$ | the acceptance probability of one draft token | $[0, 1]$ |
| $\mathrm{TV}(p, q) = \tfrac12 \sum_v \lvert p(v) - q(v) \rvert$ | total variation distance; $\alpha = 1 - \mathrm{TV}$ | $[0, 1]$ |

### 2.1 The round, and the cache

The target cache holds the positions that are **kept**. One round:

1. the draft proposes $x_1..x_m$ after the context (prompt plus generated);
2. the target runs **one** forward pass over the tokens not yet in its cache: the **pending** token (the previous round's last token, or the whole prompt in the first round) followed by $x_1..x_m$. Its last $m + 1$ rows of logits are the target's predictions after the context, after $x_1$, ..., after $x_m$;
3. `verify_draft` keeps $x_1..x_n$ for some $n \le m$ and adds one more token $y$ of the target's own (a correction, or a bonus when all $m$ were kept);
4. the cache now holds $m$ draft positions, but only $n$ are kept: `KVCache.truncate(committed + len(pending) + n)` drops the rest. $y$ was never fed, so it becomes the next round's pending token.

Every round emits $n + 1 \ge 1$ tokens for one target pass, so speculation can never be slower in passes than plain decoding (with $k = 0$ it **is** plain decoding).

### 2.2 Exactness: why rejection sampling gives exactly $p$

Draw $x \sim q$. Keep it with probability $\min(1, p(x)/q(x))$ (M07.6's `rejection_accept`: $u < p(x)/q(x)$). If it is rejected, draw $y$ from the **residual** $r = \max(0, p - q) / Z$ with $Z = \sum_v \max(0, p(v) - q(v))$. For any token $v$:

$$P(\text{out} = v) = \underbrace{q(v) \min\!\left(1, \tfrac{p(v)}{q(v)}\right)}_{\text{kept}} + \underbrace{\left(1 - \alpha\right) r(v)}_{\text{replaced}} = \min(q(v), p(v)) + \max(0, p(v) - q(v)) = p(v),$$

because $1 - \alpha = 1 - \sum_v \min(p, q) = \sum_v \max(0, p - q) = Z$. Nothing in this argument needs $q$ to be good: a bad draft is rejected more often, never wrong. A **deterministic** draft (prompt lookup, greedy n-gram) proposes a fixed $x$, which is a draw from the one-hot $q = e_x$: then $x$ is kept with probability $p(x)$ and the residual is $p$ with $x$ removed, renormalized.

At position $i > 1$ the same step runs with $p_i$, the target's distribution **after** $x_1..x_{i-1}$ (which the target already computed in the same pass), and only if $x_{i-1}$ was kept. Conditioned on the kept prefix, each position is a fresh exact step, so every emitted token follows the target's distribution given the tokens before it. Penalties (repetition, presence, frequency) are part of $p_i$: their history includes the drafts kept so far in this round.

### 2.3 Greedy is the temperature-0 limit

At temperature 0 the target's distribution is one-hot on its argmax $g_i$ (L8.1, ties to the lowest id). The rule becomes: keep $x_i$ while $x_i = g_i$; at the first mismatch emit $g_i$ and stop; after $m$ matches emit $g_{m+1}$. No uniform is drawn. The emitted tokens are exactly the target's greedy tokens, which is the claim the milestone checks on a real model.

### 2.4 How fast: expected tokens per pass

If each draft token is accepted independently with probability $\alpha$, a round of $k$ drafts emits $1 + \alpha + \alpha^2 + \dots + \alpha^k = \frac{1 - \alpha^{k+1}}{1 - \alpha}$ tokens on average. With $\alpha = 0.8$ and $k = 4$ that is 3.36 tokens per target pass. The cost side is the draft: an n-gram lookup or a prompt lookup is free, a smaller model costs $k$ of its own passes per round. `acceptance_rate` (accepted drafts over proposed drafts) is the number that tells you whether a draft is worth it. **Prompt lookup** shines when the output repeats the input (code edits, summaries quoting their source); an **n-gram** model trained on similar text catches common continuations; a **model draft** generalizes best and costs the most.

### 2.5 Randomness, in a fixed order

One request generator (`request_rng(seed)`, L8.1) serves the draft and the verifier in program order. Per verified draft position the verifier always draws **two** uniforms, `u_accept` then `u_resample` (M07.6's `speculative_step` takes both, and drawing both keeps the count independent of the outcome); the bonus token is L8.1's `sample` (one uniform). The Rust port (L10.8) draws in the same order, so on the same logits it emits the same ids.

## 3. Worked example by hand

**Sampled.** $V = 3$, target $p = [\tfrac12, \tfrac14, \tfrac14]$, draft $q = [\tfrac14, \tfrac12, \tfrac14]$, draft token $x = 1$.

| Step | Computation | Result |
|---|---|---|
| acceptance probability | $\min(1, p_1/q_1) = \min(1, \tfrac{1/4}{1/2})$ | $\tfrac12$ |
| $u_{accept} = 0.7$ | $0.7 < 0.5$? no | rejected |
| residual | $\max(0, p - q) = [\tfrac14, 0, 0]$, $Z = \tfrac14$ | $r = [1, 0, 0]$ |
| $u_{resample} = 0.4$ | inverse CDF of $r$ | token 0; emitted `[0]`, 0 accepted |

With $u_{accept} = 0.3 < 0.5$ the draft is kept, $u_{resample}$ is drawn and ignored, and the bonus token comes from the target's next row with a third uniform. Averaged over the draft as well ($x \sim q$): token 1 comes out only when $x = 1$ is kept, with probability $q_1 \cdot \tfrac12 = \tfrac14 = p_1$; token 0 when $x = 0$ (always kept, $\tfrac14$) or when $x = 1$ is rejected ($\tfrac12 \cdot \tfrac12$), total $\tfrac12 = p_0$. That is section 2.2's identity, which `test_one_token_output_is_the_target_exactly` checks by enumeration on a larger example.

**Greedy.** Target rows with argmax 2, 0, 1 (`[[0, 1, 3], [2, 1, 0], [0, 5, 1]]`) and draft `[2, 1]`: position 0, draft 2 = argmax 2, keep; position 1, draft 1 $\ne$ argmax 0, emit 0 and stop. Result `[2, 0]`, one accepted, no uniform drawn. Draft `[2, 0]` matches twice and the bonus is row 2's argmax, giving `[2, 0, 1]`.

**Prompt lookup.** Context `1 2 3 9 1 2`, $k = 3$, `max_ngram = 3`: the last 3-gram `9 1 2` never occurred earlier; the last 2-gram `1 2` occurred at the start, followed by `3 9 1`: that is the draft.

## 4. The interface

```python
class DraftModel(Protocol):
    def propose(self, ctx: Sequence[int], k: int, rng: Optional[UniformSource]) -> tuple[list[int], Optional[NDArray]]
class NGramDraft:        def __init__(self, lm: NGramLM)
class PromptLookupDraft: def __init__(self, max_ngram: int = 3, min_ngram: int = 1)
class ModelDraft:        def __init__(self, model, temperature: float = 1.0)
def verify_draft(target_logits, draft_ids, draft_probs, p, history, rng, prompt=()) -> tuple[list[int], int]
def speculative_generate(target, draft, tok, prompt, p, k=4, kv_dtype=np.float32, eos_ids=()) -> Generation
```

`rng=None` asks a draft for its greedy guess (and no distribution). `speculative_generate` reports `drafted`, `accepted`, `acceptance_rate`, and `target_calls` in `Generation.stats`, besides L8.2's fields. The tests use `_toy.BagLM`, a model whose logits at a position depend only on the tokens up to it and are computed row by row in float64, so whether the target scores one token or five cannot change a bit; and it asserts on every call that the cache holds exactly the positions before the chunk.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example` | unit, smoke | section 3, sampled: `[0]` with 0 accepted for $u = 0.7, 0.4$; `[1, 2]` for $0.3, 0.9, 0.1$; three uniforms | the draw order L10.8 must follow |
| `test_hand_example_greedy` | unit, smoke | section 3, greedy: `[2, 0]`, `[2, 0, 1]`, `[2]`; no draw | greedy verification |
| `test_one_token_output_is_the_target_exactly` | statistical | $V = 5$, all outcomes enumerated: the first token's distribution equals $p$ as fractions | the exactness theorem |
| `test_two_token_draft_is_exact_at_each_position` | statistical | $k = 2$: first token $\sim p_1$; after a kept $x_1 = y_1$, second $\sim p(\cdot \mid y_1)$ | the right row and history per position |
| `test_deterministic_draft_is_exact` | statistical | one-hot drafts (every $x$, including one with $p(x) = 0$) give exactly $p$ | prompt lookup and n-gram drafts at temperature > 0 |
| `test_penalties_see_the_accepted_drafts` | unit | a repetition penalty rejects a second copy of a just-kept token | penalties match plain `generate` |
| `test_verify_checks_shapes` | boundary | rows must be $m + 1$, draft distributions $m \times V$ | no off-by-one row |
| `test_prompt_lookup_hand_example` | unit, smoke | section 3; most recent occurrence; longest n-gram first; $k = 0$ | the lookup rule L10.8 ports |
| `test_ngram_draft_follows_the_model` | unit | greedy ids are the argmax chain; sampled rows are the model's distribution | the n-gram draft |
| `test_model_draft_syncs_its_cache` | unit | the draft's own cache follows a changing context; proposals equal fresh greedy decoding | a model draft after rejections |
| `test_speculative_greedy_equals_greedy` | differential | four drafts: ids equal L8.2's `generate` and a plain argmax loop | the MS-L8 claim |
| `test_rejected_drafts_are_rolled_back` | property | every target pass starts at the kept length with the right token | the cache rollback |
| `test_stats_and_one_pass_per_round` | unit | self-draft: 40 tokens in 8 passes, acceptance 1; $k = 0$: 40 passes | the speedup is real |
| `test_sampled_speculation_matches_the_target_distribution` | statistical | end to end, 400 seeds, chi-square $p > 10^{-3}$ | sampling through the whole loop |
| `test_eos_stop_and_budget` | boundary | EOS ends without being emitted; a stop string cuts the text; never more than `max_tokens` | `generate`'s contract kept |
| `test_bad_arguments` | boundary | $k < 0$, empty prompt, prompt longer than the model | caller errors early |
| `test_logprobs_are_the_targets` | unit | logprobs equal plain `generate`'s | the API reports the target's numbers |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. treating the draft as if it sampled from the target ($q = p$), or ignoring `draft_probs` | every draft kept: the output follows the draft, not the target | `test_one_token_output_is_the_target_exactly` (mutants `s01`, `s07`) |
| 2. a deterministic draft treated as uniform, or its one-hot at the wrong index | biased output, or a crash on $q(x) = 0$ | `test_deterministic_draft_is_exact` (mutants `s02`, `s09`) |
| 3. resampling from $p$ instead of the residual after a rejection | tokens the draft favors are over-represented | `test_one_token_output_is_the_target_exactly` (mutant `s03`) |
| 4. drawing the uniforms in another order | Python and Rust streams diverge | `test_hand_example` (mutant `s04`) |
| 5. greedy: keeping the draft on a mismatch | the output changes with the draft | `test_hand_example_greedy` (mutant `s05`) |
| 6. the bonus from the wrong row | the token after a fully kept draft is wrong | `test_hand_example_greedy` (mutant `s06`) |
| 7. penalties that do not see the drafts kept this round | sampled and greedy output differ from plain `generate` with penalties | `test_penalties_see_the_accepted_drafts` (mutant `s08`) |
| 8. prompt lookup: oldest occurrence, shortest n-gram first, or ignoring $k$ | different drafts than L10.8; longer drafts than asked | `test_prompt_lookup_hand_example` (mutants `s10`, `s11`, `s12`) |
| 9. a model draft that does not truncate or re-feed its own cache | stale keys: proposals drift from the draft model's real greedy output | `test_model_draft_syncs_its_cache` (mutants `s15`, `s16`) |
| 10. no rollback, or the extra token counted as cached, or never fed | the next pass attends to rejected tokens; output diverges | `test_rejected_drafts_are_rolled_back` (mutants `s17`, `s18`, `s19`) |
| 11. EOS kept in `ids`, or a stop string not cut | the API returns text past the stop | `test_eos_stop_and_budget` (mutants `s22`, `s23`) |
| 12. a round allowed $k + 1$ tokens when fewer remain | more than `max_tokens` ids | `test_eos_stop_and_budget` (mutant `s24`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M07.6` | `speculative_step` is the sampled verification of one position |
| Back | `L8.2` | `KVCache.truncate` is the rollback; `cache_dims` sizes the cache; `Generation` is the result; `generate` is the reference the tests compare with |
| Back | `L2.1` | `NGramDraft` drafts from an `NGramLM`'s `logprobs` |
| Back | `L8.1` | `sample` (greedy and bonus draws), `sampling_distribution` (the $p_i$), `token_logprobs`, `request_rng` |
| Back | `M11.1` | total variation and why $\alpha = 1 - \mathrm{TV}(p, q)$ (reading) |
| Forward | `L10.8` | the Rust engine's prompt-lookup and n-gram speculation, held to `verify_draft` on shared logits and to greedy equality |

If you skip this module, `ol check L10.8` stops with `BLOCKED ... needs L8.6`: build it, or pass `--ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `verify_draft` | vLLM's rejection sampler | batched verification on the GPU, per-request draft lengths, the residual in log space | vLLM `v1/sample/rejection_sampler.py` |
| `PromptLookupDraft` | vLLM's n-gram proposer, HF `prompt_lookup_num_tokens` | a KMP-style search over the whole context, minimum and maximum n-gram sizes | vLLM `v1/spec_decode/ngram_proposer.py` |
| `ModelDraft` | EAGLE, Medusa | a draft head on the target's own hidden states; a tree of candidates verified with one tree-masked attention pass | Li et al., *EAGLE* (2024); Cai et al., *Medusa* (2024) |
| one draft path | SGLang and vLLM tree speculation | several candidate continuations per round, verified together | SGLang `srt/speculative/` |
