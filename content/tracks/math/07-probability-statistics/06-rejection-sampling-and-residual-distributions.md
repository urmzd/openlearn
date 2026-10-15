<!-- ol:module M07.6 -->
# Rejection sampling and residual distributions

## Overview

| | |
|---|---|
| **Module** | `M07.6` · build · Python · Pass 6 · 2 to 3 h |
| **You build** | `python/tinyllm/prob/rejection.py`: `rejection_accept`, `acceptance_probability`, `residual_distribution`, `speculative_step`, `rejection_sample` |
| **Contract** | [`course/contracts/py/tinyllm/prob/rejection.pyi`](../../../../course/contracts/py/tinyllm/prob/rejection.pyi) · the draw order: [`spec/sampling.md`](../../../../course/contracts/spec/sampling.md) (draw accounting) |
| **Tests** | `course/tests/M07.6/` (what they check: section 4) |
| **Needs** | `M07.1` (`sample_categorical`, the inverse CDF, and `UniformSource`) · reading: `M07.0` random variables, `M06.3` PCG32 (your uniforms) |
| **Used by** | `L8.6` speculative decoding verifies every draft token with `speculative_step`, and `L10.8` ports it to Rust |
| **Milestone** | `MS-P6` (Pass 6 gate: every math module of the pass checks green) |
| **Optional depth** | Devroye, *Non-Uniform Random Variate Generation* (1986), ch. 2.3; Leviathan, Kalman, and Matias, "Fast Inference from Transformers via Speculative Decoding" (2023), appendix A.1; Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling" (2023) |

## Key Takeaways

- Keeping a draft $x \sim q$ with probability $\min(1, p_x/q_x)$ keeps exactly $\min(p_x, q_x)$ of every token's mass; the total kept is $\sum_x \min(p_x, q_x) = 1 - \mathrm{TV}(p, q)$ (`test_acceptance_rate_matches_acceptance_probability`).
- What rejection removes is exactly the residual $\max(0, p - q)$, so resampling from its normalized form restores $p$ token for token: the output of `speculative_step` is distributed as $p$ for every $q$ (`test_speculative_output_is_exactly_p`).
- "Accept when $u < r$" with $u$ uniform on $[0, 1)$ has probability exactly $\min(1, r)$; "$u \le r$" counts one extra point and accepts tokens $p$ never produces (`test_accept_is_strict`).
- Von Neumann's sampler accepts a proposal with probability $p_x / (m q_x)$; each try succeeds with probability $1/m$, so the cost is a geometric number of tries with mean $m$ (`test_rejection_sample_tries_are_geometric`).

## How to work this chapter

```bash
ol start M07.6              # stubs rejection.py into your repo
ol tests M07.6              # read the test catalog first: rung R0, you write no tests here
ol check M07.6              # exit code is the verdict
ol check M07.6 --ref-deps   # only if you skipped M07.1
ol diff  M07.6              # after passing: your code against the reference
```

---

## 1. Why now

Your sampler (`L8.1`) emits one token per forward pass of the model, and each forward pass reads every weight once. In Part 8 you will make decoding faster without changing what it says: a cheap draft (an n-gram table, a prompt lookup, a small model) proposes several tokens, and the target model checks all of them in one forward pass (`L8.6`). Greedy decoding only needs "did the draft guess the argmax". Sampled decoding needs more: the accepted text must be distributed exactly as if the target had sampled it alone, or the speedup silently changes the model's output distribution, which no eval would attribute to the cache. This module is the piece of probability that makes that exact: an acceptance test and a residual distribution, proved by enumeration before `L8.6` builds on them.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $n$ | number of ids (the vocabulary) | integer |
| $p \in \Delta^{n-1}$ | the target distribution (the big model's sampler output) | `float64[n]`, sums to 1 |
| $q \in \Delta^{n-1}$ | the proposal distribution (the draft) | `float64[n]`, sums to 1 |
| $x$ | a draft token, drawn from $q$ | integer in $[0, n)$ |
| $u, u'$ | independent uniforms on $[0, 1)$ | float |
| $r_x = p_x / q_x$ | the acceptance ratio of $x$ | float $\ge 0$ |
| $\alpha = \sum_i \min(p_i, q_i)$ | the acceptance probability | float in $[0, 1]$ |
| $\mathrm{TV}(p, q) = \tfrac12 \sum_i \lvert p_i - q_i\rvert$ | total variation distance | float in $[0, 1]$ |
| $\mathrm{res}(p, q)_i = \max(0, p_i - q_i) / Z$ | the residual distribution, $Z = \sum_i \max(0, p_i - q_i)$ | `float64[n]` |
| $m$ | an envelope: $p_i \le m q_i$ for every $i$ | float $\ge 1$ |

### 2.1 Von Neumann's rejection sampler

You can sample $q$ cheaply but want samples of $p$. If $p_i \le m\, q_i$ for every $i$, repeat: draw $x \sim q$, draw $u$, and accept $x$ when $u < p_x / (m\, q_x)$. On one try,

$$P(\text{propose } x \text{ and accept}) = q_x \cdot \frac{p_x}{m\, q_x} = \frac{p_x}{m}, \qquad P(\text{accept}) = \sum_x \frac{p_x}{m} = \frac{1}{m}.$$

Conditioned on acceptance, $x$ has probability $(p_x/m)/(1/m) = p_x$: exactly the target. Each try is an independent coin with success probability $1/m$, so the number of tries is geometric, with mean $m$ and variance $m(m - 1)$. A loose envelope costs time, never correctness; an envelope that is too small ($p_x > m q_x$ for some $x$) caps the acceptance of $x$ at 1 and under-samples it, which is why the contract checks it.

### 2.2 The acceptance test, exactly

For $u$ uniform on $[0, 1)$ and any $r \ge 0$, $P(u < r) = \min(1, r)$. The comparison must be strict: $P(u \le r)$ is the same for continuous $u$, but floating-point uniforms are discrete (`M06.3` gives multiples of $2^{-53}$), and the point $u = 0$ with $p_x = 0$ would accept a token the target never produces. With $r = p_x / q_x \ge 1$, every $u < 1$ is below it: an under-proposed token is always kept.

### 2.3 Accept, or draw from the residual

Speculative decoding has no retry loop: the target's forward pass already happened, so a rejected position must produce a token right away. Keep $x \sim q$ with probability $\min(1, p_x/q_x)$. The kept mass of token $t$ is

$$q_t \min\!\left(1, \frac{p_t}{q_t}\right) = \min(p_t, q_t),$$

and the total kept mass is $\alpha = \sum_t \min(p_t, q_t)$. Since $\min(a, b) = \tfrac12(a + b - \lvert a - b\rvert)$ and both distributions sum to 1, $\alpha = 1 - \mathrm{TV}(p, q)$: the closer the draft, the more it is accepted. What is missing from $p$ after acceptance is $p_t - \min(p_t, q_t) = \max(0, p_t - q_t)$, which sums to $Z = 1 - \alpha$. Normalizing gives the residual.

### 2.4 The theorem

Draw $x \sim q$, accept with probability $\min(1, p_x/q_x)$, otherwise draw $t \sim \mathrm{res}(p, q)$. Then

$$P(\text{output} = t) = \underbrace{\min(p_t, q_t)}_{\text{accepted } t} + \underbrace{(1 - \alpha)}_{\text{rejected}} \cdot \frac{\max(0, p_t - q_t)}{1 - \alpha} = \min(p_t, q_t) + \max(0, p_t - q_t) = p_t.$$

Nothing was assumed about $q$ beyond being a distribution: a bad draft makes the method slow, never wrong. When $p = q$, $Z = 0$ and the residual is undefined, but then $\alpha = 1$ and it is never sampled; the contract returns $p$ so the function stays total.

### 2.5 Determinism

The sums in $\alpha$ and $Z$ are left-to-right loops over ascending ids in float64, and $u$, $u'$ come from the request's generator in the order `L8.6` fixes. The Rust engine (`L10.8`) repeats the same arithmetic, so both accept and reject the same drafts for the same seed. numpy's `np.sum` sums pairwise and can differ in the last bit, which moves a boundary of the residual's CDF.

## 3. Worked example by hand

Four ids, $p = [\tfrac12, \tfrac14, \tfrac14, 0]$ (the target), $q = [\tfrac14, \tfrac12, 0, \tfrac14]$ (the draft).

| id $t$ | $p_t$ | $q_t$ | $r_t = p_t/q_t$ | $P(\text{keep} \mid x = t)$ | kept mass $\min(p_t, q_t)$ | $\max(0, p_t - q_t)$ |
|---|---|---|---|---|---|---|
| 0 | 1/2 | 1/4 | 2 | 1 | 1/4 | 1/4 |
| 1 | 1/4 | 1/2 | 1/2 | 1/2 | 1/4 | 0 |
| 2 | 1/4 | 0 | (never proposed) | | 0 | 1/4 |
| 3 | 0 | 1/4 | 0 | 0 | 0 | 0 |

The acceptance probability is $\alpha = \tfrac14 + \tfrac14 = \tfrac12$; the residual mass is $Z = \tfrac12$, and the residual is $[\tfrac12, 0, \tfrac12, 0]$. The output distribution:

- id 0: kept $\tfrac14$, plus rejected mass $\tfrac12$ times residual $\tfrac12$: $\tfrac12$.
- id 1: kept $\tfrac14$, residual 0: $\tfrac14$.
- id 2: never proposed, residual: $\tfrac12 \cdot \tfrac12 = \tfrac14$.
- id 3: proposed a quarter of the time and always rejected: 0.

That is $p$. One concrete draw: the draft proposes $x = 1$, $u = 0.75 \ge r_1 = 0.5$ rejects it, and $u' = 0.75$ walks the residual's CDF $[0.5, 0.5, 1.0, 1.0]$ to id 2. With $u = 0.25$ instead, id 1 is kept and $u'$ is not looked at. These are the first cases in section 4: `test_hand_example`, `test_hand_example_output_is_p`, and `test_speculative_accepts_without_resampling`.

## 4. The interface

```python
# python/tinyllm/prob/rejection.py
def rejection_accept(p_x: float, q_x: float, u: float) -> bool               # u < p_x / q_x
def acceptance_probability(p: ArrayLike, q: ArrayLike) -> float              # sum min(p, q)
def residual_distribution(p: ArrayLike, q: ArrayLike) -> NDArray             # normalize(max(0, p - q)); p if p == q
def speculative_step(p, q, x: int, u_accept: float, u_resample: float) -> tuple[int, bool]
def rejection_sample(p, q, m: float, rng: UniformSource, max_tries: int = 10000) -> tuple[int, int]
```

`speculative_step` resamples with your `M07.1` `sample_categorical`. `rejection_sample` draws exactly two uniforms per try, proposal first. The contract has every error rule.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example` | unit | section 3's ratios, $\alpha = 1/2$, residual $[1/2, 0, 1/2, 0]$ | you and the test agree on the definitions |
| `test_hand_example_output_is_p` | statistical | section 3's output distribution, enumerated exactly | the theorem on numbers you can check by hand |
| `test_accept_is_strict` | boundary | $u = r$ rejects, $p_x = 0$ never accepts | discrete uniforms make $\le$ wrong |
| `test_accept_always_when_target_dominates` | boundary | $r \ge 1$ accepts every $u < 1$ | under-proposed tokens are always kept |
| `test_accept_rejects_bad_arguments` | boundary | $q_x = 0$, $u \notin [0, 1)$, $p_x < 0$ raise | a silent answer biases the output |
| `test_acceptance_rate_matches_acceptance_probability` | statistical | enumerated acceptance $= \sum \min(p, q)$ | `L8.6` reports it as its acceptance rate |
| `test_residual_is_a_distribution` | property | non-negative, sums to 1, $Z = 1 - \alpha$ | `sample_categorical` demands a distribution |
| `test_residual_when_p_equals_q` | boundary | $p = q$ gives a copy of $p$ | no 0/0, no aliasing of the caller's array |
| `test_residual_disjoint_supports_is_p` | boundary | $\alpha = 0$ gives residual $p$ | a useless draft is slow, not wrong |
| `test_residual_sum_is_ascending` | unit | the normalizer is a left-to-right loop | bit parity with the Rust port (`L10.8`) |
| `test_shapes_and_distributions_are_checked` | boundary | mismatched shapes and non-distributions raise | vocabularies of draft and target must match |
| `test_speculative_output_is_exactly_p` | statistical | output distribution $= p$ exactly, random $p, q$ in eighths | the theorem `L8.6` relies on |
| `test_speculative_chi_square` | statistical | 20000 seeded draws match $p$ (chi-square) | the same law on arbitrary distributions |
| `test_speculative_accepts_without_resampling` | unit | an accepted draft ignores $u'$ | the draw order is part of the parity contract |
| `test_speculative_rejects_bad_drafts` | boundary | $x$ outside the vocabulary or $q_x = 0$ raise | draft and proposal disagreeing is a bug |
| `test_rejection_sample_first_try_enumerated` | statistical | one try accepts $x$ with probability $p_x/m$, exactly | the von Neumann law |
| `test_rejection_sample_tries_are_geometric` | statistical | mean tries $= m$ within 4 standard errors | the cost of a loose envelope |
| `test_rejection_sample_draw_order` | unit | proposal uniform first, then acceptance; two per try | reproducible across languages |
| `test_rejection_sample_checks_the_envelope` | boundary | $p_x > m q_x$, $m < 1$, and no acceptance in `max_tries` raise | a wrong envelope silently biases |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. accepting on $u \le r$ | a token with $p_x = 0$ is accepted when $u = 0$ | `test_accept_is_strict` (mutant `s01`) |
| 2. the ratio upside down, or "capped" by inverting it | over-proposed tokens kept, under-proposed ones dropped | `test_accept_always_when_target_dominates` (mutants `s02`, `s12`) |
| 3. residual $\max(0, q - p)$ or $\lvert p - q\rvert$ | resamples the tokens the draft over-proposed | `test_residual_is_a_distribution` (mutants `s03`, `s05`) |
| 4. forgetting to normalize the residual | `sample_categorical` rejects it, or the CDF ends below 1 | `test_residual_is_a_distribution` (mutant `s04`) |
| 5. on rejection, resampling from $p$, from $q$, or returning the draft | output no longer $p$: accepted tokens counted twice | `test_speculative_output_is_exactly_p` (mutants `s06`, `s07`, `s13`) |
| 6. $p = q$ handled as 0/0, or returning the caller's array | NaN, or a later in-place edit corrupts the target | `test_residual_when_p_equals_q` (mutants `s08`, `s14`) |
| 7. von Neumann without $m$, or without checking it | the accepted tokens follow $\min(p, q)$-ish weights, not $p$ | `test_rejection_sample_first_try_enumerated` (mutant `s15`), `test_rejection_sample_checks_the_envelope` (mutant `s19`) |
| 8. summing with numpy's pairwise `np.sum` | last-bit differences against the Rust port | `test_residual_sum_is_ascending` (mutant `s09`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M07.1` | `sample_categorical` draws the residual; `UniformSource` is the generator type |
| Back | `M07.0` | uniform random variables and independence, behind every probability above |
| Forward | `L8.6` | speculative decoding: one `speculative_step` per draft position, `acceptance_probability` as the reported acceptance rate |
| Forward | `L10.8` | the Rust engine's speculative decoding, held to the same accept and reject decisions |

If you skip this module, `L8.6` cannot check: `ol check L8.6` stops with `L8.6 needs M07.6`, and `--ref-deps` substitutes the reference.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `speculative_step` | vLLM `RejectionSampler` | batched acceptance over a whole draft tree on the GPU, recovered tokens from the residual | `vllm/v1/sample/rejection_sampler.py` |
| `residual_distribution` | Hugging Face assisted generation | the same residual for model drafts (`_speculative_sampling`) | `transformers/generation/utils.py` |
| `rejection_sample` | NumPy's Gamma and binomial samplers | rejection with tight squeeze functions to avoid evaluating the target | `numpy/random/src/distributions/distributions.c` |
