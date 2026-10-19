# Rejection sampling and residual distributions

## Overview

| | |
|---|---|
| **Module** | `M07.6` · build · Python · Pass 6 · 1 to 1.5 h reading |
| **Solution** | [`course/solutions/python/tinyllm/prob/rejection.py`](../../../../course/solutions/python/tinyllm/prob/rejection.py): `rejection_accept`, `acceptance_probability`, `residual_distribution`, `speculative_step`, `rejection_sample` |
| **Contract** | [`course/contracts/py/tinyllm/prob/rejection.pyi`](../../../../course/contracts/py/tinyllm/prob/rejection.pyi) · the draw order: [`spec/sampling.md`](../../../../course/contracts/spec/sampling.md) (draw accounting) |
| **Needs** | `M07.1` (`sample_categorical`, the inverse CDF, and `UniformSource`; [`sampling.py`](../../../../course/solutions/python/tinyllm/prob/sampling.py)) · reading: `M07.0` random variables, `M06.3` PCG32 (the uniforms) |
| **Used by** | `L8.6` speculative decoding verifies every draft token with `speculative_step`, and `L10.8` ports it to Rust; one of the Pass 6 math modules |
| **Optional depth** | Devroye, *Non-Uniform Random Variate Generation* (1986), ch. 2.3; Leviathan, Kalman, and Matias, "Fast Inference from Transformers via Speculative Decoding" (2023), appendix A.1; Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling" (2023) |

## Key Takeaways

- Keeping a draft $x \sim q$ with probability $\min(1, p_x/q_x)$ keeps exactly $\min(p_x, q_x)$ of every token's mass; the total kept is $\sum_x \min(p_x, q_x) = 1 - \mathrm{TV}(p, q)$.
- What rejection removes is exactly the residual $\max(0, p - q)$, so resampling from its normalized form restores $p$ token for token: the output of `speculative_step` is distributed as $p$ for every $q$.
- "Accept when $u < r$" with $u$ uniform on $[0, 1)$ has probability exactly $\min(1, r)$; "$u \le r$" counts one extra point and accepts tokens $p$ never produces.
- Von Neumann's sampler accepts a proposal with probability $p_x / (m q_x)$; each try succeeds with probability $1/m$, so the cost is a geometric number of tries with mean $m$.

## How to read this chapter

Sections 1 and 2 derive von Neumann's sampler, the acceptance test, and the residual distribution, and prove that speculative decoding samples exactly $p$. Section 3 works the theorem by hand. Section 4 fixes the interface, and section 5 walks through the solution line by line and explains why each choice is the right one. Try to predict each step of the solution before reading it.

---

## 1. Why now

The sampler (`L8.1`) emits one token per forward pass of the model, and each forward pass reads every weight once. Part 8 makes decoding faster without changing what it says: a cheap draft (an n-gram table, a prompt lookup, a small model) proposes several tokens, and the target model checks all of them in one forward pass (`L8.6`). Greedy decoding only needs "did the draft guess the argmax". Sampled decoding needs more: the accepted text must be distributed exactly as if the target had sampled it alone, or the speedup silently changes the model's output distribution, which no eval would attribute to the cache. This module is the piece of probability that makes that exact: an acceptance test and a residual distribution, proved by enumeration before `L8.6` builds on them.

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

Suppose $q$ is cheap to sample but samples of $p$ are wanted. If $p_i \le m\, q_i$ for every $i$, repeat: draw $x \sim q$, draw $u$, and accept $x$ when $u < p_x / (m\, q_x)$. On one try,

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

That is $p$. One concrete draw: the draft proposes $x = 1$, $u = 0.75 \ge r_1 = 0.5$ rejects it, and $u' = 0.75$ walks the residual's CDF $[0.5, 0.5, 1.0, 1.0]$ to id 2. With $u = 0.25$ instead, id 1 is kept and $u'$ is not looked at. These are the first cases in section 4.

## 4. The interface

```python
# python/tinyllm/prob/rejection.py
def rejection_accept(p_x: float, q_x: float, u: float) -> bool               # u < p_x / q_x
def acceptance_probability(p: ArrayLike, q: ArrayLike) -> float              # sum min(p, q)
def residual_distribution(p: ArrayLike, q: ArrayLike) -> NDArray             # normalize(max(0, p - q)); p if p == q
def speculative_step(p, q, x: int, u_accept: float, u_resample: float) -> tuple[int, bool]
def rejection_sample(p, q, m: float, rng: UniformSource, max_tries: int = 10000) -> tuple[int, int]
```

`speculative_step` resamples with `M07.1`'s `sample_categorical`. `rejection_sample` draws exactly two uniforms per try, proposal first. The contract has every error rule.

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the hand example | unit | section 3's ratios, $\alpha = 1/2$, residual $[1/2, 0, 1/2, 0]$ | everyone agrees on the definitions |
| the hand example's output is $p$ | statistical | section 3's output distribution, enumerated exactly | the theorem on numbers checkable by hand |
| strict acceptance | boundary | $u = r$ rejects, $p_x = 0$ never accepts | discrete uniforms make $\le$ wrong |
| target dominates | boundary | $r \ge 1$ accepts every $u < 1$ | under-proposed tokens are always kept |
| bad acceptance arguments | boundary | $q_x = 0$, $u \notin [0, 1)$, $p_x < 0$ raise | a silent answer biases the output |
| acceptance rate | statistical | enumerated acceptance $= \sum \min(p, q)$ | `L8.6` reports it as its acceptance rate |
| the residual is a distribution | property | non-negative, sums to 1, $Z = 1 - \alpha$ | `sample_categorical` demands a distribution |
| $p = q$ | boundary | gives a copy of $p$ | no 0/0, no aliasing of the caller's array |
| disjoint supports | boundary | $\alpha = 0$ gives residual $p$ | a useless draft is slow, not wrong |
| ascending sums | unit | the normalizer is a left-to-right loop | bit parity with the Rust port (`L10.8`) |
| shapes and distributions | boundary | mismatched shapes and non-distributions raise | vocabularies of draft and target must match |
| speculative output is exactly $p$ | statistical | output distribution $= p$ exactly, random $p, q$ in eighths | the theorem `L8.6` relies on |
| speculative chi-square | statistical | 20000 seeded draws match $p$ (chi-square) | the same law on arbitrary distributions |
| accepted drafts ignore $u'$ | unit | an accepted draft does not look at $u'$ | the draw order is part of the parity contract |
| bad drafts | boundary | $x$ outside the vocabulary or $q_x = 0$ raise | draft and proposal disagreeing is a bug |
| von Neumann's first try | statistical | one try accepts $x$ with probability $p_x/m$, exactly | the von Neumann law |
| geometric tries | statistical | mean tries $= m$ within 4 standard errors | the cost of a loose envelope |
| von Neumann draw order | unit | proposal uniform first, then acceptance; two per try | reproducible across languages |
| the envelope | boundary | $p_x > m q_x$, $m < 1$, and no acceptance in `max_tries` raise | a wrong envelope silently biases |

## 5. The solution, step by step

### From first principles

1. **Both inputs are distributions over one vocabulary.** Every function starts by checking $p$ and $q$ with `M07.1`'s rules (1-D, finite, non-negative, summing to 1) and that their shapes match: a draft over a different vocabulary is a bug, not a case to handle.
2. **Acceptance is one strict comparison.** Section 2.2: `u < p_x / q_x` has probability exactly $\min(1, r)$ for $u$ on the discrete grid in $[0, 1)$. No explicit `min` is needed, because $u < 1 \le r$ already accepts. $q_x = 0$ means the draft proposed a token it gave probability 0, so raise rather than divide.
3. **The acceptance probability and the residual's normalizer are left-to-right sums.** Section 2.5: a plain loop over ascending ids, so the Rust port gets the same last bit and the residual's CDF has the same boundaries.
4. **The residual is $\max(0, p - q)/Z$, or $p$ when $Z = 0$.** Section 2.3. $Z = 0$ happens exactly when $p \le q$ everywhere, which for two distributions means $p = q$; then nothing is ever rejected, and returning a copy of $p$ keeps the function total without aliasing the caller's array.
5. **One speculative step: accept, or one draw from the residual.** Section 2.4: if the acceptance test passes, return $x$ without reading $u'$; otherwise `sample_categorical(residual, u')`. That draw order is part of the parity contract with `L8.6` and `L10.8`.
6. **Von Neumann needs a checked envelope and two uniforms per try.** Section 2.1: check $m \ge 1$ and $p \le mq$ everywhere first, then loop: propose with one uniform, accept with another when $u < p_x/(m q_x)$, return the token and the number of tries. A bounded loop turns a pathological envelope into an error rather than a hang.

### The code, annotated

```python
def rejection_accept(p_x: float, q_x: float, u: float) -> bool:
    _check_u(u, "u")                                   # u in [0, 1)
    if not q_x > 0.0:
        raise ValueError(f"q_x must be > 0 (q proposed this token), got {q_x!r}")
    if not p_x >= 0.0:
        raise ValueError(f"p_x must be >= 0, got {p_x!r}")
    # P(u < r) = min(1, r) for u uniform on [0, 1): strict, so p_x = 0 (r = 0)
    # never accepts and r >= 1 always does.
    return u < p_x / q_x


def acceptance_probability(p: ArrayLike, q: ArrayLike) -> float:
    a, b = _pair(p, q)                                 # two checked distributions, same shape
    s = 0.0
    for pi, qi in zip(a.tolist(), b.tolist()):         # ascending ids, left to right
        s += min(pi, qi)                               # the kept mass of token i
    return s                                           # = 1 - TV(p, q)
```

On section 3, the per-token kept masses are $\tfrac14, \tfrac14, 0, 0$, so $\alpha = \tfrac12$.

```python
def residual_distribution(p: ArrayLike, q: ArrayLike) -> NDArray:
    a, b = _pair(p, q)
    r = np.maximum(a - b, 0.0)                         # what acceptance left missing from p
    z = 0.0
    for v in r.tolist():                               # not np.sum: pairwise summation differs in the last bit
        z += v
    if z <= 0.0:
        # p == q: nothing is ever rejected, but the result must stay a distribution.
        return a.copy()
    return r / z


def speculative_step(p, q, x: int, u_accept: float, u_resample: float) -> tuple[int, bool]:
    a, b = _pair(p, q)
    if not 0 <= x < a.size:
        raise ValueError(f"draft token {x} outside [0, {a.size})")
    _check_u(u_resample, "u_resample")                 # validated even when not used
    if rejection_accept(float(a[x]), float(b[x]), u_accept):
        return int(x), True                            # accepted: u_resample is not read
    return sample_categorical(residual_distribution(a, b), u_resample), False
```

Section 3's draw: $x = 1$, $u = 0.75 \ge 0.5$ rejects; the residual is $[\tfrac12, 0, \tfrac12, 0]$ with running sums $0.5, 0.5, 1.0, 1.0$, and $u' = 0.75$ first falls below $1.0$ at id 2. With $u = 0.25 < 0.5$ the function returns `(1, True)`. If the disjoint-support case $\alpha = 0$ occurs, $r = p$ and $Z = 1$, so the residual is $p$ itself: a useless draft is always rejected and the output is still a draw of $p$.

```python
def rejection_sample(p, q, m: float, rng, max_tries: int = 10000) -> tuple[int, int]:
    a, b = _pair(p, q)
    if not m >= 1.0:                                   # an envelope below 1 cannot cover p
        raise ValueError(f"the envelope m must be >= 1, got {m!r}")
    if (a > m * b).any():                              # p_x > m q_x: x would be under-sampled
        i = int(np.flatnonzero(a > m * b)[0])
        raise ValueError(f"m = {m} is not an envelope: p[{i}] = {a[i]} > m q[{i}] = {m * b[i]}")
    for tries in range(1, max_tries + 1):
        x = sample_categorical(b, rng.uniform())       # proposal uniform first
        if rng.uniform() < a[x] / (m * b[x]):          # then the acceptance uniform; b[x] > 0 here
            return x, tries
    raise RuntimeError(f"no acceptance in {max_tries} tries")
```

`sample_categorical` never returns an id with $q_x = 0$ (`M07.1`'s strict comparison), so the division is safe; and the envelope check guarantees that any id with $p_x > 0$ has $q_x > 0$.

Full solution: [`rejection.py`](../../../../course/solutions/python/tinyllm/prob/rejection.py).

### Why this solution

- **Cost.** `rejection_accept` is $O(1)$; `acceptance_probability` and `residual_distribution` are $O(n)$; `speculative_step` is $O(n)$ (validation, the residual, and an inverse-CDF draw) and is only computed in full on rejection. `rejection_sample` costs $O(n)$ checks plus a geometric number of tries with mean $m$, each $O(n)$ for the inverse CDF.
- **Residual over retry.** A retry loop (von Neumann) would also be exact, but each retry in speculative decoding would need another target forward pass; the residual produces a token from the pass already done, so every verified position yields exactly one token.
- **Loops over `np.sum`.** numpy's pairwise sum is more accurate for long arrays, but it is not what the Rust port computes. The sums here are of at most a vocabulary of non-negative terms, where left-to-right error is tiny, and bit parity matters more.
- **Tradeoffs accepted.** Validating $p$ and $q$ on every call costs $O(n)$ per step; for a 50 000-token vocabulary that is small next to a forward pass, and it turns a draft/target vocabulary mismatch into an immediate error.

### Explain it back

**1. Why is the residual $\max(0, p - q)$ and not $\max(0, q - p)$ or $\lvert p - q\rvert$?**

<details><summary>Answer</summary>

After acceptance each token has $\min(p_t, q_t)$ of its target mass; what is missing is $p_t - \min(p_t, q_t) = \max(0, p_t - q_t)$, the tokens the draft under-proposed. $\max(0, q - p)$ would resample the tokens the draft over-proposed, which already have more than enough, and $\lvert p - q \rvert$ mixes both; neither adds up to $p$.

</details>

**2. Why does `rejection_accept` need no `min(1, ...)`?**

<details><summary>Answer</summary>

$u$ is always below 1, so when $r \ge 1$ the test `u < r` is always true: the acceptance probability is already $\min(1, r)$. Writing an explicit cap is harmless, but "capping" by inverting the ratio when it exceeds 1 would turn the always-accept case into a partial one and drop under-proposed tokens.

</details>

**3. On acceptance, why must `speculative_step` return without using $u'$?**

<details><summary>Answer</summary>

Correctness: the theorem's accept branch outputs $x$ itself, contributing $\min(p_x, q_x)$; letting $u'$ change the result there would replace accepted tokens with residual draws and break that term. Parity: the contract says $u'$ matters only on rejection, so `L8.6` and the Rust engine (`L10.8`) agree on every decision for the same uniforms.

</details>

**4. Why does `residual_distribution` return `a.copy()` when $Z = 0$ and not `a`?**

<details><summary>Answer</summary>

$Z = 0$ means $p = q$, so the residual is never sampled, but the function must still return a valid distribution and $p$ is the natural choice. Returning the caller's array would let a later in-place edit of the result (a mask, a temperature) silently change the target distribution itself.

</details>

**5. In von Neumann's sampler, what does a loose envelope (large $m$) cost, and what does a too-small one break?**

<details><summary>Answer</summary>

A loose envelope only costs time: each try accepts with probability $1/m$, so the mean number of tries is $m$. A too-small one breaks correctness: for an $x$ with $p_x > m q_x$ the acceptance ratio exceeds 1 and is effectively capped, so $x$ is drawn with probability below $p_x$ and the others are over-sampled. That is why the function checks the envelope before drawing.

</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. accepting on $u \le r$ | a token with $p_x = 0$ is accepted when $u = 0$ | `return u < p_x / q_x`; step 2 |
| 2. the ratio upside down, or "capped" by inverting it | over-proposed tokens kept, under-proposed ones dropped | the ratio is `p_x / q_x`, and $u < 1$ does the capping; step 2 |
| 3. residual $\max(0, q - p)$ or $\lvert p - q\rvert$ | resamples the tokens the draft over-proposed | `np.maximum(a - b, 0.0)`; step 4 |
| 4. forgetting to normalize the residual | `sample_categorical` rejects it, or the CDF ends below 1 | `r / z`; step 4 |
| 5. on rejection, resampling from $p$, from $q$, or returning the draft | output no longer $p$: accepted tokens counted twice | samples `residual_distribution(a, b)` on rejection; step 5 |
| 6. $p = q$ handled as 0/0, or returning the caller's array | NaN, or a later in-place edit corrupts the target | `if z <= 0.0: return a.copy()`; step 4 |
| 7. von Neumann without $m$, or without checking it | the accepted tokens follow $\min(p, q)$-ish weights, not $p$ | accepts on `a[x] / (m * b[x])` after checking `m >= 1` and `a <= m * b`; step 6 |
| 8. summing with numpy's pairwise `np.sum` | last-bit differences against the Rust port | explicit left-to-right loops; step 3 |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M07.1` | `sample_categorical` draws the residual; `UniformSource` is the generator type |
| Back | `M07.0` | uniform random variables and independence, behind every probability above |
| Forward | `L8.6` | speculative decoding: one `speculative_step` per draft position, `acceptance_probability` as the reported acceptance rate |
| Forward | `L10.8` | the Rust engine's speculative decoding, held to the same accept and reject decisions |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `speculative_step` | vLLM `RejectionSampler` | batched acceptance over a whole draft tree on the GPU, recovered tokens from the residual | `vllm/v1/sample/rejection_sampler.py` |
| `residual_distribution` | Hugging Face assisted generation | the same residual for model drafts (`_speculative_sampling`) | `transformers/generation/utils.py` |
| `rejection_sample` | NumPy's Gamma and binomial samplers | rejection with tight squeeze functions to avoid evaluating the target | `numpy/random/src/distributions/distributions.c` |
