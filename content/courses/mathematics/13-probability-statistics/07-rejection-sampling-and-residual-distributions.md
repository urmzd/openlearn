# Rejection sampling and residual distributions

A language model's sampler emits one token per forward pass, and each forward pass reads every weight once. The fastest way to decode more quickly without changing what the model says is to let something cheap guess ahead: a draft (an n-gram table, a prompt lookup, a small model) proposes several tokens, and the target model checks all of them in a single forward pass. For greedy decoding the check is easy, since you only ask whether the draft guessed the argmax. For sampled decoding it is subtle. The accepted text must be distributed *exactly* as if the target had sampled it alone, or the speedup silently changes the model's output distribution, and no evaluation would ever attribute that drift to the decoding trick.

This section supplies the piece of probability that makes the check exact: an acceptance test and a residual distribution. You will derive von Neumann's classic rejection sampler, adapt it to the setting where there is no time to retry, and prove (first algebraically, then by enumerating a small example by hand) that the result samples the target distribution for *every* draft. The code lives in [`course/solutions/python/tinyllm/prob/rejection.py`](../../../../course/solutions/python/tinyllm/prob/rejection.py), with the signatures and error rules in the contract [`course/contracts/py/tinyllm/prob/rejection.pyi`](../../../../course/contracts/py/tinyllm/prob/rejection.pyi) and the draw order in [`spec/sampling.md`](../../../../course/contracts/spec/sampling.md) (draw accounting).

This section builds on [Categorical sampling: inverse CDF, Gumbel-max, alias method](02-categorical-sampling.md), whose `sample_categorical`, inverse CDF, and `UniformSource` it reuses (see [`sampling.py`](../../../../course/solutions/python/tinyllm/prob/sampling.py)). It also assumes the language of [Random variables, expectation, variance, and normal draws by Box-Muller](01-random-variables-and-the-normal.md) and the uniforms produced by [Modular arithmetic, hashing, and PCG32 in Python](../12-discrete-math-2/03-modular-arithmetic-hashing-and-pcg32.md).

> **Learning objectives**
> - Derive von Neumann's rejection sampler and show that the number of tries is geometric with mean $m$.
> - Explain why the acceptance test must be the strict comparison $u < r$ for floating-point uniforms.
> - Derive the residual distribution $\max(0, p - q)/Z$ and prove that accept-or-resample outputs exactly $p$.
> - Show that the acceptance probability equals $\sum_i \min(p_i, q_i) = 1 - \mathrm{TV}(p, q)$.
> - Implement the five functions so that their sums and draw order match a Rust port bit for bit.

## Notation

The section uses the following symbols throughout.

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

## Von Neumann's rejection sampler

Start with the classic problem. Suppose $q$ is cheap to sample but you want samples of $p$. If $p_i \le m\, q_i$ for every $i$, repeat the following: draw $x \sim q$, draw $u$, and accept $x$ when $u < p_x / (m\, q_x)$. On one try,

$$P(\text{propose } x \text{ and accept}) = q_x \cdot \frac{p_x}{m\, q_x} = \frac{p_x}{m}, \qquad P(\text{accept}) = \sum_x \frac{p_x}{m} = \frac{1}{m}.$$

Conditioned on acceptance, $x$ has probability $(p_x/m)/(1/m) = p_x$: exactly the target. Each try is an independent coin with success probability $1/m$, so the number of tries is geometric, with mean $m$ and variance $m(m - 1)$. A loose envelope costs time, never correctness. An envelope that is too small ($p_x > m q_x$ for some $x$) is a different matter: it caps the acceptance of $x$ at 1 and under-samples it, which is why the contract checks the envelope before drawing anything.

## The acceptance test, exactly

Every acceptance in this section is a comparison of a uniform against a ratio, so it pays to get that comparison exactly right. For $u$ uniform on $[0, 1)$ and any $r \ge 0$, $P(u < r) = \min(1, r)$. The comparison must be strict. For a continuous $u$, $P(u \le r)$ would be the same, but floating-point uniforms are discrete (PCG32 gives multiples of $2^{-53}$), and the point $u = 0$ combined with $p_x = 0$ would accept a token the target never produces. Notice also what happens at the other end: with $r = p_x / q_x \ge 1$, every $u < 1$ is below it, so an under-proposed token is always kept.

## Accept, or draw from the residual

Speculative decoding cannot use von Neumann's retry loop as it stands. The target's forward pass has already happened, and a rejected position must produce a token right away; another try would mean another forward pass. So change the procedure: keep $x \sim q$ with probability $\min(1, p_x/q_x)$, and on rejection draw from a carefully chosen fallback. To find that fallback, look at what acceptance keeps. The kept mass of token $t$ is

$$q_t \min\!\left(1, \frac{p_t}{q_t}\right) = \min(p_t, q_t),$$

and the total kept mass is $\alpha = \sum_t \min(p_t, q_t)$. Since $\min(a, b) = \tfrac12(a + b - \lvert a - b\rvert)$ and both distributions sum to 1, $\alpha = 1 - \mathrm{TV}(p, q)$: the closer the draft is to the target, the more of it is accepted. What is missing from $p$ after acceptance is $p_t - \min(p_t, q_t) = \max(0, p_t - q_t)$, which sums to $Z = 1 - \alpha$. Normalizing that missing mass gives the residual distribution.

## The theorem

You now have both halves of the procedure, and they fit together exactly. Draw $x \sim q$, accept with probability $\min(1, p_x/q_x)$, and otherwise draw $t \sim \mathrm{res}(p, q)$. Then

$$P(\text{output} = t) = \underbrace{\min(p_t, q_t)}_{\text{accepted } t} + \underbrace{(1 - \alpha)}_{\text{rejected}} \cdot \frac{\max(0, p_t - q_t)}{1 - \alpha} = \min(p_t, q_t) + \max(0, p_t - q_t) = p_t.$$

Nothing was assumed about $q$ beyond its being a distribution: a bad draft makes the method slow, never wrong. When $p = q$, $Z = 0$ and the residual is undefined, but then $\alpha = 1$ and the residual is never sampled; the contract returns $p$ in that case so that the function stays total.

## Determinism

Correct in distribution is not quite enough. The same request, with the same seed, should accept and reject the same drafts in the Python reference and in the Rust engine of [Speculative decoding in the engine](../../ai-engineering/23-model-serving/09-speculative-decoding-in-the-engine.md) in *AI Engineering*. The sums in $\alpha$ and $Z$ are therefore left-to-right loops over ascending ids in float64, and $u$, $u'$ come from the request's generator in the order fixed by [Speculative decoding: n-gram, prompt-lookup, and model drafts](../../ai-engineering/21-inference/06-speculative-decoding.md). The Rust engine repeats the same arithmetic. numpy's `np.sum` sums pairwise and can differ in the last bit, which would move a boundary of the residual's CDF and change a draw.

## Working it by hand

The theorem is short enough that it is worth checking on numbers. Take four ids, with target $p = [\tfrac12, \tfrac14, \tfrac14, 0]$ and draft $q = [\tfrac14, \tfrac12, 0, \tfrac14]$. The table lays out each token's ratio, its chance of being kept when proposed, the mass it keeps, and the mass left missing.

| id $t$ | $p_t$ | $q_t$ | $r_t = p_t/q_t$ | $P(\text{keep} \mid x = t)$ | kept mass $\min(p_t, q_t)$ | $\max(0, p_t - q_t)$ |
|---|---|---|---|---|---|---|
| 0 | 1/2 | 1/4 | 2 | 1 | 1/4 | 1/4 |
| 1 | 1/4 | 1/2 | 1/2 | 1/2 | 1/4 | 0 |
| 2 | 1/4 | 0 | (never proposed) | | 0 | 1/4 |
| 3 | 0 | 1/4 | 0 | 0 | 0 | 0 |

The acceptance probability is $\alpha = \tfrac14 + \tfrac14 = \tfrac12$; the residual mass is $Z = \tfrac12$, and the residual is $[\tfrac12, 0, \tfrac12, 0]$. Now add up the output distribution token by token. Id 0 is kept with mass $\tfrac14$, plus the rejected mass $\tfrac12$ times its residual probability $\tfrac12$, for a total of $\tfrac12$. Id 1 is kept with mass $\tfrac14$ and has residual 0, so it gets $\tfrac14$. Id 2 is never proposed but comes back through the residual: $\tfrac12 \cdot \tfrac12 = \tfrac14$. Id 3 is proposed a quarter of the time and always rejected, so it gets 0. That is exactly $p$.

Follow one concrete draw too. The draft proposes $x = 1$; $u = 0.75 \ge r_1 = 0.5$ rejects it, and $u' = 0.75$ walks the residual's CDF $[0.5, 0.5, 1.0, 1.0]$ to id 2. With $u = 0.25$ instead, id 1 is kept and $u'$ is never looked at. These are the first cases the tests check.

## Designing the interface

The module exposes five functions: the single acceptance test, the acceptance probability, the residual, one speculative step, and a full von Neumann sampler.

```python
# python/tinyllm/prob/rejection.py
def rejection_accept(p_x: float, q_x: float, u: float) -> bool               # u < p_x / q_x
def acceptance_probability(p: ArrayLike, q: ArrayLike) -> float              # sum min(p, q)
def residual_distribution(p: ArrayLike, q: ArrayLike) -> NDArray             # normalize(max(0, p - q)); p if p == q
def speculative_step(p, q, x: int, u_accept: float, u_resample: float) -> tuple[int, bool]
def rejection_sample(p, q, m: float, rng: UniformSource, max_tries: int = 10000) -> tuple[int, int]
```

`speculative_step` resamples with `sample_categorical` from [Categorical sampling](02-categorical-sampling.md). `rejection_sample` draws exactly two uniforms per try, proposal first. The contract has every error rule.

## Building it step by step

The implementation follows the derivation closely, and each choice traces back to one of the principles above.

First, both inputs are distributions over one vocabulary. Every function starts by checking $p$ and $q$ with the rules from [Categorical sampling](02-categorical-sampling.md) (1-D, finite, non-negative, summing to 1) and that their shapes match. A draft over a different vocabulary is a bug, not a case to handle.

Second, acceptance is one strict comparison. As shown above, `u < p_x / q_x` has probability exactly $\min(1, r)$ for $u$ on the discrete grid in $[0, 1)$. No explicit `min` is needed, because $u < 1 \le r$ already accepts. If $q_x = 0$, the draft proposed a token it gave probability 0, so the function raises rather than divides.

Third, the acceptance probability and the residual's normalizer are left-to-right sums: a plain loop over ascending ids, so the Rust port gets the same last bit and the residual's CDF has the same boundaries.

Fourth, the residual is $\max(0, p - q)/Z$, or $p$ when $Z = 0$. $Z = 0$ happens exactly when $p \le q$ everywhere, which for two distributions means $p = q$; then nothing is ever rejected, and returning a copy of $p$ keeps the function total without aliasing the caller's array.

Fifth, one speculative step is accept, or one draw from the residual. If the acceptance test passes, return $x$ without reading $u'$; otherwise call `sample_categorical(residual, u')`. That draw order is part of the parity contract with the Python and Rust speculative decoders.

Sixth, von Neumann needs a checked envelope and two uniforms per try. Check $m \ge 1$ and $p \le mq$ everywhere first, then loop: propose with one uniform, accept with another when $u < p_x/(m q_x)$, and return the token and the number of tries. A bounded loop turns a pathological envelope into an error rather than a hang.

The first two functions are the acceptance test and its probability.

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

On the hand example, the per-token kept masses are $\tfrac14, \tfrac14, 0, 0$, so $\alpha = \tfrac12$. The residual and the speculative step come next.

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

Trace the hand example's draw through this code: $x = 1$, and $u = 0.75 \ge 0.5$ rejects; the residual is $[\tfrac12, 0, \tfrac12, 0]$ with running sums $0.5, 0.5, 1.0, 1.0$, and $u' = 0.75$ first falls below $1.0$ at id 2. With $u = 0.25 < 0.5$ the function returns `(1, True)`. In the disjoint-support case $\alpha = 0$, $r = p$ and $Z = 1$, so the residual is $p$ itself: a useless draft is always rejected and the output is still a draw of $p$.

Finally, the von Neumann sampler checks its envelope and loops.

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

`sample_categorical` never returns an id with $q_x = 0$ (its comparison is strict), so the division is safe; and the envelope check guarantees that any id with $p_x > 0$ has $q_x > 0$. The full solution is [`rejection.py`](../../../../course/solutions/python/tinyllm/prob/rejection.py).

### Why this design

Consider the cost first. `rejection_accept` is $O(1)$; `acceptance_probability` and `residual_distribution` are $O(n)$; `speculative_step` is $O(n)$ (validation, the residual, and an inverse-CDF draw) and is only computed in full on rejection. `rejection_sample` costs $O(n)$ checks plus a geometric number of tries with mean $m$, each $O(n)$ for the inverse CDF.

The residual beats a retry loop for speculative decoding. A von Neumann retry would also be exact, but each retry would need another target forward pass; the residual produces a token from the pass already done, so every verified position yields exactly one token.

Explicit loops are chosen over `np.sum` deliberately. numpy's pairwise sum is more accurate for long arrays, but it is not what the Rust port computes. The sums here are of at most a vocabulary of non-negative terms, where left-to-right error is tiny, and bit parity matters more.

One tradeoff is accepted knowingly: validating $p$ and $q$ on every call costs $O(n)$ per step. For a 50 000-token vocabulary that is small next to a forward pass, and it turns a draft/target vocabulary mismatch into an immediate error.

## What can go wrong

A correct implementation has to survive a specific set of checks. The hand example anchors the definitions: its ratios, $\alpha = 1/2$, and the residual $[1/2, 0, 1/2, 0]$ must come out exactly, and its output distribution, enumerated exactly, must equal $p$. Strict acceptance is a boundary check: $u = r$ rejects and $p_x = 0$ never accepts, because discrete uniforms make $\le$ wrong. When the target dominates, $r \ge 1$ must accept every $u < 1$, so under-proposed tokens are always kept. Bad acceptance arguments ($q_x = 0$, $u \notin [0, 1)$, $p_x < 0$) must raise, since a silent answer biases the output.

The statistical checks cover the theorem itself. The enumerated acceptance rate must equal $\sum \min(p, q)$, because the speculative decoder reports it as its acceptance rate. The residual must be a distribution (non-negative, summing to 1, with $Z = 1 - \alpha$), since `sample_categorical` demands one. For random $p$ and $q$ in eighths, the speculative output distribution must equal $p$ exactly by enumeration, and 20000 seeded draws on arbitrary distributions must match $p$ under a chi-square test. For von Neumann, one try must accept $x$ with probability exactly $p_x/m$, and the mean number of tries must equal $m$ within 4 standard errors.

The remaining boundary and unit checks protect parity and the edges. $p = q$ must give a copy of $p$: no 0/0, and no aliasing of the caller's array. Disjoint supports give $\alpha = 0$ and residual $p$, so a useless draft is slow, not wrong. The normalizer must be a left-to-right loop over ascending ids for bit parity with the Rust port. Mismatched shapes and non-distributions must raise, because draft and target vocabularies must match. An accepted draft must not look at $u'$, because the draw order is part of the parity contract, and a draft $x$ outside the vocabulary or with $q_x = 0$ must raise, because draft and proposal disagreeing is a bug. Von Neumann must draw the proposal uniform first and then the acceptance uniform, two per try, to be reproducible across languages, and it must raise for $p_x > m q_x$, for $m < 1$, and when nothing is accepted within `max_tries`, since a wrong envelope silently biases.

The classic mistakes are worth naming one by one, with the line of the solution that avoids each.

> **Warning:** Accepting on $u \le r$ accepts a token with $p_x = 0$ whenever $u = 0$. The solution returns `u < p_x / q_x`.

> **Warning:** Writing the ratio upside down, or "capping" it by inverting it when it exceeds 1, keeps over-proposed tokens and drops under-proposed ones. The ratio is `p_x / q_x`, and $u < 1$ does the capping for free.

The residual has its own traps. Using $\max(0, q - p)$ or $\lvert p - q\rvert$ resamples the tokens the draft over-proposed; the solution computes `np.maximum(a - b, 0.0)`. Forgetting to normalize makes `sample_categorical` reject the residual, or leaves its CDF ending below 1; the solution divides `r / z`. Handling $p = q$ as 0/0 produces NaN, and returning the caller's array lets a later in-place edit corrupt the target; the solution returns `a.copy()` when `z <= 0.0`.

On rejection, resampling from $p$, from $q$, or simply returning the draft means the output is no longer $p$, because accepted tokens are effectively counted twice; the solution samples `residual_distribution(a, b)`. Running von Neumann without $m$, or without checking it, makes accepted tokens follow $\min(p, q)$-ish weights rather than $p$; the solution accepts on `a[x] / (m * b[x])` after checking `m >= 1` and `a <= m * b`. And summing with numpy's pairwise `np.sum` produces last-bit differences against the Rust port, which the explicit left-to-right loops avoid.

## Where this leads

The residual draw uses `sample_categorical` and the `UniformSource` generator type from [Categorical sampling](02-categorical-sampling.md), and every probability above rests on the uniform random variables and independence of [Random variables, expectation, variance, and normal draws by Box-Muller](01-random-variables-and-the-normal.md). Looking forward, [Speculative decoding: n-gram, prompt-lookup, and model drafts](../../ai-engineering/21-inference/06-speculative-decoding.md) in *AI Engineering* calls one `speculative_step` per draft position and reports `acceptance_probability` as its acceptance rate, building on the token sampler of [Sampling and logit processors](../../ai-engineering/21-inference/01-sampling-and-logit-processors.md). [Speculative decoding in the engine](../../ai-engineering/23-model-serving/09-speculative-decoding-in-the-engine.md) ports the same logic to Rust and is held to the same accept and reject decisions.

Production systems use the same ideas at larger scale. vLLM's `RejectionSampler` (`vllm/v1/sample/rejection_sampler.py`) performs batched acceptance over a whole draft tree on the GPU, recovering tokens from the residual. Hugging Face assisted generation uses the same residual for model drafts in `_speculative_sampling` (`transformers/generation/utils.py`). NumPy's Gamma and binomial samplers (`numpy/random/src/distributions/distributions.c`) use rejection with tight squeeze functions to avoid evaluating the target.

### Further reading

- Devroye, *Non-Uniform Random Variate Generation* (1986), ch. 2.3.
- Leviathan, Kalman, and Matias, "Fast Inference from Transformers via Speculative Decoding" (2023), appendix A.1.
- Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling" (2023).

## Key terms

- **Rejection sampling**: drawing from a proposal $q$ and accepting each draw with a probability chosen so that accepted draws follow the target $p$.
- **Envelope**: a constant $m \ge 1$ with $p_i \le m q_i$ for all $i$; von Neumann's sampler needs one, and its mean cost is $m$ tries.
- **Acceptance ratio**: $r_x = p_x / q_x$; the draft token $x$ is kept with probability $\min(1, r_x)$.
- **Acceptance probability**: $\alpha = \sum_i \min(p_i, q_i)$, the chance a draft token is kept.
- **Total variation distance**: $\mathrm{TV}(p, q) = \tfrac12 \sum_i \lvert p_i - q_i\rvert$; it satisfies $\alpha = 1 - \mathrm{TV}(p, q)$.
- **Residual distribution**: $\max(0, p - q)/Z$, the normalized mass that acceptance leaves missing from $p$.

## Summary

- Keeping a draft $x \sim q$ with probability $\min(1, p_x/q_x)$ keeps exactly $\min(p_x, q_x)$ of every token's mass; the total kept is $\sum_x \min(p_x, q_x) = 1 - \mathrm{TV}(p, q)$.
- What rejection removes is exactly the residual $\max(0, p - q)$, so resampling from its normalized form restores $p$ token for token: the output of `speculative_step` is distributed as $p$ for every $q$.
- "Accept when $u < r$" with $u$ uniform on $[0, 1)$ has probability exactly $\min(1, r)$; "$u \le r$" counts one extra point and accepts tokens $p$ never produces.
- Von Neumann's sampler accepts a proposal with probability $p_x / (m q_x)$; each try succeeds with probability $1/m$, so the cost is a geometric number of tries with mean $m$.
- Left-to-right sums and a fixed draw order make the Python and Rust implementations agree on every decision for the same seed.

## Review questions

1. Why is the residual $\max(0, p - q)$ and not $\max(0, q - p)$ or $\lvert p - q\rvert$?

   <details><summary>Answer</summary>

   After acceptance each token has $\min(p_t, q_t)$ of its target mass; what is missing is $p_t - \min(p_t, q_t) = \max(0, p_t - q_t)$, the tokens the draft under-proposed. $\max(0, q - p)$ would resample the tokens the draft over-proposed, which already have more than enough, and $\lvert p - q \rvert$ mixes both; neither adds up to $p$.

   </details>

2. Why does `rejection_accept` need no `min(1, ...)`?

   <details><summary>Answer</summary>

   $u$ is always below 1, so when $r \ge 1$ the test `u < r` is always true: the acceptance probability is already $\min(1, r)$. Writing an explicit cap is harmless, but "capping" by inverting the ratio when it exceeds 1 would turn the always-accept case into a partial one and drop under-proposed tokens.

   </details>

3. On acceptance, why must `speculative_step` return without using $u'$?

   <details><summary>Answer</summary>

   Correctness: the theorem's accept branch outputs $x$ itself, contributing $\min(p_x, q_x)$; letting $u'$ change the result there would replace accepted tokens with residual draws and break that term. Parity: the contract says $u'$ matters only on rejection, so the Python speculative decoder and the Rust engine agree on every decision for the same uniforms.

   </details>

4. Why does `residual_distribution` return `a.copy()` when $Z = 0$ and not `a`?

   <details><summary>Answer</summary>

   $Z = 0$ means $p = q$, so the residual is never sampled, but the function must still return a valid distribution and $p$ is the natural choice. Returning the caller's array would let a later in-place edit of the result (a mask, a temperature) silently change the target distribution itself.

   </details>

5. In von Neumann's sampler, what does a loose envelope (large $m$) cost, and what does a too-small one break?

   <details><summary>Answer</summary>

   A loose envelope only costs time: each try accepts with probability $1/m$, so the mean number of tries is $m$. A too-small one breaks correctness: for an $x$ with $p_x > m q_x$ the acceptance ratio exceeds 1 and is effectively capped, so $x$ is drawn with probability below $p_x$ and the others are over-sampled. That is why the function checks the envelope before drawing.

   </details>

6. In the hand example, $q_2 = 0$ but the output still produces id 2 a quarter of the time. Where does that probability come from?

   <details><summary>Answer</summary>

   Entirely from the residual. Id 2 is never proposed, so its kept mass is 0, but its missing mass $\max(0, p_2 - q_2) = \tfrac14$ makes up half of $Z = \tfrac12$, so its residual probability is $\tfrac12$. The rejection probability is $1 - \alpha = \tfrac12$, giving $\tfrac12 \cdot \tfrac12 = \tfrac14 = p_2$.

   </details>
