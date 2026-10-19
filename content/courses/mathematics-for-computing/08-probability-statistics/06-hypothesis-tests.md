# Hypothesis tests: paired and two-sample permutation, McNemar, Holm

Once you can train real models, every change to one is a claim: "pre-LN trains better", "this LoRA rank matches full fine-tuning", "int8 [quantization](../../build-an-llm/09-inference/05-quantization.md) costs nothing". The evidence is a score on a finite eval set, and two runs of the same model already differ by a few tenths of a point. The previous section, [LLN, CLT, confidence intervals, bootstrap](05-lln-clt-confidence-intervals-bootstrap.md), put an interval on each number. This section answers the comparison question directly: is the gap between model A and model B on the same 200 prompts larger than the gap chance alone produces?

The [model zoo](../../build-an-llm/07-objectives/07-lm-evaluation-harness-and-the-model-zoo.md) in *Build Your Own LLM System* prints a p-value next to every comparison, the quantization and [speculative decoding](../../build-an-llm/09-inference/06-speculative-decoding.md) sections fail their "no regression" checks on one, and the zoo compares many models at once, which needs a correction or it will "discover" a winner by luck. All of these call the six functions you build here: a paired and a two-sample permutation test, McNemar's test for classifiers, and Holm's correction for many comparisons.

No code from earlier sections is needed. The reading background is the sampling distributions and intervals of the previous section and [problem set C](11-problem-set-c.md). The reference implementation is [`course/solutions/python/tinyllm/prob/tests.py`](../../../../course/solutions/python/tinyllm/prob/tests.py), with its contract in [`tests.pyi`](../../../../course/contracts/py/tinyllm/prob/tests.pyi); the generator behind `rng` is specified in [`spec/pcg32.md`](../../../../course/contracts/spec/pcg32.md).

> **Learning objectives**
> - State what a p-value is and is not, and what type-I error rate a valid test promises.
> - Implement paired and two-sample permutation tests with the add-one p-value, two-sidedness, and tie-safe comparison.
> - Implement a correct Fisher-Yates shuffle and explain why the naive shuffle is biased.
> - Compute McNemar's exact and approximate p-values for two classifiers.
> - Apply Holm's step-down procedure and compute adjusted p-values.

## Notation

| Symbol | Meaning | Type / shape |
|---|---|---|
| $H_0$ | the null hypothesis: "no difference" in a precise form (exchangeable labels) | |
| $T$ | the test statistic, larger meaning more extreme | `float` |
| $t_{\text{obs}}$ | $T$ on the data as observed | `float` |
| $p$ | the p-value, $P(T \ge t_{\text{obs}} \mid H_0)$ | `float` in $[0, 1]$ |
| $\alpha$ | the significance level: reject $H_0$ when $p \le \alpha$ | `float`, often 0.05 |
| $a_i, b_i$ | the scores of models A and B on prompt $i$ (paired) | `float64[n]` |
| $d_i = a_i - b_i$ | paired differences | `float64[n]` |
| $s_i \in \{-1, +1\}$ | a random sign per pair | |
| $B$ | the number of random permutations, `n_perm` | `int` |
| $b_{01}, b_{10}$ | counts of examples where only A is right, and only B is right | `int` |
| $m$, $p_{(k)}$ | the number of tests in a family, and the $k$-th smallest p-value ($k = 0, \ldots, m-1$) | |

## Testing as a thought experiment

Suppose B scores 0.8 points above A on 50 prompts. Two stories explain it: B is better, or the prompts happened to favour B. A test makes the second story precise as a null hypothesis $H_0$, picks a statistic $T$ that is large when the data disagree with $H_0$, and asks how often $H_0$ would produce a $T$ at least as large as the observed one. That probability is the **p-value**. A small p-value says the data would be surprising if $H_0$ were true; it is not the probability that $H_0$ is true, and a large one does not prove $H_0$ (the test may just lack data).

## What a p-value promises

The decision rule is to reject $H_0$ whenever $p \le \alpha$. If $H_0$ is true, a valid p-value satisfies $P(p \le \alpha) \le \alpha$: a rejection is a false alarm (a **type-I error**) at most a fraction $\alpha$ of the time. Not rejecting a false $H_0$ is a **type-II error**; the probability of rejecting when the effect is real is the **power**. A test that never rejects has a perfect type-I rate and no power, so a correct solution is judged on both. "Different" is usually **two-sided**: B worse is as interesting as B better, so $T$ is a magnitude, $|\cdot|$, and swapping A and B must not change $p$.

## The paired permutation test

When both models answer the same prompts, compare them prompt by prompt: $d_i = a_i - b_i$. Under $H_0$ ("which model produced which score of a pair is arbitrary"), each $d_i$ was as likely to come out as $-d_i$. So the null world is: flip each sign with probability $\frac{1}{2}$. With $T(s) = |\sum_i s_i d_i|$ and $t_{\text{obs}} = T(1, \ldots, 1)$, the exact p-value is the fraction of all $2^n$ sign vectors with $T(s) \ge t_{\text{obs}}$.

For $n = 50$ that is $10^{15}$ vectors, so draw $B$ of them at random. The natural estimate $\text{count}/B$ can be 0, which no true p-value is. Counting the observed labelling as one more draw,

$$p = \frac{1 + \#\{b : T(s^{(b)}) \ge t_{\text{obs}}\}}{1 + B},$$

fixes that, and it is exactly valid: under $H_0$ the observed $T$ is one of $B + 1$ exchangeable values, its rank is uniform, so $P(p \le \alpha) = \lfloor \alpha (B + 1) \rfloor / (B + 1) \le \alpha$. With $B = 99$, $P(p \le 0.1)$ is exactly 0.1: the rate a type-I check measures. Pairing matters: prompt difficulty varies far more than the model gap, and pairing cancels it.

## The two-sample permutation test

Not every comparison is paired. When the two samples are not (latencies of two server builds, scores on different eval sets), $H_0$ says the group labels are arbitrary: all $N = n_a + n_b$ values come from one distribution. The null world deals the pooled values into two groups of the original sizes at random, and $T = |\text{stat}(a', b')|$, by default the difference of means.

A uniformly random order of $N$ items comes from **Fisher-Yates**: for $i = N - 1$ down to 1, pick $j$ uniformly in $\{0, \ldots, i\}$ and swap positions $i$ and $j$. Each of the $N!$ orders comes out with probability $1/N!$. The tempting shortcut, $j$ uniform over all $N$ positions every time, produces $N^N$ equally likely swap sequences, which is not a multiple of $N!$, so some orders are more likely than others.

## McNemar's test

Classifiers give right-or-wrong answers, and for those there is an exact test that needs no simulation. Two classifiers on the same $n$ examples give a 2 by 2 table: both right, both wrong, only A right ($b_{01}$), only B right ($b_{10}$). The examples where they agree say nothing about which is better. Under $H_0$ each disagreement is a fair coin, so $b_{01} \sim \text{Binomial}(b_{01} + b_{10}, \frac{1}{2})$, and the exact two-sided p-value doubles the smaller tail:

$$p = \min\Bigl(1,\ 2 \sum_{i=0}^{k} \binom{n}{i} 2^{-n}\Bigr), \qquad n = b_{01} + b_{10},\ k = \min(b_{01}, b_{10}).$$

The doubling can exceed 1 when $b_{01} = b_{10}$, hence the cap. For large $n$ the classic approximation is $\chi^2 = (|b_{01} - b_{10}| - 1)^2 / n$ on one degree of freedom (the $-1$ is Edwards' continuity correction), with $p = \operatorname{erfc}(\sqrt{\chi^2/2})$. With no disagreements at all there is no evidence: $p = 1$.

## Many comparisons and Holm's method

Each test so far controls its own false-alarm rate. A table of comparisons needs more. Test $m = 10$ true null hypotheses at $\alpha = 0.05$ and the chance of at least one false alarm is $1 - 0.95^{10} = 0.40$. The **family-wise error rate** (FWER) is that chance. **Bonferroni** tests each at $\alpha/m$, which bounds the FWER by $\alpha$ (a union bound) but is needlessly strict.

**Holm** is uniformly better. Sort the p-values, compare the smallest with $\alpha/m$, the next with $\alpha/(m-1)$, and so on, rejecting while each passes and **stopping at the first that does not**. Once a hypothesis is rejected, only $m - 1$ can still be true nulls, which is why the bar relaxes. The **adjusted p-values** $\tilde p_{(k)} = \max_{j \le k} \min(1, (m - j) p_{(j)})$ make Holm a table column: reject exactly where $\tilde p \le \alpha$. The running maximum keeps the order: a smaller raw p-value never gets a larger adjusted one.

## Working it by hand

**Paired.** Model B minus model A on three prompts: $d = (2, 1, 3)$, $t_{\text{obs}} = 6$. All eight sign patterns:

| $s$ | $+{+}{+}$ | $+{+}{-}$ | $+{-}{+}$ | $+{-}{-}$ | $-{+}{+}$ | $-{+}{-}$ | $-{-}{+}$ | $-{-}{-}$ |
|---|---|---|---|---|---|---|---|---|
| $\sum s_i d_i$ | 6 | 0 | 4 | $-2$ | 2 | $-4$ | 0 | $-6$ |
| $\lvert \cdot \rvert \ge 6$ | yes | | | | | | | yes |

The exact p-value is $2/8 = 1/4$: with three prompts, even a unanimous win is not convincing. Fed exactly these eight patterns as its $B = 8$ permutations, the add-one estimate is $(1 + 2)/(1 + 8) = 1/3$.

**McNemar.** On 40 prompts, A alone is right on $b_{01} = 1$, B alone on $b_{10} = 7$. Then $n = 8$, $k = 1$, the tail is $\binom{8}{0} + \binom{8}{1} = 9$ out of $2^8 = 256$, and $p = 2 \cdot 9/256 = 9/128 = 0.0703$. Not significant at 0.05, despite 7 to 1. The approximation gives $\chi^2 = (6 - 1)^2/8 = 3.125$ and $p = \operatorname{erfc}(1.25) = 0.0771$.

**Holm.** Four comparisons with p-values $(0.01, 0.04, 0.02, 0.005)$ at $\alpha = 0.05$ step down as follows.

| sorted $k$ | $p_{(k)}$ | bar $\alpha/(m-k)$ | reject? | $(m-k)\,p_{(k)}$ | adjusted |
|---|---|---|---|---|---|
| 0 | 0.005 | 0.0125 | yes | 0.02 | 0.02 |
| 1 | 0.01 | 0.0167 | yes | 0.03 | 0.03 |
| 2 | 0.02 | 0.025 | yes | 0.04 | 0.04 |
| 3 | 0.04 | 0.05 | yes | 0.04 | 0.04 |

Holm rejects all four; Bonferroni (every p against 0.0125) rejects only 0.005 and 0.01. In input order the adjusted values are $(0.03, 0.04, 0.04, 0.02)$. These are the first three test cases a correct implementation must reproduce.

## Designing the interface

Six functions cover the section:

```python
def mean_difference(a: NDArray, b: NDArray) -> float: ...
def paired_permutation_test(a: ArrayLike, b: ArrayLike, n_perm: int, rng: UniformSource) -> float: ...
def permutation_test(a: ArrayLike, b: ArrayLike, stat: Callable[[NDArray, NDArray], float],
                     n_perm: int, rng: UniformSource) -> float: ...
def mcnemar(b01: int, b10: int, exact: bool = True) -> float: ...
def holm(pvalues: ArrayLike, alpha: float = 0.05) -> NDArray: ...      # bool, True = rejected
def holm_adjust(pvalues: ArrayLike) -> NDArray: ...
```

The draw order is part of the contract, because the [run comparison and regression gate](../../build-an-llm/11-serving/12-run-comparison-and-regression-gate.md) reimplements the two-sample test in Go and must reproduce these p-values from the same seed. The paired test draws one uniform per pair per permutation ($u < 0.5$ flips). The two-sample test restarts from the identity order each permutation and runs Fisher-Yates from the end with $j = \min(\lfloor u(i+1) \rfloor, i)$. A permuted statistic counts as "at least as extreme" when it is at least $t_{\text{obs}}(1 - 10^{-9})$, so rounding never splits a tie.

## Building it step by step

Seven decisions produce the implementation.

1. **A permutation test is "count how often the null world is as extreme".** Compute $t_{\text{obs}}$ once, draw $B$ relabellings the null says are harmless, count those with $T \ge t_{\text{obs}}$, return $(1 + \text{count})/(1 + B)$. Everything else is which relabelling and which $T$.
2. **Two-sided means a magnitude.** $T = |\cdot|$ on both the observed and the permuted statistic, so swapping A and B (or negating the statistic) cannot change $p$.
3. **Ties need a relative bar.** Scores like 0.7 and 0.2 tie in exact arithmetic but their float sums can differ by an ulp depending on the order of addition. Comparing against $t_{\text{obs}}(1 - 10^{-9})$ counts those ties as "at least as extreme", which is what the exact test does.
4. **Paired relabelling is a sign flip per pair, one uniform each.** $u < 0.5$ flips; $B n$ uniforms in all, in pair order, so the stream is fixed.
5. **Two-sample relabelling is a fresh Fisher-Yates shuffle each time.** Start from the identity, for $i = N-1$ down to 1 pick $j = \min(\lfloor u(i+1) \rfloor, i)$, swap; then the first $n_a$ are group A. $N - 1$ uniforms per permutation. Restarting from the identity is what makes the permutation a function of its own $N - 1$ uniforms only.
6. **McNemar exact is a binomial tail in integers.** $p = \min(1, 2\sum_{i \le k}\binom{n}{i}/2^n)$, with Python's exact integers until one final division, so $n$ in the hundreds neither overflows nor underflows. The approximation applies Edwards' $-1$ and clamps at 0 so a tie gives $\chi^2 = 0$, $p = 1$.
7. **Holm sorts, compares with a relaxing bar, and stops.** Reject the $k$-th smallest while $p_{(k)} \le \alpha/(m - k)$; break at the first failure. The adjusted p-values are the running maximum of $\min(1, (m-k)p_{(k)})$, written back to input positions.

The paired test comes first:

```python
TIE_RTOL = 1e-9


def paired_permutation_test(a, b, n_perm: int, rng) -> float:
    x, y = _vector(a, "a"), _vector(b, "b")          # non-empty, 1-D, finite
    if x.shape != y.shape:
        raise ValueError(f"a and b must pair up, got lengths {x.size} and {y.size}")
    n_perm = _n_perm(n_perm)                         # an integer >= 1
    d = x - y
    t_obs = abs(float(np.sum(d)))                    # two-sided: a magnitude
    bar = t_obs * (1.0 - TIE_RTOL)                   # ties a rounding error below still count
    count = 0
    for _ in range(n_perm):
        # Under H0 the two scores of a pair could have come from either model:
        # each difference keeps or flips its sign with probability 1/2.
        s = np.array([-1.0 if rng.uniform() < 0.5 else 1.0 for _ in range(d.size)])
        if abs(float(np.sum(s * d))) >= bar:
            count += 1
    return (1 + count) / (1 + n_perm)                # the observed labelling is one more draw
```

When all differences are 0, $t_{\text{obs}} = 0$, `bar = 0`, every permutation counts, and $p = (1 + B)/(1 + B) = 1$: no evidence, as it should be. With the hand example's $d = (2, 1, 3)$, `bar` is just below 6, and the patterns $+{+}{+}$ and $-{-}{-}$ count.

The two-sample test differs only in the relabelling:

```python
def permutation_test(a, b, stat, n_perm: int, rng) -> float:
    x, y = _vector(a, "a"), _vector(b, "b")
    n_perm = _n_perm(n_perm)
    z = np.concatenate([x, y])                        # the pooled values
    na, total = x.size, z.size
    t_obs = abs(float(stat(x, y)))
    bar = t_obs * (1.0 - TIE_RTOL)
    count = 0
    for _ in range(n_perm):
        # Under H0 the group labels are arbitrary: deal the pooled values into
        # two groups of the original sizes, uniformly over all orders.
        idx = list(range(total))                      # restart from the identity every time
        for i in range(total - 1, 0, -1):             # Fisher-Yates from the end
            j = min(int(rng.uniform() * (i + 1)), i)  # j uniform on {0, ..., i}; clamp guards rounding
            idx[i], idx[j] = idx[j], idx[i]
        p = z[idx]
        if abs(float(stat(p[:na], p[na:]))) >= bar:   # same sizes as the data
            count += 1
    return (1 + count) / (1 + n_perm)
```

`stat` is any function of the two groups; `mean_difference` (`np.mean(a) - np.mean(b)`) is the default the callers pass. The absolute value is applied here, not inside `stat`, so a caller's signed statistic still gives a two-sided test.

McNemar has an exact and an approximate branch:

```python
def mcnemar(b01: int, b10: int, exact: bool = True) -> float:
    # (integer >= 0 checks elided)
    n = b01 + b10
    if n == 0:
        return 1.0                                     # no disagreements: no evidence
    if exact:
        # Under H0 each discordant pair is a fair coin: b01 ~ Binomial(n, 1/2).
        # Two-sided: double the smaller tail. Integers until the one division.
        k = min(b01, b10)
        tail = sum(math.comb(n, i) for i in range(k + 1))
        return min(1.0, 2 * tail / 2**n)               # cap: doubling a tie's tail exceeds 1
    chi2 = max(abs(b01 - b10) - 1, 0) ** 2 / n         # Edwards' continuity correction, clamped
    return math.erfc(math.sqrt(chi2 / 2.0))            # chi-square(1) survival via the normal
```

On the hand example, `tail = 1 + 8 = 9` and `2 * 9 / 256 = 9/128`. The approximate branch gives $\chi^2 = 25/8$ and $\operatorname{erfc}(\sqrt{1.5625}) = \operatorname{erfc}(1.25) = 0.0771$. The identity $P(\chi^2_1 > c) = \operatorname{erfc}(\sqrt{c/2})$ holds because $\chi^2_1$ is the square of a standard normal.

Holm and its adjustment share a stable sort:

```python
def holm(pvalues, alpha: float = 0.05) -> NDArray:
    p = _pvalues(pvalues)                              # 1-D, every value in [0, 1]
    # (alpha check elided)
    m = p.size
    order = np.argsort(p, kind="stable")               # stable: ties keep input order
    reject = np.zeros(m, dtype=bool)
    for k, i in enumerate(order):
        # The smallest p-value faces the strictest bar alpha / m; each
        # rejection removes one hypothesis from the family.
        if p[i] > alpha / (m - k):
            break                                      # step-down: nothing after a failure is rejected
        reject[i] = True                               # equality rejects
    return reject


def holm_adjust(pvalues) -> NDArray:
    p = _pvalues(pvalues)
    m = p.size
    order = np.argsort(p, kind="stable")
    adj = np.empty(m, dtype=np.float64)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (m - k) * p[i]))   # running max keeps the order
        adj[i] = running                                   # written back in input order
    return adj
```

The hand example's family sorts to $(0.005, 0.01, 0.02, 0.04)$ with multipliers $4, 3, 2, 1$: $0.02, 0.03, 0.04, 0.04$, and in input order $(0.03, 0.04, 0.04, 0.02)$. `holm(p, alpha)` rejects exactly where `holm_adjust(p) <= alpha`. The full solution is [`tests.py`](../../../../course/solutions/python/tinyllm/prob/tests.py).

### Why this design

The paired test is $O(Bn)$ time and $B n$ uniforms; the two-sample test $O(BN)$ plus $B$ calls to `stat`, with $B(N-1)$ uniforms. Exact McNemar is $O(k)$ big-integer binomials; Holm and its adjustment are $O(m \log m)$ for the sort.

Permutation tests are chosen over $t$-tests. A paired $t$-test assumes roughly normal differences; per-prompt scores are 0/1, bounded, or heavy-tailed. The permutation test assumes only exchangeability under $H_0$, which is exactly what "the models are equally good" means here, and its add-one p-value has an exact type-I guarantee at any $n$.

Monte Carlo is chosen over exact enumeration. $2^n$ sign patterns or $\binom{N}{n_a}$ splits are out of reach for real eval sets; $B$ random relabellings give an unbiased estimate whose error shrinks like $1/\sqrt B$, and the add-one form remains valid for any $B$.

Holm is chosen over Bonferroni. Holm controls the same family-wise error rate and rejects a superset of what Bonferroni rejects, at the cost of a sort. Benjamini-Hochberg would control the false discovery rate instead, a weaker promise than "no false winner in the table".

The tradeoff accepted is speed. The scalar loops and a fixed draw order make the functions slower than vectorized numpy, in exchange for a stream that the Go port reproduces draw for draw.

## What can go wrong

A correct solution reproduces the hand values, agrees with exact enumeration and with scipy, keeps its type-I promise, has power, and handles ties and edges. The full list:

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| paired test by hand | unit | $(1 + 2)/(1 + 8) = 1/3$ over the eight patterns | everyone agrees on the statistic and the add-one rule |
| McNemar by hand | unit | $9/128$ exact, $\operatorname{erfc}(1.25)$ approximate | classifier comparisons in the model zoo |
| Holm by hand | unit | all four rejected, adjusted $(0.03, 0.04, 0.04, 0.02)$ | the zoo's comparison table |
| paired against exact enumeration | golden | Monte Carlo within 4.5 standard errors of scipy's exact enumeration | the estimate converges to the right number |
| two-sample against exact enumeration | golden | the same for 6 + 6, 7 + 5, 8 + 8 splits | latency and eval comparisons |
| replay of fixed seeds | golden | exact p-values for fixed seeds | the Go regression gate reproduces them |
| McNemar against scipy | golden | `binomtest` and the chi-square, up to 360 pairs | exact arithmetic, no overflow |
| Holm against a reference | golden | adjusted p and rejections on five families | ties, 0 and 1, a single test |
| paired type-I rate | statistical | $P(p \le 0.1) = 0.1$ under $H_0$ over 300 datasets | false alarms at the promised rate |
| two-sample type-I rate | statistical | the same with groups of 6 and 9 | the shuffle really mixes the groups |
| power | statistical | real shifts are detected at 0.05 | a test that never rejects is useless |
| draw counts | unit | $B n$ and $B(N - 1)$ uniforms | one stream in every language |
| Fisher-Yates order | unit | $u \to 1$ is the identity, $u = 0$ deals $(5, 1 \mid 2, 3)$ | the shuffle as specified |
| two-sidedness | property | swapping A and B, or negating the statistic, leaves $p$ unchanged | "different", not "better" |
| add-one and degenerate data | boundary | $p > 0$ always; no differences give $p = 1$ | no "p = 0" in a report |
| rounding never splits a tie | boundary | $d = (-0.7, 0.2, 0.7, 0.1)$ counts 12 of 16 ties | decimal scores tie in exact arithmetic |
| Holm steps down and stops | property | stops at the first failure; equality rejects; superset of Bonferroni | the step-down rule |
| Holm adjustment is monotone and capped | property | adjusted p sorted like raw p, at most 1 | a trustworthy table column |
| McNemar edges | boundary | no disagreements or a tie give 1; symmetric | small eval sets |
| bad arguments | boundary | bad lengths, NaN, `n_perm < 1`, bad counts, p outside $[0, 1]$ raise | bugs surface at the call |

The pitfalls group by the property they break.

Validity breaks first. Forgetting the $|\cdot|$ makes a comparison one-sided by accident, so $p$ is near 1 when B is worse and too small when it is better; the solution applies `abs(...)` to both the observed and permuted statistic. Returning $\text{count}/B$ without the add-one puts "p = 0.000" in a report and breaks the type-I promise; the solution returns `(1 + count) / (1 + n_perm)`. Comparing permuted statistics with a bare $\ge$ misses exact ties that land an ulp below, so $p$ is too small; `bar = t_obs * (1.0 - TIE_RTOL)` catches them.

> **Warning:** The naive shuffle, $j$ drawn over the whole array, makes some orders more likely than others and produces a different stream from the Go port. Use `j = min(int(rng.uniform() * (i + 1)), i)`. Shuffling on from the previous permutation instead of restarting from the identity is still a valid test, but it is not the contract's stream.

A relabelling that never mixes the groups makes every permutation equal the data: $p = 1$ and no power. The solution runs a full Fisher-Yates over the pooled values, then splits at the original group sizes.

McNemar has three ways to go wrong: without doubling $p$ is halved, without the cap it can exceed 1, and without the continuity correction it is too small. The solution uses `min(1.0, 2 * tail / 2**n)` and `max(abs(b01 - b10) - 1, 0)`.

Holm has three more. Using Bonferroni where Holm was meant misses real improvements; the bar is `alpha / (m - k)`, not `alpha / m`. Not stopping at the first failure rejects a later hypothesis past a failed bar and breaks the FWER; the `break` prevents it. Computing adjusted p-values without the running maximum lets a smaller raw p get a larger adjusted one; `running = max(running, ...)` keeps them ordered.

## Where this leads

[Problem set D](12-problem-set-d.md) works these tests by hand in its first four questions. In *Build Your Own LLM System*, the [model zoo](../../build-an-llm/07-objectives/07-lm-evaluation-harness-and-the-model-zoo.md) prints a paired p-value for each comparison against the baseline, Holm-adjusted over the table. [Quantization](../../build-an-llm/09-inference/05-quantization.md) claims "quantization costs nothing", so a paired test on per-prompt losses must not reject. [Speculative decoding](../../build-an-llm/09-inference/06-speculative-decoding.md) must not change quality, checked with McNemar on exact-match outcomes. The [run comparison and regression gate](../../build-an-llm/11-serving/12-run-comparison-and-regression-gate.md) implements the two-sample test in Go for p50 and p99 latency comparisons, bit-identical to this one. The [TinyStories capstone](../../build-an-llm/capstones/01-tinystories.md) and the [post-trained capstone](../../build-an-llm/capstones/02-post-trained.md) use these tests for A/B ablation verdicts over seeds.

### Further reading

- Wasserman, *All of Statistics*, chapter 10 (hypothesis testing and p-values).
- Good, *Permutation, Parametric, and Bootstrap Tests of Hypotheses* (2005), chapter 3.
- Phipson and Smyth, "Permutation p-values should never be zero" (2010).
- Holm, "A simple sequentially rejective multiple test procedure" (1979).
- `scipy.stats.permutation_test` (`scipy/stats/_resampling.py`): vectorized batches, exact enumeration when it is small enough, one- and two-sided alternatives.
- `statsmodels.stats.contingency_tables.mcnemar` (`statsmodels/stats/contingency_tables.py`): the same two variants, plus Cochran's Q for more than two classifiers.
- `statsmodels.stats.multitest.multipletests` (`statsmodels/stats/multitest.py`): Hochberg, Benjamini-Hochberg (false discovery rate), and others.
- lm-evaluation-harness and HELM (`stderr_for_metric` in `lm_eval/api/metrics.py`): bootstrap standard errors per task; paired comparisons across models.
- Johari et al., "Peeking at A/B Tests" (2017): sequential testing with always-valid p-values, to stop an A/B test early without inflating false alarms.

## Key terms

- **Null hypothesis**: the precise "no difference" claim a test tries to reject, here that labels are exchangeable.
- **p-value**: the probability under the null of a statistic at least as extreme as the one observed.
- **Type-I error**: rejecting a true null hypothesis; a valid test does so at rate at most $\alpha$.
- **Power**: the probability of rejecting the null when the effect is real.
- **Permutation test**: a test whose null distribution comes from relabelling the data in ways the null says are harmless.
- **Fisher-Yates shuffle**: the swap-from-the-end algorithm that produces each of the $N!$ orders with equal probability.
- **McNemar's test**: an exact binomial test on the examples two classifiers disagree on.
- **Family-wise error rate**: the probability of at least one false rejection in a family of tests.
- **Holm's method**: a step-down correction that controls the FWER and rejects at least as much as Bonferroni.

## Summary

- A p-value is the probability, if nothing really changed, of a difference at least as large as the one observed; a valid test rejects a true null at rate $\alpha$.
- A permutation test builds that "nothing changed" world by relabelling: flip which model each paired score belongs to, or reshuffle two pools; no normality assumption.
- Count the observed labelling as one more permutation: $p = (1 + \text{count})/(1 + B)$ is never 0 and keeps the type-I promise exactly.
- McNemar compares two classifiers on the same examples using only the pairs they disagree on: an exact binomial test.
- Ten comparisons at 0.05 give a false alarm 40% of the time; Holm controls the chance of any false alarm and rejects at least as much as Bonferroni.

## Review questions

1. Why $(1 + \text{count})/(1 + B)$ and not $\text{count}/B$?

   <details><summary>Answer</summary>

   Under $H_0$ the observed labelling is exchangeable with the $B$ random ones, so its statistic's rank among all $B + 1$ is uniform. The add-one p-value is exactly that rank's upper-tail fraction, giving $P(p \le \alpha) \le \alpha$ exactly. $\text{count}/B$ can be 0, which overstates the evidence (no true p-value is 0) and breaks the type-I promise for small $B$.

   </details>

2. Why compare the permuted statistic with `t_obs * (1 - 1e-9)` rather than `t_obs`?

   <details><summary>Answer</summary>

   Decimal scores tie in exact arithmetic but not always in floating point: $-0.7 + 0.2 + 0.7 + 0.1$ and $0.7 + 0.2 - 0.7 + 0.1$ can round differently, landing an ulp below $t_{\text{obs}}$. A bare $\ge$ would miss those ties and report too small a p-value. The relative margin is far larger than rounding and far smaller than any real difference.

   </details>

3. Why must Fisher-Yates pick $j$ from $\{0, \ldots, i\}$ rather than from all $N$ positions?

   <details><summary>Answer</summary>

   Picking from $\{0, \ldots, i\}$ at step $i$ gives $N!$ equally likely swap sequences, one per order. Picking from all $N$ each time gives $N^N$ sequences, and since $N^N$ is not a multiple of $N!$ for $N > 2$ they cannot map evenly onto orders: some groupings become more likely, and the test's null distribution is wrong. It also changes the uniform-to-order map the Go port reproduces.

   </details>

4. Why does `holm` `break` at the first failure instead of testing the rest?

   <details><summary>Answer</summary>

   Holm's FWER proof relies on stepping down: the bar $\alpha/(m-k)$ is only justified once the $k$ smaller p-values were rejected. A later p-value can pass its looser bar after an earlier one failed (for example $(0.03, 0.04)$ with $m = 2$: $0.03 > 0.025$ fails, $0.04 \le 0.05$ would pass), and rejecting it would exceed the promised error rate.

   </details>

5. Why does exact McNemar sum `math.comb` integers and divide once by `2**n`?

   <details><summary>Answer</summary>

   For $n$ in the hundreds, $\binom{n}{i}$ overflows float64 and $2^{-n}$ underflows; computing each term as a float loses the tail or produces inf/inf. Python integers are exact at any size, so the tail is exact and the single division rounds once to the nearest float.

   </details>
