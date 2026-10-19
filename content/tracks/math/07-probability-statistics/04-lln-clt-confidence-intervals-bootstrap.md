# LLN, CLT, confidence intervals, bootstrap

## Overview

| | |
|---|---|
| **Module** | `M07.4` · build · Python · Pass 4 · 2 to 2.5 h reading |
| **Solution** | [`course/solutions/python/tinyllm/prob/stats.py`](../../../../course/solutions/python/tinyllm/prob/stats.py): `normal_cdf`, `normal_ppf`, `t_cdf`, `t_ppf`, `standard_error`, `mean_ci`, `wilson_interval`, `quantile`, `bootstrap_ci` |
| **Contract** | [`course/contracts/py/tinyllm/prob/stats.pyi`](../../../../course/contracts/py/tinyllm/prob/stats.pyi) · the generator behind `rng`: [`spec/pcg32.md`](../../../../course/contracts/spec/pcg32.md) |
| **Needs** | no code from earlier modules · reading: [`M07.1`](01-categorical-sampling.md) one uniform per draw, [`M07.0`](00-random-variables-and-the-normal.md) expectation, variance, the normal, [`S-M07b`](91-problem-set-b.md) the variance of a sum |
| **Used by** | later: `L4.5` sequence metrics with bootstrap CIs · `L6.7` every evaluation metric carries a CI · `L8.5` quantization verdicts · `L10.7` · `ag.12` re-implements the bootstrap in Go · `M07.5` hypothesis tests build on it · later: `L12.3`, `ethics.04` · `L14.5` WER, VQA, and caption intervals · from Pass 4 (sequence models) on, every reported metric carries one of these intervals |
| **Optional depth** | Wasserman, *All of Statistics*, ch. 5 to 8 (convergence, the bootstrap); Efron and Tibshirani, *An Introduction to the Bootstrap* (1993), ch. 13; Brown, Cai, DasGupta, "Interval Estimation for a Binomial Proportion" (2001) |

## Key Takeaways

- The law of large numbers says a sample mean converges to the true mean; the central limit theorem says the error is close to normal with standard deviation $\sigma/\sqrt{n}$, so quadrupling the data halves the error bar.
- When $\sigma$ is estimated from the same $n$ values, the right quantile is Student's $t$ with $n - 1$ degrees of freedom, not the normal one; at $n = 10$ the normal quantile covers only about 92% instead of 95%.
- For a proportion near 0 or 1 the Wilson interval stays honest where the textbook $\hat p \pm z\sqrt{\hat p(1-\hat p)/n}$ collapses to a point.
- The percentile bootstrap gives an interval for any statistic, BLEU or a median included, by resampling the data; with one uniform per index it is reproducible across languages.

## How to read this chapter

Sections 1 and 2 build the law of large numbers, the CLT, the $t$ and Wilson intervals, and the bootstrap from first principles, and section 3 works the intervals by hand. Section 4 fixes the interface, and section 5 walks through the solution line by line and explains why each choice is the right one. Try to predict each step of the solution before reading it.

---

## 1. Why now

Pass 4 is the first time the course compares models. The LSTM language model (`L3.6`) reports a bits-per-character number, the seq2seq model (`L4.1`) a BLEU score (`L4.5`), and the question is never "what is the number" but "is model A better than model B, or did I get a lucky test set". A BLEU of 21.3 against 20.8 on 200 sentences means nothing until you know how much BLEU moves when you draw another 200 sentences. From here on every metric the course prints (`L4.5`, `L6.7`, the quantization verdicts of `L8.5`) carries a 95% interval, and the comparison tests of `M07.5` build on the same machinery. This module gives that interval three ways: from a formula for means, from a formula for proportions, and by resampling for everything else.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $X_1, \dots, X_n$ | independent draws from one distribution (per-example scores) | reals |
| $\mu, \sigma^2$ | the true mean and variance of that distribution | reals |
| $\bar X_n = \frac{1}{n}\sum_i X_i$ | the sample mean | real |
| $s^2 = \frac{1}{n-1}\sum_i (X_i - \bar X_n)^2$ | the sample variance (Bessel's $n - 1$) | real |
| $\mathrm{se} = s/\sqrt{n}$ | the standard error: estimated sd of $\bar X_n$ | real |
| $\alpha$ | the miss rate; $1 - \alpha$ is the confidence level (0.95) | real in $(0, 1)$ |
| $\Phi(z)$, $\Phi^{-1}(p)$ | the standard normal CDF and its quantile | functions |
| $t_\nu$, $F_\nu(t)$, $F_\nu^{-1}(p)$ | Student's t with $\nu$ degrees of freedom, its CDF and quantile | functions |
| $k, n, \hat p = k/n$ | successes, trials, observed proportion | integers, real |
| $B$, $\theta^*_b$ | number of bootstrap resamples, the statistic on resample $b$ | integer, reals |

**The law of large numbers.** By linearity $E[\bar X_n] = \mu$, and because independent variances add, $\operatorname{Var}(\bar X_n) = n\sigma^2/n^2 = \sigma^2/n$ (`S-M07b` q3 with all covariances 0). Chebyshev's inequality then gives $P(|\bar X_n - \mu| \ge \varepsilon) \le \sigma^2/(n\varepsilon^2) \to 0$: the mean of more data is closer to the truth (proved in [`S-M07c`](92-problem-set-c.md) q5). It also gives the rate: the typical error is $\sigma/\sqrt{n}$, so four times the eval set halves the error, it does not quarter it.

**The central limit theorem.** Whatever the distribution of each $X_i$ (finite variance is all it needs), the standardized mean $\sqrt{n}(\bar X_n - \mu)/\sigma$ tends to the standard normal $\mathcal{N}(0, 1)$. So $P(|\bar X_n - \mu| \le z\,\sigma/\sqrt{n}) \approx 2\Phi(z) - 1$, and choosing $z = \Phi^{-1}(1 - \alpha/2)$ (1.96 for 95%) turns that into an interval: $\bar X_n \pm z\,\sigma/\sqrt{n}$ covers $\mu$ in about $1 - \alpha$ of repeated samples. Per-example eval scores are never normal (0/1 accuracy, skewed losses), and the interval still works once $n$ is in the hundreds: that is the CLT.

**Quantiles from first principles.** $\Phi(z) = \frac{1}{2}\operatorname{erfc}(-z/\sqrt{2})$, with `math.erfc` (it keeps full relative precision in the lower tail, where $1 - \Phi$ computed by subtraction would lose every digit). The quantile $\Phi^{-1}(p)$ has no closed form, but $\Phi$ is increasing, so **bisection** finds it: keep a bracket $[l, h]$ with $\Phi(l) < p \le \Phi(h)$ and halve it until the midpoint equals an end, which is the last bit of a double. For $p > 1/2$ use the symmetry $\Phi^{-1}(p) = -\Phi^{-1}(1 - p)$; $1 - p$ is exact in float64 there.

**Student's t.** The interval above needs $\sigma$, and you only have $s$, computed from the same data. The ratio $T = (\bar X_n - \mu)/(s/\sqrt{n})$ is wider-tailed than the normal (when $s$ happens to be small, $T$ is large), and for normal data it follows Student's t with $\nu = n - 1$ degrees of freedom exactly (Gosset, 1908). So the interval is

$$\bar X_n \pm F_{n-1}^{-1}(1 - \alpha/2)\,\frac{s}{\sqrt{n}}.$$

Two details carry it: the $n - 1$ in $s^2$ (the deviations are measured from $\bar X_n$, which sits closer to the data than $\mu$ does, so dividing by $n$ underestimates $\sigma^2$), and $\alpha/2$ (the interval misses on both sides). For integer $\nu$ the CDF has a closed form (Abramowitz and Stegun 26.7.3 and 26.7.4). With $\theta = \arctan(|t|/\sqrt{\nu})$ and $A = P(|T| < |t|)$:

- $\nu$ even: $A = \sin\theta\,\bigl(1 + \tfrac{1}{2}\cos^2\theta + \tfrac{1 \cdot 3}{2 \cdot 4}\cos^4\theta + \dots + \tfrac{1 \cdot 3 \cdots (\nu-3)}{2 \cdot 4 \cdots (\nu-2)}\cos^{\nu-2}\theta\bigr)$,
- $\nu$ odd: $A = \tfrac{2}{\pi}\bigl(\theta + \sin\theta\,(\cos\theta + \tfrac{2}{3}\cos^3\theta + \dots + \tfrac{2 \cdot 4 \cdots (\nu-3)}{1 \cdot 3 \cdots (\nu-2)}\cos^{\nu-2}\theta)\bigr)$, just $2\theta/\pi$ for $\nu = 1$,

and $F_\nu(t) = \tfrac{1}{2} + \tfrac{1}{2}\operatorname{sign}(t)\,A$. Each term is the previous one times a ratio ($\tfrac{2k-1}{2k}\cos^2\theta$ or $\tfrac{2k}{2k+1}\cos^2\theta$), so a cumulative product computes the series in $O(\nu)$. The quantile is bisection again, on a bracket $[0, b]$ whose top doubles until $F_\nu(b) \ge p$ ($\nu = 1$ needs $t = 636.6$ at $p = 0.9995$, so no fixed bound is safe). As $\nu$ grows, $t_\nu$ tends to the normal: $F_{1000}^{-1}(0.975) = 1.9623$ against $1.9600$.

**Proportions: the Wilson interval.** An accuracy is a mean of 0/1 values, but the formula $\hat p \pm z\sqrt{\hat p(1-\hat p)/n}$ (the Wald interval) fails exactly where safety evals live: with 0 failures in 10 trials it says $[0, 0]$, certainty. Wilson (1927) inverts the test instead: the interval is every $p$ with $|\hat p - p| \le z\sqrt{p(1-p)/n}$. Squaring and solving the quadratic in $p$ gives

$$\frac{\hat p + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}},$$

whose center is pulled toward $1/2$ and which is never empty. At $k = 0$ its lower end is exactly 0 and at $k = n$ its upper end exactly 1; rounding can leave them a hair off, so the code pins them.

**The bootstrap.** For a statistic with no variance formula (BLEU, chrF, a median, pass@k), Efron's idea (1979) is to let the data stand in for the population: draw $n$ indices with replacement, recompute the statistic on that resample, repeat $B$ times, and read the spread of $\theta^*_1, \dots, \theta^*_B$. The **percentile interval** is their $\alpha/2$ and $1 - \alpha/2$ quantiles. Two rules make it reproducible across languages (`ag.12` re-implements it in Go): each index is $i = \min(\lfloor u n \rfloor, n - 1)$ from exactly one `rng.uniform()`, drawn in order (resample 0's indices first); and the quantile is **type 7**: sort, $h = (B-1)q$, $i = \lfloor h \rfloor$, value $s_i + (h - i)(s_{i+1} - s_i)$ (numpy's default). The point estimate stays $\theta(x)$ on the original data.

## 3. Worked example by hand

**A t interval.** $x = [2, 4, 4, 5, 7, 8]$, $n = 6$.

1. Mean: $30/6 = 5$.
2. Deviations: $-3, -1, -1, 0, 2, 3$; squares sum to $9 + 1 + 1 + 0 + 4 + 9 = 24$.
3. $s^2 = 24/(6 - 1) = 4.8$, $s = 2.19089$; $\mathrm{se} = \sqrt{4.8/6} = \sqrt{0.8} = 0.894427$.
4. $\nu = 5$, $1 - \alpha/2 = 0.975$: $F_5^{-1}(0.975) = 2.570582$ (bisection on the odd series, which for $\nu = 5$ is $\theta + \sin\theta(\cos\theta + \frac{2}{3}\cos^3\theta)$).
5. Half-width $2.570582 \times 0.894427 = 2.299198$; interval $[2.700802, 7.299198]$.

With the normal 1.959964 instead, the interval would be $[3.246955, 6.753045]$, a quarter narrower: at $n = 6$ that is a large overstatement of certainty.

**One value of the series.** $F_3(1)$: $\theta = \arctan(1/\sqrt{3}) = \pi/6$, $\sin\theta = 1/2$, $\cos\theta = \sqrt{3}/2$. The odd series with $\nu = 3$ stops after $\cos\theta$: $A = \frac{2}{\pi}(\frac{\pi}{6} + \frac{1}{2}\cdot\frac{\sqrt{3}}{2}) = \frac{1}{3} + \frac{\sqrt{3}}{2\pi} = 0.608998$, so $F_3(1) = \frac{1}{2} + \frac{1}{2}A = 0.804499$.

**A Wilson interval.** $k = 0$, $n = 10$, $z = 1.959964$, $z^2 = 3.841459$: $1 + z^2/n = 1.384146$, center $= (0 + 0.192073)/1.384146 = 0.138766$, half-width $= 1.959964 \times \sqrt{0 + 3.841459/400}/1.384146 = 0.138766$. The interval is $[0, 0.277533]$; the Wald interval is $[0, 0]$.

These are the first two cases in section 4.

## 4. The interface

```python
# python/tinyllm/prob/stats.py
def normal_cdf(z: float) -> float
def normal_ppf(p: float) -> float                          # bisection, symmetric for p > 1/2
def t_cdf(t: float, df: int) -> float                      # A&S 26.7.3 / 26.7.4
def t_ppf(p: float, df: int) -> float                      # bracket doubled, then bisection
def standard_error(x) -> float                             # s / sqrt(n), divisor n - 1
def mean_ci(x, alpha: float = 0.05) -> tuple[float, float, float]           # (mean, lo, hi)
def wilson_interval(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]
def quantile(x, q: float) -> float                         # type 7
def bootstrap_ci(x, stat, n_boot: int, alpha: float, rng) -> tuple[float, float, float]   # (stat(x), lo, hi)
```

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the $t$ interval by hand | unit | section 3: se $= \sqrt{0.8}$, interval $[2.700802, 7.299198]$, $F_3(1) = 0.804499$ | everyone agrees on Bessel, $\alpha/2$, and $t$ |
| the Wilson interval by hand | unit | section 3: $[0, 0.277533]$ at $k = 0$ | safety failure rates (`ethics.04`) |
| the normal against scipy | golden | $\Phi$ and $\Phi^{-1}$ to 1e-11 relative, tails to $p = 10^{-12}$ | every $z$ in the course |
| Student's $t$ against scipy | golden | $F_\nu$ and $F_\nu^{-1}$ for 13 values of $\nu$ from 1 to 1000 | small eval sets get the right width |
| `mean_ci` against scipy | golden | `standard_error` and `mean_ci` equal `scipy.stats.sem` and `t.interval` | `L6.7` reports |
| Wilson against scipy | golden | scipy's Wilson interval at the extremes and in the middle | accuracy CIs |
| type 7 quantiles | golden | equal to numpy's default | Go's bootstrap (`ag.12`) agrees |
| an independent bootstrap | golden | same seed, same interval as an independent implementation, mean and median | reproducible CIs for BLEU (`L4.5`) |
| coverage at small $n$ | statistical | 2000 samples of $n = 10$: coverage in $[0.93, 0.97]$ | what "95%" promises |
| coverage on skewed data | statistical | exponential data, $n = 200$: coverage in $[0.93, 0.97]$ | the CLT on non-normal scores |
| the $\sqrt{n}$ law | property | four copies of the data halve the standard error | sizing eval sets |
| quantile consistency | property | $F(F^{-1}(p)) = p$, symmetry, $t_\nu \to$ normal | the quantiles are consistent |
| $t$ closed forms | unit | $\nu = 1$ (Cauchy) and $\nu = 2$ closed forms | both series are right at their shortest |
| bootstrap draws and determinism | unit | $B n$ uniforms; same seed same result; point is $\theta(x)$ | cross-language reproducibility |
| the bootstrap index rule | boundary | $u = 0.74$, $n = 4$ is index 2, $u \to 1$ is the last index | the floor rule |
| shift equivariance | property | $x + c$ moves the mean's interval by $c$ | resampling does not look at values |
| quantile edges | boundary | $q = 0$, $q = 1$, one value, unsorted input | no index past the end |
| Wilson inside $[0, 1]$ | boundary | exact 0 and 1 at the ends; $\hat p$ inside | intervals stay probabilities |
| bad arguments | boundary | $n < 2$, $\alpha \notin (0,1)$, $k > n$, NaN, empty input | errors at the call site |

## 5. The solution, step by step

### From first principles

1. **Every quantile is the inverse of an increasing CDF, so one bisection serves them all.** Keep $f(\mathit{lo}) < \mathit{target} \le f(\mathit{hi})$ and halve until the midpoint equals an end: that is the last bit of a double, about 60 halvings, with no tolerance to choose.
2. **$\Phi$ from `erfc`, and the upper half by symmetry.** $\Phi(z) = \tfrac12\operatorname{erfc}(-z/\sqrt2)$ keeps relative precision in the lower tail. For $p > \tfrac12$, $1 - p$ is exact (Sterbenz), so $\Phi^{-1}(p) = -\Phi^{-1}(1-p)$ never bisects in the region where $\Phi$ is flat near 1.
3. **The $t$ CDF from the finite series.** For integer $\nu$, A&S 26.7.3/4 give $P(|T| < |t|)$ in $O(\nu)$ terms, each the previous times a ratio, so a cumulative product computes them. The quantile needs a bracket that grows with the tail: double `hi` until $F_\nu(\mathit{hi}) \ge p$, then bisect.
4. **The standard error uses Bessel and $\sqrt n$.** $s^2 = \sum (x_i - \bar x)^2/(n-1)$, then $s/\sqrt n$. Two passes, as in `M07.0`, so a large mean does not cancel.
5. **The mean interval is $\bar x \pm F^{-1}_{n-1}(1 - \alpha/2)\cdot\mathrm{se}$.** Both the $t$ (estimated $\sigma$) and the $\alpha/2$ (two-sided) are required for the promised coverage.
6. **Wilson is the closed form of an inverted test, with pinned ends.** Compute center and half-width from section 2; set the lower end to exactly 0 when $k = 0$ and the upper to exactly 1 when $k = n$, and clamp otherwise.
7. **The bootstrap is reproducible only with a fixed draw rule.** Index $\min(\lfloor un \rfloor, n-1)$ from one uniform, resample by resample, row by row; the interval is the type 7 quantiles of the replicates; the point estimate is the statistic on the original data.

### The code, annotated

```python
def _bisect(f: Callable[[float], float], target: float, lo: float, hi: float) -> float:
    """The x in [lo, hi] where the increasing f crosses target, to the last bit:
    keeps f(lo) < target <= f(hi) and halves until the midpoint is an end."""
    for _ in range(2000):                 # a safety cap; ~60-1100 halvings reach adjacent doubles
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:        # lo and hi are neighbouring floats: done
            break
        if f(mid) < target:
            lo = mid
        else:
            hi = mid
    return hi                             # the smallest x found with f(x) >= target


def normal_cdf(z: float) -> float:
    return 0.5 * math.erfc(-float(z) / math.sqrt(2.0))   # no 1 - (...) subtraction in the lower tail


def normal_ppf(p: float) -> float:
    p = float(p)
    if not 0.0 < p < 1.0:
        raise ValueError(f"normal_ppf needs 0 < p < 1, got {p!r}")
    if p == 0.5:
        return 0.0
    if p > 0.5:
        # 1 - p is exact for p in [0.5, 1) (Sterbenz), so the upper tail keeps
        # its precision by symmetry.
        return -normal_ppf(1.0 - p)
    return _bisect(normal_cdf, p, -40.0, 0.0)   # Phi(-40) underflows to 0, below any p > 0
```

The loop stops on floats, not on a tolerance, so the answer is as close as float64 allows whatever the scale of $x$; the 2000 cap only guards against a non-terminating `f`.

```python
def _t_abs_prob(t: float, df: int) -> float:
    """A = P(|T| < |t|) by Abramowitz and Stegun 26.7.3 (odd df) and 26.7.4 (even df)."""
    theta = math.atan(abs(t) / math.sqrt(df))
    s, c = math.sin(theta), math.cos(theta)
    c2 = c * c
    if df % 2 == 0:
        # 1 + (1/2) c^2 + (1 3)/(2 4) c^4 + ... : k-th ratio (2k - 1) / (2k) c^2
        m = (df - 2) // 2
        if m == 0:
            return s                                   # df = 2: A = sin(theta)
        k = np.arange(1, m + 1, dtype=np.float64)
        terms = np.cumprod((2.0 * k - 1.0) / (2.0 * k) * c2)
        return s * (1.0 + float(np.sum(terms)))
    if df == 1:
        return 2.0 * theta / math.pi                   # Cauchy
    # c + (2/3) c^3 + (2 4)/(3 5) c^5 + ... : k-th ratio (2k) / (2k + 1) c^2
    m = (df - 3) // 2
    k = np.arange(1, m + 1, dtype=np.float64)
    terms = c * np.cumprod((2.0 * k) / (2.0 * k + 1.0) * c2) if m else np.zeros(0)
    return 2.0 / math.pi * (theta + s * (c + float(np.sum(terms))))


def t_cdf(t: float, df: int) -> float:
    df = _check_df(df)                                 # an integer >= 1
    t = float(t)
    if math.isnan(t):
        raise ValueError("t_cdf of NaN")
    if math.isinf(t):
        return 1.0 if t > 0 else 0.0
    a = _t_abs_prob(t, df)
    return 0.5 + 0.5 * a if t >= 0 else 0.5 - 0.5 * a  # F = 1/2 + sign(t) A / 2


def t_ppf(p: float, df: int) -> float:
    # (argument checks elided)
    if p == 0.5:
        return 0.0
    if p < 0.5:
        return -t_ppf(1.0 - p, df)                     # symmetry
    hi = 1.0
    while t_cdf(hi, df) < p:                           # no fixed bound is safe: nu = 1 needs 636.6 at 0.9995
        hi *= 2.0
    return _bisect(lambda v: t_cdf(v, df), p, 0.0, hi)
```

Section 3's $F_3(1)$: $\theta = \pi/6$, `m = 0`, so `terms` is empty and the odd branch returns $\frac{2}{\pi}(\frac{\pi}{6} + \frac12 \cdot \frac{\sqrt3}{2}) = 0.608998$; `t_cdf` gives $0.804499$.

```python
def standard_error(x: ArrayLike) -> float:
    a = _sample(x, 2)                       # 1-D, finite, at least 2 values
    n = a.size
    mean = math.fsum(a.tolist()) / n
    ss = math.fsum(((a - mean) ** 2).tolist())   # two passes: centre, then square
    # Bessel: the deviations are measured from the sample mean, which sits
    # closer to the data than mu does, so divide by n - 1, not n.
    return math.sqrt(ss / (n - 1)) / math.sqrt(n)


def mean_ci(x: ArrayLike, alpha: float = 0.05) -> tuple[float, float, float]:
    a = _sample(x, 2)
    alpha = _check_alpha(alpha)
    mean = math.fsum(a.tolist()) / a.size
    half = t_ppf(1.0 - alpha / 2.0, a.size - 1) * standard_error(a)   # t with n - 1 df, two-sided
    return mean, mean - half, mean + half
```

On $[2, 4, 4, 5, 7, 8]$: `ss = 24`, $\mathrm{se} = \sqrt{4.8}/\sqrt6 = \sqrt{0.8}$, `t_ppf(0.975, 5) = 2.570582`, half-width $2.299198$.

```python
def wilson_interval(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    # (integer and range checks elided: 0 <= k <= n, n >= 1)
    z = normal_ppf(1.0 - _check_alpha(alpha) / 2.0)
    z2 = z * z
    phat = k / n
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / denom            # pulled toward 1/2
    half = z * math.sqrt(phat * (1.0 - phat) / n + z2 / (4.0 * n * n)) / denom   # never 0
    # In exact arithmetic the interval ends at 0 when k = 0 and at 1 when
    # k = n; rounding can leave it a hair inside or outside, so pin the ends.
    lo = 0.0 if k == 0 else max(0.0, center - half)
    hi = 1.0 if k == n else min(1.0, center + half)
    return lo, hi
```

```python
def quantile(x: ArrayLike, q: float) -> float:
    # (empty, NaN, and q range checks elided)
    s = np.sort(a)
    h = (s.size - 1) * q                    # type 7: position on 0 .. n - 1
    i = int(math.floor(h))
    if i >= s.size - 1:
        return float(s[-1])                 # q = 1 (or one value): no s[i + 1] to read
    return float(s[i] + (h - i) * (s[i + 1] - s[i]))   # linear interpolation


def bootstrap_ci(x, stat, n_boot: int, alpha: float, rng) -> tuple[float, float, float]:
    a = _sample(x, 1)
    alpha = _check_alpha(alpha)
    # (n_boot check elided: an integer >= 1)
    n = a.size
    thetas = np.empty(int(n_boot), dtype=np.float64)
    for b in range(int(n_boot)):
        idx = np.empty(n, dtype=np.int64)   # a fresh resample every replicate
        for i in range(n):
            # One uniform per index, in order: the same seed resamples the same
            # rows in Python and in Go (ag.12).
            idx[i] = min(int(rng.uniform() * n), n - 1)   # floor, clamped against rounding
        thetas[b] = float(stat(a[idx]))
    return (
        float(stat(a)),                     # the point estimate is on the original data
        quantile(thetas, alpha / 2.0),
        quantile(thetas, 1.0 - alpha / 2.0),
    )
```

The index rule on the boundary case: $u = 0.74$, $n = 4$ gives $\lfloor 2.96 \rfloor = 2$; rounding would give 3. Rounding gives index 0 only the uniforms below $0.125$ and pushes the top eighth onto the last index, so the rows stop being equally likely; the floor gives each row exactly a quarter of $[0, 1)$.

Full solution: [`stats.py`](../../../../course/solutions/python/tinyllm/prob/stats.py).

### Why this solution

- **Cost.** $\Phi^{-1}$ is about 60 to 1100 `erfc` calls (the bracket starts at $-40$ and halves to adjacent doubles); $F_\nu^{-1}$ is $O(\log \mathit{hi})$ doublings plus a bisection, each step $O(\nu)$. The bootstrap is $B \cdot n$ uniforms plus $B$ calls to `stat`, and the quantiles sort $B$ values: $O(Bn + B\log B)$ plus the statistic's cost.
- **Bisection over rational approximations.** scipy's Cephes uses rational approximations with Newton polishing, which is faster but hard to verify and to port. Bisection on a correct CDF is obviously correct, ends at the last bit, and gives the Go port a definition it can reproduce exactly.
- **$t$ over normal; Wilson over Wald.** Both alternatives are simpler and both fail where evaluation lives: small $n$ (the normal covers 92% at $n = 10$) and proportions near 0 or 1 (Wald says $[0, 0]$). Wilson costs one more square root.
- **Percentile bootstrap over BCa.** BCa corrects bias and skew and has better coverage for skewed statistics, but needs a jackknife pass and more code to port. The percentile interval is what the Go harness reimplements and is adequate for metric reporting.
- **Tradeoffs accepted.** The scalar Python loops are slow for large $B n$ (a few seconds at $B = 1000$, $n = 1000$); vectorized resampling would draw the same uniforms but obscure the one-uniform-per-index rule that makes cross-language reproduction possible.

### Explain it back

**1. Why does `mean_ci` use `t_ppf(1 - alpha/2, n - 1)` rather than `normal_ppf(1 - alpha/2)`?**

<details><summary>Answer</summary>

The standard error uses $s$, estimated from the same $n$ values. When $s$ is small by chance, $(\bar X - \mu)/(s/\sqrt n)$ is large, so this ratio has heavier tails than the normal: Student's $t$ with $n - 1$ degrees of freedom. The normal quantile ignores that and covers only about 92% at $n = 10$. As $n$ grows the two agree.

</details>

**2. Why does `normal_ppf` recurse on $1 - p$ for $p > \tfrac12$ instead of bisecting on $[0, 40]$?**

<details><summary>Answer</summary>

Near the top $\Phi(z)$ is within $10^{-12}$ of 1, where float64 values are spaced $1.1 \times 10^{-16}$ apart, so bisecting against a target like $1 - 10^{-12}$ compares nearly equal numbers and loses digits. $1 - p$ is computed exactly for $p \ge \tfrac12$, and the lower tail of $\Phi$ via `erfc` keeps full relative precision, so the symmetric problem is solved accurately.

</details>

**3. Why does `t_ppf` double `hi` in a loop instead of starting from a fixed bracket like $[0, 100]$?**

<details><summary>Answer</summary>

The heavy tails of small $\nu$ put the quantile far out: for $\nu = 1$ (Cauchy), $F_1^{-1}(0.9995) = 636.6$, and more extreme $p$ go further. Any fixed bound fails for some valid input. Doubling finds a valid upper end in $O(\log t)$ steps for any $\nu$ and $p$.

</details>

**4. Why pin `lo = 0.0` when $k = 0$ in the Wilson interval, when the formula already gives 0?**

<details><summary>Answer</summary>

It gives 0 in exact arithmetic: center and half-width are equal at $k = 0$. In floating point they are computed by different operations and can differ in the last bit, leaving `lo` at $\pm 10^{-17}$: a negative probability or an interval that excludes the observed $\hat p = 0$. Pinning restores the exact property.

</details>

**5. What goes wrong if `bootstrap_ci` draws the index array once and reuses it for every replicate?**

<details><summary>Answer</summary>

Every replicate is the same resample, so all $\theta^*_b$ are equal and the interval has zero width: the method reports certainty. The bootstrap's spread comes from many independent resamples; each must draw $n$ fresh uniforms, $Bn$ in total.

</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. the normal quantile with an estimated $\sigma$ | at $n = 10$ the "95%" interval covers 92% | `t_ppf(..., a.size - 1)` in `mean_ci`; step 5 |
| 2. dividing by $n$ instead of $n - 1$, or by $n$ instead of $\sqrt{n}$ | intervals too narrow, by a little or by a lot | `math.sqrt(ss / (n - 1)) / math.sqrt(n)`; step 4 |
| 3. the quantile at $1 - \alpha$ instead of $1 - \alpha/2$ | a 90% interval labelled 95% | `1.0 - alpha / 2.0` in `mean_ci` and `wilson_interval`; step 5 |
| 4. the Wald interval for a proportion, or unpinned ends | $[0, 0]$ after zero failures; ends a hair outside $[0, 1]$ | Wilson's closed form with `lo = 0.0 if k == 0`, `hi = 1.0 if k == n`; step 6 |
| 5. rounding $u n$ to pick a bootstrap index | the last row drawn more often than the first; Go and Python disagree | `min(int(rng.uniform() * n), n - 1)`; step 7 |
| 6. drawing one resample and reusing it | every replicate identical: a zero-width interval | a new `idx` inside the replicate loop; step 7 |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M07.1` | one uniform per draw, the same discipline as the bootstrap's index rule |
| Back | `M07.0` | expectation, variance, the normal distribution |
| Back | `S-M07b` | the variance of a sum, which is why the standard error is $\sigma/\sqrt{n}$ |
| Forward | `S-M07c` | the LLN proof, the CLT approximation, and these intervals by hand |
| Forward | `L4.5` | BLEU and chrF with bootstrap intervals over sentences |
| Forward | `L6.7` | every evaluation metric carries `mean_ci` or `bootstrap_ci` |
| Forward | `L8.5` | quantization verdicts: is the perplexity change inside the noise |
| Forward | `M07.5` | permutation tests and McNemar build on the same resampling |
| Forward | `ag.12` | the Go eval harness re-implements `bootstrap_ci` from the same seed |
| Forward | `L12.3` | a later call site of these intervals |
| Forward | `ethics.04` | safety failure rates with Wilson intervals |
| Forward | `L14.5` | WER, VQA, caption, and zero-shot intervals resample items with `bootstrap_ci` |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `t_ppf`, `normal_ppf` | scipy's `stdtrit` and `ndtri` (Cephes) | rational approximations plus Newton steps, $O(1)$ for any $\nu$ | `scipy/special/cephes/stdtr.c`, `ndtri.c` |
| `bootstrap_ci` (percentile) | `scipy.stats.bootstrap` (BCa) | bias-corrected and accelerated intervals, vectorized resampling | `scipy/stats/_resampling.py` |
| `wilson_interval` | `statsmodels.stats.proportion.proportion_confint` | Agresti-Coull, Jeffreys, Clopper-Pearson side by side | `statsmodels/stats/proportion.py` |
| interval on eval scores | HELM and lm-evaluation-harness stderr | clustered standard errors when examples share a prompt | `lm_eval/api/metrics.py` (`bootstrap_stderr`) |
