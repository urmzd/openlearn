<!-- ol:module M07.4 -->
# LLN, CLT, confidence intervals, bootstrap

## Overview

| | |
|---|---|
| **Module** | `M07.4` · build · Python · Pass 4 · 4 to 5 h |
| **You build** | `python/tinyllm/prob/stats.py`: `normal_cdf`, `normal_ppf`, `t_cdf`, `t_ppf`, `standard_error`, `mean_ci`, `wilson_interval`, `quantile`, `bootstrap_ci` |
| **Contract** | [`course/contracts/py/tinyllm/prob/stats.pyi`](../../course/contracts/py/tinyllm/prob/stats.pyi) · the generator behind `rng`: [`spec/pcg32.md`](../../course/contracts/spec/pcg32.md) |
| **Tests** | `course/tests/M07.4/` (what they check: section 4), golden values from scipy 1.17.1 in `course/fixtures/M07.4/scipy_golden.json` |
| **Needs** | no code from earlier modules · reading: [`M07.1`](01-categorical-sampling.md) one uniform per draw, [`M07.0`](00-random-variables-and-the-normal.md) expectation, variance, the normal, [`S-M07b`](91-problem-set-b.md) the variance of a sum |
| **Used by** | later: `L4.5` sequence metrics with bootstrap CIs · `L6.7` every evaluation metric carries a CI · `L8.5` quantization verdicts · `L10.7` · `ag.12` re-implements the bootstrap in Go · `M07.5` hypothesis tests build on it · later: `L12.3`, `ethics.04` |
| **Milestone** | `MS-P4` (sequence models; the pass gate runs `ol check` on every math module of the pass) |
| **Optional depth** | Wasserman, *All of Statistics*, ch. 5 to 8 (convergence, the bootstrap); Efron and Tibshirani, *An Introduction to the Bootstrap* (1993), ch. 13; Brown, Cai, DasGupta, "Interval Estimation for a Binomial Proportion" (2001) |

## Key Takeaways

- The law of large numbers says a sample mean converges to the true mean; the central limit theorem says the error is close to normal with standard deviation $\sigma/\sqrt{n}$, so quadrupling the data halves the error bar (`test_standard_error_shrinks_like_root_n`, `test_mean_ci_coverage_skewed_data_by_clt`).
- When $\sigma$ is estimated from the same $n$ values, the right quantile is Student's $t$ with $n - 1$ degrees of freedom, not the normal one; at $n = 10$ the normal quantile covers only about 92% instead of 95% (`test_hand_example_mean_ci`, `test_mean_ci_coverage_normal_small_n`).
- For a proportion near 0 or 1 the Wilson interval stays honest where the textbook $\hat p \pm z\sqrt{\hat p(1-\hat p)/n}$ collapses to a point (`test_hand_example_wilson`, `test_wilson_matches_scipy`).
- The percentile bootstrap gives an interval for any statistic, BLEU or a median included, by resampling the data; with one uniform per index it is reproducible across languages (`test_bootstrap_matches_independent_resampling`, `test_bootstrap_draws_and_determinism`).

## How to work this chapter

```bash
ol start M07.4              # stubs stats.py into your repo
ol tests M07.4              # read the test catalog first
ol check M07.4              # exit code is the verdict
ol diff  M07.4              # after passing: your code against the reference
```

---

## 1. Why now

Pass 4 is the first time you compare models. Your LSTM language model (`L3.6`) will report a bits-per-character number, your seq2seq model (`L4.1`) a BLEU score (`L4.5`), and the question is never "what is the number" but "is model A better than model B, or did I get a lucky test set". A BLEU of 21.3 against 20.8 on 200 sentences means nothing until you know how much BLEU moves when you draw another 200 sentences. From here on every metric the course prints (`L4.5`, `L6.7`, the quantization verdicts of `L8.5`) carries a 95% interval, and the comparison tests of `M07.5` build on the same machinery. This module gives you that interval three ways: from a formula for means, from a formula for proportions, and by resampling for everything else.

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

**The law of large numbers.** By linearity $E[\bar X_n] = \mu$, and because independent variances add, $\operatorname{Var}(\bar X_n) = n\sigma^2/n^2 = \sigma^2/n$ (`S-M07b` q3 with all covariances 0). Chebyshev's inequality then gives $P(|\bar X_n - \mu| \ge \varepsilon) \le \sigma^2/(n\varepsilon^2) \to 0$: the mean of more data is closer to the truth (you prove this in `S-M07c` q5). It also gives the rate: the typical error is $\sigma/\sqrt{n}$, so four times the eval set halves the error, it does not quarter it.

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

These are `test_hand_example_mean_ci` and `test_hand_example_wilson`.

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

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_mean_ci` | unit | section 3: se $= \sqrt{0.8}$, interval $[2.700802, 7.299198]$, $F_3(1) = 0.804499$ | you and the test agree on Bessel, $\alpha/2$, and $t$ |
| `test_hand_example_wilson` | unit | section 3: $[0, 0.277533]$ at $k = 0$ | safety failure rates (`ethics.04`) |
| `test_normal_matches_scipy` | golden | $\Phi$ and $\Phi^{-1}$ to 1e-11 relative, tails to $p = 10^{-12}$ | every $z$ in the course |
| `test_t_matches_scipy` | golden | $F_\nu$ and $F_\nu^{-1}$ for 13 values of $\nu$ from 1 to 1000 | small eval sets get the right width |
| `test_mean_ci_matches_scipy` | golden | `standard_error` and `mean_ci` equal `scipy.stats.sem` and `t.interval` | `L6.7` reports |
| `test_wilson_matches_scipy` | golden | scipy's Wilson interval at the extremes and in the middle | accuracy CIs |
| `test_quantile_matches_numpy_type7` | golden | type 7 quantiles | Go's bootstrap (`ag.12`) agrees |
| `test_bootstrap_matches_independent_resampling` | golden | same seed, same interval as an independent implementation, mean and median | reproducible CIs for BLEU (`L4.5`) |
| `test_mean_ci_coverage_normal_small_n` | statistical | 2000 samples of $n = 10$: coverage in $[0.93, 0.97]$ | what "95%" promises |
| `test_mean_ci_coverage_skewed_data_by_clt` | statistical | exponential data, $n = 200$: coverage in $[0.93, 0.97]$ | the CLT on non-normal scores |
| `test_standard_error_shrinks_like_root_n` | property | four copies of the data halve the standard error | sizing eval sets |
| `test_ppf_inverts_cdf_and_t_tends_to_normal` | property | $F(F^{-1}(p)) = p$, symmetry, $t_\nu \to$ normal | the quantiles are consistent |
| `test_t_closed_forms` | unit | $\nu = 1$ (Cauchy) and $\nu = 2$ closed forms | both series are right at their shortest |
| `test_bootstrap_draws_and_determinism` | unit | $B n$ uniforms; same seed same result; point is $\theta(x)$ | cross-language reproducibility |
| `test_bootstrap_index_rule` | boundary | $u = 0.74$, $n = 4$ is index 2, $u \to 1$ is the last index | the floor rule |
| `test_bootstrap_shift_equivariance` | property | $x + c$ moves the mean's interval by $c$ | resampling does not look at values |
| `test_quantile_edges` | boundary | $q = 0$, $q = 1$, one value, unsorted input | no index past the end |
| `test_wilson_stays_inside_zero_one` | boundary | exact 0 and 1 at the ends; $\hat p$ inside | intervals stay probabilities |
| `test_rejects_bad_arguments` | boundary | $n < 2$, $\alpha \notin (0,1)$, $k > n$, NaN, empty input | errors at the call site |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. the normal quantile with an estimated $\sigma$ | at $n = 10$ the "95%" interval covers 92% | `test_hand_example_mean_ci`, `test_mean_ci_coverage_normal_small_n` (mutant `s01`) |
| 2. dividing by $n$ instead of $n - 1$, or by $n$ instead of $\sqrt{n}$ | intervals too narrow, by a little or by a lot | `test_hand_example_mean_ci`, `test_standard_error_shrinks_like_root_n` (mutants `s02`, `s11`) |
| 3. the quantile at $1 - \alpha$ instead of $1 - \alpha/2$ | a 90% interval labelled 95% | `test_hand_example_mean_ci` (mutant `s03`) |
| 4. the Wald interval for a proportion, or unpinned ends | $[0, 0]$ after zero failures; ends a hair outside $[0, 1]$ | `test_hand_example_wilson`, `test_wilson_stays_inside_zero_one` (mutants `s04`, `s13`) |
| 5. rounding $u n$ to pick a bootstrap index | the last row drawn twice as often as the first; Go and Python disagree | `test_bootstrap_index_rule` (mutant `s05`) |
| 6. drawing one resample and reusing it | every replicate identical: a zero-width interval | `test_bootstrap_matches_independent_resampling` (mutant `s06`) |

## 6. Where it's used next
| Forward | `L12.3` | Registered module relationship. |
| Forward | `ethics.04` | Registered call site uses this module. |

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

If you skip this module, `ol check L6.7` stops with `L6.7 needs M07.4`: build it, or rerun with `--ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `t_ppf`, `normal_ppf` | scipy's `stdtrit` and `ndtri` (Cephes) | rational approximations plus Newton steps, $O(1)$ for any $\nu$ | `scipy/special/cephes/stdtr.c`, `ndtri.c` |
| `bootstrap_ci` (percentile) | `scipy.stats.bootstrap` (BCa) | bias-corrected and accelerated intervals, vectorized resampling | `scipy/stats/_resampling.py` |
| `wilson_interval` | `statsmodels.stats.proportion.proportion_confint` | Agresti-Coull, Jeffreys, Clopper-Pearson side by side | `statsmodels/stats/proportion.py` |
| interval on eval scores | HELM and lm-evaluation-harness stderr | clustered standard errors when examples share a prompt | `lm_eval/api/metrics.py` (`bootstrap_stderr`) |
