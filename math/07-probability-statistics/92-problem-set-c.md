<!-- ol:module S-M07c -->
# Solve set: LLN/CLT, confidence intervals

## Overview

| | |
|---|---|
| **Module** | `S-M07c` · solve · none · Pass 4 · 2 to 3 h |
| **You build** | answers in `solve/S-M07c.toml` (9 checked by SymPy) and 1 proof in `solve/S-M07c/q5.md` (self-graded against its rubric) |
| **Contract** | none: a pen and paper set |
| **Tests** | `course/solve/S-M07c/key.toml` (hidden): typed answers plus reject canaries; the problems are in `course/solve/S-M07c/problems.md` and in section 4 |
| **Needs** | `S-M07b` (the variance of a sum). Reading: [`M07.4` LLN, CLT, confidence intervals, bootstrap](04-lln-clt-confidence-intervals-bootstrap.md) and the [Probability and Statistics topic](README.md), limit theorems and estimation sections |
| **Used by** | no call site (a solve set). Do it alongside `M07.4`: q3, q6, q7, and q8 are its functions by hand, q5 is why its intervals shrink; part `S-M07d` follows with hypothesis tests |
| **Milestone** | `MS-P4` (the Pass 4 gate runs `ol check` on every solve part of the pass) |
| **Optional depth** | Blitzstein and Hwang, *Introduction to Probability* (free), ch. 10 "Inequalities and limit theorems"; Wasserman, *All of Statistics*, ch. 5 to 8 |

## Key Takeaways

- The mean of $n$ independent draws has the same mean and $1/n$ of the variance, so its error shrinks like $1/\sqrt{n}$; halving an error bar costs four times the data (q1, q4).
- Chebyshev's inequality turns that variance into the weak law of large numbers in three lines (q2, q5).
- The central limit theorem lets a normal table answer questions about sums of non-normal variables, such as coin flips (q3).
- A confidence interval is a statement about the procedure, not about one interval: 95% of the intervals it would produce cover the true value (q9).
- Student's t, Wilson, and the type 7 percentile rule are each a few lines of arithmetic, the same numbers `M07.4` computes (q6, q7, q8).

## How to work this chapter

```bash
ol start S-M07c             # writes solve/S-M07c.toml and the proof file
ol check S-M07c             # SymPy checks the answers, then asks the proof rubric (y/n)
ol check S-M07c --regrade   # ask the rubric again after you change the proof
```

---

## 1. Why now

`M07.4` puts an interval on every number your models report from Pass 4 on: the bits per character of `L3.6`, the BLEU of `L4.4`, the accuracies of `L6.7`. Code computes those intervals without complaint whether or not you understand them, and a misread interval is worse than none: "95% sure model B is better" is not what a 95% interval says. This set makes the three ideas concrete by hand, so that the code is a transcription: why averaging works (the law of large numbers), why the error is normal and shrinks like $1/\sqrt{n}$ (the central limit theorem), and what exactly an interval promises.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $X_1, \dots, X_n$ | independent draws, each with mean $\mu$ and variance $\sigma^2$ (`mu`, `s2`) | reals |
| $\bar X_n$ | the sample mean $(X_1 + \dots + X_n)/n$ | real |
| $\varepsilon$ (`eps`) | a tolerance, $\varepsilon > 0$ | real |
| $\Phi$ | the standard normal CDF | function |
| $s$ | sample standard deviation, divisor $n - 1$ | real |
| $t_{1-\alpha/2,\,\nu}$ | Student's t quantile with $\nu$ degrees of freedom | real |
| $k, n, \hat p, z$ | successes, trials, $k/n$, the normal quantile | integers, reals |
| $s_0 \le \dots \le s_{B-1}$ | sorted bootstrap replicates | reals |

### 2.1 Averages

Linearity of expectation gives $E[\bar X_n] = \mu$. Independence makes every covariance in $\operatorname{Var}(\sum X_i)$ vanish (`S-M07b` q3), so the variance of the sum is $n\sigma^2$, and dividing by $n$ divides the variance by $n^2$: $\operatorname{Var}(\bar X_n) = \sigma^2/n$. The standard deviation of the mean is $\sigma/\sqrt{n}$.

### 2.2 Chebyshev and the law of large numbers

Markov's inequality, $P(Z \ge a) \le E[Z]/a$ for $Z \ge 0$ and $a > 0$, holds because $a\,\mathbf{1}[Z \ge a] \le Z$. Applied to $Z = (Y - EY)^2$ with $a = \varepsilon^2$ it becomes Chebyshev's inequality, $P(|Y - EY| \ge \varepsilon) \le \operatorname{Var}(Y)/\varepsilon^2$. With $Y = \bar X_n$ the bound shrinks to 0 as $n$ grows: that is the weak law of large numbers.

### 2.3 The central limit theorem

For large $n$, $(\bar X_n - \mu)/(\sigma/\sqrt{n})$ is approximately standard normal whatever the distribution of each $X_i$. Equivalently a sum $S_n = n\bar X_n$ is approximately normal with mean $n\mu$ and variance $n\sigma^2$, so $P(S_n \ge c) \approx 1 - \Phi\bigl((c - n\mu)/\sqrt{n\sigma^2}\bigr)$.

### 2.4 Intervals

A 95% confidence interval is a **procedure**: data in, interval out, built so that over repeated samples 95% of the intervals contain the fixed, unknown $\mu$. Once computed, a particular interval either contains $\mu$ or not; the 95% belongs to the procedure. The three procedures of `M07.4`:

- **Student t**: $\bar x \pm t_{1-\alpha/2,\,n-1}\, s/\sqrt{n}$.
- **Wilson** for a proportion: $\bigl(\hat p + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^2}{4n^2}}\bigr) / (1 + \frac{z^2}{n})$.
- **Percentile bootstrap**: the type 7 quantiles at $\alpha/2$ and $1 - \alpha/2$ of the replicates: $h = (B - 1)q$, $i = \lfloor h \rfloor$, value $s_i + (h - i)(s_{i+1} - s_i)$.

## 3. Worked example by hand

These are siblings of q3, q6, and q8, not graded problems.

**A CLT tail.** 400 fair coin flips; $P(\text{at least } 220 \text{ heads})$. The count has mean $400 \cdot \frac{1}{2} = 200$ and variance $400 \cdot \frac{1}{4} = 100$, sd 10, so $z = (220 - 200)/10 = 2$ and the probability is about $1 - \Phi(2) = 1 - 0.97725 = 0.02275$. Twenty extra heads out of 400 is as surprising as ten out of 100: the sd grows like $\sqrt{n}$, not $n$.

**A t interval.** $n = 16$, $\bar x = 3$, $s = 4$, $t_{0.975,\,15} = 2.131$: the half-width is $2.131 \times 4/\sqrt{16} = 2.131$, so the interval is $[0.869, 5.131]$. In `solve/` this would be `answer = "[0.869, 5.131]"`.

**A percentile interval.** Five replicates $2, 3, 3, 7, 10$ and $q = 0.25$: $h = 4 \times 0.25 = 1$, $i = 1$, value $s_1 = 3$; $q = 0.75$: $h = 3$, value $s_3 = 7$. The 50% interval is $[3, 7]$.

## 4. The problem set

Write each answer in `solve/S-M07c.toml`; lettered parts are their own tables:

```toml
[q1.a]
answer = "mu"
[q3]
answer = "0.02275"
[q6]
answer = "[9.1744, 10.8256]"
[q9]
answer = "c"
[q5]
proof = "S-M07c/q5.md"
```

Write $\sigma^2$ as `s2`, $\varepsilon$ as `eps`, $\mu$ as `mu`. An interval is written `[lo, hi]`; exact fractions such as `2/7` are welcome.

<!-- ol:problems S-M07c -->

### The law of large numbers and the CLT

**q1.** $X_1, \dots, X_n$ are independent, each with mean $\mu$ and variance $\sigma^2$, and $\bar X_n = (X_1 + \dots + X_n)/n$. (a) $E[\bar X_n]$. `[expr in mu]` (b) $\operatorname{Var}(\bar X_n)$. `[expr in s2, n]`

**q2.** For the same $\bar X_n$ and any $\varepsilon > 0$, Chebyshev's inequality $P(|Y - E Y| \ge \varepsilon) \le \operatorname{Var}(Y)/\varepsilon^2$ bounds $P(|\bar X_n - \mu| \ge \varepsilon)$. Give the bound. `[expr in s2, n, eps]`

**q3.** You flip a fair coin 100 times. Using the central limit theorem without a continuity correction, and $\Phi(2) = 0.97725$ for the standard normal CDF, approximate $P(\text{at least } 60 \text{ heads})$. `[number, 3 significant digits]`

**q4.** A 95% interval for an eval score is $\bar x \pm 1.96\, s/\sqrt{n}$. By what factor must $n$ grow to halve its width (with $s$ unchanged)? `[number, exact]`

**q5.** Prove the weak law of large numbers for finite variance: if $X_1, X_2, \dots$ are independent with mean $\mu$ and variance $\sigma^2 < \infty$, then for every $\varepsilon > 0$, $P(|\bar X_n - \mu| \ge \varepsilon) \to 0$ as $n \to \infty$. `[proof]`

### Confidence intervals

**q6.** An eval of $n = 25$ prompts has mean score $\bar x = 10$ and sample standard deviation $s = 2$ (divisor $n - 1$). With $t_{0.975,\,24} = 2.064$, give the 95% Student t interval for the true mean. `[interval, exact decimals]`

**q7.** A safety check fails on 0 of $n = 10$ prompts. Give the Wilson score interval for the failure rate with $z = 2$:
$$\frac{\hat p + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat p(1 - \hat p)}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}.$$
`[interval, exact]`

**q8.** Nine bootstrap replicates of a statistic, sorted, are $1, 2, 4, 5, 5, 6, 8, 9, 13$. With the type 7 quantile (for sorted $s_0, \dots, s_{B-1}$ and level $q$: $h = (B - 1)q$, $i = \lfloor h \rfloor$, value $s_i + (h - i)(s_{i+1} - s_i)$), give the 80% percentile interval $[\text{quantile}(0.1), \text{quantile}(0.9)]$. `[interval, exact]`

**q9.** From one sample you compute the 95% confidence interval $[3.1, 4.7]$ for a mean $\mu$. Which statement is correct? `[choice]`
(a) $P(3.1 \le \mu \le 4.7) = 0.95$, with $\mu$ random.
(b) 95% of the observations lie in $[3.1, 4.7]$.
(c) The procedure that produced it covers the true $\mu$ in 95% of repeated samples.
(d) A new sample's mean lands in $[3.1, 4.7]$ with probability 0.95.

<!-- /ol:problems -->

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| Forgetting that $1/n$ is squared in a variance | the variance of a mean reported as $\sigma^2$ or $\sigma^2/n^2$ | q1 (canaries s2 and s2/n^2) |
| Confusing the variance with the standard error | $\sqrt{\sigma^2/n}$ where a variance is asked | q1 (canary sqrt(s2/n)) |
| Chebyshev on one draw instead of the mean | a bound that does not shrink with $n$ | q2 (canary s2/eps^2) |
| A two-sided tail for a one-sided question | 0.0455 instead of 0.02275 | q3 (canary) |
| Width proportional to $1/n$ | "double the data, halve the bar" | q4 (canary 2) |
| The normal 1.96 with an estimated $s$ at small $n$, or $s$ without $\sqrt{n}$ | intervals too narrow, or 5 times too wide | q6 (canaries) |
| The Wald interval at $k = 0$ | $[0, 0]$: certainty from ten trials | q7 (canary) |
| Nearest-rank instead of interpolated quantiles | a bootstrap interval that disagrees with numpy and Go | q8 (canary [1, 9]) |
| Reading the 95% as a probability about this one interval | "95% chance the mean is in here" | q9 (canary a) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `S-M07b` | the variance of a sum with covariances, q3 there |
| Back | `M07.4` | `mean_ci`, `wilson_interval`, `quantile`, and `bootstrap_ci` are q6, q7, and q8 in code (reading) |
| Forward | `S-M07d` | hypothesis tests: the same sampling distributions, read as p-values |
| Forward | `L6.7` | every metric with its interval, and what the interval does and does not claim |
| Forward | `L4.5` | bootstrap intervals over sentences for BLEU |
