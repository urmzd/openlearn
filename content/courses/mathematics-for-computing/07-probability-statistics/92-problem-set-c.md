# Solve set: LLN/CLT, confidence intervals

## Overview

| | |
|---|---|
| **Module** | `S-M07c` · solve · none · Pass 4 · 2 to 3 h |
| **Solution** | a worked answer after every problem in section 4: 9 answers (counting lettered parts) and 1 proof (q5; model proof also in [`course/solutions/solve/S-M07c/q5.md`](../../../../course/solutions/solve/S-M07c/q5.md)) |
| **Contract** | none: a pen and paper set |
| **Needs** | `S-M07b` (the variance of a sum). Reading: [`M07.4` LLN, CLT, confidence intervals, bootstrap](04-lln-clt-confidence-intervals-bootstrap.md) and the [Probability and Statistics topic](README.md), limit theorems and estimation sections |
| **Used by** | no call site (a solve set). Read it alongside `M07.4`: q3, q6, q7, and q8 are its functions by hand, q5 is why its intervals shrink; part `S-M07d` follows with hypothesis tests |
| **Optional depth** | Blitzstein and Hwang, *Introduction to Probability* (free), ch. 10 "Inequalities and limit theorems"; Wasserman, *All of Statistics*, ch. 5 to 8 |

## Key Takeaways

- The mean of $n$ independent draws has the same mean and $1/n$ of the variance, so its error shrinks like $1/\sqrt{n}$; halving an error bar costs four times the data (q1, q4).
- Chebyshev's inequality turns that variance into the weak law of large numbers in three lines (q2, q5).
- The central limit theorem lets a normal table answer questions about sums of non-normal variables, such as coin flips (q3).
- A confidence interval is a statement about the procedure, not about one interval: 95% of the intervals it would produce cover the true value (q9).
- Student's t, Wilson, and the type 7 percentile rule are each a few lines of arithmetic, the same numbers `M07.4` computes (q6, q7, q8).

## How to read this chapter

Sections 1 to 3 review the ideas and work three examples by hand. Section 4 is the problem set: every problem is followed by a collapsed worked answer with the final result, the derivation, and the common wrong answers and why they are wrong. Try each problem before opening its answer.

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

These are siblings of q3, q6, and q8, not problems of the set.

**A CLT tail.** 400 fair coin flips; $P(\text{at least } 220 \text{ heads})$. The count has mean $400 \cdot \frac{1}{2} = 200$ and variance $400 \cdot \frac{1}{4} = 100$, sd 10, so $z = (220 - 200)/10 = 2$ and the probability is about $1 - \Phi(2) = 1 - 0.97725 = 0.02275$. Twenty extra heads out of 400 is as surprising as ten out of 100: the sd grows like $\sqrt{n}$, not $n$.

**A t interval.** $n = 16$, $\bar x = 3$, $s = 4$, $t_{0.975,\,15} = 2.131$: the half-width is $2.131 \times 4/\sqrt{16} = 2.131$, so the interval is $[0.869, 5.131]$.

**A percentile interval.** Five replicates $2, 3, 3, 7, 10$ and $q = 0.25$: $h = 4 \times 0.25 = 1$, $i = 1$, value $s_1 = 3$; $q = 0.75$: $h = 3$, value $s_3 = 7$. The 50% interval is $[3, 7]$.

## 4. The problem set, with worked answers

$\sigma^2$ is written `s2`, $\varepsilon$ is `eps`, $\mu$ is `mu`. An interval is written $[lo, hi]$, and exact fractions such as $2/7$ are kept exact. Lettered parts are answered separately. The tag after each problem says what kind of answer is expected.

### The law of large numbers and the CLT

**q1.** $X_1, \dots, X_n$ are independent, each with mean $\mu$ and variance $\sigma^2$, and $\bar X_n = (X_1 + \dots + X_n)/n$. (a) $E[\bar X_n]$. `[expr in mu]` (b) $\operatorname{Var}(\bar X_n)$. `[expr in s2, n]`

<details><summary>Answer</summary>

**(a) $\mu$. (b) $\sigma^2/n$** (`s2/n`). (a) Linearity: $E[\bar X_n] = \frac1n\sum_i E[X_i] = \frac1n \cdot n\mu = \mu$; averaging keeps the mean. (b) Independence removes every covariance, so $\operatorname{Var}(\sum_i X_i) = n\sigma^2$, and the factor $1/n$ comes out squared: $\frac{1}{n^2} \cdot n\sigma^2 = \frac{\sigma^2}{n}$. Common mistakes: $\mu/n$ for (a), dividing once too often; $\sigma^2$ for (b), the variance of one draw; $\sigma^2/n^2$, squaring the $1/n$ but forgetting the sum has $n$ terms; $\sqrt{\sigma^2/n}$, the standard error, not the variance.

</details>

**q2.** For the same $\bar X_n$ and any $\varepsilon > 0$, Chebyshev's inequality $P(|Y - E Y| \ge \varepsilon) \le \operatorname{Var}(Y)/\varepsilon^2$ bounds $P(|\bar X_n - \mu| \ge \varepsilon)$. Give the bound. `[expr in s2, n, eps]`

<details><summary>Answer</summary>

**$\dfrac{\sigma^2}{n\varepsilon^2}$** (`s2/(n*eps^2)`). Apply Chebyshev to $Y = \bar X_n$, whose mean is $\mu$ and variance $\sigma^2/n$ (q1): $P(|\bar X_n - \mu| \ge \varepsilon) \le \frac{\sigma^2/n}{\varepsilon^2}$. For fixed $\varepsilon$ this goes to 0 as $n$ grows, which is the law of large numbers (q5). Common mistakes: $\sigma^2/\varepsilon^2$, the bound for one draw, which does not shrink; $\sigma^2/(n\varepsilon)$, with $\varepsilon$ not squared.

</details>

**q3.** You flip a fair coin 100 times. Using the central limit theorem without a continuity correction, and $\Phi(2) = 0.97725$ for the standard normal CDF, approximate $P(\text{at least } 60 \text{ heads})$. `[number, 3 significant digits]`

<details><summary>Answer</summary>

**$1 - 0.97725 = 0.02275$** ($\approx 0.0228$). The number of heads has mean $100 \cdot \frac12 = 50$ and variance $100 \cdot \frac14 = 25$, sd 5. The CLT treats it as normal: $z = (60 - 50)/5 = 2$, and the one-sided tail is $1 - \Phi(2) = 0.02275$. Common mistakes: $2(1 - 0.97725) = 0.0455$, the two-sided tail ("at least 60 or at most 40") when only one side was asked; $0.97725$, the CDF instead of the tail.

</details>

**q4.** A 95% interval for an eval score is $\bar x \pm 1.96\, s/\sqrt{n}$. By what factor must $n$ grow to halve its width (with $s$ unchanged)? `[number, exact]`

<details><summary>Answer</summary>

**$4$.** The half-width is $1.96\,s/\sqrt n$, proportional to $1/\sqrt n$. Halving it needs $\sqrt{n'} = 2\sqrt n$, so $n' = 4n$. Common mistakes: 2, treating the width as proportional to $1/n$; 16, squaring twice.

</details>

**q5.** Prove the weak law of large numbers for finite variance: if $X_1, X_2, \dots$ are independent with mean $\mu$ and variance $\sigma^2 < \infty$, then for every $\varepsilon > 0$, $P(|\bar X_n - \mu| \ge \varepsilon) \to 0$ as $n \to \infty$. `[proof]`

<details><summary>Answer</summary>

**Claim.** If $X_1, X_2, \dots$ are independent, each with mean $\mu$ and finite variance $\sigma^2$, and $\bar X_n = \frac1n\sum_{i=1}^n X_i$, then for every $\varepsilon > 0$, $P(|\bar X_n - \mu| \ge \varepsilon) \to 0$ as $n \to \infty$.

**Proof.** Fix $\varepsilon > 0$.

1. **Mean.** By linearity, $E[\bar X_n] = \frac1n \cdot n\mu = \mu$.
2. **Variance.** Independent variables have zero covariances, so $\operatorname{Var}(\sum_i X_i) = n\sigma^2$, and scaling by $1/n$ multiplies the variance by $1/n^2$: $\operatorname{Var}(\bar X_n) = \sigma^2/n$. This is where $\sigma^2 < \infty$ is used: every term is a finite number.
3. **Chebyshev.** For $Y$ with finite variance, Markov's inequality ($P(Z \ge a) \le E[Z]/a$ for $Z \ge 0$, $a > 0$, because $a\,\mathbf 1[Z \ge a] \le Z$) applied to $Z = (Y - EY)^2$ and $a = \varepsilon^2$ gives $P(|Y - EY| \ge \varepsilon) \le \operatorname{Var}(Y)/\varepsilon^2$.
4. **Combine.** With $Y = \bar X_n$: $P(|\bar X_n - \mu| \ge \varepsilon) \le \frac{\sigma^2}{n\varepsilon^2}$.
5. **Limit.** The right side is a constant times $1/n$ and tends to 0; probabilities are at least 0, so the left side tends to 0 too.

Hence $\bar X_n \to \mu$ in probability. A complete proof computes the mean and variance (using independence for the variance), states or derives Chebyshev, takes the limit for every fixed $\varepsilon$, and says where finite variance is used. Model proof also in [`q5.md`](../../../../course/solutions/solve/S-M07c/q5.md).

</details>

### Confidence intervals

**q6.** An eval of $n = 25$ prompts has mean score $\bar x = 10$ and sample standard deviation $s = 2$ (divisor $n - 1$). With $t_{0.975,\,24} = 2.064$, give the 95% Student t interval for the true mean. `[interval, exact decimals]`

<details><summary>Answer</summary>

**$[9.1744, 10.8256]$.** The standard error is $s/\sqrt n = 2/5 = 0.4$, and the half-width is $2.064 \times 0.4 = 0.8256$, so $10 \mp 0.8256$. Common mistakes: $[9.216, 10.784]$, using the normal 1.96 when $s$ is estimated from 25 points (Student's t is wider to pay for that estimate); $[5.872, 14.128]$, multiplying $t$ by $s$ instead of $s/\sqrt n$.

</details>

**q7.** A safety check fails on 0 of $n = 10$ prompts. Give the Wilson score interval for the failure rate with $z = 2$:
$$\frac{\hat p + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat p(1 - \hat p)}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}.$$
`[interval, exact]`

<details><summary>Answer</summary>

**$[0, 2/7]$.** With $\hat p = 0$, $n = 10$, $z = 2$: the centre term is $\frac{z^2}{2n} = \frac4{20} = \frac15$, the root is $z\sqrt{0 + \frac{z^2}{4n^2}} = 2\sqrt{\frac4{400}} = \frac15$, and the denominator is $1 + \frac4{10} = \frac75$. Lower: $(\frac15 - \frac15)/\frac75 = 0$. Upper: $\frac25 / \frac75 = \frac27 \approx 0.286$. Zero failures in ten trials still leave a failure rate up to about 29% plausible. Common mistakes: $[0, 0]$, the Wald interval $\hat p \pm z\sqrt{\hat p(1 - \hat p)/n}$, which claims certainty from ten trials; $[0, 4/10]$, dropping the denominator $1 + z^2/n$.

</details>

**q8.** Nine bootstrap replicates of a statistic, sorted, are $1, 2, 4, 5, 5, 6, 8, 9, 13$. With the type 7 quantile (for sorted $s_0, \dots, s_{B-1}$ and level $q$: $h = (B - 1)q$, $i = \lfloor h \rfloor$, value $s_i + (h - i)(s_{i+1} - s_i)$), give the 80% percentile interval $[\text{quantile}(0.1), \text{quantile}(0.9)]$. `[interval, exact]`

<details><summary>Answer</summary>

**$[9/5, 49/5] = [1.8, 9.8]$.** $B = 9$. For $q = 0.1$: $h = 8 \times 0.1 = 0.8$, $i = 0$, value $s_0 + 0.8(s_1 - s_0) = 1 + 0.8 = 1.8$. For $q = 0.9$: $h = 7.2$, $i = 7$, value $s_7 + 0.2(s_8 - s_7) = 9 + 0.2 \times 4 = 9.8$. This is the rule numpy uses by default, so `M07.4`'s Python and Go agree with it. Common mistakes: $[1, 9]$, taking $s_{\lfloor h \rfloor}$ without interpolating; $[2, 9]$, a nearest-rank rule.

</details>

**q9.** From one sample you compute the 95% confidence interval $[3.1, 4.7]$ for a mean $\mu$. Which statement is correct? `[choice]`
(a) $P(3.1 \le \mu \le 4.7) = 0.95$, with $\mu$ random.
(b) 95% of the observations lie in $[3.1, 4.7]$.
(c) The procedure that produced it covers the true $\mu$ in 95% of repeated samples.
(d) A new sample's mean lands in $[3.1, 4.7]$ with probability 0.95.

<details><summary>Answer</summary>

**(c).** The parameter $\mu$ is fixed; the interval is the random object. Before sampling, the procedure has a 95% chance of producing an interval that covers $\mu$; once $[3.1, 4.7]$ is computed, it either contains $\mu$ or not. Common mistakes: (a), which treats $\mu$ as random and this one interval as carrying the 95%; (d), which confuses a confidence interval with a prediction interval for a new sample mean (a new mean is itself random and lands in the interval less often). (b) confuses the interval for the mean with the spread of the observations.

</details>

## 5. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| Forgetting that $1/n$ is squared in a variance | the variance of a mean reported as $\sigma^2$ or $\sigma^2/n^2$ | $n\sigma^2$ for the sum, times $1/n^2$ (q1 b) |
| Confusing the variance with the standard error | $\sqrt{\sigma^2/n}$ where a variance is asked | reports the variance, $\sigma^2/n$, and keeps the root for the interval (q1 b, q6) |
| Chebyshev on one draw instead of the mean | a bound that does not shrink with $n$ | applies it to $\bar X_n$, variance $\sigma^2/n$ (q2) |
| A two-sided tail for a one-sided question | 0.0455 instead of 0.02275 | takes $1 - \Phi(z)$ once (q3) |
| Width proportional to $1/n$ | "double the data, halve the bar" | width $\propto 1/\sqrt n$, so four times the data (q4) |
| The normal 1.96 with an estimated $s$ at small $n$, or $s$ without $\sqrt{n}$ | intervals too narrow, or 5 times too wide | uses $t_{0.975,\,n-1}$ times $s/\sqrt n$ (q6) |
| The Wald interval at $k = 0$ | $[0, 0]$: certainty from ten trials | uses Wilson, which stays open at $k = 0$ (q7) |
| Nearest-rank instead of interpolated quantiles | a bootstrap interval that disagrees with numpy and Go | interpolates with type 7 (q8) |
| Reading the 95% as a probability about this one interval | "95% chance the mean is in here" | attaches the 95% to the procedure (q9) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `S-M07b` | the variance of a sum with covariances, q3 there |
| Back | `M07.4` | `mean_ci`, `wilson_interval`, `quantile`, and `bootstrap_ci` are q6, q7, and q8 in code (reading) |
| Forward | `S-M07d` | hypothesis tests: the same sampling distributions, read as p-values |
| Forward | `L6.7` | every metric with its interval, and what the interval does and does not claim |
| Forward | `L4.5` | bootstrap intervals over sentences for BLEU |
