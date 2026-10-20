# Probability problem set, part c: limit theorems and confidence intervals

[LLN, CLT, confidence intervals, bootstrap](05-lln-clt-confidence-intervals-bootstrap.md) puts an interval on every number your models report: the bits per character of the [RNN language model](../../ai-engineering/09-recurrent-networks/06-rnn-language-model-stateful-tbptt.md), the BLEU of [beam search](../../ai-engineering/10-attention/04-beam-search.md), the accuracies of the [evaluation harness](../../ai-engineering/12-objectives-and-adaptation/07-lm-evaluation-harness-and-the-model-zoo.md) in *AI Engineering*. Code computes those intervals without complaint whether or not you understand them, and a misread interval is worse than none: "95% sure model B is better" is not what a 95% interval says.

This problem set makes three ideas concrete by hand, so that the code becomes a transcription: why averaging works (the law of large numbers), why the error is normal and shrinks like $1/\sqrt{n}$ (the central limit theorem), and what exactly an interval promises. It is a pen-and-paper set with 9 answers (counting lettered parts) and one proof. A short review and three examples by hand come first; then the problems, each followed by a collapsed worked answer with the final result, the derivation, and the common wrong answers and why they are wrong. Try each problem before opening its answer. Read it alongside the limit-theorems section: Problems 3, 6, 7, and 8 are its functions by hand, and Problem 5 is why its intervals shrink. [Problem set D](12-problem-set-d.md) follows with hypothesis tests.

The set builds on the variance of a sum from [problem set B](10-problem-set-b.md) and on the limit-theorem and inference material in the [chapter opening](README.md).

> **Learning objectives**
> - Compute the mean and variance of a sample mean and explain why its error shrinks like $1/\sqrt{n}$.
> - Derive Chebyshev's inequality from Markov's and use it to prove the weak law of large numbers.
> - Approximate tail probabilities of sums with the central limit theorem.
> - Compute Student t, Wilson, and type 7 percentile intervals by hand.
> - State correctly what a 95% confidence interval does and does not promise.

## Notation

The set uses these symbols.

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

## Averages

Linearity of expectation gives $E[\bar X_n] = \mu$. Independence makes every covariance in $\operatorname{Var}(\sum X_i)$ vanish (Problem 3 of [set B](10-problem-set-b.md)), so the variance of the sum is $n\sigma^2$, and dividing by $n$ divides the variance by $n^2$: $\operatorname{Var}(\bar X_n) = \sigma^2/n$. The standard deviation of the mean is $\sigma/\sqrt{n}$.

## Chebyshev and the law of large numbers

That variance turns into a law in three lines. Markov's inequality, $P(Z \ge a) \le E[Z]/a$ for $Z \ge 0$ and $a > 0$, holds because $a\,\mathbf{1}[Z \ge a] \le Z$. Applied to $Z = (Y - EY)^2$ with $a = \varepsilon^2$ it becomes Chebyshev's inequality, $P(|Y - EY| \ge \varepsilon) \le \operatorname{Var}(Y)/\varepsilon^2$. With $Y = \bar X_n$ the bound shrinks to 0 as $n$ grows: that is the weak law of large numbers.

## The central limit theorem

The law of large numbers says the error goes to zero; the central limit theorem says what shape it has on the way. For large $n$, $(\bar X_n - \mu)/(\sigma/\sqrt{n})$ is approximately standard normal whatever the distribution of each $X_i$. Equivalently, a sum $S_n = n\bar X_n$ is approximately normal with mean $n\mu$ and variance $n\sigma^2$, so $P(S_n \ge c) \approx 1 - \Phi\bigl((c - n\mu)/\sqrt{n\sigma^2}\bigr)$.

## Intervals

A 95% confidence interval is a **procedure**: data in, interval out, built so that over repeated samples 95% of the intervals contain the fixed, unknown $\mu$. Once computed, a particular interval either contains $\mu$ or not; the 95% belongs to the procedure. The limit-theorems section implements three procedures.

- **Student t**: $\bar x \pm t_{1-\alpha/2,\,n-1}\, s/\sqrt{n}$.
- **Wilson** for a proportion: $\bigl(\hat p + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^2}{4n^2}}\bigr) / (1 + \frac{z^2}{n})$.
- **Percentile bootstrap**: the type 7 quantiles at $\alpha/2$ and $1 - \alpha/2$ of the replicates: $h = (B - 1)q$, $i = \lfloor h \rfloor$, value $s_i + (h - i)(s_{i+1} - s_i)$.

## Three examples by hand

These are siblings of Problems 3, 6, and 8, not problems of the set.

**A CLT tail.** Take 400 fair coin flips and ask for $P(\text{at least } 220 \text{ heads})$. The count has mean $400 \cdot \frac{1}{2} = 200$ and variance $400 \cdot \frac{1}{4} = 100$, sd 10, so $z = (220 - 200)/10 = 2$ and the probability is about $1 - \Phi(2) = 1 - 0.97725 = 0.02275$. Twenty extra heads out of 400 is as surprising as ten out of 100: the sd grows like $\sqrt{n}$, not $n$.

**A t interval.** With $n = 16$, $\bar x = 3$, $s = 4$, and $t_{0.975,\,15} = 2.131$, the half-width is $2.131 \times 4/\sqrt{16} = 2.131$, so the interval is $[0.869, 5.131]$.

**A percentile interval.** Take five replicates $2, 3, 3, 7, 10$. For $q = 0.25$: $h = 4 \times 0.25 = 1$, $i = 1$, value $s_1 = 3$. For $q = 0.75$: $h = 3$, value $s_3 = 7$. The 50% interval is $[3, 7]$.

## The problems

$\sigma^2$ is written `s2`, $\varepsilon$ is `eps`, and $\mu$ is `mu`. An interval is written $[lo, hi]$, and exact fractions such as $2/7$ are kept exact. Lettered parts are answered separately, and the tag after each problem says what kind of answer is expected.

### The law of large numbers and the CLT

The first five problems build from the variance of a mean to a full proof of the weak law.

**Problem 1.** $X_1, \dots, X_n$ are independent, each with mean $\mu$ and variance $\sigma^2$, and $\bar X_n = (X_1 + \dots + X_n)/n$. (a) $E[\bar X_n]$. `[expr in mu]` (b) $\operatorname{Var}(\bar X_n)$. `[expr in s2, n]`

<details><summary>Answer</summary>

**(a) $\mu$. (b) $\sigma^2/n$** (`s2/n`). (a) Linearity: $E[\bar X_n] = \frac1n\sum_i E[X_i] = \frac1n \cdot n\mu = \mu$; averaging keeps the mean. (b) Independence removes every covariance, so $\operatorname{Var}(\sum_i X_i) = n\sigma^2$, and the factor $1/n$ comes out squared: $\frac{1}{n^2} \cdot n\sigma^2 = \frac{\sigma^2}{n}$. Common mistakes: $\mu/n$ for (a), dividing once too often; $\sigma^2$ for (b), the variance of one draw; $\sigma^2/n^2$, squaring the $1/n$ but forgetting the sum has $n$ terms; $\sqrt{\sigma^2/n}$, the standard error, not the variance.

</details>

**Problem 2.** For the same $\bar X_n$ and any $\varepsilon > 0$, Chebyshev's inequality $P(|Y - E Y| \ge \varepsilon) \le \operatorname{Var}(Y)/\varepsilon^2$ bounds $P(|\bar X_n - \mu| \ge \varepsilon)$. Give the bound. `[expr in s2, n, eps]`

<details><summary>Answer</summary>

**$\dfrac{\sigma^2}{n\varepsilon^2}$** (`s2/(n*eps^2)`). Apply Chebyshev to $Y = \bar X_n$, whose mean is $\mu$ and variance $\sigma^2/n$ (Problem 1): $P(|\bar X_n - \mu| \ge \varepsilon) \le \frac{\sigma^2/n}{\varepsilon^2}$. For fixed $\varepsilon$ this goes to 0 as $n$ grows, which is the law of large numbers (Problem 5). Common mistakes: $\sigma^2/\varepsilon^2$, the bound for one draw, which does not shrink; $\sigma^2/(n\varepsilon)$, with $\varepsilon$ not squared.

</details>

**Problem 3.** You flip a fair coin 100 times. Using the central limit theorem without a continuity correction, and $\Phi(2) = 0.97725$ for the standard normal CDF, approximate $P(\text{at least } 60 \text{ heads})$. `[number, 3 significant digits]`

<details><summary>Answer</summary>

**$1 - 0.97725 = 0.02275$** ($\approx 0.0228$). The number of heads has mean $100 \cdot \frac12 = 50$ and variance $100 \cdot \frac14 = 25$, sd 5. The CLT treats it as normal: $z = (60 - 50)/5 = 2$, and the one-sided tail is $1 - \Phi(2) = 0.02275$. Common mistakes: $2(1 - 0.97725) = 0.0455$, the two-sided tail ("at least 60 or at most 40") when only one side was asked; $0.97725$, the CDF instead of the tail.

</details>

**Problem 4.** A 95% interval for an eval score is $\bar x \pm 1.96\, s/\sqrt{n}$. By what factor must $n$ grow to halve its width (with $s$ unchanged)? `[number, exact]`

<details><summary>Answer</summary>

**$4$.** The half-width is $1.96\,s/\sqrt n$, proportional to $1/\sqrt n$. Halving it needs $\sqrt{n'} = 2\sqrt n$, so $n' = 4n$. Common mistakes: 2, treating the width as proportional to $1/n$; 16, squaring twice.

</details>

**Problem 5.** Prove the weak law of large numbers for finite variance: if $X_1, X_2, \dots$ are independent with mean $\mu$ and variance $\sigma^2 < \infty$, then for every $\varepsilon > 0$, $P(|\bar X_n - \mu| \ge \varepsilon) \to 0$ as $n \to \infty$. `[proof]`

<details><summary>Answer</summary>

**Claim.** If $X_1, X_2, \dots$ are independent, each with mean $\mu$ and finite variance $\sigma^2$, and $\bar X_n = \frac1n\sum_{i=1}^n X_i$, then for every $\varepsilon > 0$, $P(|\bar X_n - \mu| \ge \varepsilon) \to 0$ as $n \to \infty$.

**Proof.** Fix $\varepsilon > 0$.

1. **Mean.** By linearity, $E[\bar X_n] = \frac1n \cdot n\mu = \mu$.
2. **Variance.** Independent variables have zero covariances, so $\operatorname{Var}(\sum_i X_i) = n\sigma^2$, and scaling by $1/n$ multiplies the variance by $1/n^2$: $\operatorname{Var}(\bar X_n) = \sigma^2/n$. This is where $\sigma^2 < \infty$ is used: every term is a finite number.
3. **Chebyshev.** For $Y$ with finite variance, Markov's inequality ($P(Z \ge a) \le E[Z]/a$ for $Z \ge 0$, $a > 0$, because $a\,\mathbf 1[Z \ge a] \le Z$) applied to $Z = (Y - EY)^2$ and $a = \varepsilon^2$ gives $P(|Y - EY| \ge \varepsilon) \le \operatorname{Var}(Y)/\varepsilon^2$.
4. **Combine.** With $Y = \bar X_n$: $P(|\bar X_n - \mu| \ge \varepsilon) \le \frac{\sigma^2}{n\varepsilon^2}$.
5. **Limit.** The right side is a constant times $1/n$ and tends to 0; probabilities are at least 0, so the left side tends to 0 too.

Hence $\bar X_n \to \mu$ in probability. A complete proof computes the mean and variance (using independence for the variance), states or derives Chebyshev, takes the limit for every fixed $\varepsilon$, and says where finite variance is used. The model proof is also in [`course/solutions/solve/S-M07c/q5.md`](../../../../course/solutions/solve/S-M07c/q5.md).

</details>

### Confidence intervals

The last four problems compute each of the three intervals and then ask what one of them means.

**Problem 6.** An eval of $n = 25$ prompts has mean score $\bar x = 10$ and sample standard deviation $s = 2$ (divisor $n - 1$). With $t_{0.975,\,24} = 2.064$, give the 95% Student t interval for the true mean. `[interval, exact decimals]`

<details><summary>Answer</summary>

**$[9.1744, 10.8256]$.** The standard error is $s/\sqrt n = 2/5 = 0.4$, and the half-width is $2.064 \times 0.4 = 0.8256$, so $10 \mp 0.8256$. Common mistakes: $[9.216, 10.784]$, using the normal 1.96 when $s$ is estimated from 25 points (Student's t is wider to pay for that estimate); $[5.872, 14.128]$, multiplying $t$ by $s$ instead of $s/\sqrt n$.

</details>

**Problem 7.** A safety check fails on 0 of $n = 10$ prompts. Give the Wilson score interval for the failure rate with $z = 2$:
$$\frac{\hat p + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat p(1 - \hat p)}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}.$$
`[interval, exact]`

<details><summary>Answer</summary>

**$[0, 2/7]$.** With $\hat p = 0$, $n = 10$, $z = 2$: the centre term is $\frac{z^2}{2n} = \frac4{20} = \frac15$, the root is $z\sqrt{0 + \frac{z^2}{4n^2}} = 2\sqrt{\frac4{400}} = \frac15$, and the denominator is $1 + \frac4{10} = \frac75$. Lower: $(\frac15 - \frac15)/\frac75 = 0$. Upper: $\frac25 / \frac75 = \frac27 \approx 0.286$. Zero failures in ten trials still leave a failure rate up to about 29% plausible. Common mistakes: $[0, 0]$, the Wald interval $\hat p \pm z\sqrt{\hat p(1 - \hat p)/n}$, which claims certainty from ten trials; $[0, 4/10]$, dropping the denominator $1 + z^2/n$.

</details>

**Problem 8.** Nine bootstrap replicates of a statistic, sorted, are $1, 2, 4, 5, 5, 6, 8, 9, 13$. With the type 7 quantile (for sorted $s_0, \dots, s_{B-1}$ and level $q$: $h = (B - 1)q$, $i = \lfloor h \rfloor$, value $s_i + (h - i)(s_{i+1} - s_i)$), give the 80% percentile interval $[\text{quantile}(0.1), \text{quantile}(0.9)]$. `[interval, exact]`

<details><summary>Answer</summary>

**$[9/5, 49/5] = [1.8, 9.8]$.** $B = 9$. For $q = 0.1$: $h = 8 \times 0.1 = 0.8$, $i = 0$, value $s_0 + 0.8(s_1 - s_0) = 1 + 0.8 = 1.8$. For $q = 0.9$: $h = 7.2$, $i = 7$, value $s_7 + 0.2(s_8 - s_7) = 9 + 0.2 \times 4 = 9.8$. This is the rule numpy uses by default, so the Python and Go implementations of the limit-theorems section agree with it. Common mistakes: $[1, 9]$, taking $s_{\lfloor h \rfloor}$ without interpolating; $[2, 9]$, a nearest-rank rule.

</details>

**Problem 9.** From one sample you compute the 95% confidence interval $[3.1, 4.7]$ for a mean $\mu$. Which statement is correct? `[choice]`
(a) $P(3.1 \le \mu \le 4.7) = 0.95$, with $\mu$ random.
(b) 95% of the observations lie in $[3.1, 4.7]$.
(c) The procedure that produced it covers the true $\mu$ in 95% of repeated samples.
(d) A new sample's mean lands in $[3.1, 4.7]$ with probability 0.95.

<details><summary>Answer</summary>

**(c).** The parameter $\mu$ is fixed; the interval is the random object. Before sampling, the procedure has a 95% chance of producing an interval that covers $\mu$; once $[3.1, 4.7]$ is computed, it either contains $\mu$ or not. Common mistakes: (a), which treats $\mu$ as random and this one interval as carrying the 95%; (d), which confuses a confidence interval with a prediction interval for a new sample mean (a new mean is itself random and lands in the interval less often). (b) confuses the interval for the mean with the spread of the observations.

</details>

## What can go wrong

The variance mistakes come first and propagate into everything else. Forgetting that $1/n$ is squared in a variance reports the variance of a mean as $\sigma^2$ or $\sigma^2/n^2$; the sum has variance $n\sigma^2$, times $1/n^2$ (Problem 1b). Confusing the variance with the standard error puts $\sqrt{\sigma^2/n}$ where a variance is asked; report $\sigma^2/n$ and keep the root for the interval (Problems 1b and 6). Applying Chebyshev to one draw instead of the mean gives a bound that does not shrink with $n$; apply it to $\bar X_n$, whose variance is $\sigma^2/n$ (Problem 2). Answering a one-sided question with a two-sided tail gives 0.0455 instead of 0.02275; take $1 - \Phi(z)$ once (Problem 3). And treating width as proportional to $1/n$ leads to "double the data, halve the bar"; width goes as $1/\sqrt n$, so it takes four times the data (Problem 4).

> **Warning:** Using the normal 1.96 with an estimated $s$ at small $n$ gives intervals that are too narrow, and multiplying by $s$ without dividing by $\sqrt{n}$ gives intervals 5 times too wide. Use $t_{0.975,\,n-1}$ times $s/\sqrt n$ (Problem 6).

The interval-specific pitfalls are about procedure. The Wald interval at $k = 0$ gives $[0, 0]$, certainty from ten trials; Wilson stays open at $k = 0$ (Problem 7). Nearest-rank quantiles instead of interpolated ones produce a bootstrap interval that disagrees with numpy and Go; interpolate with type 7 (Problem 8). And reading the 95% as a probability about this one interval ("95% chance the mean is in here") misattributes it; the 95% belongs to the procedure (Problem 9).

## Where this leads

The variance of a sum with covariances is Problem 3 of [problem set B](10-problem-set-b.md), and [LLN, CLT, confidence intervals, bootstrap](05-lln-clt-confidence-intervals-bootstrap.md) implements `mean_ci`, `wilson_interval`, `quantile`, and `bootstrap_ci`, which are Problems 6, 7, and 8 in code. [Problem set D](12-problem-set-d.md) reads the same sampling distributions as p-values. In *AI Engineering*, [The evaluation harness and the model zoo](../../ai-engineering/12-objectives-and-adaptation/07-lm-evaluation-harness-and-the-model-zoo.md) reports every metric with its interval, together with what the interval does and does not claim, and [Sequence metrics: exact match, BLEU, chrF](../../ai-engineering/10-attention/05-sequence-metrics.md) bootstraps intervals over sentences for BLEU.

### Further reading

- Blitzstein and Hwang, *Introduction to Probability* (free), ch. 10 "Inequalities and limit theorems".
- Wasserman, *All of Statistics*, ch. 5 to 8.

## Key terms

- **Standard error**: $\sigma/\sqrt{n}$, the standard deviation of a sample mean.
- **Chebyshev's inequality**: $P(|Y - EY| \ge \varepsilon) \le \operatorname{Var}(Y)/\varepsilon^2$, a distribution-free tail bound derived from Markov's inequality.
- **Weak law of large numbers**: the sample mean converges in probability to the true mean.
- **Central limit theorem**: the standardized sample mean is approximately standard normal for large $n$, whatever the distribution of each draw.
- **Confidence interval**: a procedure that, over repeated samples, produces intervals covering the fixed true parameter at the stated rate.
- **Wilson interval**: a confidence interval for a proportion that stays sensible at 0 or $n$ successes.
- **Type 7 quantile**: the linearly interpolated quantile $s_i + (h - i)(s_{i+1} - s_i)$ with $h = (B - 1)q$, numpy's default.

## Summary

- The mean of $n$ independent draws has the same mean and $1/n$ of the variance, so its error shrinks like $1/\sqrt{n}$; halving an error bar costs four times the data (Problems 1 and 4).
- Chebyshev's inequality turns that variance into the weak law of large numbers in three lines (Problems 2 and 5).
- The central limit theorem lets a normal table answer questions about sums of non-normal variables, such as coin flips (Problem 3).
- A confidence interval is a statement about the procedure, not about one interval: 95% of the intervals it would produce cover the true value (Problem 9).
- Student's t, Wilson, and the type 7 percentile rule are each a few lines of arithmetic, the same numbers the limit-theorems section computes (Problems 6, 7, and 8).
