# S-M07c problems: the law of large numbers, the CLT, confidence intervals

Answer every question in `solve/S-M07c.toml` (written by `ol start S-M07c`).
The tag after each question is its answer type: `[expr]` is a formula in the
named variables (write $\sigma^2$ as `s2`, $\varepsilon$ as `eps`, $\mu$ as
`mu`), `[number]` a value (exact where the question says so), `[interval]`
an interval such as `[1/2, 3]`, `[choice]` one letter, and `[proof]` a file
`solve/S-M07c/q5.md` graded against its rubric.

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
