# S-M07d problems: hypothesis tests, regression, queueing and Little's law, balls into bins

Answer every question in `solve/S-M07d.toml` (written by `ol start S-M07d`).
The tag after each question is its answer type: `[number]` a value (exact
where the question says so: fractions such as `79/2048`, `E` for $e$,
`exp(-1)`), `[expr]` a formula in the named variables (write $\lambda$ as
`lam`, $\mu$ as `mu`), `[set]` a set such as `{1, 3}`, `[choice]` one letter,
and `[proof]` a file `solve/S-M07d/q9.md` graded against its rubric.

### Hypothesis tests

**q1.** Model B minus model A on three prompts gives the paired differences $d = (2, 1, 3)$. Give the exact two-sided sign-flip p-value: the fraction of the $2^3$ sign vectors $s$ with $|\sum_i s_i d_i| \ge |\sum_i d_i|$. `[number, exact]`

**q2.** Two unpaired samples, $a = \{3, 5\}$ and $b = \{1, 2\}$, with statistic $T = |\bar a - \bar b|$. Give the exact two-sided permutation p-value: the fraction of the ways to split the pooled values $\{1, 2, 3, 5\}$ into two groups of two whose $T$ is at least the observed one. `[number, exact]`

**q3.** Two classifiers on the same eval set: A alone is right on $b_{01} = 2$ examples, B alone on $b_{10} = 10$. Give the exact two-sided McNemar p-value $\min\bigl(1, 2\sum_{i=0}^{k}\binom{n}{i}2^{-n}\bigr)$, $n = b_{01} + b_{10}$, $k = \min(b_{01}, b_{10})$. `[number, exact]`

**q4.** Five comparisons have p-values $(p_1, \dots, p_5) = (0.03, 0.002, 0.04, 0.012, 0.3)$. Which hypotheses does Holm's procedure reject at family-wise level $\alpha = 0.05$? Answer with the set of their indices. `[set]`

### Regression

**q5.** Fit the least-squares line $y = \beta_0 + \beta_1 x$ to the points $(0, 1), (1, 3), (2, 4), (3, 8)$. (a) $\beta_1$. `[number, exact]` (b) $\beta_0$. `[number, exact]`

**q6.** A fitted logistic regression has log-odds $\log\frac{p}{1-p} = -2 + 0.5x$. By what factor do the odds of $y = 1$ multiply when $x$ increases by 2? `[number, exact]`

**q7.** For one example $(x, y)$ with $x \in \mathbb{R}$, $y \in \{0, 1\}$, and $p = \sigma(wx) = 1/(1 + e^{-wx})$, the loss is $\ell(w) = -[y\log p + (1 - y)\log(1 - p)]$. Give $\frac{d\ell}{dw}$. `[expr in w, x, y]`

**q8.** A classifier scores three positives $0.8, 0.5, 0.35$ and three negatives $0.6, 0.35, 0.1$. Give its ROC-AUC, counting a tied pair as one half. `[number, exact]`

**q9.** Prove that the penalized logistic loss $L(w) = -\sum_i [y_i\log p_i + (1 - y_i)\log(1 - p_i)] + \frac{\lambda}{2}\lVert w\rVert^2$, $p_i = \sigma(x_i \cdot w)$, $\lambda \ge 0$, is convex, and say why that makes every stationary point a global minimum and Newton's step well defined when $\lambda > 0$. `[proof]`

### Queueing and Little's law

**q10.** A gateway receives $\lambda = 40$ requests per second and the mean time a request spends inside it is $W = 0.25$ s. By Little's law, how many requests are inside on average? `[number, exact]`

**q11.** A single-server queue with Poisson arrivals at $\lambda = 8$ per second and exponential service at rate $\mu = 10$ per second (M/M/1). (a) The utilization $\rho = \lambda/\mu$. `[number, exact]` (b) The mean number in the system $L = \rho/(1 - \rho)$. `[number, exact]` (c) The mean time in the system $W$, by Little's law. `[number, exact]`

**q12.** Give the mean time in an M/M/1 system as a formula in the arrival rate $\lambda$ and the service rate $\mu > \lambda$. `[expr in lam, mu]`

**q13.** Keep the service rate $\mu$ fixed. By what factor is the mean time in an M/M/1 system larger at utilization $\rho = 0.9$ than at $\rho = 0.5$? `[number, exact]`

### Balls into bins

**q14.** Throw $n$ balls independently and uniformly into $n$ bins. Give the expected number of empty bins. `[expr in n]`

**q15.** As $n \to \infty$, what fraction of the bins in q14 is empty? `[number, exact]`

**q16.** Throw $m$ balls independently and uniformly into $n$ bins. Give the expected number of pairs of balls that land in the same bin. `[expr in m, n]`

**q17.** $n$ requests go to $n$ servers. Each request either picks one server uniformly at random, or samples two servers and joins the less loaded one (the power of two choices). With high probability the maximum load grows like: `[choice]`
(a) $\ln n/\ln\ln n$ with two choices, the same as with one.
(b) $\ln\ln n/\ln 2$ with two choices, against $\ln n/\ln\ln n$ with one.
(c) $\sqrt{n}$ with one choice, $\ln n$ with two.
(d) a constant with two choices.
