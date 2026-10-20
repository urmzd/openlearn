# Probability problem set, part d: hypothesis tests, regression, queueing, balls into bins

There comes a point where a system starts making claims and serving load at the same time. [Hypothesis tests: paired and two-sample permutation, McNemar, Holm](06-hypothesis-tests.md) turns "model B is better" into a p-value, and [Logistic regression (IRLS), ROC-AUC, calibration (ECE)](08-logistic-regression-roc-auc-calibration.md) fits and scores the gateway's policy classifier; both are short functions whose results are easy to misread. Later, the gateway routes requests across engine replicas and a load generator measures latency, and two pieces of probability decide whether those numbers make sense: Little's law, which links throughput, latency, and concurrency, and the balls-into-bins arithmetic behind random and two-choice load balancing.

This problem set works each idea once by hand, so the code and the dashboards that follow are transcriptions of things you have already computed. It is a pen-and-paper set with 19 answers (counting lettered parts) and one proof. A short review and three examples by hand come first; then the problems, each followed by a collapsed worked answer with the final result, the derivation, and the common wrong answers and why they are wrong. Try each problem before opening its answer. Read it alongside the hypothesis-tests and logistic-regression sections: Problems 1 to 4 are their tests by hand, Problems 5 to 9 the regression behind the policy head, and Problems 10 to 17 the queueing and load-balancing arithmetic that the [load generator](../../ai-engineering/23-model-serving/11-load-generator.md) and the gateway's [router](../../ai-engineering/25-ai-gateway/04-routing.md) rely on.

The set builds on the sampling distributions of [problem set C](11-problem-set-c.md) and on the hypothesis-testing and regression material in the [chapter opening](README.md).

> **Learning objectives**
> - Compute exact permutation, sign-flip, and McNemar p-values by enumeration, and apply Holm's correction.
> - Fit a least-squares line by hand, interpret a logistic coefficient as a log odds ratio, and derive the logistic loss gradient.
> - Compute an AUC with ties and prove that the penalized logistic loss is convex.
> - Apply Little's law and the M/M/1 formulas to relate arrival rate, utilization, concurrency, and latency.
> - Use linearity of expectation for empty bins and collisions, and state the power-of-two-choices bound.

## Notation

The set uses these symbols.

| Symbol | Meaning | Type / shape |
|---|---|---|
| $d_i$, $s_i$ | paired differences and random signs $\pm 1$ | reals |
| $b_{01}$, $b_{10}$ | discordant counts (only A right, only B right) | integers |
| $p_{(k)}$, $m$, $\alpha$ | sorted p-values, family size, level | reals, integer |
| $\beta_0$, $\beta_1$ | intercept and slope of a least-squares line | reals |
| $\sigma(z)$ | the sigmoid $1/(1 + e^{-z})$ | function |
| $\lambda$ (`lam`), $\mu$ (`mu`) | arrival rate and service rate | positive reals |
| $\rho = \lambda/\mu$ | utilization | in $[0, 1)$ |
| $L$, $W$ | mean number in the system and mean time in the system | reals |
| $n$, $m$ | bins and balls | positive integers |

## Tests

A p-value is the probability under $H_0$ of a statistic at least as extreme as observed. Exact permutation p-values count the relabellings that $H_0$ makes equally likely: $2^n$ sign vectors for pairs, $\binom{N}{n_a}$ splits for two samples. McNemar uses only the discordant pairs, each a fair coin under $H_0$. Holm sorts the p-values and rejects the $k$-th smallest while $p_{(k)} \le \alpha/(m - k)$ (with $k$ counted from 0).

## Regression

Least squares minimizes $\sum_i (y_i - \beta_0 - \beta_1 x_i)^2$; setting both partial derivatives to 0 gives $\beta_1 = S_{xy}/S_{xx}$ with $S_{xy} = \sum (x_i - \bar x)(y_i - \bar y)$, $S_{xx} = \sum (x_i - \bar x)^2$, and $\beta_0 = \bar y - \beta_1 \bar x$. In logistic regression the log-odds are linear, so a coefficient $w$ multiplies the odds by $e^{w}$ per unit of its feature, and $\frac{d}{dz}[-y\log\sigma(z) - (1-y)\log(1-\sigma(z))] = \sigma(z) - y$. The AUC is the fraction of (positive, negative) pairs ranked correctly, ties counting one half.

## Queueing

The second half of the set leaves statistics for systems. **Little's law** says that in any system where items arrive at long-run rate $\lambda$ and stay a mean time $W$, the mean number inside is $L = \lambda W$, whatever the arrival or service distributions. The proof is an area argument: the area under "number inside over time" counts each item's time inside once. In the **M/M/1** queue (Poisson arrivals, exponential service, one server), the number in the system is geometric, $P(N = k) = (1 - \rho)\rho^k$, so $L = \rho/(1 - \rho)$ and, by Little, $W = L/\lambda = 1/(\mu - \lambda)$.

## Balls into bins

With $m$ balls thrown uniformly into $n$ bins, a given bin is missed by every ball with probability $(1 - 1/n)^m$, and linearity of expectation adds such indicators over bins or over pairs of balls with no independence needed. With $m = n$, $(1 - 1/n)^n \to e^{-1}$. The most loaded bin holds about $\ln n/\ln\ln n$ balls with one random choice; Azar, Broder, Karlin, and Upfal (1994) showed that sampling two bins and taking the emptier gives $\ln\ln n/\ln 2 + O(1)$, an exponential improvement.

## Three examples by hand

These are siblings of Problems 2, 11, and 14, not problems of the set.

**A two-sample p-value.** Take $a = \{4, 6\}$ and $b = \{1, 3\}$, so $T = |5 - 2| = 3$. The six ways to pick group $a$ from $\{1, 3, 4, 6\}$ give $|\bar a - \bar b|$ of $\{1,3\}: 3$, $\{1,4\}: 2$, $\{1,6\}: 0$, $\{3,4\}: 0$, $\{3,6\}: 2$, $\{4,6\}: 3$. Two of six reach 3: $p = 1/3$.

**An M/M/1 queue.** With $\lambda = 3$ and $\mu = 4$ per second, $\rho = 3/4$, $L = (3/4)/(1/4) = 3$, and $W = L/\lambda = 1$ s, and indeed $1/(\mu - \lambda) = 1$. The service time alone is $1/4$ s; queueing quadrupled it.

**Empty bins.** Throw $n = 3$ balls into 3 bins. A bin is empty with probability $(2/3)^3 = 8/27$, so the expected number of empty bins is $3 \cdot 8/27 = 8/9$. Enumerating the 27 assignments agrees: 6 hit every bin (0 empty), 18 leave one empty, 3 leave two empty, and $(18 + 6)/27 = 8/9$.

## The problems

$\lambda$ is written `lam` and $\mu$ is `mu` in formulas; answers are exact ($79/2048$, $e$), not decimals. Lettered parts are answered separately, and the tag after each problem says what kind of answer is expected.

### Hypothesis tests

The first four problems compute each test of the hypothesis-tests section by exhaustive enumeration.

**Problem 1.** Model B minus model A on three prompts gives the paired differences $d = (2, 1, 3)$. Give the exact two-sided sign-flip p-value: the fraction of the $2^3$ sign vectors $s$ with $|\sum_i s_i d_i| \ge |\sum_i d_i|$. `[number, exact]`

<details><summary>Answer</summary>

**$1/4$.** $\sum_i d_i = 6$. Of the $2^3 = 8$ sign vectors, only $(+,+,+)$ and $(-,-,-)$ give $|\sum_i s_i d_i| = 6$; every other pattern flips at least one term and lands at 0, 2, or 4. So $p = 2/8 = 1/4$: with three pairs, even a unanimous win cannot reach $p < 0.05$. Common mistakes: $1/8$, the one-sided count (only $+++$); $1/3$, the add-one Monte Carlo estimate $(c + 1)/(R + 1)$, which the hypothesis-tests section uses when it samples, not the exact enumeration asked here.

</details>

**Problem 2.** Two unpaired samples, $a = \{3, 5\}$ and $b = \{1, 2\}$, with statistic $T = |\bar a - \bar b|$. Give the exact two-sided permutation p-value: the fraction of the ways to split the pooled values $\{1, 2, 3, 5\}$ into two groups of two whose $T$ is at least the observed one. `[number, exact]`

<details><summary>Answer</summary>

**$1/3$.** Observed $T = |4 - 1.5| = 2.5$. The six ways to choose group $a$ from $\{1, 2, 3, 5\}$ give $|\bar a - \bar b|$: $\{1,2\}: 2.5$, $\{1,3\}: 1.5$, $\{1,5\}: 0.5$, $\{2,3\}: 0.5$, $\{2,5\}: 1.5$, $\{3,5\}: 2.5$. Two of six reach 2.5, so $p = 2/6 = 1/3$. Common mistakes: $1/6$, the one-sided count ($\bar a - \bar b \ge 2.5$ only); $1/4$.

</details>

**Problem 3.** Two classifiers on the same eval set: A alone is right on $b_{01} = 2$ examples, B alone on $b_{10} = 10$. Give the exact two-sided McNemar p-value $\min\bigl(1, 2\sum_{i=0}^{k}\binom{n}{i}2^{-n}\bigr)$, $n = b_{01} + b_{10}$, $k = \min(b_{01}, b_{10})$. `[number, exact]`

<details><summary>Answer</summary>

**$79/2048$** ($\approx 0.0386$). $n = 12$, $k = 2$. The tail is $\sum_{i=0}^{2}\binom{12}{i} = 1 + 12 + 66 = 79$ out of $2^{12} = 4096$, doubled for two sides: $2 \cdot 79/4096 = 79/2048$. Under the null, each discordant example is a fair coin between A and B, which is why only the $b_{01} + b_{10}$ disagreements matter. Common mistakes: $79/4096$, the tail not doubled; $2(1 + 12)/4096$, a tail that stops at $k - 1$ and leaves out the observed count.

</details>

**Problem 4.** Five comparisons have p-values $(p_1, \dots, p_5) = (0.03, 0.002, 0.04, 0.012, 0.3)$. Which hypotheses does Holm's procedure reject at family-wise level $\alpha = 0.05$? Answer with the set of their indices. `[set]`

<details><summary>Answer</summary>

**$\{2, 4\}$.** Sort: $p_2 = 0.002$, $p_4 = 0.012$, $p_1 = 0.03$, $p_3 = 0.04$, $p_5 = 0.3$. Holm compares the $j$-th smallest with $\alpha/(m - j + 1)$: $0.002 \le 0.05/5 = 0.01$ (reject 2); $0.012 \le 0.05/4 = 0.0125$ (reject 4); $0.03 > 0.05/3 \approx 0.0167$: stop, and keep every remaining hypothesis, including $p_3$. Common mistakes: $\{2\}$, Bonferroni (every p against $0.01$), valid but weaker; $\{1, 2, 3, 4\}$, no correction; $\{2, 3, 4\}$, a step-down that does not stop at the first failure: once $p_1$ fails its bar, every larger p-value is kept.

</details>

### Regression

The next five problems are the regression behind the policy head: a least-squares line, odds, the loss gradient, an AUC with a tie, and the convexity proof.

**Problem 5.** Fit the least-squares line $y = \beta_0 + \beta_1 x$ to the points $(0, 1), (1, 3), (2, 4), (3, 8)$. (a) $\beta_1$. `[number, exact]` (b) $\beta_0$. `[number, exact]`

<details><summary>Answer</summary>

**(a) $11/5$. (b) $7/10$.** $\bar x = 3/2$, $\bar y = 4$. With deviations $x - \bar x = (-\frac32, -\frac12, \frac12, \frac32)$ and $y - \bar y = (-3, -1, 0, 4)$: $S_{xy} = \frac92 + \frac12 + 0 + 6 = 11$ and $S_{xx} = \frac94 + \frac14 + \frac14 + \frac94 = 5$. (a) $\beta_1 = S_{xy}/S_{xx} = 11/5$. (b) The line passes through $(\bar x, \bar y)$: $\beta_0 = 4 - \frac{11}5 \cdot \frac32 = \frac7{10}$. Common mistakes: $26/11 = S_{yy}/S_{xy}$, the reciprocal of the slope from regressing $x$ on $y$; $7/3$, the slope through the two end points; $1$ for (b), the first $y$; $4$ for (b), $\bar y$.

</details>

**Problem 6.** A fitted logistic regression has log-odds $\log\frac{p}{1-p} = -2 + 0.5x$. By what factor do the odds of $y = 1$ multiply when $x$ increases by 2? `[number, exact]`

<details><summary>Answer</summary>

**$e$.** The log-odds grow by $0.5 \times 2 = 1$, so the odds $p/(1 - p) = e^{\text{log-odds}}$ multiply by $e^1 \approx 2.718$. A logistic coefficient is a log odds ratio per unit of $x$. Common mistakes: $e^{1/2}$, the factor for one unit; $1$, the change in log-odds reported as if it were the odds factor.

</details>

**Problem 7.** For one example $(x, y)$ with $x \in \mathbb{R}$, $y \in \{0, 1\}$, and $p = \sigma(wx) = 1/(1 + e^{-wx})$, the loss is $\ell(w) = -[y\log p + (1 - y)\log(1 - p)]$. Give $\frac{d\ell}{dw}$. `[expr in w, x, y]`

<details><summary>Answer</summary>

**$\frac{d\ell}{dw} = (\sigma(wx) - y)\,x$** (`(1/(1 + exp(-w*x)) - y)*x`). Let $z = wx$. Using $\ln p = -\ln(1 + e^{-z})$ and $\ln(1 - p) = -z - \ln(1 + e^{-z})$, $\ell = \ln(1 + e^{-z}) + (1 - y)z$, so $d\ell/dz = -(1 - p) + (1 - y) = p - y$. The chain rule multiplies by $dz/dw = x$. Prediction minus label, times the input. Common mistakes: $(y - \sigma(wx))x$, the gradient of the log-likelihood rather than of the loss (a step along it walks uphill on the loss); $\sigma(wx) - y$, the chain-rule factor $x$ missing.

</details>

**Problem 8.** A classifier scores three positives $0.8, 0.5, 0.35$ and three negatives $0.6, 0.35, 0.1$. Give its ROC-AUC, counting a tied pair as one half. `[number, exact]`

<details><summary>Answer</summary>

**$13/18$.** There are $3 \times 3 = 9$ positive-negative pairs. $0.8$ beats all three negatives (3); $0.5$ beats $0.35$ and $0.1$ (2); $0.35$ beats $0.1$ and ties $0.35$ (1.5). AUC $= 6.5/9 = 13/18$. Common mistakes: $2/3$, the tie counted as a loss; $7/9$, the tie counted as a win.

</details>

**Problem 9.** Prove that the penalized logistic loss $L(w) = -\sum_i [y_i\log p_i + (1 - y_i)\log(1 - p_i)] + \frac{\lambda}{2}\lVert w\rVert^2$, $p_i = \sigma(x_i \cdot w)$, $\lambda \ge 0$, is convex, and say why that makes every stationary point a global minimum and Newton's step well defined when $\lambda > 0$. `[proof]`

<details><summary>Answer</summary>

**Setup.** Examples $(x_i, y_i)$, $i = 1, \dots, n$, with $x_i \in \mathbb R^d$, $y_i \in \{0, 1\}$; $X$ is the $n \times d$ matrix with rows $x_i$; $\sigma(z) = 1/(1 + e^{-z})$, $p_i = \sigma(x_i \cdot w)$, $p$ the vector of the $p_i$; $\lambda \ge 0$.

**Gradient and Hessian.** With $z_i = x_i \cdot w$, each term is $\ell_i = \ln(1 + e^{-z_i}) + (1 - y_i)z_i$, so $d\ell_i/dz_i = p_i - y_i$ (Problem 7) and, since $\sigma' = \sigma(1 - \sigma)$, $d^2\ell_i/dz_i^2 = p_i(1 - p_i)$. By the chain rule,
$$\nabla L(w) = X^\top(p - y) + \lambda w, \qquad H(w) = X^\top S X + \lambda I, \quad S = \operatorname{diag}(p_i(1 - p_i)).$$

**Positive semidefinite.** For every $v \in \mathbb R^d$,
$$v^\top H v = \sum_i p_i(1 - p_i)(x_i \cdot v)^2 + \lambda\lVert v\rVert^2 \ge 0,$$
since $0 < p_i < 1$ makes every weight positive. When $\lambda > 0$ and $v \ne 0$ the last term is strictly positive, so $H$ is positive definite.

**Conclusion.** A twice-differentiable function with a positive semidefinite Hessian everywhere is convex, so $L$ is convex (strictly when $\lambda > 0$). For convex differentiable $L$, $L(u) \ge L(w) + \nabla L(w) \cdot (u - w)$ for all $u$, so a point with $\nabla L(w) = 0$ has $L(u) \ge L(w)$ for every $u$: a global minimum, unique when $\lambda > 0$. Newton's step solves $H\Delta = -\nabla L$; with $\lambda > 0$, $H$ is positive definite, hence invertible, so the system has exactly one solution. This is why the IRLS fitter of [Logistic regression (IRLS), ROC-AUC, calibration (ECE)](08-logistic-regression-roc-auc-calibration.md) converges.

A complete proof derives the gradient and Hessian, shows $v^\top H v \ge 0$ from $0 < p_i < 1$, draws both conclusions, and defines every symbol. The model proof is also in [`course/solutions/solve/S-M07d/q9.md`](../../../../course/solutions/solve/S-M07d/q9.md).

</details>

### Queueing and Little's law

The next four problems move from statistics to systems: concurrency from throughput and latency, and how latency grows as a server fills.

**Problem 10.** A gateway receives $\lambda = 40$ requests per second and the mean time a request spends inside it is $W = 0.25$ s. By Little's law, how many requests are inside on average? `[number, exact]`

<details><summary>Answer</summary>

**$10$.** $L = \lambda W = 40 \times 0.25 = 10$ requests in flight on average. Little's law needs only a stable system, no distributional assumption. Common mistakes: $160 = \lambda/W$; $1/160$.

</details>

**Problem 11.** A single-server queue with Poisson arrivals at $\lambda = 8$ per second and exponential service at rate $\mu = 10$ per second (M/M/1). (a) The utilization $\rho = \lambda/\mu$. `[number, exact]` (b) The mean number in the system $L = \rho/(1 - \rho)$. `[number, exact]` (c) The mean time in the system $W$, by Little's law. `[number, exact]`

<details><summary>Answer</summary>

**(a) $4/5$. (b) $4$. (c) $1/2$ s.** (a) $\rho = 8/10$. (b) $L = \frac{4/5}{1/5} = 4$. (c) $W = L/\lambda = 4/8 = 1/2$ s, which agrees with $1/(\mu - \lambda) = 1/2$. Common mistakes: $5/4 = \mu/\lambda$ for (a); $4/5$ for (b), the utilization reported as the number in system; $16/5 = \rho^2/(1 - \rho)$ for (b), the number waiting in queue without the one in service; $1/10$ for (c), the service time alone; $2/5$ for (c), the queue wait $W - 1/\mu$ without the service.

</details>

**Problem 12.** Give the mean time in an M/M/1 system as a formula in the arrival rate $\lambda$ and the service rate $\mu > \lambda$. `[expr in lam, mu]`

<details><summary>Answer</summary>

**$W = \dfrac{1}{\mu - \lambda}$** (`1/(mu - lam)`). $L = \rho/(1 - \rho) = \lambda/(\mu - \lambda)$, and Little's law gives $W = L/\lambda = 1/(\mu - \lambda)$. Common mistakes: $1/\mu$, the service time with no queueing; $\lambda/(\mu - \lambda)$, which is $L$, not $W$.

</details>

**Problem 13.** Keep the service rate $\mu$ fixed. By what factor is the mean time in an M/M/1 system larger at utilization $\rho = 0.9$ than at $\rho = 0.5$? `[number, exact]`

<details><summary>Answer</summary>

**$5$.** $W = \frac{1}{\mu - \lambda} = \frac{1}{\mu(1 - \rho)}$, so with $\mu$ fixed the ratio is $\frac{1 - 0.5}{1 - 0.9} = 5$. Going from half to 90% busy multiplies latency by five; this is the curve behind keeping a server below saturation. Common mistakes: $9/5$, the ratio of the utilizations; $9$, the ratio of $L = \rho/(1 - \rho)$ values ($9$ against $1$), which compares counts, not times (the arrival rates differ).

</details>

### Balls into bins

The last four problems are the load-balancing arithmetic: empty bins, collisions, and the power of two choices.

**Problem 14.** Throw $n$ balls independently and uniformly into $n$ bins. Give the expected number of empty bins. `[expr in n]`

<details><summary>Answer</summary>

**$n(1 - 1/n)^n$.** A given bin is missed by one ball with probability $1 - 1/n$ and by all $n$ independent balls with probability $(1 - 1/n)^n$. Linearity of expectation adds this indicator over the $n$ bins, with no independence between bins needed. Common mistakes: $(1 - 1/n)^n$, the probability for one bin, not the expected count; $n/e$, the large-$n$ limit, not the exact value.

</details>

**Problem 15.** As $n \to \infty$, what fraction of the bins in Problem 14 is empty? `[number, exact]`

<details><summary>Answer</summary>

**$1/e$** ($\approx 0.368$). The fraction is $(1 - 1/n)^n \to e^{-1}$. Common mistakes: 0; $1 - 1/e$, the occupied fraction.

</details>

**Problem 16.** Throw $m$ balls independently and uniformly into $n$ bins. Give the expected number of pairs of balls that land in the same bin. `[expr in m, n]`

<details><summary>Answer</summary>

**$\dfrac{m(m - 1)}{2n}$** $= \binom m2/n$. Each of the $\binom m2$ unordered pairs of distinct balls lands in the same bin with probability $1/n$ (the second ball matches the first's bin); linearity adds them. This is the birthday problem: collisions appear once $m \approx \sqrt{2n}$. Common mistakes: $m^2/n$, counting ordered pairs and self-pairs; $m/n$, the mean load, not the collisions.

</details>

**Problem 17.** $n$ requests go to $n$ servers. Each request either picks one server uniformly at random, or samples two servers and joins the less loaded one (the power of two choices). With high probability the maximum load grows like: `[choice]`
(a) $\ln n/\ln\ln n$ with two choices, the same as with one.
(b) $\ln\ln n/\ln 2$ with two choices, against $\ln n/\ln\ln n$ with one.
(c) $\sqrt{n}$ with one choice, $\ln n$ with two.
(d) a constant with two choices.

<details><summary>Answer</summary>

**(b).** With one uniform choice the fullest bin holds about $\ln n/\ln\ln n$; with the less loaded of two random choices it holds $\ln\ln n/\ln 2 + O(1)$ (Azar, Broder, Karlin, and Upfal), an exponential improvement. Common mistakes: (a), believing the second choice only shaves a constant; (d), a constant: the bound still grows, just very slowly; (c) has the wrong rates for both.

</details>

## What can go wrong

In the tests, the commonest slip is a one-sided count for a two-sided question, which halves $p$; count both tails, or double the binomial tail (Problems 1, 2, and 3). Quoting the add-one Monte Carlo estimate where the exact $p$ was asked gives 1/3 instead of 1/4; enumerate all $2^3$ sign vectors (Problem 1). Using Bonferroni, no correction, or a Holm that does not stop gives the wrong set of discoveries; sort, compare with $\alpha/(m - j + 1)$, and stop at the first failure (Problem 4).

The regression slips change the answer's scale or sign. Regressing $x$ on $y$, or drawing a line through two points, gives a different slope; use $S_{xy}/S_{xx}$ over all four points (Problem 5). Reporting the change in log-odds as the odds factor gives 1 instead of $e$; exponentiate the change (Problem 6). Counting ties as wins or losses gives 7/9 or 2/3 instead of 13/18; a tie counts one half (Problem 8).

> **Warning:** Taking the likelihood's gradient instead of the loss's makes Newton walk uphill, and dropping the chain-rule factor gives the wrong scale. Differentiate the loss: $(p - y)x$ (Problem 7).

The queueing slips mix up the three quantities of Little's law. Computing $\lambda/W$ instead of $\lambda W$ gives 160 requests, and reporting the utilization as $L$ gives 4/5; use $L = \lambda W$ and $L = \rho/(1 - \rho)$ (Problems 10 and 11). Reporting the service time or the queue wait as the time in system gives 1/10 or 2/5 s; $W = L/\lambda$ is wait plus service (Problem 11c).

In balls into bins, giving the limit $n/e$ where the exact count was asked is approximate where exact was wanted; keep $n(1 - 1/n)^n$ (Problem 14). Counting ordered pairs or self-pairs gives $m^2/n$ collisions; count $\binom m2$ unordered pairs (Problem 16). And believing two choices only shave a constant leads to answer (a); the bound drops from $\ln n/\ln\ln n$ to $\ln\ln n/\ln 2$ (Problem 17).

## Where this leads

This set reads the sampling distributions and intervals of [problem set C](11-problem-set-c.md) as tests. [Hypothesis tests: paired and two-sample permutation, McNemar, Holm](06-hypothesis-tests.md) implements `paired_permutation_test`, `permutation_test`, `mcnemar`, and `holm`, which are Problems 1 to 4 in code, and [Logistic regression (IRLS), ROC-AUC, calibration (ECE)](08-logistic-regression-roc-auc-calibration.md) implements IRLS and `roc_auc` and relies on the convexity of Problem 9. In *AI Engineering*, the comparison table of [The evaluation harness and the model zoo](../../ai-engineering/12-objectives-and-adaptation/07-lm-evaluation-harness-and-the-model-zoo.md) runs Problems 1 to 4 at scale, and the [open-loop load generator](../../ai-engineering/23-model-serving/11-load-generator.md) reports concurrency, throughput, and latency, tied together by Little's law. In *AI Engineering*, [Routing, cascades, and failover](../../ai-engineering/25-ai-gateway/04-routing.md) routes across replicas with random choice against least-loaded of two.

### Further reading

- Wasserman, *All of Statistics*, ch. 10 and 13.
- Harchol-Balter, *Performance Modeling and Design of Computer Systems*, ch. 6 (Little's law) and 14 (M/M/1).
- Mitzenmacher and Upfal, *Probability and Computing*, ch. 5 (balls and bins) and 17 (the power of two choices).

## Key terms

- **Exact permutation p-value**: the fraction of equally likely relabellings under $H_0$ whose statistic is at least as extreme as observed.
- **McNemar test**: an exact binomial test on the discordant pairs of two classifiers evaluated on the same examples.
- **Holm's procedure**: a step-down correction that compares the sorted p-values with $\alpha/m, \alpha/(m-1), \dots$ and stops at the first failure.
- **Log odds ratio**: a logistic coefficient; increasing its feature by one multiplies the odds by $e^{w}$.
- **Little's law**: $L = \lambda W$ for any stable system.
- **Utilization**: $\rho = \lambda/\mu$, the fraction of time a server is busy.
- **Power of two choices**: sampling two bins and joining the emptier, which cuts the maximum load from $\ln n/\ln\ln n$ to $\ln\ln n/\ln 2 + O(1)$.

## Summary

- An exact permutation p-value is a count over a finite set of relabellings; with three pairs, even a unanimous win has $p = 1/4$ (Problems 1 and 2).
- Holm stops at the first p-value above its bar and still rejects more than Bonferroni (Problem 4).
- A logistic coefficient is a log odds ratio, and the loss gradient is "prediction minus label, times the input" (Problems 6 and 7); that loss is convex, which is why IRLS converges (Problem 9).
- Little's law $L = \lambda W$ holds for any stable system; in M/M/1 the time in system $1/(\mu - \lambda)$ explodes as utilization approaches 1 (Problems 10 to 13).
- Throwing $n$ balls into $n$ bins leaves about $1/e$ of them empty, and asking two bins instead of one shrinks the fullest bin from $\ln n/\ln\ln n$ to $\ln\ln n/\ln 2$ (Problems 14 to 17).
