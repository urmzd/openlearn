<!-- ol:module S-M07d -->
# Solve set: hypothesis tests, regression, queueing and Little's law, balls into bins

## Overview

| | |
|---|---|
| **Module** | `S-M07d` · solve · none · Pass 5 · 3 to 4 h |
| **You build** | answers in `solve/S-M07d.toml` (19 checked by SymPy) and 1 proof in `solve/S-M07d/q9.md` (self-graded against its rubric) |
| **Contract** | none: a pen and paper set |
| **Tests** | `course/solve/S-M07d/key.toml` (hidden): typed answers plus reject canaries; the problems are in `course/solve/S-M07d/problems.md` and in section 4 |
| **Needs** | `S-M07c` (sampling distributions). Reading: [`M07.5` hypothesis tests](05-hypothesis-tests.md), [`M07.7` logistic regression, ROC-AUC, calibration](07-logistic-regression-roc-auc-calibration.md), and the [Probability and Statistics topic](README.md), hypothesis testing and regression sections |
| **Used by** | no call site (a solve set). Do it alongside `M07.5` and `M07.7`: q1 to q4 are their tests by hand, q5 to q9 the regression behind the policy head; q10 to q17 are the queueing and load-balancing arithmetic `load.01` and the gateway's router (`gw.05`) rely on |
| **Milestone** | `MS-P5` (the Pass 5 gate runs `ol check` on every solve part of the pass) |
| **Optional depth** | Wasserman, *All of Statistics*, ch. 10 and 13; Harchol-Balter, *Performance Modeling and Design of Computer Systems*, ch. 6 (Little's law) and 14 (M/M/1); Mitzenmacher and Upfal, *Probability and Computing*, ch. 5 (balls and bins) and 17 (the power of two choices) |

## Key Takeaways

- An exact permutation p-value is a count over a finite set of relabellings; with three pairs, even a unanimous win has $p = 1/4$ (q1, q2).
- Holm stops at the first p-value above its bar and still rejects more than Bonferroni (q4).
- A logistic coefficient is a log odds ratio, and the loss gradient is "prediction minus label, times the input" (q6, q7); that loss is convex, which is why IRLS converges (q9).
- Little's law $L = \lambda W$ holds for any stable system; in M/M/1 the time in system $1/(\mu - \lambda)$ explodes as utilization approaches 1 (q10 to q13).
- Throwing $n$ balls into $n$ bins leaves about $1/e$ of them empty, and asking two bins instead of one shrinks the fullest bin from $\ln n/\ln\ln n$ to $\ln\ln n/\ln 2$ (q14 to q17).

## How to work this chapter

```bash
ol start S-M07d             # writes solve/S-M07d.toml and the proof file
ol check S-M07d             # SymPy checks the answers, then asks the proof rubric (y/n)
ol check S-M07d --regrade   # ask the rubric again after you change the proof
```

---

## 1. Why now

Pass 5 is where your system starts making claims and serving load at the same time. `M07.5` turns "model B is better" into a p-value and `M07.7` fits and scores the gateway's policy classifier; both are short functions whose results are easy to misread. In Pass 7 the gateway routes requests across engine replicas and the load generator measures latency, and two pieces of probability decide whether those numbers make sense: Little's law, which links throughput, latency, and concurrency, and the balls-into-bins arithmetic behind random and two-choice load balancing. This set works each idea once by hand, so the code and the dashboards that follow are transcriptions of things you have already computed.

## 2. Principles

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

### 2.1 Tests

A p-value is the probability under $H_0$ of a statistic at least as extreme as observed. Exact permutation p-values count the relabellings that $H_0$ makes equally likely: $2^n$ sign vectors for pairs, $\binom{N}{n_a}$ splits for two samples. McNemar uses only the discordant pairs, a fair coin under $H_0$. Holm sorts the p-values and rejects the $k$-th smallest while $p_{(k)} \le \alpha/(m - k)$.

### 2.2 Regression

Least squares minimizes $\sum_i (y_i - \beta_0 - \beta_1 x_i)^2$; setting both partial derivatives to 0 gives $\beta_1 = S_{xy}/S_{xx}$ with $S_{xy} = \sum (x_i - \bar x)(y_i - \bar y)$, $S_{xx} = \sum (x_i - \bar x)^2$, and $\beta_0 = \bar y - \beta_1 \bar x$. In logistic regression the log-odds are linear, so a coefficient $w$ multiplies the odds by $e^{w}$ per unit of its feature, and $\frac{d}{dz}[-y\log\sigma(z) - (1-y)\log(1-\sigma(z))] = \sigma(z) - y$. The AUC is the fraction of (positive, negative) pairs ranked correctly, ties counting one half.

### 2.3 Queueing

**Little's law**: in any system where items arrive at long-run rate $\lambda$ and stay a mean time $W$, the mean number inside is $L = \lambda W$, whatever the arrival or service distributions. The proof is an area argument: the area under "number inside over time" counts each item's time inside once. In the **M/M/1** queue (Poisson arrivals, exponential service, one server), the number in the system is geometric: $P(N = k) = (1 - \rho)\rho^k$, so $L = \rho/(1 - \rho)$ and, by Little, $W = L/\lambda = 1/(\mu - \lambda)$.

### 2.4 Balls into bins

With $m$ balls thrown uniformly into $n$ bins, a given bin is missed by every ball with probability $(1 - 1/n)^m$, and linearity of expectation adds such indicators over bins or over pairs of balls with no independence needed. With $m = n$, $(1 - 1/n)^n \to e^{-1}$. The most loaded bin holds about $\ln n/\ln\ln n$ balls with one random choice; Azar, Broder, Karlin, and Upfal (1994) showed that sampling two bins and taking the emptier gives $\ln\ln n/\ln 2 + O(1)$, an exponential improvement.

## 3. Worked example by hand

These are siblings of q2, q11, and q14, not graded problems.

**A two-sample p-value.** $a = \{4, 6\}$, $b = \{1, 3\}$: $T = |5 - 2| = 3$. The six ways to pick group $a$ from $\{1, 3, 4, 6\}$ give $|\bar a - \bar b|$ of $\{1,3\}: 3$, $\{1,4\}: 2$, $\{1,6\}: 0$, $\{3,4\}: 0$, $\{3,6\}: 2$, $\{4,6\}: 3$. Two of six reach 3: $p = 1/3$. In `solve/` this would be `answer = "1/3"`.

**An M/M/1 queue.** $\lambda = 3$, $\mu = 4$ per second: $\rho = 3/4$, $L = (3/4)/(1/4) = 3$, $W = L/\lambda = 1$ s, and indeed $1/(\mu - \lambda) = 1$. The service time alone is $1/4$ s; queueing quadrupled it.

**Empty bins.** $n = 3$ balls into 3 bins: a bin is empty with probability $(2/3)^3 = 8/27$, so the expected number of empty bins is $3 \cdot 8/27 = 8/9$. Enumerating the 27 assignments agrees: 6 hit every bin (0 empty), 18 leave one empty, 3 leave two empty, $(18 + 6)/27 = 8/9$.

## 4. The problem set

Write each answer in `solve/S-M07d.toml`; lettered parts are their own tables:

```toml
[q1]
answer = "1/4"
[q4]
answer = "{1, 3}"
[q5.a]
answer = "2"
[q12]
answer = "1/(mu - lam)"
[q17]
answer = "a"
[q9]
proof = "S-M07d/q9.md"
```

Write $\lambda$ as `lam` and $\mu$ as `mu`; exact fractions such as `79/2048` are welcome, $e$ is `E` or `exp(1)`.

<!-- ol:problems S-M07d -->

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

<!-- /ol:problems -->

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| A one-sided count for a two-sided question | p halved | q1, q2, q3 (canaries) |
| The add-one Monte Carlo estimate where the exact p was asked | 1/3 instead of 1/4 | q1 (canary 1/3) |
| Bonferroni, no correction, or a Holm that does not stop | the wrong set of discoveries | q4 (canaries) |
| Regressing $x$ on $y$, or a line through two points | a different slope | q5 (canaries) |
| The change in log-odds reported as the odds factor | 1 instead of $e$ | q6 (canary) |
| The likelihood's gradient instead of the loss's, or a missing chain-rule factor | Newton walks uphill, or a wrong scale | q7 (canaries) |
| Ties counted as wins or losses | 7/9 or 2/3 instead of 13/18 | q8 (canaries) |
| $\lambda/W$ instead of $\lambda W$; the utilization reported as $L$ | 160 requests, or 4/5 | q10, q11 (canaries) |
| Service time or queue wait reported as the time in system | 1/10 or 2/5 s | q11 (canaries) |
| The limit $n/e$ given for the exact count | approximate where exact was asked | q14 (canary) |
| Ordered pairs or self-pairs counted | $m^2/n$ collisions | q16 (canary) |
| Believing two choices only shave a constant | answer (a) | q17 (canary) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `S-M07c` | sampling distributions and intervals, read as tests here |
| Back | `M07.5` | `paired_permutation_test`, `permutation_test`, `mcnemar`, `holm` are q1 to q4 in code (reading) |
| Back | `M07.7` | IRLS, `roc_auc`, and the convexity of q9 (reading) |
| Forward | `L6.7` | the zoo's comparison table: q1 to q4 at scale |
| Forward | `load.01` | the load generator reports concurrency, throughput, and latency, tied together by Little's law |
| Forward | `gw.05` | routing across replicas: random choice against least-loaded of two |
