# S-M07b problems: joint distributions, covariance, maximum likelihood

Answer every question in `solve/S-M07b.toml` (written by `ol start S-M07b`).
The tag after each question is its answer type: `[number]` is an exact value
(`3/8`, `-1/16`, `0`), `[expr]` a formula in the named variables
(`k/n`; write $\sigma_X^2$ as `vx`), `[bool]` is `true` or `false`, and
`[proof]` a file `solve/S-M07b/q9.md` graded against its rubric.

### Joint distributions and covariance

**q1.** $X \in \{0, 1\}$ and $Y \in \{0, 1, 2\}$ have this joint mass function $P(X = x, Y = y)$:

| | $Y = 0$ | $Y = 1$ | $Y = 2$ |
|---|---|---|---|
| $X = 0$ | $1/8$ | $1/4$ | $1/8$ |
| $X = 1$ | $1/4$ | $1/8$ | $1/8$ |

(a) $P(Y = 1)$. (b) $P(X = 1 \mid Y = 0)$. `[number]` (c) Are $X$ and $Y$ independent? `[bool]`

**q2.** For the same table: (a) $E[XY]$. (b) $\operatorname{Cov}(X, Y)$. `[number]`

**q3.** $X$ and $Y$ have variances $\sigma_X^2$ (`vx`) and $\sigma_Y^2$ (`vy`) and covariance $c$; $a$ and $b$ are constants. Give $\operatorname{Var}(aX + bY)$. `[expr in a, b, vx, vy, c]`

**q4.** $X$ is uniform on $\{-1, 0, 1\}$ and $Y = X^2$. (a) $\operatorname{Cov}(X, Y)$. `[number]` (b) Are $X$ and $Y$ independent? `[bool]`

### Maximum likelihood

**q5.** A tokenized corpus holds $N = 10$ tokens: "the" 4 times, "cat" 3, "sat" 2, "mat" 1. The vocabulary also has a fifth word, "dog", never seen, so $V = 5$. (a) The maximum-likelihood estimate of $p(\text{cat})$. (b) The Laplace (add-one) estimate of $p(\text{dog})$ over the $V = 5$ words. `[number]`

**q6.** $n$ independent Bernoulli($p$) trials give $k$ successes ($0 < k < n$). (a) Give the log-likelihood $\ell(p) = \ln P(\text{data} \mid p)$ of the observed sequence. `[expr in k, n, p]` (b) Give the $p$ that maximizes it. `[expr in k, n]`

**q7.** The gaps between requests at your gateway are modeled as independent Exponential($\lambda$) with density $\lambda e^{-\lambda x}$ for $x \ge 0$. You observe $n$ gaps whose sum is $s$. Give the maximum-likelihood estimate of $\lambda$. `[expr in n, s]`

**q8.** A Unigram tokenizer (`L1.4`) has three pieces with probabilities $P(a) = 1/4$, $P(b) = 1/4$, $P(ab) = 1/2$, and the training corpus is the single word "ab". It can be segmented as $a + b$ or as $ab$, and a segmentation's probability is the product of its pieces' probabilities. One EM step computes each piece's expected count (each segmentation's count of the piece, weighted by the segmentation's posterior probability given the word), then sets each probability to its expected count divided by the total expected count. Give (a) the expected count of $a$ and (b) the new $P(ab)$. `[number]`

**q9.** Counts $c_1, \dots, c_V \ge 0$ with $N = c_1 + \dots + c_V > 0$ are observed from a categorical distribution $p = (p_1, \dots, p_V)$. Prove that the log-likelihood $\sum_k c_k \ln p_k$ is maximized over all probability vectors $p$ by $p_k = c_k / N$. `[proof]`
