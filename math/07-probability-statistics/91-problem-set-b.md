<!-- ol:module S-M07b -->
# Solve set: joint and covariance, MLE

## Overview

| | |
|---|---|
| **Module** | `S-M07b` · solve · none · Pass 3 · 3 to 4 h |
| **You build** | answers in `solve/S-M07b.toml` (15 checked by SymPy) and 1 proof in `solve/S-M07b/q9.md` (self-graded against its rubric) |
| **Contract** | none: a pen and paper set |
| **Tests** | `course/solve/S-M07b/key.toml` (hidden): typed answers plus reject canaries; the problems are in `course/solve/S-M07b/problems.md` and in section 4 |
| **Needs** | `S-M07a` (axioms, conditioning, expectation, variance). Reading: [`M07.2` MLE, Laplace, absolute discounting](02-mle-laplace-absolute-discounting.md) and the [Probability and Statistics topic](README.md), joint distributions and estimation sections |
| **Used by** | no call site (a solve set). Do it alongside `M07.2` and before `L1.4` (Unigram EM: q8 is one of its steps by hand, q9 is its M-step); part `S-M07c` follows `M07.4` |
| **Milestone** | `MS-P3` (the Pass 3 gate runs `ol check` on every solve part of the pass) |
| **Optional depth** | Blitzstein and Hwang, *Introduction to Probability* (free), ch. 7 "Joint Distributions"; Wasserman, *All of Statistics*, ch. 9 "Parametric Inference" |

## Key Takeaways

- A joint distribution holds everything: marginals are its row and column sums, conditionals are a renormalized row or column, and independence means every cell factors (q1).
- Covariance $E[XY] - E[X]E[Y]$ measures linear co-movement; it is the cross term in $\operatorname{Var}(aX + bY)$, and zero covariance does not mean independence (q2, q3, q4).
- The maximum-likelihood estimate of a categorical is count over total; Laplace smoothing adds one to every vocabulary entry, seen or not (q5, q9).
- MLE by calculus: write the log-likelihood, differentiate, set to zero; the Bernoulli MLE is the success fraction and the exponential rate is the reciprocal of the mean gap (q6, q7).
- When the data is hidden (which segmentation produced a word), EM replaces counts by expected counts and then takes the same MLE; one step of `L1.4`'s trainer fits on a page (q8).

## How to work this chapter

```bash
ol start S-M07b             # writes solve/S-M07b.toml and the proof file
ol check S-M07b             # SymPy checks the answers, then asks the proof rubric (y/n)
ol check S-M07b --regrade   # ask the rubric again after you change the proof
```

---

## 1. Why now

Pass 3 turns counts into probabilities everywhere. `M07.2` estimates a categorical distribution from token counts and smooths it so unseen words keep some mass. `L1.4` trains a Unigram tokenizer whose training data never says which segmentation produced a word, so it re-estimates piece probabilities from expected counts, again and again (EM). The language models of `L2` are tables of conditional distributions over pairs of tokens, which is a joint distribution read one row at a time. And when you evaluate, per-token losses on one document are correlated, so the variance of their mean is not $\sigma^2/n$. This set gives you the tools behind all of it: joint and conditional distributions, covariance, and maximum likelihood, including the one EM step you will implement.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $p_{X,Y}(x, y)$ | joint mass function, $P(X = x, Y = y)$ | real |
| $p_X(x) = \sum_y p_{X,Y}(x, y)$ | marginal of $X$ | real |
| $P(X = x \mid Y = y)$ | conditional, $p_{X,Y}(x, y) / p_Y(y)$ for $p_Y(y) > 0$ | real |
| $\operatorname{Cov}(X, Y)$ | covariance, $E[(X - E X)(Y - E Y)] = E[XY] - E[X]E[Y]$ | real |
| $\sigma_X^2$ (`vx`) | variance of $X$, $\operatorname{Cov}(X, X)$ | real |
| $\rho$ | correlation, $\operatorname{Cov}(X, Y) / (\sigma_X \sigma_Y)$ | real in $[-1, 1]$ |
| $\theta$ | the parameter of a model, $p_\theta$ | real or vector |
| $L(\theta)$, $\ell(\theta)$ | likelihood $\prod_i p_\theta(x_i)$ and log-likelihood $\sum_i \ln p_\theta(x_i)$ | real |
| $\hat\theta$ | the maximum-likelihood estimate, $\arg\max_\theta \ell(\theta)$ | real or vector |
| $c_k$, $N$, $V$ | count of outcome $k$, total count, vocabulary size | integers |

### 2.1 Joint, marginal, conditional

Two discrete random variables on the same outcome have a **joint** mass function $p_{X,Y}$, a table that sums to 1. Summing a row gives the **marginal** of one variable; dividing a cell by its column sum gives the **conditional** of the row variable given the column. $X$ and $Y$ are **independent** when $p_{X,Y}(x, y) = p_X(x)\,p_Y(y)$ for **every** cell; one cell that fails is enough to refute it. A bigram model (`L2.1`) is exactly a table of conditionals $P(\text{next} \mid \text{current})$.

### 2.2 Covariance and the variance of a sum

$\operatorname{Cov}(X, Y) = E[XY] - E[X]E[Y]$ is positive when the variables tend to be large together and negative when one is large while the other is small. Expanding the square of $aX + bY - E[aX + bY]$ gives
$$\operatorname{Var}(aX + bY) = a^2 \sigma_X^2 + b^2 \sigma_Y^2 + 2ab\,\operatorname{Cov}(X, Y).$$
Independence makes the covariance 0 (because $E[XY] = E[X]E[Y]$), but not the other way round: covariance only sees **linear** dependence, and $Y = X^2$ with $X$ symmetric around 0 is completely determined by $X$ with zero covariance (q4). For $n$ losses with common variance $\sigma^2$ and pairwise covariance $c$, the variance of their mean is $\sigma^2/n + (n - 1)c/n$, which does not go to 0 when $c > 0$: correlated evaluation tokens give wider error bars than independent ones.

### 2.3 Maximum likelihood

Given data $x_1, \dots, x_n$ drawn independently from $p_\theta$, the **likelihood** $L(\theta) = \prod_i p_\theta(x_i)$ is how probable the data is under $\theta$, and the **MLE** $\hat\theta$ maximizes it. Take logs first: products become sums, and $\ln$ is increasing, so the maximizer is the same. For a smooth one-parameter model, solve $\ell'(\theta) = 0$ and check it is a maximum.

- **Bernoulli.** $k$ successes in $n$ trials: $\ell(p) = k \ln p + (n - k)\ln(1 - p)$, and $\ell'(p) = 0$ at the success fraction (q6).
- **Exponential.** Gaps $x_i$ with density $\lambda e^{-\lambda x}$: $\ell(\lambda) = n \ln \lambda - \lambda \sum_i x_i$ (q7).
- **Categorical.** Counts $c_k$: $\ell(p) = \sum_k c_k \ln p_k$ subject to $\sum_k p_k = 1$; the maximizer is $p_k = c_k / N$ (q9, the proof). An outcome never seen gets probability 0, which is why `M07.2` smooths: **Laplace** (add-one) uses $(c_k + 1)/(N + V)$, one pseudo-count for every entry of the vocabulary, seen or not (q5).

### 2.4 Hidden data and EM

Sometimes the data that would make counting easy is hidden. A Unigram tokenizer (`L1.4`) sees the word "ab" but not whether it was produced as $a + b$ or as $ab$. EM replaces each count by its **expected** value under the current model: weight each possible segmentation by its posterior probability given the word (its probability divided by the sum over all segmentations), count the pieces in each, and add up. Then take the categorical MLE of those expected counts, count over total. Repeating never lowers the likelihood of the observed words.

## 3. Worked example by hand

This is a sibling of q1, q2, and q8, not one of the graded problems.

**Joint table.** $X, Y \in \{0, 1\}$ with $P(0, 0) = 1/2$, $P(0, 1) = 0$, $P(1, 0) = 1/4$, $P(1, 1) = 1/4$. Marginals: $P(X = 1) = 1/2$, $P(Y = 1) = 1/4$. Conditional: $P(Y = 1 \mid X = 1) = (1/4)/(1/2) = 1/2$. Independence fails at the cell $(0, 1)$: $0 \ne (1/2)(1/4)$. $E[XY] = 1 \cdot 1 \cdot 1/4 = 1/4$, $E[X] E[Y] = 1/8$, so $\operatorname{Cov}(X, Y) = 1/8 > 0$: $Y$ is 1 only when $X$ is.

**One EM step.** A Unigram with pieces $x$, $y$, $xy$ at probabilities $1/2$, $1/4$, $1/4$, and one observed word "xy". Segmentations: $x + y$ with probability $1/2 \cdot 1/4 = 1/8$, and $xy$ with $1/4$. Their sum is $3/8$, so the posteriors are $1/3$ and $2/3$. Expected counts: $x$ is used once in the first, so $1/3$; $y$ also $1/3$; $xy$ is $2/3$. The total is $4/3$, and the M-step gives $P(xy) = (2/3)/(4/3) = 1/2$, up from $1/4$, and $P(x) = P(y) = 1/4$. In `solve/` this would be `answer = "1/2"`; `answer = "0.5"` fails as inexact.

## 4. The problem set

Write each answer in `solve/S-M07b.toml`; lettered parts are their own tables:

```toml
[q1.a]
answer = "3/8"
[q3]
answer = "a^2*vx + b^2*vy + 2*a*b*c"
[q6.b]
answer = "k/n"
[q9]
proof = "S-M07b/q9.md"
```

Probabilities are exact fractions; `0.375` fails a question marked exact. Write $\sigma_X^2$ as `vx`, $\sigma_Y^2$ as `vy`, the covariance as `c`, and the sum of the gaps in q7 as `s`.

<!-- ol:problems S-M07b -->

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

<!-- /ol:problems -->

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| Reading one cell as a marginal | $P(Y = 1)$ quoted as a joint probability | q1 (canary 1/4) |
| Confusing a joint with a conditional | $P(X = 1 \mid Y = 0)$ answered as $P(X = 1, Y = 0)$ | q1 (canary 1/4) |
| $E[XY] = E[X]E[Y]$ without independence | covariance reported as 0 for dependent variables | q2 (canary 7/16) |
| Dropping or halving the covariance term | error bars on correlated losses too narrow | q3 (canaries without $2abc$, with $abc$) |
| Zero covariance read as independence | a deterministic relation treated as noise | q4 (canary true) |
| Laplace over the seen words only, or added to $N$ only | unseen words get the wrong mass; `M07.2`'s rows do not sum to 1 | q5 (canaries 1/14 and 1/11) |
| The likelihood instead of its log, or the failures dropped | a derivative that cannot be solved by hand, or $\hat p = 1$ | q6 (canaries) |
| The mean gap instead of the rate | arrival rates inverted in the load generator | q7 (canary s/n) |
| Skipping the M-step normalization in EM | probabilities that do not sum to 1 after one step | q8 (canary 8/9) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `S-M07a` | axioms, conditioning, expectation and variance of one variable |
| Back | `M07.2` | `mle` and `laplace` are q5 and q9 in code (reading) |
| Forward | `L1.4` | Unigram EM: expected counts by forward-backward (q8), then count over total (q9) |
| Forward | `L2.1` | bigram and n-gram tables are joint and conditional distributions over tokens (q1) |
| Forward | `S-M07c` | the law of large numbers and confidence intervals build on q3's variance of a sum |
| Forward | `load.01` | Poisson arrivals with exponential gaps: q7 fits their rate from a trace |
