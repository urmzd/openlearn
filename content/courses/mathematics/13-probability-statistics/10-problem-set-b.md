# Probability problem set, part b: joint distributions, covariance, MLE

Once a system has text, it turns counts into probabilities everywhere. [MLE, Laplace, absolute discounting](03-mle-laplace-absolute-discounting.md) estimates a categorical distribution from token counts and smooths it so unseen words keep some mass. [The Unigram language model tokenizer](../../ai-engineering/07-tokenizers/04-unigram-lm.md) in *AI Engineering* trains a tokenizer whose training data never says which segmentation produced a word, so it re-estimates piece probabilities from expected counts, again and again: that is EM. The statistical language models of [n-gram language model with interpolated modified Kneser-Ney](../../ai-engineering/08-statistical-language-models/01-ngram-kneser-ney.md) are tables of conditional distributions over pairs of tokens, which is a joint distribution read one row at a time. And when you evaluate, per-token losses on one document are correlated, so the variance of their mean is not $\sigma^2/n$.

This problem set covers the tools behind all of that: joint and conditional distributions, covariance, and maximum likelihood, including one EM step of the kind the Unigram trainer runs. It is a pen-and-paper set with 15 exact answers (counting lettered parts) and one proof. A short review and two examples by hand come first; then the problems, each followed by a collapsed worked answer with the final result, the derivation, and the common wrong answers and why they are wrong. Try each problem before opening its answer. Read it alongside [MLE, Laplace, absolute discounting](03-mle-laplace-absolute-discounting.md) and before the Unigram tokenizer; [problem set C](11-problem-set-c.md) follows the limit-theorems section.

The set builds on [problem set A](09-problem-set-a.md) (axioms, conditioning, expectation, and variance of one variable) and on the joint-distribution and estimation material in the [chapter opening](README.md).

> **Learning objectives**
> - Read marginals, conditionals, and independence off a joint table.
> - Compute covariance and use it in the variance of a linear combination, and explain why zero covariance is not independence.
> - Derive maximum-likelihood estimates for Bernoulli, exponential, and categorical models, and apply Laplace smoothing.
> - Run one EM step by hand: expected counts from posteriors, then the categorical MLE.
> - Prove that count over total maximizes the categorical log-likelihood.

## Notation

The set uses these symbols.

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

## Joint, marginal, conditional

Two discrete random variables on the same outcome have a **joint** mass function $p_{X,Y}$, a table that sums to 1. Summing a row gives the **marginal** of one variable; dividing a cell by its column sum gives the **conditional** of the row variable given the column. $X$ and $Y$ are **independent** when $p_{X,Y}(x, y) = p_X(x)\,p_Y(y)$ for **every** cell, so one cell that fails is enough to refute it. A bigram model is exactly a table of conditionals $P(\text{next} \mid \text{current})$.

## Covariance and the variance of a sum

$\operatorname{Cov}(X, Y) = E[XY] - E[X]E[Y]$ is positive when the variables tend to be large together and negative when one is large while the other is small. Expanding the square of $aX + bY - E[aX + bY]$ gives
$$\operatorname{Var}(aX + bY) = a^2 \sigma_X^2 + b^2 \sigma_Y^2 + 2ab\,\operatorname{Cov}(X, Y).$$
Independence makes the covariance 0 (because $E[XY] = E[X]E[Y]$), but not the other way round: covariance only sees **linear** dependence, and $Y = X^2$ with $X$ symmetric around 0 is completely determined by $X$ yet has zero covariance with it (Problem 4). For $n$ losses with common variance $\sigma^2$ and pairwise covariance $c$, the variance of their mean is $\sigma^2/n + (n - 1)c/n$, which does not go to 0 when $c > 0$: correlated evaluation tokens give wider error bars than independent ones.

## Maximum likelihood

Given data $x_1, \dots, x_n$ drawn independently from $p_\theta$, the **likelihood** $L(\theta) = \prod_i p_\theta(x_i)$ is how probable the data is under $\theta$, and the **MLE** $\hat\theta$ maximizes it. Take logs first: products become sums, and $\ln$ is increasing, so the maximizer is the same. For a smooth one-parameter model, solve $\ell'(\theta) = 0$ and check that it is a maximum. Three models recur.

- **Bernoulli.** $k$ successes in $n$ trials: $\ell(p) = k \ln p + (n - k)\ln(1 - p)$, and $\ell'(p) = 0$ at the success fraction (Problem 6).
- **Exponential.** Gaps $x_i$ with density $\lambda e^{-\lambda x}$: $\ell(\lambda) = n \ln \lambda - \lambda \sum_i x_i$ (Problem 7).
- **Categorical.** Counts $c_k$: $\ell(p) = \sum_k c_k \ln p_k$ subject to $\sum_k p_k = 1$; the maximizer is $p_k = c_k / N$ (Problem 9 is the proof). An outcome never seen gets probability 0, which is why the estimation section smooths: **Laplace** (add-one) uses $(c_k + 1)/(N + V)$, one pseudo-count for every entry of the vocabulary, seen or not (Problem 5).

## Hidden data and EM

Sometimes the data that would make counting easy is hidden. A Unigram tokenizer sees the word "ab" but not whether it was produced as $a + b$ or as $ab$. EM replaces each count by its **expected** value under the current model: weight each possible segmentation by its posterior probability given the word (its probability divided by the sum over all segmentations), count the pieces in each, and add up. Then take the categorical MLE of those expected counts, count over total. Repeating never lowers the likelihood of the observed words.

## Two examples by hand

These are siblings of Problems 1, 2, and 8, not problems of the set.

**Joint table.** Let $X, Y \in \{0, 1\}$ with $P(0, 0) = 1/2$, $P(0, 1) = 0$, $P(1, 0) = 1/4$, $P(1, 1) = 1/4$. The marginals are $P(X = 1) = 1/2$ and $P(Y = 1) = 1/4$. The conditional is $P(Y = 1 \mid X = 1) = (1/4)/(1/2) = 1/2$. Independence fails at the cell $(0, 1)$: $0 \ne (1/2)(1/4)$. $E[XY] = 1 \cdot 1 \cdot 1/4 = 1/4$ and $E[X] E[Y] = 1/8$, so $\operatorname{Cov}(X, Y) = 1/8 > 0$: $Y$ is 1 only when $X$ is.

**One EM step.** Take a Unigram with pieces $x$, $y$, $xy$ at probabilities $1/2$, $1/4$, $1/4$, and one observed word "xy". The segmentations are $x + y$ with probability $1/2 \cdot 1/4 = 1/8$, and $xy$ with $1/4$. Their sum is $3/8$, so the posteriors are $1/3$ and $2/3$. Expected counts: $x$ is used once in the first, so $1/3$; $y$ also $1/3$; $xy$ is $2/3$. The total is $4/3$, and the M-step gives $P(xy) = (2/3)/(4/3) = 1/2$, up from $1/4$, and $P(x) = P(y) = 1/4$. Keep it as the exact fraction $1/2$: every answer in this set is exact.

## The problems

Probabilities are exact fractions ($3/8$, not 0.375), and lettered parts are answered separately. $\sigma_X^2$ is written `vx`, $\sigma_Y^2$ is `vy`, the covariance is $c$, and the sum of the gaps in Problem 7 is $s$. The tag after each problem says what kind of answer is expected.

### Joint distributions and covariance

The first four problems work with one joint table and then with a pair of variables that are dependent but uncorrelated.

**Problem 1.** $X \in \{0, 1\}$ and $Y \in \{0, 1, 2\}$ have this joint mass function $P(X = x, Y = y)$:

| | $Y = 0$ | $Y = 1$ | $Y = 2$ |
|---|---|---|---|
| $X = 0$ | $1/8$ | $1/4$ | $1/8$ |
| $X = 1$ | $1/4$ | $1/8$ | $1/8$ |

(a) $P(Y = 1)$. (b) $P(X = 1 \mid Y = 0)$. `[number]` (c) Are $X$ and $Y$ independent? `[bool]`

<details><summary>Answer</summary>

**(a) $3/8$. (b) $2/3$. (c) No.** (a) A marginal sums the joint over the other variable: $P(Y = 1) = 1/4 + 1/8 = 3/8$. (b) A conditional renormalizes a column: $P(Y = 0) = 1/8 + 1/4 = 3/8$, so $P(X = 1 \mid Y = 0) = (1/4)/(3/8) = 2/3$. (c) Independence needs every cell to factor. $P(X = 0) = 1/8 + 1/4 + 1/8 = 1/2$ and $P(Y = 0) = 3/8$, so independence would need $P(0, 0) = 3/16$, but the table says $1/8$; one failing cell is enough. Common mistakes: $1/4$ for (a), reading one cell instead of the column sum; $1/4$ for (b), the joint $P(X = 1, Y = 0)$ rather than the conditional; $1/2$ for (b), the marginal $P(X = 1)$.

</details>

**Problem 2.** For the same table: (a) $E[XY]$. (b) $\operatorname{Cov}(X, Y)$. `[number]`

<details><summary>Answer</summary>

**(a) $3/8$. (b) $-1/16$.** (a) Only cells with $X = 1$ and $Y \ne 0$ contribute: $E[XY] = 1 \cdot 1 \cdot \frac18 + 1 \cdot 2 \cdot \frac18 = \frac38$. (b) $E[X] = 1/2$ and $E[Y] = 0 \cdot \frac38 + 1 \cdot \frac38 + 2 \cdot \frac14 = \frac78$, so $\operatorname{Cov}(X, Y) = \frac38 - \frac12 \cdot \frac78 = \frac{6 - 7}{16} = -\frac1{16}$. The sign says $Y$ tends to be a little smaller when $X = 1$ (row $X = 1$ puts its weight on $Y = 0$). Common mistakes: $7/16$ for (a), which is $E[X]E[Y]$ and equals $E[XY]$ only when the covariance is zero; $+1/16$ for (b), the subtraction reversed; $3/8$ for (b), $E[XY]$ without centering.

</details>

**Problem 3.** $X$ and $Y$ have variances $\sigma_X^2$ (`vx`) and $\sigma_Y^2$ (`vy`) and covariance $c$; $a$ and $b$ are constants. Give $\operatorname{Var}(aX + bY)$. `[expr in a, b, vx, vy, c]`

<details><summary>Answer</summary>

**$a^2\sigma_X^2 + b^2\sigma_Y^2 + 2abc$** (`a^2*vx + b^2*vy + 2*a*b*c`). Let $\tilde X = X - E X$ and $\tilde Y = Y - E Y$. Then $\operatorname{Var}(aX + bY) = E[(a\tilde X + b\tilde Y)^2] = a^2 E[\tilde X^2] + 2ab\,E[\tilde X\tilde Y] + b^2 E[\tilde Y^2]$, which is the formula. The cross term is why correlated per-token losses average down slower than independent ones. Common mistakes: dropping the covariance term (true only when $c = 0$); $a\,\sigma_X^2 + b\,\sigma_Y^2 + 2abc$, forgetting that scaling a variable by $a$ scales its variance by $a^2$; and $abc$ instead of $2abc$, counting the cross term once when the square has two of them.

</details>

**Problem 4.** $X$ is uniform on $\{-1, 0, 1\}$ and $Y = X^2$. (a) $\operatorname{Cov}(X, Y)$. `[number]` (b) Are $X$ and $Y$ independent? `[bool]`

<details><summary>Answer</summary>

**(a) $0$. (b) No.** (a) $E[X] = 0$ by symmetry and $E[XY] = E[X^3] = \frac13(-1 + 0 + 1) = 0$, so $\operatorname{Cov}(X, Y) = 0 - 0 \cdot E[Y] = 0$. (b) $Y$ is a function of $X$: $P(X = 0, Y = 1) = 0$, while $P(X = 0)\,P(Y = 1) = \frac13 \cdot \frac23 = \frac29$. Zero covariance rules out only a linear relation. Common mistakes: $2/3$ for (a), which is $E[Y]$; "yes" for (b), reading zero covariance as independence.

</details>

### Maximum likelihood

The remaining five problems estimate parameters: smoothing a categorical, two calculus MLEs, one EM step, and the proof that count over total is optimal.

**Problem 5.** A tokenized corpus holds $N = 10$ tokens: "the" 4 times, "cat" 3, "sat" 2, "mat" 1. The vocabulary also has a fifth word, "dog", never seen, so $V = 5$. (a) The maximum-likelihood estimate of $p(\text{cat})$. (b) The Laplace (add-one) estimate of $p(\text{dog})$ over the $V = 5$ words. `[number]`

<details><summary>Answer</summary>

**(a) $3/10$. (b) $1/15$.** (a) The categorical MLE is count over total: $3/10$. (b) Laplace adds one pseudo-count to every vocabulary entry, so the denominator grows by $V$: $(0 + 1)/(10 + 5) = 1/15$. The five smoothed estimates $5/15, 4/15, 3/15, 2/15, 1/15$ sum to 1. Common mistakes: $1/4$ for (a), one over the number of distinct seen words; $4/15$ for (a), the Laplace estimate of "cat" when the MLE was asked; $0$ for (b), the unsmoothed MLE; $1/14$ for (b), a vocabulary of the 4 seen words only; $1/11$ for (b), adding one to $N$ only, after which the estimates no longer sum to 1.

</details>

**Problem 6.** $n$ independent Bernoulli($p$) trials give $k$ successes ($0 < k < n$). (a) Give the log-likelihood $\ell(p) = \ln P(\text{data} \mid p)$ of the observed sequence. `[expr in k, n, p]` (b) Give the $p$ that maximizes it. `[expr in k, n]`

<details><summary>Answer</summary>

**(a) $\ell(p) = k\ln p + (n - k)\ln(1 - p)$. (b) $\hat p = k/n$.** (a) The observed sequence has probability $p^k(1 - p)^{n - k}$ (independent trials multiply); its log is the sum. (b) $\ell'(p) = \frac kp - \frac{n - k}{1 - p} = 0$ gives $k(1 - p) = (n - k)p$, so $p = k/n$. It is a maximum because $\ell''(p) = -\frac k{p^2} - \frac{n - k}{(1 - p)^2} < 0$. Common mistakes: $k \ln p$ for (a), dropping the failures (its maximizer is $p = 1$); $p^k(1-p)^{n-k}$ for (a), the likelihood rather than its log; $(k + 1)/(n + 2)$ for (b), Laplace's rule of succession, a smoothed estimate rather than the MLE; and $n/k$.

</details>

**Problem 7.** The gaps between requests at your gateway are modeled as independent Exponential($\lambda$) with density $\lambda e^{-\lambda x}$ for $x \ge 0$. You observe $n$ gaps whose sum is $s$. Give the maximum-likelihood estimate of $\lambda$. `[expr in n, s]`

<details><summary>Answer</summary>

**$\hat\lambda = n/s$.** The log-likelihood is $\ell(\lambda) = \sum_i (\ln\lambda - \lambda x_i) = n\ln\lambda - \lambda s$. Setting $\ell'(\lambda) = n/\lambda - s = 0$ gives $\lambda = n/s$, and $\ell''(\lambda) = -n/\lambda^2 < 0$ makes it a maximum. The rate is the reciprocal of the mean gap $s/n$; the arrival schedule of the [open-loop load generator](../../ai-engineering/23-model-serving/11-load-generator.md) is fitted this way. Common mistakes: $s/n$, the mean gap instead of the rate; $(n - 1)/s$, the unbiased estimator of $\lambda$, which is not the MLE.

</details>

**Problem 8.** A Unigram tokenizer has three pieces with probabilities $P(a) = 1/4$, $P(b) = 1/4$, $P(ab) = 1/2$, and the training corpus is the single word "ab". It can be segmented as $a + b$ or as $ab$, and a segmentation's probability is the product of its pieces' probabilities. One EM step computes each piece's expected count (each segmentation's count of the piece, weighted by the segmentation's posterior probability given the word), then sets each probability to its expected count divided by the total expected count. Give (a) the expected count of $a$ and (b) the new $P(ab)$. `[number]`

<details><summary>Answer</summary>

**(a) $1/9$. (b) $4/5$.** E-step: $P(a + b) = \frac14 \cdot \frac14 = \frac1{16}$ and $P(ab) = \frac12 = \frac8{16}$, total $\frac9{16}$, so the posteriors given the word are $\frac19$ and $\frac89$. Expected counts: $a$ appears once in $a + b$, so $\frac19$; $b$ also $\frac19$; $ab$ is $\frac89$. The total expected count is $\frac{10}9$. M-step: $P(ab) = \frac{8/9}{10/9} = \frac45$ (and $P(a) = P(b) = \frac1{10}$; the three sum to 1). Common mistakes: $1/4$ for (a), the prior probability of $a$ rather than its expected count; $1/2$ for (a), weighting both segmentations equally instead of by their posteriors; $8/9$ for (b), the expected count without dividing by the total $10/9$; $1/2$ for (b), the old probability, with no M-step.

</details>

**Problem 9.** Counts $c_1, \dots, c_V \ge 0$ with $N = c_1 + \dots + c_V > 0$ are observed from a categorical distribution $p = (p_1, \dots, p_V)$. Prove that the log-likelihood $\sum_k c_k \ln p_k$ is maximized over all probability vectors $p$ by $p_k = c_k / N$. `[proof]`

<details><summary>Answer</summary>

**Claim.** Let $c_1, \dots, c_V \ge 0$ with $N = c_1 + \dots + c_V > 0$. Over all probability vectors $p$ ($p_k \ge 0$, $\sum_k p_k = 1$), $\ell(p) = \sum_k c_k \ln p_k$ is maximized by $\hat p_k = c_k / N$, with the convention $0 \ln 0 = 0$.

**Zero counts.** If some $p_k = 0$ with $c_k > 0$, then $\ell(p) = -\infty$, so such a $p$ is never a maximizer; assume $p_k > 0$ wherever $c_k > 0$. Terms with $c_k = 0$ contribute 0 whatever $p_k$ is, and $\hat p_k = 0$ there.

**Proof (Gibbs' inequality).** Let $S = \{k : c_k > 0\}$. Then
$$\ell(p) = N \sum_{k \in S} \hat p_k \ln p_k = N \sum_{k \in S} \hat p_k \ln \hat p_k - N \sum_{k \in S} \hat p_k \ln \frac{\hat p_k}{p_k} = -N\bigl(H(\hat p) + \mathrm{KL}(\hat p \,\Vert\, p)\bigr),$$
where $H(\hat p)$ does not depend on $p$. Since $\ln t \le t - 1$ with equality only at $t = 1$,
$$-\mathrm{KL}(\hat p \Vert p) = \sum_{k \in S} \hat p_k \ln \frac{p_k}{\hat p_k} \le \sum_{k \in S} \hat p_k \Bigl(\frac{p_k}{\hat p_k} - 1\Bigr) = \sum_{k \in S} p_k - 1 \le 0,$$
with equality only if $p_k = \hat p_k$ for every $k \in S$ and $p_k = 0$ off $S$. So $\ell(p) \le -N H(\hat p) = \ell(\hat p)$, with equality exactly at $p = \hat p$: count over total is the unique maximizer.

A Lagrange-multiplier argument also works: stationarity of $\sum_k c_k \ln p_k - \mu(\sum_k p_k - 1)$ gives $c_k/p_k = \mu$, so $p_k \propto c_k$ and $\mu = N$; it must then be argued that this stationary point is the maximum (the objective is concave). A complete proof states the claim, handles zero counts, and proves the maximum is attained only at $c/N$. The model proof is also in [`course/solutions/solve/S-M07b/q9.md`](../../../../course/solutions/solve/S-M07b/q9.md).

</details>

## What can go wrong

Most wrong answers here come from reading the wrong piece of a table or dropping a term. Reading one cell as a marginal quotes $P(Y = 1)$ as a joint probability; sum the column instead (Problem 1a). Confusing a joint with a conditional answers $P(X = 1 \mid Y = 0)$ as $P(X = 1, Y = 0)$; divide the cell by its column sum (Problem 1b). Assuming $E[XY] = E[X]E[Y]$ without independence reports zero covariance for dependent variables; compute $E[XY]$ cell by cell (Problem 2). And reading zero covariance as independence treats a deterministic relation as noise; check a cell of the joint, not the covariance (Problem 4b).

> **Warning:** Dropping or halving the covariance term makes error bars on correlated losses too narrow. Expand the square and keep both cross terms, $2abc$ (Problem 3).

The estimation pitfalls are just as specific. Applying Laplace over the seen words only, or adding one to $N$ only, gives unseen words the wrong mass and leaves the smoothed rows of the estimation section not summing to 1; add one per vocabulary entry and $V$ to the denominator (Problem 5b). Differentiating the likelihood instead of its log, or dropping the failures, gives a derivative that cannot be solved by hand, or $\hat p = 1$; differentiate the full log-likelihood (Problem 6). Reporting the mean gap instead of the rate inverts arrival rates in the load generator; solve $\ell'(\lambda) = 0$ for the rate, $n/s$ (Problem 7). And skipping the M-step normalization in EM leaves probabilities that do not sum to 1 after one step; divide each expected count by the total, $10/9$ in Problem 8b.

## Where this leads

This set extends the single-variable expectation and variance of [problem set A](09-problem-set-a.md), and [MLE, Laplace, absolute discounting](03-mle-laplace-absolute-discounting.md) implements `mle` and `laplace`, which are Problems 5 and 9 in code. In *AI Engineering*, [The Unigram language model tokenizer](../../ai-engineering/07-tokenizers/04-unigram-lm.md) computes the expected counts of Problem 8 by forward-backward and then takes count over total (Problem 9); [n-gram language model with interpolated modified Kneser-Ney](../../ai-engineering/08-statistical-language-models/01-ngram-kneser-ney.md) builds bigram and n-gram tables that are joint and conditional distributions over tokens (Problem 1); and the [open-loop load generator](../../ai-engineering/23-model-serving/11-load-generator.md) uses Poisson arrivals with exponential gaps, whose rate Problem 7 fits from a trace. [Problem set C](11-problem-set-c.md) builds the law of large numbers and confidence intervals on Problem 3's variance of a sum.

### Further reading

- Blitzstein and Hwang, *Introduction to Probability* (free), ch. 7 "Joint Distributions".
- Wasserman, *All of Statistics*, ch. 9 "Parametric Inference".

## Key terms

- **Joint distribution**: the table $P(X = x, Y = y)$ over pairs of values; it determines marginals and conditionals.
- **Marginal**: the distribution of one variable, obtained by summing the joint over the other.
- **Covariance**: $E[XY] - E[X]E[Y]$, a measure of linear co-movement; zero does not imply independence.
- **Likelihood**: $\prod_i p_\theta(x_i)$, the probability of the observed data as a function of the parameter.
- **Maximum-likelihood estimate**: the parameter value that maximizes the (log-)likelihood.
- **Laplace smoothing**: adding one pseudo-count per vocabulary entry, $(c_k + 1)/(N + V)$.
- **EM**: expectation-maximization; replace hidden counts by expected counts under the current model, then take the MLE, and repeat.

## Summary

- A joint distribution holds everything: marginals are its row and column sums, conditionals are a renormalized row or column, and independence means every cell factors (Problem 1).
- Covariance $E[XY] - E[X]E[Y]$ measures linear co-movement; it is the cross term in $\operatorname{Var}(aX + bY)$, and zero covariance does not mean independence (Problems 2, 3, and 4).
- The maximum-likelihood estimate of a categorical is count over total; Laplace smoothing adds one to every vocabulary entry, seen or not (Problems 5 and 9).
- MLE by calculus: write the log-likelihood, differentiate, set to zero; the Bernoulli MLE is the success fraction and the exponential rate is the reciprocal of the mean gap (Problems 6 and 7).
- When the data is hidden (which segmentation produced a word), EM replaces counts by expected counts and then takes the same MLE; one step of the Unigram trainer fits on a page (Problem 8).
