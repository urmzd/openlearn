# Probability and Statistics

Probability is the mathematics of reasoning under uncertainty, and statistics is the craft of drawing conclusions from data. For anyone who builds software, and especially anyone who builds learning systems, both are everywhere. Every weight a neural network starts with is a random draw; every token a language model emits is sampled from a distribution; every count-based model is an estimate; and every claim that "model B is better" is a statement about noise. Beyond machine learning, the same ideas analyse randomized algorithms and hashing, size queues and load balancers, and run A/B tests in production.

The bridge between the two subjects is the central limit theorem: whatever the underlying distribution, sample means are approximately normal, and that is why confidence intervals and hypothesis tests work at all. This chapter walks that bridge in the order a system needs it. It starts with random variables, expectation, and variance, and turns uniform random numbers into normal ones. It then samples categorical distributions three ways, estimates distributions from counts, and uses variance to choose how neural network weights are initialized. With those in hand it builds the inferential layer: the law of large numbers, the central limit theorem, confidence intervals and the bootstrap, and permutation tests for comparing models. It closes with rejection sampling, the exact probability behind speculative decoding, and with logistic regression, ROC-AUC, and calibration, the tools for fitting and trusting a classifier. Four problem sets, placed alongside the sections they support, let you work every idea by hand.

Each build section comes with a reference implementation under [`course/solutions/python/tinyllm/prob/`](../../../../course/solutions/python/tinyllm/prob/), and each problem set has worked answers inline.

> **Learning objectives**
> - Use the axioms, conditional probability, and Bayes' rule, and compute expectations, variances, and covariances of discrete and continuous random variables.
> - Sample any distribution from uniform random numbers: by inverse CDF, by Box-Muller, by Gumbel-max, by alias tables, and by rejection.
> - Estimate distributions by maximum likelihood, smooth them, and fit logistic regression by Newton's method.
> - Explain the law of large numbers and the central limit theorem, and build t, Wilson, and bootstrap intervals.
> - Compare models with permutation tests, McNemar, and Holm's correction, and evaluate a classifier with ROC-AUC and ECE.

## Before you begin

The chapter assumes integration from [Calculus 2](../08-calculus-2/README.md), including the Gaussian integral, and counting and sets from [Discrete Math 1](../11-discrete-math-1/README.md). Several sections draw their uniform random numbers from the PCG32 generator built in [Modular arithmetic, hashing, and PCG32 in Python](../12-discrete-math-2/03-modular-arithmetic-hashing-and-pcg32.md), and the last section reuses LU from [Linear Algebra](../09-linear-algebra/README.md) and Newton's method and the trapezoid rule from [Calculus 1](../07-calculus-1/README.md). The overview below reviews the classical foundations; if they are new to you, read it with one of the textbooks listed under "Further reading" open beside it.

## In this chapter

The sections and problem sets are listed in the order you should read them.

1. [Probability problem set, part a: axioms, Bayes, random variables, Box-Muller](09-problem-set-a.md). The vocabulary for the whole chapter, by hand: axioms, conditioning, Bayes' rule, discrete and continuous random variables, their moments, and the normal.
2. [Random variables, expectation, variance, and normal draws by Box-Muller](01-random-variables-and-the-normal.md). Defines expectation and variance, computes variance stably in two passes, and turns PCG32's uniforms into normals in a fixed draw order.
3. [Categorical sampling: inverse CDF, Gumbel-max, alias method](02-categorical-sampling.md). Three exact ways to draw a token id from a distribution, each a pure function of its uniforms, plus Poisson arrivals from exponential gaps.
4. [MLE, Laplace, absolute discounting](03-mle-laplace-absolute-discounting.md). Why count over total is the maximum-likelihood estimate, why it fails on unseen outcomes, and how add-alpha smoothing and absolute discounting repair it.
5. [Probability problem set, part b: joint distributions, covariance, MLE](10-problem-set-b.md). Joint and conditional tables, covariance and the variance of a sum, maximum likelihood by calculus, and one EM step by hand.
6. [Variance propagation and initialization](04-variance-propagation-and-initialization.md). Derives the variance of a layer's pre-activation and from it the Xavier and Kaiming initializations and residual scaling.
7. [LLN, CLT, confidence intervals, bootstrap](05-lln-clt-confidence-intervals-bootstrap.md). Why error bars shrink as $1/\sqrt{n}$, and how to build t, Wilson, and percentile-bootstrap intervals reproducibly.
8. [Probability problem set, part c: limit theorems and confidence intervals](11-problem-set-c.md). The weak law proved from Chebyshev, CLT tail approximations, the three intervals by hand, and what a 95% interval actually promises.
9. [Hypothesis tests: paired and two-sample permutation, McNemar, Holm](06-hypothesis-tests.md). P-values built by relabelling, exact tests for classifiers on the same examples, and a correction for many comparisons.
10. [Rejection sampling and residual distributions](07-rejection-sampling-and-residual-distributions.md). Von Neumann's sampler and the accept-or-resample rule that makes speculative decoding sample the target distribution exactly.
11. [Logistic regression (IRLS), ROC-AUC, calibration (ECE)](08-logistic-regression-roc-auc-calibration.md). Fits a classifier by Newton's method and measures whether it ranks well and whether its probabilities mean what they say.
12. [Probability problem set, part d: hypothesis tests, regression, queueing, balls into bins](12-problem-set-d.md). The tests and regression by hand, then Little's law, the M/M/1 queue, and the balls-into-bins arithmetic behind load balancing.

## The foundations at a glance

The sections above build specific tools. This overview collects the classical results they rest on, with one worked example for each topic, so that you can check a definition quickly or fill a gap before a section needs it.

### Probability foundations

A **sample space** $\Omega$ is the set of all possible outcomes, and an **event** is a subset of $\Omega$. Probability satisfies three axioms: $P(A) \ge 0$; $P(\Omega) = 1$; and $P(A \cup B) = P(A) + P(B)$ for disjoint $A$ and $B$, extending to countable unions. **Conditional probability** is $P(A \mid B) = P(A \cap B)/P(B)$, and $A$ and $B$ are **independent** if $P(A \cap B) = P(A)P(B)$.

Three theorems follow. **Bayes' theorem**, $P(A \mid B) = P(B \mid A)P(A)/P(B)$, reverses a conditional probability: weight the likelihood $P(B \mid A)$ by the prior $P(A)$ and normalize. It is the engine of Bayesian inference. The **law of total probability**, $P(B) = \sum_i P(B \mid A_i)P(A_i)$ for a partition $\{A_i\}$ of $\Omega$, computes the probability of $B$ by considering all the mutually exclusive ways it can happen. And **inclusion-exclusion**, $P(A \cup B) = P(A) + P(B) - P(A \cap B)$, is the set identity carried over to probabilities.

Here is the classic example. A test for a disease is 99% sensitive ($P(+ \mid \text{disease}) = 0.99$) and 95% specific ($P(- \mid \text{no disease}) = 0.95$), and the prevalence is 1%. By total probability, $P(+) = 0.99 \cdot 0.01 + 0.05 \cdot 0.99 = 0.0099 + 0.0495 = 0.0594$, and by Bayes, $P(\text{disease} \mid +) = 0.0099/0.0594 = 0.167$. Despite 99% sensitivity, the positive predictive value is only 16.7%, because the disease is rare.

### Discrete random variables

A **random variable** is a function $X: \Omega \to \mathbb{R}$ that assigns a number to each outcome. Its **probability mass function** gives $P(X = x)$ for each value, and its **cumulative distribution function** is $F(x) = P(X \le x)$. The **expected value** is $E[X] = \sum_x x\,P(X = x)$, the **variance** is $\operatorname{Var}(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2$, and the **standard deviation** is $\sqrt{\operatorname{Var}(X)}$.

Four discrete distributions cover most needs. A **Bernoulli**($p$) variable takes values in $\{0, 1\}$ with $P(X = 1) = p$, so $E[X] = p$ and $\operatorname{Var}(X) = p(1 - p)$. A **Binomial**($n, p$) counts successes in $n$ independent Bernoulli trials: $P(X = k) = \binom{n}{k}p^k(1-p)^{n-k}$, $E[X] = np$, $\operatorname{Var}(X) = np(1-p)$. A **Geometric**($p$) counts trials until the first success: $P(X = k) = (1-p)^{k-1}p$ and $E[X] = 1/p$. A **Poisson**($\lambda$) counts events in a fixed interval: $P(X = k) = e^{-\lambda}\lambda^k/k!$ and $E[X] = \operatorname{Var}(X) = \lambda$.

**Linearity of expectation**, $E[aX + bY] = aE[X] + bE[Y]$, always holds, even for dependent variables; it is one of the most useful properties in all of probability. For independent $X$ and $Y$, $\operatorname{Var}(X + Y) = \operatorname{Var}(X) + \operatorname{Var}(Y)$: independent fluctuations add in quadrature.

For example, roll a fair die 10 times and let $X$ count the sixes, so $X \sim \text{Binomial}(10, 1/6)$. Then $E[X] = 10/6 = 5/3$, $\operatorname{Var}(X) = 10 \cdot \frac16 \cdot \frac56 = 25/18$, $P(X = 0) = (5/6)^{10} = 0.1615$, and $P(X \ge 3) = 1 - (5/6)^{10} - \binom{10}{1}\frac16(5/6)^9 - \binom{10}{2}(1/6)^2(5/6)^8 = 0.2248$. Linearity also solves the coupon collector problem: the expected number of draws needed to collect all $n$ coupon types is $n H_n = n(1 + 1/2 + \cdots + 1/n)$, a sum of geometric waiting times.

### Continuous random variables

A continuous variable has a **probability density function** $f(x) \ge 0$ with $\int_{-\infty}^{\infty} f(x)\,dx = 1$ and $P(a \le X \le b) = \int_a^b f(x)\,dx$. Its CDF is $F(x) = \int_{-\infty}^{x} f(t)\,dt$, so $f = F'$. Expectation and variance become integrals: $E[X] = \int x f(x)\,dx$ and $\operatorname{Var}(X) = \int (x - \mu)^2 f(x)\,dx$.

The **Uniform**($a, b$) density is $1/(b - a)$ on $[a, b]$, with $E[X] = (a + b)/2$ and $\operatorname{Var}(X) = (b - a)^2/12$. The **Exponential**($\lambda$) density is $\lambda e^{-\lambda x}$ for $x \ge 0$, with $E[X] = 1/\lambda$ and $\operatorname{Var}(X) = 1/\lambda^2$. The **Normal**($\mu, \sigma^2$) density is $\frac{1}{\sigma\sqrt{2\pi}}\exp\bigl(-(x - \mu)^2/(2\sigma^2)\bigr)$, the bell curve, with $E[X] = \mu$ and $\operatorname{Var}(X) = \sigma^2$; standardizing, $Z = (X - \mu)/\sigma \sim N(0, 1)$.

Two facts are worth memorizing. The **68-95-99.7 rule** says a normal puts about 68% of its mass within one standard deviation of the mean, 95% within two, and 99.7% within three: it concentrates tightly. The exponential is **memoryless**, $P(X > s + t \mid X > s) = P(X > t)$: it forgets how long you have been waiting, so the remaining wait has the same distribution regardless. It is the continuous analogue of the geometric distribution.

For example, if scores are Normal(72, 64), so the standard deviation is 8, the fraction above 88 is $P(Z > (88 - 72)/8) = P(Z > 2) = 1 - 0.9772 = 0.0228$, about 2.3%. And if service time is Exponential($\lambda = 0.5$), then $P(X > 4) = e^{-0.5 \cdot 4} = e^{-2} = 0.1353$.

### Joint distributions

A **joint** mass function $P(X = x, Y = y)$ or density $f(x, y)$ gives probabilities for pairs. The **marginal** of $X$ is $f_X(x) = \sum_y f(x, y)$ (or $\int f(x, y)\,dy$), and the **conditional** is $f(y \mid x) = f(x, y)/f_X(x)$. **Covariance**, $\operatorname{Cov}(X, Y) = E[XY] - E[X]E[Y]$, measures linear association, and **correlation**, $\rho(X, Y) = \operatorname{Cov}(X, Y)/(\operatorname{SD}(X)\operatorname{SD}(Y))$, always lies in $[-1, 1]$.

$X$ and $Y$ are independent if and only if $f(x, y) = f_X(x)f_Y(y)$ for all $x, y$: knowing $X$ tells you nothing about $Y$. In general $\operatorname{Var}(X + Y) = \operatorname{Var}(X) + \operatorname{Var}(Y) + 2\operatorname{Cov}(X, Y)$, so positively correlated variables sum to something that fluctuates more than independent ones would. Covariance is bilinear: $\operatorname{Cov}(aX, bY) = ab\operatorname{Cov}(X, Y)$, $\operatorname{Cov}(X + Y, Z) = \operatorname{Cov}(X, Z) + \operatorname{Cov}(Y, Z)$, and $\operatorname{Cov}(X, X) = \operatorname{Var}(X)$.

For example, let $P(0,0) = 0.1$, $P(0,1) = 0.2$, $P(1,0) = 0.3$, $P(1,1) = 0.4$. The marginals are $P(X = 0) = 0.3$, $P(X = 1) = 0.7$, $P(Y = 0) = 0.4$, $P(Y = 1) = 0.6$. Since $P(X = 0)P(Y = 0) = 0.12 \ne 0.1$, the variables are not independent. With $E[X] = 0.7$, $E[Y] = 0.6$, and $E[XY] = 0.4$ (only the cell $(1, 1)$ contributes), $\operatorname{Cov}(X, Y) = 0.4 - 0.7 \cdot 0.6 = -0.02$.

### Limit theorems

The **sample mean** is $\bar X = (X_1 + \cdots + X_n)/n$, and $X_n$ **converges in probability** to $c$ if $P(|X_n - c| > \varepsilon) \to 0$ for every $\varepsilon > 0$. The **weak law of large numbers** says that if $X_1, X_2, \ldots$ are iid with mean $\mu$, then $\bar X \to \mu$ in probability: the average of many independent measurements converges to the true mean, which is why sampling works. The **central limit theorem** says that if the $X_i$ are iid with mean $\mu$ and variance $\sigma^2$, then $(\bar X - \mu)/(\sigma/\sqrt{n})$ converges in distribution to $N(0, 1)$: whatever the shape of the original distribution, the sample mean becomes approximately normal for large $n$, which is why the normal appears everywhere. **Chebyshev's inequality**, $P(|X - \mu| \ge k\sigma) \le 1/k^2$, is a distribution-free bound on spread: weak, but universally applicable.

For example, a factory produces items with mean weight 500 g and standard deviation 10 g. For a sample of $n = 100$, the CLT gives $\bar X \approx N(500, 100/100) = N(500, 1)$, so $P(\bar X > 502) = P(Z > 2) = 0.0228$: about a 2.3% chance. Chebyshev, for any distribution with $\mu = 100$ and $\sigma = 15$, gives $P(|X - 100| \ge 45) \le (15/45)^2 = 1/9$: at most 11.1% of values lie three or more standard deviations out, against 0.3% for a normal. Chebyshev is loose but general.

### Descriptive statistics

The **measures of center** are the mean (arithmetic average), median (middle value), and mode (most frequent value). The **measures of spread** are the range, the interquartile range $\text{IQR} = Q_3 - Q_1$, the variance, and the standard deviation, where $Q_1$, $Q_2$ (the median), and $Q_3$ are the 25th, 50th, and 75th percentiles. A **box plot** displays the minimum, $Q_1$, median, $Q_3$, and maximum, and flags outliers beyond $1.5 \times \text{IQR}$ from the quartiles. A distribution is left-skewed when mean < median, right-skewed when mean > median, and symmetric when they are equal.

For example, for the data 3, 7, 7, 12, 15, 18, 22: the mean is $84/7 = 12$ and the median is 12 (the 4th value); $Q_1 = 7$, $Q_3 = 18$, and $\text{IQR} = 11$. The outlier fences are $7 - 1.5 \cdot 11 = -9.5$ and $18 + 1.5 \cdot 11 = 34.5$, so there are no outliers.

### Inference

A **point estimate** is a single value estimating a parameter, as $\bar X$ estimates $\mu$. A **confidence interval** is an interval $[L, U]$ with $P(L \le \theta \le U) = 1 - \alpha$, for example $\bar X \pm z_{\alpha/2}\,\sigma/\sqrt{n}$. Its interpretation is about the procedure, not any single interval: if you repeat the experiment many times, 95% of the 95% intervals constructed will contain the true parameter.

A test starts from a **null hypothesis** $H_0$, the default claim (say $\mu = \mu_0$), and an **alternative** $H_a$, the claim you seek evidence for ($\mu \ne \mu_0$, or $\mu > \mu_0$). The **test statistic** is $Z = (\bar X - \mu_0)/(\sigma/\sqrt{n})$, or $T = (\bar X - \mu_0)/(s/\sqrt{n})$ when $\sigma$ is estimated. The **p-value** is the probability, assuming $H_0$, of a statistic as extreme as or more extreme than the one observed. In the Neyman-Pearson framework you choose the significance level $\alpha$ before the experiment and reject $H_0$ when $p < \alpha$, so $\alpha$ controls how often you falsely declare an effect. A **type I error** rejects a true $H_0$ (probability $\alpha$); a **type II error** fails to reject a false one (probability $\beta$).

For example, a company claims its batteries last 500 hours, and a sample of 36 gives $\bar X = 490$ and $s = 30$. Testing $H_0: \mu = 500$ against $H_a: \mu < 500$ at $\alpha = 0.05$, $Z = (490 - 500)/(30/6) = -10/5 = -2.0$ and $p = P(Z < -2.0) = 0.0228 < 0.05$, so you reject $H_0$: there is significant evidence that the mean lifetime is below 500 hours. The 95% interval $490 \pm 1.96 \cdot (30/6) = 490 \pm 9.8 = (480.2, 499.8)$ excludes 500, consistent with the rejection.

### Regression

**Simple linear regression** fits $\hat y = b_0 + b_1 x$ with $b_1 = \sum(x_i - \bar x)(y_i - \bar y)/\sum(x_i - \bar x)^2$ and $b_0 = \bar y - b_1\bar x$. A **residual** is $e_i = y_i - \hat y_i$, observed minus predicted. The **coefficient of determination** $R^2 = 1 - SS_\text{res}/SS_\text{tot}$ is the proportion of the variance in $y$ explained by $x$, and the correlation coefficient is $r = b_1(s_x/s_y)$, with $R^2 = r^2$ for simple regression.

The least-squares estimates minimize $\sum e_i^2$: among all lines, the regression line makes the total squared error smallest, which is orthogonal projection from linear algebra, $A^\top A x = A^\top b$. The **Gauss-Markov theorem** adds that under the regression assumptions (linearity, independence, homoscedasticity, no perfect multicollinearity), the least-squares estimators are the best linear unbiased estimators: no unbiased linear estimator has smaller variance.

For example, take the points (1, 2), (2, 4), (3, 5), (4, 4), (5, 5), so $\bar x = 3$ and $\bar y = 4$. Then $\sum(x_i - \bar x)(y_i - \bar y) = (-2)(-2) + (-1)(0) + 0(1) + 1(0) + 2(1) = 6$ and $\sum(x_i - \bar x)^2 = 4 + 1 + 0 + 1 + 4 = 10$, giving $b_1 = 0.6$ and $b_0 = 4 - 0.6 \cdot 3 = 2.2$. The line is $\hat y = 2.2 + 0.6x$, predicting 4.0 at $x = 3$. The residual sum of squares is $(2 - 2.8)^2 + (4 - 3.4)^2 + (5 - 4.0)^2 + (4 - 4.6)^2 + (5 - 5.2)^2 = 0.64 + 0.36 + 1.0 + 0.36 + 0.04 = 2.4$, and $SS_\text{tot} = 4 + 0 + 1 + 0 + 1 = 6$, so $R^2 = 1 - 2.4/6 = 0.6$: the model explains 60% of the variance in $y$.

### Which technique when

The table collects the techniques above by the question each answers.

| Technique | When to use | Key formula or idea |
|---|---|---|
| Bayes' theorem | Reverse a conditional probability | $P(A \mid B) = P(B \mid A)P(A)/P(B)$ |
| Law of total probability | Decompose by cases | $P(B) = \sum_i P(B \mid A_i)P(A_i)$ |
| Linearity of expectation | Expected value of a sum | $E[X + Y] = E[X] + E[Y]$, always |
| CLT approximation | Distribution of a sample mean | $\bar X \approx N(\mu, \sigma^2/n)$ for large $n$ |
| Z-test or t-test | Test a hypothesis about a mean | Compare the statistic with a critical value |
| Confidence interval | Estimate a parameter with uncertainty | $\bar X \pm z\,\sigma/\sqrt{n}$ |
| Least squares | Fit a linear model | Minimize the sum of squared residuals |
| Standardization | Convert to the standard normal | $Z = (X - \mu)/\sigma$ |

### Where these ideas appear in computing

The same handful of ideas turns up across the field. Bayes' theorem drives spam filters (naive Bayes) and probabilistic classification, and conditional probability underlies hidden Markov models and Bayesian networks. Poisson and exponential distributions model network traffic and queues. Linearity of expectation is the main tool for analysing randomized algorithms such as quicksort and hashing. The CLT and hypothesis testing are what make A/B tests and feature-flag experiments trustworthy, and regression is the baseline for machine learning. Variance and covariance appear in portfolio theory, PCA, and feature engineering.

In industry, this is the daily language of experimentation and data science: A/B testing at scale and experiment design at large web companies; demand forecasting, recommendation confidence, and pricing experiments in retail; risk models, option pricing, and stochastic processes in quantitative finance; Bayesian inference and evaluation metrics in machine learning; and Bayesian rating systems such as TrueSkill and Elo for matchmaking in games.

## Further reading

- Grinstead and Snell, *Introduction to Probability* (free): https://math.dartmouth.edu/~prob/prob/prob.pdf. Chapters 1 and 4 cover the foundations, 2 and 5 continuous densities, 5 and 6 discrete random variables, 7 joint distributions, and 8 and 9 the limit theorems. Useful exercises: Ch. 1 #1-8, #13-18; Ch. 2 #1-10; Ch. 4 #1-10, #17-22 (challenge #25-30); Ch. 5 #1-22 (continuous sections #10-20, challenge #25-30); Ch. 6 #1-8 (challenge #15-20); Ch. 7 #1-10, #15-20 (challenge #23-28); Ch. 8 #1-8; Ch. 9 #1-6 (challenge #10-14).
- OpenStax, *Introductory Statistics* (CC BY 4.0): https://openstax.org/details/books/introductory-statistics. Chapters 1 and 2 cover descriptive statistics, 8 to 10 inference, and 12 regression. Useful exercises: Ch. 2 #1-12, #20-28 (challenge #45-52); Ch. 8 #1-12; Ch. 9 #1-14 (challenge #60-70); Ch. 10 #1-10 (challenge #40-48); Ch. 12 #1-14, #30-40 (challenge #55-65).
- [Seeing Theory](https://seeing-theory.brown.edu), interactive visualizations of distributions, the CLT, and regression.
- Simulating experiments in Python with NumPy and SciPy is the fastest way to check a theoretical result; solve by hand first, then verify.
