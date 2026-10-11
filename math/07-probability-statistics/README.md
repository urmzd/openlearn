# Probability & Statistics

## Overview
- **Textbook (Probability)**: *Introduction to Probability* by Grinstead & Snell -- https://math.dartmouth.edu/~prob/prob/prob.pdf (free)
- **Textbook (Statistics)**: *Introductory Statistics* -- https://openstax.org/details/books/introductory-statistics (CC BY 4.0)
- **Supplementary**: [Seeing Theory](https://seeing-theory.brown.edu) (interactive visualizations)
- **Prerequisites**: [Calculus 2](../02-calculus-2/), [Discrete Math 1](../05-discrete-math-1/)
- **Estimated time**: 3-4 weeks at 10-12 hrs/week

## Key Takeaways
- Formalize probability with axioms and understand conditional probability and Bayes' theorem
- Master the major discrete and continuous distributions and know when each applies
- Connect random variables to expectation, variance, and covariance -- the language of data science
- Understand the Law of Large Numbers and Central Limit Theorem as the theoretical foundation of statistical inference
- Perform hypothesis testing and confidence intervals -- the tools behind A/B testing and experimental design
- Fit simple linear regression models and interpret coefficients, residuals, and R^2

## How to Study
- Read Grinstead & Snell for probability (Chapters 1-9) and OpenStax for statistics
- Use Seeing Theory for interactive exploration of distributions, CLT, and regression
- Solve problems by hand before using calculators or code
- Simulate experiments in Python (NumPy/SciPy) to verify theoretical results
- Focus on building intuition for when to use each distribution and each test

---

# Concepts & Techniques

## Core Insight

Probability provides the mathematical framework for reasoning about uncertainty, and statistics
provides the tools for drawing conclusions from data. The Central Limit Theorem bridges them:
no matter the underlying distribution, sample means are approximately normal, which is why
confidence intervals and hypothesis tests work. For CS, probability is everywhere -- from
randomized algorithms and hashing to machine learning, A/B testing, and queueing theory.

## 1. Probability Foundations

**Textbook sections**: Grinstead & Snell, Ch 1 and Ch 4

**Key definitions**:
- **Sample space**: Omega = the set of all possible outcomes
- **Event**: A subset of Omega
- **Probability axioms**: (1) P(A) >= 0; (2) P(Omega) = 1; (3) P(A union B) = P(A) + P(B) for disjoint A, B (extends to countable unions)
- **Conditional probability**: P(A | B) = P(A intersect B) / P(B)
- **Independence**: A and B are independent if P(A intersect B) = P(A) * P(B)

**Key theorems**:
- **Bayes' Theorem**: P(A | B) = P(B | A) * P(A) / P(B). *Intuition*: to "reverse" a conditional probability, weight the likelihood P(B|A) by the prior P(A) and normalize. This is the engine of Bayesian inference.
- **Law of Total Probability**: P(B) = Sum over i of P(B | A_i) * P(A_i) where {A_i} partitions Omega. *Intuition*: compute the probability of B by considering all the mutually exclusive ways B can happen.
- **Inclusion-Exclusion for Probability**: P(A union B) = P(A) + P(B) - P(A intersect B). *Intuition*: same as set inclusion-exclusion but with probabilities.

**Worked example**:
> A test for a disease is 99% sensitive (P(+|disease) = 0.99) and 95% specific (P(-|no disease) = 0.95). The disease prevalence is 1%. What is P(disease | +)? By Bayes': P(disease|+) = P(+|disease)*P(disease) / P(+). P(+) = P(+|disease)*0.01 + P(+|no disease)*0.99 = 0.99*0.01 + 0.05*0.99 = 0.0099 + 0.0495 = 0.0594. P(disease|+) = 0.0099/0.0594 = 0.167. Despite 99% sensitivity, the positive predictive value is only 16.7% due to low prevalence.

**Essential problems**: Grinstead & Snell Ch 1: #1-8, #13-18; Ch 4: #1-10, #17-22
**Challenge problems**: Grinstead & Snell Ch 4: #25-30

## 2. Discrete Random Variables

**Textbook sections**: Grinstead & Snell, Ch 5 and Ch 6

**Key definitions**:
- **Random variable**: A function X: Omega -> R that assigns a numerical value to each outcome
- **PMF (probability mass function)**: P(X = x) for each possible value x
- **CDF (cumulative distribution function)**: F(x) = P(X <= x)
- **Expected value**: E[X] = Sum over x of x * P(X = x)
- **Variance**: Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
- **Standard deviation**: SD(X) = sqrt(Var(X))

**Key distributions**:
- **Bernoulli(p)**: X in {0,1}, P(X=1) = p. E[X] = p, Var(X) = p(1-p)
- **Binomial(n,p)**: Number of successes in n independent Bernoulli trials. P(X=k) = C(n,k)*p^k*(1-p)^(n-k). E[X] = np, Var(X) = np(1-p)
- **Geometric(p)**: Number of trials until first success. P(X=k) = (1-p)^(k-1)*p. E[X] = 1/p
- **Poisson(lambda)**: Number of events in a fixed interval. P(X=k) = e^(-lambda)*lambda^k/k!. E[X] = Var(X) = lambda

**Key theorems**:
- **Linearity of Expectation**: E[aX + bY] = aE[X] + bE[Y], always (even for dependent variables). *Intuition*: expectation is a linear operator, no independence needed. This is one of the most useful properties in all of probability.
- **Variance of Sum (independent)**: Var(X + Y) = Var(X) + Var(Y) when X, Y are independent. *Intuition*: independent fluctuations add in quadrature.

**Worked example**:
> A fair die is rolled 10 times. Let X = number of sixes. X ~ Binomial(10, 1/6). E[X] = 10*(1/6) = 5/3. Var(X) = 10*(1/6)*(5/6) = 25/18. P(X=0) = (5/6)^10 = 0.1615. P(X >= 3) = 1 - P(X=0) - P(X=1) - P(X=2) = 1 - (5/6)^10 - C(10,1)*(1/6)*(5/6)^9 - C(10,2)*(1/6)^2*(5/6)^8 = 0.2248.

> Expected number of distinct coupons when collecting n types: E = n * H_n = n * (1 + 1/2 + ... + 1/n) (the Coupon Collector Problem, using linearity of expectation over geometric waiting times).

**Essential problems**: Grinstead & Snell Ch 5: #1-10, #15-22; Ch 6: #1-8
**Challenge problems**: Grinstead & Snell Ch 6: #15-20

## 3. Continuous Random Variables

**Textbook sections**: Grinstead & Snell, Ch 2 (continuous density) and Ch 5

**Key definitions**:
- **PDF (probability density function)**: f(x) >= 0 and integral from -inf to inf of f(x) dx = 1; P(a <= X <= b) = integral from a to b of f(x) dx
- **CDF**: F(x) = P(X <= x) = integral from -inf to x of f(t) dt; f(x) = F'(x)
- **Expected value**: E[X] = integral from -inf to inf of x * f(x) dx
- **Variance**: Var(X) = integral from -inf to inf of (x - mu)^2 * f(x) dx

**Key distributions**:
- **Uniform(a,b)**: f(x) = 1/(b-a) on [a,b]. E[X] = (a+b)/2, Var(X) = (b-a)^2/12
- **Exponential(lambda)**: f(x) = lambda*e^(-lambda*x) for x >= 0. E[X] = 1/lambda, Var(X) = 1/lambda^2. Memoryless property: P(X > s+t | X > s) = P(X > t)
- **Normal(mu, sigma^2)**: f(x) = (1/(sigma*sqrt(2*pi))) * exp(-(x-mu)^2/(2*sigma^2)). The "bell curve." E[X] = mu, Var(X) = sigma^2
- **Standard Normal**: Z = (X - mu)/sigma ~ Normal(0,1)

**Key theorems**:
- **68-95-99.7 Rule**: For a normal distribution, approximately 68% of values fall within 1 SD, 95% within 2 SD, 99.7% within 3 SD. *Intuition*: the normal distribution concentrates tightly around the mean.
- **Memoryless Property of Exponential**: P(X > s+t | X > s) = P(X > t). *Intuition*: the exponential distribution "forgets" how long you have been waiting -- the remaining wait time has the same distribution regardless. This is the continuous analogue of the geometric distribution.

**Worked example**:
> Scores are Normal(72, 64) (mean 72, variance 64, so SD = 8). What fraction scores above 88? Standardize: Z = (88-72)/8 = 2. P(Z > 2) = 1 - 0.9772 = 0.0228. About 2.3% score above 88.

> If service time is Exponential(lambda = 0.5), what is P(service > 4 minutes)? P(X > 4) = e^(-0.5*4) = e^(-2) = 0.1353.

**Essential problems**: Grinstead & Snell Ch 2: #1-10; Ch 5: #10-20 (continuous sections)
**Challenge problems**: Grinstead & Snell Ch 5: #25-30

## 4. Joint Distributions

**Textbook sections**: Grinstead & Snell, Ch 7

**Key definitions**:
- **Joint PMF/PDF**: P(X=x, Y=y) or f(x,y) giving probabilities for pairs
- **Marginal distribution**: f_X(x) = Sum over y of f(x,y) (discrete) or integral over y of f(x,y) dy (continuous)
- **Conditional distribution**: f(y|x) = f(x,y) / f_X(x)
- **Covariance**: Cov(X,Y) = E[XY] - E[X]*E[Y]; measures linear association
- **Correlation**: rho(X,Y) = Cov(X,Y) / (SD(X)*SD(Y)); always in [-1,1]

**Key theorems**:
- **Independence Characterization**: X and Y are independent iff f(x,y) = f_X(x)*f_Y(y) for all x,y. *Intuition*: knowing X tells you nothing about Y.
- **Variance of Sum (general)**: Var(X+Y) = Var(X) + Var(Y) + 2*Cov(X,Y). *Intuition*: when variables are positively correlated, their sum fluctuates more than independent variables would.
- **Covariance Properties**: Cov(aX, bY) = ab*Cov(X,Y). Cov(X+Y, Z) = Cov(X,Z) + Cov(Y,Z). Cov(X,X) = Var(X).

**Worked example**:
> Let X and Y have joint PMF: P(0,0)=0.1, P(0,1)=0.2, P(1,0)=0.3, P(1,1)=0.4. Marginals: P(X=0) = 0.3, P(X=1) = 0.7, P(Y=0) = 0.4, P(Y=1) = 0.6. Check independence: P(X=0)*P(Y=0) = 0.12 != 0.1 = P(X=0,Y=0). So X and Y are not independent. E[X] = 0.7, E[Y] = 0.6, E[XY] = 0*0*0.1 + 0*1*0.2 + 1*0*0.3 + 1*1*0.4 = 0.4. Cov(X,Y) = 0.4 - 0.7*0.6 = -0.02.

**Essential problems**: Grinstead & Snell Ch 7: #1-10, #15-20
**Challenge problems**: Grinstead & Snell Ch 7: #23-28

## 5. Limit Theorems

**Textbook sections**: Grinstead & Snell, Ch 8 and Ch 9

**Key definitions**:
- **Sample mean**: X_bar = (X_1 + X_2 + ... + X_n) / n
- **Convergence in probability**: X_n converges to c if P(|X_n - c| > epsilon) -> 0 for every epsilon > 0

**Key theorems**:
- **Law of Large Numbers (Weak)**: If X_1, X_2, ... are iid with mean mu, then X_bar converges to mu in probability as n -> inf. *Intuition*: the average of many independent measurements converges to the true mean. This is why sampling works.
- **Central Limit Theorem**: If X_1, ..., X_n are iid with mean mu and variance sigma^2, then (X_bar - mu) / (sigma/sqrt(n)) converges in distribution to N(0,1) as n -> inf. *Intuition*: no matter the shape of the original distribution, the distribution of the sample mean becomes approximately normal for large n. This is why the normal distribution appears everywhere.
- **Chebyshev's Inequality**: P(|X - mu| >= k*sigma) <= 1/k^2. *Intuition*: a distribution-free bound on how spread out a random variable can be. Weak but universally applicable.

**Worked example**:
> A factory produces items with mean weight 500g and SD 10g. For a sample of n = 100, what is P(X_bar > 502)? By CLT: X_bar ~ approx. N(500, 100/100) = N(500, 1). Z = (502 - 500)/1 = 2. P(Z > 2) = 0.0228. There is about a 2.3% chance the sample mean exceeds 502g.

> Chebyshev: For any distribution with mu = 100, sigma = 15, P(|X - 100| >= 45) <= (15/45)^2 = 1/9. At most 11.1% of values can be more than 3 standard deviations from the mean (compare to 0.3% for normal -- Chebyshev is loose but general).

**Essential problems**: Grinstead & Snell Ch 8: #1-8; Ch 9: #1-6
**Challenge problems**: Grinstead & Snell Ch 9: #10-14

## 6. Descriptive Statistics

**Textbook sections**: OpenStax Introductory Statistics, Ch 1-2

**Key definitions**:
- **Measures of center**: Mean (arithmetic average), median (middle value), mode (most frequent)
- **Measures of spread**: Range, interquartile range (IQR = Q3 - Q1), variance, standard deviation
- **Percentiles and quartiles**: Q1 (25th percentile), Q2 (median), Q3 (75th percentile)
- **Box plot**: Displays min, Q1, median, Q3, max; outliers beyond 1.5*IQR from quartiles
- **Skewness**: Left-skewed (mean < median), right-skewed (mean > median), symmetric (mean = median)

**Worked example**:
> Data: 3, 7, 7, 12, 15, 18, 22. Mean = 84/7 = 12. Median = 12 (4th value). Q1 = 7, Q3 = 18, IQR = 11. Outlier fences: 7 - 1.5*11 = -9.5 (lower), 18 + 1.5*11 = 34.5 (upper). No outliers.

**Essential problems**: OpenStax Statistics Ch 2: #1-12, #20-28
**Challenge problems**: OpenStax Statistics Ch 2: #45-52

## 7. Inference

**Textbook sections**: OpenStax Introductory Statistics, Ch 8-10

**Key definitions**:
- **Point estimate**: A single value estimating a parameter (e.g., X_bar estimates mu)
- **Confidence interval**: An interval [L, U] such that P(L <= theta <= U) = 1 - alpha; e.g., X_bar +/- z_(alpha/2) * sigma/sqrt(n)
- **Null hypothesis (H_0)**: The default claim (e.g., mu = mu_0)
- **Alternative hypothesis (H_a)**: The claim we seek evidence for (e.g., mu != mu_0, or mu > mu_0)
- **Test statistic**: Z = (X_bar - mu_0) / (sigma/sqrt(n)) or T = (X_bar - mu_0) / (s/sqrt(n))
- **p-value**: The probability of observing a test statistic as extreme as (or more extreme than) the one observed, assuming H_0 is true
- **Type I error**: Rejecting H_0 when it is true (probability = alpha)
- **Type II error**: Failing to reject H_0 when it is false (probability = beta)

**Key theorems**:
- **CI Interpretation**: A 95% confidence interval means that if we repeat the experiment many times, 95% of the constructed intervals will contain the true parameter. *Intuition*: it is a statement about the procedure, not about any single interval.
- **Neyman-Pearson Framework**: Choose alpha (significance level) before the experiment. Reject H_0 if p-value < alpha. *Intuition*: alpha controls the rate of false positives -- how often you incorrectly declare an effect.

**Worked example**:
> A company claims its batteries last 500 hours. A sample of 36 gives X_bar = 490, s = 30. Test at alpha = 0.05. H_0: mu = 500, H_a: mu < 500. Test statistic: Z = (490 - 500)/(30/6) = -10/5 = -2.0. p-value = P(Z < -2.0) = 0.0228 < 0.05. Reject H_0. There is significant evidence that the mean lifetime is less than 500 hours.

> 95% CI for a population mean: X_bar +/- 1.96 * s/sqrt(n) = 490 +/- 1.96*(30/6) = 490 +/- 9.8 = (480.2, 499.8). Since 500 is outside this interval, this is consistent with rejecting H_0.

**Essential problems**: OpenStax Statistics Ch 8: #1-12; Ch 9: #1-14; Ch 10: #1-10
**Challenge problems**: OpenStax Statistics Ch 9: #60-70; Ch 10: #40-48

## 8. Regression

**Textbook sections**: OpenStax Introductory Statistics, Ch 12

**Key definitions**:
- **Simple linear regression**: y_hat = b_0 + b_1*x, where b_1 = Sum((x_i - x_bar)(y_i - y_bar)) / Sum((x_i - x_bar)^2) and b_0 = y_bar - b_1*x_bar
- **Residual**: e_i = y_i - y_hat_i (observed minus predicted)
- **Coefficient of determination**: R^2 = 1 - OL_res / OL_tot = the proportion of variance in y explained by x
- **Correlation coefficient**: r = b_1 * (s_x / s_y); R^2 = r^2 for simple linear regression

**Key theorems**:
- **Least Squares Optimality**: The OLS estimates b_0, b_1 minimize the sum of squared residuals Sum(e_i^2). *Intuition*: among all lines, the regression line makes the total squared error as small as possible -- this connects directly to orthogonal projection in linear algebra (A^T A x = A^T b).
- **Gauss-Markov Theorem**: Under the assumptions of linear regression (linearity, independence, homoscedasticity, no perfect multicollinearity), OLS estimators are the Best Linear Unbiased Estimators (BLUE). *Intuition*: you cannot find an unbiased linear estimator with smaller variance.

**Worked example**:
> Data: (1,2), (2,4), (3,5), (4,4), (5,5). x_bar = 3, y_bar = 4. Sum((x_i - x_bar)(y_i - y_bar)): (-2)(-2) + (-1)(0) + 0(1) + 1(0) + 2(1) = 4+0+0+0+2 = 6. Sum((x_i - x_bar)^2) = 4+1+0+1+4 = 10. b_1 = 6/10 = 0.6. b_0 = 4 - 0.6*3 = 2.2. Regression line: y_hat = 2.2 + 0.6x. Predicted at x=3: y_hat = 4.0. R^2: OL_res = (2-2.8)^2 + (4-3.4)^2 + (5-4.0)^2 + (4-4.6)^2 + (5-5.2)^2 = 0.64+0.36+1.0+0.36+0.04 = 2.4. OL_tot = 4+0+1+0+1 = 6. R^2 = 1 - 2.4/6 = 0.6. The model explains 60% of the variance in y.

**Essential problems**: OpenStax Statistics Ch 12: #1-14, #30-40
**Challenge problems**: OpenStax Statistics Ch 12: #55-65

---

## Technique Catalog

| Technique | When to Use | Key Formula/Idea |
|-----------|-------------|------------------|
| Bayes' Theorem | Reverse conditional probabilities | P(A|B) = P(B|A)*P(A) / P(B) |
| Law of Total Probability | Decompose by cases | P(B) = Sum P(B|A_i)*P(A_i) |
| Linearity of Expectation | Compute expected values of sums | E[X+Y] = E[X] + E[Y], always |
| CLT Approximation | Approximate distribution of sample mean | X_bar ~ N(mu, sigma^2/n) for large n |
| Z-test / T-test | Hypothesis testing for means | Compare test statistic to critical value |
| Confidence Interval | Estimate a parameter with uncertainty | X_bar +/- z * sigma/sqrt(n) |
| Least Squares | Fit a linear model to data | Minimize sum of squared residuals |
| Standardization | Convert to standard normal | Z = (X - mu) / sigma |

## Connections to CS & Algorithms

| Math Concept | CS Application | Repo Link |
|-------------|----------------|-----------|
| Bayes' Theorem | Spam filters (Naive Bayes), ML classification | ML track |
| Distributions (Poisson, exponential) | Queueing theory, network traffic modeling | systems track |
| Expected value / linearity | Randomized algorithm analysis (QuickSort, hashing) | algorithms track |
| CLT | A/B testing, confidence intervals in production | systems track |
| Hypothesis testing | Feature flag experiments, statistical significance | systems track |
| Regression | Machine learning (linear models as baseline) | ML track |
| Conditional probability | Hidden Markov Models, Bayesian networks | ML track |
| Variance / covariance | Portfolio theory, PCA, feature engineering | ML track |

## Company Relevance

| Company | How This Appears | Difficulty |
|---------|-----------------|------------|
| Google/Meta | A/B testing at scale, experiment design, statistical significance | Medium |
| Amazon | Demand forecasting, recommendation confidence, pricing experiments | Medium |
| Quantitative Finance | Risk modeling (VaR), option pricing, stochastic processes | Hard |
| ML/AI Startups | Bayesian inference, probabilistic models, evaluation metrics | Hard |
| Any Data Science role | Hypothesis testing, regression, confidence intervals daily | Medium |
| Gaming (Riot, Blizzard) | Matchmaking algorithms use Bayesian rating systems (TrueSkill, Elo) | Medium |

## Chapters

<!-- ol:chapters -->
| # | Module | Chapter | Kind | Pass |
|---|---|---|---|---|
| 1 | `M07.0` | [Random variables, expectation, variance, and normal draws by Box-Muller](00-random-variables-and-the-normal.md) | build | 2 |
| 2 | `M07.1` | [Categorical sampling: inverse CDF, Gumbel-max, alias method](01-categorical-sampling.md) | build | 3 |
| 3 | `M07.2` | [MLE, Laplace, absolute discounting](02-mle-laplace-absolute-discounting.md) | build | 3 |
| 4 | `M07.3` | [Variance propagation and initialization](03-variance-propagation-and-initialization.md) | build | 2 |
| 5 | `M07.4` | [LLN, CLT, confidence intervals, bootstrap](04-lln-clt-confidence-intervals-bootstrap.md) | build | 4 |
| 6 | `M07.5` | [Hypothesis tests: paired and two-sample permutation, McNemar, Holm](05-hypothesis-tests.md) | build | 5 |
| 7 | `M07.6` | [Rejection sampling and residual distributions](06-rejection-sampling-and-residual-distributions.md) | build | 6 |
| 8 | `M07.7` | [Logistic regression (IRLS), ROC-AUC, calibration (ECE)](07-logistic-regression-roc-auc-calibration.md) | build | 5 |
| 9 | `S-M07a` | [Probability problem set, part a: axioms, Bayes, random variables, Box-Muller](90-problem-set-a.md) | solve | 2 |
| 10 | `S-M07b` | [Solve set: joint and covariance, MLE](91-problem-set-b.md) | solve | 3 |
| 11 | `S-M07c` | [Solve set: LLN/CLT, confidence intervals](92-problem-set-c.md) | solve | 4 |
| 12 | `S-M07d` | [Solve set: hypothesis tests, regression, queueing and Little's law, balls into bins](93-problem-set-d.md) | solve | 5 |
<!-- /ol:chapters -->
