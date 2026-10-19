# Statistical Learning

## Overview

- **Primary textbook**: *An Introduction to Statistical Learning* (ISLR) -- [free online](https://www.statlearning.com/)
- **Advanced reference**: *The Elements of Statistical Learning* (ESL) -- [free PDF](https://hastie.su.domains/ElemStatLearn/)
- **Prerequisites**: [Linear Algebra](../../mathematics-for-computing/03-linear-algebra/), [Probability & Statistics](../../mathematics-for-computing/07-probability-statistics/)
- **Estimated time**: 4-5 weeks at 10-12 hrs/week

## Key Takeaways

- The bias-variance tradeoff is the central tension in all of statistical learning
- Simple models (linear regression, logistic regression) are not just "easy" -- they're often the right choice
- Resampling methods (cross-validation, bootstrap) are how you honestly evaluate any model
- Ensemble methods (random forests, boosting) dominate tabular data in practice

## How to Study

- Read ISLR chapters in order -- it's deliberately sequenced for pedagogy
- Do the conceptual exercises first, then applied exercises in R or Python
- Move to ESL chapters for mathematical depth on topics that interest you
- Implement at least linear regression, logistic regression, and K-means from scratch

---

# Concepts & Techniques

## Core Insight

Statistical learning is about finding f(X) that predicts Y, while understanding the uncertainty in that prediction. Every method navigates the same tradeoff: more flexible models fit training data better but generalize worse (variance); simpler models miss patterns (bias).

## 1. Statistical Learning Framework

**ISLR sections**: Ch 2

**Key definitions**:
- **Reducible error**: error from using f-hat instead of true f; can be minimized by choosing a better method
- **Irreducible error**: Var(epsilon); inherent noise in the system that no model can eliminate
- **Bias**: E[f-hat(x)] - f(x); error from approximating a complex problem with a simple model
- **Variance**: Var[f-hat(x)]; how much f-hat changes with different training data

**Key theorem**:
- **Bias-Variance Decomposition**: E[(Y - f-hat(x))^2] = Bias^2 + Variance + Irreducible Error. *Intuition*: test error decomposes into three independent sources. You can't reduce all three simultaneously.

**Worked example**:
> Fitting polynomials of degree 1, 5, and 15 to noisy data from a cubic function. Degree 1 (high bias, low variance) -- systematic undershoot. Degree 15 (low bias, high variance) -- fits training data perfectly but oscillates wildly on test data. Degree 5 (balanced) -- best test error.

**Essential problems**: ISLR Ch 2: Conceptual #1-4, Applied #8-10

## 2. Linear Regression

**ISLR sections**: Ch 3

**Key ideas**:
- **Normal equation**: w = (X^T X)^{-1} X^T y -- closed-form solution when X^T X is invertible
- **Hypothesis testing**: t-statistics and p-values for each coefficient; F-statistic for overall significance
- **Model selection**: R^2, adjusted R^2, AIC, BIC, Mallow's Cp

**Key theorem**:
- **Gauss-Markov**: Among all linear unbiased estimators, OLS has the smallest variance. *Intuition*: if the true relationship is linear, OLS is the best you can do without regularization.

**Essential problems**: ISLR Ch 3: Conceptual #1-3, Applied #8, #9, #15

## 3. Classification

**ISLR sections**: Ch 4

**Key ideas**:
- **Logistic regression**: models P(Y=1|X) = 1/(1 + exp(-X^T w)); decision boundary is linear
- **LDA**: assumes Gaussian class-conditional densities with shared covariance; Bayes-optimal if assumptions hold
- **QDA**: like LDA but allows separate covariance per class; more flexible, more parameters
- **ROC and AUC**: ROC plots TPR vs FPR across all thresholds; AUC = probability that a random positive ranks higher than a random negative

**Essential problems**: ISLR Ch 4: Conceptual #1-5, Applied #13

## 4. Resampling Methods

**ISLR sections**: Ch 5

**Key ideas**:
- **k-fold cross-validation**: split data into k folds, train on k-1, test on held-out fold, rotate. k=5 or 10 is standard
- **LOOCV**: k-fold with k=n; low bias but high variance and computationally expensive
- **Bootstrap**: sample n observations with replacement B times; estimate standard errors of any statistic

**Key insight**: the training error is NOT a good estimate of test error. Cross-validation gives you an honest estimate.

**Essential problems**: ISLR Ch 5: Conceptual #1-4, Applied #5-9

## 5. Regularization

**ISLR sections**: Ch 6

**Key ideas**:
- **Ridge regression**: minimize ||y - Xw||^2 + lambda * ||w||^2. Shrinks coefficients toward zero but never to exactly zero
- **Lasso**: minimize ||y - Xw||^2 + lambda * ||w||_1. Can shrink coefficients to exactly zero (feature selection)
- **Elastic net**: combines L1 and L2 penalties; best of both worlds
- **PCR/PLS**: dimension reduction approaches; project features onto principal components before regression

**Key insight**: regularization trades increased bias for decreased variance. The lambda parameter controls this tradeoff explicitly.

**Essential problems**: ISLR Ch 6: Conceptual #1-4, Applied #8-11

## 6. Tree-Based Methods

**ISLR sections**: Ch 8

**Key ideas**:
- **Decision trees**: recursive binary splitting; greedy, interpretable, but high variance
- **Bagging**: average many trees trained on bootstrap samples; reduces variance
- **Random forests**: bagging + random feature subsets at each split; decorrelates trees
- **Boosting**: sequential fitting of weak learners to residuals; controls bias and variance
- **XGBoost/LightGBM**: optimized gradient boosting with regularization; dominant on tabular data

**Key insight**: a single decision tree is interpretable but unstable. Ensembles sacrifice interpretability for dramatic accuracy gains.

**Essential problems**: ISLR Ch 8: Conceptual #1-5, Applied #7-12

## 7. Support Vector Machines

**ISLR sections**: Ch 9

**Key ideas**:
- **Maximal margin classifier**: find the hyperplane with the largest margin between classes
- **Support vector classifier**: soft margin allows some misclassifications; controlled by C parameter
- **Kernel trick**: map to higher dimensions implicitly; RBF kernel for non-linear boundaries; no need to compute the mapping explicitly

**Key theorem**:
- **Representer theorem**: the SVM solution depends only on inner products between data points, enabling the kernel trick

**Essential problems**: ISLR Ch 9: Conceptual #1-3, Applied #7-8

## 8. Unsupervised Learning

**ISLR sections**: Ch 12

**Key ideas**:
- **PCA**: find directions of maximum variance; eigendecomposition of covariance matrix
- **K-means**: iterative assignment + centroid update; converges but to local optimum
- **Hierarchical clustering**: agglomerative (bottom-up) with linkage choices (complete, average, single)
- **Choosing K**: elbow method (inertia), silhouette score, gap statistic

**Essential problems**: ISLR Ch 12: Conceptual #1-2, Applied #7-10

## 9. Advanced Topics (ESL)

**ESL chapters**: Ch 5, 8, 9, 14

**Key ideas**:
- **Additive models & splines**: flexible nonparametric regression (ESL Ch 5)
- **EM algorithm**: iterative optimization for latent variable models (GMM, missing data) (ESL Ch 8)
- **Ensemble methods in depth**: stacking, mixture of experts (ESL Ch 8)
- **Kernel smoothing**: local regression, Nadaraya-Watson estimator (ESL Ch 6)

---

## Technique Catalog

| Method | Type | Bias | Variance | Interpretability | Best For |
|--------|------|------|----------|-----------------|----------|
| Linear regression | Supervised | High | Low | High | Linear relationships, baseline |
| Logistic regression | Supervised | High | Low | High | Binary classification, baseline |
| LDA/QDA | Supervised | Medium | Medium | Medium | Gaussian-distributed features |
| Ridge/Lasso | Supervised | Medium | Low | Medium | High-dimensional, regularization |
| Decision tree | Supervised | Low | High | High | Interpretable models |
| Random forest | Supervised | Low | Low | Low | Tabular data, default choice |
| XGBoost | Supervised | Low | Low | Low | Kaggle, production tabular ML |
| SVM | Supervised | Low | Medium | Low | Small-medium datasets, kernels |
| K-means | Unsupervised | -- | -- | Medium | Spherical clusters |
| PCA | Unsupervised | -- | -- | Medium | Dimensionality reduction |

## Connections to Other Tracks

| Concept | Connected Track | Application |
|---------|-----------------|-------------|
| Linear algebra (SVD, eigendecomposition) | [Linear Algebra](../../mathematics-for-computing/03-linear-algebra/) | PCA, regression normal equations |
| Probability (Bayes, distributions) | [Probability](../../mathematics-for-computing/07-probability-statistics/) | LDA, Naive Bayes, generative models |
| Gradient descent | [Deep Learning](../02-deep-learning/) | Foundation for neural network optimization |
| Bias-variance | [Deep Learning](../02-deep-learning/) | Extends to double descent in deep learning |

## Company Relevance

| Company | How This Appears | Difficulty |
|---------|-----------------|------------|
| Google | ML system design, bias-variance analysis, feature engineering | Advanced |
| Meta | Ranking/recommendation, logistic regression at scale, A/B testing | Advanced |
| Netflix | Collaborative filtering, evaluation metrics, recommendation | Advanced |
| Anthropic | Model evaluation, statistical rigor in AI safety | Advanced |
| Jane Street | Factor models, regression, statistical arbitrage | Expert |
| Two Sigma | Time series, regression, ensemble methods | Expert |
