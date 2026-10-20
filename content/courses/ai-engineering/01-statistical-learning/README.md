# Statistical Learning

Statistical learning is the study of finding a function $f(X)$ that predicts an outcome $Y$ from inputs $X$, while understanding how uncertain that prediction is. It is the foundation the rest of *AI Engineering* stands on: deep networks, reinforcement learning agents, and large language models are all, at bottom, flexible estimators of $f$ judged by how well they generalize. This chapter is where you learn the vocabulary for that judgment.

One tension runs through every method in the chapter. More flexible models fit the training data better but generalize worse, because they also fit its noise (variance); simpler models generalize more stably but miss real patterns (bias). The chapter opens by making that tradeoff precise, then walks through the classical toolbox in the order *An Introduction to Statistical Learning* (ISLR) teaches it: linear regression, classification, resampling, regularization, tree-based ensembles, support vector machines, and unsupervised learning. Each method is a different answer to the same question of how much flexibility to buy and how to pay for it. It closes with pointers into the more advanced material of *The Elements of Statistical Learning* (ESL) and a catalog that compares the methods side by side.

Two lessons are worth carrying out of the chapter. Simple models such as linear and logistic regression are not merely "easy"; they are often the right choice. And ensemble methods such as random forests and gradient boosting dominate tabular data in practice.

> **Learning objectives**
> - State the bias-variance decomposition and use it to explain underfitting and overfitting.
> - Fit and interpret linear and logistic regression, and compare logistic regression, LDA, and QDA as classifiers.
> - Estimate test error honestly with cross-validation and quantify uncertainty with the bootstrap.
> - Explain how ridge, lasso, bagging, random forests, and boosting each trade bias against variance.
> - Describe the kernel trick, PCA, and K-means, and choose a reasonable method for a new tabular problem.

## Before you begin

You need linear algebra (matrix products, inverses, eigendecomposition) and basic probability and statistics (expectation, variance, distributions). Both are covered in *Mathematics*: see [Linear Algebra](../../mathematics/09-linear-algebra/README.md) and [Probability and Statistics](../../mathematics/13-probability-statistics/README.md).

The chapter follows two free textbooks. The primary text is [*An Introduction to Statistical Learning*](https://www.statlearning.com/) (ISLR), which is deliberately sequenced for teaching, so read its chapters in order. The advanced reference is [*The Elements of Statistical Learning*](https://hastie.su.domains/ElemStatLearn/) (ESL), which gives the mathematical depth behind the same ideas; turn to it for topics that interest you. For each topic, do ISLR's conceptual exercises first and then its applied exercises in R or Python. The exercises listed in each section are the essential ones. Finally, implement at least linear regression, logistic regression, and K-means from scratch; nothing else makes the mechanics stick as well.

## In this chapter

- [The statistical learning framework](#the-statistical-learning-framework): reducible and irreducible error, bias, variance, and the decomposition that ties them together.
- [Linear regression](#linear-regression): the normal equation, hypothesis tests on coefficients, model selection, and why OLS is optimal among linear unbiased estimators.
- [Classification](#classification): logistic regression, LDA, QDA, and evaluating classifiers with ROC curves and AUC.
- [Resampling methods](#resampling-methods): cross-validation for honest test error and the bootstrap for uncertainty.
- [Regularization](#regularization): ridge, lasso, elastic net, and dimension-reduction regression.
- [Tree-based methods](#tree-based-methods): decision trees and the ensembles that stabilize them.
- [Support vector machines](#support-vector-machines): margins, soft margins, and the kernel trick.
- [Unsupervised learning](#unsupervised-learning): PCA, K-means, hierarchical clustering, and choosing the number of clusters.
- [Advanced topics from ESL](#advanced-topics-from-esl) and [Comparing the methods](#comparing-the-methods): where to go deeper, and a side-by-side catalog.

## The statistical learning framework

*Reading: ISLR Chapter 2.*

Suppose the data are generated as $Y = f(X) + \varepsilon$, where $\varepsilon$ is noise with mean zero. You never see $f$; you estimate it with $\hat f$ from a training set. The error of your predictions has two sources. The **reducible error** comes from using $\hat f$ instead of the true $f$, and a better method can shrink it. The **irreducible error** is $\operatorname{Var}(\varepsilon)$, noise inherent in the system that no model can remove.

The reducible part splits further. **Bias**, $E[\hat f(x)] - f(x)$, is the error from approximating a complex problem with a model too simple to capture it. **Variance**, $\operatorname{Var}[\hat f(x)]$, measures how much $\hat f$ would change if you trained it on a different sample. The bias-variance decomposition puts these together for the expected squared test error at a point $x$:

$$E\big[(Y - \hat f(x))^2\big] = \operatorname{Bias}[\hat f(x)]^2 + \operatorname{Var}[\hat f(x)] + \operatorname{Var}(\varepsilon).$$

Test error, in other words, comes from three separate sources. The last is out of your hands, and the first two pull against each other: making a model more flexible lowers its bias and raises its variance, so you cannot drive both to zero at once. The art is finding the flexibility that minimizes their sum.

A concrete case makes this vivid. Fit polynomials of degree 1, 5, and 15 to noisy data drawn from a cubic function. The degree-1 fit has high bias and low variance: it systematically undershoots the curve. The degree-15 fit has low bias and high variance: it passes through the training points almost perfectly but oscillates wildly between them, so its test error is poor. The degree-5 fit is balanced and achieves the best test error.

*Exercises: ISLR Chapter 2, Conceptual 1-4 and Applied 8-10.*

## Linear regression

*Reading: ISLR Chapter 3.*

With that framework in place, start with the simplest flexible model: $Y \approx Xw$. Least squares chooses $w$ to minimize $\lVert y - Xw \rVert^2$, and setting the gradient to zero gives the **normal equation**

$$w = (X^\top X)^{-1} X^\top y,$$

a closed-form solution whenever $X^\top X$ is invertible.

Because the model is so simple, you can reason about it statistically. Each coefficient has a t-statistic and p-value that test whether it differs from zero, and the F-statistic tests whether any predictor matters at all. Choosing among models with different predictors uses $R^2$, adjusted $R^2$, AIC, BIC, and Mallows' $C_p$, the last four of which penalize model size so that adding predictors does not automatically look like an improvement.

The **Gauss-Markov theorem** explains why ordinary least squares (OLS) is the default: among all linear unbiased estimators, OLS has the smallest variance. If the true relationship is linear and you insist on an unbiased estimate, OLS is the best you can do. The way to beat it is to accept some bias, which is exactly what regularization does later in the chapter.

*Exercises: ISLR Chapter 3, Conceptual 1-3 and Applied 8, 9, and 15.*

## Classification

*Reading: ISLR Chapter 4.*

When $Y$ is a category rather than a number, the question becomes how to model class probabilities. **Logistic regression** models

$$P(Y = 1 \mid X) = \frac{1}{1 + \exp(-X^\top w)},$$

which keeps probabilities between 0 and 1 and gives a linear decision boundary.

**Linear discriminant analysis** (LDA) takes the generative route instead: it assumes each class's inputs are Gaussian with a shared covariance matrix and applies Bayes' rule, which is Bayes-optimal when those assumptions hold. **Quadratic discriminant analysis** (QDA) allows each class its own covariance, which bends the boundary into a quadratic; it is more flexible but has more parameters to estimate, so it needs more data. The three classifiers sit at different points on the bias-variance curve.

To evaluate a classifier across all possible thresholds, plot its **ROC curve**, the true positive rate against the false positive rate as the threshold sweeps. The area under it, **AUC**, has a clean interpretation: it is the probability that a randomly chosen positive example is ranked above a randomly chosen negative one.

*Exercises: ISLR Chapter 4, Conceptual 1-5 and Applied 13.*

## Resampling methods

*Reading: ISLR Chapter 5.*

Everything so far assumes you can measure test error, and the key insight of this section is that training error is not a good estimate of it: a flexible model can drive training error toward zero while test error climbs. Resampling methods give you an honest estimate from the data you have.

**k-fold cross-validation** splits the data into $k$ folds, trains on $k-1$ of them, tests on the held-out fold, and rotates until every fold has been the test set; $k = 5$ or $10$ is standard. **Leave-one-out cross-validation** (LOOCV) is the extreme case $k = n$. Its estimate has low bias, since each model trains on almost all the data, but high variance, because the $n$ training sets are nearly identical, and it is computationally expensive.

The **bootstrap** answers a different question: how uncertain is a statistic? Draw $n$ observations with replacement from your sample, compute the statistic, and repeat $B$ times. The spread of those $B$ values estimates the statistic's standard error, for almost any statistic you can compute.

*Exercises: ISLR Chapter 5, Conceptual 1-4 and Applied 5-9.*

## Regularization

*Reading: ISLR Chapter 6.*

Cross-validation tells you when a model is overfitting; regularization gives you a dial to fix it. The idea is to add a penalty on the size of the coefficients, trading a little bias for a large reduction in variance, with the parameter $\lambda$ controlling the trade explicitly.

**Ridge regression** minimizes $\lVert y - Xw \rVert^2 + \lambda \lVert w \rVert_2^2$. It shrinks coefficients toward zero but never to exactly zero. The **lasso** minimizes $\lVert y - Xw \rVert^2 + \lambda \lVert w \rVert_1$; the corners of the $L_1$ penalty let it set coefficients exactly to zero, so it performs feature selection as it fits. The **elastic net** combines the $L_1$ and $L_2$ penalties, getting the lasso's sparsity with ridge's stability when predictors are correlated.

A different way to reduce variance is to reduce dimension before regressing. **Principal components regression** (PCR) and **partial least squares** (PLS) project the features onto a few derived directions, the principal components in PCR's case, and regress on those.

*Exercises: ISLR Chapter 6, Conceptual 1-4 and Applied 8-11.*

## Tree-based methods

*Reading: ISLR Chapter 8.*

Trees take the opposite starting point from linear models: very low bias and very high variance. A **decision tree** splits the data by recursive binary splitting, greedily choosing the split that most improves the fit at each step. The result is interpretable, but it is unstable; a small change in the data can produce a very different tree.

Ensembles fix the instability by combining many trees. **Bagging** trains trees on bootstrap samples and averages them, which reduces variance. **Random forests** add a second source of randomness, considering only a random subset of features at each split; this decorrelates the trees, so averaging them reduces variance further. **Boosting** works sequentially instead, fitting each weak learner to the residuals of the ensemble so far, which mainly drives down bias while shrinkage and small trees keep variance in check. **XGBoost** and **LightGBM** are optimized, regularized implementations of gradient boosting, and they dominate tabular data in competitions and production alike.

The tradeoff is worth stating plainly: a single tree is interpretable but unstable, and ensembles give up that interpretability in exchange for dramatic gains in accuracy.

*Exercises: ISLR Chapter 8, Conceptual 1-5 and Applied 7-12.*

## Support vector machines

*Reading: ISLR Chapter 9.*

Support vector machines approach classification geometrically. The **maximal margin classifier** finds the separating hyperplane with the largest margin between the classes. Real data rarely separate cleanly, so the **support vector classifier** uses a soft margin that tolerates some misclassifications, with the parameter $C$ controlling how many.

For non-linear boundaries, the **kernel trick** maps the data into a higher-dimensional space implicitly. The optimization depends on the data only through inner products between pairs of points, so replacing the inner product with a kernel function, such as the RBF kernel, fits a linear boundary in the high-dimensional space without ever computing the mapping. The **representer theorem** is the general result behind this: the solution can be written as a combination of kernel evaluations at the training points.

*Exercises: ISLR Chapter 9, Conceptual 1-3 and Applied 7-8.*

## Unsupervised learning

*Reading: ISLR Chapter 12.*

Without labels, the goal shifts from prediction to finding structure. **Principal component analysis** (PCA) finds the directions of maximum variance in the data, which are the eigenvectors of the covariance matrix. **K-means** clusters by alternating two steps, assigning each point to its nearest centroid and moving each centroid to the mean of its points; it always converges, but only to a local optimum, so run it from several starts. **Hierarchical clustering** builds clusters bottom-up (agglomeratively), merging the closest pair at each step, where "closest" depends on the linkage you choose: complete, average, or single.

Choosing the number of clusters $K$ has no single right answer. The common tools are the elbow method (look for where inertia stops dropping sharply), the silhouette score, and the gap statistic.

*Exercises: ISLR Chapter 12, Conceptual 1-2 and Applied 7-10.*

## Advanced topics from ESL

*Reading: ESL Chapters 5, 6, 8, 9, and 14.*

Once ISLR is comfortable, ESL takes the same ideas further. Additive models and splines give flexible nonparametric regression (ESL Chapter 5). Kernel smoothing covers local regression and the Nadaraya-Watson estimator (ESL Chapter 6). The EM algorithm is the iterative method for fitting latent-variable models such as Gaussian mixtures and for handling missing data (ESL Chapter 8), and the same chapter treats ensembles in more depth, including stacking and mixtures of experts.

## Comparing the methods

The methods in this chapter can be compared along the same axes the chapter has used throughout: their bias, their variance, and how interpretable they are. The catalog below summarizes them:

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

## Where this leads

The linear algebra and probability you used here keep paying off. SVD and eigendecomposition from [Linear Algebra](../../mathematics/09-linear-algebra/README.md) underlie PCA and the normal equations, and Bayes' rule and distributions from [Probability and Statistics](../../mathematics/13-probability-statistics/README.md) underlie LDA, Naive Bayes, and generative models generally. The next chapter, [Deep Learning](../02-deep-learning/README.md), builds on two ideas from this one: gradient-based fitting becomes the foundation of neural network optimization, and the bias-variance tradeoff is extended (and complicated) by the double-descent behavior of very large models.

These topics also come up constantly in interviews, which is one reason to learn them well. The table below shows how they appear at several companies covered in *Interview Preparation*:

| Company | How This Appears | Difficulty |
|---------|-----------------|------------|
| Google | ML system design, bias-variance analysis, feature engineering | Advanced |
| Meta | Ranking/recommendation, logistic regression at scale, A/B testing | Advanced |
| Netflix | Collaborative filtering, evaluation metrics, recommendation | Advanced |
| Anthropic | Model evaluation, statistical rigor in AI safety | Advanced |
| Jane Street | Factor models, regression, statistical arbitrage | Expert |
| Two Sigma | Time series, regression, ensemble methods | Expert |

### Further reading

- James, Witten, Hastie, and Tibshirani, [*An Introduction to Statistical Learning*](https://www.statlearning.com/) (free online).
- Hastie, Tibshirani, and Friedman, [*The Elements of Statistical Learning*](https://hastie.su.domains/ElemStatLearn/) (free PDF).

## Key terms

- **Reducible error**: the part of prediction error due to using $\hat f$ instead of $f$, which a better method can reduce.
- **Irreducible error**: $\operatorname{Var}(\varepsilon)$, noise that no model can remove.
- **Bias**: $E[\hat f(x)] - f(x)$, error from a model too simple for the problem.
- **Variance**: $\operatorname{Var}[\hat f(x)]$, how much the fitted model changes across training sets.
- **Cross-validation**: estimating test error by repeatedly training on part of the data and testing on the rest.
- **Bootstrap**: resampling with replacement to estimate the uncertainty of a statistic.
- **Regularization**: penalizing model complexity to trade bias for lower variance.
- **Kernel trick**: replacing inner products with a kernel to fit non-linear boundaries without computing the feature map.

## Summary

- The bias-variance tradeoff is the central tension in statistical learning: expected test error is squared bias plus variance plus irreducible noise.
- Simple models such as linear and logistic regression are often the right choice, and OLS is optimal among linear unbiased estimators.
- Training error is not test error; cross-validation and the bootstrap are how you evaluate any model honestly.
- Regularization (ridge, lasso, elastic net) and ensembles (bagging, random forests, boosting) are explicit ways to manage the tradeoff.
- Ensembles of trees dominate tabular data in practice, at the cost of interpretability.

## Review questions

1. Write the bias-variance decomposition and explain what happens to each term as model flexibility increases.
   <details><summary>Answer</summary>$E[(Y - \hat f(x))^2] = \operatorname{Bias}^2 + \operatorname{Var} + \operatorname{Var}(\varepsilon)$. As flexibility grows, squared bias falls and variance rises; the irreducible term stays fixed. Test error is U-shaped in flexibility, minimized where the sum of the first two terms is smallest.</details>
2. In the polynomial example, why does the degree-15 fit have low training error but poor test error?
   <details><summary>Answer</summary>It has enough flexibility to pass through nearly every training point, fitting the noise as well as the cubic signal. That gives low bias but high variance: on new data the fitted curve's oscillations between training points produce large errors.</details>
3. Why can the lasso set coefficients exactly to zero while ridge regression cannot?
   <details><summary>Answer</summary>The $L_1$ penalty has corners on the coordinate axes, so the constrained optimum often lands where some coefficients are exactly zero. The $L_2$ penalty is smooth and round, so it shrinks coefficients toward zero proportionally without making them exactly zero.</details>
4. How do random forests improve on bagging?
   <details><summary>Answer</summary>Bagged trees are correlated because the strongest predictors dominate the top splits in every tree, which limits how much averaging reduces variance. Random forests consider a random subset of features at each split, decorrelating the trees so that averaging removes more variance.</details>
5. What does an AUC of 0.8 mean?
   <details><summary>Answer</summary>If you pick a random positive example and a random negative one, the classifier scores the positive higher 80% of the time. It summarizes ranking quality across all thresholds.</details>
6. Compare LOOCV with 10-fold cross-validation.
   <details><summary>Answer</summary>LOOCV trains on $n-1$ points each time, so its estimate has low bias, but its $n$ training sets are nearly identical, so the estimate has high variance, and it requires $n$ fits. 10-fold CV has slightly more bias but lower variance and costs only 10 fits, which is why $k = 5$ or $10$ is standard.</details>
