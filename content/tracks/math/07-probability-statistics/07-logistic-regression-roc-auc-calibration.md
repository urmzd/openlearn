# Logistic regression (IRLS), ROC-AUC, calibration (ECE)

## Overview

| | |
|---|---|
| **Module** | `M07.7` · build · Python · Pass 5 · 1.5 to 2 h reading |
| **Solution** | [`course/solutions/python/tinyllm/prob/metrics.py`](../../../../course/solutions/python/tinyllm/prob/metrics.py): `logistic_regression_fit`, `logistic_predict_proba`, `roc_curve`, `roc_auc`, `reliability_bins`, `ece`, `fit_temperature` |
| **Contract** | [`course/contracts/py/tinyllm/prob/metrics.pyi`](../../../../course/contracts/py/tinyllm/prob/metrics.pyi) · the head it fits: [`formats/linear-head.schema.json`](../../../../course/contracts/formats/linear-head.schema.json) |
| **Needs** | [`M03.2` LU](../03-linear-algebra/02-gaussian-elimination-and-lu.md) (`lu`, `lu_solve`: one solve per Newton step) · [`M01.4` trapezoid](../01-calculus-1/04-definite-integrals-trapezoid-simpson.md) (the area under the ROC curve) · [`M01.2` Newton's method](../01-calculus-1/02-newtons-method.md) (`newton`: the temperature's root) · reading: [`M07.5` hypothesis tests](05-hypothesis-tests.md) |
| **Used by** | later: `L6.5` fits the usage-policy linear head and reports its AUC and ECE (D33) · `L3.5` scores probes · `ethics.04` puts both numbers in the safety report; part of the Pass 5 evaluation layer |
| **Optional depth** | Hastie, Tibshirani, Friedman, *The Elements of Statistical Learning* (free), section 4.4 (logistic regression and IRLS); Fawcett, "An introduction to ROC analysis" (2006); Guo, Pleiss, Sun, Weinberger, "On Calibration of Modern Neural Networks" (2017) |

## Key Takeaways

- **Logistic regression** models $P(y = 1 \mid x) = \sigma(w \cdot x)$; its loss is convex, so the minimum is where the gradient $X^\top(p - y) + \lambda w$ vanishes.
- **Newton's method** on that loss is IRLS: each step is one linear solve with the Hessian $X^\top S X + \lambda I$, $S = \operatorname{diag}(p(1-p))$, and the digits double per step.
- The **ROC curve** sweeps every threshold; its area, the **AUC**, is the probability a random positive outscores a random negative, ties counting half.
- **Calibration** asks whether "0.8" means right 80% of the time; the **ECE** is the count-weighted gap between confidence and accuracy over bins.
- A ranking metric and a calibration metric answer different questions: sharpening probabilities leaves the AUC unchanged and wrecks the ECE.

## How to read this chapter

Sections 1 and 2 derive logistic regression and its Newton (IRLS) fit, the ROC curve and AUC, the ECE, and temperature scaling from first principles, and section 3 works each by hand. Section 4 fixes the interface, and section 5 walks through the solution line by line and explains why each choice is the right one. Try to predict each step of the solution before reading it.

---

## 1. Why now

Pass 5 gives the system its first classifiers. `L6.5` puts a classification head on BERT and ELECTRA, and the gateway's usage policy (D33) is a linear head over the engine's embeddings: a dot product and a sigmoid that Go evaluates on every request. That head has to be fitted, and gradient descent with a learning rate to tune is the wrong tool for a convex problem with a few hundred weights: Newton's method solves it in five or six steps, each step a linear system solved by `M03.2`'s LU. Once fitted, the head needs two numbers before anyone trusts it. Does it rank unsafe requests above safe ones (AUC)? And when it says 0.9, is it right 90% of the time, so a threshold of 0.9 means what the operator thinks (ECE)? This module builds the fitter and both metrics.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $X$ | feature matrix, one row per example | `float64[n, d]` |
| $X_a$ | $X$ with a column of ones appended (the intercept column, last) | `float64[n, d+1]` |
| $y_i \in \{0, 1\}$ | labels | `float64[n]` |
| $w$ | weights; with an intercept, $w_{d}$ (the last) is the bias $b$ | `float64[d+1]` |
| $z_i = x_i \cdot w$ | the logit (log-odds) of example $i$ | `float64[n]` |
| $\sigma(z) = 1/(1 + e^{-z})$ | the sigmoid | |
| $p_i = \sigma(z_i)$ | predicted $P(y_i = 1)$ | `float64[n]` |
| $\lambda$ (`l2`) | ridge penalty on the feature weights | `float` $\ge 0$ |
| $g$, $H$ | gradient and Hessian of the loss | `[d+1]`, `[d+1, d+1]` |
| $S$ | $\operatorname{diag}(p_i (1 - p_i))$ | `[n, n]` (never formed) |
| TPR, FPR | true and false positive rates at a threshold | |
| $B$ (`n_bins`) | number of confidence bins | `int` |

### 2.1 The model and its loss

A linear score $z = w \cdot x$ can be any real number; a probability must lie in $(0, 1)$. The sigmoid maps one to the other, and its inverse is the log-odds: $z = \log \frac{p}{1 - p}$. So logistic regression says the log-odds are linear in the features; a unit step in feature $j$ multiplies the odds by $e^{w_j}$. Fitting maximizes the likelihood $\prod_i p_i^{y_i}(1 - p_i)^{1 - y_i}$, or equivalently minimizes

$$L(w) = -\sum_i \bigl[y_i \log p_i + (1 - y_i) \log(1 - p_i)\bigr] + \tfrac{\lambda}{2}\lVert w_{\text{feat}} \rVert^2 .$$

The derivative of the bracket with respect to $z_i$ is just $p_i - y_i$ (because $\sigma' = \sigma(1 - \sigma)$; `S-M07d` q7 asks for it), so by the chain rule $g = X_a^\top (p - y) + \lambda w_{\text{feat}}$. Differentiating once more, $H = X_a^\top S X_a + \lambda I_{\text{feat}}$. For any vector $v$, $v^\top H v = \sum_i p_i(1 - p_i)(x_i \cdot v)^2 + \lambda \lVert v_{\text{feat}} \rVert^2 \ge 0$: the loss is **convex**, every stationary point is the global minimum (the proof is `S-M07d` q9). With an intercept, the intercept row of $g = 0$ reads $\sum_i (p_i - y_i) = 0$: the fitted probabilities average to the base rate.

### 2.2 Fitting by Newton's method

`M01.2` found a root of $f$ by $x \leftarrow x - f(x)/f'(x)$. Minimizing $L$ is finding a root of its gradient, so the step is $w \leftarrow w - H^{-1} g$: solve $H \delta = g$ (one `lu` and one `lu_solve`) and subtract. Expanding $w - H^{-1}g$ shows each step is a weighted least-squares solve with weights $p_i(1 - p_i)$ that change every iteration, hence **iteratively reweighted least squares**. Near the optimum Newton converges **quadratically**: the error is roughly squared each step, so $10^{-2}$ becomes $10^{-4}$, then $10^{-8}$, then rounding. Five or six steps from $w = 0$ are typical; the solution stops once a step is below $10^{-12}$ relative. Without a penalty and with perfectly separable data there is no finite minimum (the weights grow forever), which is one reason the head is always fitted with $\lambda > 0$.

The sigmoid needs care: $e^{-z}$ overflows for $z < -709$ and $e^{z}/(1 + e^{z})$ is $\infty/\infty$ for $z > 709$. Computing $e^{-|z|}$, which is at most 1, and choosing the formula by the sign of $z$ is exact everywhere.

### 2.3 Regularization

The penalty $\frac{\lambda}{2}\lVert w_{\text{feat}} \rVert^2$ pulls feature weights toward 0; it adds $\lambda$ to the Hessian diagonal, which also makes $H$ safely invertible. The **intercept is not penalized**: shrinking it toward 0 would bias every probability toward 0.5 instead of toward the base rate. With $\lambda \to \infty$ the features are switched off and $\sigma(b)$ becomes the fraction of positives. This is scikit-learn's convention with $C = 1/\lambda$.

### 2.4 The ROC curve

A threshold $t$ turns scores into decisions: say 1 when score $\ge t$. Then TPR $= \#\{\text{positives} \ge t\}/P$ and FPR $= \#\{\text{negatives} \ge t\}/N$. Lowering $t$ from $+\infty$ to below the smallest score walks from $(0, 0)$ to $(1, 1)$: each positive passed is a step up, each negative a step right. That staircase is the **ROC curve**, one corner per distinct score. Several examples with the **same score** cannot be separated by any threshold, so they enter together: a diagonal step. Stepping through them one at a time would make the curve depend on the order of the input.

### 2.5 AUC is a ranking probability

The area under the staircase, `trapezoid(tpr, fpr)` from `M01.4`, has a meaning: each up-step at horizontal position FPR contributes the fraction of negatives already below that positive. Summed, the AUC is the fraction of (positive, negative) pairs in which the positive scores higher, a tie counting $\frac{1}{2}$ (the diagonal step's triangle). That is the Mann-Whitney $U$ statistic divided by $PN$. Consequences: AUC ignores the scores' scale (any increasing transform leaves it unchanged), flipping the labels gives $1 - \text{AUC}$, random scores give 0.5.

### 2.6 Calibration and ECE

A classifier is **calibrated** if, among the predictions made with confidence $c$, a fraction $c$ is correct. Group the predictions into $B$ equal-width bins of confidence, $(b/B, (b + 1)/B]$, and compare each bin's mean confidence with its accuracy: plotted, that is the reliability diagram. The **expected calibration error** is the count-weighted mean gap:

$$\text{ECE} = \sum_{b} \frac{n_b}{n}\,\lvert \text{acc}_b - \text{conf}_b \rvert .$$

For a binary probability $p$ the prediction is 1 when $p \ge 0.5$ and the confidence is $\max(p, 1 - p)$; with class probabilities, the top class and its probability (the "top-label" ECE of Guo et al.). Even a perfectly calibrated classifier has an ECE of a few hundredths from sampling noise in each bin, which is why a sensible check compares calibrated against overconfident rather than against 0.

### 2.7 Temperature scaling

A model that ranks well but is overconfident has one cheap fix (Guo et al. 2017): divide every logit row by one temperature $T > 0$ fitted on held-out data. The argmax cannot change, so accuracy stays; only the confidences move. With $\beta = 1/T$ and $p = \operatorname{softmax}(\beta z_i)$, the mean NLL is

$$\ell(\beta) = \frac1n \sum_i \Bigl[\log \textstyle\sum_c e^{\beta z_{ic}} - \beta z_{i y_i}\Bigr], \qquad \ell'(\beta) = \frac1n \sum_i \bigl(\mathbb{E}_p[z_i] - z_{i y_i}\bigr), \qquad \ell''(\beta) = \frac1n \sum_i \operatorname{Var}_p[z_i] \ge 0 .$$

So $\ell$ is convex in $\beta$, and its minimum is the root of a scalar function whose derivative is known in closed form: exactly what `M01.2`'s `newton(f, df, x0)` solves. `fit_temperature` calls it with $f = \ell'$, $df = \ell''$, and $x_0 = 0$. The start matters: at $\beta = 0$ the softmax is uniform and the curvature is largest, so the steps climb to the root; from $\beta = 1$ a peaked softmax has almost no curvature, and for a model three times too confident the first step lands thousands of units away. A root at $\beta \le 0$ means the logits rank the labels no better than chance, and there is no temperature to report.

## 3. Worked example by hand

**One Newton step.** One feature, no intercept: $x = (1, -1)$, $y = (1, 0)$, $\lambda = 1$. At $w = 0$ both $p_i = \frac{1}{2}$.

| Quantity | Computation | Value |
|---|---|---|
| $g$ | $(\tfrac12 - 1)(1) + (\tfrac12 - 0)(-1) + 1 \cdot 0$ | $-1$ |
| $H$ | $\tfrac14 \cdot 1 + \tfrac14 \cdot 1 + 1$ | $\tfrac32$ |
| $w_1$ | $0 - (-1)/\tfrac32$ | $\tfrac23$ |

The optimum satisfies $g = 0$: $2(1 - \sigma(w)) = w$, so $w^\star = 0.6748\ldots$; one step already got within 0.008.

**ROC and AUC.** Scores $(0.9, 0.7, 0.6, 0.4, 0.3)$ with labels $(1, 0, 1, 1, 0)$: $P = 3$, $N = 2$.

| threshold | $+\infty$ | 0.9 | 0.7 | 0.6 | 0.4 | 0.3 |
|---|---|---|---|---|---|---|
| (FPR, TPR) | (0, 0) | (0, 1/3) | (1/2, 1/3) | (1/2, 2/3) | (1/2, 1) | (1, 1) |

The area is $\frac12 \cdot \frac13 + \frac12 \cdot 1 = \frac23$. Check by pairs: positive 0.9 beats both negatives, 0.6 beats only 0.3, 0.4 beats only 0.3: 4 of 6 pairs.

**ECE.** Probabilities $(0.9, 0.8, 0.3, 0.6)$, labels $(1, 0, 0, 1)$, $B = 5$. Predictions $(1, 1, 0, 1)$, confidences $(0.9, 0.8, 0.7, 0.6)$, correct $(\text{yes}, \text{no}, \text{yes}, \text{yes})$. Bin indices $\lceil 5c \rceil - 1$: $4, 3, 3, 2$. Bin 4: conf 0.9, acc 1, gap 0.1; bin 3: conf 0.75, acc 0.5, gap 0.25; bin 2: conf 0.6, acc 1, gap 0.4. ECE $= \frac14 (0.1) + \frac24 (0.25) + \frac14 (0.4) = 0.25$. These three examples are the first three cases in section 4.

**Temperature.** Three items with logits $(0, 1)$, labels $(1, 1, 0)$. Here $\mathbb{E}_p[z] = \sigma(\beta)$, so $\ell'(\beta) = \sigma(\beta) - \frac23$: the calibrated confidence is the observed $\frac23$, $\beta = \ln 2$, and $T = 1/\ln 2 = 1.4427$. With logits $(0, 2)$ the same confidence needs $2\beta = \ln 2$, so $T$ doubles.

## 4. The interface

```python
def logistic_regression_fit(X: ArrayLike, y: ArrayLike, l2: float, iters: int, fit_intercept: bool = True) -> NDArray: ...
def logistic_predict_proba(X: ArrayLike, w: ArrayLike, fit_intercept: bool = True) -> NDArray: ...
def roc_curve(scores: ArrayLike, labels: ArrayLike) -> tuple[NDArray, NDArray, NDArray]: ...   # fpr, tpr, thresholds
def roc_auc(scores: ArrayLike, labels: ArrayLike) -> float: ...
def reliability_bins(probs: ArrayLike, labels: ArrayLike, n_bins: int = 15) -> tuple[NDArray, NDArray, NDArray]: ...
def ece(probs: ArrayLike, labels: ArrayLike, n_bins: int = 15) -> float: ...
def fit_temperature(logits: ArrayLike, labels: ArrayLike, max_iter: int = 50) -> float: ...  # T, by M01.2's newton
```

The intercept is the **last** weight, the layout `L6.5` writes into the linear head's `b`. `roc_curve` starts at $(0, 0)$ with threshold $+\infty$ and has one more point per distinct score; `roc_auc` is `trapezoid(tpr, fpr)`. Bins are $(b/B, (b+1)/B]$ with index $\operatorname{clip}(\lceil cB \rceil - 1, 0, B - 1)$.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_newton_step` | unit, smoke | one step gives $2/3$; converged $w = 2(1 - \sigma(w))$ | you and the tests agree on $g$ and $H$ |
| `test_hand_example_roc_auc` | unit, smoke | the six corners and $2/3$ | `L6.5`'s reported AUC |
| `test_hand_example_ece` | unit, smoke | bins $(0, 0, 1, 2, 1)$ and ECE 0.25 | the safety report's calibration line |
| `test_logistic_matches_scipy` | golden | IRLS equals scipy's BFGS minimizer on four datasets | the policy head is the right fit |
| `test_fit_is_a_stationary_point` | property | gradient below $10^{-9}$; mean $p$ = base rate | section 2.1 |
| `test_newton_converges_quadratically` | property | errors square per step; 8 steps equal 50 | section 2.2 |
| `test_l2_shrinks_feature_weights` | property | weight norm falls with $\lambda$; intercept free | section 2.3 |
| `test_sigmoid_never_overflows` | boundary | $z = \pm 1000$ gives exactly 1 and 0, never NaN | embeddings with large norms |
| `test_predict_proba_intercept_last` | unit | the intercept is the last weight | the linear-head format |
| `test_roc_matches_brute_force` | golden | corners and thresholds on continuous and tied scores | section 2.4 |
| `test_auc_is_mann_whitney` | golden | AUC = $U/(PN)$ | section 2.5 |
| `test_auc_properties` | property | monotone invariance, label flip $1 - \text{AUC}$, 1 and 0.5 | AUC is about ranking only |
| `test_ties_take_a_diagonal_step` | boundary | two tied examples give 0.5 in either order | quantized scores tie often |
| `test_reliability_and_ece_match_reference` | golden | bins and ECE, binary and 3-class | section 2.6 |
| `test_ece_calibrated_vs_overconfident` | statistical | calibrated under 0.04, sharpened above 0.1 | ECE measures the right gap |
| `test_bin_edges` | boundary | 0.5, 0.75, 1.0, and 0 land in the right bins | cross-language agreement |
| `test_multiclass_top_label` | unit | first argmax, row max confidence | `L6.5`'s multi-class heads |
| `test_rejects_bad_arguments` | boundary | bad labels, shapes, penalties, one-class ROC, bad probabilities raise | bugs surface at the call |
| `test_hand_example_temperature` | unit, smoke | $T = 1/\ln 2$ for gap 1, twice that for gap 2 | section 3 |
| `test_temperature_calibrates_an_overconfident_model` | property | logits 3 times too sharp give $T$ near 3, a true NLL minimum, the same argmax, half the ECE | section 2.7 |
| `test_temperature_rejects_bad_input` | boundary | 1-D logits, bad classes, NaN, worse than chance raise | no negative temperatures |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. $e^{z}/(1 + e^{z})$ or $1/(1 + e^{-z})$ without care | NaN probabilities for large logits | `test_sigmoid_never_overflows` (mutant `s01`) |
| 2. penalizing the intercept | probabilities pulled toward 0.5; base rate lost | `test_l2_shrinks_feature_weights` (mutant `s02`) |
| 3. the gradient of the likelihood instead of the loss | Newton walks uphill and diverges | `test_hand_example_newton_step` (mutant `s03`) |
| 4. a Hessian without the weights $p(1-p)$ | linear, not quadratic, convergence; wrong after few steps | `test_newton_converges_quadratically` (mutant `s04`) |
| 5. stepping through tied scores one at a time | the AUC depends on the input order | `test_ties_take_a_diagonal_step` (mutant `s05`) |
| 6. sorting scores ascending | AUC reported as $1 - \text{AUC}$ | `test_hand_example_roc_auc` (mutant `s06`) |
| 7. a curve that does not start at $(0, 0)$ | the first corner is missing from plots and thresholds | `test_roc_matches_brute_force` (mutant `s07`) |
| 8. an unweighted mean over bins | one stray prediction in an empty corner dominates the ECE | `test_reliability_and_ece_match_reference` (mutant `s08`) |
| 9. binary confidence $p$ instead of $\max(p, 1 - p)$ | confident negatives counted as unconfident | `test_hand_example_ece` (mutant `s09`) |
| 10. bins $[\text{lo}, \text{hi})$ by floor | edge values one bin up; another number than Go and the paper | `test_bin_edges` (mutant `s10`) |
| 11. the intercept column first | the head's `b` holds a feature weight | `test_predict_proba_intercept_last` (mutant `s11`) |
| 12. returning $\beta$ instead of $T = 1/\beta$ | an overconfident model gets sharper, not softer | `test_hand_example_temperature` (mutant `s12`) |
| 13. $\ell'$ without the label's logit | the root no longer depends on the labels | `test_hand_example_temperature` (mutant `s13`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M03.2` | `lu` and `lu_solve` solve $H \delta = g$ each Newton step |
| Back | `M01.4` | `trapezoid(tpr, fpr)` is the AUC |
| Back | `M01.2` | `newton(f, df, 0.0)` finds the root of $\ell'(\beta)$ for `fit_temperature`; IRLS is the same method on a gradient vector |
| Forward | `L6.5` | fits the usage-policy head on embeddings and exports it as `linear-head.schema.json` with its AUC and ECE |
| Forward | `L3.5` | probing classifiers over frozen representations |
| Forward | `ethics.04` | the safety report's ROC and reliability diagram |
| Forward | `S-M07d` | q5 to q9: regression, odds, the loss gradient, AUC by hand, and the convexity proof |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `logistic_regression_fit` | `sklearn.linear_model.LogisticRegression` | L-BFGS and Newton-CG solvers that never form $H$, L1 penalties, multinomial fits | `sklearn/linear_model/_logistic.py` |
| IRLS | `statsmodels` GLM | the same IRLS for every exponential family, with standard errors from $H^{-1}$ | `statsmodels/genmod/generalized_linear_model.py` |
| `roc_curve`, `roc_auc` | `sklearn.metrics.roc_curve`, `roc_auc_score` | dropping collinear corners, multiclass one-vs-rest averaging | `sklearn/metrics/_ranking.py` |
| `ece` | temperature scaling | one scalar fitted on held-out logits that fixes most overconfidence | Guo et al. (2017), section 4.2 |
| the policy head | Llama Guard, OpenAI moderation | a classifier model instead of a linear head, with per-category thresholds | Inan et al., "Llama Guard" (2023) |
