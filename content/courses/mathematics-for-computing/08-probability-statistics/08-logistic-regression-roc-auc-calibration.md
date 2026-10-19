# Logistic regression (IRLS), ROC-AUC, calibration (ECE)

Sooner or later every system that generates text also has to classify it. A fine-tuned encoder gets a classification head; a gateway's usage policy is a linear head over the engine's embeddings, a dot product and a sigmoid evaluated on every request. That head has to be fitted, and gradient descent with a learning rate to tune is the wrong tool for a convex problem with a few hundred weights. Newton's method solves it in five or six steps, each step one linear system solved by LU.

Once fitted, the head needs two numbers before anyone should trust it. Does it rank unsafe requests above safe ones? That is the area under the ROC curve, the AUC. And when it says 0.9, is it right 90% of the time, so that a threshold of 0.9 means what the operator thinks it means? That is the expected calibration error, the ECE. This section builds the fitter and both metrics, plus temperature scaling, the standard one-parameter fix for an overconfident model. The code is [`course/solutions/python/tinyllm/prob/metrics.py`](../../../../course/solutions/python/tinyllm/prob/metrics.py) (`logistic_regression_fit`, `logistic_predict_proba`, `roc_curve`, `roc_auc`, `reliability_bins`, `ece`, `fit_temperature`); its contract is [`course/contracts/py/tinyllm/prob/metrics.pyi`](../../../../course/contracts/py/tinyllm/prob/metrics.pyi), and the head it fits follows [`formats/linear-head.schema.json`](../../../../course/contracts/formats/linear-head.schema.json).

This section reuses three earlier tools: `lu` and `lu_solve` from [Gaussian elimination and LU with partial pivoting](../04-linear-algebra/02-gaussian-elimination-and-lu.md) (one solve per Newton step), `trapezoid` from [Definite Integrals: Trapezoid and Simpson](../02-calculus-1/04-definite-integrals-trapezoid-simpson.md) (the area under the ROC curve), and `newton` from [Newton's Method](../02-calculus-1/02-newtons-method.md) (the temperature's root). It also helps to have read [Hypothesis tests: paired and two-sample permutation, McNemar, Holm](06-hypothesis-tests.md), the other half of the evaluation toolkit.

> **Learning objectives**
> - Derive the logistic loss, its gradient $X^\top(p - y) + \lambda w$, and its Hessian, and prove the loss is convex.
> - Fit logistic regression by Newton's method (IRLS) with an unpenalized intercept and a numerically stable sigmoid.
> - Build the ROC curve with ties handled as diagonal steps, and explain why its area is a ranking probability.
> - Compute reliability bins and the ECE, and explain why ranking and calibration are different questions.
> - Fit a temperature by Newton's method on a convex scalar loss and explain why the starting point matters.

## Notation

The section uses these symbols throughout.

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

## The model and its loss

A linear score $z = w \cdot x$ can be any real number, but a probability must lie in $(0, 1)$. The sigmoid maps one to the other, and its inverse is the log-odds: $z = \log \frac{p}{1 - p}$. So logistic regression says the log-odds are linear in the features; a unit step in feature $j$ multiplies the odds by $e^{w_j}$. Fitting maximizes the likelihood $\prod_i p_i^{y_i}(1 - p_i)^{1 - y_i}$, or equivalently minimizes

$$L(w) = -\sum_i \bigl[y_i \log p_i + (1 - y_i) \log(1 - p_i)\bigr] + \tfrac{\lambda}{2}\lVert w_{\text{feat}} \rVert^2 .$$

The derivative of the bracket with respect to $z_i$ is just $p_i - y_i$ (because $\sigma' = \sigma(1 - \sigma)$; question 7 of [problem set D](12-problem-set-d.md) asks you to show it), so by the chain rule $g = X_a^\top (p - y) + \lambda w_{\text{feat}}$. Differentiating once more, $H = X_a^\top S X_a + \lambda I_{\text{feat}}$. For any vector $v$, $v^\top H v = \sum_i p_i(1 - p_i)(x_i \cdot v)^2 + \lambda \lVert v_{\text{feat}} \rVert^2 \ge 0$: the loss is **convex**, and every stationary point is the global minimum (the proof is question 9 of the same set). With an intercept, the intercept row of $g = 0$ reads $\sum_i (p_i - y_i) = 0$: the fitted probabilities average to the base rate.

## Fitting by Newton's method

Convexity tells you there is one minimum; Newton's method tells you how to reach it fast. [Newton's Method](../02-calculus-1/02-newtons-method.md) found a root of $f$ by $x \leftarrow x - f(x)/f'(x)$. Minimizing $L$ is finding a root of its gradient, so the step is $w \leftarrow w - H^{-1} g$: solve $H \delta = g$ (one `lu` and one `lu_solve`) and subtract. Expanding $w - H^{-1}g$ shows each step is a weighted least-squares solve with weights $p_i(1 - p_i)$ that change every iteration, hence the name **iteratively reweighted least squares** (IRLS). Near the optimum Newton converges **quadratically**: the error is roughly squared each step, so $10^{-2}$ becomes $10^{-4}$, then $10^{-8}$, then rounding. Five or six steps from $w = 0$ are typical; the solution stops once a step is below $10^{-12}$ relative. Without a penalty and with perfectly separable data there is no finite minimum (the weights grow forever), which is one reason the head is always fitted with $\lambda > 0$.

The sigmoid itself needs care. $e^{-z}$ overflows for $z < -709$, and $e^{z}/(1 + e^{z})$ is $\infty/\infty$ for $z > 709$. Computing $e^{-|z|}$, which is at most 1, and choosing the formula by the sign of $z$ is exact everywhere.

## Regularization

The penalty $\frac{\lambda}{2}\lVert w_{\text{feat}} \rVert^2$ pulls feature weights toward 0, and it adds $\lambda$ to the Hessian diagonal, which also makes $H$ safely invertible. The **intercept is not penalized**: shrinking it toward 0 would bias every probability toward 0.5 instead of toward the base rate. With $\lambda \to \infty$ the features are switched off and $\sigma(b)$ becomes the fraction of positives. This is scikit-learn's convention with $C = 1/\lambda$.

## The ROC curve

With a fitted model in hand, the next question is how to judge it, starting with ranking. A threshold $t$ turns scores into decisions: say 1 when score $\ge t$. Then TPR $= \#\{\text{positives} \ge t\}/P$ and FPR $= \#\{\text{negatives} \ge t\}/N$. Lowering $t$ from $+\infty$ to below the smallest score walks from $(0, 0)$ to $(1, 1)$: each positive passed is a step up, each negative a step right. That staircase is the **ROC curve**, with one corner per distinct score. Several examples with the **same score** cannot be separated by any threshold, so they enter together as a diagonal step. Stepping through them one at a time would make the curve depend on the order of the input.

## AUC is a ranking probability

The area under the staircase, `trapezoid(tpr, fpr)`, has a meaning. Each up-step at horizontal position FPR contributes the fraction of negatives already below that positive. Summed, the AUC is the fraction of (positive, negative) pairs in which the positive scores higher, with a tie counting $\frac{1}{2}$ (the diagonal step's triangle). That is the Mann-Whitney $U$ statistic divided by $PN$. Several consequences follow: AUC ignores the scores' scale (any increasing transform leaves it unchanged), flipping the labels gives $1 - \text{AUC}$, and random scores give 0.5.

## Calibration and ECE

Ranking says nothing about whether the numbers mean what they say. A classifier is **calibrated** if, among the predictions made with confidence $c$, a fraction $c$ is correct. Group the predictions into $B$ equal-width bins of confidence, $(b/B, (b + 1)/B]$, and compare each bin's mean confidence with its accuracy; plotted, that comparison is the reliability diagram. The **expected calibration error** is the count-weighted mean gap:

$$\text{ECE} = \sum_{b} \frac{n_b}{n}\,\lvert \text{acc}_b - \text{conf}_b \rvert .$$

For a binary probability $p$ the prediction is 1 when $p \ge 0.5$ and the confidence is $\max(p, 1 - p)$; with class probabilities, it is the top class and its probability (the "top-label" ECE of Guo et al.). Even a perfectly calibrated classifier has an ECE of a few hundredths from sampling noise in each bin, which is why a sensible check compares calibrated against overconfident rather than against 0.

The two metrics answer different questions. Sharpening probabilities (pushing them toward 0 and 1) preserves their order, so it leaves the AUC unchanged, and it can wreck the ECE.

## Temperature scaling

That observation suggests a cheap fix for a model that ranks well but is overconfident (Guo et al. 2017): divide every logit row by one temperature $T > 0$ fitted on held-out data. The argmax cannot change, so accuracy stays; only the confidences move. With $\beta = 1/T$ and $p = \operatorname{softmax}(\beta z_i)$, the mean NLL is

$$\ell(\beta) = \frac1n \sum_i \Bigl[\log \textstyle\sum_c e^{\beta z_{ic}} - \beta z_{i y_i}\Bigr], \qquad \ell'(\beta) = \frac1n \sum_i \bigl(\mathbb{E}_p[z_i] - z_{i y_i}\bigr), \qquad \ell''(\beta) = \frac1n \sum_i \operatorname{Var}_p[z_i] \ge 0 .$$

So $\ell$ is convex in $\beta$, and its minimum is the root of a scalar function whose derivative is known in closed form: exactly what `newton(f, df, x0)` solves. `fit_temperature` calls it with $f = \ell'$, $df = \ell''$, and $x_0 = 0$. The start matters. At $\beta = 0$ the softmax is uniform and the curvature is largest, so the steps climb to the root; from $\beta = 1$ a peaked softmax has almost no curvature, and for a model three times too confident the first step lands thousands of units away. A root at $\beta \le 0$ means the logits rank the labels no better than chance, and there is no temperature to report.

## Working it by hand

Each of the four ideas has a small example you can do on paper, and the tests check all of them.

**One Newton step.** Take one feature and no intercept: $x = (1, -1)$, $y = (1, 0)$, $\lambda = 1$. At $w = 0$ both $p_i = \frac{1}{2}$, and the step works out as follows.

| Quantity | Computation | Value |
|---|---|---|
| $g$ | $(\tfrac12 - 1)(1) + (\tfrac12 - 0)(-1) + 1 \cdot 0$ | $-1$ |
| $H$ | $\tfrac14 \cdot 1 + \tfrac14 \cdot 1 + 1$ | $\tfrac32$ |
| $w_1$ | $0 - (-1)/\tfrac32$ | $\tfrac23$ |

The optimum satisfies $g = 0$: $2(1 - \sigma(w)) = w$, so $w^\star = 0.6748\ldots$; one step already got within 0.008.

**ROC and AUC.** Take scores $(0.9, 0.7, 0.6, 0.4, 0.3)$ with labels $(1, 0, 1, 1, 0)$, so $P = 3$ and $N = 2$. Lowering the threshold through each score gives these corners.

| threshold | $+\infty$ | 0.9 | 0.7 | 0.6 | 0.4 | 0.3 |
|---|---|---|---|---|---|---|
| (FPR, TPR) | (0, 0) | (0, 1/3) | (1/2, 1/3) | (1/2, 2/3) | (1/2, 1) | (1, 1) |

The area is $\frac12 \cdot \frac13 + \frac12 \cdot 1 = \frac23$. Check it by pairs: positive 0.9 beats both negatives, 0.6 beats only 0.3, and 0.4 beats only 0.3, which is 4 of 6 pairs.

**ECE.** Take probabilities $(0.9, 0.8, 0.3, 0.6)$, labels $(1, 0, 0, 1)$, and $B = 5$. The predictions are $(1, 1, 0, 1)$, the confidences $(0.9, 0.8, 0.7, 0.6)$, and the correctness (yes, no, yes, yes). Bin indices $\lceil 5c \rceil - 1$ are $4, 3, 3, 2$. Bin 4 has confidence 0.9, accuracy 1, gap 0.1; bin 3 has confidence 0.75, accuracy 0.5, gap 0.25; bin 2 has confidence 0.6, accuracy 1, gap 0.4. So ECE $= \frac14 (0.1) + \frac24 (0.25) + \frac14 (0.4) = 0.25$.

**Temperature.** Take three items with logits $(0, 1)$ and labels $(1, 1, 0)$. Here $\mathbb{E}_p[z] = \sigma(\beta)$, so $\ell'(\beta) = \sigma(\beta) - \frac23$: the calibrated confidence is the observed $\frac23$, $\beta = \ln 2$, and $T = 1/\ln 2 = 1.4427$. With logits $(0, 2)$ the same confidence needs $2\beta = \ln 2$, so $T$ doubles.

## Designing the interface

Seven functions cover fitting, prediction, ranking, calibration, and temperature.

```python
def logistic_regression_fit(X: ArrayLike, y: ArrayLike, l2: float, iters: int, fit_intercept: bool = True) -> NDArray: ...
def logistic_predict_proba(X: ArrayLike, w: ArrayLike, fit_intercept: bool = True) -> NDArray: ...
def roc_curve(scores: ArrayLike, labels: ArrayLike) -> tuple[NDArray, NDArray, NDArray]: ...   # fpr, tpr, thresholds
def roc_auc(scores: ArrayLike, labels: ArrayLike) -> float: ...
def reliability_bins(probs: ArrayLike, labels: ArrayLike, n_bins: int = 15) -> tuple[NDArray, NDArray, NDArray]: ...
def ece(probs: ArrayLike, labels: ArrayLike, n_bins: int = 15) -> float: ...
def fit_temperature(logits: ArrayLike, labels: ArrayLike, max_iter: int = 50) -> float: ...  # T, by M01.2's newton
```

The intercept is the **last** weight, which is the layout the linear-head export writes into the head's `b`. `roc_curve` starts at $(0, 0)$ with threshold $+\infty$ and has one more point per distinct score; `roc_auc` is `trapezoid(tpr, fpr)`. Bins are $(b/B, (b+1)/B]$ with index $\operatorname{clip}(\lceil cB \rceil - 1, 0, B - 1)$.

## Building it step by step

The reasoning comes first, as a chain of eight decisions, and then the annotated code.

Start with a stable sigmoid: compute $e = e^{-|z|} \le 1$ and return $1/(1+e)$ for $z \ge 0$, $e/(1+e)$ otherwise. Both branches are the exact sigmoid and neither can overflow. Next, append the intercept as the last column. Then the model is $\sigma(X_a w)$ in all cases, and the last weight is the head's `b`. Third, penalize features, not the intercept: a penalty vector `pen` equal to $\lambda$ except 0 in the intercept slot enters the gradient as `pen * w` and the Hessian as `diag(pen)`.

Fourth, each IRLS iteration is one Newton step. From $w = 0$: compute $p = \sigma(X_a w)$, $g = X_a^\top(p - y) + \text{pen}\, w$, $H = X_a^\top S X_a + \operatorname{diag}(\text{pen})$, solve $H\delta = g$ with LU, and set $w \leftarrow w - \delta$. $S$ is applied by scaling the rows of $X_a$, never formed as an $n \times n$ matrix. Stop when the step is below $10^{-12}$ relative, or after `iters` steps.

Fifth, for the ROC curve, sort descending, cumulate, and keep one point per distinct score. Cumulative sums of $y$ and $1 - y$ give TP and FP at every cut; keeping only the last index of each run of equal scores makes ties enter together. Prepend $(0, 0)$ with threshold $+\infty$. Sixth, the AUC is the trapezoid of that staircase. Vertical steps add no area, horizontal ones add their height, and diagonal (tie) steps add a trapezoid, which is the $\tfrac12$ credit for tied pairs.

Seventh, reliability bins need a confidence and a correctness per example, then three bincounts. Binary: prediction $p \ge 0.5$, confidence $\max(p, 1-p)$. Multi-class: argmax and row max. The bin index is $\operatorname{clip}(\lceil cB \rceil - 1, 0, B-1)$ for right-closed bins, and the ECE weights each bin's gap by its count. Eighth, the temperature is Newton on $\ell'(\beta)$ from $\beta = 0$. The derivative and curvature are the means of $\mathbb{E}_p[z] - z_y$ and of $\operatorname{Var}_p[z]$, computed from a max-shifted softmax; reject $\beta \le 0$ and return $T = 1/\beta$.

Here are the sigmoid and the fitter.

```python
def _sigmoid(z: NDArray) -> NDArray:
    # exp of a non-positive number never overflows: use e^{-|z|} on both sides.
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def logistic_regression_fit(X, y, l2: float, iters: int, fit_intercept: bool = True) -> NDArray:
    Xa = _design(X, fit_intercept)            # 2-D, finite; a column of ones appended last
    n, k = Xa.shape
    t = _binary(y, n)                         # labels as 0.0 / 1.0
    # (l2 >= 0 and iters >= 1 checks elided)
    pen = np.full(k, l2)
    if fit_intercept:
        pen[-1] = 0.0  # the intercept is never shrunk toward 0
    w = np.zeros(k)
    for _ in range(int(iters)):
        p = _sigmoid(Xa @ w)
        g = Xa.T @ (p - t) + pen * w          # gradient of the loss (not the likelihood)
        # Hessian: X^T S X with S = diag(p (1 - p)), the Bernoulli variances.
        H = Xa.T @ (Xa * (p * (1.0 - p))[:, None]) + np.diag(pen)
        P, L, U = lu(H)                       # M03.2: one factorization per step
        delta = lu_solve(P, L, U, g)
        w = w - delta                         # Newton: w - H^{-1} g
        if np.max(np.abs(delta)) <= 1e-12 * max(1.0, float(np.max(np.abs(w)))):
            break                             # relative step test, as in M01.2
    return w
```

Trace the hand example's step through it: $X_a = [[1], [-1]]$ (no intercept), $p = (\tfrac12, \tfrac12)$, `g = 1·(-1/2) + (-1)·(1/2) + 0 = -1`, `H = 1/4 + 1/4 + 1 = 3/2`, `delta = -2/3`, so $w = 2/3$. Note that `Xa * s[:, None]` scales row $i$ by $s_i$, so `Xa.T @ (...)` is $X_a^\top S X_a$ in $O(nk^2)$ without an $n \times n$ matrix.

Prediction is short, but it checks the weight layout.

```python
def logistic_predict_proba(X, w, fit_intercept: bool = True) -> NDArray:
    Xa = _design(X, fit_intercept)
    wa = np.asarray(w, dtype=np.float64)
    if wa.shape != (Xa.shape[1],):            # a weight vector from another layout fails loudly
        raise ValueError(f"w must have shape ({Xa.shape[1]},), got {wa.shape}")
    return _sigmoid(Xa @ wa)
```

The ROC curve and its area come next.

```python
def roc_curve(scores, labels) -> tuple[NDArray, NDArray, NDArray]:
    # (1-D, finite checks elided)
    t = _binary(labels, s.size)
    P = float(np.sum(t))
    N = float(s.size) - P
    if P == 0 or N == 0:
        raise ValueError("ROC needs both classes among the labels")   # a rate over 0 examples
    order = np.argsort(-s, kind="stable")     # descending: the threshold walks down
    s, t = s[order], t[order]
    tp = np.cumsum(t)                         # positives at or above each cut
    fp = np.cumsum(1.0 - t)                   # negatives at or above each cut
    # One point per distinct score: the last index of each run of ties, so
    # tied scores enter together (a diagonal step).
    last = np.r_[np.nonzero(np.diff(s))[0], s.size - 1]
    fpr = np.r_[0.0, fp[last] / N]            # start at (0, 0), threshold +inf
    tpr = np.r_[0.0, tp[last] / P]
    thresholds = np.r_[np.inf, s[last]]
    return fpr, tpr, thresholds


def roc_auc(scores, labels) -> float:
    fpr, tpr, _ = roc_curve(scores, labels)
    return trapezoid(tpr, fpr)                # M01.4; ties get their triangle, i.e. 1/2 credit
```

On the hand example's scores (already descending, all distinct), `last` is every index, and the cumulative counts give the six corners of the table. With two tied examples of opposite labels, `last` keeps only the second, so the curve goes straight from $(0, 0)$ to $(1, 1)$ and the area is 0.5 whichever comes first in the input.

Calibration is three bincounts.

```python
def reliability_bins(probs, labels, n_bins: int = 15) -> tuple[NDArray, NDArray, NDArray]:
    # (shape, range, and label checks elided)
    if p.ndim == 1:                           # binary: p is P(y = 1)
        t = _binary(y, p.size)
        pred = (p >= 0.5).astype(np.float64)
        conf = np.maximum(p, 1.0 - p)         # confidence in the predicted class
        correct = pred == t
    else:                                     # [n, C]: top-label
        pred = np.argmax(p, axis=1)           # first argmax on ties
        conf = np.max(p, axis=1)
        correct = pred == yf.astype(np.int64)
    # Right-closed bins (b/B, (b+1)/B]; 0 goes to bin 0, 1.0 to bin B - 1.
    idx = np.clip(np.ceil(conf * n_bins).astype(np.int64) - 1, 0, n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins).astype(np.int64)
    csum = np.bincount(idx, weights=conf, minlength=n_bins)
    asum = np.bincount(idx, weights=correct.astype(np.float64), minlength=n_bins)
    safe = np.maximum(counts, 1)              # empty bins report 0, not 0/0
    return counts, csum / safe, asum / safe


def ece(probs, labels, n_bins: int = 15) -> float:
    counts, conf, acc = reliability_bins(probs, labels, n_bins)
    n = float(np.sum(counts))
    return float(np.sum(counts / n * np.abs(acc - conf)))   # count-weighted; empty bins weigh 0
```

On the hand example, confidences $(0.9, 0.8, 0.7, 0.6)$ map to bins $4, 3, 3, 2$; the counts are $(0, 0, 1, 2, 1)$, and the weighted gaps sum to $0.25$.

Finally, the temperature.

```python
def fit_temperature(logits, labels, max_iter: int = 50) -> float:
    # (shape, finiteness, and label checks elided: [n, C], C >= 2, classes in range)
    zy = z[np.arange(z.shape[0]), yf.astype(np.int64)]     # the label's logit per row

    def moments(beta: float) -> tuple[NDArray, NDArray]:
        a = beta * z
        p = np.exp(a - a.max(axis=1, keepdims=True))       # max-shifted softmax: no overflow
        p /= p.sum(axis=1, keepdims=True)
        mean = (p * z).sum(axis=1)                         # E_p[z] per row
        return mean, (p * z * z).sum(axis=1) - mean * mean # Var_p[z] per row

    # NLL(beta) = mean(logsumexp(beta z) - beta z_y) is convex in beta = 1/T:
    # its derivative is the mean of E_p[z] - z_y, its second derivative the
    # mean variance of z under p. Newton (M01.2) finds the derivative's root.
    # It starts at beta = 0 (p uniform), where the variance is largest: from
    # beta = 1 a peaked softmax has almost no curvature and the first step
    # overshoots far past the root.
    beta, _ = newton(
        lambda b: float(np.mean(moments(b)[0] - zy)),
        lambda b: float(np.mean(moments(b)[1])),
        0.0,
        tol=1e-12,
        max_iter=max_iter,
    )
    if not beta > 0:
        raise ValueError(f"no positive temperature: the NLL is smallest at 1/T = {beta}")
    return 1.0 / beta                                      # report T, not beta
```

On the hand example's three items with logits $(0, 1)$, $\mathbb{E}_p[z] = \sigma(\beta)$ for every row and the mean $z_y$ is $\tfrac23$, so `newton` finds $\sigma(\beta) = \tfrac23$, $\beta = \ln 2$, and the function returns $1/\ln 2 = 1.4427$.

The full solution is [`metrics.py`](../../../../course/solutions/python/tinyllm/prob/metrics.py). It calls [`lu.py`](../../../../course/solutions/python/tinyllm/linalg/lu.py), [`integrate.py`](../../../../course/solutions/python/tinyllm/num/integrate.py), and [`newton.py`](../../../../course/solutions/python/tinyllm/num/newton.py).

### Why this design

On cost: one IRLS step is $O(nk^2)$ to build $H$ plus $O(k^3)$ for the LU, and five or six steps suffice. `roc_curve` is $O(n \log n)$ for the sort; `reliability_bins` is $O(nC)$; each Newton step of `fit_temperature` is $O(nC)$.

Newton beats gradient descent here. For a convex problem with a few hundred weights the $k \times k$ solve is cheap, there is no learning rate to tune, and quadratic convergence reaches rounding in a handful of steps. L-BFGS (scikit-learn's default) avoids forming $H$, which matters only when $k$ is in the many thousands.

Sort-and-cumulate beats pairwise AUC. Counting all $PN$ pairs is $O(PN)$; the staircase gives the same number (ties included) in $O(n \log n)$, plus the curve and thresholds for free.

Right-closed bins and top-label confidence match Guo et al. and the Go port, so a reported ECE means the same number everywhere; the floor rule $[\text{lo}, \text{hi})$ would move edge values like 0.8 up a bin.

Two tradeoffs are accepted. Unpenalized fits on separable data have no finite optimum and stop at `iters`, so callers always pass $\lambda > 0$. And ECE with equal-width bins is noisy for small $n$ and ignores where in a bin the confidences fall; it is a coarse but standard summary.

## What can go wrong

The tests a correct solution must pass fall into a few groups. The hand examples come first: one Newton step gives $2/3$ and the converged $w$ satisfies $w = 2(1 - \sigma(w))$, so everyone agrees on $g$ and $H$; the ROC gives the six corners and AUC $2/3$; the ECE example gives bins $(0, 0, 1, 2, 1)$ and ECE 0.25; and the temperature is $T = 1/\ln 2$ for a logit gap of 1 and twice that for a gap of 2.

The fitter is checked against a golden reference and against its theory. IRLS must equal scipy's BFGS minimizer on four datasets. At the result, the gradient must be below $10^{-9}$ and the mean $p$ must equal the base rate. Errors must square per step, so 8 steps equal 50. The weight norm must fall as $\lambda$ grows while the intercept stays free. The sigmoid must give exactly 1 and 0 at $z = \pm 1000$, never NaN, since embeddings can have large norms. And the intercept must be the last weight, as the linear-head format requires.

The ranking metrics are checked the same way. The ROC corners and thresholds must match a brute-force computation on continuous and tied scores, and the AUC must equal $U/(PN)$. It must be invariant to monotone transforms, give $1 - \text{AUC}$ under a label flip, and give 1 and 0.5 in the obvious cases. Two tied examples must give 0.5 in either order, which matters because quantized scores tie often.

Calibration has golden checks for bins and ECE, binary and 3-class, and a statistical check that a calibrated model scores under 0.04 while a sharpened one scores above 0.1. Bin edges 0.5, 0.75, 1.0, and 0 must land in the right bins for cross-language agreement, and the multi-class top label uses the first argmax and the row-max confidence. Temperature scaling of logits three times too sharp must give $T$ near 3, a true NLL minimum, the same argmax, and half the ECE. Finally, bad arguments must raise: bad labels, shapes, penalties, a one-class ROC, bad probabilities, and, for the temperature, 1-D logits, bad classes, NaN, or logits worse than chance, so no negative temperature is ever reported.

The pitfalls below are the mistakes those checks catch, each paired with the line of the solution that avoids it.

> **Warning:** Computing $e^{z}/(1 + e^{z})$ or $1/(1 + e^{-z})$ without care gives NaN probabilities for large logits. `_sigmoid` uses $e^{-\lvert z \rvert}$ and picks the formula by sign.

> **Warning:** Using the gradient of the likelihood instead of the loss makes Newton walk uphill and diverge. The solution computes `g = Xa.T @ (p - t) + pen * w`.

In the fitter, penalizing the intercept pulls probabilities toward 0.5 and loses the base rate (the fix is `pen[-1] = 0.0`). A Hessian without the weights $p(1-p)$ gives linear rather than quadratic convergence and wrong answers after a few steps (the rows are scaled by `p * (1.0 - p)`). Putting the intercept column first leaves a feature weight in the head's `b` (`_design` appends the ones column last).

In the ROC code, stepping through tied scores one at a time makes the AUC depend on the input order (`last` keeps one point per distinct score). Sorting scores ascending reports $1 - \text{AUC}$ (the solution uses `np.argsort(-s, ...)`). A curve that does not start at $(0, 0)$ drops the first corner from plots and thresholds (hence `np.r_[0.0, ...]` and `np.r_[np.inf, ...]`).

In calibration, an unweighted mean over bins lets one stray prediction in an empty corner dominate the ECE (the solution weights by `counts / n * np.abs(acc - conf)`). Using binary confidence $p$ instead of $\max(p, 1 - p)$ counts confident negatives as unconfident (`conf = np.maximum(p, 1.0 - p)`). Floor-based bins $[\text{lo}, \text{hi})$ move edge values one bin up and give a different number from Go and the paper (`np.ceil(conf * n_bins) - 1`, clipped).

In temperature scaling, returning $\beta$ instead of $T = 1/\beta$ makes an overconfident model sharper, not softer (`return 1.0 / beta`), and computing $\ell'$ without the label's logit makes the root independent of the labels (`moments(b)[0] - zy`).

## Where this leads

Looking back, `lu` and `lu_solve` from [Gaussian elimination and LU](../04-linear-algebra/02-gaussian-elimination-and-lu.md) solve $H \delta = g$ in each Newton step; `trapezoid(tpr, fpr)` from [Definite Integrals: Trapezoid and Simpson](../02-calculus-1/04-definite-integrals-trapezoid-simpson.md) is the AUC; and `newton(f, df, 0.0)` from [Newton's Method](../02-calculus-1/02-newtons-method.md) finds the root of $\ell'(\beta)$ for `fit_temperature`, while IRLS is the same method on a gradient vector.

Looking forward, [Fine-tuning heads and the linear-head export](../../build-an-llm/07-objectives/05-fine-tuning-heads-and-the-linear-head-export.md) in *Build Your Own LLM System* fits the usage-policy head on embeddings and exports it as `linear-head.schema.json` with its AUC and ECE. [ELMo: biLM, ScalarMix, linear probes](../../build-an-llm/04-recurrent/05-elmo-bilm-scalarmix-probes.md) uses the same fitter for probing classifiers over frozen representations, and [Bias and safety evals](../../responsible-ai/04-bias-and-safety-evals/01-bias-and-safety-evals.md) in *Responsible AI* puts the ROC and the reliability diagram in the safety report. Questions 5 to 9 of [problem set D](12-problem-set-d.md) practise regression, odds, the loss gradient, AUC by hand, and the convexity proof.

Production libraries extend each piece. `sklearn.linear_model.LogisticRegression` (`sklearn/linear_model/_logistic.py`) adds L-BFGS and Newton-CG solvers that never form $H$, L1 penalties, and multinomial fits. The `statsmodels` GLM (`statsmodels/genmod/generalized_linear_model.py`) runs the same IRLS for every exponential family, with standard errors from $H^{-1}$. `sklearn.metrics.roc_curve` and `roc_auc_score` (`sklearn/metrics/_ranking.py`) drop collinear corners and average multiclass one-vs-rest. Temperature scaling itself is Guo et al. (2017), section 4.2: one scalar fitted on held-out logits that fixes most overconfidence. And in place of a linear policy head, systems such as Llama Guard and OpenAI moderation use a classifier model with per-category thresholds (Inan et al., "Llama Guard", 2023).

### Further reading

- Hastie, Tibshirani, and Friedman, *The Elements of Statistical Learning* (free), section 4.4 (logistic regression and IRLS).
- Fawcett, "An introduction to ROC analysis" (2006).
- Guo, Pleiss, Sun, and Weinberger, "On Calibration of Modern Neural Networks" (2017).

## Key terms

- **Logistic regression**: a model $P(y = 1 \mid x) = \sigma(w \cdot x)$ in which the log-odds are linear in the features.
- **IRLS**: iteratively reweighted least squares, Newton's method on the logistic loss, where each step is a weighted least-squares solve with weights $p_i(1 - p_i)$.
- **ROC curve**: the staircase of (FPR, TPR) pairs traced as the decision threshold falls through every distinct score.
- **AUC**: the area under the ROC curve, equal to the probability that a random positive outscores a random negative, ties counting half.
- **Calibration**: the property that predictions made with confidence $c$ are correct a fraction $c$ of the time.
- **ECE**: expected calibration error, the count-weighted mean gap between accuracy and confidence over confidence bins.
- **Temperature scaling**: dividing logits by one fitted scalar $T > 0$ to fix over- or underconfidence without changing the argmax.

## Summary

- Logistic regression models $P(y = 1 \mid x) = \sigma(w \cdot x)$; its loss is convex, so the minimum is where the gradient $X^\top(p - y) + \lambda w$ vanishes.
- Newton's method on that loss is IRLS: each step is one linear solve with the Hessian $X^\top S X + \lambda I$, $S = \operatorname{diag}(p(1-p))$, and the digits double per step.
- The ROC curve sweeps every threshold; its area, the AUC, is the probability a random positive outscores a random negative, ties counting half.
- Calibration asks whether "0.8" means right 80% of the time; the ECE is the count-weighted gap between confidence and accuracy over bins.
- A ranking metric and a calibration metric answer different questions: sharpening probabilities leaves the AUC unchanged and wrecks the ECE, and temperature scaling fixes the ECE without touching the ranking.

## Review questions

1. Why is the gradient $X_a^\top(p - y)$ and not $X_a^\top(y - p)$?

   <details><summary>Answer</summary>

   The function minimizes the negative log-likelihood; its derivative with respect to each logit is $p_i - y_i$. $X_a^\top(y - p)$ is the gradient of the likelihood, and $w - H^{-1}(\text{that})$ steps uphill: on the hand example it would give $w = -2/3$ and move away from the optimum $0.6748$.

   </details>

2. Why is the intercept's penalty set to 0?

   <details><summary>Answer</summary>

   The intercept sets the base rate: at the optimum $\sum_i (p_i - y_i) = 0$, so the mean prediction equals the fraction of positives. Shrinking it toward 0 pulls every probability toward $\sigma(0) = 0.5$, a bias unrelated to overfitting the features. As $\lambda \to \infty$ the unpenalized intercept still lets $\sigma(b)$ equal the base rate.

   </details>

3. Why does `roc_curve` keep only the last index of each run of tied scores?

   <details><summary>Answer</summary>

   No threshold can separate examples with the same score, so they must enter the curve together. Keeping intermediate points would create corners that depend on the input order: a positive listed before its tied negative gives AUC 1 for that pair, after it gives 0. Jumping to the end of the run gives a diagonal step whose trapezoid is the $\tfrac12$ credit Mann-Whitney assigns to ties.

   </details>

4. For a binary probability, why is the confidence $\max(p, 1 - p)$ and not $p$?

   <details><summary>Answer</summary>

   Calibration compares confidence in the *predicted* class with how often that prediction is right. A prediction $p = 0.1$ predicts class 0 with confidence 0.9; filing it in the 0.1 bin would compare a confident correct negative against an accuracy that measures a different event and inflate the ECE.

   </details>

5. Why does `fit_temperature` start Newton at $\beta = 0$ rather than at $\beta = 1$ (no change)?

   <details><summary>Answer</summary>

   $\ell''(\beta)$ is the mean variance of the logits under the softmax. At $\beta = 0$ the softmax is uniform and that variance is largest, so the first step is moderate and the iterates climb toward the root. At $\beta = 1$ an overconfident model's softmax is nearly one-hot, the variance is almost 0, and the step $-\ell'/\ell''$ is enormous, overshooting by thousands of units.

   </details>

6. A colleague squares every predicted probability's odds (pushing them toward 0 and 1). What happens to the AUC and to the ECE?

   <details><summary>Answer</summary>

   Squaring the odds is an increasing transform of the scores, so every pairwise comparison between a positive and a negative is unchanged and the AUC stays exactly the same. The confidences move toward 0 and 1 while the accuracies in each region stay put, so a model that was calibrated becomes overconfident and its ECE rises. That is the sense in which ranking and calibration are separate questions.

   </details>
