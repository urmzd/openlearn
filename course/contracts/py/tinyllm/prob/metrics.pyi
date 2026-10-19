# contracts/py/tinyllm/prob/metrics.pyi (M07.7)
# chapter: content/courses/mathematics-for-computing/07-probability-statistics/07-logistic-regression-roc-auc-calibration.md
#
# A binary classifier and the three numbers that judge it: logistic
# regression fitted by Newton's method (iteratively reweighted least squares,
# IRLS, each step one linear solve with M03.2's LU), the area under the ROC
# curve (M01.4's trapezoid over the curve's corners), and the expected
# calibration error, with the one-parameter fix for a miscalibrated model:
# temperature scaling, fitted by M01.2's scalar newton. L6.5 fits the usage-policy head (D33,
# formats/linear-head.schema.json) with logistic_regression_fit and reports
# roc_auc and ece; ethics.04 puts them in the safety report.
#
# Labels are 0 or 1 (bool or integer arrays); all arithmetic is float64.
from numpy.typing import ArrayLike, NDArray

def logistic_regression_fit(
    X: ArrayLike, y: ArrayLike, l2: float, iters: int, fit_intercept: bool = True
) -> NDArray:
    """Weights w minimizing the penalized negative log-likelihood
        L(w) = -sum_i [y_i log p_i + (1 - y_i) log(1 - p_i)] + (l2 / 2) ||w_feat||^2,
        p_i = sigmoid(z_i), z = Xa @ w,
    where Xa is X with a column of ones appended when fit_intercept (so w has
    d + 1 entries and the intercept is LAST, w[-1]), else X itself (d
    entries). The intercept is never penalized; w_feat is every other entry.
    (This is scikit-learn's LogisticRegression with C = 1 / l2.)
    Starts at w = 0 and takes at most `iters` Newton steps:
        g = Xa^T (p - y) + l2 * w_feat        (zeros at the intercept)
        H = Xa^T diag(p (1 - p)) Xa + l2 * I_feat
        solve H delta = g with tinyllm.linalg.lu (lu, then lu_solve)
        w = w - delta
    and stops early once max|delta| <= 1e-12 * max(1, max|w|).
    sigmoid is evaluated without overflow for either sign of z.
    ValueError unless X is 2-D [n, d] and finite, y is [n] with values in
    {0, 1}, l2 >= 0 and finite, and iters is an integer >= 1; ValueError also
    when H is singular (for example l2 = 0 with a constant feature)."""

def logistic_predict_proba(
    X: ArrayLike, w: ArrayLike, fit_intercept: bool = True
) -> NDArray:
    """sigmoid(X @ w[:-1] + w[-1]) when fit_intercept, else sigmoid(X @ w):
    float64 [n] probabilities of class 1. ValueError when the shapes disagree."""

def roc_curve(scores: ArrayLike, labels: ArrayLike) -> tuple[NDArray, NDArray, NDArray]:
    """(fpr, tpr, thresholds), float64, one point per distinct score plus a
    first point (0, 0) at threshold +inf. Thresholds are the distinct scores
    in decreasing order; at threshold t the classifier says 1 when
    score >= t, so tpr = #{positives with score >= t} / P and fpr =
    #{negatives with score >= t} / N. Tied scores move fpr and tpr together
    (one diagonal step). The last point is (1, 1).
    ValueError unless scores and labels are 1-D, the same length, scores are
    finite, labels are in {0, 1}, and both classes are present."""

def roc_auc(scores: ArrayLike, labels: ArrayLike) -> float:
    """The area under roc_curve: trapezoid(tpr, fpr) (M01.4). Equal to the
    probability that a random positive outscores a random negative, ties
    counting 1/2 (the Mann-Whitney U statistic over P * N). 1.0 for a perfect
    ranking, 0.5 for constant scores. ValueError as roc_curve."""

def reliability_bins(
    probs: ArrayLike, labels: ArrayLike, n_bins: int = 15
) -> tuple[NDArray, NDArray, NDArray]:
    """(counts int64 [n_bins], confidence float64 [n_bins], accuracy float64
    [n_bins]) for the top-label reliability diagram.
    probs is [n] (the probability of class 1; the prediction is 1 when
    p >= 0.5, the confidence max(p, 1 - p)) or [n, C] rows of class
    probabilities (the prediction is the first argmax, the confidence the row
    max); labels is [n] integer classes. Bin b covers (b / n_bins,
    (b + 1) / n_bins], with index b = clip(ceil(confidence * n_bins) - 1, 0,
    n_bins - 1), so a confidence of exactly 0 falls in bin 0. confidence[b] and
    accuracy[b] are the mean confidence and the fraction of correct
    predictions in bin b, 0.0 for an empty bin.
    ValueError unless probs is finite and in [0, 1], the lengths agree, n >= 1,
    labels are valid classes, and n_bins is an integer >= 1."""

def ece(probs: ArrayLike, labels: ArrayLike, n_bins: int = 15) -> float:
    """Expected calibration error (Guo et al. 2017): sum over non-empty bins of
    (counts[b] / n) * |accuracy[b] - confidence[b]|, with the bins of
    reliability_bins. 0 for a perfectly calibrated classifier.
    ValueError as reliability_bins."""

def fit_temperature(logits: ArrayLike, labels: ArrayLike, max_iter: int = 50) -> float:
    """Temperature scaling (Guo et al. 2017): the T > 0 minimizing the mean
    NLL of softmax(logits / T) on held-out (logits [n, C], labels [n]).
    With beta = 1 / T, NLL(beta) = mean_i(logsumexp(beta z_i) - beta z_i,y_i)
    is convex; its derivative g(beta) = mean_i(E_p[z_i] - z_i,y_i) and second
    derivative g'(beta) = mean_i Var_p[z_i] (p = softmax(beta z_i)) go to
    M01.2's newton(g, g', 0.0, tol=1e-12, max_iter=max_iter), started at
    beta = 0 (uniform p, the largest curvature), and the result is 1 / beta. Dividing by T never changes the argmax, so accuracy stays
    and only the confidences move (ECE falls).
    ValueError unless logits is finite [n, C] with n >= 1, C >= 2, and
    labels are [n] classes in [0, C); ValueError when the root has beta <= 0
    (the logits rank the labels no better than chance); newton's
    RuntimeError when it does not converge in max_iter steps. Held-out data
    that the logits separate perfectly has no minimum (the NLL keeps falling
    as T -> 0): fit on data with errors in it."""

