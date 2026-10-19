"""Logistic regression (IRLS), ROC-AUC, calibration (ECE) (M07.7).

Logistic regression models P(y = 1 | x) as sigmoid(w . x). Its negative
log-likelihood is convex, so Newton's method finds the minimum in a handful
of steps, and each Newton step is a weighted least-squares solve (hence
"iteratively reweighted least squares"). A fitted classifier is then judged
twice: by how well its scores rank positives above negatives (the area under
the ROC curve, threshold free) and by whether its probabilities mean what
they say (the expected calibration error).

Contract: contracts/py/tinyllm/prob/metrics.pyi. L6.5 fits the usage-policy
linear head with logistic_regression_fit; L6.5, L3.5, and ethics.04 report
roc_auc and ece.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tinyllm.linalg.lu import lu, lu_solve
from tinyllm.num.integrate import trapezoid
from tinyllm.num.newton import newton


def _sigmoid(z: NDArray) -> NDArray:
    # exp of a non-positive number never overflows: use e^{-|z|} on both sides.
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def _binary(y: ArrayLike, n: int) -> NDArray:
    a = np.asarray(y)
    if a.ndim != 1 or a.size != n:
        raise ValueError(f"labels must be 1-D of length {n}, got shape {a.shape}")
    f = a.astype(np.float64)
    if not np.all((f == 0.0) | (f == 1.0)):
        raise ValueError("labels must be 0 or 1")
    return f


def _design(X: ArrayLike, fit_intercept: bool) -> NDArray:
    a = np.asarray(X, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"X must be 2-D [n, d], got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError("X must be finite")
    if fit_intercept:
        a = np.concatenate([a, np.ones((a.shape[0], 1))], axis=1)
    return a


def logistic_regression_fit(
    X: ArrayLike, y: ArrayLike, l2: float, iters: int, fit_intercept: bool = True
) -> NDArray:
    Xa = _design(X, fit_intercept)
    n, k = Xa.shape
    t = _binary(y, n)
    l2 = float(l2)
    if not (math.isfinite(l2) and l2 >= 0.0):
        raise ValueError(f"l2 must be finite and >= 0, got {l2}")
    if int(iters) != iters or iters < 1:
        raise ValueError(f"iters must be an integer >= 1, got {iters!r}")
    pen = np.full(k, l2)
    if fit_intercept:
        pen[-1] = 0.0  # the intercept is never shrunk toward 0
    w = np.zeros(k)
    for _ in range(int(iters)):
        p = _sigmoid(Xa @ w)
        g = Xa.T @ (p - t) + pen * w
        # Hessian: X^T S X with S = diag(p (1 - p)), the Bernoulli variances.
        H = Xa.T @ (Xa * (p * (1.0 - p))[:, None]) + np.diag(pen)
        P, L, U = lu(H)
        delta = lu_solve(P, L, U, g)
        w = w - delta
        if np.max(np.abs(delta)) <= 1e-12 * max(1.0, float(np.max(np.abs(w)))):
            break
    return w


def logistic_predict_proba(
    X: ArrayLike, w: ArrayLike, fit_intercept: bool = True
) -> NDArray:
    Xa = _design(X, fit_intercept)
    wa = np.asarray(w, dtype=np.float64)
    if wa.shape != (Xa.shape[1],):
        raise ValueError(f"w must have shape ({Xa.shape[1]},), got {wa.shape}")
    return _sigmoid(Xa @ wa)


def roc_curve(scores: ArrayLike, labels: ArrayLike) -> tuple[NDArray, NDArray, NDArray]:
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 1:
        raise ValueError(f"scores must be 1-D, got shape {s.shape}")
    if not np.all(np.isfinite(s)):
        raise ValueError("scores must be finite")
    t = _binary(labels, s.size)
    P = float(np.sum(t))
    N = float(s.size) - P
    if P == 0 or N == 0:
        raise ValueError("ROC needs both classes among the labels")
    order = np.argsort(-s, kind="stable")
    s, t = s[order], t[order]
    tp = np.cumsum(t)
    fp = np.cumsum(1.0 - t)
    # One point per distinct score: the last index of each run of ties, so
    # tied scores enter together (a diagonal step).
    last = np.r_[np.nonzero(np.diff(s))[0], s.size - 1]
    fpr = np.r_[0.0, fp[last] / N]
    tpr = np.r_[0.0, tp[last] / P]
    thresholds = np.r_[np.inf, s[last]]
    return fpr, tpr, thresholds


def roc_auc(scores: ArrayLike, labels: ArrayLike) -> float:
    fpr, tpr, _ = roc_curve(scores, labels)
    return trapezoid(tpr, fpr)


def reliability_bins(
    probs: ArrayLike, labels: ArrayLike, n_bins: int = 15
) -> tuple[NDArray, NDArray, NDArray]:
    p = np.asarray(probs, dtype=np.float64)
    if int(n_bins) != n_bins or n_bins < 1:
        raise ValueError(f"n_bins must be an integer >= 1, got {n_bins!r}")
    n_bins = int(n_bins)
    if p.ndim not in (1, 2) or p.shape[0] == 0:
        raise ValueError(
            f"probs must be [n] or [n, C] with n >= 1, got shape {p.shape}"
        )
    if not np.all(np.isfinite(p)) or np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("probs must be finite and in [0, 1]")
    y = np.asarray(labels)
    if y.ndim != 1 or y.shape[0] != p.shape[0]:
        raise ValueError(f"labels must be [{p.shape[0]}], got shape {y.shape}")
    if p.ndim == 1:
        t = _binary(y, p.size)
        pred = (p >= 0.5).astype(np.float64)
        conf = np.maximum(p, 1.0 - p)
        correct = pred == t
    else:
        yf = y.astype(np.float64)
        if not np.all((yf == np.round(yf)) & (yf >= 0) & (yf < p.shape[1])):
            raise ValueError(f"labels must be classes in [0, {p.shape[1]})")
        pred = np.argmax(p, axis=1)
        conf = np.max(p, axis=1)
        correct = pred == yf.astype(np.int64)
    idx = np.clip(np.ceil(conf * n_bins).astype(np.int64) - 1, 0, n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins).astype(np.int64)
    csum = np.bincount(idx, weights=conf, minlength=n_bins)
    asum = np.bincount(idx, weights=correct.astype(np.float64), minlength=n_bins)
    safe = np.maximum(counts, 1)
    return counts, csum / safe, asum / safe


def ece(probs: ArrayLike, labels: ArrayLike, n_bins: int = 15) -> float:
    counts, conf, acc = reliability_bins(probs, labels, n_bins)
    n = float(np.sum(counts))
    return float(np.sum(counts / n * np.abs(acc - conf)))


def fit_temperature(logits: ArrayLike, labels: ArrayLike, max_iter: int = 50) -> float:
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim != 2 or z.shape[0] == 0 or z.shape[1] < 2:
        raise ValueError(
            f"logits must be [n, C] with n >= 1 and C >= 2, got shape {z.shape}"
        )
    if not np.all(np.isfinite(z)):
        raise ValueError("logits must be finite")
    y = np.asarray(labels)
    yf = y.astype(np.float64) if y.ndim == 1 else np.zeros(0)
    if (
        y.ndim != 1
        or y.shape[0] != z.shape[0]
        or not np.all((yf == np.round(yf)) & (yf >= 0) & (yf < z.shape[1]))
    ):
        raise ValueError(f"labels must be [{z.shape[0]}] classes in [0, {z.shape[1]})")
    zy = z[np.arange(z.shape[0]), yf.astype(np.int64)]

    def moments(beta: float) -> tuple[NDArray, NDArray]:
        a = beta * z
        p = np.exp(a - a.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        mean = (p * z).sum(axis=1)
        return mean, (p * z * z).sum(axis=1) - mean * mean

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
        raise ValueError(
            f"no positive temperature: the NLL is smallest at 1/T = {beta}"
        )
    return 1.0 / beta
