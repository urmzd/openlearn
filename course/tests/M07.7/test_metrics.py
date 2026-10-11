"""Course tests for M07.7: logistic regression (IRLS), ROC-AUC, calibration
(ECE) (tinyllm/prob/metrics.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/M07.7), and the chapter section it comes from.

The chapter's worked example (section 3):
  * one feature, no intercept, x = (1, -1), y = (1, 0), l2 = 1. From w = 0:
    p = (1/2, 1/2), g = -1/2 - 1/2 = -1, H = 1/4 + 1/4 + 1 = 3/2, so one
    Newton step gives w = 2/3; the optimum solves w = 2 (1 - sigmoid(w)),
    w* = 0.674...
  * scores (0.9, 0.7, 0.6, 0.4, 0.3) with labels (1, 0, 1, 1, 0): ROC corners
    (0,0) (0,1/3) (1/2,1/3) (1/2,2/3) (1/2,1) (1,1), area 2/3 = 4 of the 6
    positive-negative pairs ranked correctly.
  * probabilities (0.9, 0.8, 0.3, 0.6), labels (1, 0, 0, 1), 5 bins:
    confidences 0.9 | 0.8, 0.7 | 0.6 in bins 4 | 3 | 2, ECE =
    1/4 * 0.1 + 2/4 * 0.25 + 1/4 * 0.4 = 0.25.
  * temperature: three items with logits (0, 1), two labelled 1 and one 0.
    The calibrated confidence is 2/3, so sigmoid(1 / T) = 2/3, 1 / T = ln 2,
    T = 1 / ln 2 = 1.4427; with logits (0, 2), T doubles.

Golden values come from scipy 1.17.1 (course/fixtures/M07.7/scipy_golden.json,
written by course/oracle/M07.7/scipy_golden.py: BFGS for the fit,
Mann-Whitney for the AUC, exact rational bins for the ECE).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32
from tinyllm.prob.metrics import (
    ece,
    fit_temperature,
    logistic_predict_proba,
    logistic_regression_fit,
    reliability_bins,
    roc_auc,
    roc_curve,
)

FIX = (
    Path(os.environ.get("TINYLLM_FIXTURES", "course/fixtures"))
    / "M07.7"
    / "scipy_golden.json"
)
HAND_SCORES = [0.9, 0.7, 0.6, 0.4, 0.3]
HAND_LABELS = [1, 0, 1, 1, 0]


def seed() -> int:
    return int(os.environ.get("OL_SEED", "0"))


def golden() -> dict:
    return json.loads(FIX.read_text())


def sig(z):
    return 1.0 / (1.0 + math.exp(-z))


def nll_grad(X, y, w, l2, fit_intercept=True):
    """The gradient of the contract's objective, written out independently."""
    X = np.asarray(X, dtype=np.float64)
    Xa = np.c_[X, np.ones(len(X))] if fit_intercept else X
    p = 1.0 / (1.0 + np.exp(-(Xa @ w)))
    pen = np.full(Xa.shape[1], l2)
    if fit_intercept:
        pen[-1] = 0.0
    return Xa.T @ (p - np.asarray(y, dtype=np.float64)) + pen * w


def make_data(g: PCG32, n: int, w, b):
    X = np.array([[g.normal() for _ in range(len(w))] for _ in range(n)])
    y = np.array(
        [1.0 if g.uniform() < sig(float(x @ np.asarray(w)) + b) else 0.0 for x in X]
    )
    return X, y


# --- the worked example -----------------------------------------------------------


def test_hand_example_newton_step():
    # WHY: section 3: from w = 0 the gradient is -1 and the Hessian 3/2, so
    #      one Newton step is exactly 2/3; converged, w satisfies the
    #      first-order condition w = 2 (1 - sigmoid(w)).
    # KIND: unit, smoke
    # CATCHES: s03, s04, m08
    # CHAPTER: M07.7 section 3, Worked example by hand
    X, y = [[1.0], [-1.0]], [1, 0]
    w1 = logistic_regression_fit(X, y, 1.0, 1, fit_intercept=False)
    assert w1.shape == (1,)
    assert_close(w1, [2.0 / 3.0], rtol=1e-14, atol=1e-15)
    w = float(logistic_regression_fit(X, y, 1.0, 30, fit_intercept=False)[0])
    assert abs(w - 2.0 * (1.0 - sig(w))) <= 1e-12
    assert 0.674 < w < 0.675


def test_hand_example_roc_auc():
    # WHY: section 3: the five scores give the corners (0,0) (0,1/3) (1/2,1/3)
    #      (1/2,2/3) (1/2,1) (1,1) at thresholds +inf, 0.9, 0.7, 0.6, 0.4, 0.3,
    #      and the area 1/6 + 1/2 = 2/3: 4 of the 6 pairs ranked right.
    # KIND: unit, smoke
    # CATCHES: s06, s07, m07
    # CHAPTER: M07.7 section 3, Worked example by hand
    fpr, tpr, thr = roc_curve(HAND_SCORES, HAND_LABELS)
    assert_close(fpr, [0, 0, 0.5, 0.5, 0.5, 1.0], rtol=0, atol=1e-15)
    assert_close(tpr, [0, 1 / 3, 1 / 3, 2 / 3, 1.0, 1.0], rtol=0, atol=1e-15)
    assert thr.tolist() == [math.inf, 0.9, 0.7, 0.6, 0.4, 0.3]
    assert_close(roc_auc(HAND_SCORES, HAND_LABELS), 2 / 3, rtol=1e-14, atol=1e-15)


def test_hand_example_ece():
    # WHY: section 3: top-label confidences 0.9, 0.8, 0.7, 0.6 (predictions
    #      1, 1, 0, 1; correct, wrong, right, right) fall in bins 4, 3, 3, 2 of
    #      (b/5, (b+1)/5]; ECE = 0.025 + 0.125 + 0.1 = 0.25.
    # KIND: unit, smoke
    # CATCHES: s09, s10, m05
    # CHAPTER: M07.7 section 3, Worked example by hand
    probs, labels = [0.9, 0.8, 0.3, 0.6], [1, 0, 0, 1]
    counts, conf, acc = reliability_bins(probs, labels, 5)
    assert counts.tolist() == [0, 0, 1, 2, 1]
    assert_close(conf, [0, 0, 0.6, 0.75, 0.9], rtol=1e-14, atol=1e-15)
    assert_close(acc, [0, 0, 1.0, 0.5, 1.0], rtol=1e-14, atol=1e-15)
    assert_close(ece(probs, labels, 5), 0.25, rtol=1e-14, atol=1e-15)


# --- logistic regression ------------------------------------------------------------


def test_logistic_matches_scipy():
    # WHY: on four datasets (ridge and weak penalties, no intercept, the
    #      unpenalized MLE) the IRLS answer must equal the minimizer scipy's
    #      BFGS finds for the same objective, intercept last and unpenalized.
    # KIND: golden
    # CATCHES: s02, s03, s11, m03, m08
    # CHAPTER: M07.7 section 2.2, Fitting by Newton's method
    for c in golden()["logistic"]:
        w = logistic_regression_fit(c["X"], c["y"], c["l2"], 50, c["fit_intercept"])
        assert_close(w, c["w"], rtol=1e-6, atol=1e-7, msg=c["name"])


def test_fit_is_a_stationary_point():
    # WHY: the objective is convex, so the minimum is where its gradient
    #      X^T (p - y) + l2 w_feat vanishes. With an intercept that includes
    #      sum(p - y) = 0: the fitted probabilities average to the base rate.
    # KIND: property
    # CATCHES: s02, s03, m08
    # CHAPTER: M07.7 section 2.1, The model and its loss
    g = PCG32(seed=seed() + 10)
    for l2 in (0.3, 2.0):
        X, y = make_data(g, 70, [1.0, -0.5, 0.25], 0.4)
        w = logistic_regression_fit(X, y, l2, 40)
        assert np.max(np.abs(nll_grad(X, y, w, l2))) <= 1e-9
        p = logistic_predict_proba(X, w)
        assert abs(p.mean() - y.mean()) <= 1e-11


def test_newton_converges_quadratically():
    # WHY: Newton's method doubles the correct digits per step near the
    #      optimum: errors like 1e-2, 1e-4, 1e-8, 1e-16. Eight steps must
    #      already equal fifty to rounding; a Hessian without the weights
    #      p (1 - p) (least squares) converges only linearly.
    # KIND: property
    # CATCHES: s04
    # CHAPTER: M07.7 section 2.2, Fitting by Newton's method
    g = PCG32(seed=seed() + 20)
    X, y = make_data(g, 100, [0.9, -1.1], -0.2)
    w_star = logistic_regression_fit(X, y, 0.5, 50)
    errs = [
        np.max(np.abs(logistic_regression_fit(X, y, 0.5, k) - w_star))
        for k in (1, 2, 3, 4, 8)
    ]
    assert errs[-1] <= 1e-13
    assert errs[2] <= errs[1] ** 1.5 + 1e-15, errs
    assert errs[3] <= errs[2] ** 1.5 + 1e-15, errs


def test_l2_shrinks_feature_weights():
    # WHY: a larger l2 pulls the feature weights toward 0 (their norm falls
    #      monotonically) while the intercept stays free: with huge l2 the
    #      model predicts the base rate, sigmoid(b) = mean(y).
    # KIND: property
    # CATCHES: s02
    # CHAPTER: M07.7 section 2.3, Regularization
    g = PCG32(seed=seed() + 30)
    X, y = make_data(g, 80, [1.2, -0.8], 0.7)
    norms = [
        np.linalg.norm(logistic_regression_fit(X, y, l2, 50)[:-1])
        for l2 in (0.01, 1.0, 10.0, 100.0)
    ]
    assert all(a > b for a, b in zip(norms, norms[1:])), norms
    w = logistic_regression_fit(X, y, 1e9, 50)
    assert np.max(np.abs(w[:-1])) < 1e-6
    assert abs(sig(w[-1]) - y.mean()) < 1e-6


def test_sigmoid_never_overflows():
    # WHY: features of size 1000 put z far outside exp's range (exp(710) is
    #      inf). The stable sigmoid uses exp(-|z|) on both sides, so the
    #      probabilities are exactly 0 or 1 at the ends and never NaN.
    # KIND: boundary
    # CATCHES: s01
    # CHAPTER: M07.7 section 5, Pitfalls
    X = np.array([[1000.0], [-1000.0], [0.0]])
    p = logistic_predict_proba(X, [1.0, 0.0])
    assert p.tolist() == [1.0, 0.0, 0.5]
    w = logistic_regression_fit(
        [[800.0], [-900.0], [1.0], [-1.0]], [1, 0, 0, 1], 1.0, 20
    )
    assert np.all(np.isfinite(w))


def test_predict_proba_intercept_last():
    # WHY: the contract puts the intercept LAST (w = [w_1 .. w_d, b]), the
    #      layout L6.5 writes into the linear head; without an intercept w
    #      has exactly d entries.
    # KIND: unit
    # CATCHES: s11
    # CHAPTER: M07.7 section 4, The interface
    X = np.array([[1.0, 2.0], [0.0, -1.0]])
    p = logistic_predict_proba(X, [0.5, -0.25, 1.0])
    assert_close(p, [sig(0.5 - 0.5 + 1.0), sig(0.25 + 1.0)], rtol=1e-14, atol=1e-15)
    q = logistic_predict_proba(X, [0.5, -0.25], fit_intercept=False)
    assert_close(q, [sig(0.0), sig(0.25)], rtol=1e-14, atol=1e-15)
    g = PCG32(seed=seed() + 40)
    X2, y2 = make_data(g, 300, [0.3, -0.3], 2.5)
    w = logistic_regression_fit(X2, y2, 0.1, 30)
    assert w.shape == (3,) and w[-1] > 1.5 > max(abs(w[0]), abs(w[1]))


# --- ROC and AUC --------------------------------------------------------------------


def test_roc_matches_brute_force():
    # WHY: the corners, one per distinct score, counted directly (scores >= t
    #      per class) on continuous and on heavily tied scores, plus the
    #      thresholds in decreasing order after +inf.
    # KIND: golden
    # CATCHES: s05, s06, s07, m07
    # CHAPTER: M07.7 section 2.4, The ROC curve
    for c in golden()["roc"]:
        fpr, tpr, thr = roc_curve(c["scores"], c["labels"])
        assert_close(fpr, c["fpr"], rtol=0, atol=1e-15, msg=c["name"])
        assert_close(tpr, c["tpr"], rtol=0, atol=1e-15, msg=c["name"])
        assert thr[0] == math.inf and thr[1:].tolist() == c["thresholds"]


def test_auc_is_mann_whitney():
    # WHY: the area under the corners equals the probability that a random
    #      positive outscores a random negative, ties counting 1/2: scipy's
    #      Mann-Whitney U over P * N.
    # KIND: golden
    # CATCHES: s05, s06
    # CHAPTER: M07.7 section 2.5, AUC is a ranking probability
    for c in golden()["roc"]:
        assert_close(
            roc_auc(c["scores"], c["labels"]),
            c["auc"],
            rtol=1e-13,
            atol=1e-15,
            msg=c["name"],
        )


def test_auc_properties():
    # WHY: AUC depends only on the ranking: any increasing transform of the
    #      scores leaves it unchanged, flipping the labels gives 1 - AUC, a
    #      perfect ranking gives 1, and constant scores give 0.5.
    # KIND: property
    # CATCHES: s05, s06
    # CHAPTER: M07.7 section 2.5, AUC is a ranking probability
    g = PCG32(seed=seed() + 50)
    for _ in range(20):
        n = 10 + g.below(30)
        lab = np.array([g.below(2) for _ in range(n)])
        lab[0], lab[1] = 0, 1
        s = np.array([round(g.normal() + lab[i], 1) for i in range(n)])
        a = roc_auc(s, lab)
        assert_close(roc_auc(np.exp(s) * 3.0 + 1.0, lab), a, rtol=1e-12, atol=1e-14)
        assert_close(roc_auc(s, 1 - lab), 1.0 - a, rtol=1e-12, atol=1e-14)
    assert roc_auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert roc_auc([0.4] * 5, [0, 1, 0, 1, 1]) == 0.5


def test_ties_take_a_diagonal_step():
    # WHY: a positive and a negative with the same score cannot be ordered:
    #      they enter the curve together, one diagonal step, worth 1/2 of a
    #      pair. Stepping through them one at a time makes the area depend on
    #      the input order (0 or 1 here instead of 1/2).
    # KIND: boundary
    # CATCHES: s05
    # CHAPTER: M07.7 section 2.4, The ROC curve
    assert roc_auc([0.5, 0.5], [1, 0]) == 0.5
    assert roc_auc([0.5, 0.5], [0, 1]) == 0.5
    fpr, tpr, _ = roc_curve([0.5, 0.5], [0, 1])
    assert fpr.tolist() == [0.0, 1.0] and tpr.tolist() == [0.0, 1.0]


# --- calibration ---------------------------------------------------------------------


def test_reliability_and_ece_match_reference():
    # WHY: bins, mean confidence, accuracy, and ECE on 400 binary predictions
    #      (10 and 15 bins) and 300 three-class rows, against bins computed
    #      with exact rational arithmetic.
    # KIND: golden
    # CATCHES: s08, s09, m05
    # CHAPTER: M07.7 section 2.6, Calibration and ECE
    for c in golden()["ece"]:
        counts, conf, acc = reliability_bins(c["probs"], c["labels"], c["n_bins"])
        assert counts.dtype.kind == "i" and counts.tolist() == c["counts"], c["name"]
        assert_close(conf, c["confidence"], rtol=1e-12, atol=1e-15, msg=c["name"])
        assert_close(acc, c["accuracy"], rtol=1e-12, atol=1e-15, msg=c["name"])
        assert_close(
            ece(c["probs"], c["labels"], c["n_bins"]), c["ece"], rtol=1e-12, atol=1e-15
        )


def test_ece_calibrated_vs_overconfident():
    # WHY: when y ~ Bernoulli(p) the probabilities are calibrated and ECE is
    #      only sampling noise (under 0.04 at n = 4000, 10 bins); sharpening
    #      the same probabilities (logit times 3) is overconfidence and ECE
    #      jumps above 0.1. ECE measures exactly the gap the diagram shows.
    # KIND: statistical
    # CATCHES: s09, m05
    # CHAPTER: M07.7 section 2.6, Calibration and ECE
    g = PCG32(seed=seed() + 60)
    p = np.array([g.uniform() for _ in range(4000)])
    y = np.array([1 if g.uniform() < q else 0 for q in p])
    assert ece(p, y, 10) < 0.04
    z = np.log(np.clip(p, 1e-12, None) / np.clip(1 - p, 1e-12, None)) * 3.0
    assert ece(1.0 / (1.0 + np.exp(-z)), y, 10) > 0.1


def test_bin_edges():
    # WHY: bins are (b/B, (b+1)/B]: a confidence exactly on an edge belongs
    #      to the lower bin (0.5 in bin 0 of 2, 0.75 in bin 2 of 4), 1.0 to
    #      the last bin, and an all-zero row's confidence 0 to bin 0.
    # KIND: boundary
    # CATCHES: s10
    # CHAPTER: M07.7 section 2.6, Calibration and ECE
    assert reliability_bins([0.5], [1], 2)[0].tolist() == [1, 0]
    assert reliability_bins([0.75], [1], 4)[0].tolist() == [0, 0, 1, 0]
    assert reliability_bins([1.0, 0.0], [1, 0], 3)[0].tolist() == [0, 0, 2]
    assert reliability_bins([[0.0, 0.0, 0.0]], [0], 4)[0].tolist() == [1, 0, 0, 0]


def test_multiclass_top_label():
    # WHY: with class probabilities the prediction is the first argmax and
    #      the confidence the row maximum: (0.1, 0.2, 0.7) predicts 2 (right),
    #      (0.6, 0.3, 0.1) predicts 0 for class 1 (wrong), and the tie
    #      (0.4, 0.4, 0.2) predicts the first maximum, 0 (right).
    # KIND: unit
    # CATCHES: s10
    # CHAPTER: M07.7 section 2.6, Calibration and ECE
    probs = [[0.1, 0.2, 0.7], [0.6, 0.3, 0.1], [0.4, 0.4, 0.2]]
    counts, conf, acc = reliability_bins(probs, [2, 1, 0], 5)
    assert counts.tolist() == [0, 1, 1, 1, 0]
    assert_close(conf, [0, 0.4, 0.6, 0.7, 0], rtol=1e-14, atol=1e-15)
    assert_close(acc, [0, 1.0, 0.0, 1.0, 0], rtol=1e-14, atol=1e-15)


def test_rejects_bad_arguments():
    # WHY: labels outside {0, 1}, non-finite or mismatched inputs, a negative
    #      penalty, zero iterations, one-class ROC data, probabilities outside
    #      [0, 1], and zero bins are caller bugs that must raise, not yield a
    #      metric someone ships.
    # KIND: boundary
    # CATCHES: m01, m02, m04, m06
    # CHAPTER: M07.7 section 4, The interface
    X = [[1.0], [2.0]]
    for args in (
        (X, [1, 2], 1.0, 5),
        (X, [1], 1.0, 5),
        ([[math.nan], [1.0]], [1, 0], 1.0, 5),
        ([1.0, 2.0], [1, 0], 1.0, 5),
        (X, [1, 0], -1.0, 5),
        (X, [1, 0], math.inf, 5),
        (X, [1, 0], 1.0, 0),
        (X, [1, 0], 1.0, 2.5),
    ):
        with pytest.raises(ValueError):
            logistic_regression_fit(*args)
    with pytest.raises(ValueError):
        logistic_predict_proba(X, [1.0, 2.0, 3.0])
    for s, lab in (
        ([0.1, 0.2], [1, 1]),
        ([0.1, 0.2], [0, 0]),
        ([0.1], [0, 1]),
        ([0.1, math.nan], [0, 1]),
        ([0.1, 0.2], [0, 3]),
    ):
        with pytest.raises(ValueError):
            roc_curve(s, lab)
        with pytest.raises(ValueError):
            roc_auc(s, lab)
    for p, lab, b in (
        ([1.5], [1], 5),
        ([-0.1], [0], 5),
        ([0.5], [2], 5),
        ([0.5], [1], 0),
        ([], [], 5),
        ([[0.5, 0.5]], [2], 5),
        ([0.2, 0.3], [1], 5),
    ):
        with pytest.raises(ValueError):
            reliability_bins(p, lab, b)
        with pytest.raises(ValueError):
            ece(p, lab, b)


# --- temperature scaling -------------------------------------------------------------


def softmax_rows(z):
    a = np.asarray(z, dtype=np.float64)
    e = np.exp(a - a.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def mean_nll(z, y, T):
    p = softmax_rows(np.asarray(z) / T)
    return float(-np.mean(np.log(p[np.arange(len(y)), y])))


def test_hand_example_temperature():
    # WHY: section 3: two of three items with logit gap 1 are right, so the
    #      calibrated confidence is 2/3: sigmoid(1 / T) = 2/3 gives
    #      1 / T = ln 2. Newton (M01.2) on the NLL's derivative lands there,
    #      and a gap of 2 needs twice the temperature.
    # KIND: unit, smoke
    # CATCHES: s12, s13
    # CHAPTER: M07.7 section 3, Worked example by hand
    z, y = [[0.0, 1.0]] * 3, [1, 1, 0]
    assert_close(fit_temperature(z, y), 1.0 / math.log(2.0), rtol=1e-12, atol=0)
    assert_close(
        fit_temperature([[0.0, 2.0]] * 3, y), 2.0 / math.log(2.0), rtol=1e-12, atol=0
    )


def test_temperature_calibrates_an_overconfident_model():
    # WHY: labels drawn from softmax(z0), logits reported as 3 z0: the model
    #      is right as often as before but three times too sure. The fitted
    #      T is near 3, it is the NLL's minimum (nudging it either way only
    #      raises the NLL), the argmax never changes, and ECE falls.
    # KIND: property
    # CHAPTER: M07.7 section 2
    g = PCG32(seed(), 77)
    n, c = 3000, 4
    z0 = np.array([[g.normal() for _ in range(c)] for _ in range(n)])
    p0 = softmax_rows(z0)
    y = np.array([int(np.searchsorted(np.cumsum(p), g.uniform())) for p in p0])
    y = np.minimum(y, c - 1)
    z = 3.0 * z0
    T = fit_temperature(z, y)
    assert 2.6 < T < 3.4, T
    best = mean_nll(z, y, T)
    assert best <= mean_nll(z, y, T * 1.01) and best <= mean_nll(z, y, T / 1.01)
    cal = softmax_rows(z / T)
    assert (cal.argmax(axis=1) == z.argmax(axis=1)).all()
    assert ece(cal, y) < 0.5 * ece(softmax_rows(z), y)


def test_temperature_rejects_bad_input():
    # WHY: a temperature only exists when the logits beat chance; logits
    #      that rank the labels worse must fail loudly, not return a
    #      negative T, and malformed input is refused before any Newton
    #      step.
    # KIND: boundary
    # CHAPTER: M07.7 section 4
    with pytest.raises(ValueError):
        fit_temperature([0.0, 1.0], [1, 0])  # 1-D logits
    with pytest.raises(ValueError):
        fit_temperature([[0.0, 1.0]] * 2, [1, 2])  # class out of range
    with pytest.raises(ValueError):
        fit_temperature([[0.0, np.nan]], [1])
    with pytest.raises(ValueError):
        fit_temperature([[0.0, 1.0]] * 3, [1, 0, 0])  # worse than chance: 1/T < 0
