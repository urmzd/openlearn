"""Course tests for M11.1: entropy, cross-entropy, KL, JS, k3 (tinyllm/info/entropy.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/M11.1), and the chapter section it comes from.

The worked example of the chapter (section 3) is the pair
p = [1/2, 1/4, 1/4], q = [1/4, 1/4, 1/2]: H(p) = 1.5 ln 2,
H(p, q) = 1.75 ln 2, KL(p || q) = 0.25 ln 2, JS = 1.25 ln 2 - 0.75 ln 3.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest
from _lib.close import assert_close, assert_close_bounded
from _lib.pcg32 import PCG32
from tinyllm.info.entropy import (
    cross_entropy,
    entropy,
    entropy_from_logits,
    js,
    kl,
    kl_from_logprobs,
    kl_k3,
)

GOLDEN = Path(os.environ.get("TINYLLM_FIXTURES", "")) / "M11.1" / "scipy_golden.npz"
LN2, LN3 = math.log(2), math.log(3)
P = np.array([0.5, 0.25, 0.25])
Q = np.array([0.25, 0.25, 0.5])


def seed() -> int:
    return int(os.environ.get("OL_SEED", "0"))


def random_dist(rng: PCG32, shape, zeros: float = 0.0) -> np.ndarray:
    """Random distributions along the last axis; a share `zeros` of entries
    set to 0 (each row keeps its first entry positive)."""
    a = rng.uniform_array(shape) + 1e-3
    if zeros:
        a[rng.uniform_array(shape) < zeros] = 0.0
        a[..., 0] += 0.5
    return a / a.sum(axis=-1, keepdims=True)


def stable_log_softmax(z: np.ndarray) -> np.ndarray:
    """The test's own log-softmax (not your M09.2): z - max - log sum exp(z - max)."""
    m = z.max(axis=-1, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(axis=-1, keepdims=True))


# --- the worked example --------------------------------------------------------


def test_hand_example():
    # WHY: the chapter's worked example, number for number, in nats:
    #      H(p) = 1.5 ln 2, H(p, q) = 1.75 ln 2, KL(p || q) = 0.25 ln 2,
    #      JS(p, q) = 1.25 ln 2 - 0.75 ln 3, the three k3 values
    #      [1/2 + ln 2 - 1, 0, 1 - ln 2], and the entropy of
    #      softmax([ln 2, 0, 0]) = p. You and the test agree on every
    #      definition (and on natural logs) before any edge case.
    # KIND: unit
    # CATCHES: s09, s11, m02
    # CHAPTER: M11.1 section 3, Worked example by hand
    assert_close(entropy(P), 1.5 * LN2)
    assert_close(cross_entropy(P, Q), 1.75 * LN2)
    assert_close(kl(P, Q), 0.25 * LN2)
    assert_close(js(P, Q), 1.25 * LN2 - 0.75 * LN3)
    assert_close(kl_from_logprobs(np.log(P), np.log(Q)), 0.25 * LN2)
    k3 = kl_k3(np.log(Q), np.log(P))
    assert_close(k3, [0.5 + LN2 - 1.0, 0.0, 1.0 - LN2])
    assert_close(
        float(np.dot(P, k3)), 0.25 * LN2
    )  # the mean of k3 under p is KL(p || q)
    assert_close(entropy_from_logits(np.array([LN2, 0.0, 0.0])), 1.5 * LN2)


def test_matches_scipy_golden():
    # WHY: scipy.stats.entropy (H and KL), their sum (cross-entropy), and the
    #      square of scipy's Jensen-Shannon distance, on dense, sparse, peaked,
    #      and column-wise distributions, recorded by
    #      course/oracle/M11.1/scipy_golden.py.
    # KIND: golden
    # CATCHES: s01, s03, s04, s09, s11, m01, m02
    # CHAPTER: M11.1 section 2, Principles
    data = np.load(GOLDEN, allow_pickle=False)
    names = [k[: -len("/p")] for k in data.files if k.endswith("/p")]
    assert len(names) == 4
    for name in names:
        p, q, axis = data[f"{name}/p"], data[f"{name}/q"], int(data[f"{name}/axis"])
        assert_close(
            entropy(p, axis=axis), data[f"{name}/entropy"], msg=f"entropy {name}"
        )
        assert_close(kl(p, q, axis=axis), data[f"{name}/kl"], msg=f"kl {name}")
        assert_close(
            cross_entropy(p, q, axis=axis),
            data[f"{name}/cross_entropy"],
            msg=f"cross_entropy {name}",
        )
        if f"{name}/js" in data.files:
            assert_close(js(p, q), data[f"{name}/js"], msg=f"js {name}")


# --- properties ----------------------------------------------------------------


def test_gibbs_kl_nonnegative():
    # WHY: Gibbs' inequality: KL(p || q) >= 0, with equality exactly when
    #      p = q. A KL penalty that goes negative (L12.3) would reward the
    #      policy for drifting. Checked on 200 random pairs, sparse ones too.
    # KIND: property
    # CATCHES: s14
    # CHAPTER: M11.1 section 2, Principles (Gibbs' inequality)
    rng = PCG32(seed=seed())
    p = random_dist(rng, (200, 7), zeros=0.3)
    q = random_dist(rng, (200, 7))
    d = kl(p, q)
    assert d.shape == (200,)
    assert (d >= -1e-15).all()
    assert (kl(p, p) == 0.0).all()
    assert (kl_from_logprobs(np.log(q), np.log(q)) == 0.0).all()


def test_cross_entropy_decomposes():
    # WHY: H(p, q) = H(p) + KL(p || q): the cross-entropy loss is the data's
    #      own entropy (a floor no model can beat) plus the model's excess.
    #      L0.3's loss and L6.7's perplexity both read this way.
    # KIND: property
    # CATCHES: s01, s04, s11
    # CHAPTER: M11.1 section 2, Principles (cross-entropy)
    rng = PCG32(seed=seed())
    p = random_dist(rng, (50, 9), zeros=0.2)
    q = random_dist(rng, (50, 9))
    assert_close_bounded(
        cross_entropy(p, q), entropy(p) + kl(p, q), k=9, dtype="float64"
    )


def test_kl_is_not_symmetric():
    # WHY: KL(p || q) weighs the log ratio by p. For p = [0.9, 0.1] and the
    #      uniform q, KL(p || q) = 0.9 ln 1.8 + 0.1 ln 0.2 = 0.36806 but
    #      KL(q || p) = 0.5 ln(5/9) + 0.5 ln 5 = 0.51083. The direction is a
    #      design decision (forward vs reverse KL in L12), not a detail.
    # KIND: unit
    # CATCHES: s03
    # CHAPTER: M11.1 section 5, Pitfalls, item 3
    p, q = np.array([0.9, 0.1]), np.array([0.5, 0.5])
    assert_close(kl(p, q), 0.9 * math.log(1.8) + 0.1 * math.log(0.2))
    assert_close(kl(q, p), 0.5 * math.log(5 / 9) + 0.5 * math.log(5))


def test_zero_probability_conventions():
    # WHY: 0 log 0 = 0 (an outcome that never happens costs nothing), and
    #      p log(p / 0) = +inf (q rules out something p produces). A numpy
    #      0 * log 0 is nan; adding an epsilon to q turns a true +inf into a
    #      large finite number and biases every other answer.
    # KIND: boundary
    # CATCHES: s01, s02, s04
    # CHAPTER: M11.1 section 5, Pitfalls, item 1
    p = np.array([0.5, 0.5, 0.0])
    q = np.array([0.5, 0.25, 0.25])
    assert_close(entropy(p), LN2)
    assert_close(cross_entropy(p, q), 1.5 * LN2)
    assert_close(kl(p, q), 0.5 * LN2)
    assert_close(kl(np.array([0.5, 0.5, 0.0]), np.array([0.5, 0.5, 0.0])), 0.0)
    r = np.array([1.0, 0.0, 0.0])
    assert kl(q, r) == np.inf and cross_entropy(q, r) == np.inf
    assert entropy(r) == 0.0
    assert_close(kl(r, q), math.log(2))


def test_rejects_negative_probabilities():
    # WHY: a negative "probability" is a bug upstream (a subtraction gone
    #      wrong, a logit passed where a probability belongs). Refuse it
    #      instead of returning a number that looks plausible.
    # KIND: boundary
    # CATCHES: s12
    # CHAPTER: M11.1 section 4, The interface
    bad = np.array([0.6, 0.5, -0.1])
    for fn in (
        lambda: entropy(bad),
        lambda: kl(bad, Q),
        lambda: kl(P, bad),
        lambda: cross_entropy(P, bad),
        lambda: js(bad, Q),
    ):
        with pytest.raises(ValueError):
            fn()


def test_entropy_bounds():
    # WHY: 0 <= H(p) <= ln n, with 0 for a certain outcome and ln n for the
    #      uniform distribution. L8.1 logs the entropy of each sampling step;
    #      the bounds are how you read that number.
    # KIND: property
    # CATCHES: s01, s11
    # CHAPTER: M11.1 section 2, Principles (entropy)
    for n in (2, 3, 10, 256):
        assert_close(entropy(np.full(n, 1.0 / n)), math.log(n))
        one_hot = np.zeros(n)
        one_hot[n // 2] = 1.0
        assert entropy(one_hot) == 0.0
    rng = PCG32(seed=seed())
    h = entropy(random_dist(rng, (100, 12), zeros=0.4))
    assert (h >= 0).all() and (h <= math.log(12) + 1e-12).all()


def test_axis_and_batch_shapes():
    # WHY: a batch of distributions [B, V] reduces over V to [B]; with
    #      axis=0 the columns are the distributions. The reduced axis is
    #      dropped, like numpy's sum.
    # KIND: unit
    # CATCHES: m01
    # CHAPTER: M11.1 section 4, The interface
    rng = PCG32(seed=seed())
    p = random_dist(rng, (4, 6))
    q = random_dist(rng, (4, 6))
    for fn in (
        lambda a, b, ax: kl(a, b, axis=ax),
        lambda a, b, ax: cross_entropy(a, b, axis=ax),
    ):
        assert fn(p, q, -1).shape == (4,)
        cols = fn(p.T, q.T, 0)
        assert cols.shape == (4,)
        assert_close(cols, fn(p, q, -1))
    assert entropy(p).shape == (4,)
    assert_close(entropy(p.T, axis=0), entropy(p))
    assert kl_from_logprobs(np.log(p), np.log(q)).shape == (4,)
    assert_close(kl_from_logprobs(np.log(p.T), np.log(q.T), axis=0), kl(p, q))
    assert js(p, q).shape == (4,)


def test_js_properties():
    # WHY: Jensen-Shannon compares both distributions with their average m,
    #      which is positive wherever either is, so it is symmetric and finite
    #      even for disjoint supports, where it reaches its maximum ln 2.
    # KIND: property
    # CATCHES: s09, m02
    # CHAPTER: M11.1 section 2, Principles (Jensen-Shannon)
    rng = PCG32(seed=seed())
    p = random_dist(rng, (30, 5), zeros=0.3)
    q = random_dist(rng, (30, 5), zeros=0.3)
    d = js(p, q)
    assert_close(d, js(q, p))
    assert (d >= -1e-15).all() and (d <= LN2 + 1e-15).all()
    assert_close(js(np.array([1.0, 0.0]), np.array([0.0, 1.0])), LN2)
    assert_close(js(P, P), 0.0)


# --- KL from log-probabilities ------------------------------------------------


def test_kl_from_logprobs_matches_kl():
    # WHY: the same divergence from log-probabilities: what a model actually
    #      outputs (L0.3, L12.3). Includes -inf entries where p is 0.
    # KIND: differential
    # CATCHES: s06
    # CHAPTER: M11.1 section 2, Principles (KL)
    rng = PCG32(seed=seed())
    p = random_dist(rng, (40, 8), zeros=0.3)
    q = random_dist(rng, (40, 8))
    with np.errstate(divide="ignore"):
        lp = np.log(p)
    assert np.isneginf(lp).any()
    assert_close_bounded(
        kl_from_logprobs(lp, np.log(q)), kl(p, q), k=8, dtype="float64"
    )


def test_kl_from_logprobs_extreme_logits():
    # WHY: log-probabilities from logits near 1e4 (L8.1 after a temperature
    #      of 0.01, L12.3's policy late in training) are fine numbers, but
    #      exp(logp) / exp(logq) is 0 / 0 or inf / inf. Working with the
    #      difference logp - logq keeps KL finite and right.
    # KIND: boundary
    # CATCHES: s05
    # CHAPTER: M11.1 section 5, Pitfalls, item 2
    zp = np.array([1e4, 1e4 - 1.0, -1e4])
    zq = np.array([1e4 - 2.0, 1e4, -1e4])
    lp, lq = stable_log_softmax(zp), stable_log_softmax(zq)
    p = np.exp(lp)
    expect = float(np.sum(p * (lp - lq)))
    out = kl_from_logprobs(lp, lq)
    assert np.isfinite(out)
    assert_close(out, expect)


def test_kl_from_logprobs_infinite_cases():
    # WHY: logp = -inf is p = 0: it costs nothing even against logq = -inf
    #      (-inf - -inf is nan). A finite logp against logq = -inf is +inf,
    #      including logp = -800, where exp(logp) underflows to 0 and
    #      0 * inf would be nan.
    # KIND: boundary
    # CATCHES: s06, s13
    # CHAPTER: M11.1 section 5, Pitfalls, item 1
    ninf = -np.inf
    assert_close(kl_from_logprobs(np.array([0.0, ninf]), np.array([0.0, ninf])), 0.0)
    assert kl_from_logprobs(np.array([0.0, -800.0]), np.array([0.0, ninf])) == np.inf
    assert (
        kl_from_logprobs(np.array([math.log(0.5)] * 2), np.array([0.0, ninf])) == np.inf
    )


# --- the k3 estimator --------------------------------------------------------


def test_kl_k3_is_nonnegative():
    # WHY: k3 = e^r - r - 1 >= 0 for every r (the exponential lies above its
    #      tangent at 0), with 0 exactly at r = 0. Each sample's KL penalty in
    #      L12.3 is therefore nonnegative, unlike the plain estimator -r.
    # KIND: property
    # CATCHES: s08
    # CHAPTER: M11.1 section 2, Principles (the k3 estimator)
    rng = PCG32(seed=seed())
    a = rng.normal_array(1000, scale=3.0)
    b = rng.normal_array(1000, scale=3.0)
    k = kl_k3(a, b)
    assert k.shape == (1000,) and (k >= 0).all()
    assert (kl_k3(a, a) == 0.0).all()
    assert_close(kl_k3(np.array([1.0]), np.array([0.0])), [math.e - 2.0])
    assert kl_k3(np.zeros((2, 3)), np.zeros(3)).shape == (2, 3)


def test_kl_k3_small_r_precision():
    # WHY: for small r, e^r - r - 1 = r^2/2 + r^3/6 + ... is tiny, while
    #      e^r itself is about 1 with an absolute rounding error near 1e-16.
    #      At r = 1e-6 the true value is 5.0000017e-13, and exp(r) - r - 1
    #      carries a relative error near 1e-4. expm1(r) - r keeps it under
    #      1e-9. Policies that barely moved from the reference (the start of
    #      L12.3) live exactly here.
    # KIND: boundary
    # CATCHES: s07
    # CHAPTER: M11.1 section 5, Pitfalls, item 4
    for r in (1e-6, -1e-6, 3e-5, -4e-6):
        series = r * r / 2 + r**3 / 6 + r**4 / 24 + r**5 / 120
        out = float(kl_k3(np.array(r), np.array(0.0)))
        assert abs(out - series) <= 1e-9 * series, f"r = {r}"


def test_kl_k3_mean_is_kl():
    # WHY: k3 is an unbiased estimator: sum_x pi(x) k3(log ref(x) - log pi(x))
    #      = sum ref - 1 + KL(pi || ref) = KL(pi || ref). Checked exactly by
    #      enumerating a 6-symbol pi, then by sampling 20000 draws from pi
    #      (frozen PCG32, inverse CDF): the sample mean is within 4 standard
    #      errors of KL. Swapping the arguments gives a different, biased
    #      number.
    # KIND: statistical
    # CATCHES: s08
    # CHAPTER: M11.1 section 2, Principles (the k3 estimator)
    pi = np.array([0.3, 0.25, 0.2, 0.1, 0.1, 0.05])
    ref = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])
    truth = float(np.sum(pi * np.log(pi / ref)))
    exact = float(np.dot(pi, kl_k3(np.log(ref), np.log(pi))))
    assert_close(exact, truth)
    rng = PCG32(seed=seed())
    cdf = np.cumsum(pi)
    xs = np.minimum(np.searchsorted(cdf, rng.uniform_array(20000), side="right"), 5)
    k = kl_k3(np.log(ref)[xs], np.log(pi)[xs])
    se = float(k.std(ddof=1)) / math.sqrt(k.size)
    assert abs(float(k.mean()) - truth) < 4 * se


# --- entropy from logits -----------------------------------------------------


def test_entropy_from_logits():
    # WHY: the sampler of L8.1 logs the entropy of each step from logits.
    #      It must equal H(softmax(z)), stay finite for logits near 1e4
    #      (where log(softmax) is log 0 = -inf and 0 * -inf is nan), and
    #      treat masked entries (-inf logits) as probability 0.
    # KIND: boundary
    # CATCHES: s10
    # CHAPTER: M11.1 section 5, Pitfalls, item 5
    rng = PCG32(seed=seed())
    z = rng.normal_array((8, 11), scale=2.0)
    p = np.exp(stable_log_softmax(z))
    assert_close_bounded(entropy_from_logits(z), entropy(p), k=11, dtype="float64")
    big = np.array([1e4, 1e4, -1e4, 0.0])
    out = entropy_from_logits(big)
    assert np.isfinite(out)
    assert_close(out, LN2)
    masked = np.array([[0.0, 0.0, -np.inf, -np.inf], [5.0, -np.inf, -np.inf, -np.inf]])
    assert_close(entropy_from_logits(masked), [LN2, 0.0])
    assert_close(entropy_from_logits(z.T, axis=0), entropy_from_logits(z))
