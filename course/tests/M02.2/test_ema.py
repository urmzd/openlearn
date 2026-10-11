"""Course tests for M02.2: series convergence, the EMA as a geometric series,
and bias correction (tinyllm/num/ema.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/M02.2), and the chapter section it comes from.

The worked example of the chapter (section 3) is beta = 0.9 and the inputs
1, 2, 3: m = 0.1, 0.29, 0.561, the weight totals 1 - 0.9^t = 0.1, 0.19,
0.271, and the debiased values 1, 1.5263158, 2.0701107.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32
from tinyllm.num.ema import EMA, ema_weights

SEED = int(os.environ.get("OL_SEED", "0"))
EPS = np.finfo(np.float64).eps


# --- the worked example -------------------------------------------------------


def test_hand_example():
    # WHY: the chapter's worked example, number for number. m starts at 0,
    #      so after one update it is 0.1 * 1 = 0.1, far below the only input;
    #      dividing by the weight total 1 - 0.9 = 0.1 gives 1 back.
    # KIND: unit
    # CATCHES: s01, s02, s03, s04, m01, m03
    # CHAPTER: M02.2 section 3, Worked example by hand
    e = EMA(0.9)
    assert e.t == 0 and e.value == 0.0
    want_m = [0.1, 0.29, 0.561]
    want_d = [1.0, 0.29 / 0.19, 0.561 / 0.271]
    for k, x in enumerate([1.0, 2.0, 3.0]):
        m = e.update(x)
        assert type(m) is float
        assert_close(m, want_m[k], rtol=1e-15, atol=0)
        assert_close(e.value, want_m[k], rtol=1e-15, atol=0)
        assert_close(e.value_debiased(), want_d[k], rtol=1e-15, atol=0)
        assert e.t == k + 1
    assert_close(e.value_debiased(), 2.0701107, rtol=1e-7, atol=0)


def test_weights_hand_example():
    # WHY: unrolled, m_3 = 0.081 x_1 + 0.09 x_2 + 0.1 x_3: the weights form a
    #      geometric sequence (ratio 0.9 going back in time) whose total is
    #      0.271 = 1 - 0.9^3, the number bias correction divides by.
    # KIND: unit
    # CATCHES: s05, m02
    # CHAPTER: M02.2 section 3, Worked example by hand
    w = ema_weights(0.9, 3)
    assert w.dtype == np.float64 and w.shape == (3,)
    assert_close(w, [0.081, 0.09, 0.1], rtol=1e-15, atol=0)
    assert_close(w.sum(), 0.271, rtol=1e-15, atol=0)
    assert ema_weights(0.5, 0).shape == (0,)


# --- the geometric series -----------------------------------------------------


def test_weights_sum_to_one_minus_beta_power():
    # WHY: the partial sum of a geometric series, (1 - b) (1 + b + ... + b^(t-1))
    #      = 1 - b^t (section 2.2): it tends to 1 only as t grows, which is the
    #      whole reason for the bias.
    # KIND: property
    # CATCHES: s05, m02
    # CHAPTER: M02.2 section 2.2, Geometric series
    for beta in (0.0, 0.5, 0.9, 0.99, 0.999):
        for t in (1, 2, 10, 100, 1000):
            w = ema_weights(beta, t)
            assert_close(
                w.sum(), 1 - beta**t, rtol=1e-12, atol=1e-15, msg=f"{beta} {t}"
            )
            if beta > 0 and t > 1:
                assert_close(w[:-1] / w[1:], np.full(t - 1, beta), rtol=1e-12, atol=0)


def test_update_equals_weighted_sum():
    # WHY: the recursion and the unrolled series are the same number: m_t is
    #      the dot product of ema_weights(beta, t) with x_1..x_t. A recursion
    #      with beta and 1 - beta swapped is still a weighted sum, just of the
    #      wrong weights.
    # KIND: differential
    # CATCHES: s01, s05
    # CHAPTER: M02.2 section 2.3, The EMA unrolled
    rng = PCG32(seed=SEED)
    xs = rng.normal_array((200,))
    for beta in (0.3, 0.9, 0.99):
        e = EMA(beta)
        for t, x in enumerate(xs, start=1):
            e.update(float(x))
            if t in (1, 7, 50, 200):
                want = float(ema_weights(beta, t) @ xs[:t])
                assert_close(
                    e.value, want, rtol=1e-12, atol=1e-14, msg=f"beta {beta} t {t}"
                )


# --- bias correction ----------------------------------------------------------


def test_debiased_constant_is_exact():
    # WHY: the design's property: for a constant input c, m_t = c (1 - b^t)
    #      exactly in real arithmetic, so the debiased value is c at every t,
    #      including t = 1 where the biased m is only (1 - b) c. In floating
    #      point the error stays below a few eps / (1 - b).
    # KIND: property
    # CATCHES: s02, s03, s04, s06
    # CHAPTER: M02.2 section 2.4, Bias correction
    for beta in (0.0, 0.5, 0.9, 0.99, 0.999, 0.9999):
        for c in (3.7, -1e-3, 1e6):
            e = EMA(beta)
            tol = 16 * EPS / (1 - beta)
            for t in range(1, 3001):
                e.update(c)
                if t <= 20 or t % 97 == 0:
                    got = e.value_debiased()
                    assert abs(got - c) <= tol * abs(c), (beta, c, t, got)


def test_matches_exact_rational_arithmetic():
    # WHY: the oracle is the definition evaluated with no rounding at all:
    #      fractions.Fraction holds beta and every input exactly (they are
    #      binary floats), runs m_t = b m + (1 - b) x and m_t / (1 - b^t) in
    #      rational arithmetic, and rounds once at the end. The float64
    #      recursion must stay within a few hundred ulps of it (the inputs
    #      are of size 1) after 200 seeded steps, biased and debiased, so
    #      any change to the
    #      recursion or the correction shows up.
    # KIND: golden
    # CATCHES: s01, s03, s04, s06
    # CHAPTER: M02.2 section 2.4, Bias correction
    from fractions import Fraction

    rng = PCG32(seed=SEED)
    xs = [float(v) for v in rng.normal_array((200,))]
    for beta in (0.9, 0.99):
        b = Fraction(beta)
        m = Fraction(0)
        e = EMA(beta)
        for t, x in enumerate(xs, start=1):
            m = b * m + (1 - b) * Fraction(x)
            e.update(x)
            if t in (1, 2, 3, 10, 200):
                want_m = float(m)
                want_d = float(m / (1 - b**t))
                tol = 256 * EPS / (1 - beta)
                assert abs(e.value - want_m) <= tol * max(abs(want_m), 1.0), (beta, t)
                got_d = e.value_debiased()
                assert abs(got_d - want_d) <= tol * max(abs(want_d), 1.0), (beta, t)


def test_bias_fades_without_correction():
    # WHY: the biased value converges to the constant as b^t -> 0: after
    #      t = 50 steps with b = 0.9 the gap is 0.9^50 = 0.5 percent of c. The
    #      correction matters early and is harmless late.
    # KIND: property
    # CATCHES: s01
    # CHAPTER: M02.2 section 2.4, Bias correction
    e = EMA(0.9)
    for _ in range(50):
        e.update(10.0)
    assert_close(e.value, 10.0 * (1 - 0.9**50), rtol=1e-13, atol=0)
    assert_close(e.value_debiased(), 10.0, rtol=1e-13, atol=0)


def test_beta_zero_is_the_last_value():
    # WHY: beta = 0 keeps no history: m_t = x_t and the correction divides
    #      by 1. A useful sanity case for a schedule that anneals beta.
    # KIND: boundary
    # CATCHES: s07
    # CHAPTER: M02.2 section 4, The interface
    e = EMA(0.0)
    for x in (5.0, -2.0, 7.5):
        assert e.update(x) == x
        assert e.value_debiased() == x


# --- arrays and state ---------------------------------------------------------


def test_arrays_are_averaged_elementwise():
    # WHY: C1 keeps an EMA of every weight tensor and Adam (M10.3) one of
    #      every gradient; an array EMA must equal one scalar EMA per element
    #      and return a new array, not a view the caller could mutate.
    # KIND: differential
    # CATCHES: s08, m04
    # CHAPTER: M02.2 section 4, The interface
    rng = PCG32(seed=SEED)
    xs = rng.normal_array((30, 2, 3))
    e = EMA(0.8)
    scal = [EMA(0.8) for _ in range(6)]
    for x in xs:
        out = e.update(x)
        for s, v in zip(scal, x.reshape(-1)):
            s.update(float(v))
    assert (
        isinstance(out, np.ndarray) and out.shape == (2, 3) and out.dtype == np.float64
    )
    assert_close(
        e.value_debiased().reshape(-1),
        [s.value_debiased() for s in scal],
        rtol=1e-15,
        atol=0,
    )
    out[...] = 0.0
    assert not np.all(e.value == 0.0)
    v = e.value
    v[...] = 0.0
    assert not np.all(e.value == 0.0)
    with pytest.raises(ValueError):
        e.update(np.zeros(3))


def test_rejects_nonfinite_input_and_keeps_state():
    # WHY: one nan in a loss curve would poison every later average (nan
    #      times anything is nan). The update must refuse it and leave the
    #      average exactly as it was.
    # KIND: boundary
    # CATCHES: s09
    # CHAPTER: M02.2 section 5, Pitfalls, item 4
    e = EMA(0.9)
    e.update(1.0)
    for bad in (float("nan"), float("inf"), np.array([1.0, np.nan])):
        with pytest.raises(ValueError):
            e.update(bad)
    assert e.t == 1
    assert_close(e.value, 0.1, rtol=1e-15, atol=0)


def test_debiased_before_update_raises():
    # WHY: at t = 0 the correction is 0 / 0: nothing has been averaged.
    #      RuntimeError says so instead of returning nan or 0.
    # KIND: boundary
    # CATCHES: s10
    # CHAPTER: M02.2 section 4, The interface
    with pytest.raises(RuntimeError):
        EMA(0.9).value_debiased()


@pytest.mark.parametrize("beta", [-0.1, 1.0, 1.5, float("nan")])
def test_rejects_bad_beta(beta):
    # WHY: beta = 1 never moves (the weight total 1 - 1^t is 0 forever), and
    #      beta outside [0, 1) is not an average. Both are caller bugs.
    # KIND: boundary
    # CATCHES: m05, m06
    # CHAPTER: M02.2 section 2.2, Geometric series
    with pytest.raises(ValueError):
        EMA(beta)
    with pytest.raises(ValueError):
        ema_weights(beta, 3)
