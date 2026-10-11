"""Course tests for M02.1: Taylor series, remainder bounds, range reduction
(tinyllm/num/taylor.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/M02.1), and the chapter section it comes from.

The worked example of the chapter (section 3) is exp(1) with degree 3:
k = round(1 / ln 2) = 1, r = 1 - ln 2 = 0.30685, p(r) = 1 + r + r^2/2 + r^3/6
= 1.35875, and 2 * p(r) = 2.71750 against e = 2.71828 (relative error 2.9e-4,
inside the Lagrange bound r^4 / 4! = 3.7e-4).
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32
from tinyllm.num.taylor import (
    ERF_CUTOFF,
    erf_series,
    exp_range_reduced,
    exp_taylor_coeffs,
)

SEED = int(os.environ.get("OL_SEED", "0"))
LN2 = math.log(2.0)


def lagrange_rel_bound(r: np.ndarray, deg: int) -> np.ndarray:
    """|e^r - p(r)| / e^r <= e^(|r| - r) |r|^(deg+1) / (deg+1)!  (chapter 2.3)."""
    return np.exp(np.abs(r) - r) * np.abs(r) ** (deg + 1) / math.factorial(deg + 1)


def reduced(x: np.ndarray) -> np.ndarray:
    """The test's own r = x - k ln 2 with k = round(x / ln 2), for small |x|."""
    return x - np.rint(x / LN2) * LN2


# --- the worked example -------------------------------------------------------


def test_hand_example():
    # WHY: the chapter's worked example: exp(1) with degree 3 is
    #      2 * (1 + r + r^2/2 + r^3/6) with r = 1 - ln 2, which is
    #      2.7174952 to 8 digits, a relative error of 2.9e-4, inside the
    #      Lagrange bound r^4 / 4! = 3.7e-4.
    # KIND: unit
    # CATCHES: s01, s02, s03
    # CHAPTER: M02.1 section 3, Worked example by hand
    r = 1.0 - LN2
    want = 2.0 * (1 + r + r * r / 2 + r**3 / 6)
    got = exp_range_reduced(1.0, deg=3)
    assert np.shape(got) == ()
    assert_close(got, want, rtol=1e-15, atol=0)
    assert_close(got, 2.7174952, rtol=1e-7, atol=0)
    assert 2.8e-4 < 1 - float(got) / math.e < r**4 / 24


def test_coefficients_are_inverse_factorials():
    # WHY: the Maclaurin coefficient of x^i in exp is 1/i!: degree n gives
    #      n + 1 numbers, starting with 1/0! = 1. An off-by-one in the length
    #      or the index silently drops or shifts a term.
    # KIND: unit
    # CATCHES: s03, m02
    # CHAPTER: M02.1 section 2.2, The Taylor polynomial
    c = exp_taylor_coeffs(4)
    assert c.dtype == np.float64 and c.shape == (5,)
    assert_close(c, [1, 1, 1 / 2, 1 / 6, 1 / 24], rtol=1e-16, atol=0)
    assert exp_taylor_coeffs(0).tolist() == [1.0]
    c30 = exp_taylor_coeffs(30)
    assert_close(c30[30], 1 / math.factorial(30), rtol=1e-14, atol=0)
    with pytest.raises(ValueError):
        exp_taylor_coeffs(-1)


# --- remainder bounds ---------------------------------------------------------


def test_lagrange_bound_holds():
    # WHY: Taylor's theorem with the Lagrange remainder bounds the error of
    #      the degree-d polynomial on |r| <= ln(2)/2. The bound is the
    #      design's property test: it must hold at every point and every
    #      degree, with room only for float64 rounding.
    # KIND: property
    # CATCHES: s01, s02, s04, m01
    # CHAPTER: M02.1 section 2.3, The remainder
    x = np.linspace(-20.0, 20.0, 4001)
    true = np.exp(x)  # the oracle; float64 exp is correct to about 1 ulp
    r = reduced(x)
    for deg in range(1, 13):
        got = exp_range_reduced(x, deg)
        rel = np.abs(got - true) / true
        assert np.all(rel <= lagrange_rel_bound(r, deg) + 8e-16), deg


def test_degree_six_is_float32_accurate():
    # WHY: degree 6 after reduction to |r| <= 0.347 gives relative error at
    #      most 2.4e-7, about 2 float32 ulps: the polynomial M09.6 ports to C
    #      as tl_expf. Reducing with floor (r up to 0.693) would need
    #      degree 9 for the same accuracy.
    # KIND: golden
    # CATCHES: s04, m03
    # CHAPTER: M02.1 section 2.4, Range reduction
    x = np.linspace(-87.0, 88.0, 20001)
    rel = np.abs(exp_range_reduced(x) - np.exp(x)) / np.exp(x)
    assert rel.max() <= 2.4e-7


def test_full_precision_across_the_range():
    # WHY: with degree 20 the polynomial is exact to rounding, so what is
    #      left is the reduction: k ln 2 must be subtracted in two parts
    #      (Cody and Waite). One rounded ln 2 times k = 1000 already costs
    #      1e-13 relative; split, the error stays at the 1e-16 level.
    # KIND: golden
    # CATCHES: s05
    # CHAPTER: M02.1 section 5, Pitfalls, item 3
    x = np.concatenate(
        [np.linspace(-700.0, -600.0, 1001), np.linspace(600.0, 709.0, 1001)]
    )
    rel = np.abs(exp_range_reduced(x, 20) - np.exp(x)) / np.exp(x)
    assert rel.max() <= 1e-15


def test_special_values_and_extremes():
    # WHY: the C port and every softmax downstream rely on exp(-inf) = 0
    #      (masked logits), exp(inf) = inf, nan in nan out, overflow to inf
    #      above 709.78 and underflow to 0 below -745, with no warnings.
    # KIND: boundary
    # CATCHES: s06, m04
    # CHAPTER: M02.1 section 4, The interface
    x = np.array([np.inf, -np.inf, np.nan, 710.0, 1e300, -750.0, -1e300, 0.0])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        y = exp_range_reduced(x)
    assert y[0] == np.inf and y[1] == 0.0 and np.isnan(y[2])
    assert y[3] == np.inf and y[4] == np.inf
    assert y[5] == 0.0 and y[6] == 0.0
    assert y[7] == 1.0
    assert_close(exp_range_reduced(709.7, 20), math.exp(709.7), rtol=1e-15, atol=0)
    sub = exp_range_reduced(-740.0, 20)  # subnormal: few significant bits left
    assert 0 < sub < 1e-320


def test_shape_and_scalars():
    # WHY: callers pass scalars, vectors, and matrices; the result has x's
    #      shape and dtype float64, and integer input is accepted.
    # KIND: unit
    # CATCHES: m05
    # CHAPTER: M02.1 section 4, The interface
    assert exp_range_reduced(0).dtype == np.float64
    m = exp_range_reduced(np.arange(6).reshape(2, 3), 20)
    assert m.shape == (2, 3)
    assert_close(m, np.exp(np.arange(6.0).reshape(2, 3)), rtol=1e-15, atol=0)
    with pytest.raises(ValueError):
        exp_range_reduced(1.0, -1)


# --- erf ----------------------------------------------------------------------


def test_erf_hand_example():
    # WHY: the chapter's second example, erf(0.5) with three terms:
    #      (2/sqrt(pi)) e^-0.25 (0.5 + 2 * 0.125 / 3 + 4 * 0.03125 / 15)
    #      = 0.8787826 * 0.5916667 = 0.5199464, against erf(0.5) = 0.5204999.
    # KIND: unit
    # CATCHES: s07, s08
    # CHAPTER: M02.1 section 3, Worked example by hand
    want = 2 / math.sqrt(math.pi) * math.exp(-0.25) * (0.5 + 0.25 / 3 + 0.125 / 15)
    assert_close(erf_series(0.5, 3), want, rtol=1e-15, atol=0)
    assert_close(
        erf_series(0.5, 1),
        2 / math.sqrt(math.pi) * math.exp(-0.25) * 0.5,
        rtol=1e-15,
        atol=0,
    )


def test_erf_matches_math_erf():
    # WHY: with 160 terms the series is within 3e-15 of erf on the whole
    #      line, the accuracy gelu_erf (M01.3) needs to match PyTorch.
    #      math.erf (correctly rounded to about 1 ulp) is the oracle.
    # KIND: golden
    # CATCHES: s07, s09, m06
    # CHAPTER: M02.1 section 2.5, A series for erf
    rng = PCG32(seed=SEED)
    x = np.concatenate(
        [np.linspace(-7.0, 7.0, 2801), rng.uniform_array((300,), -6.0, 6.0)]
    )
    want = np.array([math.erf(v) for v in x])
    assert_close(erf_series(x, 160), want, rtol=0, atol=3e-15)


def test_erf_no_cancellation_at_x_5():
    # WHY: the textbook alternating series sum (-1)^n x^(2n+1) / (n! (2n+1))
    #      adds terms as large as 6e8 at x = 5 to get a number below 1: the
    #      cancellation leaves 7 correct digits. The positive-term series
    #      loses nothing.
    # KIND: boundary
    # CATCHES: s09
    # CHAPTER: M02.1 section 5, Pitfalls, item 4
    for v in (3.0, 4.0, 5.0, -5.5):
        assert abs(float(erf_series(v, 160)) - math.erf(v)) <= 3e-15, v


def test_erf_cutoff_and_specials():
    # WHY: beyond |x| = 6, erf(x) rounds to +-1 and the series would need
    #      hundreds of terms (and overflows its partial sums near |x| = 27):
    #      the cutoff returns sign(x). GELU feeds x / sqrt(2) up to 28 here.
    # KIND: boundary
    # CATCHES: s10, m06
    # CHAPTER: M02.1 section 2.5, A series for erf
    assert ERF_CUTOFF == 6.0
    y = erf_series(
        np.array([6.0, -6.0, 10.0, -28.3, 1e300, np.inf, -np.inf, np.nan, 0.0]), 160
    )
    assert y[:7].tolist() == [1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0]
    assert np.isnan(y[7]) and y[8] == 0.0
    with pytest.raises(ValueError):
        erf_series(1.0, 0)


def test_erf_tail_bound_holds():
    # WHY: after N terms, once the term ratio 2x^2 / (2N + 3) is below 1, the
    #      omitted tail is at most t_N / (1 - ratio) (a geometric series
    #      bounds it): the remainder bound the contract states, checked at
    #      many N and x.
    # KIND: property
    # CATCHES: s07, s09
    # CHAPTER: M02.1 section 2.5, A series for erf
    pref = 2 / math.sqrt(math.pi)
    for x in (0.3, 1.0, 2.0, 3.5, 5.0):
        t = x  # t_0
        for n in range(1, 200):
            t_next = t * 2 * x * x / (2 * n + 1)  # t_n
            ratio = 2 * x * x / (2 * n + 3)
            if ratio < 1 and n % 7 == 0:
                bound = pref * math.exp(-x * x) * t_next / (1 - ratio)
                err = abs(float(erf_series(x, n)) - math.erf(x))
                assert err <= bound + 4e-15, (x, n, err, bound)
            t = t_next
