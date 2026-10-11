"""Course tests for M01.1: finite-difference derivatives (tinyllm/num/diff.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/M01.1), and the chapter section it comes from.

The worked example of the chapter (section 3) is f(x) = x**3 at x = 2 with
h = 0.1: forward 12.61, central 12.01, one Richardson level 12, true 12.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32
from tinyllm.num.diff import central_diff, forward_diff, richardson

SEED = int(os.environ.get("OL_SEED", "0"))


def cube(x: float) -> float:
    return x**3


# Smooth functions with their exact derivatives: the oracle for every
# accuracy test. Each f is evaluated at float64 points only.
SMOOTH = [
    ("sin", math.sin, math.cos),
    ("exp", math.exp, math.exp),
    ("log1p_sq", lambda x: math.log(1 + x * x), lambda x: 2 * x / (1 + x * x)),
    ("tanh", math.tanh, lambda x: 1 - math.tanh(x) ** 2),
    ("quintic", lambda x: x**5 - 2 * x, lambda x: 5 * x**4 - 2),
]


def slope(hs: np.ndarray, errs: list[float]) -> float:
    """Least-squares slope of log(err) against log(h): the order of the method."""
    return float(np.polyfit(np.log(hs), np.log(errs), 1)[0])


# --- the worked example -------------------------------------------------------


def test_hand_example():
    # WHY: the chapter's worked example, number for number: x**3 at 2 with
    #      h = 0.1 gives forward 12.61 (error 0.61 = 3 x h + h**2), central
    #      12.01 (error h**2 = 0.01), and one Richardson level removes the
    #      h**2 term entirely: 12.
    # KIND: unit
    # CATCHES: s01, s06, s07, m01
    # CHAPTER: M01.1 section 3, Worked example by hand
    assert_close(forward_diff(cube, 2.0, 0.1), 12.61, rtol=1e-12, atol=1e-12)
    assert_close(central_diff(cube, 2.0, 0.1), 12.01, rtol=1e-12, atol=1e-12)
    assert_close(richardson(cube, 2.0, 0.1, levels=1), 12.0, rtol=1e-12, atol=1e-12)


def test_returns_python_float():
    # WHY: callers (gradcheck in M04.1, Newton in M01.2) store the result in
    #      float64 arrays and compare it; the contract says a Python float,
    #      even when f returns a numpy scalar or a 0-d array.
    # KIND: unit
    # CATCHES: m02
    # CHAPTER: M01.1 section 4, The interface
    for fn in (forward_diff, central_diff):
        out = fn(lambda t: np.float32(t) * np.float64(2.0), 1.0, 0.5)
        assert type(out) is float
        assert out == 2.0
    assert type(richardson(lambda t: np.array(t * t), 1.0, 0.5, 1)) is float


# --- order of accuracy --------------------------------------------------------


def test_central_error_slope_is_two():
    # WHY: halving h divides the central-difference error by 4: on a log-log
    #      plot the error against h has slope 2. A one-sided formula has
    #      slope 1, and a test at a single h cannot tell them apart.
    # KIND: property
    # CATCHES: s01
    # CHAPTER: M01.1 section 2.3, Truncation error
    hs = 2.0 ** -np.arange(2, 8)
    for name, f, df in SMOOTH[:3]:
        errs = [abs(central_diff(f, 0.7, h) - df(0.7)) for h in hs]
        assert 1.8 <= slope(hs, errs) <= 2.2, f"{name}: slope {slope(hs, errs):.3f}"


def test_forward_error_slope_is_one():
    # WHY: the forward difference drops f''(x) h / 2: first order. The
    #      chapter's step rule (sqrt(eps) for forward, cbrt(eps) for
    #      central) comes from exactly these two orders.
    # KIND: property
    # CATCHES: m03
    # CHAPTER: M01.1 section 2.3, Truncation error
    hs = 2.0 ** -np.arange(2, 8)
    for name, f, df in SMOOTH[:3]:
        errs = [abs(forward_diff(f, 0.7, h) - df(0.7)) for h in hs]
        assert 0.8 <= slope(hs, errs) <= 1.2, f"{name}: slope {slope(hs, errs):.3f}"


# --- the default step ---------------------------------------------------------


def test_default_step_is_accurate():
    # WHY: with h = cbrt(eps) * max(1, |x|) the truncation and rounding errors
    #      balance near eps**(2/3), about 4e-11. The bound 1e-9 rejects a
    #      central difference with the forward step sqrt(eps) (about 6e-9
    #      here) and any step off by orders of magnitude.
    # KIND: golden
    # CATCHES: s04, m04
    # CHAPTER: M01.1 section 2.5, Choosing h
    rng = PCG32(seed=SEED)
    xs = [-3.0, -0.5, 0.0, 0.25, 1.0, 2.5] + [6 * rng.uniform() - 3 for _ in range(20)]
    for name, f, df in SMOOTH:
        for x in xs:
            want = df(x)
            got = central_diff(f, x)
            assert abs(got - want) <= 1e-9 * max(1.0, abs(want)), (name, x, got, want)


def test_forward_default_step_is_accurate():
    # WHY: forward_diff's default h = sqrt(eps) * max(1, |x|) gets about
    #      eps**(1/2), 1.5e-8. The bound 1e-7 rejects the central step
    #      cbrt(eps) used one-sidedly (error about 6e-6).
    # KIND: golden
    # CATCHES: s08
    # CHAPTER: M01.1 section 2.5, Choosing h
    for name, f, df in SMOOTH:
        for x in (-2.0, -0.3, 0.0, 0.8, 2.0):
            want = df(x)
            got = forward_diff(f, x)
            assert abs(got - want) <= 1e-7 * max(1.0, abs(want)), (name, x, got, want)


def test_default_step_scales_with_x():
    # WHY: at x = 1e8 an absolute step of 6e-6 is a relative change of 6e-14:
    #      f(x + h) - f(x - h) is then mostly rounding noise (log(1e8) is known
    #      to 4e-15, divided by 1.2e-5). Scaling h by |x| keeps the step a
    #      fixed fraction of x. The derivative of log at 1e8 is 1e-8.
    # KIND: boundary
    # CATCHES: s02
    # CHAPTER: M01.1 section 5, Pitfalls, item 2
    for x in (1e8, -1e8, 3e5):
        got = central_diff(lambda t: math.log(abs(t)), x)
        assert abs(got - 1 / x) <= 1e-7 * abs(1 / x), (x, got)


def test_default_step_at_zero():
    # WHY: max(1, |x|), not |x|: a step proportional to x alone is 0 at
    #      x = 0, and 0 / 0 is not a derivative. Activations (M01.3) are
    #      checked exactly there.
    # KIND: boundary
    # CATCHES: s03
    # CHAPTER: M01.1 section 5, Pitfalls, item 3
    assert abs(central_diff(math.sin, 0.0) - 1.0) <= 1e-10
    assert abs(forward_diff(math.exp, 0.0) - 1.0) <= 1e-7


# --- floating-point details ---------------------------------------------------


def test_divides_by_the_step_actually_taken():
    # WHY: x + h is rounded, so (x + h) - (x - h) is usually not 2h. Dividing
    #      by the step actually taken makes f(x) = x come out as exactly 1,
    #      where dividing by 2h gives 0.9999999999996.
    # KIND: boundary
    # CATCHES: s05, m05
    # CHAPTER: M01.1 section 5, Pitfalls, item 4
    for x, h in ((0.1, 1e-5), (1.7, 3e-6), (-2.3, 7e-4)):
        assert central_diff(lambda t: t, x, h) == 1.0
        assert forward_diff(lambda t: t, x, h) == 1.0


def test_given_step_is_used_as_is():
    # WHY: a caller that passes h means that h: gradcheck (M04.1) passes
    #      eps = 1e-6 and its tolerance assumes that step. Rescaling a given
    #      h by |x| changes the answer at x = 100.
    # KIND: unit
    # CATCHES: s09
    # CHAPTER: M01.1 section 4, The interface
    got = central_diff(cube, 100.0, 0.5)
    # ((100.5)^3 - (99.5)^3) / 1 = 30000.25 exactly in real arithmetic
    assert_close(got, 30000.25, rtol=1e-13, atol=0)


# --- Richardson extrapolation -------------------------------------------------


def test_richardson_levels_raise_the_order():
    # WHY: each level cancels the next even power of h, so the error falls
    #      from about 3e-3 (level 0) to 3e-7, 5e-12, and the rounding floor.
    #      Level 0 must be the plain central difference.
    # KIND: property
    # CATCHES: s06, s07, m06
    # CHAPTER: M01.1 section 2.6, Richardson extrapolation
    true = math.exp(0.5)
    errs = [abs(richardson(math.exp, 0.5, 0.1, lv) - true) for lv in range(4)]
    assert richardson(math.exp, 0.5, 0.1, 0) == central_diff(math.exp, 0.5, 0.1)
    assert 1e-3 < errs[0] < 1e-2
    assert 1e-7 < errs[1] < 1e-6
    assert errs[2] < 1e-10
    assert errs[3] < 1e-12


def test_richardson_exact_on_polynomials():
    # WHY: with L levels every term up to h**(2L) is gone, so polynomials of
    #      degree <= 2L + 2 are differentiated exactly, at any h.
    # KIND: property
    # CATCHES: s06, s07
    # CHAPTER: M01.1 section 2.6, Richardson extrapolation
    def quartic(x):
        return x**4 - 3 * x**3 + x

    def dquartic(x):
        return 4 * x**3 - 9 * x**2 + 1

    for h in (0.25, 0.5, 1.0):
        assert_close(
            richardson(quartic, 1.3, h, 1), dquartic(1.3), rtol=1e-12, atol=1e-12
        )
    sextic = lambda x: x**6 - x  # noqa: E731
    assert_close(
        richardson(sextic, 0.9, 0.5, 2), 6 * 0.9**5 - 1, rtol=1e-12, atol=1e-12
    )


def test_richardson_default_levels_is_two():
    # WHY: the contract's default is two levels (error O(h**6)); a default of
    #      one would silently cost two orders of accuracy.
    # KIND: unit
    # CATCHES: m07
    # CHAPTER: M01.1 section 4, The interface
    assert richardson(math.sin, 0.3, 0.2) == richardson(math.sin, 0.3, 0.2, 2)
    assert abs(richardson(math.sin, 0.3, 0.2) - math.cos(0.3)) < 1e-9


# --- argument checks ----------------------------------------------------------


@pytest.mark.parametrize("h", [0.0, -1e-3, float("nan"), float("inf")])
def test_rejects_bad_steps(h):
    # WHY: h = 0 divides by zero and a negative h hides a sign error in the
    #      caller; both are bugs to report, not numbers to return.
    # KIND: boundary
    # CATCHES: m08, s10
    # CHAPTER: M01.1 section 4, The interface
    for fn in (forward_diff, central_diff):
        with pytest.raises(ValueError):
            fn(math.sin, 1.0, h)
    with pytest.raises(ValueError):
        richardson(math.sin, 1.0, h, 1)


def test_rejects_bad_points_and_levels():
    # WHY: a nan or inf x gives a nan derivative that would travel silently
    #      into a gradient check; a negative level count is meaningless.
    # KIND: boundary
    # CATCHES: m09, m10
    # CHAPTER: M01.1 section 4, The interface
    for x in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            central_diff(math.sin, x)
        with pytest.raises(ValueError):
            forward_diff(math.sin, x)
    with pytest.raises(ValueError):
        richardson(math.sin, 1.0, 0.1, -1)
