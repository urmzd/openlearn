"""Course tests for M01.4: definite integrals, trapezoid, Simpson
(tinyllm/num/integrate.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/M01.4), and the chapter section it comes from.

The worked example of the chapter (section 3) is the integral of x**3 from
0 to 2, exactly 4: the trapezoid rule gives 5 with 2 steps and 4.25 with 4
(the error drops from 1 to 1/4), Simpson gives exactly 4 with 2 steps, and
(4 * 4.25 - 5) / 3 = 4 is Richardson's step from trapezoid to Simpson. The
samples (0, 0), (0.5, 0.75), (1, 1), the corners of an ROC curve, have
trapezoid area 0.625.

Golden values come from scipy 1.17.1 (course/fixtures/M01.4/scipy_golden.json,
written by course/oracle/M01.4/scipy_golden.py).
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
from tinyllm.num.integrate import simpson, trapezoid, trapezoid_rule

FIX = (
    Path(os.environ.get("TINYLLM_FIXTURES", "course/fixtures"))
    / "M01.4"
    / "scipy_golden.json"
)
SEED = int(os.environ.get("OL_SEED", "0"))

# The same table as the oracle: name -> vectorized float64 function.
FUNCS = {
    "sin": np.sin,
    "exp": np.exp,
    "runge": lambda x: 1.0 / (1.0 + 25.0 * x * x),
    "quartic": lambda x: x**4 - 3.0 * x**2 + x,
    "sqrt1p": lambda x: np.sqrt(1.0 + x),
    "gauss": lambda x: np.exp(-x * x / 2.0),
}


def cube(x):
    return x**3


def golden() -> dict:
    return json.loads(FIX.read_text())


def slope(hs, errs) -> float:
    """Least-squares slope of log(err) against log(h): the order of the rule."""
    return float(np.polyfit(np.log(hs), np.log(errs), 1)[0])


# --- the worked example -------------------------------------------------------


def test_hand_example():
    # WHY: the chapter's worked example, number for number: x**3 on [0, 2]
    #      (exactly 4). Trapezoid with 2 steps: 1 * (0/2 + 1 + 8/2) = 5; with
    #      4 steps: 0.5 * (0 + 0.125 + 1 + 3.375 + 4) = 4.25, a quarter of the
    #      error. Simpson with 2 steps: 1/3 * (0 + 4 * 1 + 8) = 4, exact.
    # KIND: unit, smoke
    # CATCHES: s02, s04, s06, s07, m05, m06
    # CHAPTER: M01.4 section 3, Worked example by hand
    assert_close(trapezoid_rule(cube, 0.0, 2.0, 2), 5.0, rtol=1e-14, atol=1e-14)
    assert_close(trapezoid_rule(cube, 0.0, 2.0, 4), 4.25, rtol=1e-14, atol=1e-14)
    assert_close(simpson(cube, 0.0, 2.0, 2), 4.0, rtol=1e-14, atol=1e-14)
    assert_close(simpson(cube, 0.0, 2.0, 4), 4.0, rtol=1e-14, atol=1e-14)


def test_hand_example_samples():
    # WHY: the ROC-shaped samples of section 3: two trapezoids,
    #      0.5 * (0 + 0.75) / 2 + 0.5 * (0.75 + 1) / 2 = 0.1875 + 0.4375 =
    #      0.625. A left Riemann sum (heights at the left end only) gives 0.375.
    # KIND: unit, smoke
    # CATCHES: s01
    # CHAPTER: M01.4 section 3, Worked example by hand
    assert trapezoid([0.0, 0.75, 1.0], [0.0, 0.5, 1.0]) == 0.625


# --- against scipy ------------------------------------------------------------


def test_rules_match_scipy():
    # WHY: on six functions (smooth, peaked, oscillating) and a reversed
    #      interval, trapezoid_rule and simpson on n equal steps must equal
    #      scipy.integrate.trapezoid and scipy.integrate.simpson on the same
    #      nodes, numpy.linspace(a, b, n + 1).
    # KIND: golden
    # CATCHES: s02, s04, s06, s07, m05, m06
    # CHAPTER: M01.4 section 2.3, The composite rules
    for name, a, b, n, want_t, want_s, _exact in golden()["rules"]:
        f = FUNCS[name]
        assert_close(
            trapezoid_rule(f, a, b, n),
            want_t,
            rtol=1e-12,
            atol=1e-13,
            msg=f"T {name} n={n}",
        )
        assert_close(
            simpson(f, a, b, n), want_s, rtol=1e-12, atol=1e-13, msg=f"S {name} n={n}"
        )


def test_trapezoid_samples_match_scipy():
    # WHY: samples off a grid (uneven steps, an unsorted order, a decreasing
    #      order) are integrated step by step with each step's own width and
    #      sign, exactly as scipy.integrate.trapezoid(y, x). ROC curves and
    #      reliability curves (M07.7, ethics.04) are such samples.
    # KIND: golden
    # CATCHES: s01, s05, s08
    # CHAPTER: M01.4 section 2.4, Integrating samples
    for kind, x, y, want in golden()["samples"]:
        assert_close(trapezoid(y, x), want, rtol=1e-12, atol=1e-13, msg=kind)


# --- order of accuracy --------------------------------------------------------


def test_trapezoid_error_order_two():
    # WHY: halving h divides the trapezoid error by 4: the log-log slope of
    #      the error against h is 2 (section 2.5). Exact values from scipy's
    #      quad. A rule that does not halve its end points has slope 1.
    # KIND: property
    # CATCHES: s04
    # CHAPTER: M01.4 section 2.5, Error and order
    exact = {r[0]: r[6] for r in golden()["rules"] if r[1] < r[2]}
    for name, a, b in [("sin", 0.0, math.pi), ("exp", 0.0, 1.0), ("sqrt1p", 0.0, 3.0)]:
        ns = np.array([4, 8, 16, 32, 64])
        errs = [
            abs(trapezoid_rule(FUNCS[name], a, b, int(n)) - exact[name]) for n in ns
        ]
        assert 1.9 <= slope((b - a) / ns, errs) <= 2.1, (name, errs)


def test_simpson_error_order_four():
    # WHY: Simpson's error falls like h**4: halving h divides it by 16. The
    #      log-log slope is 4; wrong weights fall back to order 2 or worse.
    # KIND: property
    # CATCHES: s02, s06
    # CHAPTER: M01.4 section 2.5, Error and order
    exact = {r[0]: r[6] for r in golden()["rules"] if r[1] < r[2]}
    for name, a, b in [("sin", 0.0, math.pi), ("exp", 0.0, 1.0), ("gauss", -3.0, 3.0)]:
        ns = np.array([8, 16, 32, 64])
        errs = [abs(simpson(FUNCS[name], a, b, int(n)) - exact[name]) for n in ns]
        assert 3.8 <= slope((b - a) / ns, errs) <= 4.2, (name, errs)


# --- exactness ----------------------------------------------------------------


def test_simpson_exact_on_cubics():
    # WHY: a parabola through three points integrates every cubic exactly
    #      (the x**3 error terms cancel by symmetry, section 2.5): for random
    #      cubics on random intervals and any even n, Simpson equals the exact
    #      antiderivative difference to rounding.
    # KIND: property
    # CATCHES: s02, s06, s07, m06
    # CHAPTER: M01.4 section 2.5, Error and order
    rng = PCG32(seed=SEED)
    for _ in range(40):
        c = [4.0 * rng.uniform() - 2.0 for _ in range(4)]
        a = 6.0 * rng.uniform() - 3.0
        b = a + 0.5 + 3.0 * rng.uniform()

        def p(x, c=c):
            return c[0] + c[1] * x + c[2] * x**2 + c[3] * x**3

        def anti(x, c=c):
            return c[0] * x + c[1] * x**2 / 2 + c[2] * x**3 / 3 + c[3] * x**4 / 4

        want = anti(b) - anti(a)
        scale = sum(abs(t) for t in c) * max(abs(a), abs(b), 1.0) ** 4
        for n in (2, 4, 10):
            assert abs(simpson(p, a, b, n) - want) <= 1e-13 * scale, (c, a, b, n)


def test_trapezoid_rule_exact_on_lines():
    # WHY: a trapezoid under a straight line is the exact area, so the rule
    #      is exact on f(x) = m x + c for any n; it is not exact on x**2.
    # KIND: property
    # CATCHES: s04, m05
    # CHAPTER: M01.4 section 2.2, The trapezoid
    rng = PCG32(seed=SEED + 1)
    for _ in range(20):
        m, c = 4.0 * rng.uniform() - 2.0, 4.0 * rng.uniform() - 2.0
        a, b = -1.0 - rng.uniform(), 1.0 + rng.uniform()
        want = m * (b * b - a * a) / 2 + c * (b - a)
        for n in (1, 3, 7):
            assert abs(trapezoid_rule(lambda x: m * x + c, a, b, n) - want) <= 1e-13
    assert trapezoid_rule(lambda x: x * x, 0.0, 1.0, 1) == 0.5  # exact is 1/3


def test_simpson_is_richardson_of_trapezoid():
    # WHY: the trapezoid error is c2 h**2 + c4 h**4 + ...; (4 T(h/2) - T(h)) / 3
    #      cancels the h**2 term, and that combination is Simpson's rule with
    #      the same nodes (section 2.6, the M01.1 Richardson step). Equal to
    #      rounding for every function, not just on average.
    # KIND: property
    # CATCHES: s02, s06, s07
    # CHAPTER: M01.4 section 2.6, Simpson is Richardson
    for name, a, b in [("sin", 0.0, 2.0), ("runge", -1.0, 1.0), ("gauss", -2.0, 3.0)]:
        f = FUNCS[name]
        for n in (2, 6, 20):
            rich = (
                4.0 * trapezoid_rule(f, a, b, n) - trapezoid_rule(f, a, b, n // 2)
            ) / 3.0
            assert_close(
                simpson(f, a, b, n), rich, rtol=1e-13, atol=1e-14, msg=f"{name} n={n}"
            )


# --- boundaries and the interface ---------------------------------------------


def test_reversed_and_empty_intervals():
    # WHY: the integral from b to a is minus the integral from a to b, and
    #      over [a, a] it is 0. The nodes run from a to b, so h is negative
    #      and the sign comes out by itself: no special case, no abs().
    # KIND: boundary
    # CATCHES: s05
    # CHAPTER: M01.4 section 2.1, The definite integral
    for rule, n in ((trapezoid_rule, 6), (simpson, 6)):
        fwd = rule(np.exp, -0.5, 1.0, n)
        assert_close(rule(np.exp, 1.0, -0.5, n), -fwd, rtol=1e-14, atol=1e-15)
        assert rule(np.exp, 0.3, 0.3, n) == 0.0
    assert trapezoid([1.0, 3.0], [2.0, 0.0]) == -4.0


def test_f_called_once_on_linspace_nodes():
    # WHY: f is evaluated once, vectorized, on numpy.linspace(a, b, n + 1):
    #      the contract fixes the nodes (both ends exact) so every language
    #      and scipy agree to the last bit. A per-node Python loop is n times
    #      slower on the safety report's curves.
    # KIND: unit
    # CATCHES: s06
    # CHAPTER: M01.4 section 4, The interface
    for rule in (trapezoid_rule, simpson):
        seen = []

        def f(x):
            seen.append(np.array(x, copy=True))
            return np.cos(x)

        rule(f, -0.3, 1.7, 8)
        assert len(seen) == 1
        assert seen[0].dtype == np.float64
        assert np.array_equal(seen[0], np.linspace(-0.3, 1.7, 9))


def test_returns_python_float():
    # WHY: callers (roc_auc in M07.7, the safety report) store and compare
    #      the result as a number; the contract says a Python float, not a
    #      numpy scalar.
    # KIND: unit
    # CATCHES: m04
    # CHAPTER: M01.4 section 4, The interface
    assert type(trapezoid(np.array([1.0, 2.0]), np.array([0.0, 1.0]))) is float
    assert type(trapezoid_rule(np.sin, 0.0, 1.0, 3)) is float
    assert type(simpson(np.sin, 0.0, 1.0, 4)) is float


def test_trapezoid_short_inputs():
    # WHY: one sample or none encloses no area: 0.0, as numpy.trapezoid. An
    #      ROC curve of one point (one threshold) is degenerate, not an error.
    # KIND: boundary, smoke
    # CHAPTER: M01.4 section 2.4, Integrating samples
    assert trapezoid([], []) == 0.0
    assert trapezoid([5.0], [1.0]) == 0.0


def test_rejects_bad_arguments():
    # WHY: Simpson's weights only fit an even number of steps; an odd n must
    #      raise, never be rounded or silently drop a step. Non-finite limits,
    #      a wrong-shaped or non-finite f, and mismatched samples are caller
    #      bugs that must surface here, not as a NaN AUC two modules later.
    # KIND: boundary
    # CATCHES: s03, m01, m02, m03, m07, m08
    # CHAPTER: M01.4 section 5, Pitfalls
    for n in (1, 3, 7, 0, -2, 2.5):
        with pytest.raises(ValueError):
            simpson(np.sin, 0.0, 1.0, n)
    for n in (0, -1, 2.5):
        with pytest.raises(ValueError):
            trapezoid_rule(np.sin, 0.0, 1.0, n)
    for rule, n in ((trapezoid_rule, 4), (simpson, 4)):
        with pytest.raises(ValueError):
            rule(np.sin, 0.0, math.inf, n)
        with pytest.raises(ValueError):
            rule(np.sin, math.nan, 1.0, n)
        with pytest.raises(ValueError):
            rule(lambda x: 1.0, 0.0, 1.0, n)  # a scalar, not one value per node
        with pytest.raises(ValueError):
            rule(lambda x: np.full_like(x, np.nan), -1.0, 1.0, n)  # NaN on the nodes
    with pytest.raises(ValueError):
        trapezoid([1.0, 2.0], [0.0, 1.0, 2.0])
    with pytest.raises(ValueError):
        trapezoid([[1.0, 2.0]], [[0.0, 1.0]])
    with pytest.raises(ValueError):
        trapezoid([1.0, math.nan], [0.0, 1.0])
    with pytest.raises(ValueError):
        trapezoid([1.0, 2.0], [0.0, math.inf])
