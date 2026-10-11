"""Course tests for M00.4: polynomials, Horner, stable quadratic roots
(tinyllm/num/poly.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/M00.4), and the chapter section it comes from.

The chapter's worked examples (section 3): p(x) = 1 + 2x + 3x^2 at x = 2 is
1 + 2 (2 + 3 * 2) = 17 by Horner; x^2 - 5x + 6 has roots 2 and 3; and
x^2 - 10^8 x + 1 has roots 1e-8 and 1e8, where the textbook formula returns
7.45e-9 for the small one.
"""

from __future__ import annotations

import json
import math
import os
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32
from tinyllm.num.poly import horner, quadratic_roots

GOLDEN = Path(os.environ.get("TINYLLM_FIXTURES", "")) / "M00.4" / "poly.json"
U = 2.0**-53  # float64 unit roundoff


def seed() -> int:
    return int(os.environ.get("OL_SEED", "0"))


def gamma(k: int) -> float:
    """gamma_k = k u / (1 - k u): the bound on k rounded operations."""
    return k * U / (1 - k * U)


def horner_bound(coeffs, x: float) -> float:
    """Horner's error bound for degree n: gamma_{2n} * sum_k |c_k| |x|^k."""
    n = max(len(coeffs) - 1, 0)
    return gamma(2 * n) * sum(abs(c) * abs(x) ** k for k, c in enumerate(coeffs))


# --- the worked examples ----------------------------------------------------------


def test_hand_example_horner():
    # WHY: the chapter's worked example: p(x) = 1 + 2x + 3x^2 at x = 2.
    #      Horner starts at the top: 3; 3 * 2 + 2 = 8; 8 * 2 + 1 = 17.
    # KIND: unit
    # CATCHES: s01
    # CHAPTER: M00.4 section 3, Worked example by hand
    assert_close(horner([1.0, 2.0, 3.0], 2.0), 17.0, dtype="float64")
    assert_close(
        horner([1.0, 2.0, 3.0], [0.0, 1.0, -1.0, 2.0]),
        [1.0, 6.0, 2.0, 17.0],
        dtype="float64",
    )


def test_hand_example_roots():
    # WHY: x^2 - 5x + 6 = (x - 2)(x - 3), and the section 3 cancellation case:
    #      x^2 - 1e8 x + 1 has roots 1e-8 and 1e8 (to 16 digits). The
    #      textbook (-b - sqrt(b^2 - 4ac)) / 2a gives 7.45e-9 for the small
    #      one: 25% wrong.
    # KIND: unit
    # CATCHES: s02, s04, s05, m01, m02
    # CHAPTER: M00.4 section 3, Worked example by hand
    r = quadratic_roots(1.0, -5.0, 6.0)
    assert isinstance(r, tuple) and len(r) == 2
    assert_close(r, (2.0, 3.0), dtype="float64")
    assert_close(quadratic_roots(1.0, -1e8, 1.0), (1e-8, 1e8), dtype="float64")


# --- Horner ---------------------------------------------------------------------------------


def test_coefficients_are_ascending():
    # WHY: coeffs[k] multiplies x^k. [1, 0] is the constant 1 and [0, 1] is
    #      x; numpy.polyval uses the opposite (descending) order, and mixing
    #      them up is pitfall 1.
    # KIND: boundary
    # CATCHES: s01
    # CHAPTER: M00.4 section 5, Pitfalls, item 1
    assert_close(horner([1.0, 0.0], 5.0), 1.0, dtype="float64")
    assert_close(horner([0.0, 1.0], 5.0), 5.0, dtype="float64")
    assert_close(horner([2.0, 0.0, 0.0, 1.0], 3.0), 29.0, dtype="float64")


def test_horner_exact_on_integer_polynomials():
    # WHY: with integer coefficients and points, every partial result of
    #      Horner is an integer below 2^53, so float64 computes it exactly.
    #      The check is Python's exact integer arithmetic.
    # KIND: property
    # CATCHES: s01
    # CHAPTER: M00.4 section 2.2, Horner's rule
    rng = PCG32(seed=seed())
    for _ in range(100):
        deg = rng.below(7)
        coeffs = [rng.below(41) - 20 for _ in range(deg + 1)]
        xs = [rng.below(21) - 10 for _ in range(5)]
        got = horner([float(c) for c in coeffs], np.array(xs, dtype=np.float64))
        want = [sum(c * x**k for k, c in enumerate(coeffs)) for x in xs]
        assert got.tolist() == [float(w) for w in want], (coeffs, xs)


def test_horner_within_error_bound():
    # WHY: on real coefficients Horner rounds 2n times, so its error is at
    #      most gamma_{2n} * sum |c_k| |x|^k (section 2.3). The reference
    #      value is exact rational arithmetic on the same float64 inputs.
    # KIND: property
    # CATCHES: s01, s06
    # CHAPTER: M00.4 section 2.3, How accurate is Horner?
    rng = PCG32(seed=seed())
    for _ in range(60):
        deg = 1 + rng.below(9)
        coeffs = [rng.uniform() * 4 - 2 for _ in range(deg + 1)]
        x = rng.uniform() * 4 - 2
        exact = sum(Fraction(c) * Fraction(x) ** k for k, c in enumerate(coeffs))
        err = abs(Fraction(float(horner(coeffs, x))) - exact)
        assert float(err) <= horner_bound(coeffs, x), (coeffs, x)


def test_horner_golden():
    # WHY: Taylor polynomials of exp and cos (what M02.1 and M09.6 evaluate)
    #      and (x-1)(x-2)(x-3)(x-4)(x-5), against mpmath at 80 digits, within
    #      the Horner error bound.
    # KIND: golden
    # CATCHES: s01, s06
    # CHAPTER: M00.4 section 2.3, How accurate is Horner?
    for case in json.loads(GOLDEN.read_text())["horner"]:
        got = horner(case["coeffs"], np.array(case["x"]))
        for x, g, want in zip(case["x"], got, case["p"]):
            bound = horner_bound(case["coeffs"], x) + abs(want) * U
            assert abs(g - want) <= bound, (case["name"], x, g, want)


def test_horner_shapes():
    # WHY: x may be a scalar (a float64 scalar comes back) or any array (same shape); the
    #      result is float64. No coefficients is the zero polynomial, and a
    #      single coefficient is a constant broadcast over x.
    # KIND: boundary
    # CATCHES: s01, s06, s07
    # CHAPTER: M00.4 section 4, The interface
    s = horner([1.0, 1.0], 2)
    assert np.shape(s) == () and np.asarray(s).dtype == np.float64 and float(s) == 3.0
    g = horner([1.0, -1.0, 0.5], np.zeros((2, 3), dtype=np.float32))
    assert g.shape == (2, 3) and g.dtype == np.float64
    assert_close(g, np.ones((2, 3)), dtype="float64")
    z = horner([], np.array([1.0, 2.0]))
    assert z.shape == (2,) and z.tolist() == [0.0, 0.0]
    assert horner([4.0], np.arange(3.0)).tolist() == [4.0, 4.0, 4.0]


# --- quadratic roots -----------------------------------------------------------------------


def test_roots_golden_cancellation():
    # WHY: 15 quadratics with b^2 much larger than 4ac (the small root is
    #      1e-15 of the large one), solved by mpmath at 80 digits. Both roots
    #      must be right to float64 precision; the textbook formula loses up
    #      to all the digits of the small root.
    # KIND: golden
    # CATCHES: s02, s04, s05, m01, m02
    # CHAPTER: M00.4 section 2.5, Cancellation and the stable formula
    for case in json.loads(GOLDEN.read_text())["quadratics"]:
        got = quadratic_roots(case["a"], case["b"], case["c"])
        assert_close(
            got,
            case["roots"],
            dtype="float64",
            msg=f"{case['a']}, {case['b']}, {case['c']}",
        )


def test_b_zero():
    # WHY: x^2 - 4 has b = 0. Taking sign(0) = 0, as numpy.sign does, makes
    #      q = 0 and then c / q divides by zero (pitfall 3).
    # KIND: boundary
    # CATCHES: s03, s04, m01
    # CHAPTER: M00.4 section 5, Pitfalls, item 3
    assert_close(quadratic_roots(1.0, 0.0, -4.0), (-2.0, 2.0), dtype="float64")
    assert_close(quadratic_roots(2.0, 0.0, -8.0), (-2.0, 2.0), dtype="float64")


def test_special_roots():
    # WHY: c = 0 makes one root exactly 0; a double root comes back twice;
    #      a negative leading coefficient has the same roots as its negation;
    #      and a != 1 checks that the second root is c / q, not c / x1.
    # KIND: unit
    # CATCHES: s04, s05, m01, m02
    # CHAPTER: M00.4 section 2.4, The quadratic formula
    assert_close(quadratic_roots(1.0, 3.0, 0.0), (-3.0, 0.0), dtype="float64")
    assert_close(quadratic_roots(1.0, -2.0, 1.0), (1.0, 1.0), dtype="float64")
    assert_close(quadratic_roots(-1.0, 5.0, -6.0), (2.0, 3.0), dtype="float64")
    assert_close(quadratic_roots(2.0, -10.0, 12.0), (2.0, 3.0), dtype="float64")
    assert quadratic_roots(5.0, 0.0, 0.0) == (0.0, 0.0)


def test_roots_ascending():
    # WHY: the first root is the smaller one, always, whatever the signs.
    # KIND: unit
    # CATCHES: s05, m01
    # CHAPTER: M00.4 section 4, The interface
    for a, b, c in [
        (1.0, -5.0, 6.0),
        (1.0, 5.0, 6.0),
        (-3.0, 1.0, 2.0),
        (1.0, -1e8, 1.0),
    ]:
        r1, r2 = quadratic_roots(a, b, c)
        assert r1 <= r2


def test_vieta_relations():
    # WHY: for any real-rooted quadratic, x1 + x2 = -b/a and x1 x2 = c/a
    #      (expand a (x - x1)(x - x2)). Random well-separated cases.
    # KIND: property
    # CATCHES: s04, m02
    # CHAPTER: M00.4 section 2.4, The quadratic formula
    rng = PCG32(seed=seed())
    for _ in range(200):
        x1, x2 = rng.uniform() * 20 - 10, rng.uniform() * 20 - 10
        if abs(x1 - x2) < 0.1:  # nearly double roots are ill-conditioned; skip them
            continue
        a = (rng.uniform() * 4 + 0.5) * (1 if rng.below(2) else -1)
        b, c = -a * (x1 + x2), a * x1 * x2
        r1, r2 = quadratic_roots(a, b, c)
        scale = abs(x1) + abs(x2) + 1.0
        assert_close(r1 + r2, -b / a, rtol=1e-9, atol=1e-9 * scale)
        assert_close(r1 * r2, c / a, rtol=1e-9, atol=1e-9 * scale * scale)


def test_huge_b_does_not_overflow():
    # WHY: b = 1e200 makes b * b overflow to inf, and the textbook formula
    #      returns inf and 0. Scaling the three coefficients by a power of
    #      two first (exact) keeps the roots -1e200 and -1e-200.
    # KIND: boundary
    # CATCHES: s08, m01, m02
    # CHAPTER: M00.4 section 5, Pitfalls, item 4
    assert_close(quadratic_roots(1.0, 1e200, 1.0), (-1e200, -1e-200), dtype="float64")
    assert_close(
        quadratic_roots(1e-200, 1.0, 1e-200), (-1e200, -1e-200), dtype="float64"
    )


@pytest.mark.parametrize(
    "a,b,c",
    [
        (0.0, 2.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
        (math.nan, 1.0, 0.0),
        (1.0, math.inf, 0.0),
    ],
)
def test_rejects_non_quadratics(a, b, c):
    # WHY: a = 0 is a linear equation; a negative discriminant has complex
    #      roots; NaN and inf are not coefficients. ValueError, never a
    #      ZeroDivisionError or a pair of NaNs.
    # KIND: boundary
    # CATCHES: s09, s10
    # CHAPTER: M00.4 section 4, The interface
    with pytest.raises(ValueError):
        quadratic_roots(a, b, c)
