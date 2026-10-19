"""Polynomials, Horner's rule, and stable quadratic roots (M00.4).

A polynomial p(x) = c[0] + c[1] x + ... + c[n] x^n is stored by its
coefficients in ASCENDING order of power. Horner's rule evaluates it as
c[0] + x (c[1] + x (c[2] + ... + x c[n])) with n multiplications and n
additions. M02.1 evaluates Taylor polynomials with it, and M09.6 runs the
same loop in C inside tl_expf.

Contract: contracts/py/tinyllm/num/poly.pyi.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


def horner(coeffs: Sequence[float], x: ArrayLike) -> NDArray:
    c = [float(v) for v in coeffs]
    xs = np.asarray(x, dtype=np.float64)
    acc = np.zeros_like(xs)  # the empty sum: the zero polynomial
    # Innermost bracket first: start at the highest power and work down.
    for ck in reversed(c):
        acc = acc * xs + ck
    return acc


def quadratic_roots(a: float, b: float, c: float) -> tuple[float, float]:
    a, b, c = float(a), float(b), float(c)
    if not all(math.isfinite(v) for v in (a, b, c)):
        raise ValueError(f"coefficients must be finite, got {a}, {b}, {c}")
    if a == 0.0:
        raise ValueError("a == 0: the equation is linear, not quadratic")
    # Scale by a power of two (exact) so that b * b cannot overflow when |b|
    # is near 1e200; the roots of s a x^2 + s b x + s c are the same.
    e = math.frexp(max(abs(a), abs(b), abs(c)))[1]
    a, b, c = math.ldexp(a, -e), math.ldexp(b, -e), math.ldexp(c, -e)
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        raise ValueError(f"discriminant {disc} < 0: the roots are complex")
    # q = -(b + sign(b) sqrt(disc)) / 2 adds two numbers of the same sign, so
    # nothing cancels. sign(0) is taken as +1: numpy's sign(0) = 0 would
    # make q = 0 for x^2 - 4.
    sign = 1.0 if b >= 0.0 else -1.0
    q = -0.5 * (b + sign * math.sqrt(disc))
    if q == 0.0:  # b == 0 and disc == 0, so c == 0: the double root 0
        return (0.0, 0.0)
    # One root is q / a; Vieta (x1 x2 = c / a) gives the other as c / q.
    r1, r2 = q / a, c / q
    return (r1, r2) if r1 <= r2 else (r2, r1)
