"""Finite-difference derivatives (M01.1).

A derivative is a limit, f'(x) = lim_{h -> 0} (f(x + h) - f(x)) / h, and a
computer cannot take a limit: it picks one h. Two errors pull in opposite
directions. Truncation (the Taylor terms the formula drops) shrinks with h:
O(h) for the forward difference, O(h**2) for the central one. Rounding
(f is only known to about eps * |f|, and that error is divided by h) grows
as h shrinks. The default steps balance the two: sqrt(eps) for forward,
cbrt(eps) for central, scaled by max(1, |x|) so the step is never lost
below the spacing of floats near x.

Contract: contracts/py/tinyllm/num/diff.pyi. M04.1's gradcheck calls
central_diff once per coordinate; M01.2's Newton uses it when no derivative
is given.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

EPS = 2.0**-52  # float64 machine epsilon


def _check_x(x: float) -> float:
    x = float(x)
    if not math.isfinite(x):
        raise ValueError(f"x must be finite, got {x}")
    return x


def _check_h(h: float) -> float:
    h = float(h)
    if not math.isfinite(h) or h <= 0:
        raise ValueError(f"h must be finite and > 0, got {h}")
    return h


def forward_diff(
    f: Callable[[float], float], x: float, h: Optional[float] = None
) -> float:
    x = _check_x(x)
    h = math.sqrt(EPS) * max(1.0, abs(x)) if h is None else _check_h(h)
    xp = x + h
    # Divide by the step the hardware actually took: x + h is rounded, so
    # (x + h) - x is usually not h. Using it makes f(x) = x exact.
    return (float(f(xp)) - float(f(x))) / (xp - x)


def central_diff(
    f: Callable[[float], float], x: float, h: Optional[float] = None
) -> float:
    x = _check_x(x)
    h = math.cbrt(EPS) * max(1.0, abs(x)) if h is None else _check_h(h)
    xp, xm = x + h, x - h
    return (float(f(xp)) - float(f(xm))) / (xp - xm)


def richardson(
    f: Callable[[float], float], x: float, h: float, levels: int = 2
) -> float:
    x = _check_x(x)
    h = _check_h(h)
    if int(levels) != levels or levels < 0:
        raise ValueError(f"levels must be an integer >= 0, got {levels}")
    levels = int(levels)
    # Column 0: central differences at h, h/2, h/4, ... Each later column
    # cancels the next even power of h: D(h) = f' + c2 h^2 + c4 h^4 + ...,
    # so (4^k D(h/2) - D(h)) / (4^k - 1) removes the h^(2k) term.
    col = [central_diff(f, x, h / 2.0**j) for j in range(levels + 1)]
    for k in range(1, levels + 1):
        p = 4.0**k
        col = [(p * col[j + 1] - col[j]) / (p - 1.0) for j in range(len(col) - 1)]
    return col[0]
