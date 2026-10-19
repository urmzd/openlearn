"""Definite integrals by quadrature (M01.4).

The integral of f from a to b is the signed area under its graph, the limit
of sums of thin slices. A computer stops at n slices and approximates f on
each: by a straight line (the trapezoid rule, error O(h**2)) or by a parabola
through three points (Simpson's rule, error O(h**4)). Samples that are not on
a grid (an ROC curve's corners) are integrated by the trapezoid rule over the
given points.

Contract: contracts/py/tinyllm/num/integrate.pyi. M07.7's roc_auc is
trapezoid(tpr, fpr); ethics.04 integrates its reliability and ROC curves the
same way.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _samples(y: ArrayLike, x: ArrayLike) -> tuple[NDArray, NDArray]:
    ya = np.asarray(y, dtype=np.float64)
    xa = np.asarray(x, dtype=np.float64)
    if ya.ndim != 1 or xa.ndim != 1:
        raise ValueError(f"x and y must be 1-D, got shapes {xa.shape} and {ya.shape}")
    if ya.shape != xa.shape:
        raise ValueError(
            f"x and y must have the same length, got {xa.size} and {ya.size}"
        )
    if not (np.all(np.isfinite(ya)) and np.all(np.isfinite(xa))):
        raise ValueError("x and y must be finite")
    return ya, xa


def trapezoid(y: ArrayLike, x: ArrayLike) -> float:
    ya, xa = _samples(y, x)
    if ya.size < 2:
        return 0.0
    # One trapezoid per step: width times the mean of its two heights. The
    # widths keep their sign, so the order of the samples is the direction.
    return float(np.sum(np.diff(xa) * (ya[:-1] + ya[1:])) / 2.0)


def _nodes(
    f: Callable[[NDArray], NDArray], a: float, b: float, n: int
) -> tuple[NDArray, float]:
    a, b = float(a), float(b)
    if not (math.isfinite(a) and math.isfinite(b)):
        raise ValueError(f"a and b must be finite, got {a} and {b}")
    xs = np.linspace(a, b, n + 1)
    fx = np.asarray(f(xs), dtype=np.float64)
    if fx.shape != xs.shape:
        raise ValueError(f"f must return shape {xs.shape}, got {fx.shape}")
    if not np.all(np.isfinite(fx)):
        raise ValueError("f returned a non-finite value on the nodes")
    return fx, (b - a) / n


def _steps(n: int) -> int:
    if int(n) != n:
        raise ValueError(f"n must be an integer, got {n!r}")
    return int(n)


def trapezoid_rule(
    f: Callable[[NDArray], NDArray], a: float, b: float, n: int
) -> float:
    n = _steps(n)
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    fx, h = _nodes(f, a, b, n)
    # The two ends belong to one trapezoid each, every inner node to two.
    return float(h * (fx[0] / 2.0 + np.sum(fx[1:-1]) + fx[-1] / 2.0))


def simpson(f: Callable[[NDArray], NDArray], a: float, b: float, n: int) -> float:
    n = _steps(n)
    if n < 2 or n % 2 != 0:
        raise ValueError(f"Simpson needs an even number of steps n >= 2, got {n}")
    fx, h = _nodes(f, a, b, n)
    # Weights 1, 4, 2, 4, ..., 2, 4, 1: odd nodes are parabola midpoints,
    # even inner nodes are shared by two parabolas.
    odd = np.sum(fx[1:-1:2])
    even = np.sum(fx[2:-1:2])
    return float(h / 3.0 * (fx[0] + 4.0 * odd + 2.0 * even + fx[-1]))
