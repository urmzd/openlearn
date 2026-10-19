"""Gradient descent and the Armijo backtracking line search (M10.1).

On an L-smooth, mu-strongly convex function, gradient descent with step
1/L shrinks the optimality gap by at least (1 - 1/kappa) per step, where
kappa = L / mu is the condition number. A line search finds a step that
decreases f enough without knowing L.

Contract: contracts/py/tinyllm/optim/gd.pyi.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

MAX_REDUCTIONS = 60


def gradient_descent(
    f: Callable[[NDArray], float],
    grad: Callable[[NDArray], NDArray],
    x0: ArrayLike,
    lr: float,
    steps: int,
) -> list[NDArray]:
    if not lr > 0:
        raise ValueError(f"lr must be > 0, got {lr}")
    if steps < 0:
        raise ValueError(f"steps must be >= 0, got {steps}")
    x = np.array(x0, dtype=np.float64, copy=True)
    out = [x]
    for t in range(steps + 1):
        fx = float(f(x))
        if not (math.isfinite(fx) and np.isfinite(x).all()):
            raise FloatingPointError(f"gradient descent diverged at step {t}: f(x_t) = {fx}")
        if t == steps:
            break
        x = x - lr * np.asarray(grad(x), dtype=np.float64)
        out.append(x)
    return out


def armijo_step(
    f: Callable[[NDArray], float],
    grad: Callable[[NDArray], NDArray],
    x: ArrayLike,
    d: ArrayLike,
    alpha0: float = 1.0,
    c: float = 1e-4,
    rho: float = 0.5,
) -> float:
    if not (0 < c < 1 and 0 < rho < 1 and alpha0 > 0):
        raise ValueError(f"need 0 < c < 1, 0 < rho < 1, alpha0 > 0; got c={c}, rho={rho}, alpha0={alpha0}")
    x = np.asarray(x, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    fx = float(f(x))
    slope = float(np.dot(np.ravel(grad(x)), np.ravel(d)))
    if not slope < 0:
        raise ValueError(f"d is not a descent direction: grad(x) . d = {slope}")
    alpha = alpha0
    for _ in range(MAX_REDUCTIONS + 1):
        # Written so that a nan f(x + alpha d) fails the test and shrinks alpha.
        if f(x + alpha * d) <= fx + c * alpha * slope:
            return alpha
        alpha *= rho
    raise RuntimeError(f"no step satisfies the Armijo condition after {MAX_REDUCTIONS} reductions")
