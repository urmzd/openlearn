"""Newton's method (M01.2).

Replace f by its tangent line at x[n], f(x) ~ f(x[n]) + f'(x[n]) (x - x[n]),
and jump to where the tangent crosses zero. Near a simple root the error
squares every step (the number of correct digits doubles), which is why
two Newton steps turn a 3-digit guess for 1/sqrt(x) into a float32-accurate
one: the trick behind M09.5's C rsqrt and the RMSNorm kernel that uses it.

Contract: contracts/py/tinyllm/num/newton.pyi.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tinyllm.num.diff import central_diff


def newton(
    f: Callable[[float], float],
    df: Optional[Callable[[float], float]],
    x0: float,
    tol: float = 1e-12,
    max_iter: int = 50,
) -> tuple[float, int]:
    x = float(x0)
    if not math.isfinite(x):
        raise ValueError(f"x0 must be finite, got {x0}")
    if not tol > 0:
        raise ValueError(f"tol must be > 0, got {tol}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")
    deriv = df if df is not None else (lambda t: central_diff(f, t))
    for n in range(max_iter):
        fx = float(f(x))
        if fx == 0.0:
            return x, n
        d = float(deriv(x))
        if d == 0.0 or not math.isfinite(d):
            raise RuntimeError(
                f"derivative is {d} at x = {x!r} (step {n}): the tangent has no root"
            )
        x_new = x - fx / d
        if not math.isfinite(x_new):
            raise RuntimeError(f"iterate {n + 1} is {x_new}: Newton diverged")
        # Relative step test: near a root of size 1e10 an absolute 1e-12 is
        # below the float spacing and could never be met.
        if abs(x_new - x) <= tol * max(1.0, abs(x_new)):
            return x_new, n + 1
        x = x_new
    raise RuntimeError(f"no convergence after {max_iter} updates (last iterate {x!r})")


def rsqrt_newton(x: ArrayLike, y0: ArrayLike, iters: int) -> NDArray:
    if iters < 0:
        raise ValueError(f"iters must be >= 0, got {iters}")
    xa, ya = np.asarray(x), np.asarray(y0)
    dt = np.result_type(xa, ya)
    if not np.issubdtype(dt, np.floating):
        dt = np.dtype(np.float64)
    xa, y = np.broadcast_arrays(xa.astype(dt), ya.astype(dt))
    y = y.copy()
    half, three_halves = dt.type(0.5), dt.type(1.5)
    hx = half * xa
    for _ in range(int(iters)):
        y = y * (three_halves - hx * y * y)
    return y
