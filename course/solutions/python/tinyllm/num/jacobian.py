"""Jacobians, the multivariable chain rule, numeric JVP and VJP (M04.2).

For f: R^n -> R^m the Jacobian J is the m x n matrix of partial derivatives;
it is the best linear approximation f(x + d) ~ f(x) + J d. The chain rule
says the Jacobian of a composition is the product of the Jacobians, and
autodiff never forms J: forward mode pushes a direction through, J v;
reverse mode pulls a cotangent back, u^T J. These numeric versions are the
oracles the autodiff modules (M08.1 to M08.3) and the manual BPTT of L3.1
are checked against.

Contract: contracts/py/tinyllm/num/jacobian.pyi.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tinyllm.num.gradcheck import numerical_grad

JVP_STEP = 1e-6


def _eval(f: Callable[[NDArray], ArrayLike], x: NDArray) -> NDArray:
    return np.asarray(f(x), dtype=np.float64)


def jacobian(
    f: Callable[[NDArray], ArrayLike], x: ArrayLike, eps: float = 1e-6
) -> NDArray:
    if not (math.isfinite(eps) and eps > 0):
        raise ValueError(f"eps must be finite and > 0, got {eps}")
    xw = np.array(x, dtype=np.float64, copy=True)
    flat = xw.reshape(-1)  # a view: writing flat[j] moves xw
    shape = None
    cols = []
    for j in range(flat.size):
        orig = flat[j]
        hi_x, lo_x = orig + eps, orig - eps
        flat[j] = hi_x
        hi = _eval(f, xw)
        flat[j] = lo_x
        lo = _eval(f, xw)
        flat[j] = orig
        if shape is None:
            shape = hi.shape
        if hi.shape != shape or lo.shape != shape:
            raise ValueError(
                f"f returned shapes {hi.shape} and {lo.shape}, earlier {shape}"
            )
        # Column j: how every output moves when input j moves.
        cols.append((hi - lo).reshape(-1) / (hi_x - lo_x))
    if not cols:
        m = _eval(f, xw).size
        return np.zeros((m, 0))
    return np.stack(cols, axis=1)


def jvp_numeric(
    f: Callable[[NDArray], ArrayLike], x: ArrayLike, v: ArrayLike
) -> NDArray:
    xw = np.array(x, dtype=np.float64, copy=True)
    va = np.asarray(v, dtype=np.float64)
    if va.shape != xw.shape:
        raise ValueError(f"v has shape {va.shape}, x has {xw.shape}")
    scale = float(np.max(np.abs(va))) if va.size else 0.0
    if scale == 0.0:
        return np.zeros_like(_eval(f, xw))
    # Step so the largest coordinate moves by JVP_STEP: g(t) = f(x + t v) has
    # g'(0) = J v, and with v of size 1e6 a fixed t = 1e-6 would move x by 1.
    h = JVP_STEP / scale
    return (_eval(f, xw + h * va) - _eval(f, xw - h * va)) / (2.0 * h)


def vjp_numeric(
    f: Callable[[NDArray], ArrayLike], x: ArrayLike, u: ArrayLike
) -> NDArray:
    xw = np.array(x, dtype=np.float64, copy=True)
    ua = np.asarray(u, dtype=np.float64)
    fx = _eval(f, xw)
    if ua.shape != fx.shape:
        raise ValueError(f"u has shape {ua.shape}, f(x) has {fx.shape}")
    # u^T J is the gradient of the scalar u . f(x): one gradient, not m rows.
    return numerical_grad(lambda z: float(np.sum(ua * _eval(f, z))), [xw])[0]
