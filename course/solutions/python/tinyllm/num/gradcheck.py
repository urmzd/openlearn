"""Partial derivatives, gradients, and gradcheck (M04.1).

The gradient of a scalar f(x_1, ..., x_n) collects one partial derivative
per input element: freeze every other element and take the one-variable
derivative (M01.1) along that coordinate. A backward pass claims to compute
exactly this vector; gradcheck holds it to the numbers.

Contract: contracts/py/tinyllm/num/gradcheck.pyi. L0.2's F.gradcheck_all()
runs every op of your autograd library through this file, and M04.2 builds
vector-Jacobian products on numerical_grad.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tinyllm.num.diff import central_diff


@dataclass
class GradcheckReport:
    ok: bool
    max_abs_err: float
    max_rel_err: float
    worst_input: int
    worst_index: tuple


def _scalar(v) -> float:
    a = np.asarray(v)
    if a.size != 1:
        raise ValueError(f"f must return one number, got an array of shape {a.shape}")
    return float(a.reshape(()))


def numerical_grad(
    f: Callable[..., float], inputs: list[ArrayLike], eps: float = 1e-6
) -> list[NDArray]:
    if len(inputs) == 0:
        raise ValueError("inputs is empty: there is nothing to differentiate")
    if not (math.isfinite(eps) and eps > 0):
        raise ValueError(f"eps must be finite and > 0, got {eps}")
    # Private float64 copies: an int array would round x + eps back to x, and
    # float32 would round it to the nearest 6e-8 * |x|.
    xs = [np.array(x, dtype=np.float64, copy=True) for x in inputs]
    grads = []
    for k, x in enumerate(xs):
        g = np.zeros_like(x)
        flat, gflat = x.reshape(-1), g.reshape(-1)  # views into x and g
        for i in range(flat.size):
            orig = flat[i]

            def along(t: float, i: int = i) -> float:
                flat[i] = t
                return _scalar(f(*xs))

            try:
                gflat[i] = central_diff(along, orig, eps)
            finally:
                flat[i] = orig  # restore before the next coordinate
        grads.append(g)
    return grads


def gradcheck(
    f: Callable[..., float],
    inputs: list[ArrayLike],
    analytic: list[ArrayLike],
    eps: float = 1e-6,
    rtol: float = 1e-5,
    atol: float = 1e-7,
) -> GradcheckReport:
    if len(analytic) != len(inputs):
        raise ValueError(f"{len(analytic)} analytic gradients for {len(inputs)} inputs")
    if rtol < 0 or atol < 0:
        raise ValueError(f"rtol and atol must be >= 0, got {rtol} and {atol}")
    ana = [np.asarray(a, dtype=np.float64) for a in analytic]
    for k, (a, x) in enumerate(zip(ana, inputs)):
        if a.shape != np.shape(x):
            raise ValueError(
                f"analytic[{k}] has shape {a.shape}, input {k} has {np.shape(x)}"
            )
    num = numerical_grad(f, inputs, eps)
    ok, max_abs, max_rel = True, 0.0, 0.0
    worst = (-1.0, 0, tuple(0 for _ in num[0].shape))
    for k, (a, n) in enumerate(zip(ana, num)):
        if a.size == 0:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            diff = np.abs(a - n)
            # A nan never passes: compare the negation, since nan <= x is False.
            ok = ok and not (~(diff <= atol + rtol * np.abs(n))).any()
            diff = np.where(np.isnan(diff), np.inf, diff)
            max_abs = max(max_abs, float(diff.max()))
            rel = diff / np.maximum(
                np.abs(n), atol if atol > 0 else np.finfo(float).tiny
            )
            max_rel = max(max_rel, float(rel.max()))
            # How far each element is past its own allowance (> 1 fails).
            ratio = np.where(diff == 0, 0.0, diff / (atol + rtol * np.abs(n)))
        i = int(np.argmax(ratio))  # first maximum in row-major order
        if ratio.reshape(-1)[i] > worst[0]:
            worst = (
                float(ratio.reshape(-1)[i]),
                k,
                tuple(int(j) for j in np.unravel_index(i, a.shape)),
            )
    return GradcheckReport(
        ok=bool(ok),
        max_abs_err=max_abs,
        max_rel_err=max_rel,
        worst_input=worst[1],
        worst_index=worst[2],
    )
