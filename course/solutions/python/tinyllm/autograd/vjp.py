"""Closed-form vector-Jacobian products (M08.3).

Each rule comes from the differential of the op and the trace trick:
for a scalar loss L and Y = f(X), dL = tr(G^T dY) with G = dL/dY; write dY
in terms of dX, move dX to the right inside the trace, and read off
dL/dX as the matrix multiplying it. The derivations are in the chapter.

Contract: contracts/py/tinyllm/autograd/vjp.pyi.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tinyllm.num.stable import softmax


def _f64(x: ArrayLike) -> NDArray:
    """x as a float array (float dtypes kept, anything else float64)."""
    a = np.asarray(x)
    return a if np.issubdtype(a.dtype, np.floating) else a.astype(np.float64)


def unbroadcast(g: ArrayLike, shape: tuple[int, ...]) -> NDArray:
    out = _f64(g)
    shape = tuple(shape)
    extra = out.ndim - len(shape)
    if extra < 0:
        raise ValueError(f"gradient of shape {out.shape} cannot come from shape {shape}")
    if extra:
        out = out.sum(axis=tuple(range(extra)))
    for i, n in enumerate(shape):
        if n == 1 and out.shape[i] != 1:
            out = out.sum(axis=i, keepdims=True)
    if out.shape != shape:
        raise ValueError(f"gradient of shape {np.shape(g)} cannot come from shape {shape}")
    return out


def matmul_vjp(g: ArrayLike, A: ArrayLike, B: ArrayLike) -> tuple[NDArray, NDArray]:
    g, A, B = _f64(g), _f64(A), _f64(B)
    if A.ndim < 2 or B.ndim < 2:
        raise ValueError(f"matmul_vjp needs A and B with at least 2 axes, got {A.shape} and {B.shape}")
    dA = g @ np.swapaxes(B, -1, -2)
    dB = np.swapaxes(A, -1, -2) @ g
    return unbroadcast(dA, A.shape), unbroadcast(dB, B.shape)


def softmax_vjp(g: ArrayLike, y: ArrayLike, axis: int = -1) -> NDArray:
    g, y = _f64(g), _f64(y)
    return y * (g - np.sum(g * y, axis=axis, keepdims=True))


def log_softmax_vjp(g: ArrayLike, y: ArrayLike, axis: int = -1) -> NDArray:
    g, y = _f64(g), _f64(y)
    return g - np.exp(y) * np.sum(g, axis=axis, keepdims=True)


def _rows(rstd: ArrayLike, like: NDArray) -> NDArray:
    """rstd reshaped to like.shape[:-1] + (1,), so it broadcasts over the last axis."""
    return _f64(rstd).reshape(like.shape[:-1] + (1,))


def layernorm_vjp(
    g: ArrayLike, xhat: ArrayLike, rstd: ArrayLike, gamma: ArrayLike
) -> tuple[NDArray, NDArray, NDArray]:
    g, xhat, gamma = _f64(g), _f64(xhat), _f64(gamma)
    r = _rows(rstd, xhat)
    d = g * gamma
    dx = r * (d - d.mean(axis=-1, keepdims=True) - xhat * (d * xhat).mean(axis=-1, keepdims=True))
    lead = tuple(range(g.ndim - 1))
    return dx, (g * xhat).sum(axis=lead), g.sum(axis=lead)


def rmsnorm_vjp(
    g: ArrayLike, x: ArrayLike, rstd: ArrayLike, w: ArrayLike
) -> tuple[NDArray, NDArray]:
    g, x, w = _f64(g), _f64(x), _f64(w)
    r = _rows(rstd, x)
    d = g * w
    dx = r * (d - x * (r * r) * (d * x).mean(axis=-1, keepdims=True))
    lead = tuple(range(g.ndim - 1))
    return dx, (g * x * r).sum(axis=lead)


def cross_entropy_vjp(logits: ArrayLike, targets: ArrayLike, ignore_index: int = -100) -> NDArray:
    z = _f64(logits)
    t = np.asarray(targets)
    if t.shape != z.shape[:-1]:
        raise ValueError(f"targets have shape {t.shape}, logits {z.shape}: need logits.shape[:-1]")
    v = z.shape[-1]
    valid = t != ignore_index
    if ((t[valid] < 0) | (t[valid] >= v)).any():
        raise ValueError(f"targets must lie in [0, {v}) or equal ignore_index={ignore_index}")
    n = int(valid.sum())
    if n == 0:
        return np.zeros_like(z)
    grad = softmax(z, axis=-1)
    rows = np.nonzero(valid)
    grad[rows + (t[valid],)] -= 1.0
    grad[~valid] = 0.0
    return grad / n
