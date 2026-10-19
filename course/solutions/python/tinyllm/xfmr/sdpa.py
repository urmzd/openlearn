"""Scaled dot-product attention, forward and backward (L5.1).

Every query scores every key with a dot product, the scores are scaled by
1 / sqrt(d) so their spread does not grow with the width, masked, turned into
weights by a softmax, and the output is the weighted average of the values.
The backward is written by hand: two matrix products for V, the softmax
vector-Jacobian product for the scores, two more products for Q and K. The
whole thing is one node of the autograd graph.

Contract: contracts/py/tinyllm/xfmr/sdpa.pyi. L5.3's multi-head attention
calls scaled_dot_product_attention once per layer; L9.3's C kernel is proven
against sdpa_forward.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tinyllm.autograd.tensor import Tensor, from_op
from tinyllm.num.stable import softmax


def _t(x: NDArray) -> NDArray:
    return np.swapaxes(x, -1, -2)


def _float(x: ArrayLike, name: str) -> NDArray:
    a = np.asarray(x)
    if a.dtype.kind != "f":
        a = a.astype(np.float64)
    if a.ndim < 2:
        raise ValueError(f"{name} must be [..., T, d], got shape {a.shape}")
    return a


def _check(q: NDArray, k: NDArray, v: NDArray) -> None:
    if not (q.shape[:-2] == k.shape[:-2] == v.shape[:-2]):
        raise ValueError(f"leading dims differ: q {q.shape}, k {k.shape}, v {v.shape}")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError(f"q and k widths differ: {q.shape[-1]} and {k.shape[-1]}")
    if k.shape[-2] != v.shape[-2]:
        raise ValueError(f"k and v lengths differ: {k.shape[-2]} and {v.shape[-2]}")


def _scale(scale: Optional[float], d: int) -> float:
    if scale is None:
        return 1.0 / math.sqrt(d)
    s = float(scale)
    if not (math.isfinite(s) and s > 0.0):
        raise ValueError(f"scale must be finite and > 0, got {scale}")
    return s


def sdpa_forward(
    q: ArrayLike,
    k: ArrayLike,
    v: ArrayLike,
    mask: Optional[ArrayLike] = None,
    scale: Optional[float] = None,
) -> tuple[NDArray, NDArray]:
    qa, ka, va = _float(q, "q"), _float(k, "k"), _float(v, "v")
    _check(qa, ka, va)
    s = (qa @ _t(ka)) * qa.dtype.type(_scale(scale, qa.shape[-1]))
    if mask is not None:
        m = np.asarray(mask)
        if m.dtype != np.bool_:
            raise ValueError(f"mask must be bool (True = may attend), got {m.dtype}")
        try:
            np.broadcast_shapes(m.shape, s.shape)
        except ValueError as e:
            raise ValueError(
                f"mask {m.shape} does not broadcast to scores {s.shape}"
            ) from e
        if np.broadcast_shapes(m.shape, s.shape) != s.shape:
            raise ValueError(f"mask {m.shape} would change the scores' shape {s.shape}")
        s = np.where(m, s, s.dtype.type(-np.inf))
    # Softmax over the KEYS (the last axis): each query's weights sum to 1;
    # a row with every key masked is all zeros (M09.2), never NaN.
    p = softmax(s, axis=-1)
    return p @ va, p


def sdpa_backward(
    q: ArrayLike,
    k: ArrayLike,
    v: ArrayLike,
    p: ArrayLike,
    dout: ArrayLike,
    scale: Optional[float] = None,
    dropout_mult: Optional[ArrayLike] = None,
) -> tuple[NDArray, NDArray, NDArray]:
    qa, ka, va = _float(q, "q"), _float(k, "k"), _float(v, "v")
    _check(qa, ka, va)
    pa, g = np.asarray(p), np.asarray(dout)
    want_p = qa.shape[:-1] + (ka.shape[-2],)
    if pa.shape != want_p or g.shape != qa.shape[:-1] + (va.shape[-1],):
        raise ValueError(
            f"p must be {want_p} and dout {qa.shape[:-1] + (va.shape[-1],)}, got {pa.shape}, {g.shape}"
        )
    s = _scale(scale, qa.shape[-1])
    pd = pa if dropout_mult is None else pa * dropout_mult
    dv = _t(pd) @ g
    dp = g @ _t(va)
    if dropout_mult is not None:
        dp = dp * dropout_mult
    # Softmax VJP: dS = P * (dP - sum_j P_j dP_j), row by row.
    ds = pa * (dp - np.sum(dp * pa, axis=-1, keepdims=True))
    dq = (ds @ ka) * s
    dk = (_t(ds) @ qa) * s
    return dq.astype(qa.dtype), dk.astype(ka.dtype), dv.astype(va.dtype)


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Optional[ArrayLike] = None,
    dropout_p: float = 0.0,
    scale: Optional[float] = None,
    rng: Any = None,
) -> tuple[Tensor, Tensor]:
    qt, kt, vt = (
        x if isinstance(x, Tensor) else Tensor(x, dtype=np.asarray(x).dtype)
        for x in (q, k, v)
    )
    dropout_p = float(dropout_p)
    if not 0.0 <= dropout_p <= 1.0:
        raise ValueError(f"dropout_p must be in [0, 1], got {dropout_p}")
    if dropout_p > 0.0 and rng is None:
        raise ValueError("dropout_p > 0 needs an rng (a PCG32)")
    out, p = sdpa_forward(qt.data, kt.data, vt.data, mask, scale)
    mult = None
    if dropout_p > 0.0:
        u = np.asarray(rng.uniforms(p.size)).reshape(p.shape)
        keep = u >= dropout_p
        mult = (
            np.zeros_like(p)
            if dropout_p == 1.0
            else (keep / (1.0 - dropout_p)).astype(p.dtype)
        )
        out = (p * mult) @ vt.data

    def vjp(g: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        return sdpa_backward(qt.data, kt.data, vt.data, p, g, scale, mult)

    res = from_op(
        out.astype(qt.dtype), [qt, kt, vt], vjp, op="scaled_dot_product_attention"
    )
    return res, Tensor(p, dtype=p.dtype)
