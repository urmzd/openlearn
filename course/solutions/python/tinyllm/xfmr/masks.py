"""Attention masks: causal, padding, sliding window, additive (L5.2).

Attention lets every query read every key. A language model must not read
the future (causal), a batch must not read its own padding (padding), a long
context may read only a recent window (sliding window). Each rule is a
boolean matrix, True = may attend; the rules combine by AND, and the score
sum uses them as 0 or -inf added before the softmax.

Contract: contracts/py/tinyllm/xfmr/masks.pyi. L5.3 and L5.5 build their
attention masks here; L6.1, L7.7, and L8.2 (q_offset during decode) follow.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray


def _count(v: int, name: str, lo: int) -> int:
    if isinstance(v, (bool, np.bool_)) or int(v) != v or v < lo:
        raise ValueError(f"{name} must be an integer >= {lo}, got {v!r}")
    return int(v)


def _bool(m: ArrayLike, name: str = "mask") -> NDArray:
    a = np.asarray(m)
    if a.dtype != np.bool_:
        raise ValueError(
            f"{name} must be a bool array (True = may attend), got {a.dtype}"
        )
    return a


def causal_mask(Tq: int, Tk: int | None = None, q_offset: int = 0) -> NDArray:
    Tq = _count(Tq, "Tq", 1)
    q_offset = _count(q_offset, "q_offset", 0)
    Tk = q_offset + Tq if Tk is None else _count(Tk, "Tk", 1)
    # Query i sits at absolute position q_offset + i; key j at position j.
    pos_q = q_offset + np.arange(Tq)[:, None]
    pos_k = np.arange(Tk)[None, :]
    return pos_k <= pos_q


def padding_mask(lengths: ArrayLike, T: int) -> NDArray:
    T = _count(T, "T", 1)
    n = np.asarray(lengths)
    if n.ndim != 1 or n.dtype.kind not in "iu":
        raise ValueError(
            f"lengths must be a 1-D integer array, got {n.dtype} {n.shape}"
        )
    if np.any(n < 0) or np.any(n > T):
        raise ValueError(f"lengths must be in [0, {T}], got {n.tolist()}")
    return np.arange(T)[None, :] < n[:, None]


def sliding_window_mask(Tq: int, Tk: int, window: int, q_offset: int = 0) -> NDArray:
    window = _count(window, "window", 1)
    Tk = _count(Tk, "Tk", 1)
    causal = causal_mask(Tq, Tk, q_offset)
    pos_q = q_offset + np.arange(Tq)[:, None]
    pos_k = np.arange(Tk)[None, :]
    # At most `window` keys, the query's own position included.
    return causal & (pos_q - pos_k < window)


def combine(*masks: ArrayLike) -> NDArray:
    if not masks:
        raise ValueError("combine needs at least one mask")
    arrs = [_bool(m) for m in masks]
    try:
        shape = np.broadcast_shapes(*(a.shape for a in arrs))
    except ValueError as e:
        raise ValueError(f"masks do not broadcast: {[a.shape for a in arrs]}") from e
    out = np.ones(shape, dtype=bool)
    for a in arrs:
        out &= a
    return out


def to_additive(mask: ArrayLike, dtype: DTypeLike = np.float32) -> NDArray:
    m = _bool(mask)
    dt = np.dtype(dtype)
    if dt.kind != "f":
        raise ValueError(f"dtype must be a float type, got {dt}")
    # -inf, not a large negative number: exp(-inf) is exactly 0, and a row
    # with every key blocked stays all -inf, which the stable softmax maps
    # to zeros instead of a uniform average over forbidden keys.
    return np.where(m, dt.type(0.0), dt.type(-np.inf)).astype(dt)
