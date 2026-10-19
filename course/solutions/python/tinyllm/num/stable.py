"""Stable numerics: log-sum-exp, softmax, log-softmax, compensated sums (M09.2).

exp overflows float32 at 88.7 and float64 at 709.8, and underflows to 0 far
below. Subtracting the row maximum first makes the largest exponent e^0 = 1,
so nothing overflows, and log_softmax never takes the log of an underflowed
0. Long sums lose the low bits of every small addend; Kahan carries the lost
part in a compensation term, pairwise summation keeps partial sums of
similar size.

Contract: contracts/py/tinyllm/num/stable.pyi.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _float(x: ArrayLike) -> NDArray:
    """x as a float array: float dtypes kept, everything else float64."""
    a = np.asarray(x)
    if not np.issubdtype(a.dtype, np.floating):
        a = a.astype(np.float64)
    return a


def _safe_max(a: NDArray, axis: int) -> NDArray:
    """max along axis with keepdims, replaced by 0 where it is not finite, so
    a fully masked row (max = -inf) subtracts 0 instead of making -inf - -inf."""
    m = np.max(a, axis=axis, keepdims=True)
    return np.where(np.isfinite(m), m, np.zeros_like(m))


def logsumexp(x: ArrayLike, axis: int = -1, keepdims: bool = False) -> NDArray:
    a = _float(x)
    m = _safe_max(a, axis)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        s = np.sum(np.exp(a - m), axis=axis, keepdims=True)
        out = np.log(s) + m
    return out if keepdims else np.squeeze(out, axis=axis)


def softmax(x: ArrayLike, axis: int = -1) -> NDArray:
    a = _float(x)
    m = _safe_max(a, axis)
    with np.errstate(over="ignore", invalid="ignore"):
        e = np.exp(a - m)
        s = np.sum(e, axis=axis, keepdims=True)
        # A fully masked row has s = 0 and e = 0: divide by 1, keep the zeros.
        return e / np.where(s == 0, np.ones_like(s), s)


def log_softmax(x: ArrayLike, axis: int = -1) -> NDArray:
    a = _float(x)
    lse = logsumexp(a, axis=axis, keepdims=True)
    with np.errstate(invalid="ignore"):
        out = a - lse
    # -inf - (-inf) is nan; a fully masked row is log 0 = -inf everywhere.
    return np.where(np.isneginf(lse), np.full_like(out, -np.inf), out)


def kahan_sum(x: ArrayLike) -> float:
    a = _float(x).ravel()
    zero = a.dtype.type(0)
    s, c = zero, zero
    for v in a:
        y = v - c  # the addend, corrected by what the last step lost
        t = s + y  # the low bits of y are lost here...
        c = (t - s) - y  # ...and recovered here (algebraically 0)
        s = t
    return float(s)


def pairwise_sum(x: ArrayLike) -> float:
    a = _float(x).ravel()

    def rec(lo: int, hi: int):
        n = hi - lo
        if n == 1:
            return a[lo]
        h = lo + n // 2
        return rec(lo, h) + rec(h, hi)

    if a.size == 0:
        return 0.0
    return float(rec(0, a.size))
