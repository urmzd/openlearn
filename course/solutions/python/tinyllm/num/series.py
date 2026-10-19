"""Sequences, geometric series, and frequency ladders (M00.3).

A geometric sequence a, ar, ar^2, ... multiplies by the same ratio r at every
step. Two of them set positions in a transformer: RoPE's inverse
frequencies base^(-2i/d) (L7.3) and ALiBi's per-head slopes (L7.4). Their
partial sums are the EMA bias correction of Adam (M02.2, M10.3).

Contract: contracts/py/tinyllm/num/series.pyi.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


def _check_n(n: int) -> int:
    if isinstance(n, bool) or int(n) != n or n < 0:
        raise ValueError(f"n must be a non-negative integer, got {n!r}")
    return int(n)


def geometric(a: float, r: float, n: int) -> NDArray:
    n = _check_n(n)
    # a * r**j for each j, not a running product: a running product rounds at
    # every step and its error grows with j.
    return float(a) * np.power(float(r), np.arange(n, dtype=np.float64))


def _ones_sum(r: float, n: int) -> float:
    """1 + r + ... + r^(n-1), accurate for every r > 0, including r near 1."""
    if n == 0:
        return 0.0
    if r == 1.0:
        return float(n)
    try:
        if r > 0.0:
            # (1 - r^n) / (1 - r) = expm1(n ln r) / (r - 1). With r near 1 the
            # textbook numerator 1 - r**n subtracts two nearly equal numbers;
            # log1p and expm1 never form that difference.
            d = r - 1.0
            return math.expm1(n * math.log1p(d)) / d
        # r <= 0: 1 - r >= 1 and the closed form has no cancellation to fear,
        # except r^n near 1 (r near -1), which the contract does not promise.
        return (1.0 - r**n) / (1.0 - r)
    except OverflowError:
        # |r| > 1 and n large: the last term r^(n-1) dominates, and so does its sign.
        return math.inf if (r > 0.0 or (n - 1) % 2 == 0) else -math.inf


def geometric_sum(a: float, r: float, n: int) -> float:
    n = _check_n(n)
    return float(a) * _ones_sum(float(r), n)


def rope_inv_freq(d_rot: int, base: float) -> NDArray:
    if isinstance(d_rot, bool) or int(d_rot) != d_rot or d_rot < 2 or d_rot % 2:
        raise ValueError(f"d_rot must be a positive even integer, got {d_rot!r}")
    base = float(base)
    if not (base > 1.0 and math.isfinite(base)):
        raise ValueError(f"base must be finite and > 1, got {base}")
    d_rot = int(d_rot)
    # One frequency per pair i = 0 .. d_rot/2 - 1; the exponent steps by 2/d_rot.
    i = np.arange(d_rot // 2, dtype=np.float64)
    return base ** (-2.0 * i / d_rot)


def alibi_slopes(n_heads: int) -> NDArray:
    if isinstance(n_heads, bool) or int(n_heads) != n_heads or n_heads < 1:
        raise ValueError(f"n_heads must be a positive integer, got {n_heads!r}")
    n = int(n_heads)
    p = 1 << (n.bit_length() - 1)  # the largest power of two <= n (exact, no log2)
    start = 2.0 ** (-8.0 / p)
    slopes = geometric(start, start, p)
    if p == n:
        return slopes
    # The extra n - p heads take every other slope of the 2p-head ladder,
    # starting from the first, so they fall between the existing ones.
    start2 = 2.0 ** (-8.0 / (2 * p))
    extra = geometric(start2, start2, 2 * p)[0::2][: n - p]
    return np.concatenate([slopes, extra])
