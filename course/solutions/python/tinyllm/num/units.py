"""Exponents, logarithms, change of base, and units of information (M00.1).

A probability p carries -log_b(p) units of surprise; the base b names the
unit: b = e gives nats, b = 2 gives bits. Every loss in the course is
computed in nats (numpy's log is the natural log), and every number the
course compares across tokenizers is bits per byte.

Contract: contracts/py/tinyllm/num/units.pyi.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

LN2 = math.log(2.0)  # nats in one bit


def nats_to_bits(x: float) -> float:
    # One bit is ln 2 nats, so a quantity in nats is x / ln 2 bits.
    return float(x) / LN2


def bits_to_nats(x: float) -> float:
    return float(x) * LN2


def log_base(x: ArrayLike, b: float) -> NDArray:
    b = float(b)
    if not (b > 0.0 and b != 1.0 and math.isfinite(b)):
        raise ValueError(f"the base must be positive, finite, and not 1, got {b}")
    a = np.asarray(x, dtype=np.float64)
    if np.isnan(a).any() or (a < 0).any():
        raise ValueError(
            "log_base needs x >= 0 everywhere (the log of a negative number is not real)"
        )
    # Change of base: log_b x = ln x / ln b. log(0) = -inf is the right answer
    # (an impossible event is infinitely surprising), so silence numpy's warning.
    with np.errstate(divide="ignore"):
        return np.log(a) / math.log(b)


def bits_per_byte(nll_nats_sum: float, n_bytes: int) -> float:
    if isinstance(n_bytes, bool) or int(n_bytes) != n_bytes or n_bytes < 1:
        raise ValueError(f"n_bytes must be a positive integer, got {n_bytes!r}")
    nll = float(nll_nats_sum)
    if math.isnan(nll) or nll < 0:
        raise ValueError(f"a summed negative log-likelihood is >= 0, got {nll}")
    # Total nats -> total bits, then shared by every byte of the text.
    return nll / (int(n_bytes) * LN2)
