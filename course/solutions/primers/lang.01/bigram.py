"""lang.01 vectorized bigram count (a primer exercise, not part of the system).

A byte string is a sequence of integers 0..255. Counting which byte follows
which is the whole of training the L0.0 byte bigram; here you do the count
without a Python loop, then turn counts into per-row probabilities.
"""

from __future__ import annotations

import numpy as np


def bigram_counts(data: bytes) -> np.ndarray:
    """C[a, b] = the number of positions i with data[i] == a and data[i + 1] == b.

    Returns an int64 array of shape (256, 256). No Python loop: build one
    integer per adjacent pair and count them in a single numpy call.
    """
    x = np.frombuffer(data, dtype=np.uint8).astype(np.int64)
    pairs = x[:-1] * 256 + x[1:]
    return np.bincount(pairs, minlength=256 * 256).astype(np.int64, copy=False).reshape(256, 256)


def row_normalize(counts: np.ndarray) -> np.ndarray:
    """P[a, b] = counts[a, b] / sum over b of counts[a, b], as float64.

    Each nonzero row sums to 1. A row of zeros stays a row of zeros (no NaN).
    """
    c = np.asarray(counts, dtype=np.float64)
    totals = c.sum(axis=1, keepdims=True)
    return np.divide(c, totals, out=np.zeros_like(c), where=totals > 0)
