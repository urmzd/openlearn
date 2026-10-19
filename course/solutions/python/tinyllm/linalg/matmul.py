"""Naive float32 matrix multiplication reference (M03.1).

This Python implementation defines the semantics used by the optional C
kernel exercise. C parity vectors cross the boundary as JSON files.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def matmul_f32(
    a: ArrayLike,
    b: ArrayLike,
    *,
    trans_b: bool = False,
    alpha: float = 1.0,
    beta: float = 0.0,
    c: ArrayLike | None = None,
) -> NDArray[np.float32]:
    """Compute ``C = alpha * A @ op(B) + beta * C`` in float32.

    ``A`` and ``B`` are two-dimensional float32 matrices. With ``trans_b``,
    ``B`` is stored as ``[N, K]``. Inputs may be row-strided NumPy views.
    """
    aa = np.asarray(a)
    bb = np.asarray(b)
    if aa.dtype != np.float32 or bb.dtype != np.float32:
        raise TypeError("A and B must have dtype float32")
    if aa.ndim != 2 or bb.ndim != 2:
        raise ValueError("A and B must be two-dimensional")
    m, k = aa.shape
    if trans_b:
        n, bk = bb.shape
    else:
        bk, n = bb.shape
    if k != bk:
        raise ValueError(f"inner dimensions differ: A has {k}, op(B) has {bk}")
    if c is None:
        cc = np.empty((m, n), dtype=np.float32)
    else:
        cc = np.asarray(c)
        if cc.dtype != np.float32 or cc.shape != (m, n) or not cc.flags.writeable:
            raise ValueError(f"C must be a writable float32 array of shape {(m, n)}")
    al, be = np.float32(alpha), np.float32(beta)
    for i in range(m):
        for j in range(n):
            acc = np.float32(0.0)
            for q in range(k):
                rhs = bb[j, q] if trans_b else bb[q, j]
                acc = np.float32(acc + np.float32(aa[i, q] * rhs))
            prior = np.float32(0.0) if be == 0 else np.float32(be * cc[i, j])
            cc[i, j] = np.float32(al * acc + prior)
    return cc
