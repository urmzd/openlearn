"""tinyllm.linalg.qr (M03.3): QR by Householder reflections, and the
orthogonal initializer that L0.4 and the recurrent layers use.

A Householder reflection H = I - 2 v v^T (with ||v|| = 1) mirrors space
across the hyperplane orthogonal to v. Choosing v so that H maps the part of
column j below the diagonal onto a multiple of e_1 zeroes that column; doing
this for every column turns A into R, and the product of the reflections is
Q^T. Reflections are orthogonal by construction, so Q stays orthogonal to
working precision however ill-conditioned A is (Gram-Schmidt does not).

Contract: contracts/py/tinyllm/linalg/qr.pyi.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tinyllm.prob.rv import normal


def qr_householder(A: ArrayLike) -> tuple[NDArray, NDArray]:
    R = np.array(A, dtype=np.float64)  # a copy, reduced to R in place
    if R.ndim != 2:
        raise ValueError(f"qr_householder: A must be 2-D, got shape {R.shape}")
    m, n = R.shape
    k = min(m, n)
    Q = np.eye(m)
    for j in range(min(m - 1, n)):
        x = R[j:, j]
        norm_x = math.sqrt(float(x @ x))
        if norm_x == 0.0:
            continue  # nothing below the diagonal to zero
        # Reflect x onto alpha e_1 with alpha = -sign(x_0) ||x||: then
        # v_0 = x_0 - alpha = x_0 + sign(x_0) ||x|| adds two numbers of the
        # same sign and never cancels.
        alpha = -math.copysign(norm_x, x[0]) if x[0] != 0.0 else -norm_x
        v = x.copy()
        v[0] -= alpha
        v /= math.sqrt(float(v @ v))
        # Apply H = I - 2 v v^T to the trailing block from the left, and
        # accumulate Q = H_0 H_1 ... from the right.
        R[j:, j:] -= 2.0 * np.outer(v, v @ R[j:, j:])
        Q[:, j:] -= 2.0 * np.outer(Q[:, j:] @ v, v)
        R[j, j] = alpha
        R[j + 1 :, j] = 0.0  # exact zeros, not rounding noise
    Q, R = Q[:, :k], np.triu(R[:k, :])
    # Sign rule: flip each row of R (and column of Q) whose diagonal is
    # negative. Q D D R = Q R for D = diag(+-1), and the result is unique.
    d = np.where(np.diag(R) < 0.0, -1.0, 1.0)
    return Q * d, R * d[:, None]


def orthogonal_init(shape: tuple[int, int], gain: float, rng: Any) -> NDArray:
    rows, cols = int(shape[0]), int(shape[1])
    if rows < 1 or cols < 1:
        raise ValueError(f"orthogonal_init: shape must be positive, got {shape}")
    G = normal(rng, rows * cols).reshape(rows, cols)
    # QR needs at least as many rows as columns to give orthonormal columns;
    # for a wide matrix factor its transpose and transpose back.
    Q, _ = qr_householder(G.T if rows < cols else G)
    W = Q.T if rows < cols else Q
    return gain * W
