"""tinyllm.linalg.inner (M03.6): inner products, projections, cosine
similarity, and exact top-k retrieval.

The inner product a . b = ||a|| ||b|| cos(theta) carries both length and
angle. Dividing out the lengths leaves the angle alone: cosine similarity,
the score every embedding search uses (L2.3 analogies, L6.7's word
similarity task, and ag.07's vector retrieval, which ports this to Go).
Projection onto a direction keeps the part of x along v and drops the rest;
the residual is orthogonal to v.

Contract: contracts/py/tinyllm/linalg/inner.pyi.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _norm(x: NDArray, axis: int) -> NDArray:
    """Euclidean norm along axis, kept as a size-1 axis for broadcasting."""
    return np.sqrt(np.sum(x * x, axis=axis, keepdims=True))


def _check_lengths(a: NDArray, b: NDArray, axis: int, who: str) -> None:
    if a.ndim == 0 or b.ndim == 0 or a.shape[axis] != b.shape[axis]:
        raise ValueError(
            f"{who}: vector lengths differ along axis {axis}: {a.shape} vs {b.shape}"
        )


def normalize(x: ArrayLike, axis: int = -1, eps: float = 1e-12) -> NDArray:
    v = np.asarray(x, dtype=np.float64)
    return v / np.maximum(_norm(v, axis), eps)


def cosine_sim(
    a: ArrayLike, b: ArrayLike, axis: int = -1, eps: float = 1e-8
) -> NDArray:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    _check_lengths(x, y, axis, "cosine_sim")
    # Clamp each norm on its own: a tiny but nonzero pair of vectors still
    # has a well-defined angle, and a zero vector gives 0, never NaN.
    nx = np.maximum(_norm(x, axis), eps)
    ny = np.maximum(_norm(y, axis), eps)
    return np.squeeze(np.sum(x * y, axis=axis, keepdims=True) / (nx * ny), axis=axis)


def project(x: ArrayLike, v: ArrayLike, axis: int = -1) -> NDArray:
    xv = np.asarray(x, dtype=np.float64)
    vv = np.asarray(v, dtype=np.float64)
    _check_lengths(xv, vv, axis, "project")
    vdotv = np.sum(vv * vv, axis=axis, keepdims=True)
    if (vdotv == 0.0).any():
        raise ValueError("project: cannot project onto the zero vector")
    # Divide by v . v, not by ||v||: the coefficient multiplies v itself.
    return np.sum(xv * vv, axis=axis, keepdims=True) / vdotv * vv


def topk_cosine(query: ArrayLike, matrix: ArrayLike, k: int) -> tuple[NDArray, NDArray]:
    q = np.asarray(query, dtype=np.float64)
    M = np.asarray(matrix, dtype=np.float64)
    if M.ndim != 2 or q.ndim not in (1, 2):
        raise ValueError(
            f"topk_cosine: need query [d] or [q, d] and matrix [n, d], got {q.shape}, {M.shape}"
        )
    n, d = M.shape
    if q.shape[-1] != d:
        raise ValueError(f"topk_cosine: query has length {q.shape[-1]}, rows have {d}")
    if not 1 <= k <= n:
        raise ValueError(f"topk_cosine: need 1 <= k <= {n}, got k = {k}")
    Q = q[None, :] if q.ndim == 1 else q
    # One query at a time through cosine_sim itself: the [n, d] temporary
    # stays small, and identical rows get bitwise identical scores (a BLAS
    # matrix product may round equal rows differently in different blocks,
    # which would break ties at random).
    scores = np.stack([cosine_sim(Q[i], M) for i in range(Q.shape[0])])  # [q, n]
    # A stable sort on -score keeps equal scores in row order, so ties go to
    # the lowest index; argpartition alone would not.
    idx = np.argsort(-scores, axis=1, kind="stable")[:, :k].astype(np.int64)
    top = np.take_along_axis(scores, idx, axis=1)
    return (idx[0], top[0]) if q.ndim == 1 else (idx, top)
