"""tinyllm.linalg.svd (M03.5): the singular value decomposition, the best
low-rank approximation, and least squares.

Every real matrix maps the unit sphere onto an ellipsoid: A = U diag(S) V^T,
with V the directions that come in, U the directions that go out, and S the
stretch along each. One-sided Jacobi finds it by rotating pairs of columns of
A until all columns are mutually orthogonal: then A V = W has orthogonal
columns, their lengths are the singular values, and their directions are U.
Each rotation is orthogonal, so V stays orthogonal to working precision, and
the small singular values come out with high relative accuracy.

Truncating the decomposition after r terms gives the closest rank-r matrix
(Eckart-Young), which LoRA (L6.6), MLA (L7.6), and the PPMI-SVD embeddings
(L2.3) use. Least squares goes through QR (M03.3), not the normal equations.

Contract: contracts/py/tinyllm/linalg/svd.pyi.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tinyllm.linalg.qr import qr_householder

EPS = 2.0**-52
MAX_SWEEPS = 64


def _jacobi_tall(T: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """One-sided Jacobi on a tall [rows, cols] matrix (rows >= cols): U, S, V
    with T = U diag(S) V^T, S descending, before the sign rule."""
    W = np.array(T, dtype=np.float64)  # a copy; its columns become U * S
    rows, cols = W.shape
    V = np.eye(cols)
    tol = rows * EPS
    for _ in range(MAX_SWEEPS):
        rotated = False
        for i in range(cols - 1):
            for j in range(i + 1, cols):
                wi, wj = W[:, i], W[:, j]
                alpha = float(wi @ wi)
                beta = float(wj @ wj)
                gamma = float(wi @ wj)
                if alpha == 0.0 or beta == 0.0:
                    continue  # a zero column is orthogonal to everything
                if abs(gamma) <= tol * math.sqrt(alpha * beta):
                    continue
                rotated = True
                # The rotation angle theta with tan(2 theta) = 2 gamma /
                # (beta - alpha); t = tan(theta) is the smaller root, so
                # |theta| <= pi/4 and the iteration converges.
                zeta = (beta - alpha) / (2.0 * gamma)
                sign = 1.0 if zeta >= 0.0 else -1.0
                t = sign / (abs(zeta) + math.sqrt(1.0 + zeta * zeta))
                c = 1.0 / math.sqrt(1.0 + t * t)
                s = c * t
                W[:, [i, j]] = np.column_stack((c * wi - s * wj, s * wi + c * wj))
                vi, vj = V[:, i].copy(), V[:, j].copy()
                V[:, i] = c * vi - s * vj
                V[:, j] = s * vi + c * vj
        if not rotated:
            break
    S = np.sqrt(np.einsum("ij,ij->j", W, W))
    order = np.argsort(-S, kind="stable")
    S, W, V = S[order], W[:, order], V[:, order]
    U = np.zeros((rows, cols))
    floor = rows * EPS * (S[0] if cols else 0.0)
    for j in range(cols):
        if S[j] > floor and S[j] > 0.0:
            U[:, j] = W[:, j] / S[j]
        else:
            U[:, j] = _complete(U[:, :j], rows)
    return U, S, V


def _complete(Uprev: NDArray, rows: int) -> NDArray:
    """A unit vector orthogonal to the columns of Uprev: the first standard
    basis vector with enough of itself left after projecting them out."""
    for i in range(rows):
        e = np.zeros(rows)
        e[i] = 1.0
        for _ in range(2):  # twice is enough (Kahan, Parlett)
            e -= Uprev @ (Uprev.T @ e)
        norm = math.sqrt(float(e @ e))
        if norm > 0.5:
            return e / norm
    raise AssertionError("unreachable: fewer than rows orthonormal columns")


def svd(A: ArrayLike) -> tuple[NDArray, NDArray, NDArray]:
    M = np.asarray(A, dtype=np.float64)  # a view; _jacobi_tall copies
    if M.ndim != 2:
        raise ValueError(f"svd: A must be 2-D, got shape {M.shape}")
    if not np.isfinite(M).all():
        raise ValueError("svd: A has a NaN or infinite entry")
    m, n = M.shape
    k = min(m, n)
    if k == 0:
        return np.zeros((m, 0)), np.zeros(0), np.zeros((0, n))
    if m >= n:
        U, S, V = _jacobi_tall(M)
    else:
        # A^T = U' S V'^T, so A = V' S U'^T: the roles swap.
        Ut, S, Vt_ = _jacobi_tall(M.T)
        U, V = Vt_, Ut
    # Sign rule: the largest |entry| of each right singular vector is positive.
    big = np.argmax(np.abs(V), axis=0)
    d = np.where(V[big, np.arange(k)] < 0.0, -1.0, 1.0)
    return U * d, S, (V * d).T


def low_rank(A: ArrayLike, r: int) -> tuple[NDArray, NDArray]:
    U, S, Vt = svd(A)
    k = S.shape[0]
    if not 1 <= r <= k:
        raise ValueError(f"low_rank: need 1 <= r <= min(m, n) = {k}, got r = {r}")
    root = np.sqrt(S[:r])
    return U[:, :r] * root, root[:, None] * Vt[:r]


def lstsq(A: ArrayLike, b: ArrayLike) -> NDArray:
    M = np.array(A, dtype=np.float64)
    y = np.array(b, dtype=np.float64)
    if M.ndim != 2:
        raise ValueError(f"lstsq: A must be 2-D, got shape {M.shape}")
    m, n = M.shape
    if m < n:
        raise ValueError(f"lstsq: A is {m} x {n}; least squares needs m >= n")
    if y.ndim not in (1, 2) or y.shape[0] != m:
        raise ValueError(f"lstsq: b must have {m} rows, got shape {y.shape}")
    Q, R = qr_householder(M)
    diag = np.abs(np.diag(R))
    if n == 0:
        return np.zeros((0,) + y.shape[1:])
    if diag.min() <= max(m, n) * EPS * diag.max():
        raise ValueError("lstsq: A is rank deficient (a diagonal entry of R is ~0)")
    z = Q.T @ y  # the coordinates of b's projection onto range(A)
    x = np.zeros((n,) + y.shape[1:])
    for i in range(n - 1, -1, -1):  # back substitution, last row first
        x[i] = (z[i] - R[i, i + 1 :] @ x[i + 1 :]) / R[i, i]
    return x
