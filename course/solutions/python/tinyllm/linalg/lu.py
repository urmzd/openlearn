"""tinyllm.linalg.lu (M03.2): Gaussian elimination with partial pivoting as
a factorization P A = L U.

Elimination subtracts multiples of a pivot row from the rows below it until
the matrix is upper triangular (U). The multipliers, kept below the diagonal
of a unit lower triangular L, record how to replay the same elimination on a
right-hand side, so one O(n^3) factorization serves every later O(n^2) solve.
Partial pivoting swaps the row with the largest |entry| into the pivot
position first (P records the swaps), which keeps every multiplier at most 1
in magnitude and the computation stable.

    P, L, U = lu(A)
    x = lu_solve(P, L, U, b)       # A @ x == b

Contract: contracts/py/tinyllm/linalg/lu.pyi.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def lu(A: ArrayLike) -> tuple[NDArray, NDArray, NDArray]:
    U = np.array(A, dtype=np.float64)  # a copy: the caller's A is never touched
    if U.ndim != 2 or U.shape[0] != U.shape[1]:
        raise ValueError(f"lu: A must be a square matrix, got shape {U.shape}")
    n = U.shape[0]
    L = np.eye(n)
    perm = np.arange(n)  # row i of P A is row perm[i] of A
    for k in range(n - 1):
        # Pivot: the largest |entry| in column k at or below the diagonal.
        # np.argmax returns the first maximum, so ties go to the lowest row.
        p = k + int(np.argmax(np.abs(U[k:, k])))
        if U[p, k] == 0.0:
            continue  # the column is already zero below k: nothing to eliminate
        if p != k:
            U[[k, p], k:] = U[[p, k], k:]
            # The multipliers already stored in L belong to the rows, so they
            # move with them; L's diagonal and upper part stay put.
            L[[k, p], :k] = L[[p, k], :k]
            perm[[k, p]] = perm[[p, k]]
        m = U[k + 1 :, k] / U[k, k]  # |m| <= 1 thanks to the pivot choice
        L[k + 1 :, k] = m
        U[k + 1 :, k:] -= np.outer(m, U[k, k:])
        U[k + 1 :, k] = 0.0  # exactly zero by construction, not just rounded
    P = np.zeros((n, n))
    P[np.arange(n), perm] = 1.0
    return P, L, U


def lu_solve(P: ArrayLike, L: ArrayLike, U: ArrayLike, b: ArrayLike) -> NDArray:
    P = np.asarray(P, dtype=np.float64)
    L = np.asarray(L, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = U.shape[0]
    if P.shape != (n, n) or L.shape != (n, n) or U.shape != (n, n):
        raise ValueError(f"lu_solve: factors must be {n} x {n}")
    if b.ndim not in (1, 2) or b.shape[0] != n:
        raise ValueError(f"lu_solve: b must be [{n}] or [{n}, k], got shape {b.shape}")
    d = np.diag(U)
    if np.any(d == 0.0):
        raise ValueError("lu_solve: U has a zero pivot, A is singular")
    # Forward substitution, L y = P b. L has ones on its diagonal, so there
    # is nothing to divide by.
    y = P @ b
    for i in range(n):
        y[i] = y[i] - L[i, :i] @ y[:i]
    # Back substitution, U x = y, from the last row up.
    x = y
    for i in range(n - 1, -1, -1):
        x[i] = (x[i] - U[i, i + 1 :] @ x[i + 1 :]) / d[i]
    return x
