"""tinyllm.linalg.eig (M03.4): eigenvalues by iteration.

An eigenvector v of A satisfies A v = lambda v: A only stretches it. Applying
A again and again to almost any start vector amplifies the component along the
eigenvalue of largest modulus fastest, so the direction converges to that
eigenvector (power iteration). Inverting a shifted A turns the eigenvalue
nearest the shift into the largest one (inverse iteration). The spectral
radius rho(A) = max |lambda| decides whether A^t x grows or dies out, which is
the exploding and vanishing gradient story of recurrent networks (L3.1).

Contract: contracts/py/tinyllm/linalg/eig.pyi.
"""

from __future__ import annotations

import cmath
import math
from typing import Any, Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tinyllm.linalg.lu import lu, lu_solve
from tinyllm.linalg.qr import qr_householder


def _start(n: int, rng: Any) -> NDArray:
    """A random unit vector: almost surely not orthogonal to any eigenvector."""
    v = np.array([2.0 * rng.uniform() - 1.0 for _ in range(n)], dtype=np.float64)
    norm = math.sqrt(float(v @ v))
    if norm == 0.0:
        v = np.ones(n)  # probability zero with 53-bit uniforms; stay defined
        norm = math.sqrt(n)
    return v / norm


def power_iteration(
    matvec: Callable[[NDArray], NDArray], n: int, iters: int, rng: Any
) -> tuple[float, NDArray]:
    if n < 1:
        raise ValueError(f"power_iteration: n must be >= 1, got {n}")
    if iters < 0:
        raise ValueError(f"power_iteration: iters must be >= 0, got {iters}")
    v = _start(n, rng)
    for _ in range(iters):
        w = np.asarray(matvec(v), dtype=np.float64)
        norm = math.sqrt(float(w @ w))
        if norm == 0.0:
            return 0.0, v  # v is in the null space: eigenvalue 0
        v = w / norm  # renormalize every step, or |lambda|^t overflows
    # The Rayleigh quotient v . A v (v is a unit vector) keeps the sign that
    # ||A v|| would lose, and its error is the square of the angle error for
    # a symmetric A.
    return float(v @ np.asarray(matvec(v), dtype=np.float64)), v


def inverse_iteration(
    A: ArrayLike, shift: float, iters: int, rng: Any
) -> tuple[float, NDArray]:
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"inverse_iteration: A must be square, got shape {A.shape}")
    n = A.shape[0]
    P, L, U = lu(A - shift * np.eye(n))  # factor once ...
    if np.any(np.diag(U) == 0.0):
        raise ValueError("inverse_iteration: A - shift I is singular (shift is an eigenvalue)")
    _, v = power_iteration(lambda x: lu_solve(P, L, U, x), n, iters, rng)  # ... solve every step
    return float(v @ (A @ v)), v


def _block_moduli(T: NDArray) -> list[float]:
    """Moduli of the eigenvalues of the diagonal blocks of a nearly block
    upper triangular T: a 1x1 block where the subdiagonal entry is negligible,
    else a 2x2 block solved in closed form."""
    n = T.shape[0]
    scale = float(np.abs(T).max()) or 1.0
    out: list[float] = []
    i = 0
    while i < n:
        if i == n - 1 or abs(T[i + 1, i]) <= 1e-12 * scale:
            out.append(abs(float(T[i, i])))
            i += 1
            continue
        a, b, c, d = T[i, i], T[i, i + 1], T[i + 1, i], T[i + 1, i + 1]
        # Eigenvalues of [[a, b], [c, d]]: roots of t^2 - (a + d) t + (ad - bc).
        tr, det = a + d, a * d - b * c
        disc = cmath.sqrt(tr * tr / 4.0 - det)
        out += [abs(tr / 2.0 + disc), abs(tr / 2.0 - disc)]
        i += 2
    return out


def spectral_radius(W: ArrayLike, iters: int = 100) -> float:
    T = np.array(W, dtype=np.float64)
    if T.ndim != 2 or T.shape[0] != T.shape[1]:
        raise ValueError(f"spectral_radius: W must be square, got shape {T.shape}")
    if T.shape[0] == 0:
        return 0.0
    # A_{t+1} = R_t Q_t = Q_t^T A_t Q_t: every step is a similarity
    # transform, so the eigenvalues never change while the entries below the
    # diagonal blocks decay like |lambda_{i+1} / lambda_i|^t.
    for _ in range(iters):
        Q, R = qr_householder(T)
        T = R @ Q
    return max(_block_moduli(T))
