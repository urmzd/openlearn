"""Course tests for M03.5: svd, low_rank, and lstsq (tinyllm/linalg/svd.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/M03.5), and the chapter section it comes from.

The chapter's worked example (section 3) is A = [[3, 0], [4, 5]]: one
Jacobi rotation by 45 degrees gives S = [3 sqrt(5), sqrt(5)],
U = [[1, 3], [3, -1]] / sqrt(10), Vt = [[1, 1], [1, -1]] / sqrt(2); its best
rank-1 approximation is [[1.5, 1.5], [4.5, 4.5]], off by exactly sqrt(5).
The line through (0, 1), (1, 2), (2, 2) by least squares is 7/6 + t/2.
numpy's LAPACK SVD and least squares are the oracle for random matrices,
compared after the same sign rule.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32
from tinyllm.linalg.svd import low_rank, lstsq, svd

HAND = np.array([[3.0, 0.0], [4.0, 5.0]])


def rng(seq: int) -> PCG32:
    return PCG32(int(os.environ.get("OL_SEED", "0")), seq=seq)


def numpy_svd_sign_fixed(A: np.ndarray):
    """np.linalg.svd with the contract's sign rule: the largest |entry| of
    each row of Vt is positive (the first one on ties)."""
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    big = np.argmax(np.abs(Vt), axis=1)
    d = np.where(Vt[np.arange(Vt.shape[0]), big] < 0, -1.0, 1.0)
    return U * d, S, Vt * d[:, None]


def check_svd(A: np.ndarray, U: np.ndarray, S: np.ndarray, Vt: np.ndarray) -> None:
    m, n = A.shape
    k = min(m, n)
    assert U.shape == (m, k) and S.shape == (k,) and Vt.shape == (k, n)
    assert (S >= 0.0).all() and (np.diff(S) <= 0.0).all(), "S must be descending"
    assert_close(U.T @ U, np.eye(k), rtol=0.0, atol=1e-13 * max(m, 1))
    assert_close(Vt @ Vt.T, np.eye(k), rtol=0.0, atol=1e-13 * max(n, 1))
    scale = max(1.0, float(np.abs(A).max()))
    assert_close((U * S) @ Vt, A, rtol=0.0, atol=1e-13 * max(m, n) * scale)


def test_hand_example_svd():
    # WHY: the chapter's worked example. alpha = beta = 25 and gamma = 20
    #      give zeta = 0, t = 1, a 45 degree rotation; the next sweep finds
    #      the columns orthogonal and stops. Sorting puts sqrt(45) first, and
    #      the sign rule makes the first entry of each Vt row positive.
    # KIND: unit, smoke
    # CATCHES: s02, m01
    # CHAPTER: M03.5 section 3
    U, S, Vt = svd(HAND)
    r10, r2 = math.sqrt(10.0), math.sqrt(2.0)
    assert_close(S, [3.0 * math.sqrt(5.0), math.sqrt(5.0)], rtol=0.0, atol=1e-14)
    assert_close(U, np.array([[1.0, 3.0], [3.0, -1.0]]) / r10, rtol=0.0, atol=1e-14)
    assert_close(Vt, np.array([[1.0, 1.0], [1.0, -1.0]]) / r2, rtol=0.0, atol=1e-14)


def test_hand_example_rank_one():
    # WHY: Eckart-Young on the worked example. The best rank-1 matrix is
    #      sigma_1 u_1 v_1^T = [[1.5, 1.5], [4.5, 4.5]], the error matrix
    #      [[1.5, -1.5], [-0.5, 0.5]] has spectral and Frobenius norm
    #      sqrt(5) = sigma_2, and the balanced factors carry 45^(1/4) each.
    # KIND: unit, smoke
    # CATCHES: s06
    # CHAPTER: M03.5 section 3
    B, C = low_rank(HAND, 1)
    assert B.shape == (2, 1) and C.shape == (1, 2)
    assert_close(B @ C, [[1.5, 1.5], [4.5, 4.5]], rtol=0.0, atol=1e-14)
    q = 45.0**0.25
    assert_close(
        B[:, 0], q * np.array([1.0, 3.0]) / math.sqrt(10.0), rtol=0.0, atol=1e-14
    )
    assert_close(C[0], q * np.array([1.0, 1.0]) / math.sqrt(2.0), rtol=0.0, atol=1e-14)
    assert_close(np.linalg.norm(HAND - B @ C, 2), math.sqrt(5.0), rtol=0.0, atol=1e-14)


def test_hand_example_lstsq():
    # WHY: the chapter's line fit. A = [[1, 0], [1, 1], [1, 2]], b = [1, 2, 2]:
    #      x = [7/6, 1/2], residual [-1/6, 1/3, -1/6], orthogonal to both
    #      columns of A.
    # KIND: unit, smoke
    # CATCHES: s08
    # CHAPTER: M03.5 section 3
    A = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    x = lstsq(A, [1.0, 2.0, 2.0])
    assert x.shape == (2,)
    assert_close(x, [7.0 / 6.0, 0.5], rtol=0.0, atol=1e-15)


@pytest.mark.parametrize(
    "shape", [(1, 1), (2, 2), (6, 6), (9, 4), (4, 9), (30, 12), (1, 7)]
)
def test_random_matrices_match_numpy(shape):
    # WHY: tall, wide, and square matrices against LAPACK. Random matrices
    #      have distinct singular values, so with the sign rule the reduced
    #      SVD is unique and must agree entry by entry, not only multiply
    #      back to A.
    # KIND: differential
    # CATCHES: s03, s04, s05, s11, m01
    # CHAPTER: M03.5 section 2.3
    A = rng(91).uniform_array(shape, -1.0, 1.0)
    U, S, Vt = svd(A)
    check_svd(A, U, S, Vt)
    Un, Sn, Vtn = numpy_svd_sign_fixed(A)
    assert_close(S, Sn, rtol=0.0, atol=1e-13)
    assert_close(U, Un, rtol=0.0, atol=1e-10)
    assert_close(Vt, Vtn, rtol=0.0, atol=1e-10)


def test_scale_invariance():
    # WHY: the rotation test compares |gamma| with ||w_i|| ||w_j||, not with
    #      a fixed number. Scaled by 1e-12, every dot product is about
    #      1e-24; an absolute threshold would call the columns orthogonal
    #      and skip every rotation. svd(c A) must be c times svd(A).
    # KIND: property
    # CATCHES: s12
    # CHAPTER: M03.5 section 2.3
    A = rng(92).uniform_array((7, 5), -1.0, 1.0)
    U, S, Vt = svd(A)
    for c in (1e-12, 1e12):
        Uc, Sc, Vtc = svd(c * A)
        assert_close(Sc, c * S, rtol=1e-12, atol=0.0)
        assert_close(Uc, U, rtol=0.0, atol=1e-10)
        assert_close(Vtc, Vt, rtol=0.0, atol=1e-10)


def test_rank_deficient_completes_u():
    # WHY: a rank-2 matrix has a zero singular value, and w_3 / sigma_3 is
    #      0 / 0. U must still have orthonormal columns: the missing column
    #      is completed from the standard basis. The all-zero matrix is the
    #      extreme case: S = 0 and U, Vt still orthonormal.
    # KIND: boundary
    # CATCHES: s07
    # CHAPTER: M03.5 section 2.4
    g = rng(93)
    A = g.uniform_array((5, 2), -1.0, 1.0) @ g.uniform_array((2, 3), -1.0, 1.0)
    U, S, Vt = svd(A)
    assert np.isfinite(U).all() and np.isfinite(Vt).all()
    check_svd(A, U, S, Vt)
    assert S[2] <= 1e-14 * S[0]
    Z = np.zeros((4, 3))
    U, S, Vt = svd(Z)
    check_svd(Z, U, S, Vt)
    assert (S == 0.0).all()


@pytest.mark.parametrize("r", [1, 2, 4, 6])
def test_eckart_young_error(r):
    # WHY: Eckart-Young: no rank-r matrix is closer to A than the truncated
    #      SVD, and its error is exactly the first dropped singular value in
    #      the spectral norm (sqrt of the sum of their squares in Frobenius).
    #      r = min(m, n) reproduces A. L6.6 and L7.6 rely on this bound.
    # KIND: property
    # CATCHES: s02, m02
    # CHAPTER: M03.5 section 2.5
    A = rng(94).uniform_array((9, 6), -1.0, 1.0)
    S = np.linalg.svd(A, compute_uv=False)
    B, C = low_rank(A, r)
    assert B.shape == (9, r) and C.shape == (r, 6)
    E = A - B @ C
    spectral = S[r] if r < 6 else 0.0
    assert_close(np.linalg.norm(E, 2), spectral, rtol=1e-12, atol=1e-13)
    assert_close(
        np.linalg.norm(E, "fro"),
        math.sqrt(float((S[r:] ** 2).sum())),
        rtol=1e-12,
        atol=1e-13,
    )


def test_low_rank_factors_are_balanced():
    # WHY: the contract splits sqrt(sigma) into each factor, so
    #      B^T B = C C^T = diag(S[:r]). PiSSA (L6.6) initializes both LoRA
    #      factors this way so they start at the same scale and train at the
    #      same rate; U Sigma with V^T alone gives the same product but
    #      not this property.
    # KIND: property
    # CATCHES: s06
    # CHAPTER: M03.5 section 2.5
    A = rng(95).uniform_array((8, 5), -2.0, 2.0)
    S = np.linalg.svd(A, compute_uv=False)
    B, C = low_rank(A, 3)
    assert_close(B.T @ B, np.diag(S[:3]), rtol=0.0, atol=1e-12)
    assert_close(C @ C.T, np.diag(S[:3]), rtol=0.0, atol=1e-12)


def test_low_rank_rejects_bad_rank():
    # WHY: r = 0 is an empty adapter and r > min(m, n) asks for singular
    #      values that do not exist; both are config bugs to fail early.
    # KIND: boundary
    # CATCHES: m03
    # CHAPTER: M03.5 section 4
    A = rng(96).uniform_array((4, 3), -1.0, 1.0)
    for bad in (0, -1, 4):
        with pytest.raises(ValueError):
            low_rank(A, bad)


def test_lstsq_matches_numpy():
    # WHY: an overdetermined system with one and with three right-hand
    #      sides; the least squares solution of a full-rank A is unique.
    # KIND: differential
    # CATCHES: s08, m04
    # CHAPTER: M03.5 section 2.6
    g = rng(97)
    A = g.uniform_array((20, 6), -1.0, 1.0)
    b = g.uniform_array((20,), -1.0, 1.0)
    B = g.uniform_array((20, 3), -1.0, 1.0)
    assert_close(
        lstsq(A, b), np.linalg.lstsq(A, b, rcond=None)[0], rtol=0.0, atol=1e-12
    )
    X = lstsq(A, B)
    assert X.shape == (6, 3)
    assert_close(X, np.linalg.lstsq(A, B, rcond=None)[0], rtol=0.0, atol=1e-12)


def test_lstsq_residual_is_orthogonal():
    # WHY: the defining property of least squares: the residual b - A x is
    #      orthogonal to every column of A (A^T r = 0), so A x is the
    #      projection of b onto the column space.
    # KIND: property
    # CATCHES: s08
    # CHAPTER: M03.5 section 2.6
    g = rng(98)
    A = g.uniform_array((15, 4), -1.0, 1.0)
    b = g.uniform_array((15,), -1.0, 1.0)
    r = b - A @ lstsq(A, b)
    assert_close(A.T @ r, np.zeros(4), rtol=0.0, atol=1e-13)


def test_lstsq_ill_conditioned_beats_normal_equations():
    # WHY: a degree-9 polynomial fit at 40 points of [0, 1] has cond(A)
    #      about 3.5e6. QR's error grows like cond(A) eps (5e-10 here); the
    #      normal equations A^T A x = A^T b square the condition number and
    #      get only about 1e-3. C1's scaling-law fit is such a fit.
    # KIND: boundary
    # CATCHES: s01
    # CHAPTER: M03.5 section 2.6
    t = np.linspace(0.0, 1.0, 40)
    A = np.vander(t, 10, increasing=True)
    x_true = np.arange(1.0, 11.0)
    x = lstsq(A, A @ x_true)
    assert float(np.abs(x - x_true).max()) < 1e-6


def test_lstsq_rejects_rank_deficient_and_wide():
    # WHY: with a repeated column the minimizer is not unique and R has a
    #      zero on its diagonal; back substitution would divide by it and
    #      return inf. A wide system is underdetermined. Both raise.
    # KIND: boundary
    # CATCHES: s09, m05
    # CHAPTER: M03.5 section 4
    A = np.array([[1.0, 2.0, 2.0], [3.0, 1.0, 1.0], [0.0, 4.0, 4.0], [2.0, 2.0, 2.0]])
    with pytest.raises(ValueError):
        lstsq(A, np.ones(4))
    with pytest.raises(ValueError):
        lstsq(np.eye(2, 3), np.ones(2))
    with pytest.raises(ValueError):
        lstsq(np.ones((3, 2)) + np.eye(3, 2), np.ones(4))


def test_inputs_not_modified_and_validated():
    # WHY: low_rank is called on live weight matrices (L6.6 initializes an
    #      adapter from W and keeps W); an in-place Jacobi sweep would
    #      overwrite them with U Sigma. Non-2-D or non-finite input raises.
    # KIND: boundary
    # CATCHES: s10
    # CHAPTER: M03.5 section 4
    A = rng(99).uniform_array((6, 4), -1.0, 1.0)
    before = A.copy()
    svd(A)
    low_rank(A, 2)
    lstsq(A, np.ones(6))
    assert (A == before).all()
    with pytest.raises(ValueError):
        svd(np.ones(3))
    with pytest.raises(ValueError):
        svd(np.array([[1.0, math.nan], [0.0, 1.0]]))
