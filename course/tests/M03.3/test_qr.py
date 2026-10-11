"""Course tests for M03.3: qr_householder and orthogonal_init
(tinyllm/linalg/qr.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/M03.3), and the chapter section it comes from.

The chapter's worked example (section 3) is A = [[3, 1], [4, 2]]: one
reflection gives Q = [[0.6, -0.8], [0.8, 0.6]] and R = [[5, 2.2], [0, 0.4]].
numpy's LAPACK QR is the oracle for random matrices, compared after the same
sign rule (diagonal of R non-negative).
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32
from tinyllm.linalg.qr import orthogonal_init, qr_householder


def rng(seq: int) -> PCG32:
    return PCG32(int(os.environ.get("OL_SEED", "0")), seq=seq)


def numpy_qr_sign_fixed(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Q, R = np.linalg.qr(A)
    d = np.where(np.diag(R) < 0, -1.0, 1.0)
    return Q * d, R * d[:, None]


def check_qr(A: np.ndarray, Q: np.ndarray, R: np.ndarray) -> None:
    m, n = A.shape
    k = min(m, n)
    assert Q.shape == (m, k) and R.shape == (k, n)
    assert_close(Q.T @ Q, np.eye(k), rtol=0.0, atol=1e-13 * max(m, 1))
    assert (np.tril(R, -1) == 0.0).all(), "R must be exactly upper triangular"
    assert (np.diag(R) >= 0.0).all(), "the sign rule: diag(R) >= 0"
    scale = max(1.0, float(np.abs(A).max()))
    assert_close(Q @ R, A, rtol=0.0, atol=1e-13 * max(m, n) * scale)


def test_hand_example():
    # WHY: the chapter's worked example. x = [3, 4], ||x|| = 5, alpha = -5
    #      (opposite sign to x_0 = 3), v = x - alpha e1 = [8, 4]. Reflecting
    #      gives R = [[-5, -2.2], [0, 0.4]] and Q = H = [[-0.6, -0.8],
    #      [-0.8, 0.6]]; the sign rule flips row 0 of R and column 0 of Q.
    # KIND: unit, smoke
    # CATCHES: s05, m01
    # CHAPTER: M03.3 section 3
    Q, R = qr_householder(np.array([[3.0, 1.0], [4.0, 2.0]]))
    assert_close(Q, [[0.6, -0.8], [0.8, 0.6]], rtol=0.0, atol=1e-15)
    assert_close(R, [[5.0, 2.2], [0.0, 0.4]], rtol=0.0, atol=1e-15)


@pytest.mark.parametrize("shape", [(1, 1), (5, 5), (8, 3), (3, 8), (12, 12), (40, 7)])
def test_random_matrices_match_numpy(shape):
    # WHY: tall, wide, and square matrices against LAPACK. With the sign
    #      rule the reduced QR of a full-rank matrix is unique, so the factors
    #      must agree entry by entry, not just multiply back to A.
    # KIND: differential
    # CATCHES: s01, s05, m01, m02
    # CHAPTER: M03.3 section 2.4
    A = rng(81).uniform_array(shape, -1.0, 1.0)
    Q, R = qr_householder(A)
    check_qr(A, Q, R)
    Qn, Rn = numpy_qr_sign_fixed(A)
    assert_close(Q, Qn, rtol=0.0, atol=1e-12)
    assert_close(R, Rn, rtol=0.0, atol=1e-12)


def test_cancellation_sign_choice():
    # WHY: when column j is already close to +||x|| e1 (here [1, 1e-9]), the
    #      wrong reflector v = x - ||x|| e1 computes v_0 = 1 - 1.0000000000000000005
    #      = 0 exactly in floating point: the reflection is garbage and A = QR
    #      fails at the 1e-9 level. Choosing alpha = -sign(x_0) ||x|| adds
    #      instead of subtracting.
    # KIND: boundary
    # CATCHES: s02
    # CHAPTER: M03.3 section 2.3
    A = np.array([[1.0, 2.0], [1e-9, 3.0], [2e-9, -1.0]])
    Q, R = qr_householder(A)
    check_qr(A, Q, R)


def test_orthogonality_survives_ill_conditioning():
    # WHY: columns that are almost parallel (condition number about 1e10):
    #      classical Gram-Schmidt loses orthogonality like cond(A) * eps, but
    #      reflections are orthogonal by construction, so Q^T Q = I to 1e-13
    #      anyway. This is why the course uses Householder.
    # KIND: property
    # CATCHES: s03
    # CHAPTER: M03.3 section 2.2
    g = rng(82)
    base = g.uniform_array((30, 1), -1.0, 1.0)
    A = np.hstack(
        [base + 1e-10 * g.uniform_array((30, 1), -1.0, 1.0) for _ in range(4)]
    )
    Q, R = qr_householder(A)
    assert_close(Q.T @ Q, np.eye(4), rtol=0.0, atol=1e-13)


def test_rank_deficient_and_zero_columns():
    # WHY: a zero column has nothing to reflect (norm 0); dividing by it
    #      gives NaN everywhere. Q must stay orthonormal and A = QR must hold.
    # KIND: boundary
    # CATCHES: s04
    # CHAPTER: M03.3 section 4
    A = np.array([[0.0, 1.0, 2.0], [0.0, 2.0, 4.0], [0.0, 3.0, 6.0], [0.0, 0.0, 1.0]])
    Q, R = qr_householder(A)
    assert np.isfinite(Q).all() and np.isfinite(R).all()
    check_qr(A, Q, R)
    Q, R = qr_householder(np.zeros((3, 2)))
    check_qr(np.zeros((3, 2)), Q, R)


def test_input_not_modified_and_2d_required():
    # WHY: orthogonal_init and the QR algorithm in M03.4 pass arrays they
    #      keep using; an in-place factorization corrupts them.
    # KIND: boundary
    # CATCHES: s06
    # CHAPTER: M03.3 section 4
    A = rng(83).uniform_array((5, 4), -1.0, 1.0)
    before = A.copy()
    qr_householder(A)
    assert (A == before).all()
    with pytest.raises(ValueError):
        qr_householder(np.ones(3))


@pytest.mark.parametrize("shape", [(4, 4), (6, 3), (3, 6), (1, 5), (64, 64)])
def test_orthogonal_init_is_orthogonal_and_scaled(shape):
    # WHY: L0.4 and the recurrent layers (L3.1, L3.2) initialize weights so
    #      that W keeps vector lengths: W W^T = gain^2 I (wide) or
    #      W^T W = gain^2 I (tall). Every singular value equals gain, so the
    #      signal neither explodes nor vanishes through many steps.
    # KIND: property
    # CATCHES: s08
    # CHAPTER: M03.3 section 2.5
    rows, cols = shape
    W = orthogonal_init(shape, 1.7, rng(84))
    assert W.shape == shape
    G = W @ W.T if rows <= cols else W.T @ W
    assert_close(G, 1.7**2 * np.eye(min(shape)), rtol=0.0, atol=1e-12)


def test_orthogonal_init_draws_normals_in_c_order():
    # WHY: the initializer is deterministic given the generator: the Gaussian
    #      matrix G is normal(rng, rows * cols) reshaped in C order, and the
    #      result is the sign-fixed Q of G (or of G^T for a wide shape).
    #      Same seed, same weights in every run (P11).
    # KIND: unit
    # CATCHES: s07
    # CHAPTER: M03.3 section 2.5
    from tinyllm.prob.rv import normal

    G = normal(rng(85), 15).reshape(5, 3)
    Qn, _ = numpy_qr_sign_fixed(G)
    assert_close(orthogonal_init((5, 3), 1.0, rng(85)), Qn, rtol=0.0, atol=1e-12)
    Qw, _ = numpy_qr_sign_fixed(normal(rng(85), 15).reshape(3, 5).T)
    assert_close(orthogonal_init((3, 5), 1.0, rng(85)), Qw.T, rtol=0.0, atol=1e-12)


def test_orthogonal_init_is_uniform_on_signs():
    # WHY: without the sign rule, LAPACK-style QR returns Q whose diagonal
    #      signs follow R's, which biases the distribution of Q (Mezzadri
    #      2007). With it, the first entry of Q is positive about half the
    #      time over many seeds; a biased version is positive almost always.
    # KIND: statistical
    # CATCHES: s05
    # CHAPTER: M03.3 section 2.5
    pos = 0
    trials = 400
    for s in range(trials):
        W = orthogonal_init((3, 3), 1.0, PCG32(s, seq=86))
        pos += W[0, 0] > 0
    assert abs(pos / trials - 0.5) < 4 * (0.25 / trials) ** 0.5


def test_orthogonal_init_rejects_bad_shapes():
    # WHY: a zero-sized layer is a config bug; fail at init, not at the
    #      first forward pass.
    # KIND: boundary
    # CATCHES: m03
    # CHAPTER: M03.3 section 4
    for bad in ((0, 3), (3, 0), (-1, 2)):
        with pytest.raises(ValueError):
            orthogonal_init(bad, 1.0, rng(87))
