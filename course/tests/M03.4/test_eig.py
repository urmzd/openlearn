"""Course tests for M03.4: power_iteration, inverse_iteration, and
spectral_radius (tinyllm/linalg/eig.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/M03.4), and the chapter section it comes from.

The chapter's worked example (section 3) is A = [[2, 1], [1, 2]], with
eigenvalues 3 (eigenvector [1, 1]) and 1 (eigenvector [1, -1]).
numpy's LAPACK eigensolvers are the oracle for random matrices.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32
from tinyllm.linalg.eig import inverse_iteration, power_iteration, spectral_radius

HAND = np.array([[2.0, 1.0], [1.0, 2.0]])


def rng(seq: int) -> PCG32:
    return PCG32(int(os.environ.get("OL_SEED", "0")), seq=seq)


def with_spectrum(blocks: list, seq: int) -> np.ndarray:
    """S D S^-1 for a block diagonal D (scalars and 2x2 blocks) and a random,
    well-conditioned S: a non-symmetric matrix with a known spectrum."""
    n = sum(1 if np.isscalar(b) else 2 for b in blocks)
    D = np.zeros((n, n))
    i = 0
    for b in blocks:
        if np.isscalar(b):
            D[i, i] = b
            i += 1
        else:
            D[i : i + 2, i : i + 2] = b
            i += 2
    S = rng(seq).uniform_array((n, n), -1.0, 1.0) + 3.0 * np.eye(n)
    return S @ D @ np.linalg.inv(S)


def rotation(r: float, theta: float) -> np.ndarray:
    return r * np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
    )


def test_hand_example_power_iteration():
    # WHY: the chapter's worked example. Any start with a component along
    #      [1, 1] converges to it, the error shrinking by 1/3 per step, and the
    #      Rayleigh quotient gives 3.
    # KIND: unit, smoke
    # CATCHES: m01
    # CHAPTER: M03.4 section 3
    lam, v = power_iteration(lambda x: HAND @ x, 2, 60, rng(91))
    assert_close(lam, 3.0, rtol=1e-12, atol=0.0)
    assert_close(abs(v), [2**-0.5, 2**-0.5], rtol=1e-10, atol=0.0)
    assert_close(np.linalg.norm(v), 1.0, rtol=1e-14, atol=0.0)


def test_negative_dominant_eigenvalue_keeps_its_sign():
    # WHY: ||A v|| is |lambda|, so returning the norm reports +5 for
    #      eigenvalue -5. The Rayleigh quotient v . A v keeps the sign, which
    #      M10.5 needs to tell a maximum from a saddle.
    # KIND: boundary
    # CATCHES: s01
    # CHAPTER: M03.4 section 2.2
    A = np.diag([-5.0, 1.0, 2.0])
    lam, _ = power_iteration(lambda x: A @ x, 3, 200, rng(92))
    assert_close(lam, -5.0, rtol=1e-10, atol=0.0)


def test_random_start_finds_the_dominant_direction():
    # WHY: a fixed start such as e_1 is orthogonal to the dominant
    #      eigenvector of diag(1, 3) and stays at eigenvalue 1 forever.
    #      A random start has a non-zero component along every eigenvector
    #      with probability 1.
    # KIND: boundary
    # CATCHES: s02
    # CHAPTER: M03.4 section 5, Pitfalls
    A = np.diag([1.0, 3.0])
    lam, _ = power_iteration(lambda x: A @ x, 2, 100, rng(93))
    assert_close(lam, 3.0, rtol=1e-10, atol=0.0)


def test_start_vector_draws_from_rng_in_order():
    # WHY: v_0 is 2 u - 1 for n uniforms in order, so with zero iterations
    #      the result is that vector normalized: the run is reproducible from
    #      the seed (P11), and the caller's generator advances by exactly n.
    # KIND: unit
    # CATCHES: s02
    # CHAPTER: M03.4 section 4
    g = rng(94)
    want = np.array([2.0 * g.uniform() - 1.0 for _ in range(4)])
    want /= np.linalg.norm(want)
    _, v = power_iteration(lambda x: x, 4, 0, rng(94))
    assert_close(v, want, rtol=0.0, atol=1e-15)


def test_renormalizes_every_step():
    # WHY: with |lambda| = 10 and 400 steps, A^400 v has norm about 1e400:
    #      inf in float64, then inf / inf = NaN. Normalizing once at the end
    #      is too late.
    # KIND: boundary
    # CATCHES: s03
    # CHAPTER: M03.4 section 2.2
    A = np.diag([10.0, 1.0, -2.0])
    lam, v = power_iteration(lambda x: A @ x, 3, 400, rng(95))
    assert np.isfinite(v).all()
    assert_close(lam, 10.0, rtol=1e-12, atol=0.0)


def test_symmetric_matrices_match_numpy():
    # WHY: against LAPACK (np.linalg.eigh) on 20 random symmetric matrices
    #      whose dominant eigenvalue is +2 or -2 and the next one 1.5 in
    #      modulus: a ratio of 0.75, so 150 steps are needed for full
    #      precision. Checks the eigenvalue with its sign and the eigenvector
    #      up to sign.
    # KIND: differential
    # CATCHES: s01
    # CHAPTER: M03.4 section 2.2
    g = rng(96)
    for t in range(20):
        n = 3 + g.below(9)
        Qm, _ = np.linalg.qr(g.uniform_array((n, n), -1.0, 1.0))
        eigs = np.linspace(-1.0, 1.0, n)
        eigs[-1] = 2.0 if t % 2 == 0 else -2.0
        eigs[-2] = -1.5 if t % 2 == 0 else 1.5
        A = Qm @ np.diag(eigs) @ Qm.T
        lam, v = power_iteration(lambda x: A @ x, n, 150, g)
        w, V = np.linalg.eigh(A)
        top = int(np.argmax(np.abs(w)))
        assert_close(lam, w[top], rtol=1e-10, atol=0.0)
        assert_close(abs(v @ V[:, top]), 1.0, rtol=1e-8, atol=0.0)


def test_zero_map_returns_zero():
    # WHY: matvec(v) = 0 means v is in the null space: eigenvalue 0. Dividing
    #      by ||w|| = 0 would turn v into NaN.
    # KIND: boundary
    # CATCHES: s04
    # CHAPTER: M03.4 section 4
    lam, v = power_iteration(lambda x: np.zeros_like(x), 3, 10, rng(97))
    assert lam == 0.0 and np.isfinite(v).all()
    with pytest.raises(ValueError):
        power_iteration(lambda x: x, 0, 10, rng(97))
    with pytest.raises(ValueError):
        power_iteration(lambda x: x, 3, -1, rng(97))


def test_inverse_iteration_finds_the_eigenvalue_nearest_the_shift():
    # WHY: (A - sI)^-1 has eigenvalues 1 / (lambda_i - s); the largest is the
    #      one with lambda_i nearest s. With s = 2.9 on spectrum {1, 3, 7} they
    #      are 10, -0.53, and 0.24: the error shrinks 19-fold per step, so 20
    #      steps give full precision. A shift that is exactly an eigenvalue
    #      makes A - sI singular and must raise.
    # KIND: unit
    # CATCHES: s05
    # CHAPTER: M03.4 section 2.3
    A = with_spectrum([1.0, 3.0, 7.0], seq=98)
    lam, v = inverse_iteration(A, 2.9, 20, rng(99))
    assert_close(lam, 3.0, rtol=1e-9, atol=0.0)
    assert_close(A @ v, 3.0 * v, rtol=0.0, atol=1e-9)
    lam, _ = inverse_iteration(A, 0.0, 60, rng(99))
    assert_close(lam, 1.0, rtol=1e-9, atol=0.0)
    with pytest.raises(ValueError):  # shift exactly an eigenvalue: A - sI is singular
        inverse_iteration(np.diag([1.0, 2.0, 5.0]), 2.0, 10, rng(101))


def test_inverse_iteration_factors_once():
    # WHY: the point of LU (M03.2) is one O(n^3) factorization and an O(n^2)
    #      solve per step. Refactoring every step costs iters times more.
    # KIND: unit
    # CATCHES: s06
    # CHAPTER: M03.4 section 2.3
    import tinyllm.linalg.eig as eig

    calls = {"lu": 0}
    real_lu = eig.lu

    def counting_lu(M):
        calls["lu"] += 1
        return real_lu(M)

    eig.lu = counting_lu
    try:
        inverse_iteration(np.diag([1.0, 2.0, 5.0]), 4.0, 25, rng(100))
    finally:
        eig.lu = real_lu
    assert calls["lu"] == 1


def test_spectral_radius_of_a_rotation():
    # WHY: a rotation by 0.7 rad scaled by 1.5 has eigenvalues
    #      1.5 e^(+-0.7 i): complex, equal modulus. Power iteration never
    #      converges on it (the vector keeps turning), but the QR algorithm
    #      leaves a 2x2 block whose eigenvalues have modulus 1.5.
    # KIND: unit, smoke
    # CATCHES: s07, s08, m02
    # CHAPTER: M03.4 section 3
    assert_close(spectral_radius(rotation(1.5, 0.7)), 1.5, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize(
    "blocks, rho",
    [
        ([2.0, -1.0, 0.5], 2.0),
        ([-2.0, 1.0, 0.5, 0.25], 2.0),
        (["rot", 0.5, -0.3], 1.5),
        ([0.9, "rot_small", 0.1], 0.9),
        ([0.5, 0.4, 0.3, 0.2, 0.1], 0.5),
    ],
)
def test_spectral_radius_matches_numpy(blocks, rho):
    # WHY: non-symmetric matrices with real, negative, and complex dominant
    #      eigenvalues against np.linalg.eigvals; the diagnostic in L3.1 asks
    #      only whether rho is above or below 1, but asks it of exactly such
    #      matrices.
    # KIND: differential
    # CATCHES: s07, s08, m02, m03
    # CHAPTER: M03.4 section 2.4
    blocks = [
        rotation(1.5, 0.7)
        if b == "rot"
        else rotation(0.6, 2.0)
        if b == "rot_small"
        else b
        for b in blocks
    ]
    W = with_spectrum(blocks, seq=102)
    assert_close(max(abs(np.linalg.eigvals(W))), rho, rtol=1e-9, atol=0.0)
    assert_close(spectral_radius(W, iters=200), rho, rtol=1e-8, atol=0.0)


def test_spectral_radius_edge_cases():
    # WHY: a 1x1 matrix is its own eigenvalue (absolute value); a nilpotent
    #      matrix has only eigenvalue 0 however large its entries; non-square
    #      input has no eigenvalues.
    # KIND: boundary
    # CATCHES: m03
    # CHAPTER: M03.4 section 4
    assert spectral_radius(np.array([[-4.0]])) == 4.0
    assert spectral_radius(np.zeros((0, 0))) == 0.0
    assert spectral_radius(np.array([[0.0, 100.0], [0.0, 0.0]])) < 1e-6
    with pytest.raises(ValueError):
        spectral_radius(np.ones((2, 3)))


def test_input_not_modified():
    # WHY: the training monitor (L0.5) measures rho of live weight matrices;
    #      it must not change them.
    # KIND: boundary
    # CATCHES: s09
    # CHAPTER: M03.4 section 4
    W = with_spectrum([2.0, 1.0, 0.5], seq=103)
    before = W.copy()
    spectral_radius(W)
    assert (W == before).all()
