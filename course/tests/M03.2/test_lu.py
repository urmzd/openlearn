"""Course tests for M03.2: lu and lu_solve (tinyllm/linalg/lu.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/M03.2), and the chapter section it comes from.

The chapter's worked example (section 3) is
    A = [[2, 1, 1], [4, -6, 0], [-2, 7, 2]],  b = [5, -2, 9],  x = [1, 1, 2].
numpy's own LAPACK solver is the oracle for random systems (np.linalg.solve).
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32
from tinyllm.linalg.lu import lu, lu_solve

HAND = [[2.0, 1.0, 1.0], [4.0, -6.0, 0.0], [-2.0, 7.0, 2.0]]
HAND_B = [5.0, -2.0, 9.0]


def hand_a() -> np.ndarray:
    """A fresh copy every time: a factorization that writes into its input
    must not be able to corrupt the next test."""
    return np.array(HAND)


def rng(seq: int) -> PCG32:
    return PCG32(int(os.environ.get("OL_SEED", "0")), seq=seq)


def check_factors(A: np.ndarray, P: np.ndarray, L: np.ndarray, U: np.ndarray) -> None:
    n = A.shape[0]
    assert P.shape == L.shape == U.shape == (n, n)
    # P is a permutation matrix: 0/1 entries, one 1 per row and column.
    assert set(np.unique(P)) <= {0.0, 1.0}
    assert (P.sum(axis=0) == 1).all() and (P.sum(axis=1) == 1).all()
    assert (np.diag(L) == 1.0).all(), "L must have a unit diagonal"
    assert (np.triu(L, 1) == 0.0).all(), "L must be lower triangular"
    assert (np.tril(U, -1) == 0.0).all(), "U must be upper triangular"
    assert np.abs(L).max() <= 1.0 + 1e-15, "partial pivoting keeps |L| <= 1"
    scale = max(1.0, float(np.abs(A).max()))
    assert_close(P @ A, L @ U, rtol=0.0, atol=1e-12 * n * scale)


def test_hand_example_factors():
    # WHY: the chapter's worked example. Column 0 pivots on |4| (row 1), so
    #      P swaps rows 0 and 1; the multipliers are 1/2 and -1/2; column 1
    #      then pivots on 4 (no swap), multiplier 1. Every number is exact.
    # KIND: unit, smoke
    # CATCHES: s01, s06, m01
    # CHAPTER: M03.2 section 3
    P, L, U = lu(hand_a())
    assert P.tolist() == [[0, 1, 0], [1, 0, 0], [0, 0, 1]]
    assert L.tolist() == [[1, 0, 0], [0.5, 1, 0], [-0.5, 1, 1]]
    assert U.tolist() == [[4, -6, 0], [0, 4, 1], [0, 0, 1]]


def test_hand_example_solve():
    # WHY: forward substitution L y = P b gives y = [-2, 6, 2]; back
    #      substitution U x = y gives x = [1, 1, 2]. Exact in floating point.
    # KIND: unit, smoke
    # CATCHES: s05, m02
    # CHAPTER: M03.2 section 3
    P, L, U = lu(hand_a())
    assert lu_solve(P, L, U, np.array(HAND_B)).tolist() == [1.0, 1.0, 2.0]


def test_pa_equals_lu_on_random_matrices():
    # WHY: the defining identity on 60 random matrices of sizes 1 to 12,
    #      plus the structure every caller relies on: P a permutation, L unit
    #      lower triangular with |L| <= 1, U upper triangular.
    # KIND: property
    # CATCHES: s01, s02, s03, s06, s08, m01
    # CHAPTER: M03.2 section 2.3
    g = rng(71)
    for _ in range(60):
        n = 1 + g.below(12)
        A = g.uniform_array((n, n), -1.0, 1.0)
        check_factors(A, *lu(A))


def test_solve_matches_numpy():
    # WHY: against LAPACK (np.linalg.solve) on well-conditioned systems
    #      (diagonally shifted), with one and with several right-hand sides:
    #      M07.7's IRLS solves one system per iteration, M10.5 several.
    # KIND: differential
    # CATCHES: s05, m02
    # CHAPTER: M03.2 section 2.4
    g = rng(72)
    for n in (1, 2, 3, 7, 16, 40):
        A = g.uniform_array((n, n), -1.0, 1.0) + n * np.eye(n)
        b = g.uniform_array((n,), -1.0, 1.0)
        B = g.uniform_array((n, 3), -1.0, 1.0)
        P, L, U = lu(A)
        assert_close(lu_solve(P, L, U, b), np.linalg.solve(A, b), rtol=1e-9, atol=1e-12)
        X = lu_solve(P, L, U, B)
        assert X.shape == (n, 3)
        assert_close(X, np.linalg.solve(A, B), rtol=1e-9, atol=1e-12)


def test_zero_leading_pivot_needs_a_swap():
    # WHY: [[0, 1], [1, 1]] is perfectly invertible, but elimination without
    #      a row swap divides by the 0 in the corner. Pivoting is not an
    #      optimization; it is what makes elimination work at all.
    # KIND: boundary
    # CATCHES: s01, s09
    # CHAPTER: M03.2 section 2.3
    A = np.array([[0.0, 1.0], [1.0, 1.0]])
    P, L, U = lu(A)
    check_factors(A, P, L, U)
    assert lu_solve(P, L, U, np.array([1.0, 3.0])).tolist() == [2.0, 1.0]


def test_tiny_pivot_loses_everything_without_partial_pivoting():
    # WHY: the classic: eps = 1e-20, A = [[eps, 1], [1, 1]], b = [1, 2].
    #      The answer is x ~ [1, 1]. Pivoting on eps gives a multiplier 1e20,
    #      U[1, 1] = 1 - 1e20 rounds to -1e20, and x[0] comes out 0. With the
    #      swap, the multiplier is eps and nothing is lost.
    # KIND: boundary
    # CATCHES: s01
    # CHAPTER: M03.2 section 2.3
    eps = 1e-20
    A = np.array([[eps, 1.0], [1.0, 1.0]])
    P, L, U = lu(A)
    x = lu_solve(P, L, U, np.array([1.0, 2.0]))
    assert_close(x, [1.0, 1.0], rtol=1e-12, atol=1e-12)


def test_pivot_uses_absolute_value():
    # WHY: the pivot is the largest |entry|, not the largest entry: in column
    #      [1, -3], -3 must become the pivot, or the multiplier is -3 (|L| > 1)
    #      and errors grow by that factor at every step.
    # KIND: boundary
    # CATCHES: s02
    # CHAPTER: M03.2 section 5, Pitfalls
    A = np.array([[1.0, 2.0], [-3.0, 4.0]])
    P, L, U = lu(A)
    assert P.tolist() == [[0, 1], [1, 0]]
    assert U[0, 0] == -3.0
    check_factors(A, P, L, U)


def test_pivot_search_ignores_finished_rows():
    # WHY: after the first step, column 1 holds 100 in row 0 (already the
    #      first pivot row, finished) and 1 and -22 in rows 1 and 2. The
    #      search must look only at rows k and below and pick -22 (row 2);
    #      looking at the whole column finds row 0 and ends up pivoting on 1,
    #      with a multiplier of -22.
    # KIND: boundary
    # CATCHES: s08
    # CHAPTER: M03.2 section 2.3
    A = np.array([[4.0, 100.0, 0.0], [2.0, 51.0, 1.0], [1.0, 3.0, 5.0]])
    P, L, U = lu(A)
    check_factors(A, P, L, U)
    assert P.tolist() == [[1, 0, 0], [0, 0, 1], [0, 1, 0]]
    assert U[1, 1] == -22.0


def test_singular_matrix_factors_but_does_not_solve():
    # WHY: a singular A still has P A = L U (U gets a zero pivot), which is
    #      useful: the zero tells you the rank dropped. Solving with it must
    #      raise instead of dividing by zero into inf and NaN.
    # KIND: boundary
    # CATCHES: s07, m03
    # CHAPTER: M03.2 section 2.4
    A = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 0.0, 1.0]])
    P, L, U = lu(A)
    check_factors(A, P, L, U)
    assert U[2, 2] == 0.0
    with pytest.raises(ValueError):
        lu_solve(P, L, U, np.ones(3))
    Z = np.zeros((3, 3))
    P, L, U = lu(Z)
    assert (U == 0).all() and (L == np.eye(3)).all()


def test_input_is_not_modified_and_shapes_are_checked():
    # WHY: callers (IRLS, inverse iteration) reuse A after factoring it; an
    #      in-place elimination silently corrupts their next step.
    # KIND: boundary
    # CATCHES: s09
    # CHAPTER: M03.2 section 4
    A = hand_a()
    lu(A)
    assert A.tolist() == HAND
    for bad in (np.ones((2, 3)), np.ones(3), np.ones((2, 2, 2))):
        with pytest.raises(ValueError):
            lu(bad)
    P, L, U = lu(hand_a())
    with pytest.raises(ValueError):
        lu_solve(P, L, U, np.ones(4))


def test_permutation_applied_to_b_not_its_transpose():
    # WHY: with a 3-cycle permutation, P b and P^T b differ. lu_solve must
    #      solve L y = P b, matching P A = L U.
    # KIND: unit
    # CATCHES: s04
    # CHAPTER: M03.2 section 2.4
    A = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 5.0], [7.0, 0.0, 1.0]])
    P, L, U = lu(A)
    assert not (P == P.T).all(), "the test needs a non-symmetric permutation"
    b = np.array([1.0, 2.0, 3.0])
    assert_close(A @ lu_solve(P, L, U, b), b, rtol=1e-12, atol=1e-12)
