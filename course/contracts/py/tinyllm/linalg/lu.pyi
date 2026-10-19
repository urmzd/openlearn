# contracts/py/tinyllm/linalg/lu.pyi (M03.2)
# chapter: content/courses/mathematics-for-computing/04-linear-algebra/02-gaussian-elimination-and-lu.md
#
# Gaussian elimination with partial pivoting, kept as a factorization
# P A = L U, so one O(n^3) factorization serves many O(n^2) solves.
from numpy.typing import ArrayLike, NDArray

def lu(A: ArrayLike) -> tuple[NDArray, NDArray, NDArray]:
    """P, L, U (float64 [n, n]) with P @ A == L @ U for a square A.
    P is a permutation matrix; L is unit lower triangular with |L[i, j]| <= 1;
    U is upper triangular. Column k pivots on the row i >= k with the largest
    |value| (ties to the lowest i); a column that is already zero from row k
    down is skipped, so a singular A still factors (U then has a zero on its
    diagonal). A is not modified. ValueError unless A is 2-D and square."""

def lu_solve(P: ArrayLike, L: ArrayLike, U: ArrayLike, b: ArrayLike) -> NDArray:
    """x with A @ x == b, from the factors of A: forward substitution
    L y = P b, then back substitution U x = y. b is [n] or [n, k] (k right
    hand sides); x has b's shape, float64. ValueError when U has a zero on its
    diagonal (A is singular) or the shapes disagree."""
