# contracts/py/tinyllm/linalg/svd.pyi (M03.5)
# chapter: content/tracks/math/03-linear-algebra/05-svd-low-rank-and-least-squares.md
#
# The singular value decomposition by one-sided Jacobi rotations, the best
# rank-r approximation it gives (Eckart-Young), and least squares by QR.
# Every result is float64; inputs are never modified.
from numpy.typing import ArrayLike, NDArray

def svd(A: ArrayLike) -> tuple[NDArray, NDArray, NDArray]:
    """Reduced SVD of a real [m, n] matrix, k = min(m, n): A == U @ diag(S) @ Vt
    with U float64 [m, k] (orthonormal columns), S float64 [k] (singular
    values, non-negative, in descending order), Vt float64 [k, n]
    (orthonormal rows).

    Algorithm (one-sided Jacobi, Hestenes 1958) on the tall matrix: A itself
    when m >= n, else A^T (then the roles of U and V swap at the end).
    W = a copy of the tall matrix, V = I. Sweep over the column pairs
    (i, j), i < j, in row-major order; for each pair with
    alpha = w_i . w_i, beta = w_j . w_j, gamma = w_i . w_j, rotate both W and
    V by the plane rotation that makes columns i and j orthogonal whenever
    |gamma| > rows * eps * sqrt(alpha * beta) (eps = 2^-52, rows = the
    tall matrix's row count): zeta = (beta - alpha) / (2 gamma),
    t = sign(zeta) / (|zeta| + sqrt(1 + zeta^2)) with sign(0) = +1,
    c = 1 / sqrt(1 + t^2), s = c t, then
    (w_i, w_j) <- (c w_i - s w_j, s w_i + c w_j), the same for V's columns.
    Stop after the first sweep that rotates nothing (at most 64 sweeps).
    Then sigma_j = ||w_j||, sorted descending (stable for ties), and
    u_j = w_j / sigma_j.

    Numerically zero singular values: when sigma_j <= rows * eps * sigma_1
    (or sigma_1 == 0), u_j is not w_j / sigma_j but the first standard basis
    vector e_i (i = 0, 1, ...) whose component orthogonal to the columns of
    U chosen so far has norm > 1/2, orthogonalized against them (twice) and
    normalized. So U has orthonormal columns for rank-deficient A too.

    Sign rule: each singular pair (u_j, v_j) may be negated together. The
    returned pair is the one whose row Vt[j] has its entry of largest
    absolute value positive (the first such entry on ties). With distinct
    singular values this makes the factorization unique, so it can be
    compared entry by entry with LAPACK after the same rule.

    ValueError unless A is 2-D with finite entries. An empty dimension
    (k == 0) returns U [m, 0], S [0], Vt [0, n]."""

def low_rank(A: ArrayLike, r: int) -> tuple[NDArray, NDArray]:
    """The best rank-r approximation of A in the spectral and Frobenius
    norms (Eckart-Young), split in balanced factors: with svd(A) = U, S, Vt,
    B = U[:, :r] * sqrt(S[:r]) is float64 [m, r] and
    C = sqrt(S[:r])[:, None] * Vt[:r] is float64 [r, n], so B @ C = A_r,
    ||A - B @ C||_2 = S[r] (0 when r == k) and
    ||A - B @ C||_F = sqrt(sum of S[r:]^2). LoRA's PiSSA init (L6.6) and
    MLA's conversion (L7.6) use this split. ValueError unless
    1 <= r <= min(m, n) (and as svd)."""

def lstsq(A: ArrayLike, b: ArrayLike) -> NDArray:
    """argmin_x ||A x - b||_2 for A float [m, n] with m >= n and full column
    rank, by QR (M03.3's qr_householder): A = Q R, x = R^{-1} (Q^T b) by back
    substitution. Never forms A^T A (the normal equations square the
    condition number). b is [m] (x is [n]) or [m, p] (x is [n, p], one
    solve per column). ValueError when A is not 2-D, m < n, b's first
    dimension is not m, or A is numerically rank deficient:
    min_j |R_jj| <= max(m, n) * eps * max_j |R_jj| (eps = 2^-52)."""
