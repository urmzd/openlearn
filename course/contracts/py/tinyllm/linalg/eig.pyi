# contracts/py/tinyllm/linalg/eig.pyi (M03.4)
# chapter: content/courses/mathematics/09-linear-algebra/04-eigenvalues-and-power-iteration.md
#
# Eigenvalues by iteration: power iteration for the dominant eigenpair,
# inverse iteration (one LU, many solves) for the one nearest a shift, and the
# spectral radius rho(W) = max |lambda| of any real square matrix.
from typing import Any, Callable

from numpy.typing import ArrayLike, NDArray

def power_iteration(
    matvec: Callable[[NDArray], NDArray], n: int, iters: int, rng: Any
) -> tuple[float, NDArray]:
    """Dominant eigenpair of the linear map x -> matvec(x) on R^n.
    v_0 = n draws of 2 * rng.uniform() - 1 (C order), normalized; then
    `iters` times: w = matvec(v), v = w / ||w||. Returns (v . matvec(v), v):
    the Rayleigh quotient, which keeps the eigenvalue's sign, and the unit
    vector. If some w is exactly 0, returns (0.0, the current v).
    ValueError for n < 1 or iters < 0."""

def inverse_iteration(
    A: ArrayLike, shift: float, iters: int, rng: Any
) -> tuple[float, NDArray]:
    """The eigenpair of a square A whose eigenvalue is nearest `shift`:
    power iteration on (A - shift I)^{-1}, applied by lu_solve with a single
    lu (M03.2) of A - shift I. Start and stopping rules as power_iteration;
    returns (v . A v, v). ValueError when A - shift I is singular (shift is
    exactly an eigenvalue) or A is not square."""

def spectral_radius(W: ArrayLike, iters: int = 100) -> float:
    """rho(W) = max |lambda| over the eigenvalues of a real square W, by the
    unshifted QR algorithm with qr_householder (M03.3): A_0 = W,
    A_{t+1} = R_t Q_t. A_t tends to block upper triangular form whose
    diagonal blocks are 1x1 (real eigenvalues) or 2x2 (complex pairs, which
    power iteration cannot find); rho is the largest modulus over those
    blocks. Converges when eigenvalues of different modulus separate; the
    leading block is exact for a dominant real eigenvalue or complex pair.
    ValueError unless W is 2-D and square; 0.0 for the empty matrix."""
