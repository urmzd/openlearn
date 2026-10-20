# contracts/py/tinyllm/linalg/qr.pyi (M03.3)
# chapter: content/courses/mathematics/09-linear-algebra/03-orthogonality-and-householder-qr.md
#
# QR by Householder reflections, and the orthogonal initializer built on it.
from typing import Any

from numpy.typing import ArrayLike, NDArray

def qr_householder(A: ArrayLike) -> tuple[NDArray, NDArray]:
    """Reduced QR of a real [m, n] matrix, k = min(m, n): Q float64 [m, k]
    with orthonormal columns (Q^T Q = I), R float64 [k, n] upper triangular
    with a non-negative diagonal, and A == Q @ R. The sign rule makes the
    factors unique when A has full column rank. Reflector j maps column j
    below the diagonal onto -sign(x_0) ||x|| e_1 (sign(0) = +1), which
    avoids cancellation; a column that is already zero is skipped. A is not
    modified. ValueError unless A is 2-D."""

def orthogonal_init(shape: tuple[int, int], gain: float, rng: Any) -> NDArray:
    """float64 [rows, cols] with orthonormal rows (rows <= cols) or columns
    (rows >= cols), scaled by gain, the way torch.nn.init.orthogonal_ does it:
    draw G = normal(rng, rows * cols) (tinyllm.prob.rv, Box-Muller over
    rng.uniform()) reshaped to [rows, cols] in C order; take the QR of G, or
    of G^T when rows < cols; the sign rule above makes Q uniformly
    distributed; transpose back and multiply by gain.
    ValueError for a non-positive dimension."""
