# contracts/py/tinyllm/num/jacobian.pyi (M04.2)
# chapter: content/courses/mathematics/10-calculus-3/02-jacobians-and-the-chain-rule.md
#
# Numerical Jacobians and Jacobian-vector products of f: R^n -> R^m.
# x and f(x) may have any shape; they are flattened in row-major order, so
# n = x.size, m = f(x).size, and J[i, j] = d f(x).flat[i] / d x.flat[j].
# All arithmetic is float64 on a private copy of x; the caller's x is never
# modified. The forward-mode product J v and the reverse-mode product u^T J
# are what autodiff computes without forming J (M08.1, M08.2).
from typing import Callable

from numpy.typing import ArrayLike, NDArray

def jacobian(
    f: Callable[[NDArray], ArrayLike], x: ArrayLike, eps: float = 1e-6
) -> NDArray:
    """float64 [m, n], column j = (f(x + eps e_j) - f(x - eps e_j)) / step_j,
    where step_j = (x_j + eps) - (x_j - eps) is the step actually taken.
    2n calls of f. ValueError when eps is not finite or <= 0, or when f
    returns arrays of different shapes."""

def jvp_numeric(f: Callable[[NDArray], ArrayLike], x: ArrayLike, v: ArrayLike) -> NDArray:
    """J v, shape f(x).shape: the directional derivative
    (f(x + h v) - f(x - h v)) / (2 h) with h = 1e-6 / max|v|, so the largest
    coordinate moves by 1e-6 whatever the size of v. Two calls of f.
    v = 0 gives zeros. ValueError when v's shape is not x's."""

def vjp_numeric(f: Callable[[NDArray], ArrayLike], x: ArrayLike, u: ArrayLike) -> NDArray:
    """u^T J, shape x.shape: the gradient of the scalar sum(u * f(x)) with
    respect to x, by tinyllm.num.gradcheck.numerical_grad (M04.1) with its
    default step. ValueError when u's shape is not f(x)'s."""
