# contracts/py/tinyllm/num/newton.pyi (M01.2)
# chapter: content/courses/mathematics-for-computing/01-calculus-1/02-newtons-method.md
#
# Newton's method for a root of a scalar function, and the division-free
# Newton iteration for 1 / sqrt(x) that M09.5 ports to C.
from typing import Callable, Optional

from numpy.typing import ArrayLike, NDArray

def newton(
    f: Callable[[float], float],
    df: Optional[Callable[[float], float]],
    x0: float,
    tol: float = 1e-12,
    max_iter: int = 50,
) -> tuple[float, int]:
    """Iterate x[n+1] = x[n] - f(x[n]) / df(x[n]) from x[0] = x0 and return
    (x, n): the first iterate x[n] with |x[n] - x[n-1]| <= tol * max(1, |x[n]|),
    and the number of updates n taken to reach it. If f(x[n]) == 0 exactly,
    x[n] is returned at once (n may be 0). df = None uses
    tinyllm.num.diff.central_diff (M01.1) with its default step.
    ValueError when x0 is not finite, tol <= 0, or max_iter < 1.
    RuntimeError when df(x[n]) is zero or not finite, an iterate is not
    finite, or max_iter updates do not converge; the message says which."""

def rsqrt_newton(x: ArrayLike, y0: ArrayLike, iters: int) -> NDArray:
    """Apply y <- y * (1.5 - 0.5 * x * y * y) iters times, elementwise, from y0:
    Newton's method on g(y) = 1 / y**2 - x, which needs no division.
    The arithmetic and the result use the floating dtype of
    numpy.result_type(x, y0) (float64 for integer inputs), so float32 in
    means float32 out, step for step as the C port computes it.
    x and y0 broadcast against each other. iters = 0 returns y0 in that dtype.
    ValueError when iters < 0."""
