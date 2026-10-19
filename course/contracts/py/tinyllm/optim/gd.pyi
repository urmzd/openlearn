# contracts/py/tinyllm/optim/gd.pyi (M10.1): gradient descent and backtracking line search
# chapter: content/courses/mathematics-for-computing/11-optimization/01-gradient-descent-and-line-search.md
#
# f maps a float64 array x to a float; grad maps x to an array of x's shape.
# Iterates are float64 arrays; nothing here mutates the caller's arrays.
from typing import Callable

from numpy.typing import ArrayLike, NDArray

def gradient_descent(
    f: Callable[[NDArray], float],
    grad: Callable[[NDArray], NDArray],
    x0: ArrayLike,
    lr: float,
    steps: int,
) -> list[NDArray]:
    """x_{t+1} = x_t - lr * grad(x_t), for t = 0 .. steps - 1. Returns the
    steps + 1 iterates [x_0, x_1, ..., x_steps] as independent float64
    arrays (x_0 a copy of x0). f is evaluated at every iterate only to detect
    divergence: when f(x_t) or any entry of x_t is not finite,
    FloatingPointError naming t. ValueError when lr <= 0 or steps < 0."""

def armijo_step(
    f: Callable[[NDArray], float],
    grad: Callable[[NDArray], NDArray],
    x: ArrayLike,
    d: ArrayLike,
    alpha0: float = 1.0,
    c: float = 1e-4,
    rho: float = 0.5,
) -> float:
    """Backtracking line search: the first alpha in alpha0, rho alpha0,
    rho^2 alpha0, ... with the sufficient decrease (Armijo) condition
        f(x + alpha d) <= f(x) + c alpha grad(x) . d
    (a comparison that is False for nan, so a step into nan is rejected).
    ValueError unless d is a descent direction (grad(x) . d < 0), 0 < c < 1,
    0 < rho < 1, and alpha0 > 0. RuntimeError when 60 reductions find no
    such alpha."""
