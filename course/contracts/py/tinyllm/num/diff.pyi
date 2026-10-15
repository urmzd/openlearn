# contracts/py/tinyllm/num/diff.pyi (M01.1)
# chapter: content/tracks/math/01-calculus-1/01-derivatives-and-finite-differences.md
#
# Finite-difference derivatives of a scalar function of one real variable.
# eps below is float64 machine epsilon, 2**-52 (numpy.finfo(float).eps).
# Every function evaluates f only at float64 points and returns a Python float.
from typing import Callable, Optional

def forward_diff(
    f: Callable[[float], float], x: float, h: Optional[float] = None
) -> float:
    """(f(x + h) - f(x)) / ((x + h) - x): the one-sided difference, error O(h).
    The divisor is the step actually taken in floating point, not h.
    Default h = sqrt(eps) * max(1, |x|).
    ValueError when x is not finite, or h is given and is not finite or <= 0."""

def central_diff(
    f: Callable[[float], float], x: float, h: Optional[float] = None
) -> float:
    """(f(x + h) - f(x - h)) / ((x + h) - (x - h)): the symmetric difference,
    error O(h**2). The divisor is the step actually taken in floating point.
    Default h = cbrt(eps) * max(1, |x|); a given h is used as is.
    ValueError when x is not finite, or h is given and is not finite or <= 0."""

def richardson(
    f: Callable[[float], float], x: float, h: float, levels: int = 2
) -> float:
    """Richardson extrapolation of central differences. With
    D[0][j] = central_diff(f, x, h / 2**j) for j = 0..levels and
    D[k][j] = (4**k * D[k-1][j+1] - D[k-1][j]) / (4**k - 1),
    returns D[levels][0], whose error is O(h**(2 * levels + 2)).
    levels = 0 is central_diff(f, x, h). Exact (up to rounding) on
    polynomials of degree <= 2 * levels + 2.
    ValueError when x or h is not finite, h <= 0, or levels < 0."""
