# contracts/py/tinyllm/num/integrate.pyi (M01.4)
# chapter: content/tracks/math/01-calculus-1/04-definite-integrals-trapezoid-simpson.md
#
# Definite integrals by quadrature: the area under a curve from samples
# (trapezoid) or from a function evaluated on n equal steps (the composite
# trapezoid and Simpson rules). M07.7's roc_auc is trapezoid(tpr, fpr).
# All arithmetic is float64 and every function returns a Python float.
from typing import Callable

from numpy.typing import ArrayLike, NDArray

def trapezoid(y: ArrayLike, x: ArrayLike) -> float:
    """sum over i of (x[i+1] - x[i]) * (y[i] + y[i+1]) / 2, in the order the
    samples are given (numpy.trapezoid(y, x)): nothing is sorted, so a
    decreasing x gives the negative of the area. Fewer than two samples give
    0.0. ValueError unless x and y are 1-D, the same length, and finite."""

def trapezoid_rule(
    f: Callable[[NDArray], NDArray], a: float, b: float, n: int
) -> float:
    """The composite trapezoid rule with n equal steps, h = (b - a) / n, on the
    nodes x = numpy.linspace(a, b, n + 1):
        h * (f(x_0) / 2 + f(x_1) + ... + f(x_{n-1}) + f(x_n) / 2).
    f is called once, with the float64 array of nodes, and returns an array of
    the same shape. Error O(h**2); exact on straight lines. b < a gives the
    negative of the integral from b to a, and a == b gives 0.0.
    ValueError unless a and b are finite and n is an integer >= 1, or when f
    returns the wrong shape or a non-finite value."""

def simpson(f: Callable[[NDArray], NDArray], a: float, b: float, n: int) -> float:
    """Composite Simpson's rule with n equal steps (n even), h = (b - a) / n, on
    the nodes x = numpy.linspace(a, b, n + 1):
        h / 3 * (f(x_0) + 4 f(x_1) + 2 f(x_2) + 4 f(x_3) + ... + 4 f(x_{n-1}) + f(x_n)).
    One parabola through each pair of steps. f is called once, as for
    trapezoid_rule. Error O(h**4); exact (up to rounding) on cubics. Equal to
    (4 T(n) - T(n / 2)) / 3 with T = trapezoid_rule: Richardson extrapolation
    of the trapezoid rule. b < a and a == b behave as in trapezoid_rule.
    ValueError unless n is an even integer >= 2 (an odd n is never rounded),
    a and b are finite, and f returns finite values of the right shape."""
