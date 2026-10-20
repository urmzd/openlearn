# contracts/py/tinyllm/num/poly.pyi (M00.4)
# chapter: content/courses/mathematics/06-precalculus/04-polynomials-horner-and-stable-roots.md
#
# A polynomial p(x) = c[0] + c[1] x + ... + c[n] x^n is its list of
# coefficients in ASCENDING order of power (coeffs[k] multiplies x^k).
# numpy.polyval uses the opposite order.
from collections.abc import Sequence

from numpy.typing import ArrayLike, NDArray

def horner(coeffs: Sequence[float], x: ArrayLike) -> NDArray:
    """p(x) by Horner's rule, c[0] + x (c[1] + x (c[2] + ... + x c[n])), in
    float64, elementwise over x: the result has x's shape (a scalar x gives a
    float64 scalar). Empty coeffs is the zero polynomial (zeros). Exact when
    the coefficients and x are integers and every partial result stays below
    2^53; otherwise the error is at most gamma_{2n} * sum_k |c[k]| |x|^k
    with gamma_k = k u / (1 - k u), u = 2^-53."""

def quadratic_roots(a: float, b: float, c: float) -> tuple[float, float]:
    """The two real roots (x1, x2), x1 <= x2, of a x^2 + b x + c = 0, without
    cancellation: q = -(b + s sqrt(b^2 - 4ac)) / 2 with s = +1 when b >= 0 and
    -1 otherwise, then x = q / a and x = c / q (a double root comes back
    twice). The coefficients are first scaled by a power of two, so b^2 does
    not overflow for |b| up to about 1e300. Each root has a relative error of
    a few units in the last place whenever the roots are well separated.
    ValueError when a == 0, b^2 - 4ac < 0 (complex roots), or a coefficient
    is NaN or infinite."""
