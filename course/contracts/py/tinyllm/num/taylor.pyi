# contracts/py/tinyllm/num/taylor.pyi (M02.1)
# chapter: content/courses/mathematics-for-computing/03-calculus-2/01-taylor-series.md
#
# Truncated Taylor series with their remainder bounds, and range reduction:
# evaluate a series only where it converges fast, and map every other input
# there exactly. All arithmetic is float64; results are float64 arrays with
# the shape of x (a 0-d array for a scalar x).
from numpy.typing import ArrayLike, NDArray

ERF_CUTOFF: float  # 6.0: for |x| >= ERF_CUTOFF, erf(x) rounds to sign(x) in float64

def exp_taylor_coeffs(n: int) -> NDArray:
    """The Maclaurin coefficients of exp up to degree n: [1/0!, 1/1!, ..., 1/n!],
    float64, length n + 1. ValueError when n < 0."""

def exp_range_reduced(x: ArrayLike, deg: int = 6) -> NDArray:
    """exp(x) as 2**k * p(r), where k = round(x / ln 2) (half to even),
    r = x - k ln 2 computed in two parts (Cody and Waite: ln 2 = LN2_HI + LN2_LO,
    LN2_HI with its low 21 bits zero so k * LN2_HI is exact for |k| < 2**21),
    so |r| <= ln(2) / 2, and p is the degree-deg Taylor polynomial of exp at 0,
    evaluated by tinyllm.num.poly.horner (M00.4). Relative error at most
    e**(|r| - r) |r|**(deg + 1) / (deg + 1)! plus a few rounding errors.
    exp(+inf) = inf, exp(-inf) = 0, nan stays nan; results that overflow are
    inf and results that underflow are 0 (or subnormal), with no warnings.
    ValueError when deg < 0."""

def erf_series(x: ArrayLike, terms: int) -> NDArray:
    """erf(x) = (2 / sqrt(pi)) exp(-x**2) sum_{n=0}^{terms-1} 2**n x**(2n+1) / (2n+1)!!
    for |x| < ERF_CUTOFF, where (2n+1)!! = 1 * 3 * 5 * ... * (2n+1); every term
    has the sign of x, so nothing cancels. sign(x) for |x| >= ERF_CUTOFF,
    nan for nan. With terms >= 120 the result is within 3e-15 of erf
    everywhere. After N terms the omitted tail is at most
    t_N / (1 - 2 x**2 / (2N + 3)) times the prefactor, once 2 x**2 < 2N + 3,
    where t_N is the first omitted term.
    ValueError when terms < 1."""
