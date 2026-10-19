"""Taylor series, remainder bounds, and range reduction (M02.1).

A truncated Taylor series is a polynomial, and a polynomial is all a CPU can
evaluate: additions and multiplications. The Lagrange remainder says how
far the polynomial is from the function, and it is small only near the
expansion point. Range reduction moves every input there first:
exp(x) = 2**k exp(r) with |r| <= ln(2) / 2. The polynomial is evaluated with
M00.4's horner. M09.6 ports exp_range_reduced to C as tl_expf; M01.3's
gelu_erf calls erf_series.

Contract: contracts/py/tinyllm/num/taylor.pyi.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tinyllm.num.poly import horner

ERF_CUTOFF = 6.0  # erfc(6) = 2.2e-17 < half an ulp of 1.0: erf(x) rounds to sign(x)
LN2 = math.log(2.0)
# Cody and Waite: ln 2 split so that k * LN2_HI is exact for |k| < 2**21
# (LN2_HI has 21 trailing zero bits) and LN2_LO carries the rest.
LN2_HI = 6.93147180369123816490e-01
LN2_LO = 1.90821492927058770002e-10


def exp_taylor_coeffs(n: int) -> NDArray:
    if n < 0:
        raise ValueError(f"degree n must be >= 0, got {n}")
    c = np.empty(int(n) + 1, dtype=np.float64)
    c[0] = 1.0
    for i in range(1, int(n) + 1):
        c[i] = c[i - 1] / i  # 1/i! = (1/(i-1)!) / i, no factorial overflow
    return c


def exp_range_reduced(x: ArrayLike, deg: int = 6) -> NDArray:
    if deg < 0:
        raise ValueError(f"deg must be >= 0, got {deg}")
    x = np.asarray(x, dtype=np.float64)
    nan = np.isnan(x)
    # exp(-750) is below the smallest subnormal and exp(710) above the
    # largest double, so clipping changes no result and keeps k small.
    xc = np.clip(np.where(nan, 0.0, x), -750.0, 710.0)
    k = np.rint(xc / LN2)  # round half to even: |r| <= ln(2) / 2
    r = (xc - k * LN2_HI) - k * LN2_LO
    # Ascending coefficients, the order horner (M00.4) takes; numpy.polyval
    # wants them descending.
    p = horner(exp_taylor_coeffs(deg), r)
    with np.errstate(over="ignore", under="ignore"):
        out = np.ldexp(p, k.astype(np.int32))
    return np.where(nan, np.nan, out)


def erf_series(x: ArrayLike, terms: int) -> NDArray:
    if terms < 1:
        raise ValueError(f"terms must be >= 1, got {terms}")
    x = np.asarray(x, dtype=np.float64)
    big = np.abs(x) >= ERF_CUTOFF
    xs = np.where(big | np.isnan(x), 0.0, x)
    x2 = xs * xs
    # t_0 = x, t_{n+1} = t_n * 2 x^2 / (2n + 3): every term has the sign of x.
    term = xs.copy()
    total = xs.copy()
    for n in range(1, int(terms)):
        term = term * (2.0 * x2) / (2 * n + 1)
        total = total + term
    out = (2.0 / math.sqrt(math.pi)) * np.exp(-x2) * total
    out = np.where(big, np.sign(x), out)
    return np.where(np.isnan(x), np.nan, out)
