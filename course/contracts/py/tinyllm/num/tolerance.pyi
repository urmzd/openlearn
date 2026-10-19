# contracts/py/tinyllm/num/tolerance.pyi (M09.3)
# chapter: content/courses/mathematics-for-computing/10-numerical-methods-and-floating-point/03-error-analysis-condition-numbers-and-tolerance-budgets.md
#
# Error analysis turned into test tolerances. The model of floating point is
# fl(a op b) = (a op b)(1 + delta) with |delta| <= u, the unit roundoff of
# the format: u = 2^-p where p counts the significand bits including the
# implicit one (f64 53, f32 24, f16 11, bf16 8, e4m3 4, e5m2 3). From it,
# a sum or dot product of k terms evaluated in any order obeys
#     |fl(x . y) - x . y| <= gamma_k |x| . |y|,   gamma_k = k u / (1 - k u)
# (Higham, Accuracy and Stability of Numerical Algorithms, 3.1), a bound that
# holds for every input, which is what lets a differential test fail only on
# a real bug. `dtype` names the format the arithmetic ran in: one of "f64",
# "f32", "f16", "bf16", "e4m3", "e5m2", or the numpy names "float64",
# "float32", "float16", "bfloat16"; anything else is a ValueError.
from typing import Callable, Literal

from numpy.typing import ArrayLike, NDArray

def unit_roundoff(dtype: str) -> float:
    """u = 2^-p: half the gap between 1 and the next float (f32: 2^-24,
    f64: 2^-53, bf16: 2^-8)."""

def gamma(k: int, dtype: str) -> float:
    """gamma_k = k u / (1 - k u), the accumulated relative error of k
    roundings. gamma(0, dtype) == 0. ValueError for k < 0 or k u >= 1 (the
    bound says nothing then: bf16 at k = 256, e4m3 at k = 16)."""

def sum_error_bound(k: int, dtype: str, abs_sum: ArrayLike) -> NDArray:
    """gamma_(k-1) * abs_sum: the bound on |fl(sum of k terms) - sum| for
    recursive summation in any order, where abs_sum = sum_i |x_i| (k - 1
    additions; one term is exact). float64 array, shape of abs_sum.
    ValueError for k < 1."""

def dot_error_bound(k: int, dtype: str, abs_dot: ArrayLike) -> NDArray:
    """gamma_k * abs_dot: the bound on |fl(x . y) - x . y| for a dot product
    of k terms (k multiplications, k - 1 additions), where abs_dot = |x| . |y|
    computed exactly or in higher precision. float64 array, shape of
    abs_dot. ValueError for k < 1."""

def matmul_error_bound(A: ArrayLike, B: ArrayLike, dtype: str, dA: ArrayLike = 0.0) -> NDArray:
    """float64 [m, n] elementwise bound on |C' - A @ B| where C' is A' @ B
    computed in dtype and A' is any matrix with |A' - A| <= dA elementwise
    (dA a scalar or broadcast to A's shape: the quantization step of L8.5 is
    dA = scale / 2): dA @ |B| + gamma_k (|A| + dA) @ |B|, with k = A.shape[1]
    and the products formed in float64. dA = 0 is the plain rounding bound
    (the optional L9.1 parity check). ValueError unless A is [m, k], B is [k, n],
    k >= 1, and dA >= 0."""

def assert_close_bounded(
    actual: ArrayLike,
    expected: ArrayLike,
    k: int,
    dtype: str,
    abs_dot: ArrayLike,
    slack: float = 4.0,
) -> None:
    """Pass when |actual - expected| <= slack * dot_error_bound(k, dtype,
    abs_dot) elementwise (both NaN, or the same infinity, also pass); raise
    AssertionError otherwise, naming the first failing index, the error, the
    bound, and their ratio. Shapes must match (AssertionError). `expected` is
    the exact or float64 value; slack > 1 absorbs the error of computing
    expected itself. ValueError for slack <= 0."""

def bound_ratio(actual: ArrayLike, expected: ArrayLike, bound: ArrayLike) -> float:
    """max_i |actual_i - expected_i| / bound_i: <= 1 means within the
    budget (optional C parity reports it per operation). A zero error over a zero bound counts 0;
    a nonzero error over a zero bound is inf; an empty input gives 0.0."""

def cond(A: ArrayLike) -> float:
    """The 2-norm condition number sigma_max / sigma_min of a square or
    rectangular matrix, from M03.5's singular values (min(m, n) of them).
    inf when the matrix is singular to working precision: sigma_min <=
    max(m, n) * 2^-52 * sigma_max (M03.5's numerically-zero rule), or
    sigma_max == 0. cond(I) == 1. ValueError unless A is a finite, non-empty
    2-D array."""

def relative_condition(f: Callable[[float], float], df: Callable[[float], float], x: float) -> float:
    """The relative condition number of a scalar function at x,
    |x f'(x) / f(x)|: how much a relative change in x is amplified in f(x).
    0 when x f'(x) == 0; inf when f(x) == 0 and x f'(x) != 0."""

def fd_error_model(
    h: float,
    order: Literal[1, 2],
    dtype: str,
    f_scale: float = 1.0,
    deriv_scale: float = 1.0,
) -> float:
    """The error model of a finite difference with step h (M01.1):
    order 1 (forward): deriv_scale h / 2 + 2 u f_scale / h, deriv_scale = |f''|;
    order 2 (central): deriv_scale h^2 / 6 + u f_scale / h, deriv_scale = |f'''|.
    Truncation falls with h, rounding grows as 1/h. ValueError for h <= 0 or
    another order."""

def optimal_fd_step(
    order: Literal[1, 2], dtype: str, f_scale: float = 1.0, deriv_scale: float = 1.0
) -> float:
    """The h that minimizes fd_error_model: order 1: 2 sqrt(u f_scale /
    deriv_scale); order 2: (3 u f_scale / deriv_scale)^(1/3). In float64 with
    unit scales: 2^-25.5 (about 2.1e-8) and about 6.9e-6. ValueError for
    another order or non-positive scales."""
