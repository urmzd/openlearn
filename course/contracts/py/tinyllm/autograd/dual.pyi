# contracts/py/tinyllm/autograd/dual.pyi (M08.1): forward-mode autodiff with dual numbers
# chapter: content/tracks/math/08-matrix-calculus-and-autodiff/01-dual-numbers-forward-mode.md
#
# A dual number a + b eps with eps^2 = 0 carries a value a and a tangent b.
# Evaluating f on x + v eps gives f(x) + (J_f(x) v) eps: the value and the
# Jacobian-vector product in one pass. `val` and `eps` are both floats, or
# both float64 arrays of one shape (a vector of dual numbers; a scalar
# tangent is broadcast to the value's shape).
#
# Arithmetic mixes Dual with Python numbers, numpy scalars, and numpy
# arrays on either side. A non-Dual operand is a constant (tangent 0).
# numpy must hand the operation to Dual even when the numpy operand is on
# the left (np.float64(2.0) * d, W @ d): Dual sets __array_ufunc__ = None.
from typing import Any, Callable

from numpy.typing import ArrayLike, NDArray

class Dual:
    val: Any  # float, or float64 ndarray
    eps: Any  # same type and shape as val

    __array_ufunc__: None

    def __init__(self, val: Any, eps: Any = 0.0) -> None:
        """TypeError when val or eps is itself a Dual (no nesting)."""
    def __add__(self, other: Any) -> Dual: ...
    def __radd__(self, other: Any) -> Dual: ...
    def __sub__(self, other: Any) -> Dual: ...
    def __rsub__(self, other: Any) -> Dual: ...
    def __mul__(self, other: Any) -> Dual:
        """(a + b eps)(c + d eps) = ac + (ad + bc) eps."""
    def __rmul__(self, other: Any) -> Dual: ...
    def __truediv__(self, other: Any) -> Dual:
        """(a + b eps) / (c + d eps) = a/c + ((b c - a d) / c^2) eps."""
    def __rtruediv__(self, other: Any) -> Dual: ...
    def __neg__(self) -> Dual: ...
    def __pow__(self, other: Any) -> Dual:
        """x ** k for a constant k: k x^(k-1) tangent. x ** y for a Dual y:
        d(x^y) = x^y (y' ln x + y x' / x)."""
    def __rpow__(self, other: Any) -> Dual:
        """c ** x for a constant c > 0: c^x ln(c) tangent."""
    def __matmul__(self, other: Any) -> Dual:
        """(X + T eps) @ B = X @ B + (T @ B) eps; a Dual right operand by the
        product rule."""
    def __rmatmul__(self, other: Any) -> Dual:
        """W @ (x + t eps) = W @ x + (W @ t) eps."""
    def __getitem__(self, idx: Any) -> Dual:
        """Indexing and slicing apply to val and eps alike."""

def dual_exp(x: Any) -> Dual:
    """exp(a) + exp(a) b eps. A non-Dual argument is a constant."""

def dual_log(x: Any) -> Dual:
    """log(a) + (b / a) eps."""

def dual_tanh(x: Any) -> Dual:
    """tanh(a) + (1 - tanh(a)^2) b eps."""

def dual_erf(x: Any) -> Dual:
    """erf(a) + (2 / sqrt(pi)) exp(-a^2) b eps. The value uses math.erf,
    elementwise for arrays (numpy has no erf)."""

def derivative(f: Callable[[Dual], Any], x: float) -> float:
    """f'(x) by one forward pass: f(Dual(x, 1.0)).eps as a float. A result
    that is not a Dual (f ignores its input) has derivative 0.0."""

def jvp(f: Callable[[Dual], Any], x: ArrayLike, v: ArrayLike) -> tuple[NDArray, NDArray]:
    """(f(x), J_f(x) v) for f from float64 arrays to float64 arrays, by one
    forward pass on Dual(x, v). Both results are float64 arrays of f(x)'s
    shape; a result that is not a Dual has tangent zeros."""
