# contracts/py/tinyllm/num/activations.pyi (M01.3)
# chapter: content/courses/mathematics-for-computing/01-calculus-1/03-activation-functions.md
#
# Elementwise activation functions and their derivatives, NDArray -> NDArray.
# Every function accepts an array-like x and returns an array of the same
# shape. A float32 or float64 input keeps its dtype; any other input is
# computed in float64. No function emits a floating-point warning for any
# finite input (overflow is avoided, not silenced), and every result is
# finite for finite input. Each d<name>(x) is the derivative of <name> at x.
# Conventions (PyTorch's): drelu(0) = 0; gelu_tanh is the tanh approximation
# 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x**3))); gelu_erf is x * Phi(x)
# with Phi(x) = (1 + erf(x / sqrt(2))) / 2, erf from tinyllm.num.taylor.erf_series
# (M02.1); softplus is log(1 + e**x) with no threshold, and its derivative,
# dsoftplus, is sigmoid.
from numpy.typing import ArrayLike, NDArray

def sigmoid(x: ArrayLike) -> NDArray:
    """1 / (1 + exp(-x)), evaluated without overflow for either sign."""

def dsigmoid(x: ArrayLike) -> NDArray:
    """sigmoid(x) * sigmoid(-x); keeps full relative accuracy for large |x|."""

def tanh(x: ArrayLike) -> NDArray:
    """(e**x - e**-x) / (e**x + e**-x)."""

def dtanh(x: ArrayLike) -> NDArray:
    """1 - tanh(x)**2."""

def relu(x: ArrayLike) -> NDArray:
    """max(x, 0)."""

def drelu(x: ArrayLike) -> NDArray:
    """1 where x > 0, else 0 (including at x = 0)."""

def softplus(x: ArrayLike) -> NDArray:
    """log(1 + exp(x)) = max(x, 0) + log1p(exp(-|x|)): no overflow at x = 1000,
    full relative accuracy at x = -40."""

def dsoftplus(x: ArrayLike) -> NDArray:
    """Derivative of softplus: sigmoid(x)."""

def gelu_tanh(x: ArrayLike) -> NDArray:
    """0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x**3)))."""

def dgelu_tanh(x: ArrayLike) -> NDArray:
    """Derivative of gelu_tanh."""

def gelu_erf(x: ArrayLike) -> NDArray:
    """x * Phi(x), Phi the standard normal CDF, through erf_series (M02.1)."""

def dgelu_erf(x: ArrayLike) -> NDArray:
    """Phi(x) + x * phi(x), phi(x) = exp(-x**2 / 2) / sqrt(2 pi)."""

def silu(x: ArrayLike) -> NDArray:
    """x * sigmoid(x) (also called swish)."""

def dsilu(x: ArrayLike) -> NDArray:
    """sigmoid(x) * (1 + x * sigmoid(-x))."""
