# contracts/py/tinyllm/num/gradcheck.pyi (M04.1)
# chapter: content/courses/mathematics-for-computing/04-calculus-3/01-gradients-and-gradcheck.md
#
# Numerical gradients of a scalar function of several arrays, and the check
# that compares them with an analytic gradient. Later modules (L0.2 on) use
# it in their tests to check hand-written backward passes.
from dataclasses import dataclass
from typing import Callable

from numpy.typing import ArrayLike, NDArray

@dataclass
class GradcheckReport:
    ok: bool  # every element passes |a - n| <= atol + rtol * |n|
    max_abs_err: float  # max |a - n| over every element of every input
    max_rel_err: float  # max |a - n| / max(|n|, atol) over every element
    worst_input: int  # which input holds the worst element ...
    worst_index: tuple  # ... and where: the largest |a - n| / (atol + rtol * |n|),
    # the first such element in input order then row-major order on ties

def numerical_grad(
    f: Callable[..., float], inputs: list[ArrayLike], eps: float = 1e-6
) -> list[NDArray]:
    """Central-difference gradient of the scalar f(*inputs) with respect to
    every element of every input: one tinyllm.num.diff.central_diff (M01.1)
    per element, with step eps, on float64 copies of the inputs (integer and
    float32 inputs are promoted). Returns one float64 array per input, in the
    input's shape. The caller's arrays are never modified, and f always sees
    float64 arrays. Costs 2 * (total number of elements) calls of f.
    ValueError when inputs is empty, eps is not finite or <= 0, or f returns
    more than one number (an array with more than one element)."""

def gradcheck(
    f: Callable[..., float],
    inputs: list[ArrayLike],
    analytic: list[ArrayLike],
    eps: float = 1e-6,
    rtol: float = 1e-5,
    atol: float = 1e-7,
) -> GradcheckReport:
    """Compare analytic[k] with numerical_grad(f, inputs, eps)[k] for every k,
    elementwise: an element passes when |a - n| <= atol + rtol * |n|, and the
    report is ok when every element of every input passes. A nan in either
    gradient fails that element.
    ValueError when len(analytic) != len(inputs), any analytic[k] has a shape
    other than inputs[k]'s, or rtol or atol is negative."""
