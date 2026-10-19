# contracts/py/tinyllm/num/stable.pyi (M09.2): stable reductions over floats
# chapter: content/courses/mathematics-for-computing/10-numerical-methods-and-floating-point/02-stable-numerics.md
#
# Every function takes a numpy array (anything np.asarray accepts). Float
# inputs keep their dtype (float32 in, float32 out); integer inputs are
# computed in float64. `axis` follows numpy (negative counts from the end).
#
# Masked entries are -inf. A row whose entries are all -inf (a fully masked
# row) is defined, never NaN: logsumexp = -inf, softmax = zeros,
# log_softmax = -inf. NaN in a row propagates to that row.
from numpy.typing import ArrayLike, NDArray

def logsumexp(x: ArrayLike, axis: int = -1, keepdims: bool = False) -> NDArray:
    """log(sum(exp(x), axis)) computed as m + log(sum(exp(x - m), axis)) with
    m = max(x, axis), so no exp overflows. Finite for finite inputs of any
    size (x = [1e4, 1e4] gives 1e4 + ln 2). keepdims as in numpy."""

def softmax(x: ArrayLike, axis: int = -1) -> NDArray:
    """exp(x - m) / sum(exp(x - m)) along axis, m = max(x, axis).
    Same shape as x; every unmasked row sums to 1; invariant to adding a
    constant to a row; a fully masked row gives zeros."""

def log_softmax(x: ArrayLike, axis: int = -1) -> NDArray:
    """x - logsumexp(x, axis, keepdims=True): finite wherever x is finite,
    even where softmax underflows to 0 (x = [0, -1e4] gives [0, -1e4]).
    A fully masked row gives -inf."""

def kahan_sum(x: ArrayLike) -> float:
    """Sum of every element of x (flattened in C order), left to right with
    Kahan compensation, every operation in x's float dtype (float64 for
    integer input). Returned as a Python float. The error is at most about
    2 u |sum| + n u^2 sum|x_i| (u = unit roundoff of the dtype), independent
    of n to first order. An empty array sums to 0.0."""

def pairwise_sum(x: ArrayLike) -> float:
    """Sum of every element of x (flattened in C order) by pairwise
    (cascade) summation down to single elements, in x's float dtype: split
    the n elements at h = n // 2, sum x[:h] and x[h:] the same way, add the
    two results. (Pairing neighbours bottom up builds the same tree when n is
    a power of two.) The error is at most about ceil(log2 n) u sum|x_i|.
    An empty array sums to 0.0; one element sums to itself."""
