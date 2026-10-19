# contracts/py/tinyllm/autograd/functional.pyi (L0.2): the op library, imported as F
# chapter: content/courses/build-an-llm/01-foundations/03-op-library-and-gradcheck.md
#
#     from tinyllm.autograd import functional as F
#
# (Written `F` in prose; `tinyllm` is a namespace package with no
# __init__.py, so the module is imported by its path.)
#
# Every op takes Tensors (L0.1); an ndarray or a number where a Tensor is
# expected is a constant. Every op is built on tensor.from_op: one forward in
# numpy and one vector-Jacobian product (vjp). Forward values follow numpy
# and PyTorch (the golden tests compare with torch 2.14 forward and backward);
# float32 stays float32. `axis` is an int (negative counts from the end) or,
# for the reductions, also None or a tuple. Masked entries are -inf.
#
# Gradient conventions where the function has a kink:
#   relu'(0) = 0; max sends the gradient to every maximal entry in equal
#   shares (torch.amax); a fully masked softmax row has gradient 0.
from typing import Any, Literal, Optional, Sequence, Union

from numpy.typing import ArrayLike, NDArray

from tinyllm.autograd.tensor import Tensor
from tinyllm.num.gradcheck import GradcheckReport

Axis = Union[int, tuple[int, ...], None]

# -- elementwise (forward and derivative from tinyllm.num.activations, M01.3) --
def exp(x: Tensor) -> Tensor: ...
def log(x: Tensor) -> Tensor:
    """Natural log; the vjp is g / x."""
def tanh(x: Tensor) -> Tensor: ...
def sigmoid(x: Tensor) -> Tensor: ...
def relu(x: Tensor) -> Tensor: ...
def silu(x: Tensor) -> Tensor: ...
def gelu(x: Tensor, approximate: Literal["none", "tanh"] = "none") -> Tensor:
    """x * Phi(x) ("none", the erf form) or the tanh approximation, as torch's
    gelu(approximate=...). ValueError for another approximate."""

# -- reductions ------------------------------------------------------------------
def sum(x: Tensor, axis: Axis = None, keepdims: bool = False) -> Tensor: ...
def mean(x: Tensor, axis: Axis = None, keepdims: bool = False) -> Tensor: ...
def max(x: Tensor, axis: Axis = None, keepdims: bool = False) -> Tensor:
    """Maximum (torch.amax): ties share the gradient equally."""
def var(x: Tensor, axis: Axis = None, keepdims: bool = False, correction: int = 0) -> Tensor:
    """sum((x - mean)^2) / (n - correction): correction 0 is the population
    variance (LayerNorm), 1 the sample variance. ValueError when n <= correction."""

# -- shape ---------------------------------------------------------------------------
def reshape(x: Tensor, shape: Sequence[int]) -> Tensor:
    """numpy reshape (one dimension may be -1)."""
def transpose(x: Tensor, a: int, b: int) -> Tensor:
    """Swap axes a and b."""
def permute(x: Tensor, dims: Sequence[int]) -> Tensor:
    """Reorder every axis: output axis i is input axis dims[i]."""
def concat(xs: Sequence[Tensor], axis: int = 0) -> Tensor: ...
def stack(xs: Sequence[Tensor], axis: int = 0) -> Tensor: ...

# -- selection -------------------------------------------------------------------------
def where(cond: ArrayLike, a: Any, b: Any) -> Tensor:
    """a where cond is true, else b; cond, a, b broadcast (numpy.where).
    a and b may be Tensors or numbers; each gets its gradient where it was chosen."""
def gather(x: Tensor, idx: ArrayLike, axis: int) -> Tensor:
    """torch.gather: out[..., i, ...] = x[..., idx[..., i, ...], ...] along axis
    (numpy.take_along_axis). idx is an integer array with x's number of
    dimensions. Repeated indices add their gradients. ValueError for an index
    outside [0, x.shape[axis]) or a non-integer idx."""
def embedding(weight: Tensor, ids: ArrayLike) -> Tensor:
    """weight[ids]: rows of a [n, d] table, output shape ids.shape + (d,).
    A row used several times gets the sum of its gradients. ValueError for an
    id outside [0, n) (numpy would wrap -1 silently) or a non-integer ids."""
def masked_fill(x: Tensor, mask: ArrayLike, value: float) -> Tensor:
    """x with value wherever mask (boolean, broadcastable to x.shape) is true;
    those positions pass no gradient back to x."""

# -- softmax family (forward from tinyllm.num.stable, M09.2; vjps from tinyllm.autograd.vjp, M08.3) --
def softmax(x: Tensor, axis: int = -1) -> Tensor: ...
def log_softmax(x: Tensor, axis: int = -1) -> Tensor: ...
def logsumexp(x: Tensor, axis: int = -1, keepdims: bool = False) -> Tensor: ...

# -- the rest ---------------------------------------------------------------------------------
def dropout(x: Tensor, p: float, training: bool, rng: Any) -> Tensor:
    """Inverted dropout. When training and p > 0: draw u = rng.uniforms(x.size)
    (one uniform per element, C order, from a PCG32, M06.3), keep where
    u >= p, and scale kept values by 1 / (1 - p); p = 1 gives zeros. When not
    training, or p == 0: returns x itself and draws nothing.
    ValueError unless 0 <= p <= 1."""
def matmul(a: Tensor, b: Tensor) -> Tensor:
    """a @ b (L0.1's Tensor matmul)."""

def gradcheck_all(rtol: float = 1e-5) -> dict[str, GradcheckReport]:
    """Check every op above (and the Tensor arithmetic: add, sub, mul, div, pow,
    neg, getitem, matmul) on small float64 inputs drawn from PCG32(0): the
    backward of sum(op(inputs) * w), for a random weight w, against
    central differences through tinyllm.num.gradcheck.gradcheck (M04.1)
    with this rtol. Elementwise ops are also checked at a few points against
    forward-mode derivatives (tinyllm.autograd.dual.derivative, M08.1); a
    disagreement above 1e-9 makes their report not ok. Ops are looked up
    in this module when it runs, so a replaced op is the one checked.
    Keys are the op names; `{tinyllm} gradcheck --suite all` prints them."""
