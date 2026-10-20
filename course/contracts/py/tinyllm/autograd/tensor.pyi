# contracts/py/tinyllm/autograd/tensor.pyi (L0.1): the autograd Tensor
# chapter: content/courses/ai-engineering/06-numerics-and-autograd/02-tensor-and-broadcasting-backward.md
#
# A Tensor is a numpy array plus the record of how it was computed. An op on
# tensors that require grad returns a new tensor holding its inputs
# (`parents`) and a vector-Jacobian product (`vjp`): given the gradient of a
# scalar loss with respect to the op's output, it returns the gradient with
# respect to each input. `backward()` walks the graph from the loss to the
# leaves in reverse topological order (tinyllm.autograd.graph.toposort, M06.1)
# and calls each vjp once.
#
# Words used below:
#   leaf      a tensor made by the constructor (not by an op); only leaves
#             with requires_grad keep a `.grad` after backward
#   constant  a tensor with requires_grad False, or any non-Tensor operand
#             (a Python number or an ndarray), converted to the other
#             operand's dtype
#   broadcast numpy broadcasting (lang.01). An input that was broadcast gets
#             its gradient summed back to its own shape
#             (tinyllm.autograd.vjp.unbroadcast, M08.3)
from typing import Any, Callable, Optional, Sequence

from numpy.typing import ArrayLike, DTypeLike, NDArray

class Tensor:
    data: NDArray  # the values; float16, float32, or float64
    grad: Optional[NDArray]  # leaves only: d loss / d data, same shape and dtype as data
    requires_grad: bool

    def __init__(
        self, data: ArrayLike, requires_grad: bool = False, dtype: DTypeLike = ...
    ) -> None:
        """A leaf holding a copy of `data` as `dtype` (default float32).
        ValueError when dtype is not a floating-point type."""

    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def dtype(self) -> Any: ...
    @property
    def ndim(self) -> int: ...
    def __len__(self) -> int: ...
    def backward(self, grad: Optional[ArrayLike] = None) -> None:
        """Accumulate d self / d leaf into every reachable leaf's `.grad`
        (added to what is already there; set `.grad = None` to clear).
        `grad` is the upstream gradient, the shape of self; None means 1 and
        needs a one-element tensor. Each node's gradient is the sum over every
        path from self, so a tensor used twice gets both contributions.
        RuntimeError when self does not require grad, when grad is None for a
        tensor with more than one element, or when an op's vjp returns a
        gradient whose shape differs from its input's; ValueError when grad's
        shape differs from self's."""

    def detach(self) -> "Tensor":
        """A constant sharing this tensor's data: nothing flows back through it."""

    def numpy(self) -> NDArray:
        """The data array itself (not a copy)."""

    # Arithmetic: numpy semantics and broadcasting, gradients for every operand
    # that requires grad. Numbers and ndarrays mix in on either side
    # (ndarray + Tensor returns a Tensor: numpy defers to Tensor).
    def __add__(self, other: Any) -> "Tensor": ...
    def __radd__(self, other: Any) -> "Tensor": ...
    def __sub__(self, other: Any) -> "Tensor": ...
    def __rsub__(self, other: Any) -> "Tensor": ...
    def __mul__(self, other: Any) -> "Tensor": ...
    def __rmul__(self, other: Any) -> "Tensor": ...
    def __truediv__(self, other: Any) -> "Tensor": ...
    def __rtruediv__(self, other: Any) -> "Tensor": ...
    def __matmul__(self, other: Any) -> "Tensor":
        """numpy matmul: 1-D and N-D operands, batch dims broadcast."""
    def __rmatmul__(self, other: Any) -> "Tensor": ...
    def __pow__(self, exponent: float) -> "Tensor":
        """Elementwise power by a Python number. TypeError for a Tensor exponent."""
    def __neg__(self) -> "Tensor": ...
    def __getitem__(self, index: Any) -> "Tensor":
        """numpy indexing: ints, slices, None, Ellipsis, integer arrays
        (repeated indices accumulate their gradients), boolean masks."""

def from_op(
    data: ArrayLike,
    parents: Sequence[Tensor],
    vjp: Callable[[NDArray], Sequence[Optional[NDArray]]],
    op: str = "",
) -> Tensor:
    """The output of an op (the seam the op library, L0.2, builds on).
    It requires grad when grad mode is on (tinyllm.autograd.mode) and some
    parent requires grad; only then does it keep `parents` and `vjp`.
    `vjp(g)` gets the upstream gradient (the shape of `data`) and returns one
    gradient (or None) per parent, each the shape of that parent; `op` names
    the op in error messages."""
