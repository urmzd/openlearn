# contracts/py/tinyllm/autograd/scalar.pyi (M08.2): scalar reverse-mode autodiff
# chapter: content/courses/mathematics-for-computing/09-matrix-calculus-and-autodiff/02-scalar-reverse-mode.md
#
# Each Value is one node of a computation graph: a float `data`, the
# gradient `grad` of the final output with respect to it, the nodes it was
# computed from (`_prev`), and a closure `_backward` that pushes its own
# grad into theirs by the chain rule:
#
#     for each input x of the node y:   x.grad += dy/dx * y.grad
#
# A Python number in an operation is a constant leaf (a new Value).
from typing import Callable

class Value:
    data: float
    grad: float  # 0.0 until backward() reaches the node
    _prev: tuple[Value, ...]  # the inputs this node was computed from
    _op: str  # '+', '*', '**', 'exp', ... ('' for a leaf); for printing only
    _backward: Callable[[], None]  # a leaf's does nothing

    def __init__(self, data: float, _children: tuple = (), _op: str = "") -> None: ...
    def __add__(self, other: Value | float) -> Value: ...
    def __radd__(self, other: float) -> Value: ...
    def __sub__(self, other: Value | float) -> Value: ...
    def __rsub__(self, other: float) -> Value: ...
    def __mul__(self, other: Value | float) -> Value: ...
    def __rmul__(self, other: float) -> Value: ...
    def __truediv__(self, other: Value | float) -> Value: ...
    def __rtruediv__(self, other: float) -> Value: ...
    def __neg__(self) -> Value: ...
    def __pow__(self, k: float) -> Value:
        """x ** k for a constant int or float k (not a Value): TypeError otherwise."""
    def exp(self) -> Value: ...
    def log(self) -> Value: ...
    def tanh(self) -> Value: ...
    def relu(self) -> Value:
        """max(x, 0); its gradient at exactly 0 is 0 (as in PyTorch)."""
    def backward(self) -> None:
        """Set self.grad = 1.0, then call _backward on every node reachable
        through _prev exactly once, each node before the nodes it was
        computed from: the order of M06.1's
        toposort(self, lambda v: v._prev). Gradients accumulate (+=) into
        .grad; nodes reachable from self are not reset first, so calling
        backward twice adds the gradients twice. Never recursive: a chain of
        10^5 nodes must not raise RecursionError."""
