# contracts/py/tinyllm/optim/sgd.pyi (M10.2): the Optimizer protocol and SGD
# chapter: content/tracks/math/10-optimization/02-optimizer-protocol-and-sgd.md
#
# A parameter is any object with a float ndarray `data` and a `grad` that
# is an ndarray of the same shape or None (no gradient this step). The
# course's Tensor (L0.1) is one. Every optimizer of the course (SGD here,
# AdamW in M10.3, Muon in M10.6) implements the Optimizer protocol, so the
# training loops of L0.5, L2.2, and L3.6 never name a concrete optimizer.
from typing import Any, Iterable, Protocol, runtime_checkable

from numpy.typing import NDArray

@runtime_checkable
class Param(Protocol):
    data: NDArray
    grad: NDArray | None

@runtime_checkable
class Optimizer(Protocol):
    def step(self) -> None:
        """Update every parameter's data in place from its grad."""
    def zero_grad(self) -> None:
        """Set every parameter's grad to None (PyTorch's set_to_none)."""
    def state_dict(self) -> dict:
        """Everything needed to resume bitwise, as a deep copy (arrays copied)."""
    def load_state_dict(self, sd: dict) -> None:
        """Restore from state_dict() of an optimizer over the same parameters
        (same count and shapes; ValueError otherwise). Arrays are copied in."""

class SGD:
    """Stochastic gradient descent with momentum, Nesterov, and (coupled)
    weight decay, with PyTorch's torch.optim.SGD semantics (dampening 0).
    For each parameter p whose grad is not None, in order:

        g = grad + weight_decay * data                 (if weight_decay != 0)
        if momentum != 0:
            buf = g (a copy) on the first step, else buf = momentum * buf + g
            g = g + momentum * buf  if nesterov  else  buf
        data -= lr * g                                 (in place, data's dtype)

    A parameter with grad None is skipped entirely (no decay, no buffer
    update). The momentum buffer belongs to one parameter."""

    params: list[Any]
    lr: float
    momentum: float
    nesterov: bool
    weight_decay: float

    def __init__(
        self,
        params: Iterable[Any],
        lr: float,
        momentum: float = 0.0,
        nesterov: bool = False,
        weight_decay: float = 0.0,
    ) -> None:
        """ValueError when lr < 0, momentum < 0, weight_decay < 0, or
        nesterov is set with momentum == 0."""
    def step(self) -> None: ...
    def zero_grad(self) -> None: ...
    def state_dict(self) -> dict:
        """{"state": {i: {"momentum_buffer": ndarray}} for each parameter
        index i that has a buffer, "param_groups": [{"lr", "momentum",
        "nesterov", "weight_decay", "params": [0, 1, ...]}]}, as copies."""
    def load_state_dict(self, sd: dict) -> None: ...
