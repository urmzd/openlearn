# contracts/py/tinyllm/nn/module.pyi (L0.4): the module system
# chapter: content/courses/build-an-llm/01-foundations/05-module-system-and-layers.md
#
# A Module owns parameters (Tensors with requires_grad, L0.1) and child
# Modules, both found by assigning them as attributes in __init__ (after
# super().__init__()). Registration order is assignment order, and the
# dotted names it produces ("0.weight", "attn.q_proj.weight") ARE the
# safetensors key contract: a checkpoint written from state_dict() loads into
# any implementation that registers the same attributes in the same order,
# and PyTorch's names for the same structure are identical.
#
#   parameter  an attribute holding a Tensor with requires_grad True; a Tensor
#              without the flag is plain state and is not registered
#   child      an attribute holding a Module
#   tied       one Tensor registered under two names (weight tying): it is
#              listed once, under its first name
from typing import Any, Iterator, Mapping

from numpy.typing import NDArray

from tinyllm.autograd.tensor import Tensor

class Module:
    training: bool

    def __init__(self) -> None:
        """Start empty and in training mode. A subclass must call it before
        assigning any attribute (AttributeError otherwise, naming the fix)."""

    def __setattr__(self, name: str, value: Any) -> None:
        """Register a parameter or a child under `name` (re-assigning keeps
        its position), then set the attribute."""

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """The computation; subclasses override it. NotImplementedError here."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """forward(*args, **kwargs)."""

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Tensor]]:
        """(dotted name, parameter) for own parameters in registration order,
        then each child's (prefixed "child."), depth first; a tied Tensor once."""

    def parameters(self) -> Iterator[Tensor]:
        """The parameters of named_parameters(), in that order."""

    def named_modules(self, prefix: str = "") -> Iterator[tuple[str, "Module"]]:
        """("", self), then every descendant with its dotted name, depth first."""

    def train(self, mode: bool = True) -> "Module":
        """Set `training` on self and every descendant; returns self."""

    def eval(self) -> "Module":
        """train(False)."""

    def zero_grad(self) -> None:
        """Set .grad = None on every parameter."""

    def state_dict(self) -> dict[str, NDArray]:
        """{dotted name: a copy of the parameter's data}, in named_parameters order."""

    def load_state_dict(self, sd: Mapping[str, Any], strict: bool = True) -> None:
        """Copy each array into the parameter of the same name IN PLACE (the
        Tensor objects and their data arrays stay the ones an optimizer
        holds), converting to the parameter's dtype. ValueError for a shape
        mismatch. With strict, KeyError naming every missing and unexpected
        key, before anything is copied."""
