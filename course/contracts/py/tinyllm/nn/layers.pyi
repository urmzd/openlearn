# contracts/py/tinyllm/nn/layers.pyi (L0.4): basic layers
# chapter: content/courses/ai-engineering/06-numerics-and-autograd/05-module-system-and-layers.md
#
# Parameter names and shapes are PyTorch's (torch.nn), so HF checkpoints map
# onto these layers by name. Every forward is built from the op library
# (tinyllm.autograd.functional, L0.2), so backward comes for free.
#
# `rng` is a PCG32 (M06.3) used once, at construction, in parameter
# registration order; None means PCG32(0).substream("init") for the
# initializers and PCG32(0).substream("dropout") for Dropout (spec/pcg32.md).
# Initializers come from tinyllm.nn.init (M07.3). Parameters are float32.
from typing import Any, Iterator, Literal, Optional, Sequence

from numpy.typing import ArrayLike

from tinyllm.autograd.tensor import Tensor
from tinyllm.nn.module import Module

class Linear(Module):
    weight: Tensor  # [out_f, in_f], normal_init(std = 1 / sqrt(in_f))
    bias: Optional[Tensor]  # [out_f], zeros; None when bias=False

    def __init__(self, in_f: int, out_f: int, bias: bool = True, rng: Any = None) -> None: ...
    def forward(self, x: Tensor) -> Tensor:
        """x @ weight^T + bias over the last axis: [..., in_f] -> [..., out_f]."""

class Embedding(Module):
    weight: Tensor  # [n, d], normal_init(std = 1)

    def __init__(self, n: int, d: int, rng: Any = None) -> None: ...
    def forward(self, ids: ArrayLike) -> Tensor:
        """Rows of weight: ids.shape + (d,) (F.embedding)."""

class LayerNorm(Module):
    weight: Tensor  # [d], ones
    bias: Tensor  # [d], zeros

    def __init__(self, d: int, eps: float = 1e-5) -> None: ...
    def forward(self, x: Tensor) -> Tensor:
        """(x - mean) / sqrt(var + eps) * weight + bias over the last axis,
        var the population variance (correction 0), as torch.nn.LayerNorm."""

class Dropout(Module):
    def __init__(self, p: float, rng: Any = None) -> None:
        """ValueError unless 0 <= p <= 1."""
    def forward(self, x: Tensor) -> Tensor:
        """F.dropout(x, p, self.training, rng): the identity in eval mode."""

class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor: ...

class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor: ...

class GELU(Module):
    def __init__(self, approximate: Literal["none", "tanh"] = "none") -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...

class Sequential(Module):
    def __init__(self, *mods: Module) -> None:
        """Children named "0", "1", ... in order (torch.nn.Sequential)."""
    def forward(self, x: Any) -> Any:
        """Each child's forward applied in order."""
    def __getitem__(self, i: int) -> Module: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[Module]: ...

class ModuleList(Module):
    def __init__(self, mods: Sequence[Module] = ()) -> None:
        """Children named "0", "1", ...; no forward of its own."""
    def append(self, m: Module) -> "ModuleList": ...
    def __getitem__(self, i: int) -> Module: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[Module]: ...
