"""Basic layers with PyTorch's names and shapes (L0.4).

Every forward is a composition of op-library calls (L0.2), so backward needs
no code here. Initial weights come from M07.3's initializers, drawn from a
PCG32 (M06.3) in parameter registration order.

Contract: contracts/py/tinyllm/nn/layers.pyi.
"""

from __future__ import annotations

import math
from typing import Any, Iterator, Literal, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike

from tinyllm.autograd import functional as F
from tinyllm.autograd.tensor import Tensor
from tinyllm.nn.init import normal_init
from tinyllm.nn.module import Module
from tinyllm.num.rng import PCG32


def _rng(rng: Any, purpose: str) -> Any:
    """rng, or the default stream for `purpose` (spec/pcg32.md sub-streams)."""
    return rng if rng is not None else PCG32(0).substream(purpose)


def _param(a: ArrayLike) -> Tensor:
    return Tensor(np.asarray(a, dtype=np.float32), requires_grad=True)


class Linear(Module):
    def __init__(self, in_f: int, out_f: int, bias: bool = True, rng: Any = None) -> None:
        super().__init__()
        if in_f < 1 or out_f < 1:
            raise ValueError(f"Linear({in_f}, {out_f}): sizes must be positive")
        r = _rng(rng, "init")
        # torch's layout: one row per output, so y = x @ W^T.
        self.weight = _param(normal_init((out_f, in_f), 1.0 / math.sqrt(in_f), r))
        self.bias = _param(np.zeros(out_f)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        y = F.matmul(x, F.transpose(self.weight, 0, 1))
        return y + self.bias if self.bias is not None else y


class Embedding(Module):
    def __init__(self, n: int, d: int, rng: Any = None) -> None:
        super().__init__()
        self.weight = _param(normal_init((n, d), 1.0, _rng(rng, "init")))

    def forward(self, ids: ArrayLike) -> Tensor:
        return F.embedding(self.weight, ids)


class LayerNorm(Module):
    def __init__(self, d: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = _param(np.ones(d))
        self.bias = _param(np.zeros(d))

    def forward(self, x: Tensor) -> Tensor:
        mu = F.mean(x, axis=-1, keepdims=True)
        var = F.var(x, axis=-1, keepdims=True)  # population variance, as torch
        xhat = (x - mu) * (var + self.eps) ** -0.5
        return xhat * self.weight + self.bias


class Dropout(Module):
    def __init__(self, p: float, rng: Any = None) -> None:
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"Dropout p must be in [0, 1], got {p}")
        self.p = float(p)
        self.rng = _rng(rng, "dropout")

    def forward(self, x: Tensor) -> Tensor:
        return F.dropout(x, self.p, self.training, self.rng)


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.relu(x)


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.tanh(x)


class GELU(Module):
    def __init__(self, approximate: Literal["none", "tanh"] = "none") -> None:
        super().__init__()
        if approximate not in ("none", "tanh"):
            raise ValueError(f"approximate must be 'none' or 'tanh', got {approximate!r}")
        self.approximate = approximate

    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(x, approximate=self.approximate)


class Sequential(Module):
    def __init__(self, *mods: Module) -> None:
        super().__init__()
        for i, m in enumerate(mods):
            if not isinstance(m, Module):
                raise TypeError(f"Sequential item {i} is {type(m).__name__}, not a Module")
            setattr(self, str(i), m)  # names "0", "1", ...: torch's state_dict keys

    def forward(self, x: Any) -> Any:
        for m in self._children.values():
            x = m(x)
        return x

    def __getitem__(self, i: int) -> Module:
        return list(self._children.values())[i]

    def __len__(self) -> int:
        return len(self._children)

    def __iter__(self) -> Iterator[Module]:
        return iter(list(self._children.values()))


class ModuleList(Module):
    def __init__(self, mods: Sequence[Module] = ()) -> None:
        super().__init__()
        for m in mods:
            self.append(m)

    def append(self, m: Module) -> "ModuleList":
        if not isinstance(m, Module):
            raise TypeError(f"ModuleList holds Modules, got {type(m).__name__}")
        setattr(self, str(len(self._children)), m)
        return self

    def __getitem__(self, i: int) -> Module:
        return list(self._children.values())[i]

    def __len__(self) -> int:
        return len(self._children)

    def __iter__(self) -> Iterator[Module]:
        return iter(list(self._children.values()))
