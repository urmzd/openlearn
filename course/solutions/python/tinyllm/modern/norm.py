"""RMSNorm and the Pre-LN residual step (L7.1).

RMSNorm rescales each vector to unit root mean square, then applies a
learned per-feature gain. It drops LayerNorm's mean subtraction and bias:
cheaper, and in practice just as stable. Pre-LN puts the norm on the branch
input only, so the residual stream carries the identity from the embedding
to the last layer.

Contract: contracts/py/tinyllm/modern/norm.pyi.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from tinyllm.autograd import functional as F
from tinyllm.autograd.tensor import Tensor
from tinyllm.nn.module import Module


class RMSNorm(Module):
    def __init__(self, d: int, eps: float = 1e-6, offset: float = 0.0) -> None:
        super().__init__()
        if isinstance(d, bool) or int(d) != d or d < 1:
            raise ValueError(f"d must be a positive integer, got {d!r}")
        if not eps >= 0:
            raise ValueError(f"eps must be >= 0, got {eps!r}")
        self.d = int(d)
        self.eps = float(eps)
        self.offset = float(offset)
        # Llama starts the gain at 1; Gemma stores gain - 1 and starts it at 0.
        init = np.ones(self.d) if self.offset == 0.0 else np.zeros(self.d)
        self.weight = Tensor(init.astype(np.float32), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.d:
            raise ValueError(f"RMSNorm({self.d}) got last axis {x.shape[-1]}")
        # eps goes inside the root: sqrt(mean(x^2) + eps), as Llama and T5.
        ms = F.mean(x * x, axis=-1, keepdims=True)
        y = x * (ms + self.eps) ** -0.5
        gain = self.weight + self.offset if self.offset != 0.0 else self.weight
        return y * gain


def pre_norm_residual(x: Tensor, norm: Module, sublayer: Callable[[Tensor], Tensor]) -> Tensor:
    # The residual stream x is never normalized: only the branch sees norm(x).
    return x + sublayer(norm(x))
