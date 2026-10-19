"""Positional encodings: sinusoidal and learned (L5.4).

Attention is permutation equivariant: without a position signal the
transformer cannot tell "12+34" from "21+43". Vaswani et al. add a fixed
vector PE(p) to the embedding at position p, whose pairs of entries are a
point (sin, cos) on a circle turning at a geometric ladder of frequencies;
GPT-2 learns the table instead.

Contract: contracts/py/tinyllm/xfmr/pos.pyi.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tinyllm.autograd import functional as F
from tinyllm.autograd.tensor import Tensor
from tinyllm.nn.init import normal_init
from tinyllm.nn.module import Module
from tinyllm.num.rng import PCG32
from tinyllm.num.rotation import rotate_pairs
from tinyllm.num.series import geometric


def _check_d(d: int) -> None:
    if not isinstance(d, (int, np.integer)) or d < 2 or d % 2:
        raise ValueError(f"d must be a positive even integer, got {d}")


def sinusoidal_freqs(d: int, base: float = 10000.0) -> NDArray:
    _check_d(d)
    if not (np.isfinite(base) and base > 1.0):
        raise ValueError(f"base must be finite and > 1, got {base}")
    # omega_i = base^(-2i/d): first term 1, ratio base^(-2/d) (M00.3).
    return geometric(1.0, float(base) ** (-2.0 / d), d // 2)


def sinusoidal_pe(T: int, d: int, base: float = 10000.0, offset: int = 0) -> NDArray:
    if T < 0 or offset < 0:
        raise ValueError(f"T and offset must be non-negative, got {T}, {offset}")
    w = sinusoidal_freqs(d, base)
    # Angles in float64: at p = 10^4 a float32 angle is off by about 5e-4 rad.
    ang = np.arange(offset, offset + T, dtype=np.float64)[:, None] * w[None, :]
    pe = np.empty((T, d), dtype=np.float64)
    pe[:, 0::2] = np.sin(ang)  # PE(p, 2i)
    pe[:, 1::2] = np.cos(ang)  # PE(p, 2i + 1)
    return pe.astype(np.float32)


def shift_pe(pe: ArrayLike, k: float, d: int, base: float = 10000.0) -> NDArray:
    x = np.asarray(pe)
    if x.shape[-1] != d:
        raise ValueError(f"pe has width {x.shape[-1]}, expected d = {d}")
    # The pair (sin a, cos a) turned by -k*omega is (sin(a + k omega),
    # cos(a + k omega)): a rotation (M00.2) that depends on k, not on p.
    return rotate_pairs(x, -float(k) * sinusoidal_freqs(d, base))


class SinusoidalPE(Module):
    def __init__(self, max_len: int, d: int, base: float = 10000.0) -> None:
        super().__init__()
        if max_len < 1:
            raise ValueError(f"max_len must be positive, got {max_len}")
        self.max_len, self.d, self.base = max_len, d, float(base)
        # A constant, not a parameter: no gradient, not in the state_dict.
        self.table = sinusoidal_pe(max_len, d, base)

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        T = x.shape[-2]
        if x.shape[-1] != self.d:
            raise ValueError(f"x has width {x.shape[-1]}, expected {self.d}")
        if offset < 0 or offset + T > self.max_len:
            raise ValueError(
                f"positions {offset}..{offset + T - 1} exceed max_len {self.max_len}"
            )
        return x + self.table[offset : offset + T].astype(x.dtype)


class LearnedPE(Module):
    def __init__(
        self, max_len: int, d: int, std: float = 0.02, rng: Any = None
    ) -> None:
        super().__init__()
        if max_len < 1 or d < 1:
            raise ValueError(f"max_len and d must be positive, got {max_len}, {d}")
        r = rng if rng is not None else PCG32(0).substream("init")
        self.max_len, self.d = max_len, d
        self.weight = Tensor(normal_init((max_len, d), std, r), requires_grad=True)

    def positions(self, T: int, offset: int = 0) -> Tensor:
        if T < 0 or offset < 0 or offset + T > self.max_len:
            raise ValueError(
                f"positions {offset}..{offset + T - 1} exceed max_len {self.max_len}"
            )
        # Rows offset .. offset+T-1 of the table, with their gradient.
        return F.embedding(self.weight, np.arange(offset, offset + T))

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        if x.shape[-1] != self.d:
            raise ValueError(f"x has width {x.shape[-1]}, expected {self.d}")
        return x + self.positions(x.shape[-2], offset)
