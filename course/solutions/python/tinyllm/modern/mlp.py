"""Gated MLPs: SwiGLU and GeGLU (L7.2).

The 2017 MLP is down(act(up(x))). The gated form computes two projections of
x and lets one, through the activation, gate the other elementwise. With the
hidden width cut to 2/3 it has the same parameters and FLOPs, and trains
to a lower loss (Shazeer 2020).

Contract: contracts/py/tinyllm/modern/mlp.pyi.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from tinyllm.autograd import functional as F
from tinyllm.autograd.tensor import Tensor
from tinyllm.nn.layers import Linear
from tinyllm.nn.module import Module
from tinyllm.num.rng import PCG32


class GatedMLP(Module):
    def __init__(
        self,
        d: int,
        d_ff: int,
        act: Literal["silu", "gelu_tanh"] = "silu",
        bias: bool = False,
        rng: Any = None,
    ) -> None:
        super().__init__()
        if min(d, d_ff) < 1:
            raise ValueError(f"sizes must be positive, got d={d}, d_ff={d_ff}")
        if act not in ("silu", "gelu_tanh"):
            raise ValueError(f"act must be 'silu' or 'gelu_tanh', got {act!r}")
        r = rng if rng is not None else PCG32(0).substream("init")
        self.act = act
        # Registration order is HF's LlamaMLP order: gate, up, down.
        self.gate_proj = Linear(d, d_ff, bias=bias, rng=r)
        self.up_proj = Linear(d, d_ff, bias=bias, rng=r)
        self.down_proj = Linear(d_ff, d, bias=bias, rng=r)

    def forward(self, x: Tensor) -> Tensor:
        g = self.gate_proj(x)
        a = F.silu(g) if self.act == "silu" else F.gelu(g, approximate="tanh")
        # The activation gates the OTHER projection: act(gate) * up.
        return self.down_proj(a * self.up_proj(x))


def llama_ffn_dim(d: int, multiple_of: int = 256, ffn_dim_multiplier: Optional[float] = None) -> int:
    if d < 1 or multiple_of < 1:
        raise ValueError(f"d and multiple_of must be positive, got {d}, {multiple_of}")
    h = int(2 * (4 * d) / 3)
    if ffn_dim_multiplier is not None:
        if not ffn_dim_multiplier > 0:
            raise ValueError(f"ffn_dim_multiplier must be positive, got {ffn_dim_multiplier!r}")
        h = int(ffn_dim_multiplier * h)
    # Round UP to a multiple: hardware-friendly matmul widths.
    return multiple_of * ((h + multiple_of - 1) // multiple_of)
