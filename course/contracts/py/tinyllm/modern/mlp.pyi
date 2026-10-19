# contracts/py/tinyllm/modern/mlp.pyi (L7.2): gated MLPs (SwiGLU, GeGLU)
# chapter: content/courses/build-an-llm/08-modern-block/02-gated-mlps.md
#
# Shazeer (2020): the first projection is split in two, one half gated by an
# activation of the other:
#
#     GatedMLP(x) = down( act(gate(x)) * up(x) )        * elementwise
#
#   act = "silu"       SwiGLU (Llama, Mistral, Qwen, SmolLM2): silu(z) = z sigmoid(z)
#   act = "gelu_tanh"  GeGLU (Gemma): the tanh approximation of GELU
#
# Parameters, in registration order (= state_dict = HF LlamaMLP key order),
# float32, from L0.4's Linear drawn from one rng (PCG32, M06.3; None means
# PCG32(0).substream("init")) in this order:
#   gate_proj.weight [d_ff, d]   (gate_proj.bias [d_ff] when bias)
#   up_proj.weight   [d_ff, d]   (up_proj.bias   [d_ff] when bias)
#   down_proj.weight [d, d_ff]   (down_proj.bias [d]    when bias)
# 3 d d_ff weights: a gated MLP with d_ff = 8 d / 3 has the parameters of a
# plain 4 d MLP (2 d 4d), which is why Llama's widths look odd.
from typing import Any, Literal, Optional

from tinyllm.autograd.tensor import Tensor
from tinyllm.nn.layers import Linear
from tinyllm.nn.module import Module

class GatedMLP(Module):
    gate_proj: Linear
    up_proj: Linear
    down_proj: Linear
    act: str

    def __init__(
        self,
        d: int,
        d_ff: int,
        act: Literal["silu", "gelu_tanh"] = "silu",
        bias: bool = False,
        rng: Any = None,
    ) -> None:
        """ValueError when d or d_ff < 1 or act is not one of the two."""

    def forward(self, x: Tensor) -> Tensor:
        """[..., d] -> [..., d]: down(act(gate(x)) * up(x)), built from the
        op library (L0.2: F.silu, F.gelu(approximate="tanh")), so backward
        reaches x and all six (or three) parameters."""

def llama_ffn_dim(d: int, multiple_of: int = 256, ffn_dim_multiplier: Optional[float] = None) -> int:
    """Meta's Llama rule for d_ff: h = int(2 * 4 d / 3); with a multiplier,
    h = int(ffn_dim_multiplier * h); then round h UP to a multiple of
    multiple_of. Llama-2 7B: d = 4096 gives 11008; Llama-3 8B (multiplier
    1.3, multiple_of 1024) gives 14336. ValueError when d or multiple_of < 1
    or the multiplier is not positive."""
