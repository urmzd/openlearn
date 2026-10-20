# contracts/py/tinyllm/modern/norm.pyi (L7.1): RMSNorm and the Pre-LN residual step
# chapter: content/courses/ai-engineering/13-decoder-block/01-pre-ln-rmsnorm.md
#
# RMSNorm (Zhang and Sennrich 2019) rescales each vector to unit root mean
# square over its last axis, then multiplies by a learned gain. Unlike
# LayerNorm (L0.4) it subtracts no mean and has no bias:
#
#     rms(x) = sqrt(mean(x_i^2) + eps)        over the last axis, eps INSIDE the root
#     y      = x / rms(x) * (offset + weight)
#
# offset 0 is Llama's form (weight starts at ones); offset 1 is Gemma's form
# (weight starts at zeros, the gain is 1 + weight). The computation runs in
# x's dtype (float32 or float64), built from the op library (L0.2), so
# backward reaches x and weight.
#
# The Pre-LN residual step (Xiong et al. 2020) normalizes only the branch
# input, never the residual stream itself:
#
#     pre_norm_residual(x, norm, sublayer) = x + sublayer(norm(x))
#
# A Llama block is two of them, attention then MLP, and a final RMSNorm
# before the lm_head (L7.9).
from typing import Callable

from tinyllm.autograd.tensor import Tensor
from tinyllm.nn.module import Module

class RMSNorm(Module):
    weight: Tensor  # [d], float32: ones when offset == 0, zeros otherwise
    eps: float
    offset: float

    def __init__(self, d: int, eps: float = 1e-6, offset: float = 0.0) -> None:
        """ValueError when d < 1 or eps < 0."""

    def forward(self, x: Tensor) -> Tensor:
        """x [..., d] -> [..., d] as above. ValueError when x.shape[-1] != d."""

def pre_norm_residual(x: Tensor, norm: Module, sublayer: Callable[[Tensor], Tensor]) -> Tensor:
    """x + sublayer(norm(x)). The identity path carries x through unchanged,
    so d out / d x = I + (the branch's Jacobian): at a sublayer that outputs
    zeros the step is exactly the identity, in value and in gradient."""
