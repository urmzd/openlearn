# contracts/py/tinyllm/xfmr/mha.pyi (L5.3): multi-head attention
# chapter: content/courses/build-an-llm/p05-transformer-2017/03-multi-head-attention.md
#
# Vaswani et al. (2017), section 3.2.2. With d_head = d_model / n_heads:
#
#     Q = x_q W_q^T + b_q    K = x_kv W_k^T + b_k    V = x_kv W_v^T + b_v     [B, T, d_model]
#     split: [B, T, d_model] -> [B, T, H, d_head] -> [B, H, T, d_head]
#            (head h owns features h*d_head .. (h+1)*d_head - 1)
#     head_h = softmax(Q_h K_h^T / sqrt(d_head) + mask) V_h                    (L5.1)
#     merge: [B, H, Tq, d_head] -> [B, Tq, H, d_head] -> [B, Tq, d_model]
#     out = merge(heads) W_o^T + b_o
#
# Parameters, in registration order (= state_dict order), float32, each an
# L0.4 Linear(d_model, d_model, bias) drawn from one rng (PCG32; None means
# PCG32(0).substream("init")): q_proj, k_proj, v_proj, out_proj, so the keys
# are q_proj.weight, q_proj.bias, k_proj.weight, ..., out_proj.bias. torch's
# nn.MultiheadAttention packs the first three as in_proj_weight [3 d, d]
# (rows q | k | v); load_packed_in_proj reads that layout.
#
# Masks are bool, True = may attend (L5.2's convention), and broadcast to
# [B, H, Tq, Tk]: a [Tq, Tk] mask applies to every sequence and head, a
# [B, Tq, Tk] mask to every head of its sequence (a head axis is inserted at
# 1), and a [B, H, Tq, Tk] or [B, 1, 1, Tk] mask as it is.
#
# Attention dropout (rate `dropout`) is applied to the weights inside
# scaled_dot_product_attention (L5.1) only in training mode, drawing from
# PCG32(0).substream("dropout"); in eval mode nothing is drawn.
from typing import Any, Optional

from numpy.typing import ArrayLike, NDArray

from tinyllm.autograd.tensor import Tensor
from tinyllm.nn.layers import Linear
from tinyllm.nn.module import Module

class MultiHeadAttention(Module):
    # L5.5's encoder and decoder layers, L6.1's GPT blocks, and L6.2's BERT
    # layers all attend through this module.
    d_model: int
    n_heads: int
    d_head: int
    dropout: float
    q_proj: Linear
    k_proj: Linear
    v_proj: Linear
    out_proj: Linear

    def __init__(
        self, d_model: int, n_heads: int, dropout: float = 0.0, bias: bool = True, rng: Any = None
    ) -> None:
        """The four projections above. ValueError for a size below 1, d_model
        not divisible by n_heads, or dropout outside [0, 1)."""

    def split_heads(self, x: Tensor) -> Tensor:
        """[B, T, d_model] -> [B, H, T, d_head] as above. ValueError for
        another shape."""

    def merge_heads(self, x: Tensor) -> Tensor:
        """The inverse of split_heads: [B, H, T, d_head] -> [B, T, d_model].
        ValueError for another shape."""

    def head_mask(self, mask: Optional[ArrayLike], B: int, Tq: int, Tk: int) -> Optional[NDArray]:
        """The mask in a shape that broadcasts to [B, H, Tq, Tk] (None stays
        None). ValueError for a non-bool mask, a mask of 1 or more than 4
        dimensions, or one that does not broadcast."""

    def attend(
        self, x_q: Tensor, x_kv: Tensor, mask: Optional[ArrayLike] = None
    ) -> tuple[Tensor, Tensor]:
        """(out [B, Tq, d_model], weights [B, H, Tq, Tk]). Queries come from
        x_q [B, Tq, d_model], keys and values from x_kv [B, Tk, d_model]:
        self-attention passes the same tensor twice, cross-attention the
        encoder's output as x_kv. ValueError for x_q or x_kv not 3-D, batch
        sizes that differ, a width other than d_model, or a bad mask."""

    def forward(self, x_q: Tensor, x_kv: Tensor, mask: Optional[ArrayLike] = None) -> Tensor:
        """attend(x_q, x_kv, mask)[0]."""

    def load_packed_in_proj(self, weight: ArrayLike, bias: Optional[ArrayLike] = None) -> None:
        """Copy torch's packed in_proj_weight [3 d, d] (rows q, then k, then
        v) and in_proj_bias [3 d] into q_proj, k_proj, and v_proj, in place.
        ValueError for another shape, or a bias for a module built with
        bias=False."""
