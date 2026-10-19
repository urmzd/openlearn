"""Multi-head attention (L5.3).

H heads attend in parallel, each in its own d_head = d_model / H dimensional
subspace: project the queries, keys, and values with one Linear each, split
the last axis into H heads, run scaled dot-product attention (L5.1) on every
head at once, merge the heads back, and mix them with an output projection.

Contract: contracts/py/tinyllm/xfmr/mha.pyi.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tinyllm.autograd import functional as F
from tinyllm.autograd.tensor import Tensor
from tinyllm.nn.layers import Linear
from tinyllm.nn.module import Module
from tinyllm.num.rng import PCG32
from tinyllm.xfmr.sdpa import scaled_dot_product_attention


class MultiHeadAttention(Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        rng: Any = None,
    ) -> None:
        super().__init__()
        if d_model < 1 or n_heads < 1:
            raise ValueError(
                f"d_model and n_heads must be positive, got {d_model}, {n_heads}"
            )
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} is not divisible by n_heads {n_heads}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        r = rng if rng is not None else PCG32(0).substream("init")
        self.d_model, self.n_heads = d_model, n_heads
        self.d_head = d_model // n_heads
        self.dropout = float(dropout)
        self.q_proj = Linear(d_model, d_model, bias=bias, rng=r)
        self.k_proj = Linear(d_model, d_model, bias=bias, rng=r)
        self.v_proj = Linear(d_model, d_model, bias=bias, rng=r)
        self.out_proj = Linear(d_model, d_model, bias=bias, rng=r)
        self.dropout_rng = PCG32(0).substream("dropout")

    def split_heads(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[2] != self.d_model:
            raise ValueError(f"expected [B, T, {self.d_model}], got {x.shape}")
        B, T, _ = x.shape
        # Head h owns features h*d_head .. (h+1)*d_head - 1 of every position.
        return F.transpose(F.reshape(x, (B, T, self.n_heads, self.d_head)), 1, 2)

    def merge_heads(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != self.n_heads or x.shape[3] != self.d_head:
            raise ValueError(
                f"expected [B, {self.n_heads}, T, {self.d_head}], got {x.shape}"
            )
        B, _, T, _ = x.shape
        # Back to [B, T, H, d_head] first: positions must not mix.
        return F.reshape(F.transpose(x, 1, 2), (B, T, self.d_model))

    def head_mask(
        self, mask: Optional[ArrayLike], B: int, Tq: int, Tk: int
    ) -> Optional[NDArray]:
        if mask is None:
            return None
        m = np.asarray(mask)
        if m.dtype != np.bool_:
            raise ValueError(f"mask must be bool (True = may attend), got {m.dtype}")
        if m.ndim == 3:
            # [B, Tq, Tk]: one mask per sequence, the same for every head.
            m = m[:, None, :, :]
        elif m.ndim not in (2, 4):
            raise ValueError(
                f"mask must be [Tq, Tk], [B, Tq, Tk] or [B, H, Tq, Tk], got {m.shape}"
            )
        try:
            np.broadcast_shapes(m.shape, (B, self.n_heads, Tq, Tk))
        except ValueError:
            raise ValueError(
                f"mask {np.asarray(mask).shape} does not broadcast to [B, H, Tq, Tk] = "
                f"{(B, self.n_heads, Tq, Tk)}"
            ) from None
        return m

    def attend(
        self, x_q: Tensor, x_kv: Tensor, mask: Optional[ArrayLike] = None
    ) -> tuple[Tensor, Tensor]:
        if x_q.ndim != 3 or x_kv.ndim != 3:
            raise ValueError(
                f"x_q and x_kv must be [B, T, d], got {x_q.shape} and {x_kv.shape}"
            )
        if x_q.shape[0] != x_kv.shape[0]:
            raise ValueError(f"batch sizes differ: {x_q.shape[0]} and {x_kv.shape[0]}")
        B, Tq, Tk = x_q.shape[0], x_q.shape[1], x_kv.shape[1]
        m = self.head_mask(mask, B, Tq, Tk)
        q = self.split_heads(self.q_proj(x_q))
        k = self.split_heads(self.k_proj(x_kv))
        v = self.split_heads(self.v_proj(x_kv))
        p = self.dropout if self.training else 0.0
        # Default scale 1 / sqrt(d_head): the width of what each head compares.
        ctx, w = scaled_dot_product_attention(
            q, k, v, mask=m, dropout_p=p, rng=self.dropout_rng
        )
        return self.out_proj(self.merge_heads(ctx)), w

    def forward(
        self, x_q: Tensor, x_kv: Tensor, mask: Optional[ArrayLike] = None
    ) -> Tensor:
        return self.attend(x_q, x_kv, mask)[0]

    def load_packed_in_proj(
        self, weight: ArrayLike, bias: Optional[ArrayLike] = None
    ) -> None:
        w = np.asarray(weight, dtype=np.float32)
        d = self.d_model
        if w.shape != (3 * d, d):
            raise ValueError(
                f"packed weight must be [3 d, d] = {(3 * d, d)}, got {w.shape}"
            )
        sd = {
            "q_proj.weight": w[:d],
            "k_proj.weight": w[d : 2 * d],
            "v_proj.weight": w[2 * d :],
        }
        if bias is not None:
            b = np.asarray(bias, dtype=np.float32)
            if b.shape != (3 * d,):
                raise ValueError(
                    f"packed bias must be [3 d] = {(3 * d,)}, got {b.shape}"
                )
            if self.q_proj.bias is None:
                raise ValueError("this attention has no biases (bias=False)")
            sd.update(
                {
                    "q_proj.bias": b[:d],
                    "k_proj.bias": b[d : 2 * d],
                    "v_proj.bias": b[2 * d :],
                }
            )
        self.load_state_dict(sd, strict=False)
