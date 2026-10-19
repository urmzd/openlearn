"""Grouped-query attention with a cache hook, a sliding window, and learned sinks (L7.5).

Multi-head attention gives every query head its own keys and values, and at
inference the KV cache stores all of them for every past token. GQA lets
groups of query heads share one key/value head: the scores keep their H
query heads while the cache shrinks by H / Hkv. SmolLM2-135M runs 9 query
heads over 3 kv heads; Llama-2-70B 64 over 8.

Contract: contracts/py/tinyllm/modern/gqa.pyi.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol

import numpy as np
from numpy.typing import NDArray

from tinyllm.autograd import functional as F
from tinyllm.autograd.tensor import Tensor
from tinyllm.modern.rope import RopeSpec, apply_rope, rope_cos_sin
from tinyllm.nn.layers import Linear
from tinyllm.nn.module import Module
from tinyllm.num.rng import PCG32


class KVCacheHook(Protocol):
    """The seam L8.2's KVCache (and L8.3's paged cache) implements."""

    def update(self, layer: int, k_new: NDArray, v_new: NDArray) -> tuple[NDArray, NDArray]: ...


class ConcatKVCache:
    """The smallest cache that satisfies the hook."""

    def __init__(self) -> None:
        self._k: dict[int, NDArray] = {}
        self._v: dict[int, NDArray] = {}

    def update(self, layer: int, k_new: NDArray, v_new: NDArray) -> tuple[NDArray, NDArray]:
        k_new, v_new = np.array(k_new, copy=True), np.array(v_new, copy=True)
        if k_new.ndim != 4 or k_new.shape != v_new.shape:
            raise ValueError(f"k and v must be equal [B, Hkv, T, dh], got {k_new.shape} and {v_new.shape}")
        if layer in self._k:
            old = self._k[layer]
            if old.shape[:2] != k_new.shape[:2] or old.shape[3] != k_new.shape[3]:
                raise ValueError(f"layer {layer}: chunk {k_new.shape} does not extend {old.shape}")
            self._k[layer] = np.concatenate([old, k_new], axis=2)
            self._v[layer] = np.concatenate([self._v[layer], v_new], axis=2)
        else:
            self._k[layer], self._v[layer] = k_new, v_new
        return self._k[layer], self._v[layer]

    def seq_len(self, layer: int = 0) -> int:
        return int(self._k[layer].shape[2]) if layer in self._k else 0


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    if n_rep < 1 or x.ndim != 4:
        raise ValueError(f"need n_rep >= 1 and x [B, Hkv, T, dh], got n_rep={n_rep}, shape {x.shape}")
    if n_rep == 1:
        return x
    # head h reads kv head h // n_rep: [0, 0, 1, 1, ...], not [0, 1, 0, 1, ...]
    idx = np.repeat(np.arange(x.shape[1]), n_rep)
    return x[:, idx]


def _visible(T: int, Tk: int, window: Optional[int]) -> NDArray:
    """bool [T, Tk]: query t (key index Tk - T + t) may read key j."""
    i = np.arange(T)[:, None] + (Tk - T)
    j = np.arange(Tk)[None, :]
    vis = j <= i
    if window is not None:
        vis &= (i - j) < window
    return vis


class GQAttention(Module):
    def __init__(
        self, d: int, n_heads: int, n_kv_heads: int, d_head: Optional[int], rope: RopeSpec,
        qkv_bias: bool = False, window: Optional[int] = None, sinks: bool = False, rng: Any = None,
    ) -> None:
        super().__init__()
        if min(d, n_heads, n_kv_heads) < 1 or n_heads % n_kv_heads:
            raise ValueError(f"need sizes >= 1 and n_heads % n_kv_heads == 0, got {d}, {n_heads}, {n_kv_heads}")
        if d_head is None:
            if d % n_heads:
                raise ValueError(f"d_head None needs n_heads | d, got {d}, {n_heads}")
            d_head = d // n_heads
        if d_head < 1:
            raise ValueError(f"d_head must be >= 1, got {d_head}")
        if rope.rotary_dim % 2 or not 2 <= rope.rotary_dim <= d_head:
            raise ValueError(f"rope.rotary_dim must be even and in [2, {d_head}], got {rope.rotary_dim}")
        if window is not None and window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        r = rng if rng is not None else PCG32(0).substream("init")
        self.n_heads, self.n_kv_heads, self.d_head = n_heads, n_kv_heads, d_head
        self.d, self.window, self.rope = d, window, rope
        self.q_proj = Linear(d, n_heads * d_head, bias=qkv_bias, rng=r)
        self.k_proj = Linear(d, n_kv_heads * d_head, bias=qkv_bias, rng=r)
        self.v_proj = Linear(d, n_kv_heads * d_head, bias=qkv_bias, rng=r)
        self.o_proj = Linear(n_heads * d_head, d, bias=False, rng=r)
        self.sinks = Tensor(np.zeros(n_heads, dtype=np.float32), requires_grad=True) if sinks else None

    def _heads(self, y: Tensor, B: int, T: int, h: int) -> Tensor:
        return F.transpose(F.reshape(y, (B, T, h, self.d_head)), 1, 2)  # [B, h, T, dh]

    def forward(
        self,
        x: Tensor,
        positions: Any,
        mask: Optional[NDArray] = None,
        cache: Optional[KVCacheHook] = None,
        layer: int = 0,
    ) -> Tensor:
        if x.ndim != 3 or x.shape[2] != self.d:
            raise ValueError(f"x must be [B, T, {self.d}], got {x.shape}")
        B, T, _ = x.shape
        H, Hkv, dh = self.n_heads, self.n_kv_heads, self.d_head
        pos = np.asarray(positions)
        if pos.shape not in ((T,), (B, T)):
            raise ValueError(f"positions must be [T] or [B, T] = {(B, T)}, got {pos.shape}")
        q = self._heads(self.q_proj(x), B, T, H)
        k = self._heads(self.k_proj(x), B, T, Hkv)
        v = self._heads(self.v_proj(x), B, T, Hkv)
        cos, sin = rope_cos_sin(pos, self.rope.inv_freq, self.rope.attention_scaling)
        if pos.ndim == 2:  # per-row positions: [B, 1, T, r/2] broadcasts over heads
            cos, sin = cos[:, None], sin[:, None]
        q = apply_rope(q, cos, sin, self.rope.layout, self.rope.rotary_dim)
        k = apply_rope(k, cos, sin, self.rope.layout, self.rope.rotary_dim)
        if cache is not None:
            # The cache keeps Hkv heads, not H: that is the point of GQA.
            ka, va = cache.update(layer, k.data, v.data)
            k, v = Tensor(ka, dtype=x.dtype), Tensor(va, dtype=x.dtype)
        Tk = k.shape[2]
        vis = _visible(T, Tk, self.window)
        if mask is not None:
            m = np.asarray(mask)
            try:
                np.broadcast_shapes(m.shape, (B, H, T, Tk))
            except ValueError:
                raise ValueError(f"mask {m.shape} does not broadcast to {(B, H, T, Tk)}") from None
            vis = vis & m.astype(bool)
        n_rep = H // Hkv
        scores = F.matmul(q, F.transpose(repeat_kv(k, n_rep), 2, 3)) * (dh**-0.5)
        scores = F.masked_fill(scores, ~vis, -np.inf)
        if self.sinks is not None:
            # one extra logit per head in every row: weight it may give to nothing
            col = F.reshape(self.sinks, (1, H, 1, 1)) * np.ones((B, 1, T, 1))
            p = F.softmax(F.concat([scores, col], axis=-1), axis=-1)[..., :Tk]
        else:
            p = F.softmax(scores, axis=-1)
        o = F.matmul(p, repeat_kv(v, n_rep))  # [B, H, T, dh]
        return self.o_proj(F.reshape(F.transpose(o, 1, 2), (B, T, H * dh)))
