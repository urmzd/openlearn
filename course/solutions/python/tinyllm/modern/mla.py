"""Multi-head latent attention and weight absorption (L7.6).

GQA shrinks the KV cache by sharing key/value heads; MLA shrinks it further
by caching one low-rank latent per token from which every head's key and
value are linear projections. Because the projections are linear, a decode
step can fold them into the query and the output: attention then runs over
the cached latents directly and the per-head keys are never built.

Contract: contracts/py/tinyllm/modern/mla.pyi.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol

import numpy as np
from numpy.typing import NDArray

from tinyllm.autograd import functional as F
from tinyllm.autograd.tensor import Tensor
from tinyllm.modern.norm import RMSNorm
from tinyllm.modern.rope import RopeSpec, apply_rope, rope_cos_sin
from tinyllm.nn.layers import Linear
from tinyllm.nn.module import Module
from tinyllm.num.rng import PCG32

LATENT_EPS = 1e-6  # HF DeepseekV3RMSNorm's default for q_a_layernorm and kv_a_layernorm


class LatentCacheHook(Protocol):
    def update(self, layer: int, c_new: NDArray, k_rope_new: NDArray) -> tuple[NDArray, NDArray]: ...


class ConcatLatentCache:
    def __init__(self) -> None:
        self._c: dict[int, NDArray] = {}
        self._k: dict[int, NDArray] = {}

    def update(self, layer: int, c_new: NDArray, k_rope_new: NDArray) -> tuple[NDArray, NDArray]:
        c_new, k_new = np.array(c_new, copy=True), np.array(k_rope_new, copy=True)
        if c_new.ndim != 3 or k_new.ndim != 3 or c_new.shape[:2] != k_new.shape[:2]:
            raise ValueError(f"want c [B, T, r] and k_rope [B, T, dr], got {c_new.shape} and {k_new.shape}")
        if layer in self._c:
            oc, ok = self._c[layer], self._k[layer]
            if oc.shape[0] != c_new.shape[0] or oc.shape[2] != c_new.shape[2] or ok.shape[2] != k_new.shape[2]:
                raise ValueError(f"layer {layer}: chunk {c_new.shape}/{k_new.shape} does not extend {oc.shape}/{ok.shape}")
            self._c[layer] = np.concatenate([oc, c_new], axis=1)
            self._k[layer] = np.concatenate([ok, k_new], axis=1)
        else:
            self._c[layer], self._k[layer] = c_new, k_new
        return self._c[layer], self._k[layer]

    def seq_len(self, layer: int = 0) -> int:
        return int(self._c[layer].shape[1]) if layer in self._c else 0


def _positions(positions: Any, B: int, T: int) -> NDArray:
    pos = np.asarray(positions)
    if pos.shape not in ((T,), (B, T)):
        raise ValueError(f"positions must be [T] or [B, T] = {(B, T)}, got {pos.shape}")
    return pos


def _visible(T: int, Tk: int, mask: Optional[NDArray], B: int, H: int) -> NDArray:
    """bool broadcastable to [B, H, T, Tk]: causal by key index, AND mask."""
    i = np.arange(T)[:, None] + (Tk - T)
    vis = np.arange(Tk)[None, :] <= i
    if mask is not None:
        m = np.asarray(mask)
        try:
            np.broadcast_shapes(m.shape, (B, H, T, Tk))
        except ValueError:
            raise ValueError(f"mask {m.shape} does not broadcast to {(B, H, T, Tk)}") from None
        vis = vis & m.astype(bool)
    return vis


class MLAttention(Module):
    def __init__(
        self, d: int, n_heads: int, q_lora_rank: Optional[int], kv_lora_rank: int,
        qk_nope_dim: int, qk_rope_dim: int, v_dim: int, rope: RopeSpec,
        attention_bias: bool = False, softmax_scale: Optional[float] = None, rng: Any = None,
    ) -> None:
        super().__init__()
        if min(d, n_heads, kv_lora_rank, qk_nope_dim, v_dim) < 1:
            raise ValueError(f"sizes must be >= 1, got d={d}, H={n_heads}, r={kv_lora_rank}, dn={qk_nope_dim}, dv={v_dim}")
        if q_lora_rank is not None and q_lora_rank < 1:
            raise ValueError(f"q_lora_rank must be None or >= 1, got {q_lora_rank}")
        if qk_rope_dim < 2 or qk_rope_dim % 2:
            raise ValueError(f"qk_rope_dim must be a positive even integer, got {qk_rope_dim}")
        if rope.rotary_dim != qk_rope_dim:
            raise ValueError(f"rope.rotary_dim must equal qk_rope_dim = {qk_rope_dim}, got {rope.rotary_dim}")
        dqk = qk_nope_dim + qk_rope_dim
        if softmax_scale is None:
            softmax_scale = dqk**-0.5  # the full query width, nope AND rope
        if not softmax_scale > 0:
            raise ValueError(f"softmax_scale must be > 0, got {softmax_scale}")
        r = rng if rng is not None else PCG32(0).substream("init")
        self.d, self.n_heads, self.rope = d, n_heads, rope
        self.q_lora_rank, self.kv_lora_rank = q_lora_rank, kv_lora_rank
        self.qk_nope_dim, self.qk_rope_dim, self.v_dim = qk_nope_dim, qk_rope_dim, v_dim
        self.softmax_scale = float(softmax_scale)
        # Registration order = HF DeepseekV3Attention = the state_dict keys.
        if q_lora_rank is None:
            self.q_proj = Linear(d, n_heads * dqk, bias=False, rng=r)
            self.q_a_proj = self.q_a_layernorm = self.q_b_proj = None
        else:
            self.q_proj = None
            self.q_a_proj = Linear(d, q_lora_rank, bias=attention_bias, rng=r)
            self.q_a_layernorm = RMSNorm(q_lora_rank, eps=LATENT_EPS)
            self.q_b_proj = Linear(q_lora_rank, n_heads * dqk, bias=False, rng=r)
        self.kv_a_proj_with_mqa = Linear(d, kv_lora_rank + qk_rope_dim, bias=attention_bias, rng=r)
        self.kv_a_layernorm = RMSNorm(kv_lora_rank, eps=LATENT_EPS)
        self.kv_b_proj = Linear(kv_lora_rank, n_heads * (qk_nope_dim + v_dim), bias=False, rng=r)
        self.o_proj = Linear(n_heads * v_dim, d, bias=attention_bias, rng=r)

    def _query(self, x: Tensor) -> Tensor:
        """[B, T, d] -> [B, H, T, dn + dr]."""
        B, T, _ = x.shape
        if self.q_proj is not None:
            q = self.q_proj(x)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        return F.transpose(F.reshape(q, (B, T, self.n_heads, self.qk_nope_dim + self.qk_rope_dim)), 1, 2)

    def _latent(self, x: Tensor, cos: NDArray, sin: NDArray) -> tuple[Tensor, Tensor]:
        """The normalized latent c [B, T, r] and the rotated rope key [B, T, dr]."""
        ckv = self.kv_a_proj_with_mqa(x)
        r = self.kv_lora_rank
        c = self.kv_a_layernorm(ckv[..., :r])
        k_rope = apply_rope(ckv[..., r:], cos, sin, self.rope.layout, self.qk_rope_dim)
        return c, k_rope

    def _tables(self, pos: NDArray) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """cos, sin for [B, T, dr] keys and for [B, H, T, dr] queries."""
        cos, sin = rope_cos_sin(pos, self.rope.inv_freq, self.rope.attention_scaling)
        if pos.ndim == 2:  # per-row positions: [B, T, r/2] for keys, [B, 1, T, r/2] for queries
            return cos, sin, cos[:, None], sin[:, None]
        return cos, sin, cos, sin

    def forward(
        self,
        x: Tensor,
        positions: Any,
        mask: Optional[NDArray] = None,
        cache: Optional[LatentCacheHook] = None,
        layer: int = 0,
    ) -> Tensor:
        if x.ndim != 3 or x.shape[2] != self.d:
            raise ValueError(f"x must be [B, T, {self.d}], got {x.shape}")
        B, T, _ = x.shape
        H, dn, dv = self.n_heads, self.qk_nope_dim, self.v_dim
        pos = _positions(positions, B, T)
        kcos, ksin, qcos, qsin = self._tables(pos)
        q = self._query(x)
        q_nope = q[..., :dn]
        q_rope = apply_rope(q[..., dn:], qcos, qsin, self.rope.layout, self.qk_rope_dim)
        c, k_rope = self._latent(x, kcos, ksin)
        if cache is not None:
            # The cache holds the latent and the rotated rope key: r + dr numbers
            # per token, whatever the number of heads.
            ca, ka = cache.update(layer, c.data, k_rope.data)
            c, k_rope = Tensor(ca, dtype=x.dtype), Tensor(ka, dtype=x.dtype)
        Tk = c.shape[1]
        kv = F.transpose(F.reshape(self.kv_b_proj(c), (B, Tk, H, dn + dv)), 1, 2)  # [B, H, Tk, dn + dv]
        k_nope, v = kv[..., :dn], kv[..., dn:]
        k_r = F.reshape(k_rope, (B, 1, Tk, self.qk_rope_dim))  # one rope key, every head
        scores = F.matmul(q_nope, F.transpose(k_nope, 2, 3)) + F.matmul(q_rope, F.transpose(k_r, 2, 3))
        scores = F.masked_fill(scores * self.softmax_scale, ~_visible(T, Tk, mask, B, H), -np.inf)
        o = F.matmul(F.softmax(scores, axis=-1), v)  # [B, H, T, dv]
        return self.o_proj(F.reshape(F.transpose(o, 1, 2), (B, T, H * dv)))

    def absorb_weights(self) -> "AbsorbedMLA":
        return AbsorbedMLA(self)


class AbsorbedMLA:
    def __init__(self, mla: MLAttention) -> None:
        H, dn, dv, r = mla.n_heads, mla.qk_nope_dim, mla.v_dim, mla.kv_lora_rank
        self.mla = mla
        w = mla.kv_b_proj.weight.data.astype(np.float64).reshape(H, dn + dv, r)
        self.w_uk = np.array(w[:, :dn, :])  # [H, dn, r]: k_nope_h = W_UK_h c
        w_uv = w[:, dn:, :]  # [H, dv, r]:  v_h = W_UV_h c
        w_o = mla.o_proj.weight.data.astype(np.float64).reshape(mla.d, H, dv)  # [d, H, dv]
        self.w_ov = np.einsum("dhv,hvr->hdr", w_o, w_uv)  # [H, d, r] = W_O_h W_UV_h
        b = mla.o_proj.bias
        self.o_bias = None if b is None else np.array(b.data, dtype=np.float64)

    def forward(
        self,
        x: Any,
        positions: Any,
        cache: LatentCacheHook,
        layer: int = 0,
        mask: Optional[NDArray] = None,
    ) -> NDArray:
        m = self.mla
        xt = x if isinstance(x, Tensor) else Tensor(np.asarray(x), dtype=np.asarray(x).dtype)
        dt = xt.data.dtype
        if xt.ndim != 3 or xt.shape[2] != m.d:
            raise ValueError(f"x must be [B, T, {m.d}], got {xt.shape}")
        B, T, _ = xt.shape
        pos = _positions(positions, B, T)
        kcos, ksin, qcos, qsin = m._tables(pos)
        q = m._query(xt)
        q_nope = q.data[..., : m.qk_nope_dim]
        q_rope = apply_rope(q[..., m.qk_nope_dim :], qcos, qsin, m.rope.layout, m.qk_rope_dim).data
        c, k_rope = m._latent(xt, kcos, ksin)
        ca, ka = cache.update(layer, c.data, k_rope.data)
        ca, ka = np.asarray(ca, dtype=dt), np.asarray(ka, dtype=dt)
        Tk = ca.shape[1]
        # Query absorption: q_nope . (W_UK c) = (q_nope W_UK) . c, one latent per key.
        q_lat = np.einsum("bhtn,hnr->bhtr", q_nope, self.w_uk.astype(dt))
        scores = np.einsum("bhtr,bkr->bhtk", q_lat, ca) + np.einsum("bhtn,bkn->bhtk", q_rope, ka)
        scores = scores * dt.type(m.softmax_scale)
        vis = _visible(T, Tk, mask, B, m.n_heads)
        scores = np.where(vis, scores, -np.inf)
        mx = np.max(scores, axis=-1, keepdims=True)
        mx = np.where(np.isfinite(mx), mx, 0)
        e = np.where(vis, np.exp(scores - mx), 0)
        s = e.sum(axis=-1, keepdims=True)
        p = np.divide(e, s, out=np.zeros_like(e), where=s > 0)
        # Output absorption: W_O_h W_UV_h applied once to the weighted latent.
        o_lat = np.einsum("bhtk,bkr->bhtr", p, ca)
        out = np.einsum("bhtr,hdr->btd", o_lat, self.w_ov.astype(dt))
        if self.o_bias is not None:
            out = out + self.o_bias.astype(dt)
        return out.astype(dt)
