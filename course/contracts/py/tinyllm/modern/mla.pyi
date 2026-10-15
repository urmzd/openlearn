# contracts/py/tinyllm/modern/mla.pyi (L7.6): multi-head latent attention and weight absorption
# chapter: content/courses/build-an-llm/p07-modern-block/06-multi-head-latent-attention.md
#
# MLA (DeepSeek-V2 2024, DeepSeek-V3 2024) caches ONE small latent vector per
# token instead of a key and a value per kv head. Every head's key and value
# are linear functions of that latent, so a decode step can fold the
# up-projections into the query and the output (weight absorption) and never
# materialize per-head keys for the cache.
#
# Symbols: d model width, H heads, r = kv_lora_rank (latent width),
# rq = q_lora_rank (None: a plain query projection), dn = qk_nope_dim,
# dr = qk_rope_dim, dv = v_dim, dqk = dn + dr. Shapes: x [B, T, d].
#
# Parameters, float32, registered in HF DeepseekV3Attention order (so
# state_dict keys are HF's), Linear layers drawn from one rng (PCG32, M06.3;
# None means PCG32(0).substream("init")) in this order:
#   rq None:  q_proj.weight [H dqk, d]
#   rq set:   q_a_proj.weight [rq, d] (bias with attention_bias),
#             q_a_layernorm.weight [rq] (L7.1 RMSNorm, eps 1e-6, ones),
#             q_b_proj.weight [H dqk, rq]
#   kv_a_proj_with_mqa.weight [r + dr, d]   (bias with attention_bias)
#   kv_a_layernorm.weight     [r]           (RMSNorm, eps 1e-6, ones)
#   kv_b_proj.weight          [H (dn + dv), r]
#   o_proj.weight             [d, H dv]     (bias with attention_bias)
#
# forward(x, positions, mask, cache, layer), the naive form:
#   1. q = q_proj(x) (or q_b_proj(q_a_layernorm(q_a_proj(x)))), split into H
#      heads of dqk; each head is [q_nope (dn) | q_rope (dr)], nope FIRST.
#   2. kv_a_proj_with_mqa(x) = [c_raw (r) | k_rope (dr)];
#      c = kv_a_layernorm(c_raw) is the latent.
#   3. RoPE (L7.3) with rope_cos_sin(positions, rope.inv_freq,
#      rope.attention_scaling) and rope.layout on q_rope (every head) and on
#      k_rope, ONE rope key shared by all heads; rope.rotary_dim == dr.
#   4. cache: when given, (c, k_rope) = cache.update(layer, c.data,
#      k_rope.data): the cache stores the normalized latent [B, T, r] and
#      the ROTATED rope key [B, T, dr] of this chunk and returns everything
#      it holds as numpy arrays [B, Tk, r], [B, Tk, dr] (constants).
#   5. kv_b_proj(c) split per head into [k_nope (dn) | v (dv)], nope FIRST.
#   6. scores = (q_nope . k_nope + q_rope . k_rope) * softmax_scale,
#      softmax_scale = dqk ** -0.5 unless given; visibility as in L7.5
#      (query t is key index Tk - T + t; causal, and mask when given);
#      softmax; weighted v; heads merged [B, T, H dv]; o_proj.
#
# Weight absorption (absorb_weights): with W_UK_h [dn, r] and W_UV_h [dv, r]
# the rows of kv_b_proj for head h, and W_O_h [d, dv] the columns of o_proj
# for head h,
#   q_nope_h . (W_UK_h c_j) = (W_UK_h^T q_nope_h) . c_j        query absorption
#   W_O_h sum_j p_j W_UV_h c_j = (W_O_h W_UV_h) sum_j p_j c_j    output absorption
# so attention runs over the cached latents directly: H "latent heads" of
# width r + dr sharing one key and value per token (MQA in latent space).
from typing import Any, Optional, Protocol

from numpy.typing import NDArray

from tinyllm.autograd.tensor import Tensor
from tinyllm.modern.norm import RMSNorm
from tinyllm.modern.rope import RopeSpec
from tinyllm.nn.layers import Linear
from tinyllm.nn.module import Module

class LatentCacheHook(Protocol):
    """The seam L8.2's LatentCache implements: per layer, the latents and
    rotated rope keys of every position seen so far."""

    def update(self, layer: int, c_new: NDArray, k_rope_new: NDArray) -> tuple[NDArray, NDArray]: ...

class ConcatLatentCache:
    """The smallest latent cache: per layer, the concatenation along time of
    every chunk it was given."""

    def __init__(self) -> None: ...
    def update(self, layer: int, c_new: NDArray, k_rope_new: NDArray) -> tuple[NDArray, NDArray]:
        """Append [B, T, r] latents and [B, T, dr] rope keys (copies) and
        return the full arrays [B, Tk, r], [B, Tk, dr]. ValueError for arrays
        that are not 3-D, disagree on B or T, or differ in width from the
        layer's earlier chunks."""
    def seq_len(self, layer: int = 0) -> int:
        """Positions held for layer (0 for a layer never updated)."""

class MLAttention(Module):
    q_proj: Optional[Linear]
    q_a_proj: Optional[Linear]
    q_a_layernorm: Optional[RMSNorm]
    q_b_proj: Optional[Linear]
    kv_a_proj_with_mqa: Linear
    kv_a_layernorm: RMSNorm
    kv_b_proj: Linear
    o_proj: Linear
    n_heads: int
    kv_lora_rank: int
    qk_nope_dim: int
    qk_rope_dim: int
    v_dim: int
    softmax_scale: float

    def __init__(
        self,
        d: int,
        n_heads: int,
        q_lora_rank: Optional[int],
        kv_lora_rank: int,
        qk_nope_dim: int,
        qk_rope_dim: int,
        v_dim: int,
        rope: RopeSpec,
        attention_bias: bool = False,
        softmax_scale: Optional[float] = None,
        rng: Any = None,
    ) -> None:
        """ValueError when d, n_heads, kv_lora_rank, qk_nope_dim, v_dim < 1,
        q_lora_rank < 1 (when given), qk_rope_dim is not a positive even
        integer, rope.rotary_dim != qk_rope_dim, or softmax_scale <= 0."""

    def forward(
        self,
        x: Tensor,
        positions: Any,
        mask: Optional[NDArray] = None,
        cache: Optional[LatentCacheHook] = None,
        layer: int = 0,
    ) -> Tensor:
        """[B, T, d] -> [B, T, d], steps 1 to 6 above, built from the op
        library so backward reaches x and every parameter. positions is [T]
        or [B, T]. A row with every key hidden gives output 0. ValueError for
        a wrong x width, positions whose shape is not [T] or [B, T], or a
        mask that does not broadcast to [B, H, T, Tk]."""

    def absorb_weights(self) -> "AbsorbedMLA":
        """AbsorbedMLA(self): the absorbed form of the current weights."""

class AbsorbedMLA:
    """Inference only: plain numpy, no autograd, computed in the input's
    dtype (the absorbed weights are cast to it)."""

    w_uk: NDArray  # [H, dn, r]: query absorption, q_lat_h = q_nope_h @ w_uk[h]
    w_ov: NDArray  # [H, d, r]: output absorption, W_O_h @ W_UV_h
    o_bias: Optional[NDArray]  # [d] or None

    def __init__(self, mla: MLAttention) -> None:
        """w_uk, w_ov, and o_bias are float64 copies computed now from
        mla.kv_b_proj and mla.o_proj; the query path and kv_a_proj_with_mqa
        (with its norm) stay mla's own layers."""

    def forward(
        self,
        x: Any,
        positions: Any,
        cache: LatentCacheHook,
        layer: int = 0,
        mask: Optional[NDArray] = None,
    ) -> NDArray:
        """The same output as MLAttention.forward(x, positions, mask, cache,
        layer), computed without kv_b_proj on the cache: q_lat = q_nope W_UK,
        scores = (q_lat . c + q_rope . k_rope) * softmax_scale, o_lat = p c,
        out = sum_h w_ov[h] o_lat_h + o_bias. x is a Tensor or an ndarray
        [B, T, d]; the cache receives the same latents and rope keys as in
        the naive form. Returns [B, T, d] in x's dtype."""
