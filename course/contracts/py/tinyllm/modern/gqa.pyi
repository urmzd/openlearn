# contracts/py/tinyllm/modern/gqa.pyi (L7.5): grouped-query attention with a cache hook,
# a sliding window, and learned sinks
# chapter: content/courses/build-an-llm/08-modern-block/05-gqa-attention.md
#
# H query heads share Hkv key/value heads (Ainslie et al. 2023): kv head j
# serves query heads j * n_rep .. (j + 1) * n_rep - 1, n_rep = H / Hkv.
# Hkv = H is multi-head attention, Hkv = 1 multi-query attention. The KV cache
# holds only the Hkv heads, so it shrinks by n_rep (M05.1 kv_bytes_per_token).
#
# Shapes: x [B, T, d]; q [B, H, T, dh]; k, v [B, Hkv, T, dh] before
# repeat_kv. Scores are q k^T / sqrt(dh).
#
# Parameters, float32; Linear layers drawn from one rng (PCG32, M06.3; None
# means PCG32(0).substream("init")) in the order q, k, v, o:
#   q_proj.weight [H dh, d]    (q_proj.bias [H dh] with qkv_bias)
#   k_proj.weight [Hkv dh, d]  (k_proj.bias with qkv_bias)
#   v_proj.weight [Hkv dh, d]  (v_proj.bias with qkv_bias)
#   o_proj.weight [d, H dh]    no bias
#   sinks         [H]          only with sinks=True; zeros at construction
# state_dict keys are HF's (LlamaAttention; Qwen2Attention with qkv_bias;
# GptOssAttention with sinks) and, like torch, list the module's own
# parameter (sinks) before its children's.
#
# forward(x, positions, mask, cache, layer):
#   1. q, k, v projections, split into heads.
#   2. RoPE (L7.3) on q and k with rope_cos_sin(positions, rope.inv_freq,
#      rope.attention_scaling), rope.layout, rope.rotary_dim. positions is
#      [T] (shared by the batch) or [B, T].
#   3. cache: when given, (k, v) = cache.update(layer, k.data, v.data): the
#      cache appends this chunk's [B, Hkv, T, dh] keys and values and returns
#      everything it holds, [B, Hkv, Tk, dh] numpy arrays, which become
#      constants (inference only: no gradient flows into the cache). Without
#      a cache, Tk = T.
#   4. Visibility: query t (0-based in this chunk) sits at key index
#      i = Tk - T + t. Key j is visible when j <= i (causal, always), and
#      i - j < window when window is set (the query and window - 1 keys
#      before it), and mask allows it. mask (optional) is boolean, True =
#      may attend, broadcastable to [B, H, T, Tk] (padding). Hidden scores
#      are -inf.
#   5. Softmax over the keys. With sinks, each head h appends one extra logit
#      sinks[h] to every row before the softmax and drops its column after:
#      the weights then sum to 1 - p_sink <= 1 (a head may attend to
#      nothing). A row with every key hidden gives weights 0 (output 0), not
#      NaN.
#   6. Weighted values (kv heads repeated by repeat_kv), heads merged, o_proj.
from typing import Any, Optional, Protocol

from numpy.typing import NDArray

from tinyllm.autograd.tensor import Tensor
from tinyllm.modern.rope import RopeSpec
from tinyllm.nn.layers import Linear
from tinyllm.nn.module import Module

class KVCacheHook(Protocol):
    """The seam L8.2's KVCache (and L8.3's paged cache) implements."""

    def update(self, layer: int, k_new: NDArray, v_new: NDArray) -> tuple[NDArray, NDArray]: ...

class ConcatKVCache:
    """The smallest cache that satisfies the hook: per layer, the
    concatenation along the time axis of every chunk it was given."""

    def __init__(self) -> None: ...
    def update(self, layer: int, k_new: NDArray, v_new: NDArray) -> tuple[NDArray, NDArray]:
        """Append [B, Hkv, T, dh] chunks (copies) and return the full arrays.
        ValueError when a chunk's batch, head, or width differs from the
        layer's earlier chunks."""
    def seq_len(self, layer: int = 0) -> int:
        """Positions held for layer (0 for a layer never updated)."""

def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """[B, Hkv, T, dh] -> [B, Hkv n_rep, T, dh]: output head h is input head
    h // n_rep (each kv head repeated n_rep times in a row, HF's repeat_kv,
    torch.repeat_interleave on axis 1; NOT tiled). n_rep = 1 returns x.
    The gradient of a kv head is the sum over its n_rep copies.
    ValueError when n_rep < 1 or x is not 4-D."""

class GQAttention(Module):
    q_proj: Linear
    k_proj: Linear
    v_proj: Linear
    o_proj: Linear
    sinks: Optional[Tensor]
    n_heads: int
    n_kv_heads: int
    d_head: int
    window: Optional[int]

    def __init__(
        self,
        d: int,
        n_heads: int,
        n_kv_heads: int,
        d_head: Optional[int],
        rope: RopeSpec,
        qkv_bias: bool = False,
        window: Optional[int] = None,
        sinks: bool = False,
        rng: Any = None,
    ) -> None:
        """d_head None means d // n_heads. ValueError when n_heads % n_kv_heads
        != 0, a size is below 1, d_head is None and n_heads does not divide d,
        window < 1, or rope.rotary_dim is odd or larger than d_head."""

    def forward(
        self,
        x: Tensor,
        positions: Any,
        mask: Optional[NDArray] = None,
        cache: Optional[KVCacheHook] = None,
        layer: int = 0,
    ) -> Tensor:
        """[B, T, d] -> [B, T, d] as above. ValueError for a wrong x width,
        positions whose shape is not [T] or [B, T], or a mask that does not
        broadcast to [B, H, T, Tk]."""
