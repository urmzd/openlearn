# contracts/py/tinyllm/infer/kvcache.pyi (L8.2)
# chapter: content/courses/build-an-llm/p08-inference/02-kv-cache-and-generate.md
#
# Contiguous KV caches for incremental decoding. Keys and values of every
# position already processed are kept per layer, so a decode step computes
# attention for ONE new query against all cached keys instead of re-running
# the whole prefix. KVCache implements L7.5's KVCacheHook (update(layer,
# k_new, v_new)); LatentCache implements L7.6's LatentCacheHook (MLA caches a
# compressed latent and a small RoPE key per position, not per-head K and V).
#
# Storage is preallocated: [batch, n_kv_heads, max_len, d_head] per layer for
# K and for V, in `dtype` (float32, or float16 to halve the bytes; L8.3
# compares its paged cache against this one with float16). Each layer has its
# own fill count; `seq_len()` is the COMMITTED length, the minimum over
# layers, so it moves only after the last layer of a forward pass has
# appended (`seq_len(layer)` is one layer's count, L7.5's convention). A
# model reads `positions(T)` and `mask(T)` once at the start of a forward
# pass, before any update.
from typing import Any, Optional

from numpy.typing import NDArray

class KVCache:
    n_layers: int
    n_kv_heads: int
    d_head: int
    max_len: int
    batch: int
    dtype: Any  # a numpy floating dtype

    def __init__(
        self,
        n_layers: int,
        n_kv_heads: int,
        d_head: int,
        max_len: int,
        batch: int = 1,
        dtype: Any = ...,
    ) -> None:
        """Zeroed storage, every fill count 0. dtype defaults to float32 and
        must be a numpy floating dtype. ValueError for a size below 1 or a
        non-floating dtype."""

    def seq_len(self, layer: Optional[int] = None) -> int:
        """With no layer, the positions committed by every layer: the minimum
        of the fill counts. With a layer, that layer's fill count (what L7.7's
        attention_mask and L7.9's forward read as seq_len(layer), the same
        convention as L7.5's ConcatKVCache)."""

    def update(self, layer: int, k_new: NDArray, v_new: NDArray) -> tuple[NDArray, NDArray]:
        """Append a chunk of T positions to `layer` and return everything the
        layer holds: float32 arrays [batch, n_kv_heads, n, d_head] with
        n = the layer's fill count after the append. The chunk is converted
        to `dtype` on the way in and the returned arrays hold the STORED
        values (with float16, the new chunk comes back rounded too). The
        inputs are copied, never kept. ValueError for a layer outside
        [0, n_layers), a chunk not shaped [batch, n_kv_heads, T, d_head] with
        T >= 1, or an append past max_len (the cache is left unchanged)."""

    def positions(self, t_new: int) -> NDArray:
        """int64 [t_new]: seq_len(), seq_len() + 1, ..., the absolute positions of
        the next t_new tokens (what RoPE rotates by). ValueError for
        t_new < 1."""

    def mask(self, t_new: int) -> NDArray:
        """bool [t_new, seq_len() + t_new]: L5.2's causal_mask(t_new,
        seq_len() + t_new, q_offset=seq_len()), True where the query may attend.
        ValueError for t_new < 1."""

    def truncate(self, n: int) -> None:
        """Forget every position >= n in every layer (fill counts become n):
        the rollback speculative decoding (L8.6) needs after rejected drafts.
        Later updates overwrite from position n. ValueError unless
        0 <= n <= seq_len()."""

    def nbytes(self) -> int:
        """Bytes of the preallocated storage: 2 * n_layers * batch *
        n_kv_heads * max_len * d_head * itemsize."""

class LatentCache:
    """The same contract for MLA's cache hook: per layer, the latent c
    [batch, T, kv_lora_rank] and the shared RoPE key [batch, T, rope_dim]."""

    n_layers: int
    kv_lora_rank: int
    rope_dim: int
    max_len: int
    batch: int
    dtype: Any

    def __init__(
        self,
        n_layers: int,
        kv_lora_rank: int,
        rope_dim: int,
        max_len: int,
        batch: int = 1,
        dtype: Any = ...,
    ) -> None: ...
    def seq_len(self, layer: Optional[int] = None) -> int: ...
    def update(self, layer: int, c_new: NDArray, k_rope_new: NDArray) -> tuple[NDArray, NDArray]:
        """Append [batch, T, kv_lora_rank] and [batch, T, rope_dim] chunks to
        `layer`; return the full float32 arrays [batch, n, ...]. Same rules
        as KVCache.update."""
    def positions(self, t_new: int) -> NDArray: ...
    def mask(self, t_new: int) -> NDArray: ...
    def truncate(self, n: int) -> None: ...
    def nbytes(self) -> int:
        """n_layers * batch * max_len * (kv_lora_rank + rope_dim) * itemsize."""
