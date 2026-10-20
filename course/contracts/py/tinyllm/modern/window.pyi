# contracts/py/tinyllm/modern/window.pyi (L7.7): sliding windows, StreamingLLM sink tokens, learned sinks
# chapter: content/courses/ai-engineering/13-decoder-block/07-sliding-window-and-sinks.md
#
# Three ways to bound what a query reads, and so what the KV cache must hold:
#
#   sliding window W   (Mistral, Beltagy et al. 2020)  query at position p reads
#                      keys p - W + 1 .. p: itself and W - 1 before it
#   sink tokens S      (StreamingLLM, Xiao et al. 2023) the first S tokens of
#                      the stream stay readable forever, on top of the window:
#                      softmax needs somewhere to put weight it does not want
#                      to give, and models learn to dump it on token 0
#   learned sinks      (gpt-oss) one extra logit per head in every softmax
#                      row, attached to no value: the head may attend to
#                      nothing (L7.5's GQAttention(sinks=True))
#
# The visibility rule, for query i of a chunk at absolute position
# p_i = q_offset + i and a key at absolute position j:
#
#     visible(i, j) = j <= p_i  and  (j < S  or  p_i - j < W)
#
# window None means no window (causal only, S is then irrelevant).
# Every boolean mask uses L5.2's convention: True = may attend.
#
# SinkWindowCache keeps the sinks and the last W - 1 positions of each layer
# (what the NEXT query needs besides itself), so a stream of any length
# holds at most S + W - 1 keys per layer between updates. Keys are cached as
# the attention layer hands them over (rotated at their absolute positions),
# so attention with this cache equals full attention under the mask above.
from typing import Any, Optional

from numpy.typing import ArrayLike, NDArray

def sink_window_mask(Tq: int, Tk: int, window: Optional[int], n_sink: int = 0, q_offset: int = 0) -> NDArray:
    """bool [Tq, Tk]: visible(i, j) above, keys at positions 0 .. Tk - 1,
    queries at q_offset .. q_offset + Tq - 1. With n_sink 0 it equals L5.2's
    sliding_window_mask; with window None, L5.2's causal_mask. ValueError
    unless Tq, Tk >= 1, window is None or >= 1, n_sink >= 0, q_offset >= 0,
    all integers."""

def windowed_attention(
    q: ArrayLike,
    k: ArrayLike,
    v: ArrayLike,
    window: Optional[int] = None,
    n_sink: int = 0,
    sink_logits: Optional[ArrayLike] = None,
    q_offset: int = 0,
    scale: Optional[float] = None,
) -> tuple[NDArray, NDArray]:
    """The numpy reference L9.3's FlashAttention kernel is checked against.
    q [B, H, Tq, dh], k [B, Hkv, Tk, dh], v [B, Hkv, Tk, dv]; query head h
    reads kv head h // (H / Hkv). Scores q . k * scale (default dh ** -0.5)
    over the visible keys (sink_window_mask(Tq, Tk, window, n_sink,
    q_offset)). With sink_logits [H], each row's softmax denominator also
    holds exp(sink_logits[h]), attached to no value, so the weights sum to
    1 - p_sink. Computed in float64. Returns (out [B, H, Tq, dv],
    lse [B, H, Tq]): lse = log(sum over visible keys of exp(score) +
    exp(sink_logit)), the log of the softmax denominator. A row with no
    visible key and no sink gives out 0 and lse -inf (never NaN).
    ValueError for shapes that disagree, H % Hkv != 0, or sink_logits not of
    shape [H]."""

class SinkWindowCache:
    """A KV cache hook (L7.5's KVCacheHook) that evicts everything except the
    sink tokens and the recent window."""

    n_sink: int
    window: int

    def __init__(self, n_sink: int, window: int) -> None:
        """ValueError unless n_sink >= 0 and window >= 1 are integers."""

    def update(self, layer: int, k_new: NDArray, v_new: NDArray) -> tuple[NDArray, NDArray]:
        """Append this chunk's [B, Hkv, T, dh] keys and values (copies, at
        absolute positions seq_len(layer) .. + T - 1) and return what the
        chunk's queries may need: the held sinks and recent keys, then the
        chunk, in position order, [B, Hkv, Tk, dh]. Afterwards evict down to
        the positions < n_sink and the last window - 1 positions.
        ValueError when a chunk's batch, head, or width differs from the
        layer's earlier chunks."""

    def seq_len(self, layer: int = 0) -> int:
        """Tokens this layer has seen in total (the next chunk's first
        absolute position); 0 for a layer never updated."""

    def held(self, layer: int = 0) -> int:
        """Keys held now for layer, after eviction: at most
        n_sink + window - 1."""

    def positions(self, layer: int = 0) -> NDArray:
        """int64 [Tk]: the absolute positions of the keys the last update of
        layer returned, in order (empty before any update)."""

    def chunk_mask(self, layer: int, T: int) -> NDArray:
        """bool [T, Tk]: the mask to pass with the NEXT update of T tokens:
        query i (position seq_len + i) against the held keys followed by the
        chunk, by the visibility rule above. ValueError for T < 1."""

def attention_mask(cache: Any, layer: int, T: int, window: Optional[int], n_sink: int = 0) -> Optional[NDArray]:
    """The mask an attention layer (L7.5, L7.6) needs for a chunk of T
    queries under (window, n_sink), whatever the cache:
      window None               None (the layer's causal rule is enough)
      cache None                sink_window_mask(T, T, window, n_sink)
      a SinkWindowCache         cache.chunk_mask(layer, T)
      any cache with seq_len    sink_window_mask(T, s + T, window, n_sink, s),
                                s = cache.seq_len(layer) (it holds every key)
    ValueError for a cache without seq_len."""
