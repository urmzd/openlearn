# contracts/py/tinyllm/xfmr/masks.pyi (L5.2): attention masks
# chapter: content/courses/build-an-llm/06-transformer-2017/02-masks.md
#
# A mask says which keys each query may read. Every boolean mask here uses
# one convention, the one scaled_dot_product_attention (L5.1) takes:
# True = may attend, False = blocked. Shapes broadcast with numpy rules
# against the scores [..., Tq, Tk]:
#   causal_mask, sliding_window_mask   [Tq, Tk]   (every sequence and head)
#   padding_mask                       [B, T]     (index it as [:, None, None, :]
#                                                  for keys of [B, H, Tq, Tk] scores)
# Query i sits at absolute position q_offset + i and key j at position j,
# so a decode step with a KV cache (L8.2) is causal_mask(1, Tk, q_offset=Tk-1).
# C kernels (L9.3, L9.4) take the same rules as flags: causal, q_offset,
# window.
from numpy.typing import ArrayLike, DTypeLike, NDArray

def causal_mask(Tq: int, Tk: int | None = None, q_offset: int = 0) -> NDArray:
    """bool [Tq, Tk]: True where j <= q_offset + i (a query sees itself and
    the past, never the future). Tk defaults to q_offset + Tq, the keys up to
    the last query. causal_mask(T) is the lower triangle including the
    diagonal; with Tq < Tk and q_offset = Tk - Tq the triangle is aligned to
    the bottom right (the last query sees every key).
    ValueError unless Tq >= 1, Tk >= 1, and q_offset >= 0 are integers."""

def padding_mask(lengths: ArrayLike, T: int) -> NDArray:
    """bool [B, T]: True where t < lengths[b] (a real token), False on the
    padding after it. ValueError unless lengths is 1-D of integers in [0, T]
    and T >= 1."""

def sliding_window_mask(Tq: int, Tk: int, window: int, q_offset: int = 0) -> NDArray:
    """bool [Tq, Tk]: causal and local: True where j <= q_offset + i and
    (q_offset + i) - j < window, so each query sees at most `window` keys,
    itself included (Mistral's sliding window). window >= q_offset + Tq is
    the causal mask. ValueError unless Tq, Tk, window >= 1 and q_offset >= 0
    are integers."""

def combine(*masks: ArrayLike) -> NDArray:
    """The logical AND of one or more bool masks, broadcast together with
    numpy rules (a position is open only if every mask opens it).
    ValueError for no masks, a non-bool mask, or shapes that do not
    broadcast."""

def to_additive(mask: ArrayLike, dtype: DTypeLike = ...) -> NDArray:
    """The additive form the score sum uses: 0.0 where mask is True, -inf
    where it is False, as dtype (default float32; float16, float32, float64
    allowed). scores + to_additive(mask) followed by a softmax gives exactly
    0 weight to blocked keys, and a fully blocked row gives zeros (the stable
    softmax of M09.2), never NaN. ValueError for a non-bool mask or a
    non-float dtype."""
