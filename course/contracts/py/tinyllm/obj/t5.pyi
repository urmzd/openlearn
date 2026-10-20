# contracts/py/tinyllm/obj/t5.pyi (L6.4, optional): T5 span corruption and relative position buckets
# chapter: content/courses/ai-engineering/12-objectives-and-adaptation/04-t5-span-corruption-and-relative-buckets.md
#
# Raffel et al. (2020). Span corruption picks a noise mask of whole spans,
# T5's random_spans_noise_mask:
#
#   n_noise = clamp(round(L * noise_density), 1, L - 1)       round: half to even (np.round)
#   n_spans = clamp(round(n_noise / mean_noise_span), 1, min(n_noise, L - n_noise))
#   noise lengths = segment(n_noise, n_spans); keep lengths = segment(L - n_noise, n_spans)
#   the text is keep_1, noise_1, keep_2, noise_2, ..., keep_n, noise_n
#
# segment(m, n) splits m items into n positive lengths: a list of m - 1
# flags, the first n - 1 True, shuffled by Fisher-Yates from the end
# (for i = m - 2 down to 1: j = rng.below(i + 1), swap i and j); a segment
# starts after every True flag. The noise lengths are drawn first. (T5
# itself clamps n_spans only from below; the upper clamp keeps every span
# and every gap non-empty for short texts.)
#
# Sentinel k (k = 0, 1, ...) is the id sentinel_start_id - k (T5's
# <extra_id_k> = vocab - 1 - k). inputs: the kept tokens with each noise span
# replaced by its sentinel; targets: each noise span preceded by its
# sentinel. Replacing every sentinel of inputs by what follows it in targets
# gives the original ids back.
#
# Relative positions are key - query (memory_position - context_position).
# Buckets as Hugging Face's T5Attention._relative_position_bucket: with
# n = num_buckets (halved when bidirectional, keys after the query in the
# upper half), offsets below n // 2 get their own bucket, larger ones
#   n // 2 + trunc(log(r / (n // 2)) / log(max_distance / (n // 2)) * (n - n // 2))
# in float32, capped at n - 1.
from typing import Any, Optional

from numpy.typing import ArrayLike, NDArray

from tinyllm.autograd.tensor import Tensor
from tinyllm.nn.layers import Embedding
from tinyllm.nn.module import Module

def noise_span_counts(length: int, noise_density: float, mean_noise_span: float) -> tuple[int, int]:
    """(n_noise, n_spans) as above. ValueError for length < 2, noise_density
    outside (0, 1), or mean_noise_span < 1."""

def random_spans_noise_mask(length: int, noise_density: float, mean_noise_span: float, rng: Any) -> NDArray:
    """bool [length], True on noise tokens, as above. rng needs below(n)
    (a PCG32, M06.3). ValueError as noise_span_counts."""

def span_corrupt(
    ids: ArrayLike,
    noise_density: float,
    mean_noise_span: float,
    sentinel_start_id: int,
    rng: Any,
    eos_id: Optional[int] = None,
) -> tuple[NDArray, NDArray]:
    """(inputs, targets), int64, from one random_spans_noise_mask(len(ids),
    ...). With eos_id, it is appended to both. len(inputs) = L - n_noise +
    n_spans, len(targets) = n_noise + n_spans (each + 1 with eos_id).
    ValueError for ids not a 1-D integer array, as noise_span_counts, when
    sentinel_start_id - n_spans + 1 < 0, or when an id lies in the sentinel
    range [sentinel_start_id - n_spans + 1, sentinel_start_id]."""

def t5_relative_bucket(
    rel_pos: ArrayLike, bidirectional: bool, num_buckets: int = 32, max_distance: int = 128
) -> NDArray:
    """int64 buckets in [0, num_buckets), the shape of rel_pos, as above.
    ValueError when n // 2 < 1 or max_distance <= n // 2."""

class T5RelativeBias(Module):
    n_heads: int
    num_buckets: int
    max_distance: int
    bidirectional: bool
    relative_attention_bias: Embedding  # [num_buckets, n_heads], HF's name

    def __init__(
        self,
        n_heads: int,
        num_buckets: int = 32,
        max_distance: int = 128,
        bidirectional: bool = True,
        rng: Any = None,
    ) -> None:
        """An L0.4 Embedding built from rng. bidirectional False is T5's decoder.
        ValueError for n_heads < 1."""

    def forward(self, q_len: int, k_len: int, q_offset: int = 0) -> Tensor:
        """[n_heads, q_len, k_len]: bias[h, i, j] = weight[bucket(j - (i +
        q_offset)), h], added to head h's attention logits (T5 does not scale
        it). q_offset counts tokens already in the cache (HF's
        past_seen_tokens). ValueError for q_len or k_len < 1, q_offset < 0."""
