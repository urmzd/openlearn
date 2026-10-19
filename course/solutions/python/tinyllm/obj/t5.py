"""T5's span corruption and relative position buckets (L6.4, optional).

Span corruption (Raffel et al. 2020) turns any text into a denoising task:
drop about 15% of the tokens in a few contiguous spans, put one sentinel id
where each span was, and ask the model to write the dropped spans, each after
its sentinel. Inputs and targets together hold every original token exactly
once, so the task is self-supervised and the targets are short.

T5 has no position embeddings. Each attention head adds a learned scalar
bias that depends only on the key's offset from the query, bucketed: exact
for small offsets, logarithmic up to max_distance, one bucket beyond.

Contract: contracts/py/tinyllm/obj/t5.pyi.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tinyllm.autograd import functional as F
from tinyllm.autograd.tensor import Tensor
from tinyllm.nn.layers import Embedding
from tinyllm.nn.module import Module


def _segment_lengths(n_items: int, n_segments: int, rng: Any) -> NDArray:
    """Split n_items into n_segments positive lengths, uniformly at random."""
    # n_items - 1 gaps, n_segments - 1 of them start a new segment; the gaps
    # are shuffled with Fisher-Yates from the end, one rng.below per swap.
    starts = [i < n_segments - 1 for i in range(n_items - 1)]
    for i in range(len(starts) - 1, 0, -1):
        j = rng.below(i + 1)
        starts[i], starts[j] = starts[j], starts[i]
    first = np.array([False] + starts, dtype=bool)
    seg = np.cumsum(first)
    return np.bincount(seg, minlength=n_segments).astype(np.int64)


def noise_span_counts(length: int, noise_density: float, mean_noise_span: float) -> tuple[int, int]:
    if length < 2:
        raise ValueError(f"span corruption needs at least 2 tokens, got {length}")
    if not 0.0 < noise_density < 1.0:
        raise ValueError(f"noise_density must be in (0, 1), got {noise_density}")
    if not mean_noise_span >= 1.0:
        raise ValueError(f"mean_noise_span must be >= 1, got {mean_noise_span}")
    n_noise = int(np.round(length * noise_density))
    n_noise = min(max(n_noise, 1), length - 1)
    n_spans = max(int(np.round(n_noise / mean_noise_span)), 1)
    # Every span needs at least one token, and so does every gap before one.
    n_spans = min(n_spans, n_noise, length - n_noise)
    return n_noise, n_spans


def random_spans_noise_mask(length: int, noise_density: float, mean_noise_span: float, rng: Any) -> NDArray:
    n_noise, n_spans = noise_span_counts(length, noise_density, mean_noise_span)
    noise = _segment_lengths(n_noise, n_spans, rng)
    keep = _segment_lengths(length - n_noise, n_spans, rng)
    # keep_1, noise_1, keep_2, noise_2, ...: the text starts with kept tokens
    # and ends with a noise span.
    mask = np.zeros(length, dtype=bool)
    pos = 0
    for k, x in zip(keep, noise):
        pos += int(k)
        mask[pos : pos + int(x)] = True
        pos += int(x)
    return mask


def span_corrupt(
    ids: ArrayLike,
    noise_density: float,
    mean_noise_span: float,
    sentinel_start_id: int,
    rng: Any,
    eos_id: Optional[int] = None,
) -> tuple[NDArray, NDArray]:
    a = np.asarray(ids)
    if a.ndim != 1 or (a.size and a.dtype.kind not in "iu"):
        raise ValueError(f"ids must be a 1-D integer array, got {a.dtype} {a.shape}")
    a = a.astype(np.int64)
    mask = random_spans_noise_mask(a.size, noise_density, mean_noise_span, rng)
    n_spans = int(np.sum(mask[1:] & ~mask[:-1]) + mask[0])
    lowest = sentinel_start_id - n_spans + 1
    if lowest < 0:
        raise ValueError(f"sentinel_start_id {sentinel_start_id} leaves no room for {n_spans} sentinels")
    if np.any((a >= lowest) & (a <= sentinel_start_id)):
        raise ValueError(f"ids use the sentinel range [{lowest}, {sentinel_start_id}]")
    inputs: list[int] = []
    targets: list[int] = []
    k = -1
    for t in range(a.size):
        if mask[t]:
            if t == 0 or not mask[t - 1]:  # a new noise span: sentinel k
                k += 1
                inputs.append(sentinel_start_id - k)
                targets.append(sentinel_start_id - k)
            targets.append(int(a[t]))
        else:
            inputs.append(int(a[t]))
    if eos_id is not None:
        inputs.append(int(eos_id))
        targets.append(int(eos_id))
    return np.array(inputs, dtype=np.int64), np.array(targets, dtype=np.int64)


def t5_relative_bucket(
    rel_pos: ArrayLike, bidirectional: bool, num_buckets: int = 32, max_distance: int = 128
) -> NDArray:
    rel = np.asarray(rel_pos, dtype=np.int64)
    n = int(num_buckets)
    out = np.zeros(rel.shape, dtype=np.int64)
    if bidirectional:
        n //= 2
        out += (rel > 0).astype(np.int64) * n  # keys after the query: upper half
        rel = np.abs(rel)
    else:
        rel = -np.minimum(rel, 0)  # a causal decoder only sees keys at or before the query
    max_exact = n // 2
    if max_exact < 1 or max_distance <= max_exact:
        raise ValueError(f"need num_buckets >= {4 if bidirectional else 2} and max_distance > max_exact")
    is_small = rel < max_exact
    # T5's float32 arithmetic (as torch computes it), truncated toward zero.
    # Small offsets take the exact branch, so they are lifted to max_exact
    # here only to keep log away from 0.
    logr = np.log(np.maximum(rel, max_exact).astype(np.float32) / np.float32(max_exact))
    large = max_exact + (
        logr / np.float32(math.log(max_distance / max_exact)) * np.float32(n - max_exact)
    ).astype(np.int64)
    large = np.minimum(large, n - 1)
    return out + np.where(is_small, rel, large)


class T5RelativeBias(Module):
    def __init__(
        self,
        n_heads: int,
        num_buckets: int = 32,
        max_distance: int = 128,
        bidirectional: bool = True,
        rng: Any = None,
    ) -> None:
        super().__init__()
        if n_heads < 1:
            raise ValueError(f"n_heads must be >= 1, got {n_heads}")
        self.n_heads, self.num_buckets = int(n_heads), int(num_buckets)
        self.max_distance, self.bidirectional = int(max_distance), bool(bidirectional)
        self.relative_attention_bias = Embedding(self.num_buckets, self.n_heads, rng=rng)

    def forward(self, q_len: int, k_len: int, q_offset: int = 0) -> Tensor:
        if q_len < 1 or k_len < 1 or q_offset < 0:
            raise ValueError(f"need q_len, k_len >= 1 and q_offset >= 0, got {q_len}, {k_len}, {q_offset}")
        q = np.arange(q_len, dtype=np.int64)[:, None] + q_offset
        k = np.arange(k_len, dtype=np.int64)[None, :]
        buckets = t5_relative_bucket(k - q, self.bidirectional, self.num_buckets, self.max_distance)
        bias = self.relative_attention_bias(buckets)  # [q, k, heads]
        return F.permute(bias, (2, 0, 1))
