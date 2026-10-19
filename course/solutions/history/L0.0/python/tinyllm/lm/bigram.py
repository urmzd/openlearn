"""Byte-level bigram language model from counts (L0.0, the tracer).

The model is one table, `weight`, of shape [V, V]. Row i holds the logits of
the next token after token i, so softmax(weight[i]) is the model's
distribution over what follows i. fit_counts fills the table with the log of
add-alpha smoothed bigram frequencies; logits gathers the requested rows with NumPy.

Contract: contracts/py/tinyllm/lm/bigram.pyi.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray



def _as_ids(ids: ArrayLike, vocab_size: int) -> NDArray[np.int64]:
    """ids as a 1-D int64 array, every value checked against [0, vocab_size)."""
    a = np.asarray(ids)
    if a.ndim != 1:
        raise ValueError(f"ids must be 1-D, got shape {a.shape}")
    if a.size == 0:
        return a.astype(np.int64)
    if not np.issubdtype(a.dtype, np.integer):
        raise ValueError(f"ids must be integers, got dtype {a.dtype}")
    a = a.astype(np.int64)
    lo, hi = int(a.min()), int(a.max())
    if lo < 0 or hi >= vocab_size:
        raise ValueError(f"ids must lie in [0, {vocab_size}), got min {lo} and max {hi}")
    return a



class BigramLM:
    """A [V, V] table of next-token logits. Fit by counting, read with NumPy."""

    def __init__(self, weight: ArrayLike | None = None) -> None:
        self.weight = None
        if weight is not None:
            w = np.asarray(weight)
            if w.dtype != np.float32:
                raise ValueError(f"weight must be float32, got {w.dtype}")
            if w.ndim != 2 or w.shape[0] != w.shape[1] or w.shape[0] == 0:
                raise ValueError(f"weight must be a non-empty square matrix, got shape {w.shape}")
            if not np.isfinite(w).all():
                raise ValueError("weight must be finite (weights must be finite)")
            self.weight = np.ascontiguousarray(w)

    @property
    def vocab_size(self) -> int:
        if self.weight is None:
            raise RuntimeError("BigramLM has no weight yet: call fit_counts or pass a weight")
        return int(self.weight.shape[0])

    def fit_counts(self, ids: ArrayLike, vocab_size: int, alpha: float = 1.0) -> None:
        if not alpha > 0:
            raise ValueError(f"alpha must be > 0 (alpha = 0 gives log 0 = -inf for unseen pairs), got {alpha}")
        if vocab_size < 1:
            raise ValueError(f"vocab_size must be >= 1, got {vocab_size}")
        a = _as_ids(ids, vocab_size)
        counts = np.zeros((vocab_size, vocab_size), dtype=np.float64)
        # Row = the token we are at, column = the token that follows it.
        np.add.at(counts, (a[:-1], a[1:]), 1.0)
        smoothed = counts + alpha
        probs = smoothed / smoothed.sum(axis=1, keepdims=True)
        self.weight = np.log(probs).astype(np.float32)

    def logits(self, ids: ArrayLike) -> NDArray:
        v = self.vocab_size
        a = _as_ids(ids, v)
        return self.weight[a]

    def nll(self, ids: ArrayLike) -> float:
        a = _as_ids(ids, self.vocab_size)
        if a.size < 2:
            raise ValueError("nll needs at least two ids: one context and one prediction")
        z = self.logits(a[:-1]).astype(np.float64)
        z = z - z.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return float(-logp[np.arange(a.size - 1), a[1:]].mean())

    def sample(self, prefix: list[int], n: int, temperature: float, seed: int) -> list[int]:
        v = self.vocab_size
        if len(prefix) == 0:
            raise ValueError("prefix must hold at least one id: the byte tokenizer has no start token")
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
        if not temperature >= 0:
            raise ValueError(f"temperature must be >= 0, got {temperature}")
        cur = int(_as_ids(prefix, v)[-1])
        rng = np.random.default_rng(seed)
        out: list[int] = []
        for _ in range(n):
            row = self.logits([cur])[0].astype(np.float64)
            if temperature == 0:
                nxt = int(np.argmax(row))  # first maximum: ties go to the lowest id
            else:
                z = row / temperature
                p = np.exp(z - z.max())
                cdf = np.cumsum(p / p.sum())
                u = rng.random()
                nxt = min(int(np.searchsorted(cdf, u, side="right")), v - 1)
            out.append(nxt)
            cur = nxt
        return out
