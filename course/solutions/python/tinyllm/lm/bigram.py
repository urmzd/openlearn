"""Byte-level bigram language model: from counts (L0.0) and by gradient
descent (L0.5, which took this unit over).

The model is one table, `weight`, of shape [V, V]. Row i holds the logits of
the next token after token i, so softmax(weight[i]) is the model's
distribution over what follows i. BigramLM.fit_counts fills the table with
the log of add-alpha smoothed bigram frequencies; logits gathers rows with
NumPy. BigramLogits is the same table as a Module (L0.4) trained with the
course's autograd; to_lm() turns it into a BigramLM the CLI saves and the
unchanged tracer engine serves. Sampling draws from PCG32 (M06.3) exactly as
the Rust tracer engine does (L10.0), so a seed means the same text in both.

Contract: contracts/py/tinyllm/lm/bigram.pyi.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tinyllm.autograd import functional as F
from tinyllm.autograd.tensor import Tensor
from tinyllm.nn.module import Module
from tinyllm.num.rng import PCG32


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
                raise ValueError("weight must be finite (a -inf logit makes 0 * -inf = nan in the matmul)")
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
        # One generator per call, stream 54: the tracer engine's PCG32(seed, 54).
        rng = PCG32(seed)
        out: list[int] = []
        for _ in range(n):
            row = self.logits([cur])[0].astype(np.float64)
            if temperature == 0:
                nxt = int(np.argmax(row))  # first maximum: ties go to the lowest id
            else:
                nxt = _draw(row, float(temperature), rng.uniform())
            out.append(nxt)
            cur = nxt
        return out


def _draw(row: NDArray, temperature: float, u: float) -> int:
    """The tracer engine's draw (L10.0): weights exp(z - max z) of z = row /
    temperature in f64, summed in id order, NaN never picked; the first id
    whose running sum exceeds u * total."""
    z = [float(x) / temperature for x in row]
    finite = [x for x in z if not math.isnan(x)]
    m = max(finite) if finite else -math.inf
    w = [0.0 if math.isnan(x) or m == -math.inf else math.exp(x - m) for x in z]
    total = 0.0
    for wi in w:
        total += wi
    target = u * total
    cum, last = 0.0, 0
    for i, wi in enumerate(w):
        if wi > 0.0:
            cum += wi
            last = i
            if cum > target:
                return i
    return last


class BigramLogits(Module):
    """The bigram table as a trainable Module: logits for ids are rows of weight."""

    def __init__(self, vocab: int = 256) -> None:
        super().__init__()
        if vocab < 1:
            raise ValueError(f"vocab must be >= 1, got {vocab}")
        # Zeros: every next token starts equally likely (NLL ln V), and no
        # random init is needed for a convex problem.
        self.weight = Tensor(np.zeros((vocab, vocab)), requires_grad=True)

    def forward(self, ids: ArrayLike) -> Tensor:
        # A row gather is the one-hot matmul without the zeros: same logits,
        # and backward adds each position's gradient into its row. Any shape
        # of ids: the trainer feeds [B, T] windows. F.embedding checks them.
        return F.embedding(self.weight, ids)

    def to_lm(self) -> BigramLM:
        return BigramLM(self.weight.data.astype(np.float32, copy=True))
