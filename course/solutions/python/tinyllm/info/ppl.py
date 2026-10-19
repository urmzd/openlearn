"""Perplexity, bits per byte, and a streaming NLL accumulator (M11.2).

Every evaluation in the course reduces to one total: the sum of per-token
negative log-likelihoods in nats. This module keeps that total exactly
enough over 10^7 tokens (fsum per batch, compensated summation across
batches), counts only real tokens (masks), and turns it into the numbers
the model zoo reports.

Contract: contracts/py/tinyllm/info/ppl.pyi.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

from tinyllm.num.units import bits_per_byte


def perplexity(nll_sum: float, n_tokens: int) -> float:
    if n_tokens < 1:
        raise ValueError(
            f"perplexity needs at least one token, got n_tokens = {n_tokens}"
        )
    if math.isnan(nll_sum) or nll_sum < 0:
        raise ValueError(
            f"nll_sum must be a non-negative number of nats, got {nll_sum!r}"
        )
    try:
        return math.exp(nll_sum / n_tokens)
    except OverflowError:  # a mean above ln(DBL_MAX) = 709.78 nats
        return math.inf


class NLLAccumulator:
    """Streaming total of per-token NLLs (nats), tokens, and bytes."""

    def __init__(self) -> None:
        self._sum = 0.0  # Neumaier: the total is _sum + _comp
        self._comp = 0.0
        self.n_tokens = 0
        self.n_bytes = 0

    def _accumulate(self, x: float) -> None:
        """Neumaier's compensated add of one float into the running total."""
        if math.isinf(x) or math.isinf(self._sum):
            self._sum += x
            return
        t = self._sum + x
        if abs(self._sum) >= abs(x):
            self._comp += (self._sum - t) + x  # the low bits of x that t lost
        else:
            self._comp += (x - t) + self._sum
        self._sum = t

    def add(
        self, nll: ArrayLike, mask: ArrayLike | None = None, n_bytes: int = 0
    ) -> None:
        x = np.asarray(nll, dtype=np.float64)
        if mask is None:
            keep = np.ones(x.shape, dtype=bool)
        else:
            keep = np.asarray(mask)
            if keep.shape != x.shape:
                raise ValueError(
                    f"mask shape {keep.shape} differs from nll shape {x.shape}"
                )
            keep = keep.astype(bool)
        if n_bytes < 0:
            raise ValueError(f"n_bytes must be >= 0, got {n_bytes}")
        # Select, never multiply: padding may hold NaN, and NaN * 0 is NaN.
        vals = x[keep]
        if np.isnan(vals).any() or (vals < 0).any():
            raise ValueError("a counted NLL is negative or NaN")
        self._accumulate(math.fsum(vals.tolist()))
        self.n_tokens += int(vals.size)
        self.n_bytes += int(n_bytes)

    def merge(self, other: NLLAccumulator) -> None:
        self._accumulate(other._sum)
        self._accumulate(other._comp)
        self.n_tokens += other.n_tokens
        self.n_bytes += other.n_bytes

    def result(self) -> dict[str, float]:
        if self.n_tokens == 0:
            raise ValueError("no tokens were counted")
        total = self._sum + self._comp if math.isfinite(self._sum) else self._sum
        mean = total / self.n_tokens
        return {
            "nll_sum": float(total),
            "n_tokens": float(self.n_tokens),
            "n_bytes": float(self.n_bytes),
            "nll_mean": float(mean),
            "ppl": perplexity(total, self.n_tokens),
            "bits_per_token": float(mean / math.log(2.0)),
            "bpb": bits_per_byte(total, self.n_bytes) if self.n_bytes > 0 else math.nan,
        }
