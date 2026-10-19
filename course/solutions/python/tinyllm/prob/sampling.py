"""Categorical sampling: inverse CDF, Gumbel-max, the alias method (M07.1).

Randomness always arrives from outside: a uniform u in [0, 1), or an `rng`
whose uniform() returns one (PCG32, spec/pcg32.md). So every function here is
a deterministic map from uniforms to outcomes, which is what lets the Python
sampler (L8.1) and the Rust engine (L10.1) emit the same token for the same
seed. All arithmetic is float64.

Contract: contracts/py/tinyllm/prob/sampling.pyi.
"""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray


class UniformSource(Protocol):
    def uniform(self) -> float: ...


def _check_probs(probs: ArrayLike) -> NDArray:
    """probs as float64, after the contract's checks (ValueError otherwise)."""
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 1 or p.size == 0:
        raise ValueError(f"probs must be a non-empty 1-D array, got shape {p.shape}")
    if not np.isfinite(p).all():
        raise ValueError("probs must be finite")
    if (p < 0).any():
        raise ValueError(f"probs must be non-negative, got min {p.min()}")
    total = math.fsum(p.tolist())
    if abs(total - 1.0) > 1e-9 * p.size + 1e-12:
        raise ValueError(f"probs must sum to 1, got {total!r}")
    return p


def sample_categorical(probs: ArrayLike, u: float) -> int:
    p = _check_probs(probs)
    if not 0.0 <= u < 1.0:
        raise ValueError(f"u must lie in [0, 1), got {u!r}")
    c = 0.0
    for i, pi in enumerate(p.tolist()):
        c += pi
        if u < c:
            return i
    # Rounding left the running sum at or below u: the last id that can occur.
    return int(np.flatnonzero(p > 0)[-1])


def gumbel_noise(u: ArrayLike) -> NDArray:
    a = np.asarray(u, dtype=np.float64)
    if not ((a > 0.0) & (a < 1.0)).all():
        raise ValueError("gumbel_noise needs every u in the open interval (0, 1)")
    return -np.log(-np.log(a))


def gumbel_max(logits: ArrayLike, gumbels: ArrayLike) -> int:
    z = np.asarray(logits, dtype=np.float64)
    g = np.asarray(gumbels, dtype=np.float64)
    if z.ndim != 1 or z.size == 0 or g.shape != z.shape:
        raise ValueError(
            f"logits and gumbels must be equal non-empty 1-D arrays, got {z.shape} and {g.shape}"
        )
    if np.isnan(z).any() or (z == np.inf).any():
        raise ValueError("logits must not be NaN or +inf")
    if (z == -np.inf).all():
        raise ValueError("every logit is -inf: there is nothing to sample")
    # -inf + g stays -inf, so a masked id never wins; argmax takes the first maximum.
    return int(np.argmax(z + g))


class AliasTable:
    """Walker's alias method, built by Vose's algorithm."""

    def __init__(self, probs: ArrayLike) -> None:
        p = _check_probs(probs)
        n = p.size
        scaled = (p * n).tolist()  # mean 1: a column holds exactly one unit of mass
        self.prob = np.zeros(n, dtype=np.float64)
        self.alias = np.arange(n, dtype=np.int64)
        small = [i for i in range(n) if scaled[i] < 1.0]
        large = [i for i in range(n) if scaled[i] >= 1.0]
        while small and large:
            s, g = small.pop(), large.pop()
            # Column s keeps its own mass and is topped up from g.
            self.prob[s] = scaled[s]
            self.alias[s] = g
            scaled[g] = (scaled[g] + scaled[s]) - 1.0
            (small if scaled[g] < 1.0 else large).append(g)
        # What is left holds one unit each up to rounding: keep the column whole.
        for i in large + small:
            self.prob[i] = 1.0

    def __len__(self) -> int:
        return int(self.prob.size)

    def sample(self, rng: UniformSource, n: int) -> NDArray:
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
        k = len(self)
        u = np.fromiter((rng.uniform() for _ in range(n)), dtype=np.float64, count=n)
        x = u * k
        i = np.minimum(np.floor(x).astype(np.int64), k - 1)
        f = x - i
        return np.where(f < self.prob[i], i, self.alias[i]).astype(np.int64)


def exponential_icdf(u: float, rate: float) -> float:
    if not 0.0 <= u < 1.0:
        raise ValueError(f"u must lie in [0, 1), got {u!r}")
    if not rate > 0:
        raise ValueError(f"rate must be > 0, got {rate!r}")
    return -math.log1p(-u) / rate


def poisson_arrivals(rate: float, horizon: float, rng: UniformSource) -> NDArray:
    if not rate > 0:
        raise ValueError(f"rate must be > 0, got {rate!r}")
    if not horizon >= 0:
        raise ValueError(f"horizon must be >= 0, got {horizon!r}")
    times: list[float] = []
    t = 0.0
    while True:
        t += exponential_icdf(rng.uniform(), rate)
        if t >= horizon:
            return np.asarray(times, dtype=np.float64)
        times.append(t)
