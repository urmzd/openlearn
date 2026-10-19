"""tinyllm.prob.rv (M07.0): discrete random variables and normal draws.

A discrete random variable X is a table of values x_i with probabilities
p_i >= 0 that sum to 1. Its expectation is the probability-weighted mean and
its variance the expected squared distance from that mean. Normal draws come
from Box-Muller, which turns two independent uniforms into two independent
standard normals, using the PCG32 uniforms of spec/pcg32.md so a seed means
the same normals in every language.

Contract: contracts/py/tinyllm/prob/rv.pyi.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:  # the generator every caller passes (M06.3)
    from tinyllm.num.rng import PCG32  # noqa: F401


def _table(values: ArrayLike, probs: ArrayLike) -> tuple[NDArray, NDArray]:
    """values and probs as float64 vectors, checked to be a distribution."""
    x = np.asarray(values, dtype=np.float64)
    p = np.asarray(probs, dtype=np.float64)
    if x.ndim != 1 or p.ndim != 1 or x.shape != p.shape or x.size == 0:
        raise ValueError(f"values and probs must be 1-D of one non-zero length, got {x.shape} and {p.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError("values must be finite")
    if np.any(p < 0) or not np.all(np.isfinite(p)):
        raise ValueError("probs must be finite and >= 0")
    if abs(float(p.sum()) - 1.0) > 1e-9:
        raise ValueError(f"probs must sum to 1, got {float(p.sum())!r}")
    return x, p


def expectation(values: ArrayLike, probs: ArrayLike) -> float:
    x, p = _table(values, probs)
    return float(x @ p)


def variance(values: ArrayLike, probs: ArrayLike) -> float:
    x, p = _table(values, probs)
    mu = float(x @ p)
    d = x - mu  # two passes: center first, then square
    return float((d * d) @ p)


def box_muller(u1: float, u2: float) -> tuple[float, float]:
    # 1 - u1 lies in (0, 1], so the log is finite and r is in [0, inf).
    r = math.sqrt(-2.0 * math.log(1.0 - u1))
    theta = 2.0 * math.pi * u2
    return r * math.cos(theta), r * math.sin(theta)


def normal(rng: Any, n: int) -> NDArray:
    if n < 0:
        raise ValueError(f"normal: n must be >= 0, got {n}")
    out = np.empty(n, dtype=np.float64)
    for i in range(0, n, 2):
        u1 = rng.uniform()
        u2 = rng.uniform()
        z0, z1 = box_muller(u1, u2)
        out[i] = z0
        if i + 1 < n:
            out[i + 1] = z1
    return out
