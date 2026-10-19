"""Rejection sampling and residual distributions (M07.6).

A draft token x drawn from a cheap proposal q is kept with probability
min(1, p_x / q_x); on rejection a replacement comes from the residual
normalize(max(0, p - q)). The two branches add up to exactly p, which is why
speculative decoding (L8.6, L10.8) changes the speed of generation and not
its distribution. Von Neumann's classic sampler is the same idea with an
envelope m and a retry loop instead of a residual.

Every sum is a left-to-right loop over ascending ids in float64, so the Rust
port reproduces it bit for bit.

Contract: contracts/py/tinyllm/prob/rejection.pyi.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tinyllm.prob.sampling import UniformSource, sample_categorical


def _dist(x: ArrayLike, name: str) -> NDArray:
    """x as a checked float64 distribution (M07.1's rules)."""
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1 or a.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array, got shape {a.shape}")
    if not np.isfinite(a).all() or (a < 0).any():
        raise ValueError(f"{name} must be finite and non-negative")
    total = math.fsum(a.tolist())
    if abs(total - 1.0) > 1e-9 * a.size + 1e-12:
        raise ValueError(f"{name} must sum to 1, got {total!r}")
    return a


def _pair(p: ArrayLike, q: ArrayLike) -> tuple[NDArray, NDArray]:
    a, b = _dist(p, "p"), _dist(q, "q")
    if a.shape != b.shape:
        raise ValueError(f"p and q differ in shape: {a.shape} vs {b.shape}")
    return a, b


def _check_u(u: float, name: str) -> None:
    if not 0.0 <= u < 1.0:
        raise ValueError(f"{name} must lie in [0, 1), got {u!r}")


def rejection_accept(p_x: float, q_x: float, u: float) -> bool:
    _check_u(u, "u")
    if not q_x > 0.0:
        raise ValueError(f"q_x must be > 0 (q proposed this token), got {q_x!r}")
    if not p_x >= 0.0:
        raise ValueError(f"p_x must be >= 0, got {p_x!r}")
    # P(u < r) = min(1, r) for u uniform on [0, 1): strict, so p_x = 0 (r = 0)
    # never accepts and r >= 1 always does.
    return u < p_x / q_x


def acceptance_probability(p: ArrayLike, q: ArrayLike) -> float:
    a, b = _pair(p, q)
    s = 0.0
    for pi, qi in zip(a.tolist(), b.tolist()):
        s += min(pi, qi)
    return s


def residual_distribution(p: ArrayLike, q: ArrayLike) -> NDArray:
    a, b = _pair(p, q)
    r = np.maximum(a - b, 0.0)
    z = 0.0
    for v in r.tolist():
        z += v
    if z <= 0.0:
        # p == q: nothing is ever rejected, but the result must stay a distribution.
        return a.copy()
    return r / z


def speculative_step(
    p: ArrayLike, q: ArrayLike, x: int, u_accept: float, u_resample: float
) -> tuple[int, bool]:
    a, b = _pair(p, q)
    if not 0 <= x < a.size:
        raise ValueError(f"draft token {x} outside [0, {a.size})")
    _check_u(u_resample, "u_resample")
    if rejection_accept(float(a[x]), float(b[x]), u_accept):
        return int(x), True
    return sample_categorical(residual_distribution(a, b), u_resample), False


def rejection_sample(
    p: ArrayLike, q: ArrayLike, m: float, rng: UniformSource, max_tries: int = 10000
) -> tuple[int, int]:
    a, b = _pair(p, q)
    if not m >= 1.0:
        raise ValueError(f"the envelope m must be >= 1, got {m!r}")
    if (a > m * b).any():
        i = int(np.flatnonzero(a > m * b)[0])
        raise ValueError(f"m = {m} is not an envelope: p[{i}] = {a[i]} > m q[{i}] = {m * b[i]}")
    for tries in range(1, max_tries + 1):
        x = sample_categorical(b, rng.uniform())
        if rng.uniform() < a[x] / (m * b[x]):
            return x, tries
    raise RuntimeError(f"no acceptance in {max_tries} tries")
