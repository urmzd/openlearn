"""Estimating a categorical distribution from counts (M07.2).

mle is the maximum-likelihood estimate, which gives unseen outcomes
probability 0. laplace adds a pseudo-count to every outcome of a known
vocabulary. absolute_discount subtracts a fixed d from every seen count and
hands the freed mass to a backoff distribution, the core of Kneser-Ney
smoothing (L2.1); ney_discount estimates d from the count-of-counts.

Contract: contracts/py/tinyllm/prob/mle.pyi.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Hashable, Mapping


def _check_counts(counts: Mapping[Hashable, float]) -> float:
    """N, the total count, after checking every count is a finite
    non-negative real number (ints from counting, floats from EM)."""
    for k, c in counts.items():
        if isinstance(c, bool) or not isinstance(c, numbers.Real):
            raise ValueError(f"count of {k!r} must be a number, got {c!r}")
        if not math.isfinite(c) or c < 0:
            raise ValueError(f"count of {k!r} must be finite and >= 0, got {c}")
    return math.fsum(float(c) for c in counts.values())


def mle(counts: Mapping[Hashable, float]) -> dict:
    n = _check_counts(counts)
    if n == 0:
        raise ValueError("mle needs at least one observation (N = 0 gives 0 / 0)")
    return {k: float(c) / n for k, c in counts.items()}


def log_likelihood(
    counts: Mapping[Hashable, float], probs: Mapping[Hashable, float]
) -> float:
    _check_counts(counts)
    terms = []
    for k, c in counts.items():
        if c == 0:
            continue  # 0 * log 0 = 0: an unseen outcome costs nothing
        q = float(probs.get(k, 0.0))
        if q <= 0.0:
            return -math.inf
        terms.append(float(c) * math.log(q))
    return math.fsum(terms)


def laplace(
    counts: Mapping[Hashable, float], vocab_size: int, alpha: float = 1.0
) -> dict:
    if not alpha > 0:
        raise ValueError(f"alpha must be > 0, got {alpha!r}")
    if len(counts) > vocab_size:
        raise ValueError(f"{len(counts)} keys do not fit a vocabulary of {vocab_size}")
    n = _check_counts(counts)
    # The denominator adds alpha once per outcome of the vocabulary, seen or not.
    denom = n + alpha * vocab_size
    return {k: (float(c) + alpha) / denom for k, c in counts.items()}


def absolute_discount(
    counts: Mapping[Hashable, float], d: float, backoff: Mapping[Hashable, float]
) -> dict:
    if not 0.0 < d <= 1.0:
        raise ValueError(f"d must lie in (0, 1], got {d!r}")
    total_b = math.fsum(float(v) for v in backoff.values())
    if abs(total_b - 1.0) > 1e-9:
        raise ValueError(f"backoff must sum to 1, got {total_b!r}")
    missing = [k for k in counts if k not in backoff]
    if missing:
        raise ValueError(f"{missing[0]!r} is counted but not in the backoff vocabulary")
    n = _check_counts(counts)
    if n == 0:
        return {k: float(v) for k, v in backoff.items()}
    # Exactly the mass the discount removed: d from each seen count, or the
    # whole count when it is smaller than d (fractional EM counts). For
    # integer counts this is d * T / N with T the number of seen keys.
    lam = math.fsum(min(float(c), d) for c in counts.values()) / n
    return {
        k: max(float(counts.get(k, 0)) - d, 0.0) / n + lam * float(b)
        for k, b in backoff.items()
    }


def ney_discount(counts: Mapping[Hashable, float]) -> float:
    _check_counts(counts)
    n1 = sum(1 for c in counts.values() if c == 1)
    n2 = sum(1 for c in counts.values() if c == 2)
    if n1 == 0:
        raise ValueError("no outcome was seen exactly once: Ney's discount would be 0")
    return n1 / (n1 + 2 * n2)
