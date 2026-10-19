# contracts/py/tinyllm/prob/mle.pyi (M07.2)
# chapter: content/courses/mathematics-for-computing/08-probability-statistics/03-mle-laplace-absolute-discounting.md
#
# Estimating a categorical distribution from counts. counts maps an outcome
# (a word, an n-gram, a subword) to how often it was seen. A count is a
# finite non-negative real number: an int from counting, or a fraction from
# the expected counts of EM (L1.4); bool is not a count. A key with count 0
# is an outcome known to be in the vocabulary but not seen. N = sum of the
# counts. Probabilities are float64; every function returns a new dict.
from collections.abc import Hashable, Mapping
from typing import TypeVar

K = TypeVar("K", bound=Hashable)
Count = int | float

def mle(counts: Mapping[K, Count]) -> dict[K, float]:
    """Maximum-likelihood estimate p(k) = c(k) / N for every key of counts
    (a key with count 0 gets 0.0). ValueError when N == 0 or a count is not
    a finite non-negative real number."""

def log_likelihood(counts: Mapping[K, Count], probs: Mapping[K, float]) -> float:
    """sum over k with c(k) > 0 of c(k) * ln probs[k], in nats: the log
    probability of the observed data under probs. -inf when an observed key
    has probability 0 or is missing from probs. ValueError for a count that
    is not a finite non-negative real number."""

def laplace(counts: Mapping[K, Count], vocab_size: int, alpha: float = 1.0) -> dict[K, float]:
    """Add-alpha smoothing over a vocabulary of vocab_size outcomes:
    p(k) = (c(k) + alpha) / (N + alpha * vocab_size) for every key of counts.
    Each of the vocab_size - len(counts) outcomes that are not keys has the
    same probability alpha / (N + alpha * vocab_size), so the full
    distribution sums to 1. N == 0 gives the uniform 1 / vocab_size.
    ValueError when alpha <= 0, len(counts) > vocab_size, or a count is not
    a finite non-negative real number."""

def absolute_discount(
    counts: Mapping[K, Count], d: float, backoff: Mapping[K, float]
) -> dict[K, float]:
    """Absolute discounting interpolated with a backoff distribution (the
    lower-order model of Kneser-Ney, L2.1). With N the total count:
        p(k) = max(c(k) - d, 0) / N + lam * backoff[k],
        lam  = (sum over keys of min(c(k), d)) / N
    for every key of backoff (the vocabulary). lam is exactly the mass the
    discount removed; for integer counts it is d * T / N, T = the number of
    keys with c(k) > 0. N == 0 returns backoff unchanged (as a new dict).
    ValueError unless 0 < d <= 1, backoff sums to 1 within 1e-9, every key
    of counts is a key of backoff, and every count is a finite non-negative
    real number."""

def ney_discount(counts: Mapping[K, Count]) -> float:
    """Ney's estimate of the discount, D = n1 / (n1 + 2 * n2), where n1 and n2
    count the keys seen exactly once and exactly twice; 0 < D <= 1. ValueError
    when n1 == 0 (no singletons: the estimate would be 0, and absolute
    discounting needs d > 0), or for a count that is not a finite
    non-negative real number."""
