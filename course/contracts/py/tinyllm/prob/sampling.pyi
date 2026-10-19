# contracts/py/tinyllm/prob/sampling.pyi (M07.1)
# chapter: content/courses/mathematics-for-computing/08-probability-statistics/02-categorical-sampling.md
#
# Three exact ways to draw from a categorical distribution p over the ids
# 0..n-1, plus the inverse CDF of the exponential distribution for Poisson
# arrivals. Every function is deterministic given its uniforms: randomness
# comes in as u in [0, 1) or as an `rng` with a `uniform() -> float` method
# (the PCG32 of spec/pcg32.md: M06.3 in your repo, the frozen copy in course
# tests). Arithmetic is IEEE float64.
from typing import Protocol

from numpy.typing import ArrayLike, NDArray

class UniformSource(Protocol):
    def uniform(self) -> float:
        """One float64 uniform in [0, 1) (spec/pcg32.md uniform_f64)."""

def sample_categorical(probs: ArrayLike, u: float) -> int:
    """Inverse CDF, spec/sampling.md step 11: walk the ids in ascending order
    adding probs[i] to a float64 running sum c and return the first id with
    u < c. If rounding leaves c <= u after the last id, return the largest id
    with probs[i] > 0. An id with probs[i] == 0 is never returned.
    ValueError unless probs is 1-D, non-empty, finite, non-negative, and sums
    to 1 within 1e-9 * len(probs) + 1e-12, and unless 0 <= u < 1."""

def gumbel_noise(u: ArrayLike) -> NDArray:
    """float64 -log(-log(u)) elementwise: standard Gumbel noise from uniforms.
    ValueError unless every u lies in the open interval (0, 1)."""

def gumbel_max(logits: ArrayLike, gumbels: ArrayLike) -> int:
    """argmax_i (logits[i] + gumbels[i]) in float64, ties to the lowest id.
    With gumbels = gumbel_noise(u) for n independent uniforms, the result is
    distributed as softmax(logits). A -inf logit is never chosen.
    ValueError for mismatched 1-D shapes, an empty input, a NaN or +inf
    logit, or every logit -inf."""

class AliasTable:
    """Walker's alias method, built by Vose's algorithm in O(n): column i is
    kept with probability prob[i], else it hands over to alias[i]. Then
    P(i) = (prob[i] + sum over j with alias[j] == i of (1 - prob[j])) / n."""

    prob: NDArray  # float64 [n], each in [0, 1]
    alias: NDArray  # int64 [n], each in [0, n)

    def __init__(self, probs: ArrayLike) -> None:
        """Build the table for probs (the same checks as sample_categorical,
        which raise ValueError). An id with probs[i] == 0 has prob[i] == 0 and
        is the alias of no column with prob[j] < 1, so it is never drawn."""

    def __len__(self) -> int:
        """n, the number of ids."""

    def sample(self, rng: UniformSource, n: int) -> NDArray:
        """int64 [n] draws. Exactly one rng.uniform() per draw, in order:
        x = u * len(self), i = min(floor(x), len(self) - 1), f = x - i;
        the draw is i when f < prob[i], else alias[i]. ValueError for n < 0."""

def exponential_icdf(u: float, rate: float) -> float:
    """The inverse CDF of Exp(rate): -log1p(-u) / rate, so 0 maps to 0.
    ValueError unless 0 <= u < 1 and rate > 0."""

def poisson_arrivals(rate: float, horizon: float, rng: UniformSource) -> NDArray:
    """float64 arrival times of a Poisson process with this rate on
    [0, horizon): t = 0, then repeat t += exponential_icdf(rng.uniform(),
    rate) and append t while t < horizon. Draws one uniform per arrival plus
    the one that crosses the horizon. load.01 re-implements it in Go.
    ValueError unless rate > 0 and horizon >= 0."""
