# contracts/py/tinyllm/prob/rejection.pyi (M07.6)
# chapter: content/courses/mathematics/13-probability-statistics/07-rejection-sampling-and-residual-distributions.md
#
# Rejection sampling over a finite set of ids 0..n-1, and the one-step
# correction that makes speculative decoding (L8.6, L10.8) exact: a draft
# token x drawn from a proposal q is kept with probability min(1, p_x / q_x),
# and on rejection a replacement is drawn from the residual distribution
# normalize(max(0, p - q)). The token that comes out is distributed exactly
# as p.
#
# Randomness arrives from outside, as uniforms u in [0, 1) or an `rng` with a
# `uniform() -> float` method (PCG32, spec/pcg32.md), so every function is a
# deterministic map from uniforms to outcomes. All arithmetic is IEEE
# float64, and every sum runs over ids in ascending order with a plain
# left-to-right loop, so the Rust port (L10.8) reproduces it bit for bit.
# Distributions are checked like M07.1's sample_categorical: 1-D, non-empty,
# finite, non-negative, summing to 1 within 1e-9 * n + 1e-12 (ValueError
# otherwise).
from numpy.typing import ArrayLike, NDArray

from tinyllm.prob.sampling import UniformSource

def rejection_accept(p_x: float, q_x: float, u: float) -> bool:
    """Accept a draft x drawn from q with probability min(1, p_x / q_x):
    True iff u < p_x / q_x (one float64 division, then one comparison).
    p_x >= q_x always accepts, p_x == 0 never does, and u == p_x / q_x
    rejects, so P(accept) is exactly min(1, p_x / q_x) for u uniform on
    [0, 1). ValueError unless 0 <= u < 1, p_x >= 0, and q_x > 0 (a token q
    proposed has q_x > 0)."""

def acceptance_probability(p: ArrayLike, q: ArrayLike) -> float:
    """sum_i min(p_i, q_i) = 1 - TV(p, q): the probability that a draft from
    q survives rejection_accept. 1 when p == q, 0 for disjoint supports.
    ValueError for mismatched shapes or a bad distribution."""

def residual_distribution(p: ArrayLike, q: ArrayLike) -> NDArray:
    """float64 [n]: r_i = max(0, p_i - q_i) / Z with Z = sum_i max(0, p_i - q_i)
    (ascending ids). Z = 1 - acceptance_probability(p, q) is the rejection
    probability. When Z == 0 (p == q: a draft is never rejected) the result
    is a copy of p, so it is always a distribution. ValueError for
    mismatched shapes or a bad distribution."""

def speculative_step(
    p: ArrayLike, q: ArrayLike, x: int, u_accept: float, u_resample: float
) -> tuple[int, bool]:
    """Verify one draft token x ~ q against the target p: (x, True) when
    rejection_accept(p[x], q[x], u_accept); otherwise
    (sample_categorical(residual_distribution(p, q), u_resample), False)
    (M07.1's inverse CDF). u_resample is not looked at when x is accepted.
    For x ~ q and independent uniforms the returned token is distributed
    exactly as p. ValueError when x is outside [0, n) or q[x] == 0, or for a
    bad uniform or distribution."""

def rejection_sample(
    p: ArrayLike, q: ArrayLike, m: float, rng: UniformSource, max_tries: int = 10000
) -> tuple[int, int]:
    """Von Neumann's rejection sampler for target p with proposal q and an
    envelope m >= max_i p_i / q_i: repeat { x = sample_categorical(q,
    rng.uniform()); accept when rng.uniform() < p_x / (m q_x) } and return
    (x, tries), tries >= 1 counting proposals. Exactly two uniforms per try,
    in that order. Each try accepts with probability 1 / m, so tries is
    geometric with mean m, and x is distributed exactly as p.
    ValueError when m < 1, when p_i > m q_i for some i (m is not an
    envelope: then x would not follow p), or for a bad distribution;
    RuntimeError after max_tries rejections."""
