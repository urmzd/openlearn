# contracts/py/tinyllm/prob/rv.pyi (M07.0)
# chapter: content/courses/mathematics-for-computing/07-probability-statistics/00-random-variables-and-the-normal.md
#
# Discrete random variables as (values, probs) tables, and standard normal
# draws by Box-Muller over the PCG32 uniforms of spec/pcg32.md.
from typing import Any

from numpy.typing import ArrayLike, NDArray

def expectation(values: ArrayLike, probs: ArrayLike) -> float:
    """E[X] = sum_i values[i] * probs[i] for a discrete X. ValueError unless
    values and probs are 1-D of one non-zero length, probs >= 0, and
    |sum(probs) - 1| <= 1e-9."""

def variance(values: ArrayLike, probs: ArrayLike) -> float:
    """Var[X] = sum_i probs[i] * (values[i] - E[X])^2, computed in two passes
    (mean first), never as E[X^2] - E[X]^2, which cancels catastrophically
    when |E[X]| is large. Same ValueErrors as expectation."""

def box_muller(u1: float, u2: float) -> tuple[float, float]:
    """Two independent standard normals from two uniforms in [0, 1):
    r = sqrt(-2 ln(1 - u1)), theta = 2 pi u2, returns (r cos theta,
    r sin theta). 1 - u1 is never 0, so the log is finite."""

def normal(rng: Any, n: int) -> NDArray:
    """float64 [n] standard normals in spec/pcg32.md order: for each pair,
    u1 = rng.uniform(), u2 = rng.uniform(), then box_muller(u1, u2) gives the
    next two values, cosine first. For an odd n the last sine is dropped
    (no spare is kept between calls). normal(PCG32(s), 8) equals the
    `normal` vectors of spec/pcg32.vectors.json. ValueError for n < 0."""
