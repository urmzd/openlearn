"""Course tests for M07.0: expectation, variance, box_muller, and normal
(tinyllm/prob/rv.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/M07.0), and the chapter section it comes from.

The chapter's worked examples (section 3): a fair die, E = 7/2 and
Var = 35/12; and Box-Muller on u1 = 1 - e^-2, u2 = 1/8, which gives
r = 2, theta = pi/4, and the pair (sqrt 2, sqrt 2).
Draws come from the frozen PCG32 (course/tests/_lib), whose uniform() is
the spec/pcg32.md uniform_f64 bit for bit.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32
from tinyllm.prob.rv import box_muller, expectation, normal, variance

VECTORS = json.loads(
    (
        Path(os.environ.get("TINYLLM_FIXTURES", "")) / "M06.3" / "pcg32.vectors.json"
    ).read_text()
)
DIE = np.arange(1.0, 7.0)
FAIR = np.full(6, 1.0 / 6.0)


def seed() -> int:
    return int(os.environ.get("OL_SEED", "0"))


class Recorder:
    """A generator stand-in that hands out fixed uniforms and counts them."""

    def __init__(self, us: list[float]) -> None:
        self.us = list(us)
        self.taken = 0

    def uniform(self) -> float:
        self.taken += 1
        return self.us.pop(0)


def test_die_hand_example():
    # WHY: the chapter's worked example: E = (1 + ... + 6) / 6 = 7/2, and
    #      Var = sum (k - 7/2)^2 / 6 = (6.25 + 2.25 + 0.25) * 2 / 6 = 35/12.
    # KIND: unit, smoke
    # CATCHES: s02, m01
    # CHAPTER: M07.0 section 3
    assert_close(expectation(DIE, FAIR), 3.5, rtol=1e-15, atol=0.0)
    assert_close(variance(DIE, FAIR), 35.0 / 12.0, rtol=1e-15, atol=0.0)


def test_bernoulli_and_point_mass():
    # WHY: Bernoulli(p) has E = p and Var = p (1 - p); a point mass has
    #      variance exactly 0. These two shapes recur in every metric with a
    #      confidence interval (M07.4).
    # KIND: unit
    # CATCHES: s01, s02
    # CHAPTER: M07.0 section 2.2
    for p in (0.0, 0.1, 0.5, 0.9):
        assert_close(expectation([0, 1], [1 - p, p]), p, rtol=1e-15, atol=1e-16)
        assert_close(variance([0, 1], [1 - p, p]), p * (1 - p), rtol=1e-12, atol=1e-16)
    assert variance([42.0], [1.0]) == 0.0


def test_variance_has_no_catastrophic_cancellation():
    # WHY: X = 1e9 + {0, 1} with probability 1/2 each has Var = 1/4. The
    #      textbook shortcut E[X^2] - E[X]^2 subtracts two numbers near 1e18
    #      whose float64 spacing is 128: the answer comes out 0 or -64. Two
    #      passes (center, then square) keep every digit.
    # KIND: boundary
    # CATCHES: s03
    # CHAPTER: M07.0 section 5, Pitfalls
    assert variance([1e9, 1e9 + 1.0], [0.5, 0.5]) == 0.25


def test_variance_shift_and_scale_laws():
    # WHY: Var[a X + c] = a^2 Var[X] on 50 random distributions: the law
    #      M07.3 uses to pick initialization scales layer by layer.
    # KIND: property
    # CATCHES: s02
    # CHAPTER: M07.0 section 2.2
    g = PCG32(seed(), seq=101)
    for _ in range(50):
        k = 1 + g.below(8)
        x = g.uniform_array((k,), -5.0, 5.0)
        w = g.uniform_array((k,), 0.0, 1.0) + 1e-3
        p = w / w.sum()
        a, c = g.uniform() * 4 - 2, g.uniform() * 100
        v = variance(x, p)
        assert v >= 0.0
        assert_close(variance(a * x + c, p), a * a * v, rtol=1e-9, atol=1e-12)
        assert_close(
            expectation(a * x + c, p), a * expectation(x, p) + c, rtol=1e-9, atol=1e-9
        )


def test_rejects_tables_that_are_not_distributions():
    # WHY: probabilities must be non-negative and sum to 1; a table that
    #      does not is a bug upstream (unnormalized counts), not a value to
    #      average with.
    # KIND: boundary
    # CATCHES: s04
    # CHAPTER: M07.0 section 2.1
    for values, probs in [
        ([1, 2], [0.5, 0.6]),
        ([1, 2], [1.5, -0.5]),
        ([1, 2, 3], [0.5, 0.5]),
        ([], []),
        ([[1, 2]], [[0.5, 0.5]]),
    ]:
        with pytest.raises(ValueError):
            expectation(values, probs)
        with pytest.raises(ValueError):
            variance(values, probs)


def test_box_muller_hand_example():
    # WHY: the chapter's worked example: 1 - u1 = e^-2 gives
    #      r = sqrt(-2 ln e^-2) = 2, and u2 = 1/8 gives theta = pi/4, so the
    #      pair is (2 cos pi/4, 2 sin pi/4) = (sqrt 2, sqrt 2).
    # KIND: unit, smoke
    # CATCHES: s05, s06, s07
    # CHAPTER: M07.0 section 3
    z0, z1 = box_muller(1.0 - math.exp(-2.0), 0.125)
    assert_close([z0, z1], [math.sqrt(2.0), math.sqrt(2.0)], rtol=1e-15, atol=0.0)
    z0, z1 = box_muller(1.0 - math.exp(-2.0), 0.25)  # theta = pi/2
    assert_close([z0, z1], [0.0, 2.0], rtol=1e-15, atol=1e-15)


def test_box_muller_at_u1_zero_is_finite():
    # WHY: uniform() returns 0.0 with probability 2^-53, but over 10^9 draws
    #      that is not never. ln(1 - u1) is ln 1 = 0 there, a radius of 0;
    #      ln(u1) would be ln 0 = -inf and an infinite "normal".
    # KIND: boundary
    # CATCHES: s05
    # CHAPTER: M07.0 section 2.4
    z0, z1 = box_muller(0.0, 0.3)
    assert z0 == 0.0 and z1 == 0.0
    z0, z1 = box_muller(1.0 - 2.0**-53, 0.0)
    assert math.isfinite(z0) and z0 > 8.0


@pytest.mark.parametrize("s", ["0", "1", str(2**63)])
def test_normal_matches_spec_vectors(s):
    # WHY: spec/pcg32.md fixes the order: two uniforms per pair, cosine
    #      first, then sine. Python, Rust, and Go normals for the same seed
    #      must agree; compared to 4 ulp because they go through the
    #      platform's log, sqrt, sin, and cos.
    # KIND: golden
    # CATCHES: s05, s06, s07, s08
    # CHAPTER: M07.0 section 2.4
    z = normal(PCG32(int(s)), 8)
    want = np.array(VECTORS["normal"][s])
    assert z.dtype == np.float64 and z.shape == (8,)
    assert (np.abs(z - want) <= 4 * np.spacing(np.abs(want))).all(), (z, want)


def test_odd_n_draws_whole_pairs_and_drops_the_last_sine():
    # WHY: n = 3 takes two pairs (four uniforms) and keeps three values: the
    #      first three of normal(rng, 4). Drawing a third uniform pair, or
    #      reusing one uniform in two pairs, shifts every later draw.
    # KIND: unit
    # CATCHES: m02
    # CHAPTER: M07.0 section 4
    four = normal(PCG32(5), 4)
    g = PCG32(5)
    rec = Recorder([g.uniform() for _ in range(4)])
    three = normal(rec, 3)
    assert rec.taken == 4
    assert three.tolist() == four[:3].tolist()
    assert normal(PCG32(5), 0).shape == (0,)
    with pytest.raises(ValueError):
        normal(PCG32(5), -1)


def test_normal_moments_within_three_standard_errors():
    # WHY: 20000 draws: the sample mean has standard error 1/sqrt(n) and the
    #      sample variance about sqrt(2/n). A missing factor 2 in the radius
    #      gives variance 1/2; a missing 2 pi gives a lopsided angle.
    # KIND: statistical
    # CATCHES: s06, s07
    # CHAPTER: M07.0 section 2.3
    n = 20_000
    z = normal(PCG32(seed(), seq=102), n)
    assert abs(z.mean()) < 3 / math.sqrt(n)
    assert abs(z.var() - 1.0) < 3 * math.sqrt(2 / n)


def test_normal_shape_by_chi_square():
    # WHY: the right mean and variance are not the right shape. Map each
    #      draw through the normal CDF Phi(z) = (1 + erf(z / sqrt 2)) / 2 into
    #      20 equally likely bins; for a true normal the Pearson statistic is
    #      chi-square with 19 degrees of freedom, below 43.82 with
    #      probability 0.999 (p > 1e-3 at a fixed seed).
    # KIND: statistical
    # CATCHES: s06, s07
    # CHAPTER: M07.0 section 2.3
    n, bins = 20_000, 20
    z = normal(PCG32(seed(), seq=103), n)
    u = 0.5 * (1.0 + np.array([math.erf(x / math.sqrt(2.0)) for x in z]))
    counts = np.bincount(np.minimum((u * bins).astype(int), bins - 1), minlength=bins)
    expected = n / bins
    stat = float(((counts - expected) ** 2 / expected).sum())
    assert stat < 43.82, f"chi-square {stat:.1f} with 19 df"


def test_pair_halves_are_uncorrelated():
    # WHY: Box-Muller's two outputs are independent; using the same angle
    #      for both halves (or a cos/cos pair) makes them correlated.
    # KIND: statistical
    # CATCHES: s09
    # CHAPTER: M07.0 section 2.4
    n = 20_000
    z = normal(PCG32(seed(), seq=104), n)
    c = float(np.mean(z[0::2] * z[1::2]))
    assert abs(c) < 3 / math.sqrt(n / 2)
