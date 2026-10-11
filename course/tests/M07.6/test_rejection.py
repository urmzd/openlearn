"""Course tests for M07.6: rejection sampling and residual distributions
(tinyllm/prob/rejection.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/M07.6), and the chapter section it comes from.

The chapter's worked example (section 3) is p = [1/2, 1/4, 1/4, 0] and
q = [1/4, 1/2, 0, 1/4]. The statistical tests either enumerate grids of
uniforms on distributions whose probabilities are multiples of 1/8, where the
grid makes every acceptance count an exact integer (so the output
distribution is compared with p as exact fractions), or draw from the frozen
PCG32 at OL_SEED and apply a chi-square test at p > 1e-3 (DESIGN 4.0).
"""

from __future__ import annotations

import math
import os
from fractions import Fraction

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32
from tinyllm.prob.rejection import (
    acceptance_probability,
    rejection_accept,
    rejection_sample,
    residual_distribution,
    speculative_step,
)
from tinyllm.prob.sampling import sample_categorical

HAND_P = [0.5, 0.25, 0.25, 0.0]
HAND_Q = [0.25, 0.5, 0.0, 0.25]
# Chi-square critical values at p = 1e-3 by degrees of freedom.
CHI2_P001 = {2: 13.816, 3: 16.266, 4: 18.467, 5: 20.515, 6: 22.458, 7: 24.322}
# 840 = lcm(1..8): (a / b) * 840 is an integer for a, b in 1..8, so the
# midpoint grid accepts exactly the right fraction of uniforms.
GRID = 840


def seed() -> int:
    return int(os.environ.get("OL_SEED", "0"))


def grid(m: int) -> list[float]:
    """m uniforms at the midpoints (k + 1/2) / m: an exact enumeration of [0, 1)."""
    return [(k + 0.5) / m for k in range(m)]


class Exhausted(Exception):
    pass


class Script:
    """A uniform source that returns fixed values, counts the calls, and
    raises Exhausted when asked for more."""

    def __init__(self, values):
        self.values = list(values)
        self.calls = 0

    def uniform(self) -> float:
        if self.calls >= len(self.values):
            raise Exhausted
        v = self.values[self.calls]
        self.calls += 1
        return v


def eighths(rng: PCG32, n: int, zeros: int = 0) -> list[float]:
    """A random distribution over n ids whose probabilities are multiples of
    1/8, with `zeros` ids forced to 0 (drawn from the frozen PCG32)."""
    counts = [0] * n
    alive = list(range(n))
    for _ in range(zeros):
        alive.pop(rng.below(len(alive)))
    for _ in range(8):
        counts[alive[rng.below(len(alive))]] += 1
    return [c / 8 for c in counts]


def exact_output(p, q) -> list[Fraction]:
    """The exact distribution of speculative_step's output when x ~ q,
    enumerated: every draft x with its probability q_x, every acceptance
    uniform on the grid, and every resampling uniform on the grid."""
    n = len(p)
    out = [Fraction(0)] * n
    us = grid(GRID)
    for x in range(n):
        if q[x] == 0:
            continue
        acc, rejected_u, rej = 0, None, [0] * n
        for ua in us:
            t, accepted = speculative_step(p, q, x, ua, 0.5)
            if accepted:
                assert t == x
                acc += 1
            elif rejected_u is None:
                rejected_u = ua
        if rejected_u is not None:
            # A rejected draft resamples the same way whatever ua was: one row.
            for ur in us:
                t, accepted = speculative_step(p, q, x, rejected_u, ur)
                assert not accepted
                rej[t] += 1
        n_rej = GRID - acc
        out[x] += Fraction(q[x]) * Fraction(acc, GRID)
        for t in range(n):
            out[t] += Fraction(q[x]) * Fraction(n_rej, GRID) * Fraction(rej[t], GRID)
    return out


def chi_square(counts: np.ndarray, probs: np.ndarray) -> tuple[float, int]:
    keep = probs > 0
    n = counts.sum()
    expected = n * probs[keep]
    return float(((counts[keep] - expected) ** 2 / expected).sum()), int(keep.sum()) - 1


# --- the worked example -------------------------------------------------------


def test_hand_example():
    # WHY: section 3 by hand: the ratios p/q are [2, 1/2, -, 0], so draft 0
    #      always survives, draft 1 survives when u < 1/2, draft 3 never; the
    #      overlap sum min(p, q) is 1/2 and the residual is [1/2, 0, 1/2, 0].
    # KIND: unit, smoke
    # CATCHES: s01, s02, s03, s04, s05, s12
    # CHAPTER: M07.6 section 3
    assert rejection_accept(0.5, 0.25, 0.999) is True
    assert rejection_accept(0.25, 0.5, 0.49) is True
    assert rejection_accept(0.25, 0.5, 0.51) is False
    assert rejection_accept(0.0, 0.25, 0.0) is False
    assert acceptance_probability(HAND_P, HAND_Q) == 0.5
    r = residual_distribution(HAND_P, HAND_Q)
    assert r.dtype == np.float64
    assert r.tolist() == [0.5, 0.0, 0.5, 0.0]


def test_hand_example_output_is_p():
    # WHY: the section 3 table: accepted mass [1/4, 1/4, 0, 0] plus the
    #      rejected mass 1/2 times the residual [1/2, 0, 1/2, 0] is exactly
    #      p = [1/2, 1/4, 1/4, 0]. Enumerated exactly over grids of uniforms.
    # KIND: statistical, smoke
    # CATCHES: s02, s03, s04, s05, s06, s07, s12, s13
    # CHAPTER: M07.6 section 3
    assert exact_output(HAND_P, HAND_Q) == [
        Fraction(1, 2),
        Fraction(1, 4),
        Fraction(1, 4),
        0,
    ]


# --- the acceptance test ---------------------------------------------------------


def test_accept_is_strict():
    # WHY: P(u < r) = r for u uniform on [0, 1), but P(u <= r) counts the
    #      point u = r too: with u = 0 and p_x = 0 a "<=" accepts a token the
    #      target never produces, and a tie u = p_x / q_x must reject.
    # KIND: boundary
    # CATCHES: s01, s02
    # CHAPTER: M07.6 section 2.2
    assert rejection_accept(0.0, 0.5, 0.0) is False
    assert rejection_accept(0.25, 0.5, 0.5) is False
    assert rejection_accept(0.125, 0.5, 0.25) is False
    assert rejection_accept(0.125, 0.5, 0.2499999) is True


def test_accept_always_when_target_dominates():
    # WHY: when p_x >= q_x the draft is under-proposed, so it must always be
    #      kept: the ratio is at least 1 and every u in [0, 1) is below it,
    #      including the largest double below 1.
    # KIND: boundary
    # CATCHES: s02, s12
    # CHAPTER: M07.6 section 2.2
    u_max = 1.0 - 2.0**-53
    for p_x, q_x in [(0.3, 0.3), (0.9, 0.1), (1.0, 1e-300), (0.5, 0.25)]:
        assert rejection_accept(p_x, q_x, u_max) is True
        assert rejection_accept(p_x, q_x, 0.0) is True


def test_accept_rejects_bad_arguments():
    # WHY: q_x = 0 means q never proposes x, so asking is a caller bug; u
    #      outside [0, 1) and a negative probability are bugs too, and a
    #      silent answer would bias the output distribution.
    # KIND: boundary
    # CATCHES: s11, m01
    # CHAPTER: M07.6 section 4
    for args in [
        (0.5, 0.0, 0.1),
        (0.5, -0.1, 0.1),
        (0.5, 0.5, 1.0),
        (0.5, 0.5, -0.1),
        (-0.1, 0.5, 0.1),
    ]:
        with pytest.raises(ValueError):
            rejection_accept(*args)


def test_acceptance_rate_matches_acceptance_probability():
    # WHY: averaged over x ~ q and u, the acceptance probability is
    #      sum_x q_x min(1, p_x / q_x) = sum_x min(p_x, q_x) = 1 - TV(p, q):
    #      L8.6 reports this number as its acceptance rate.
    # KIND: statistical
    # CATCHES: s02, s10, s12
    # CHAPTER: M07.6 section 2.3
    rng = PCG32(seed(), 61)
    us = grid(GRID)
    for _ in range(6):
        p, q = eighths(rng, 5, zeros=1), eighths(rng, 5, zeros=1)
        acc = Fraction(0)
        for x in range(5):
            if q[x] > 0:
                n = sum(rejection_accept(p[x], q[x], u) for u in us)
                acc += Fraction(q[x]) * Fraction(n, GRID)
        assert acc == Fraction(acceptance_probability(p, q))
        assert acceptance_probability(p, q) == sum(min(a, b) for a, b in zip(p, q))


# --- the residual ------------------------------------------------------------------


def test_residual_is_a_distribution():
    # WHY: the residual is sampled with M07.1's inverse CDF, which demands a
    #      distribution: non-negative, summing to 1. Its normalizer is the
    #      rejection probability, 1 - acceptance_probability(p, q).
    # KIND: property
    # CATCHES: s03, s04, s05, s10
    # CHAPTER: M07.6 section 2.3
    rng = PCG32(seed(), 62)
    for _ in range(200):
        n = 2 + rng.below(9)
        p = rng.uniform_array((n,)) ** 3
        q = rng.uniform_array((n,)) ** 3
        p, q = p / p.sum(), q / q.sum()
        r = residual_distribution(p, q)
        assert (r >= 0).all()
        assert_close(r.sum(), 1.0)
        z = np.maximum(p - q, 0).sum()
        assert_close(z, 1.0 - acceptance_probability(p, q), atol=1e-12)
        # mass only where the target exceeds the proposal
        assert (r[p <= q] == 0).all()


def test_residual_when_p_equals_q():
    # WHY: p == q leaves no residual mass (0 / 0); a draft is then never
    #      rejected, but the function must still return a distribution (p),
    #      not NaN, and not the caller's own array.
    # KIND: boundary
    # CATCHES: s08, s14
    # CHAPTER: M07.6 section 2.3
    p = np.array([0.2, 0.3, 0.5])
    r = residual_distribution(p, p.copy())
    assert r.tolist() == [0.2, 0.3, 0.5]
    assert r is not p
    r[0] = 7.0
    assert p[0] == 0.2


def test_residual_disjoint_supports_is_p():
    # WHY: when q puts no mass where p does, every draft is rejected
    #      (acceptance 0) and the residual is p itself.
    # KIND: boundary
    # CATCHES: s03, s05, s10
    # CHAPTER: M07.6 section 2.3
    p, q = [0.0, 0.75, 0.25, 0.0], [0.5, 0.0, 0.0, 0.5]
    assert acceptance_probability(p, q) == 0.0
    assert residual_distribution(p, q).tolist() == [0.0, 0.75, 0.25, 0.0]


def test_residual_sum_is_ascending():
    # WHY: the normalizer is a left-to-right float64 sum in ascending ids,
    #      the order the Rust port (L10.8) uses; any other order (numpy's
    #      pairwise np.sum) can differ in the last bit of every probability.
    # KIND: unit
    # CATCHES: s03, s04, s05, s09
    # CHAPTER: M07.6 section 4
    rng = PCG32(seed(), 63)
    for _ in range(200):
        n = 9 + rng.below(40)
        p = rng.uniform_array((n,)) ** 4
        q = rng.uniform_array((n,)) ** 4
        p, q = p / p.sum(), q / q.sum()
        r = np.maximum(p - q, 0.0)
        z = 0.0
        for v in r.tolist():
            z += v
        assert residual_distribution(p, q).tolist() == (r / z).tolist()


def test_shapes_and_distributions_are_checked():
    # WHY: a residual of mismatched vocabularies, or of a "distribution" that
    #      does not sum to 1, is silently wrong; the contract raises.
    # KIND: boundary
    # CATCHES: m02
    # CHAPTER: M07.6 section 4
    with pytest.raises(ValueError):
        residual_distribution([0.5, 0.5], [1.0])
    with pytest.raises(ValueError):
        acceptance_probability([0.5, 0.5], [0.2, 0.3, 0.5])
    with pytest.raises(ValueError):
        residual_distribution([0.5, 0.6], [0.5, 0.5])
    with pytest.raises(ValueError):
        acceptance_probability([[0.5, 0.5]], [[0.5, 0.5]])
    with pytest.raises(ValueError):
        acceptance_probability([1.5, -0.5], [0.5, 0.5])


# --- speculative_step ------------------------------------------------------------------


def test_speculative_output_is_exactly_p():
    # WHY: the theorem behind speculative decoding: accept x ~ q with
    #      probability min(1, p_x/q_x), else draw from the residual, and the
    #      output is distributed as p exactly, for any q whose support covers
    #      the drafts. Enumerated exactly on random distributions in eighths.
    # KIND: statistical
    # CATCHES: s02, s03, s04, s05, s06, s07, s12, s13
    # CHAPTER: M07.6 section 2.4
    rng = PCG32(seed(), 64)
    for _ in range(4):
        p, q = eighths(rng, 5, zeros=1), eighths(rng, 5, zeros=1)
        assert exact_output(p, q) == [Fraction(v) for v in p]


def test_speculative_chi_square():
    # WHY: the same law on non-dyadic distributions with sampled uniforms:
    #      x ~ q by inverse CDF, then speculative_step; the counts must match
    #      p (chi-square, p-value above 1e-3 at a fixed seed).
    # KIND: statistical
    # CATCHES: s02, s03, s04, s05, s06, s07, s12, s13
    # CHAPTER: M07.6 section 2.4
    rng = PCG32(seed(), 65)
    p = np.array([0.05, 0.4, 0.15, 0.3, 0.1])
    q = np.array([0.3, 0.1, 0.3, 0.05, 0.25])
    counts = np.zeros(5)
    for _ in range(20000):
        x = sample_categorical(q, rng.uniform())
        t, _ = speculative_step(p, q, x, rng.uniform(), rng.uniform())
        counts[t] += 1
    stat, df = chi_square(counts, p)
    assert stat < CHI2_P001[df], f"chi-square {stat:.1f} over {df} df: counts {counts}"


def test_speculative_accepts_without_resampling():
    # WHY: an accepted draft is returned as is, with accepted = True, and the
    #      resampling uniform plays no part: L8.6 and L10.8 draw it from the
    #      same generator in a fixed order, so it must not change the token.
    # KIND: unit
    # CATCHES: s02, s03, s04, s05, s07, s12, s13
    # CHAPTER: M07.6 section 2.4
    assert speculative_step(HAND_P, HAND_Q, 0, 0.9, 0.0) == (0, True)
    assert speculative_step(HAND_P, HAND_Q, 0, 0.9, 0.999) == (0, True)
    assert speculative_step(HAND_P, HAND_Q, 1, 0.25, 0.75) == (1, True)
    t, ok = speculative_step(HAND_P, HAND_Q, 1, 0.75, 0.75)
    assert (t, ok) == (2, False)
    assert type(t) is int and type(ok) is bool


def test_speculative_rejects_bad_drafts():
    # WHY: a draft outside the vocabulary, or one q gave probability 0, means
    #      the draft and the proposal disagree: a caller bug.
    # KIND: boundary
    # CATCHES: s11
    # CHAPTER: M07.6 section 4
    with pytest.raises(ValueError):
        speculative_step(HAND_P, HAND_Q, 4, 0.5, 0.5)
    with pytest.raises(ValueError):
        speculative_step(HAND_P, HAND_Q, -1, 0.5, 0.5)
    with pytest.raises(ValueError):
        speculative_step(HAND_P, HAND_Q, 2, 0.5, 0.5)


# --- von Neumann's sampler -----------------------------------------------------------------


def test_rejection_sample_first_try_enumerated():
    # WHY: on one try, P(propose x and accept) = q_x * p_x / (m q_x) = p_x / m:
    #      the accepted token follows p and a try succeeds with probability
    #      1/m. Enumerated exactly over 8 proposal and 1680 acceptance
    #      uniforms (p and q in eighths, m = 2).
    # KIND: statistical
    # CATCHES: s15, s16, s17
    # CHAPTER: M07.6 section 2.1
    p = [0.25, 0.375, 0.375, 0.0]
    q = [0.25, 0.25, 0.25, 0.25]
    m = 2.0
    counts = [0] * 4
    for u1 in grid(8):
        for u2 in grid(1680):
            try:
                x, tries = rejection_sample(p, q, m, Script([u1, u2]))
            except Exhausted:
                continue  # rejected on the first try
            assert tries == 1
            counts[x] += 1
    total = 8 * 1680
    assert [Fraction(c, total) for c in counts] == [Fraction(v) / 2 for v in p]


def test_rejection_sample_tries_are_geometric():
    # WHY: each try accepts with probability 1/m, so the number of tries is
    #      geometric with mean m and variance m (m - 1): the cost of a loose
    #      envelope. 4000 samples, mean within 4 standard errors.
    # KIND: statistical
    # CATCHES: s15, s16, s17
    # CHAPTER: M07.6 section 2.1
    rng = PCG32(seed(), 66)
    p = np.array([0.1, 0.2, 0.3, 0.4])
    q = np.array([0.25, 0.25, 0.25, 0.25])
    m = 1.6
    tries = np.array([rejection_sample(p, q, m, rng)[1] for _ in range(4000)])
    se = math.sqrt(m * (m - 1) / len(tries))
    assert abs(tries.mean() - m) < 4 * se, (
        f"mean tries {tries.mean():.3f}, expected {m}"
    )
    assert tries.min() == 1


def test_rejection_sample_draw_order():
    # WHY: two uniforms per try, proposal first, then acceptance: the order is
    #      part of the parity contract with the Rust port (spec/sampling.md,
    #      draw accounting). Proposal 0.6 picks id 2 of uniform q and 0.3
    #      accepts it (ratio 0.5); with 0.9 instead it is rejected and the
    #      second try proposes id 0 (u = 0.1) and accepts.
    # KIND: unit
    # CATCHES: s15, s17, s18
    # CHAPTER: M07.6 section 4
    p, q = [0.5, 0.0, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]
    src = Script([0.6, 0.3, 0.1, 0.0])
    assert rejection_sample(p, q, 2.0, src) == (2, 1)
    assert src.calls == 2
    src = Script([0.6, 0.9, 0.1, 0.0])
    assert rejection_sample(p, q, 2.0, src) == (0, 2)
    assert src.calls == 4


def test_rejection_sample_checks_the_envelope():
    # WHY: with m below max p_x / q_x the accepted tokens no longer follow p
    #      (the over-represented ones are capped), and m < 1 is impossible;
    #      both are errors, and so is a loop that never accepts.
    # KIND: boundary
    # CATCHES: s15, s17, s19, m03
    # CHAPTER: M07.6 section 2.1
    p, q = [0.5, 0.0, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]
    with pytest.raises(ValueError):
        rejection_sample(p, q, 1.5, PCG32(0))
    with pytest.raises(ValueError):
        rejection_sample(q, q, 0.5, PCG32(0))
    assert rejection_sample(q, q, 1.0, PCG32(0))[1] == 1
    with pytest.raises(RuntimeError):
        rejection_sample(p, q, 2.0, Script([0.6, 0.9] * 3), max_tries=3)
    # max_tries counts every try: the third may still succeed.
    assert rejection_sample(
        p, q, 2.0, Script([0.6, 0.9] * 2 + [0.1, 0.0]), max_tries=3
    ) == (0, 3)
