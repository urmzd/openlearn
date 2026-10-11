"""Course tests for M07.5: hypothesis tests, paired and two-sample permutation,
McNemar, Holm (tinyllm/prob/tests.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/M07.5), and the chapter section it comes from.

The chapter's worked example (section 3): model B minus model A on three
prompts is d = (2, 1, 3). Of the 2^3 = 8 sign patterns, two reach
|sum| = 6, so the exact p-value is 2/8 = 1/4, and the add-one estimate over
the same 8 patterns is (1 + 2) / (1 + 8) = 1/3. McNemar with b01 = 1 and
b10 = 7: 2 * (C(8,0) + C(8,1)) / 2^8 = 18/256 = 9/128 = 0.0703125. Holm on
(0.01, 0.04, 0.02, 0.005) at alpha 0.05 rejects all four (Bonferroni only two),
and the adjusted p-values are (0.03, 0.04, 0.04, 0.02).

Golden values come from scipy 1.17.1 (course/fixtures/M07.5/scipy_golden.json,
written by course/oracle/M07.5/scipy_golden.py). Statistical tests draw from
the frozen PCG32 at OL_SEED.
"""

from __future__ import annotations

import itertools
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32
from tinyllm.prob.tests import (
    holm,
    holm_adjust,
    mcnemar,
    mean_difference,
    paired_permutation_test,
    permutation_test,
)

FIX = (
    Path(os.environ.get("TINYLLM_FIXTURES", "course/fixtures"))
    / "M07.5"
    / "scipy_golden.json"
)


def seed() -> int:
    return int(os.environ.get("OL_SEED", "0"))


def golden() -> dict:
    return json.loads(FIX.read_text())


class Counting:
    """The frozen PCG32 behind `uniform()`, counting the draws."""

    def __init__(self, s: int) -> None:
        self.g = PCG32(seed=s)
        self.n = 0

    def uniform(self) -> float:
        self.n += 1
        return self.g.uniform()


class Scripted:
    """Hands out a fixed list of uniforms, then fails loudly."""

    def __init__(self, us) -> None:
        self.us = list(us)
        self.n = 0

    def uniform(self) -> float:
        if self.n >= len(self.us):
            raise AssertionError(f"drew more than the {len(self.us)} scripted uniforms")
        u = self.us[self.n]
        self.n += 1
        return u


def every_sign_pattern(n: int) -> Scripted:
    """2^n permutations whose signs run through every pattern once:
    u = 0.25 flips (u < 0.5), u = 0.75 keeps."""
    us = []
    for signs in itertools.product((1, -1), repeat=n):
        us += [0.25 if s < 0 else 0.75 for s in signs]
    return Scripted(us)


# --- the worked example -----------------------------------------------------------


def test_hand_example_paired_enumeration():
    # WHY: section 3: d = (2, 1, 3); the 8 sign patterns give |sum| in
    #      {6, 0, 4, 2, 2, 4, 0, 6}; two reach 6. Fed every pattern once, the
    #      add-one estimate is (1 + 2) / (1 + 8) = 1/3 (the exact p is 2/8):
    #      the observed labelling is counted once more, so p is never 0.
    # KIND: unit, smoke
    # CATCHES: s01, s02, m04
    # CHAPTER: M07.5 section 3, Worked example by hand
    rng = every_sign_pattern(3)
    p = paired_permutation_test([5.0, 4.0, 6.0], [3.0, 3.0, 3.0], 8, rng)
    assert p == pytest.approx(1 / 3, rel=1e-15)
    assert rng.n == 24


def test_hand_example_mcnemar():
    # WHY: section 3: b01 = 1, b10 = 7, n = 8 discordant pairs; the exact
    #      two-sided p is 2 * (1 + 8) / 256 = 9/128; with Edwards' correction
    #      chi2 = (6 - 1)^2 / 8 = 3.125 and p = erfc(sqrt(3.125 / 2)) = erfc(1.25).
    # KIND: unit, smoke
    # CATCHES: s06, s12, m08
    # CHAPTER: M07.5 section 3, Worked example by hand
    assert mcnemar(1, 7) == 9 / 128
    assert mcnemar(7, 1) == 9 / 128
    assert_close(mcnemar(1, 7, exact=False), math.erfc(1.25), rtol=1e-14, atol=1e-16)


def test_hand_example_holm():
    # WHY: section 3: sorted 0.005 <= 0.05/4, 0.01 <= 0.05/3, 0.02 <= 0.05/2,
    #      0.04 <= 0.05/1, so Holm rejects all four where Bonferroni (every p
    #      against 0.05/4 = 0.0125) rejects only 0.01 and 0.005. Adjusted:
    #      (0.03, 0.04, 0.04, 0.02).
    # KIND: unit, smoke
    # CATCHES: s08
    # CHAPTER: M07.5 section 3, Worked example by hand
    p = [0.01, 0.04, 0.02, 0.005]
    assert holm(p, 0.05).tolist() == [True, True, True, True]
    assert_close(holm_adjust(p), [0.03, 0.04, 0.04, 0.02], rtol=1e-15, atol=1e-17)
    assert holm(p, 0.025).tolist() == [False, False, False, True]


# --- against scipy and the replayed draws --------------------------------------------


def test_paired_matches_exact_scipy():
    # WHY: with enough sign flips the Monte Carlo p-value converges to the
    #      exact one (scipy enumerates all 2^n patterns). 3000 permutations put
    #      the estimate within 4.5 binomial standard errors of it.
    # KIND: golden
    # CATCHES: s01, m04
    # CHAPTER: M07.5 section 2.3, The paired test
    for name, a, b, exact in golden()["exact_paired"]:
        n_perm = 3000
        p = paired_permutation_test(a, b, n_perm, PCG32(seed=seed() + 7))
        tol = 4.5 * math.sqrt(exact * (1 - exact) / n_perm) + 1.5 / (n_perm + 1)
        assert abs(p - exact) <= tol, (name, p, exact)


def test_two_sample_matches_exact_scipy():
    # WHY: the two-sample test against scipy's exact enumeration of every
    #      split of the pooled values into groups of the original sizes
    #      (6 + 6, 7 + 5, 8 + 8). A shuffle that never mixes the groups
    #      makes every permutation look like the data: p = 1.
    # KIND: golden
    # CATCHES: s11
    # CHAPTER: M07.5 section 2.4, The two-sample test
    for name, a, b, exact in golden()["exact_two"]:
        n_perm = 2000
        p = permutation_test(a, b, mean_difference, n_perm, PCG32(seed=seed() + 11))
        tol = 4.5 * math.sqrt(exact * (1 - exact) / n_perm) + 1.5 / (n_perm + 1)
        assert abs(p - exact) <= tol, (name, p, exact)


def test_replay_pins_the_draw_order():
    # WHY: the contract fixes which uniform decides what (one per pair in
    #      order; Fisher-Yates from the end, restarted from the identity each
    #      permutation), so a given seed gives one p-value in every language.
    #      load.02 ports the two-sample test to Go and must reproduce these.
    # KIND: golden
    # CATCHES: s03, s04, s11
    # CHAPTER: M07.5 section 4, The interface
    g = golden()
    pairs = {r[0]: r for r in g["exact_paired"]}
    twos = {r[0]: r for r in g["exact_two"]}
    for name, n_perm, s, want in g["replay_paired"]:
        _, a, b, _ = pairs[name]
        assert paired_permutation_test(a, b, n_perm, PCG32(seed=s)) == want, name
    for name, n_perm, s, want in g["replay_two"]:
        _, a, b, _ = twos[name]
        assert permutation_test(a, b, mean_difference, n_perm, PCG32(seed=s)) == want, (
            name
        )


def test_mcnemar_matches_scipy():
    # WHY: exact McNemar is a two-sided binomial test of b01 against n/2
    #      (scipy.stats.binomtest), the asymptotic one a chi-square with one
    #      degree of freedom; both from 0 to 210 discordant pairs.
    # KIND: golden
    # CATCHES: s06, s07, s12, m01, m08
    # CHAPTER: M07.5 section 2.5, McNemar
    for b01, b10, exact, asym in golden()["mcnemar"]:
        assert_close(
            mcnemar(b01, b10), exact, rtol=1e-12, atol=1e-15, msg=f"exact {b01},{b10}"
        )
        assert_close(
            mcnemar(b01, b10, exact=False),
            asym,
            rtol=1e-12,
            atol=1e-15,
            msg=f"chi2 {b01},{b10}",
        )


def test_holm_matches_reference():
    # WHY: adjusted p-values and rejections at three alphas on five families,
    #      including ties, a 0 and a 1, and a single test, from an independent
    #      numpy implementation of the same definition.
    # KIND: golden
    # CATCHES: s08, s09, s10
    # CHAPTER: M07.5 section 2.6, Many comparisons
    for p, adj, rej in golden()["holm"]:
        assert_close(holm_adjust(p), adj, rtol=1e-15, atol=1e-17, msg=str(p))
        for alpha, want in rej.items():
            assert holm(p, float(alpha)).tolist() == want, (p, alpha)


# --- the sampling behaviour -------------------------------------------------------------


def test_paired_type_one_rate():
    # WHY: a valid test rejects a true null at rate alpha. Under H0 (both
    #      scores from one distribution) the add-one p-value with 99
    #      permutations satisfies P(p <= 0.1) = 0.1 exactly; over 300 seeded
    #      datasets the rate must be 0.1 within 3.5 standard errors.
    # KIND: statistical
    # CATCHES: s01, m04
    # CHAPTER: M07.5 section 2.2, What a p-value promises
    g = PCG32(seed=seed() + 100)
    hits = 0
    for _ in range(300):
        a = [g.normal() for _ in range(10)]
        b = [g.normal() for _ in range(10)]
        hits += paired_permutation_test(a, b, 99, g) <= 0.1
    rate = hits / 300
    assert abs(rate - 0.1) <= 3.5 * math.sqrt(0.09 / 300), rate


def test_two_sample_type_one_rate():
    # WHY: the same promise for the two-sample test, with unequal groups
    #      (6 and 9). A relabelling that never mixes the groups rejects
    #      nothing: a perfect-looking test with no power.
    # KIND: statistical
    # CATCHES: s11
    # CHAPTER: M07.5 section 2.2, What a p-value promises
    g = PCG32(seed=seed() + 200)
    hits = 0
    for _ in range(300):
        a = [g.normal() for _ in range(6)]
        b = [g.normal() for _ in range(9)]
        hits += permutation_test(a, b, mean_difference, 99, g) <= 0.1
    rate = hits / 300
    assert abs(rate - 0.1) <= 3.5 * math.sqrt(0.09 / 300), rate


def test_tests_have_power():
    # WHY: a test that never rejects has a perfect type-I rate. With a real
    #      shift of one standard deviation (paired, n = 12) or 2 (two
    #      groups of 10) the tests must reject at 0.05 in most datasets.
    # KIND: statistical
    # CATCHES: s11
    # CHAPTER: M07.5 section 2.2, What a p-value promises
    g = PCG32(seed=seed() + 300)
    paired = two = 0
    for _ in range(60):
        a = [g.normal() for _ in range(12)]
        b = [x - 1.0 + 0.5 * g.normal() for x in a]
        paired += paired_permutation_test(a, b, 199, g) <= 0.05
        c = [g.normal() + 2.0 for _ in range(10)]
        d = [g.normal() for _ in range(10)]
        two += permutation_test(c, d, mean_difference, 199, g) <= 0.05
    assert paired >= 55 and two >= 50, (paired, two)


# --- the interface and boundaries -------------------------------------------------------


def test_draw_counts():
    # WHY: the contract's draw budget: n_perm * n uniforms for the paired test
    #      and n_perm * (N - 1) for the two-sample shuffle. Another count means
    #      another stream than the Go port and a different p-value.
    # KIND: unit
    # CATCHES: m09
    # CHAPTER: M07.5 section 4, The interface
    r = Counting(seed())
    paired_permutation_test([1.0, 2.0, 3.0, 4.0], [0.5, 2.5, 2.0, 3.0], 37, r)
    assert r.n == 37 * 4
    r = Counting(seed())
    permutation_test([1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0, 8.0], mean_difference, 29, r)
    assert r.n == 29 * 7


def test_fisher_yates_order():
    # WHY: Fisher-Yates from the end with j = min(floor(u (i + 1)), i):
    #      uniforms near 1 keep every element in place (the identity, so every
    #      permuted statistic equals the observed one: p = 1), and uniforms of
    #      0 swap each i with 0, dealing (5, 1 | 2, 3) from (3, 5 | 1, 2):
    #      |mean difference| 0.5 < 2.5, so p = 1 / (1 + n_perm).
    # KIND: unit
    # CATCHES: s03, s04, s11
    # CHAPTER: M07.5 section 2.4, The two-sample test
    a, b = [3.0, 5.0], [1.0, 2.0]
    assert permutation_test(a, b, mean_difference, 4, Scripted([0.999] * 12)) == 1.0
    assert permutation_test(a, b, mean_difference, 4, Scripted([0.0] * 12)) == 1 / 5


def test_two_sided():
    # WHY: "B differs from A" is two-sided: swapping the models (paired) or
    #      negating the statistic (two-sample) must give the same p-value from
    #      the same seed. A one-sided comparison halves p in one direction and
    #      sends it to 1 in the other.
    # KIND: property
    # CATCHES: s01
    # CHAPTER: M07.5 section 2.2, What a p-value promises
    g = PCG32(seed=seed() + 400)
    a = [g.normal() + 0.4 for _ in range(9)]
    b = [g.normal() for _ in range(9)]
    p1 = paired_permutation_test(a, b, 300, PCG32(seed=5))
    assert p1 == paired_permutation_test(b, a, 300, PCG32(seed=5))
    q1 = permutation_test(a, b, mean_difference, 300, PCG32(seed=6))
    q2 = permutation_test(a, b, lambda u, v: -mean_difference(u, v), 300, PCG32(seed=6))
    assert q1 == q2


def test_add_one_and_degenerate_data():
    # WHY: the add-one estimate is never 0: the most extreme data give
    #      1 / (1 + n_perm). With no differences at all every permutation ties
    #      the observed 0, so p = 1 exactly (not 0/0, not a crash).
    # KIND: boundary
    # CATCHES: s02
    # CHAPTER: M07.5 section 2.3, The paired test
    assert paired_permutation_test([1.0, 2.0], [1.0, 2.0], 50, PCG32(seed=1)) == 1.0
    assert (
        permutation_test(
            [4.0, 4.0], [4.0, 4.0, 4.0], mean_difference, 50, PCG32(seed=1)
        )
        == 1.0
    )
    p = paired_permutation_test([10.0] * 12, [0.0] * 12, 99, PCG32(seed=2))
    assert p == pytest.approx(1 / 100, rel=1e-15) or p == pytest.approx(
        2 / 100, rel=1e-15
    )
    assert p > 0


def test_rounding_never_splits_a_tie():
    # WHY: d = (-0.7, 0.2, 0.7, 0.1) has |sum| = 0.3, and four other sign
    #      patterns reach 0.3 exactly in real arithmetic but land a few ulps
    #      below it in float64. Comparing with a relative 1e-9 slack counts all
    #      12 of the 16 patterns: (1 + 12) / (1 + 16); a bare >= counts 10.
    # KIND: boundary
    # CATCHES: s05
    # CHAPTER: M07.5 section 5, Pitfalls
    rng = every_sign_pattern(4)
    p = paired_permutation_test([-0.7, 0.2, 0.7, 0.1], [0.0] * 4, 16, rng)
    assert p == pytest.approx(13 / 17, rel=1e-15)


def test_holm_steps_down_and_stops():
    # WHY: Holm is step-down: it stops at the first p-value above its bar,
    #      even if a later one would pass its own (0.04 <= 0.05 below, but
    #      0.03 > 0.025 stopped it). A bar met with equality rejects. On random
    #      families it rejects everything Bonferroni does, and holm(p, alpha)
    #      == holm_adjust(p) <= alpha.
    # KIND: property
    # CATCHES: s08, s09, s10, m05
    # CHAPTER: M07.5 section 2.6, Many comparisons
    assert holm([0.001, 0.04, 0.03], 0.05).tolist() == [True, False, False]
    assert holm([0.025, 0.05], 0.05).tolist() == [True, True]
    g = PCG32(seed=seed() + 500)
    for _ in range(200):
        m = 1 + g.below(8)
        p = np.array([g.uniform() ** 3 for _ in range(m)])
        for alpha in (0.01, 0.05, 0.2):
            r = holm(p, alpha)
            assert r.dtype == bool and r.shape == (m,)
            assert np.all(r[p <= alpha / m])
            assert r.tolist() == (holm_adjust(p) <= alpha).tolist()


def test_holm_adjust_is_monotone_and_capped():
    # WHY: adjusted p-values never decrease along the sorted order (a smaller
    #      raw p cannot get a larger verdict) and never exceed 1, so they can
    #      be reported in a table next to raw p-values.
    # KIND: property
    # CATCHES: s10, m06
    # CHAPTER: M07.5 section 2.6, Many comparisons
    g = PCG32(seed=seed() + 600)
    for _ in range(100):
        m = 1 + g.below(10)
        p = np.array([g.uniform() for _ in range(m)])
        adj = holm_adjust(p)
        srt = adj[np.argsort(p, kind="stable")]
        assert np.all(np.diff(srt) >= 0)
        assert np.all(adj <= 1.0) and np.all(adj >= p)
    assert holm(np.array([]), 0.05).shape == (0,)


def test_mcnemar_edges():
    # WHY: no discordant pairs is no evidence (p = 1, also for the
    #      chi-square, which would divide by 0); equal counts give p = 1, never
    #      above it; the test is symmetric in b01 and b10.
    # KIND: boundary
    # CATCHES: s07, m01
    # CHAPTER: M07.5 section 2.5, McNemar
    assert mcnemar(0, 0) == 1.0 and mcnemar(0, 0, exact=False) == 1.0
    assert mcnemar(5, 5) == 1.0 and mcnemar(3, 4) == 1.0
    assert mcnemar(12, 30) == mcnemar(30, 12)


def test_rejects_bad_arguments():
    # WHY: unpaired lengths, empty or non-finite samples, n_perm < 1,
    #      negative or fractional counts, p-values outside [0, 1], and alpha
    #      outside (0, 1) are caller bugs: they raise instead of returning a
    #      p-value someone will report.
    # KIND: boundary
    # CATCHES: m02, m03, m07
    # CHAPTER: M07.5 section 4, The interface
    g = PCG32(seed=0)
    for args in (
        ([1.0, 2.0], [1.0]),
        ([], []),
        ([1.0, math.nan], [1.0, 2.0]),
        ([[1.0]], [[2.0]]),
    ):
        with pytest.raises(ValueError):
            paired_permutation_test(*args, 10, g)
    for n_perm in (0, -3, 2.5):
        with pytest.raises(ValueError):
            paired_permutation_test([1.0, 2.0], [2.0, 1.0], n_perm, g)
        with pytest.raises(ValueError):
            permutation_test([1.0], [2.0], mean_difference, n_perm, g)
    for args in (([], [1.0]), ([1.0], [math.inf])):
        with pytest.raises(ValueError):
            permutation_test(*args, mean_difference, 10, g)
    for b01, b10 in ((-1, 3), (2.5, 1), (1, -2)):
        with pytest.raises(ValueError):
            mcnemar(b01, b10)
    for p in ([0.5, 1.5], [-0.1], [math.nan], [[0.1, 0.2]]):
        with pytest.raises(ValueError):
            holm(p)
        with pytest.raises(ValueError):
            holm_adjust(p)
    for alpha in (0.0, 1.0, -0.5):
        with pytest.raises(ValueError):
            holm([0.1], alpha)
