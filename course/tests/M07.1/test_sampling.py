"""Course tests for M07.1: categorical sampling (tinyllm/prob/sampling.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/M07.1), and the chapter section it comes from.

The chapter's worked example (section 3) is the 5-symbol distribution
p = [0.1, 0.2, 0.3, 0.4, 0.0]. Statistical tests either enumerate a grid of
uniforms (exact) or draw from the frozen PCG32 at OL_SEED and apply a
chi-square test at p > 1e-3 (DESIGN 4.0, KIND statistical).
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32
from tinyllm.prob.sampling import (
    AliasTable,
    exponential_icdf,
    gumbel_max,
    gumbel_noise,
    poisson_arrivals,
    sample_categorical,
)

HAND_P = [0.1, 0.2, 0.3, 0.4, 0.0]
# Chi-square critical values at p = 1e-3 by degrees of freedom.
CHI2_P001 = {2: 13.816, 3: 16.266, 4: 18.467}


def seed() -> int:
    return int(os.environ.get("OL_SEED", "0"))


class Script:
    """A uniform source that returns fixed values and counts the calls."""

    def __init__(self, values):
        self.values = list(values)
        self.calls = 0

    def uniform(self) -> float:
        v = self.values[self.calls]
        self.calls += 1
        return v


def implied(table: AliasTable) -> np.ndarray:
    """The distribution an alias table encodes, read off its two arrays:
    P(i) = (prob[i] + sum over j with alias[j] == i of (1 - prob[j])) / n."""
    n = len(table)
    out = np.array(table.prob, dtype=np.float64).copy()
    for j in range(n):
        out[int(table.alias[j])] += 1.0 - float(table.prob[j])
    return out / n


def chi_square(counts: np.ndarray, probs: np.ndarray) -> tuple[float, int]:
    """Pearson statistic over the cells with p > 0, and its degrees of freedom."""
    keep = probs > 0
    n = counts.sum()
    expected = n * probs[keep]
    stat = float(((counts[keep] - expected) ** 2 / expected).sum())
    return stat, int(keep.sum()) - 1


def grid(m: int) -> list[float]:
    """m uniforms at the midpoints (k + 1/2) / m: an exact enumeration of [0, 1)."""
    return [(k + 0.5) / m for k in range(m)]


# --- the worked example -------------------------------------------------------


def test_hand_example_inverse_cdf():
    # WHY: the chapter's worked example. The running sums are 0.1, 0.3, 0.6,
    #      1.0, 1.0, and the answer is the first id whose sum is strictly above
    #      u, so u = 0.1 lands on id 1, not id 0, and id 4 (p = 0) never wins.
    # KIND: unit
    # CATCHES: s01, m04
    # CHAPTER: M07.1 section 3, Worked example by hand
    assert sample_categorical(HAND_P, 0.05) == 0
    assert sample_categorical(HAND_P, 0.1) == 1
    assert sample_categorical(HAND_P, 0.25) == 1
    assert sample_categorical(HAND_P, 0.35) == 2
    assert sample_categorical(HAND_P, 0.99) == 3


def test_hand_example_alias_table():
    # WHY: Vose's algorithm on the worked example, step by step in the
    #      chapter: prob = [0.5, 1, 1, 0.5, 0], alias = [3, 1, 2, 2, 3]. Your
    #      table may pair columns in another order, so the test checks what
    #      any correct table must satisfy: it encodes p exactly, and a draw
    #      follows the contract's rule on your own two arrays.
    # KIND: unit
    # CATCHES: s05, s06, s07, m02
    # CHAPTER: M07.1 section 3, Worked example by hand
    t = AliasTable(HAND_P)
    assert len(t) == 5
    assert_close(implied(t), HAND_P, dtype="float64")
    us = [0.05, 0.13, 0.5, 0.95]
    got = t.sample(Script(us), len(us))
    assert got.dtype == np.int64
    want = []
    for u in us:
        x = u * 5
        i = min(int(math.floor(x)), 4)
        want.append(i if x - i < t.prob[i] else int(t.alias[i]))
    assert got.tolist() == want
    # With the chapter's table: u = 0.13 is column 0 (keep 0.5), coin 0.65: alias 3.
    ref = AliasTable(HAND_P)
    ref.prob = np.array([0.5, 1.0, 1.0, 0.5, 0.0])
    ref.alias = np.array([3, 1, 2, 2, 3], dtype=np.int64)
    assert ref.sample(Script(us), 4).tolist() == [0, 3, 2, 3]


def test_hand_example_gumbel_max():
    # WHY: with logits ln p and uniforms [0.9, 0.2, 0.5, 0.1], the noise is
    #      [2.2504, -0.4759, 0.3665, -0.8340] and the sums make id 0 win: a
    #      lucky draw lets the least likely id through, exactly as often as
    #      its probability says. Equal uniforms leave the argmax of the logits.
    # KIND: unit
    # CATCHES: s03, s04
    # CHAPTER: M07.1 section 3, Worked example by hand
    logits = np.log([0.1, 0.2, 0.3, 0.4])
    g = gumbel_noise([0.9, 0.2, 0.5, 0.1])
    assert_close(
        g, [2.250367327, -0.475884995, 0.366512921, -0.834032445], rtol=1e-9, atol=1e-9
    )
    assert gumbel_max(logits, g) == 0
    assert gumbel_max(logits, gumbel_noise([0.5] * 4)) == 3


def test_hand_example_poisson_arrivals():
    # WHY: rate 2 and uniforms 0.5, 0.75, 0.9 give gaps ln 2 / 2, ln 4 / 2,
    #      ln 10 / 2, so arrivals at 0.3466 and 1.0397; the third, 2.1910, is
    #      past the horizon 2 and is not an arrival. The load generator
    #      (load.01) replays this exact schedule in Go.
    # KIND: unit
    # CATCHES: s10, s11, m03
    # CHAPTER: M07.1 section 3, Worked example by hand
    src = Script([0.5, 0.75, 0.9])
    t = poisson_arrivals(2.0, 2.0, src)
    assert_close(
        t, [math.log(2) / 2, math.log(2) / 2 + math.log(4) / 2], dtype="float64"
    )
    assert src.calls == 3


# --- inverse CDF ------------------------------------------------------------------


def test_inverse_cdf_exact_enumeration():
    # WHY: if u is uniform, id i is returned for a set of u of length p_i.
    #      Midpoints of a 1000-cell grid hit each id exactly 1000 * p_i times
    #      when every p_i is a multiple of 1/1000: no sampling noise at all.
    # KIND: statistical
    # CATCHES: s01, s02, m04
    # CHAPTER: M07.1 section 2, Principles (inverse CDF)
    for p in ([0.1, 0.2, 0.3, 0.4, 0.0], [0.0, 0.25, 0.25, 0.0, 0.5], [0.2] * 5):
        counts = np.zeros(5, dtype=np.int64)
        for u in grid(1000):
            counts[sample_categorical(p, u)] += 1
        assert counts.tolist() == [round(1000 * x) for x in p]


def test_zero_probability_is_never_returned():
    # WHY: with "u <= c" instead of "u < c", u = 0 returns id 0 even when
    #      p_0 = 0. In the sampler (L8.1) that id is a token top-k removed:
    #      a masked token leaks out about once in 2^53 draws, in production.
    # KIND: boundary
    # CATCHES: s01
    # CHAPTER: M07.1 section 5, Pitfalls, item 1
    assert sample_categorical([0.0, 0.5, 0.5], 0.0) == 1
    assert sample_categorical([0.0, 0.0, 1.0], 0.0) == 2
    assert sample_categorical([0.5, 0.0, 0.5], 0.5) == 2


def test_rounding_fallback_skips_a_zero_tail():
    # WHY: ten 0.1s add up to 0.9999999999999999, which equals the largest
    #      uniform, 1 - 2^-53, so no running sum is above u. The fallback must
    #      be the last id that can occur, not the last id: here id 10 has
    #      p = 0 (spec/sampling.md step 11, "the largest id in K").
    # KIND: boundary
    # CATCHES: s02
    # CHAPTER: M07.1 section 5, Pitfalls, item 2
    u_max = 1.0 - 2.0**-53
    p = [0.1] * 10 + [0.0]
    c = 0.0
    for x in p:
        c += (
            x  # left to right, as the walk adds (not sum(), which compensates in 3.12+)
        )
    assert c == u_max
    assert sample_categorical(p, u_max) == 9
    assert sample_categorical([0.5, 0.5, 0.0, 0.0], u_max) == 1


def test_last_step_of_the_sampling_spec():
    # WHY: spec/sampling.md's worked example ends with q = 0.5 on ids 1 and
    #      3 and u = 0.80209...: the walk passes id 1 (c = 0.5) and returns id
    #      3. The L8.1 sampler calls this function for its step 11.
    # KIND: unit
    # CATCHES: s01, m04
    # CHAPTER: M07.1 section 6, Where it's used next
    q = [0.0, 0.5, 0.0, 0.5, 0.0]
    assert sample_categorical(q, 0.80209) == 3
    assert sample_categorical(q, 0.49999) == 1


def test_rejects_bad_arguments():
    # WHY: a distribution that does not sum to 1, a negative or NaN entry, or
    #      u outside [0, 1) is a bug upstream (a missed softmax, a broken
    #      generator); silently sampling hides it. The tolerance is loose
    #      enough for a float64 softmax.
    # KIND: boundary
    # CATCHES: s14, m01
    # CHAPTER: M07.1 section 4, The interface
    for p in (
        [0.5, 0.6],
        [1.5, -0.5],
        [0.5, float("nan")],
        [],
        [[0.5, 0.5]],
        [0.2, 0.2],
    ):
        with pytest.raises(ValueError):
            sample_categorical(p, 0.1)
        with pytest.raises(ValueError):
            AliasTable(p)
    for u in (1.0, -0.1, float("nan"), 1.5):
        with pytest.raises(ValueError):
            sample_categorical([0.5, 0.5], u)
    assert sample_categorical([1 / 3] * 3, 0.5) == 1


# --- Gumbel-max ---------------------------------------------------------------------


def test_gumbel_noise_values():
    # WHY: G = -log(-log U). The sign matters: log(-log U) is the noise of the
    #      minimum, and adding it picks ids with the wrong probabilities. U
    #      must be strictly inside (0, 1), where both logs are finite.
    # KIND: unit
    # CATCHES: s04
    # CHAPTER: M07.1 section 2, Principles (Gumbel-max)
    assert_close(gumbel_noise([math.exp(-1.0)]), [0.0], dtype="float64")
    assert_close(gumbel_noise([0.5]), [-math.log(math.log(2.0))], dtype="float64")
    for bad in ([0.0], [1.0], [0.5, 1.2]):
        with pytest.raises(ValueError):
            gumbel_noise(bad)


def test_gumbel_max_distribution_is_softmax():
    # WHY: the claim of the trick, tested: argmax(z + G) is distributed as
    #      softmax(z). 20 000 draws from the frozen PCG32, chi-square at
    #      p > 1e-3. Logits are not normalized (softmax ignores a shift).
    # KIND: statistical
    # CATCHES: s03, s04
    # CHAPTER: M07.1 section 2, Principles (Gumbel-max)
    rng = PCG32(seed=seed())
    z = np.array([1.0, 2.0, 0.5, 3.0, 2.0])
    p = np.exp(z - z.max())
    p /= p.sum()
    counts = np.zeros(5)
    for _ in range(20_000):
        u = [rng.uniform() for _ in range(5)]
        u = [x if x > 0 else 0.5 for x in u]  # u == 0 has probability 2^-53
        counts[gumbel_max(z, gumbel_noise(u))] += 1
    stat, df = chi_square(counts, p)
    assert stat < CHI2_P001[df], (counts.tolist(), (20_000 * p).tolist())


def test_gumbel_max_masks_and_ties():
    # WHY: a -inf logit (masked by top-k or a grammar in L10.9) must never
    #      win, whatever the noise; equal sums go to the lowest id, the rule
    #      every engine shares (D11); all -inf, NaN, or +inf are errors.
    # KIND: boundary
    # CATCHES: s12, s13
    # CHAPTER: M07.1 section 5, Pitfalls, item 4
    ninf = -np.inf
    assert gumbel_max([ninf, 0.0, ninf], [100.0, -100.0, 100.0]) == 1
    assert gumbel_max([1.0, 0.0, 1.0], [0.0, 1.0, 0.0]) == 0
    for z in ([ninf, ninf], [0.0, float("nan")], [np.inf, 0.0]):
        with pytest.raises(ValueError):
            gumbel_max(z, [0.0, 0.0])
    with pytest.raises(ValueError):
        gumbel_max([0.0, 1.0], [0.0])


# --- the alias method -----------------------------------------------------------------


def test_alias_table_encodes_the_distribution():
    # WHY: whatever order your algorithm pairs columns in, each column holds
    #      exactly 1/n of mass, split between its own id and its alias, and
    #      the shares add up to p. Random distributions, some with zeros.
    # KIND: property
    # CATCHES: s05, s06, m02
    # CHAPTER: M07.1 section 2, Principles (the alias method)
    rng = PCG32(seed=seed())
    for n in (1, 2, 3, 5, 17, 64):
        w = np.array([rng.uniform() if rng.uniform() > 0.3 else 0.0 for _ in range(n)])
        if w.sum() == 0:
            w[0] = 1.0
        p = w / w.sum()
        t = AliasTable(p)
        assert len(t) == n
        prob, alias = np.asarray(t.prob), np.asarray(t.alias)
        assert prob.shape == (n,) and alias.shape == (n,)
        assert ((prob >= 0) & (prob <= 1)).all()
        assert ((alias >= 0) & (alias < n)).all()
        assert_close(implied(t), p, rtol=1e-12, atol=1e-12)


def test_alias_exact_enumeration():
    # WHY: a draw is a column (the integer part of n u) and a coin (the
    #      fraction). Grid midpoints enumerate both exactly: 10 coins per
    #      column give 10 * n * p_i hits for id i when the thresholds are
    #      multiples of 0.1.
    # KIND: statistical
    # CATCHES: s05, s06, s07
    # CHAPTER: M07.1 section 2, Principles (the alias method)
    for p in ([0.1, 0.2, 0.3, 0.4, 0.0], [0.0, 0.0, 0.6, 0.0, 0.4]):
        t = AliasTable(p)
        draws = t.sample(Script(grid(50)), 50)
        counts = np.bincount(draws, minlength=5)
        assert counts.tolist() == [round(50 * x) for x in p]


def test_alias_sample_chi_square():
    # WHY: with real uniforms from the frozen PCG32, 20 000 draws fit p
    #      (chi-square, p > 1e-3) and the zero-probability id never appears.
    #      This is the negative sampler of word2vec (L2.3).
    # KIND: statistical
    # CATCHES: s05, s06, s07
    # CHAPTER: M07.1 section 2, Principles (the alias method)
    p = np.array([0.05, 0.25, 0.0, 0.3, 0.4])
    draws = AliasTable(p).sample(PCG32(seed=seed()), 20_000)
    counts = np.bincount(draws, minlength=5).astype(float)
    assert counts[2] == 0
    stat, df = chi_square(counts, p)
    assert stat < CHI2_P001[df], counts.tolist()


def test_alias_one_uniform_per_draw():
    # WHY: the contract fixes one uniform per draw (column and coin from the
    #      same u), so a seed gives the same ids in every language and a
    #      caller knows how far the generator moved. Two uniforms per draw is
    #      still a correct sampler and still a broken contract.
    # KIND: unit
    # CATCHES: s09
    # CHAPTER: M07.1 section 5, Pitfalls, item 6
    src = Script(grid(7))
    out = AliasTable(HAND_P).sample(src, 7)
    assert len(out) == 7 and src.calls == 7
    empty = AliasTable(HAND_P).sample(Script([]), 0)
    assert empty.dtype == np.int64 and empty.shape == (0,)
    with pytest.raises(ValueError):
        AliasTable(HAND_P).sample(Script([]), -1)


def test_alias_zero_probability_never_drawn_at_column_edges():
    # WHY: u = k / n lands exactly on a column edge with coin f = 0. A coin
    #      test "f <= prob[i]" keeps a column whose prob is 0, so an id with
    #      p = 0 appears. The edges are rare with real uniforms and certain
    #      with these.
    # KIND: boundary
    # CATCHES: s08
    # CHAPTER: M07.1 section 5, Pitfalls, item 5
    p = [0.0, 0.5, 0.0, 0.5]
    t = AliasTable(p)
    edges = [k / 4 for k in range(4)]
    draws = t.sample(Script(edges), 4)
    assert set(draws.tolist()) <= {1, 3}


def test_alias_single_outcome():
    # WHY: n = 1 is a column that always keeps itself, for any u.
    # KIND: boundary
    # CATCHES: s05
    # CHAPTER: M07.1 section 4, The interface
    t = AliasTable([1.0])
    assert t.sample(Script([0.0, 0.5, 1.0 - 2.0**-53]), 3).tolist() == [0, 0, 0]


# --- exponential and Poisson -----------------------------------------------------------


def test_exponential_icdf():
    # WHY: F(t) = 1 - e^(-rate t) inverts to t = -log(1 - u) / rate. Writing
    #      it as log1p(-u) keeps u near 0 accurate, and u = 0 gives 0, never
    #      log(0). Using log(u) instead is the same distribution but a
    #      different schedule from the same seed.
    # KIND: unit
    # CATCHES: s10, m03
    # CHAPTER: M07.1 section 2, Principles (continuous inverse CDF)
    assert exponential_icdf(0.0, 3.0) == 0.0
    assert_close(exponential_icdf(0.5, 2.0), math.log(2) / 2, dtype="float64")
    assert_close(exponential_icdf(1e-17, 1.0), 1e-17, dtype="float64")
    for u, r in ((1.0, 1.0), (-0.1, 1.0), (0.5, 0.0), (0.5, -1.0)):
        with pytest.raises(ValueError):
            exponential_icdf(u, r)


def test_poisson_arrivals_horizon_and_draws():
    # WHY: one uniform per arrival plus the one that crosses the horizon, and
    #      an arrival exactly at the horizon is outside [0, horizon). With
    #      horizon 0 nothing arrives even when the first gap is 0.
    # KIND: boundary
    # CATCHES: s11, m05
    # CHAPTER: M07.1 section 4, The interface
    src = Script([0.0, 0.5])
    assert poisson_arrivals(1.0, 0.0, src).tolist() == []
    assert src.calls == 1
    with pytest.raises(ValueError):
        poisson_arrivals(0.0, 1.0, Script([0.5]))
    with pytest.raises(ValueError):
        poisson_arrivals(1.0, -1.0, Script([0.5]))


def test_poisson_arrivals_rate():
    # WHY: over 200 windows of length 10 at rate 3, the count has mean 30
    #      and variance 30 (a Poisson count's variance equals its mean, index
    #      of dispersion near 1). One long run of length 2000 has gaps with
    #      mean 1/3. Gaps are read from one long run, because inside short
    #      windows the gap that crosses the edge is cut off, and long gaps
    #      are the likeliest to be cut, which biases the mean down.
    # KIND: statistical
    # CATCHES: s10, m03
    # CHAPTER: M07.1 section 2, Principles (Poisson arrivals)
    rng = PCG32(seed=seed())
    counts = np.array(
        [len(poisson_arrivals(3.0, 10.0, rng)) for _ in range(200)], dtype=float
    )
    assert abs(counts.mean() - 30.0) < 4 * math.sqrt(30.0 / 200)
    assert 0.7 < counts.var(ddof=1) / counts.mean() < 1.4
    gaps = np.diff(np.concatenate([[0.0], poisson_arrivals(3.0, 2000.0, rng)]))
    assert abs(gaps.mean() - 1 / 3) < 4 * (1 / 3) / math.sqrt(gaps.size)
