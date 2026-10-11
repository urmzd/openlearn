"""Course tests for M07.2: estimating a distribution from counts
(tinyllm/prob/mle.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/M07.2), and the chapter section it comes from.

The chapter's worked example (section 3) counts a = 3, b = 2, c = 1, d = 0
over the five-word vocabulary a, b, c, d, e (e is never a key). Every
expected value in this file is an exact fraction worked out by hand.
"""

from __future__ import annotations

import math
import os
from fractions import Fraction

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32
from tinyllm.prob.mle import (
    absolute_discount,
    laplace,
    log_likelihood,
    mle,
    ney_discount,
)

HAND = {"a": 3, "b": 2, "c": 1, "d": 0}
VOCAB = ["a", "b", "c", "d", "e"]
UNIFORM5 = {w: 0.2 for w in VOCAB}


def seed() -> int:
    return int(os.environ.get("OL_SEED", "0"))


def close_dict(got: dict, want: dict) -> None:
    assert set(got) == set(want), (sorted(got), sorted(want))
    keys = sorted(want)
    assert_close(
        [got[k] for k in keys], [float(want[k]) for k in keys], dtype="float64"
    )


def random_counts(rng: PCG32, n_keys: int, zero_rate: float = 0.3) -> dict[int, int]:
    return {
        k: (0 if rng.uniform() < zero_rate else 1 + rng.below(9)) for k in range(n_keys)
    }


# --- the worked example -------------------------------------------------------


def test_hand_example_mle():
    # WHY: N = 6 observations, so p = 3/6, 2/6, 1/6, and the counted but
    #      unseen d gets 0. A word never seen in training gets probability 0,
    #      which is the problem the other two estimators solve.
    # KIND: unit
    # CATCHES: s01
    # CHAPTER: M07.2 section 3, Worked example by hand
    close_dict(
        mle(HAND),
        {"a": Fraction(1, 2), "b": Fraction(1, 3), "c": Fraction(1, 6), "d": 0},
    )


def test_hand_example_laplace():
    # WHY: add one to each of the 5 vocabulary words: the denominator is
    #      6 + 1 * 5 = 11, so a, b, c, d get 4/11, 3/11, 2/11, 1/11, and e,
    #      which is not a key, has 1/11 too. 4 + 3 + 2 + 1 + 1 = 11.
    # KIND: unit
    # CATCHES: s02, s03
    # CHAPTER: M07.2 section 3, Worked example by hand
    got = laplace(HAND, vocab_size=5, alpha=1.0)
    close_dict(
        got,
        {
            "a": Fraction(4, 11),
            "b": Fraction(3, 11),
            "c": Fraction(2, 11),
            "d": Fraction(1, 11),
        },
    )
    unseen_e = 1 / 11
    assert_close(sum(got.values()) + unseen_e, 1.0, dtype="float64")


def test_hand_example_absolute_discount():
    # WHY: d = 1/2 takes half a count from each of the T = 3 seen words, which
    #      frees 3 * 0.5 / 6 = 1/4 of the mass; a uniform backoff spreads it
    #      as 1/20 each. a = 2.5/6 + 1/20 = 7/15, b = 3/10, c = 2/15, and d, e
    #      get only the backoff share, 1/20.
    # KIND: unit
    # CATCHES: s04, s05, s06
    # CHAPTER: M07.2 section 3, Worked example by hand
    got = absolute_discount(HAND, 0.5, UNIFORM5)
    close_dict(
        got,
        {
            "a": Fraction(7, 15),
            "b": Fraction(3, 10),
            "c": Fraction(2, 15),
            "d": Fraction(1, 20),
            "e": Fraction(1, 20),
        },
    )


def test_hand_example_ney_discount():
    # WHY: one word seen once (c) and one seen twice (b): D = 1 / (1 + 2) =
    #      1/3. This is the discount L2.1 starts from.
    # KIND: unit
    # CATCHES: s08, m04
    # CHAPTER: M07.2 section 3, Worked example by hand
    assert_close(ney_discount(HAND), 1 / 3, dtype="float64")


def test_hand_example_log_likelihood():
    # WHY: the MLE scores the data 3 ln(1/2) + 2 ln(1/3) + ln(1/6) = -6.0684
    #      nats, and Laplace scores it lower, -7.3382: smoothing always costs
    #      likelihood on the training data and buys it back on new data.
    # KIND: unit
    # CATCHES: s01, s09
    # CHAPTER: M07.2 section 3, Worked example by hand
    want_mle = 3 * math.log(1 / 2) + 2 * math.log(1 / 3) + math.log(1 / 6)
    want_lap = 3 * math.log(4 / 11) + 2 * math.log(3 / 11) + math.log(2 / 11)
    assert_close(log_likelihood(HAND, mle(HAND)), want_mle, dtype="float64")
    assert_close(log_likelihood(HAND, laplace(HAND, 5)), want_lap, dtype="float64")
    assert want_lap < want_mle


# --- MLE ----------------------------------------------------------------------


def test_mle_maximizes_the_likelihood():
    # WHY: the claim that makes it "maximum likelihood": no distribution on
    #      the same outcomes gives the counts a higher log-likelihood. Random
    #      counts against random competitors and small perturbations.
    # KIND: property
    # CATCHES: s01
    # CHAPTER: M07.2 section 2, Principles (the MLE is the frequency)
    rng = PCG32(seed=seed())
    for _ in range(30):
        counts = random_counts(rng, 6)
        if sum(counts.values()) == 0:
            counts[0] = 1
        best = log_likelihood(counts, mle(counts))
        for _ in range(5):
            w = np.array([rng.uniform() + 1e-3 for _ in counts])
            q = dict(zip(counts, (w / w.sum()).tolist()))
            assert log_likelihood(counts, q) <= best + 1e-12
        p = mle(counts)
        keys = [k for k in counts if counts[k] > 0]
        if len(keys) >= 2:
            eps = 1e-3 * min(p[keys[0]], p[keys[1]])
            q = dict(p)
            q[keys[0]] += eps
            q[keys[1]] -= eps
            assert log_likelihood(counts, q) < best


def test_mle_sums_to_one_and_zero_counts_get_zero():
    # WHY: the estimate is a distribution over the keys, and a key that was
    #      counted zero times gets exactly 0, never a NaN or a missing key.
    # KIND: property
    # CATCHES: s01
    # CHAPTER: M07.2 section 2, Principles (the MLE is the frequency)
    rng = PCG32(seed=seed())
    for _ in range(20):
        counts = random_counts(rng, 8)
        counts[0] = counts[0] or 1
        p = mle(counts)
        assert set(p) == set(counts)
        assert_close(math.fsum(p.values()), 1.0, dtype="float64")
        assert all(p[k] == 0.0 for k in counts if counts[k] == 0)


# --- Laplace ------------------------------------------------------------------


def test_laplace_full_distribution_sums_to_one():
    # WHY: the vocabulary has vocab_size outcomes, of which only some are
    #      keys. The returned probabilities plus alpha / (N + alpha V) for
    #      each missing outcome must sum to 1; dividing by N + alpha * (number
    #      of keys) instead gives more than 1.
    # KIND: property
    # CATCHES: s02, s03
    # CHAPTER: M07.2 section 5, Pitfalls, item 2
    rng = PCG32(seed=seed())
    for alpha in (0.1, 1.0, 2.5):
        for _ in range(10):
            counts = random_counts(rng, 6)
            v = 6 + rng.below(20)
            n = sum(counts.values())
            p = laplace(counts, v, alpha)
            missing = (v - len(counts)) * alpha / (n + alpha * v)
            assert_close(math.fsum(p.values()) + missing, 1.0, dtype="float64")
            assert all(x > 0 for x in p.values())


def test_laplace_limits():
    # WHY: alpha interpolates between the data and ignorance. A tiny alpha
    #      gives back the MLE, a huge one the uniform 1/V, and with no data at
    #      all Laplace is exactly uniform.
    # KIND: unit
    # CATCHES: s02, s03, m03
    # CHAPTER: M07.2 section 2, Principles (add-alpha smoothing)
    tiny, ml = laplace(HAND, 5, 1e-9), mle(HAND)
    assert_close([tiny[k] for k in HAND], [ml[k] for k in HAND], rtol=0.0, atol=1e-8)
    big = laplace(HAND, 5, 1e9)
    assert_close(list(big.values()), [0.2] * 4, rtol=0.0, atol=1e-8)
    close_dict(laplace({"x": 0, "y": 0}, 4, 0.5), {"x": 0.25, "y": 0.25})
    full = {w: 1 for w in VOCAB}
    close_dict(laplace(full, 5, 1.0), {w: 0.2 for w in VOCAB})


# --- absolute discounting -----------------------------------------------------


def test_absolute_discount_sums_to_one():
    # WHY: the mass removed from the seen keys, d * T / N, is exactly the
    #      weight given to the backoff, so the result is a distribution for
    #      any d in (0, 1] and any backoff. Counting T over every key, or
    #      dropping T, breaks it as soon as a key has count 0.
    # KIND: property
    # CATCHES: s04, s05, s06
    # CHAPTER: M07.2 section 2, Principles (absolute discounting)
    rng = PCG32(seed=seed())
    for d in (0.1, 0.5, 0.75, 1.0):
        for _ in range(10):
            counts = random_counts(rng, 6)
            counts[1] = counts[1] or 2
            counts[2] = counts[2] * rng.uniform()  # an expected count from EM
            w = np.array([rng.uniform() + 1e-3 for _ in range(9)])
            backoff = dict(zip(range(9), (w / w.sum()).tolist()))
            p = absolute_discount(counts, d, backoff)
            assert set(p) == set(backoff)
            assert_close(math.fsum(p.values()), 1.0, dtype="float64")
            assert all(x > 0 for x in p.values())


def test_absolute_discount_zero_counts_are_clamped():
    # WHY: a key counted 0 times cannot lose d: max(0 - d, 0) = 0, so it gets
    #      only its backoff share. Without the clamp its probability is
    #      negative (here -0.5/6 + 1/20 < 0).
    # KIND: boundary
    # CATCHES: s06
    # CHAPTER: M07.2 section 5, Pitfalls, item 4
    p = absolute_discount(HAND, 0.5, UNIFORM5)
    assert_close(p["d"], 0.05, dtype="float64")
    assert min(p.values()) > 0


def test_absolute_discount_with_no_data_is_the_backoff():
    # WHY: N = 0 leaves nothing to discount, so the lower-order model is the
    #      answer. It is returned as a new dict: L2.1 fills it in per
    #      context and must not edit the shared backoff.
    # KIND: boundary
    # CATCHES: s07
    # CHAPTER: M07.2 section 4, The interface
    backoff = {"x": 0.25, "y": 0.75}
    got = absolute_discount({"x": 0}, 0.5, backoff)
    assert got == backoff and got is not backoff
    got["x"] = 1.0
    assert backoff["x"] == 0.25


def test_absolute_discount_d_one_removes_singletons():
    # WHY: with d = 1 a word seen once keeps nothing of its own count and
    #      survives only through the backoff, which is why Kneser-Ney uses
    #      d < 1 for singletons (Ney's D is at most 1).
    # KIND: boundary
    # CATCHES: s04, s05
    # CHAPTER: M07.2 section 2, Principles (absolute discounting)
    p = absolute_discount({"a": 1, "b": 3}, 1.0, {"a": 0.5, "b": 0.5})
    # lambda = 1 * 2 / 4 = 1/2: a = 0 + 1/4, b = 2/4 + 1/4.
    close_dict(p, {"a": 0.25, "b": 0.75})


# --- errors -------------------------------------------------------------------


def test_rejects_bad_arguments():
    # WHY: a negative, NaN, or infinite count, a bool or a string, an empty
    #      sample, alpha or d out of range, a backoff that is not a distribution, or a counted
    #      word the backoff does not know are bugs upstream in the n-gram
    #      counter (L2.1); failing loudly here is cheaper than a NaN later.
    # KIND: boundary
    # CATCHES: s11, s12, m01, m02, m03, m05
    # CHAPTER: M07.2 section 4, The interface
    for bad in ({"a": -1}, {"a": math.nan}, {"a": math.inf}, {"a": True}, {"a": "3"}):
        with pytest.raises(ValueError):
            mle(bad)
        with pytest.raises(ValueError):
            laplace(bad, 5)
    with pytest.raises(ValueError):
        mle({"a": 0})
    with pytest.raises(ValueError):
        mle({})
    for alpha in (0.0, -1.0):
        with pytest.raises(ValueError):
            laplace(HAND, 5, alpha)
    with pytest.raises(ValueError):
        laplace(HAND, 3)
    assert len(laplace(HAND, 4)) == 4
    for d in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError):
            absolute_discount(HAND, d, UNIFORM5)
    with pytest.raises(ValueError):
        absolute_discount(HAND, 0.5, {w: 0.3 for w in VOCAB})
    with pytest.raises(ValueError):
        absolute_discount({"z": 1}, 0.5, UNIFORM5)
    assert mle({"a": np.int64(2), "b": 2}) == {"a": 0.5, "b": 0.5}


def test_fractional_counts_from_em():
    # WHY: the E step of EM (L1.4) produces expected counts such as 0.3 and
    #      2.7. The MLE of fractional counts is still c / N. A count below d
    #      can only give up what it has, so the backoff weight is the sum of
    #      min(c, d) over N, here (0.3 + 0.5) / 3, not d T / N.
    # KIND: unit
    # CATCHES: s04, s05
    # CHAPTER: M07.2 section 2, Principles (absolute discounting)
    close_dict(mle({"a": 0.5, "b": 1.5}), {"a": 0.25, "b": 0.75})
    p = absolute_discount({"a": 0.3, "b": 2.7}, 0.5, {"a": 0.5, "b": 0.5})
    lam = 0.8 / 3
    close_dict(p, {"a": lam * 0.5, "b": 2.2 / 3 + lam * 0.5})
    assert_close(math.fsum(p.values()), 1.0, dtype="float64")


def test_ney_discount_cases():
    # WHY: D = n1 / (n1 + 2 n2) lies in (0, 1]: all singletons give 1, and
    #      with no singletons the estimate would be 0, which absolute
    #      discounting cannot use.
    # KIND: unit
    # CATCHES: s08, m04
    # CHAPTER: M07.2 section 2, Principles (estimating d)
    assert ney_discount({"a": 1, "b": 1}) == 1.0
    assert_close(ney_discount({"a": 1, "b": 2, "c": 2, "d": 5}), 1 / 5, dtype="float64")
    with pytest.raises(ValueError):
        ney_discount({"a": 2, "b": 3})


def test_log_likelihood_zero_terms_and_impossible_data():
    # WHY: an outcome counted 0 times contributes 0 * log q = 0 even when
    #      q = 0 (the convention 0 log 0 = 0). An outcome that was observed
    #      but has q = 0, or no entry, makes the data impossible: -inf. This
    #      is the infinite perplexity of an unsmoothed model (M11.2).
    # KIND: boundary
    # CATCHES: s09, s10
    # CHAPTER: M07.2 section 5, Pitfalls, item 5
    assert_close(
        log_likelihood({"a": 2, "b": 0}, {"a": 1.0, "b": 0.0}), 0.0, dtype="float64"
    )
    assert log_likelihood({"a": 2, "b": 1}, {"a": 1.0, "b": 0.0}) == -math.inf
    assert log_likelihood({"a": 2, "b": 1}, {"a": 1.0}) == -math.inf
    with pytest.raises(ValueError):
        log_likelihood({"a": -1}, {"a": 1.0})
