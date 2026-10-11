"""Course tests for L0.0: the byte-level bigram model (tinyllm/lm/bigram.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/L0.0), and the chapter section it comes from.

The worked example of the chapter (section 3) is a three-symbol alphabet
a, b, c with ids 0, 1, 2 and the text "abbacab", ids [0, 1, 1, 0, 2, 0, 1].
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
from _lib.close import assert_close, assert_close_bounded
from _lib.pcg32 import PCG32
from tinyllm.lm.bigram import BigramLM

HAND_IDS = [0, 1, 1, 0, 2, 0, 1]  # "abbacab" over the alphabet a=0, b=1, c=2
# Counts + 1 per row, divided by the row total (chapter section 3).
HAND_PROBS = np.array(
    [[1 / 6, 3 / 6, 2 / 6], [2 / 5, 2 / 5, 1 / 5], [2 / 4, 1 / 4, 1 / 4]]
)
# The six predictions cost ln 2 three times, ln 2.5 twice, ln 3 once.
HAND_NLL = (3 * math.log(2) + 2 * math.log(2.5) + math.log(3)) / 6  # 0.835105882...


def hand_model() -> BigramLM:
    m = BigramLM()
    m.fit_counts(np.array(HAND_IDS), vocab_size=3, alpha=1.0)
    return m


def seed() -> int:
    return int(os.environ.get("OL_SEED", "0"))


def random_ids(rng: PCG32, n: int, vocab: int) -> np.ndarray:
    return np.array([rng.below(vocab) for _ in range(n)], dtype=np.int64)


def chi_square(model: BigramLM, ids: list[int], start: int, probs: np.ndarray) -> float:
    """Pearson chi-square of the observed transitions i -> j against probs[i, j],
    summed over every row the chain visited."""
    v = probs.shape[0]
    counts = np.zeros((v, v))
    prev = start
    for j in ids:
        counts[prev, j] += 1
        prev = j
    stat = 0.0
    for i in range(v):
        n_i = counts[i].sum()
        if n_i == 0:
            continue
        expected = n_i * probs[i]
        stat += float(((counts[i] - expected) ** 2 / expected).sum())
    return stat


# Critical value of chi-square with 6 degrees of freedom (3 rows x (3 - 1)) at p = 1e-3.
CHI2_DF6_P001 = 22.458


# --- fitting by counting ------------------------------------------------------


def test_hand_example_probabilities():
    # WHY: the chapter's worked example, number for number: you and the test
    #      agree on what "add-one smoothed bigram probability" means before
    #      any code runs. weight stores the log, so exp(weight) is the table.
    # KIND: unit
    # CATCHES: s01, s02, m02
    # CHAPTER: L0.0 section 3, Worked example by hand
    m = hand_model()
    assert m.vocab_size == 3
    assert m.weight.dtype == np.float32 and m.weight.shape == (3, 3)
    assert_close(np.exp(m.weight.astype(np.float64)), HAND_PROBS, dtype="float32")
    assert_close(
        np.exp(m.logits([0, 1, 2]).astype(np.float64)), HAND_PROBS, dtype="float32"
    )


def test_hand_example_nll():
    # WHY: the worked example's negative log-likelihood, (3 ln 2 + 2 ln 2.5 + ln 3) / 6.
    #      Six pairs, six predictions: the first token is never predicted.
    # KIND: unit
    # CATCHES: s06
    # CHAPTER: L0.0 section 3, Worked example by hand
    assert_close(hand_model().nll(np.array(HAND_IDS)), HAND_NLL, dtype="float32")


def test_counts_direction():
    # WHY: row = the token you are at, column = the token that follows. With
    #      the text "ab" only the pair (a, b) exists, so P(b | a) = 2/3 and the
    #      row of b, never seen as a context, is uniform. A transposed count
    #      table predicts the past from the future.
    # KIND: boundary
    # CATCHES: s02
    # CHAPTER: L0.0 section 5, Pitfalls, item 2
    m = BigramLM()
    m.fit_counts(np.array([0, 1]), vocab_size=2, alpha=1.0)
    assert_close(
        np.exp(m.weight.astype(np.float64)),
        [[1 / 3, 2 / 3], [1 / 2, 1 / 2]],
        dtype="float32",
    )


def test_rows_sum_to_one():
    # WHY: every row is a probability distribution, for every context,
    #      including the bytes the corpus never contains. The engine (L10.0)
    #      samples from these rows and assumes they are normalized.
    # KIND: property
    # CATCHES: s01
    # CHAPTER: L0.0 section 2, Principles (add-alpha smoothing)
    rng = PCG32(seed=seed())
    for alpha in (1.0, 0.5, 3.0):
        ids = random_ids(rng, 500, 64)  # bytes 64..255 never occur
        m = BigramLM()
        m.fit_counts(ids, vocab_size=256, alpha=alpha)
        assert m.weight.shape == (256, 256)
        assert np.isfinite(m.weight).all()
        sums = np.exp(m.weight.astype(np.float64)).sum(axis=1)
        assert_close_bounded(sums, np.ones(256), k=256, dtype="float32")


def test_nll_matches_counts():
    # WHY: the NLL of the count model is a closed form in the counts:
    #      -(1/N) sum_ij C[i, j] log((C[i, j] + alpha) / (R_i + alpha V)).
    #      The test recomputes it by plain loops, independently of your code.
    # KIND: differential
    # CATCHES: s06
    # CHAPTER: L0.0 section 2, Principles (negative log-likelihood)
    rng = PCG32(seed=seed())
    v, alpha = 5, 1.0
    ids = random_ids(rng, 400, v)
    counts = [[0] * v for _ in range(v)]
    for a, b in zip(ids[:-1], ids[1:]):
        counts[a][b] += 1
    total = 0.0
    for i in range(v):
        row = sum(counts[i])
        for j in range(v):
            if counts[i][j]:
                total -= counts[i][j] * math.log(
                    (counts[i][j] + alpha) / (row + alpha * v)
                )
    m = BigramLM()
    m.fit_counts(ids, vocab_size=v, alpha=alpha)
    assert_close(m.nll(ids), total / (len(ids) - 1), dtype="float32")


def test_nll_normalizes_logits():
    # WHY: nll must apply log-softmax to the logits, not read them as log
    #      probabilities. The count model's rows happen to be normalized
    #      already, but L0.5's trained logits are not: rows [0, 0] and [5, 5]
    #      both mean "uniform", so the NLL is ln 2 for any text.
    # KIND: unit
    # CATCHES: s10
    # CHAPTER: L0.0 section 5, Pitfalls, item 4
    m = BigramLM(np.array([[0.0, 0.0], [5.0, 5.0]], dtype=np.float32))
    assert_close(m.nll(np.array([0, 1, 1, 0])), math.log(2), dtype="float32")


def test_alpha_must_be_positive():
    # WHY: alpha = 0 is the raw maximum-likelihood estimate: an unseen pair
    #      gets probability 0 and log 0 = -inf, and a row never seen as a
    #      context is 0 / 0. Refuse it instead of writing -inf or nan.
    # KIND: boundary
    # CATCHES: m01
    # CHAPTER: L0.0 section 5, Pitfalls, item 1
    for alpha in (0.0, -1.0):
        with pytest.raises(ValueError):
            BigramLM().fit_counts(np.array(HAND_IDS), vocab_size=3, alpha=alpha)


def test_rejects_out_of_range_ids():
    # WHY: numpy indexing wraps negative ids (-1 is the last row) and a byte
    #      id of 256 does not exist. Both are bugs upstream, never data.
    # KIND: boundary
    # CATCHES: s03
    # CHAPTER: L0.0 section 5, Pitfalls, item 3
    for bad in ([0, -1, 2], [0, 3, 1]):
        with pytest.raises(ValueError):
            BigramLM().fit_counts(np.array(bad), vocab_size=3)
    with pytest.raises(ValueError):
        hand_model().logits([0, -1])


def test_rejects_non_integer_or_2d_ids():
    # WHY: ids index rows. Float ids would be cast (2.7 becomes 2) and a 2-D
    #      batch would pair tokens across row boundaries; both are caller
    #      bugs, so fit_counts refuses them instead of guessing.
    # KIND: boundary
    # CHAPTER: L0.0 section 4, The interface
    for bad in (np.array([0.0, 1.0, 2.0]), np.array([[0, 1], [1, 0]])):
        with pytest.raises(ValueError):
            BigramLM().fit_counts(bad, vocab_size=3)


def test_weight_is_checked_not_converted():
    # WHY: a loaded checkpoint becomes the model as is. A float64 table cast
    #      quietly to float32 hides a writer bug, and a non-finite logit turns
    #      into nan inside a matrix product (0 * inf). The constructor
    #      refuses them; a model with no table has no vocabulary yet.
    # KIND: boundary
    # CHAPTER: L0.0 section 4, The interface
    ok = np.zeros((3, 3), dtype=np.float32)
    assert BigramLM(ok).vocab_size == 3
    bad = [
        ok.astype(np.float64),
        np.zeros((3, 2), dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        np.array([[0.0, np.inf], [0.0, 0.0]], dtype=np.float32),
        np.array([[0.0, np.nan], [0.0, 0.0]], dtype=np.float32),
    ]
    for w in bad:
        with pytest.raises(ValueError):
            BigramLM(w)
    with pytest.raises(RuntimeError):
        _ = BigramLM().vocab_size


# --- logits by NumPy row gather -----------------------------------------------


def test_logits_are_weight_rows():
    # WHY: NumPy indexing picks row ids[t] of W directly, as float32 [T, V].
    #      This is the forward pass the Rust engine (L10.0) reproduces.
    # KIND: differential
    # CATCHES: m02
    # CHAPTER: L0.0 section 2, Principles (logits by row gather)
    rng = PCG32(seed=seed())
    m = BigramLM()
    m.fit_counts(random_ids(rng, 300, 256), vocab_size=256)
    ids = random_ids(rng, 17, 256)
    out = m.logits(ids)
    assert out.dtype == np.float32 and out.shape == (17, 256)
    assert np.array_equal(out, m.weight[ids])


# --- sampling -------------------------------------------------------------------


def test_greedy_ties_go_to_lowest_id():
    # WHY: greedy decoding takes the highest logit. In the hand model, after b
    #      the ids a and b tie at 2/5, and the rule is ties to the lowest id,
    #      so greedy from a cycles a -> b -> a. The engine and the
    #      tokens-equal matcher rely on this rule (D11).
    # KIND: unit
    # CATCHES: s07
    # CHAPTER: L0.0 section 3, Worked example by hand
    assert hand_model().sample([0], 4, temperature=0.0, seed=0) == [1, 0, 1, 0]


def test_temperature_zero_is_greedy():
    # WHY: temperature 0 means greedy whatever the seed: the limit of p ** (1/t).
    # KIND: unit
    # CATCHES: m03
    # CHAPTER: L0.0 section 2, Principles (temperature)
    m = hand_model()
    for s in (0, 1, 2):
        assert m.sample([2], 6, temperature=0.0, seed=s) == [0, 1, 0, 1, 0, 1]


def test_sample_returns_only_new_ids():
    # WHY: `generate` prints the generated ids only (spec/cli-roles.md), and
    #      the engine streams only new tokens. n = 0 is an empty list.
    # KIND: unit
    # CATCHES: s08
    # CHAPTER: L0.0 section 4, The interface
    m = hand_model()
    out = m.sample([0, 1, 2], 5, temperature=1.0, seed=3)
    assert len(out) == 5 and all(type(x) is int and 0 <= x < 3 for x in out)
    assert m.sample([0], 0, temperature=1.0, seed=3) == []


def test_sample_is_seeded():
    # WHY: the same seed gives the same ids, so a milestone can replay a run;
    #      different seeds explore different continuations.
    # KIND: property
    # CATCHES: m04
    # CHAPTER: L0.0 section 2, Principles (sampling)
    m = hand_model()
    runs = {s: m.sample([0], 40, temperature=1.0, seed=s) for s in (0, 1, 2, 3)}
    for s, ids in runs.items():
        assert m.sample([0], 40, temperature=1.0, seed=s) == ids
    assert len({tuple(x) for x in runs.values()}) > 1


def test_sample_frequencies_match_model():
    # WHY: the inverse CDF draws j with probability p[i, j]. Over 6000 steps
    #      the observed transitions must fit the model's table (chi-square,
    #      6 degrees of freedom, p > 1e-3, fixed seed). Drawing the same
    #      uniform at every step stays seeded but breaks the distribution.
    # KIND: statistical
    # CATCHES: s09
    # CHAPTER: L0.0 section 5, Pitfalls, item 6
    m = hand_model()
    ids = m.sample([0], 6000, temperature=1.0, seed=12345)
    probs = np.exp(m.weight.astype(np.float64))
    assert chi_square(m, ids, 0, probs) < CHI2_DF6_P001


def test_temperature_sharpens():
    # WHY: temperature divides the logits: softmax(z / t). At t = 0.5 each
    #      row becomes p ** 2 renormalized, sharper than the model. Multiplying
    #      by t would flatten it instead.
    # KIND: statistical
    # CATCHES: s05
    # CHAPTER: L0.0 section 2, Principles (temperature)
    m = hand_model()
    ids = m.sample([0], 6000, temperature=0.5, seed=12345)
    sharp = np.exp(2.0 * m.weight.astype(np.float64))
    sharp /= sharp.sum(axis=1, keepdims=True)
    assert chi_square(m, ids, 0, sharp) < CHI2_DF6_P001


def test_sample_rejects_bad_args():
    # WHY: the byte tokenizer has no start token, so an empty prefix has no
    #      row to read; a negative temperature or count is meaningless.
    # KIND: boundary
    # CATCHES: m05
    # CHAPTER: L0.0 section 4, The interface
    m = hand_model()
    with pytest.raises(ValueError):
        m.sample([], 3, temperature=1.0, seed=0)
    with pytest.raises(ValueError):
        m.sample([0], 3, temperature=-1.0, seed=0)
    with pytest.raises(ValueError):
        m.sample([0], -1, temperature=1.0, seed=0)
