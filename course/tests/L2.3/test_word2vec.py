"""Course tests for L2.3: word2vec SGNS and the PPMI-SVD baseline
(tinyllm/lm/word2vec.py).

Rung R0 for these course tests (your own graded tests are rung R3: write
them first, see the chapter's "How to work this chapter"). Each test names
why it exists (WHY), what kind of check it is (KIND), the planted bugs it
kills (CATCHES, mutants in course/mutants/L2.3), and the chapter section it
comes from.

The chapter's worked example (section 3) is one SGNS pair in two dimensions:
center vector v = (0.5, -1), context vector u_o = (2, 1), one negative
u_k = (0, 1). The scores are s = u_o . v = 0 and t = u_k . v = -1, the loss
is ln 2 + ln(1 + e^-1), and the gradients are g_v = (-1, -0.5 + sigma(-1)),
g_pos = (-0.25, 0.5), g_neg = sigma(-1) (0.5, -1).

The word-level corpus is the MS-L2 training text (course/oracle/MS-L2/
corpus.py) split into lower-case words; its generator also wrote the gold
word-similarity pairs in $TINYLLM_FIXTURES/L2.3/wordsim.json (3 for two
words of one class, 1 for two nouns or two adjectives or two verbs of
different classes, 0 across parts of speech). The SGNS learning test
compares against course/fixtures/ref-thresholds.tsv (the reference's mean -
3 sd of Spearman's rho over 5 seeds).
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.gradcheck import gradcheck
from _lib.pcg32 import PCG32
from _lib.thresholds import check
from tinyllm.lm.word2vec import (
    LR_FLOOR,
    NOISE_POWER,
    SkipGramNS,
    analogy,
    cooccurrence,
    ppmi_svd_embeddings,
    sgns_loss,
    spearman,
    word_similarity,
    word_vocab,
)

FIX = Path(os.environ.get("TINYLLM_FIXTURES", ""))
SIG_M1 = 1.0 / (1.0 + math.e)  # sigma(-1)
# chi-square critical value for p = 1e-3 with 4 degrees of freedom
CHI2_P001_DF4 = 18.4668


def seed() -> int:
    return int(os.environ.get("OL_SEED", "0"))


class Rng:
    """The frozen PCG32 behind the generator API of M06.3 (uniform, below)."""

    def __init__(self, s: int, seq: int = 54) -> None:
        self.g = PCG32(seed=s, seq=seq)

    def next_u32(self) -> int:
        return self.g.next_u32()

    def uniform(self) -> float:
        return self.g.uniform()

    def below(self, n: int) -> int:
        return self.g.below(n)


class Script:
    """A generator that returns scripted values and counts the calls: the
    init uniforms first, then the values the test spells out."""

    def __init__(self, uniforms=(), belows=(), init: int = 0) -> None:
        self.u = [0.5] * init + list(uniforms)
        self.b = list(belows)
        self.calls_u = self.calls_b = 0

    def uniform(self) -> float:
        v = self.u[self.calls_u]
        self.calls_u += 1
        return v

    def below(self, n: int) -> int:
        v = self.b[self.calls_b]
        assert 0 <= v < n
        self.calls_b += 1
        return v


def words() -> list[str]:
    raw = np.fromfile(FIX / "MS-L2" / "train.bin", dtype="<u2", offset=1024)
    return re.findall(r"[a-z]+", bytes(raw.astype(np.uint8)).decode("ascii").lower())


def gold() -> list:
    return json.loads((FIX / "L2.3" / "wordsim.json").read_text())["pairs"]


# --- the worked example ---------------------------------------------------------------


def test_hand_example_sgns_loss():
    # WHY: the chapter's worked example, number for number: s = 0, t = -1,
    #      loss = ln 2 + ln(1 + e^-1), and the three gradients. The positive
    #      pair pulls with weight sigma(s) - 1 = -1/2; the negative pushes with
    #      weight sigma(t) = sigma(-1).
    # KIND: unit, smoke
    # CATCHES: s05, s06
    # CHAPTER: L2.3 section 3
    loss, gv, gp, gn = sgns_loss([[0.5, -1.0]], [[2.0, 1.0]], [[[0.0, 1.0]]])
    assert (
        loss.shape == (1,)
        and gv.shape == (1, 2)
        and gp.shape == (1, 2)
        and gn.shape == (1, 1, 2)
    )
    assert_close(loss, [math.log(2.0) + math.log1p(math.exp(-1.0))], dtype="float64")
    assert_close(gv, [[-1.0, -0.5 + SIG_M1]], dtype="float64")
    assert_close(gp, [[-0.25, 0.5]], dtype="float64")
    assert_close(gn, [[[0.5 * SIG_M1, -SIG_M1]]], dtype="float64")


def test_sgns_gradcheck():
    # WHY: the gradients are derived by hand (no autograd), so the frozen
    #      central-difference check is the referee: every entry of g_v, g_pos,
    #      and g_neg against the derivative of the summed loss, in float64,
    #      for 5 pairs with 3 negatives each.
    # KIND: gradcheck
    # CATCHES: s05, s06
    # CHAPTER: L2.3 section 2.3
    g = PCG32(seed=11)
    v, up, un = (
        g.normal_array((5, 4)),
        g.normal_array((5, 4)),
        g.normal_array((5, 3, 4)),
    )
    _, gv, gp, gn = sgns_loss(v, up, un)
    gradcheck(
        lambda a, b, c: float(sgns_loss(a, b, c)[0].sum()),
        [v, up, un],
        [gv, gp, gn],
        names=["v", "u_pos", "u_neg"],
    )


def test_sgns_loss_is_stable_at_large_scores():
    # WHY: trained vectors give scores of a few hundred; log(sigmoid(x))
    #      computed literally underflows to log(0) = -inf at x = -1000 and the
    #      run is lost. The stable form gives loss 1000 + ln 2 for s = -1000
    #      and t = 0, and gradients that stay finite.
    # KIND: boundary
    # CATCHES: s04, s06
    # CHAPTER: L2.3 section 5, Pitfalls, item 4
    loss, gv, gp, gn = sgns_loss([[10.0, 0.0]], [[-100.0, 0.0]], [[[0.0, 1.0]]])
    assert_close(loss, [1000.0 + math.log(2.0)], dtype="float64")
    assert np.isfinite(gv).all() and np.isfinite(gp).all() and np.isfinite(gn).all()
    assert_close(gp, [[-10.0, 0.0]], dtype="float64")
    loss, *_ = sgns_loss([[10.0, 0.0]], [[100.0, 0.0]], [[[100.0, 0.0]]])
    assert_close(loss, [1000.0], rtol=1e-12, atol=1e-12)


# --- counts, noise, subsampling ----------------------------------------------------------


def test_noise_and_keep_probabilities():
    # WHY: negatives come from the unigram distribution raised to 0.75,
    #      which lifts rare words; subsampling keeps a token of w with
    #      probability sqrt(t / f(w)), f the FREQUENCY (count over total).
    #      Counts (10, 0, 40, 13, 1, 64) with t = 0.01: f(w5) = 0.5, so it
    #      survives with probability sqrt(0.02) = 0.1414; a word never seen
    #      gets noise probability 0.
    # KIND: unit
    # CATCHES: s02, s03, m01
    # CHAPTER: L2.3 section 2.2
    counts = np.array([10, 0, 40, 13, 1, 64])
    m = SkipGramNS(6, 2, subsample_t=0.01, rng=Rng(0))
    m.set_counts(counts)
    w = counts.astype(float) ** NOISE_POWER
    assert NOISE_POWER == 0.75
    assert_close(m.noise_probs(), w / w.sum(), dtype="float64")
    f = counts / counts.sum()
    want = np.ones(6)
    want[f > 0] = np.minimum(1.0, np.sqrt(0.01 / f[f > 0]))
    assert_close(m.keep_probs(), want, dtype="float64")
    assert_close(m.keep_probs()[5], math.sqrt(0.02), dtype="float64")
    assert m.keep_probs()[4] == 1.0 and m.keep_probs()[1] == 1.0


def test_negative_sampler_chi_square():
    # WHY: with real uniforms from the frozen PCG32, 20 000 negatives fit the
    #      noise distribution (chi-square, p > 1e-3), the word of count 0 is
    #      never drawn, and the generator moves by exactly one uniform per
    #      draw (M07.1's alias table).
    # KIND: statistical
    # CATCHES: s02
    # CHAPTER: L2.3 section 2.2
    counts = np.array([10, 0, 40, 13, 1, 64])
    m = SkipGramNS(6, 2, rng=Rng(seed(), 5))
    m.set_counts(counts)
    draws = m.negatives(20_000)
    assert draws.dtype == np.int64 and draws.shape == (20_000,)
    got = np.bincount(draws, minlength=6).astype(float)
    assert got[1] == 0
    w = counts.astype(float) ** 0.75  # the distribution itself, not your noise_probs
    p = w / w.sum()
    keep = p > 0
    stat = float((((got - 20_000 * p) ** 2)[keep] / (20_000 * p[keep])).sum())
    assert stat < CHI2_P001_DF4, got.tolist()
    s = Script(uniforms=[0.1] * 3, init=12)
    m2 = SkipGramNS(6, 2, rng=s)
    m2.set_counts(counts)
    m2.negatives(3)
    assert s.calls_u == 12 + 3


def test_pairs_hand_example():
    # WHY: with no subsampling (t = 1 keeps every token), each center j gets
    #      a window r = 1 + below(window) drawn in order, and pairs with the
    #      kept tokens at distance <= r, left to right. Ids 0 1 2 3, window 2,
    #      below() = 1, 0, 0, 1: r = 2, 1, 1, 2.
    # KIND: unit
    # CATCHES: s07, m02
    # CHAPTER: L2.3 section 3
    s = Script(uniforms=[0.0] * 4, belows=[1, 0, 0, 1], init=4 * 2)
    m = SkipGramNS(4, 2, window=2, subsample_t=1.0, rng=s)
    m.set_counts([1, 1, 1, 1])
    c, o = m.pairs(np.array([0, 1, 2, 3]))
    assert c.dtype == np.int64 and o.dtype == np.int64
    assert list(zip(c.tolist(), o.tolist())) == [
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 2),
        (2, 1),
        (2, 3),
        (3, 1),
        (3, 2),
    ]
    assert s.calls_u == 8 + 4 and s.calls_b == 4


def test_subsampling_drops_before_windowing():
    # WHY: subsampling removes frequent tokens from the sequence BEFORE the
    #      windows are cut, so the words around a dropped "the" become
    #      neighbours: that widens the effective window. Counts make word 0
    #      frequent (f = 0.9, keep 0.105 at t = 0.01) and words 1, 2 rare
    #      (f = 0.05, keep 0.447). Uniforms 0.2 drop every 0 and keep every
    #      1 and 2; window 1 then pairs 1 with 2 across the dropped 0.
    # KIND: unit
    # CATCHES: s03, s08, m02
    # CHAPTER: L2.3 section 5, Pitfalls, item 3
    s = Script(uniforms=[0.2] * 3, belows=[0, 0], init=3)
    m = SkipGramNS(3, 1, window=1, subsample_t=0.01, rng=s)
    m.set_counts([18, 1, 1])
    c, o = m.pairs([1, 0, 2])
    assert list(zip(c.tolist(), o.tolist())) == [(1, 2), (2, 1)]


# --- training -----------------------------------------------------------------------------


def test_init_and_embeddings():
    # WHY: word2vec's initialization: input vectors (u - 0.5) / dim from one
    #      uniform per entry in C order, output vectors zero; embeddings()
    #      returns a copy of the INPUT table (the vectors the zoo scores).
    # KIND: unit
    # CATCHES: s13
    # CHAPTER: L2.3 section 2.3
    m = SkipGramNS(5, 3, rng=Rng(7))
    g = PCG32(seed=7)
    want = (np.array([g.uniform() for _ in range(15)]) - 0.5) / 3
    assert m.W_in.dtype == np.float64 and m.W_out.dtype == np.float64
    assert_close(m.W_in, want.reshape(5, 3), rtol=0.0, atol=0.0)
    assert (m.W_out == 0).all()
    e = m.embeddings()
    assert_close(e, m.W_in, rtol=0.0, atol=0.0)
    e[0, 0] = 99.0
    assert m.W_in[0, 0] != 99.0


def test_step_is_a_sparse_sum_of_gradients():
    # WHY: one step updates only the rows the batch touched, and a row used
    #      twice (center 2 appears twice, and output rows repeat across
    #      contexts and negatives) receives the SUM of its gradients: numpy's
    #      W[idx] -= g keeps only the last write for a repeated index. The
    #      returned loss is the batch mean before the update. Two models with
    #      the same seed draw the same negatives, so one predicts the other.
    # KIND: unit
    # CATCHES: s01, m03
    # CHAPTER: L2.3 section 5, Pitfalls, item 1
    counts = np.array([5, 3, 9, 2, 6, 1])
    a, b = SkipGramNS(6, 3, n_neg=2, rng=Rng(3)), SkipGramNS(6, 3, n_neg=2, rng=Rng(3))
    for m in (a, b):
        m.set_counts(counts)
        m.W_out[:] = PCG32(seed=4).normal_array((6, 3)) * 0.3
    c, o = np.array([2, 0, 2, 5]), np.array([1, 2, 4, 1])
    negs = a.negatives(8).reshape(4, 2)
    W_in, W_out = a.W_in.copy(), a.W_out.copy()
    loss, gv, gp, gn = sgns_loss(W_in[c], W_out[o], W_out[negs])
    for i in range(4):  # the sum, one pair at a time
        W_in[c[i]] -= 0.5 * gv[i]
        W_out[o[i]] -= 0.5 * gp[i]
        for j in range(2):
            W_out[negs[i, j]] -= 0.5 * gn[i, j]
    got = b.step(c, o, 0.5)
    assert_close(got, float(loss.mean()), dtype="float64")
    assert_close(b.W_in, W_in, rtol=1e-12, atol=1e-15)
    assert_close(b.W_out, W_out, rtol=1e-12, atol=1e-15)
    untouched = [r for r in range(6) if r not in c]
    assert_close(b.W_in[untouched], a.W_in[untouched], rtol=0.0, atol=0.0)


def test_fit_is_pairs_then_decayed_steps():
    # WHY: fit is the whole recipe: counts from the sequence, then per epoch
    #      fresh pairs and consecutive batches whose learning rate decays
    #      linearly over the run, lr * max(1e-4, 1 - (e + b / n) / epochs).
    #      Rebuilt here step by step, the two runs agree bit for bit.
    # KIND: differential
    # CATCHES: s03, s12
    # CHAPTER: L2.3 section 2.3
    ids = np.array([(i * 7 + i // 5) % 9 for i in range(60)])
    a = SkipGramNS(9, 4, n_neg=3, window=2, subsample_t=0.05, rng=Rng(seed(), 9))
    losses = a.fit(ids, 2, 16, 0.3)
    b = SkipGramNS(9, 4, n_neg=3, window=2, subsample_t=0.05, rng=Rng(seed(), 9))
    b.set_counts(np.bincount(ids, minlength=9))
    want = []
    for e in range(2):
        c, o = b.pairs(ids)
        starts = list(range(0, c.size, 16))
        ls = [
            b.step(
                c[i : i + 16],
                o[i : i + 16],
                0.3 * max(LR_FLOOR, 1 - (e + k / len(starts)) / 2),
            )
            for k, i in enumerate(starts)
        ]
        want.append(float(np.mean(ls)))
    assert LR_FLOOR == 1e-4
    assert_close(losses, want, rtol=0.0, atol=0.0)
    assert_close(a.W_in, b.W_in, rtol=0.0, atol=0.0)


def test_sgns_learns_word_classes():
    # WHY: the real check: five epochs over the 28 600 words of the MS-L2
    #      text must order the gold pairs like the reference does (Spearman's
    #      rho at least the reference's mean - 3 sd over 5 seeds): words that
    #      fill the same slots ("the ___ ran to the park") end up close.
    # KIND: learning
    # CATCHES: s06, m03
    # CHAPTER: L2.3 section 1
    vocab, ids = word_vocab(words())
    m = SkipGramNS(
        len(vocab), 16, n_neg=5, window=2, subsample_t=1e-3, rng=Rng(seed(), 1)
    )
    losses = m.fit(ids, 5, 64, 0.1)
    assert losses[-1] < losses[0]
    rho = word_similarity(m.embeddings(), vocab, gold())
    check("L2.3/test_sgns_learns_word_classes", "spearman", rho, direction="min")


# --- PPMI-SVD ---------------------------------------------------------------------------------


def test_cooccurrence_hand_example():
    # WHY: co-occurrence counts every ordered pair of positions within the
    #      window, both directions: ids 0 1 2 0 with window 1 see (0,1),
    #      (1,2), (2,0) and their mirrors, so cooc is symmetric.
    # KIND: unit
    # CATCHES: s14, m04
    # CHAPTER: L2.3 section 3
    c = cooccurrence(np.array([0, 1, 2, 0]), 3, 1)
    assert c.dtype == np.float64
    assert c.tolist() == [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    c = cooccurrence(np.array([0, 1, 2, 0]), 3, 2)
    assert c.tolist() == [[0, 2, 2], [2, 0, 1], [2, 1, 0]]


def numpy_ppmi_svd(cooc: np.ndarray, dim: int) -> np.ndarray:
    """An independent PPMI-SVD with numpy's LAPACK: PMI with context
    smoothing 0.75, clipped at 0, factored by np.linalg.svd."""
    total = cooc.sum()
    pw = cooc.sum(axis=1) / total
    cc = cooc.sum(axis=0) ** 0.75
    pc = cc / cc.sum()
    with np.errstate(divide="ignore"):
        pmi = np.log(cooc / total) - np.log(pw)[:, None] - np.log(pc)[None, :]
    m = np.maximum(pmi, 0.0)
    u, s, _ = np.linalg.svd(m)
    return u[:, :dim] * np.sqrt(s[:dim])


def cosines(e: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    n = e / np.linalg.norm(e, axis=1, keepdims=True)
    return np.array([n[a] @ n[b] for a, b in pairs])


def test_ppmi_svd_matches_numpy():
    # WHY: your PPMI (M11.4) and Jacobi SVD (M03.5) against numpy's LAPACK on
    #      the corpus's 151-word co-occurrence matrix: the embeddings are
    #      U_d sqrt(S_d), unique up to the sign of each column, so the cosine
    #      of every gold pair must agree, and the baseline already orders the
    #      pairs well (Spearman above 0.7).
    # KIND: differential
    # CATCHES: s09, s14
    # CHAPTER: L2.3 section 2.4
    vocab, ids = word_vocab(words())
    cooc = cooccurrence(ids, len(vocab), 2)
    e = ppmi_svd_embeddings(cooc, 16)
    assert e.shape == (len(vocab), 16) and e.dtype == np.float64
    idx = {w: i for i, w in enumerate(vocab)}
    pairs = [(idx[a], idx[b]) for a, b, _ in gold() if a in idx and b in idx]
    assert_close(
        cosines(e, pairs),
        cosines(numpy_ppmi_svd(cooc, 16), pairs),
        rtol=1e-6,
        atol=1e-8,
    )
    assert word_similarity(e, vocab, gold()) > 0.7


# --- utilities ----------------------------------------------------------------------------------


def test_analogy_hand_example():
    # WHY: "man is to king as woman is to ?" by 3CosAdd: the nearest word to
    #      king - man + woman. In these vectors the nearest word of all is
    #      woman (cosine 0.794, queen 0.783): a query word, as so often with
    #      real embeddings, so the three query words must be excluded. Then
    #      queen wins, and k = 2 adds the runner-up, apple.
    # KIND: unit
    # CATCHES: s10
    # CHAPTER: L2.3 section 3
    vocab = ["man", "woman", "king", "queen", "apple"]
    e = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.5, 1.0, 0.5],
            [0.0, 0.0, -1.0],
        ]
    )
    assert analogy(e, vocab, "man", "king", "woman") == ["queen"]
    assert analogy(e, vocab, "man", "king", "woman", k=2) == ["queen", "apple"]


def test_spearman_with_ties():
    # WHY: gold scores have only three levels (0, 1, 3), so ties are the
    #      rule: tied values share their average rank. x = (1, 2, 2, 3) has
    #      ranks (1, 2.5, 2.5, 4); against (1, 3, 2, 4) rho = 0.94868.
    # KIND: unit
    # CATCHES: s11
    # CHAPTER: L2.3 section 2.5
    assert_close(
        spearman([1, 2, 2, 3], [1, 3, 2, 4]),
        4.5 / math.sqrt(4.5 * 5.0),
        dtype="float64",
    )
    assert_close(spearman([3, 1, 2], [30, 10, 20]), 1.0, dtype="float64")
    assert_close(spearman([3, 1, 2], [-3, -1, -2]), -1.0, dtype="float64")
    with pytest.raises(ValueError):
        spearman([1, 1, 1], [1, 2, 3])


def test_word_similarity_covers_known_words_only():
    # WHY: the zoo scores any embedding table against a fixed pair list; a
    #      pair with a word outside the vocabulary is skipped, not an error,
    #      and the score is Spearman of cosine similarity against gold.
    # KIND: unit
    # CATCHES: m05
    # CHAPTER: L2.3 section 4
    vocab = ["a", "b", "c"]
    e = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    pairs = [("a", "b", 3), ("a", "c", 0), ("b", "c", 1), ("a", "zzz", 3)]
    assert_close(word_similarity(e, vocab, pairs), 1.0, dtype="float64")
    with pytest.raises(ValueError):
        word_similarity(e, vocab, [("a", "b", 1), ("x", "y", 0)])


def test_word_vocab_order():
    # WHY: ids are assigned by count, most frequent first, ties broken by
    #      the word itself (c before d, although d comes first in the text),
    #      so the same text gives the same ids on every machine; words below
    #      min_count are dropped from the ids.
    # KIND: unit
    # CATCHES: s15
    # CHAPTER: L2.3 section 4
    vocab, ids = word_vocab("b a d a b c a".split())
    assert (
        vocab == ["a", "b", "c", "d"]
        and ids.tolist() == [1, 0, 3, 0, 1, 2, 0]
        and ids.dtype == np.int64
    )
    vocab, ids = word_vocab("b a d a b c a".split(), min_count=2)
    assert vocab == ["a", "b"] and ids.tolist() == [1, 0, 0, 1, 0]


def test_input_validation():
    # WHY: sizes below 1, a non-positive subsampling threshold, counts of the
    #      wrong shape or sign, ids outside the vocabulary, and calls before
    #      set_counts are caller bugs that must fail at once.
    # KIND: boundary
    # CATCHES: m06
    # CHAPTER: L2.3 section 4
    for bad in (
        {"vocab": 0, "dim": 2},
        {"vocab": 3, "dim": 0},
        {"vocab": 3, "dim": 2, "n_neg": 0},
        {"vocab": 3, "dim": 2, "window": 0},
        {"vocab": 3, "dim": 2, "subsample_t": 0.0},
    ):
        with pytest.raises(ValueError):
            SkipGramNS(rng=Rng(0), **bad)
    m = SkipGramNS(3, 2, rng=Rng(0))
    with pytest.raises(RuntimeError):
        m.noise_probs()
    with pytest.raises(RuntimeError):
        m.pairs([0, 1])
    for counts in ([1, 2], [1, -1, 2], [0, 0, 0]):
        with pytest.raises(ValueError):
            m.set_counts(counts)
    m.set_counts([1, 1, 1])
    with pytest.raises(ValueError):
        m.pairs([0, 3])
    with pytest.raises(ValueError):
        m.step([0, 1], [1], 0.1)
    with pytest.raises(ValueError):
        m.step([], [], 0.1)
    with pytest.raises(ValueError):
        cooccurrence([0, 5], 3, 1)
    with pytest.raises(ValueError):
        ppmi_svd_embeddings(np.ones((3, 3)), 4)
    with pytest.raises(ValueError):
        analogy(np.eye(4), ["a", "b", "c", "d"], "a", "b", "x")
    with pytest.raises(ValueError):
        sgns_loss(np.ones((2, 3)), np.ones((2, 3)), np.ones((2, 1, 2)))
