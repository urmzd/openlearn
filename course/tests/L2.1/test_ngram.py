"""Course tests for L2.1: NGramLM, interpolated modified Kneser-Ney
(tinyllm/lm/ngram.py).

Rung R0 for these course tests (your own graded tests are rung R3: write
them first, see the chapter's "How to work this chapter"). Each test names
why it exists (WHY), what kind of check it is (KIND), the planted bugs it
kills (CATCHES, mutants in course/mutants/L2.1), and the chapter section it
comes from.

The chapter's worked example (section 3) is the corpus a b c / a b a b /
c a b c over V = 3 tokens (a = 0, b = 1, c = 2), a bigram model with a
fixed discount d = 1/2. b is the most frequent token after a but follows
nothing else, so its continuation count is 1 and the unigram level gives it
1/6, not the 4/11 raw counts would. The golden values in
$TINYLLM_FIXTURES/L2.1/kn_golden.json come from an independent exact
(fractions.Fraction) implementation, course/oracle/L2.1/kn_golden.py, on
that corpus and on the bytes of course/fixtures/MS-P1/corpus.txt.
"""

from __future__ import annotations

import json
import math
import os
import struct
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32
from tinyllm.lm.ngram import FALLBACK_DISCOUNTS, NGramLM

FIX = Path(os.environ.get("TINYLLM_FIXTURES", ""))
A, B, C = 0, 1, 2
HAND = [[A, B, C], [A, B, A, B], [C, A, B, C]]


def rng(seq: int) -> PCG32:
    return PCG32(int(os.environ.get("OL_SEED", "0")), seq=seq)


def golden() -> dict:
    return json.loads((FIX / "L2.1" / "kn_golden.json").read_text())["cases"]


def corpus_lines() -> list[list[int]]:
    text = (FIX / "MS-P1" / "corpus.txt").read_bytes()
    return [list(line) for line in text.split(b"\n") if line]


def fitted(seqs, n, discount="modified", V=None) -> NGramLM:
    lm = NGramLM(n, discount, V)
    lm.fit(seqs)
    return lm


def ctx_of(key: str) -> list[int]:
    return [] if key == "<s>" else [int(x) for x in key.split()]


def test_hand_example_bigram():
    # WHY: the chapter's worked example. Unigram level from continuation
    #      counts a: 3 (after <s>, b, c), b: 1 (after a only), c: 2:
    #      T = 6, gamma = 3 * 1/2 / 6 = 1/4, so p1 = (1/2, 1/6, 1/3).
    #      After b (b a once, b c twice): p(. | b) = (1/3, 1/18, 11/18).
    # KIND: unit, smoke
    # CATCHES: s01, m01
    # CHAPTER: L2.1 section 3
    lm = fitted(HAND, 2, 0.5, 3)
    assert_close(
        [lm.prob([B], w) for w in (A, B, C)],
        [1 / 3, 1 / 18, 11 / 18],
        rtol=0.0,
        atol=1e-15,
    )
    # After c (only c a): p(. | c) = (1/2 + 1/2 * 1/2, 1/2 * 1/6, 1/2 * 1/3).
    assert_close(
        [lm.prob([C], w) for w in (A, B, C)],
        [3 / 4, 1 / 12, 1 / 6],
        rtol=0.0,
        atol=1e-15,
    )
    assert lm.discounts(1) == (0.5, 0.5, 0.5) and lm.discounts(2) == (0.5, 0.5, 0.5)


def test_hand_example_sequence_start():
    # WHY: the first token is predicted from <s>, interpolated with the
    #      KN unigram: <s> a twice, <s> c once gives T = 3, gamma = 1/3,
    #      p(. | <s>) = (2/3, 1/18, 5/18). (In a trigram, <s> a is a lower
    #      order and still keeps its raw count, since nothing precedes <s>;
    #      test_golden_hand_cases checks that.)
    # KIND: unit, smoke
    # CATCHES: s01, m01
    # CHAPTER: L2.1 section 3
    lm = fitted(HAND, 2, 0.5, 3)
    assert_close(
        [lm.prob([], w) for w in (A, B, C)],
        [2 / 3, 1 / 18, 5 / 18],
        rtol=0.0,
        atol=1e-15,
    )


def test_golden_hand_cases():
    # WHY: every context of the worked example (bigram, d = 1/2) and the same
    #      corpus as a modified-KN trigram, against exact fractions from the
    #      independent oracle. The trigram mixes sequence-start histories
    #      (<s> a) with mid-sequence ones (b a).
    # KIND: golden
    # CATCHES: s01, s02, s03, s06, m01
    # CHAPTER: L2.1 section 2.4
    for name in ("hand", "hand_trigram"):
        case = golden()[name]
        disc = (
            case["discount"]
            if case["discount"] == "modified"
            else float(case["discount"])
        )
        lm = fitted(case["sequences"], case["n"], disc, case["vocab_size"])
        for key, probs in case["probs"].items():
            got = [lm.prob(ctx_of(key), w) for w in range(case["vocab_size"])]
            assert_close(
                got,
                [float(Fraction(p)) for p in probs],
                rtol=0.0,
                atol=1e-15,
                msg=f"{name} {key}",
            )


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_golden_byte_corpus(n):
    # WHY: the MS-P1 corpus as bytes (V = 256), trained on all but the last
    #      five lines, scored on those five: the discounts of every order
    #      (Chen and Goodman's formulas from the counts of counts, or the
    #      fallback), held-out perplexities, and probabilities after "the "
    #      must match the exact oracle, for modified KN and for d = 0.75.
    # KIND: golden
    # CATCHES: s01, s07, s08, s11, m01
    # CHAPTER: L2.1 section 2.5
    lines = corpus_lines()
    train, held = lines[:-5], lines[-5:]
    for mode in ("modified", "0.75"):
        case = golden()[f"ms_p1_n{n}_{mode}"]
        lm = fitted(train, n, mode if mode == "modified" else 0.75, 256)
        for k, ds in case["discounts"].items():
            assert_close(
                lm.discounts(int(k)),
                [float(Fraction(d)) for d in ds],
                rtol=1e-14,
                atol=0.0,
            )
        assert_close(
            [lm.perplexity(s) for s in held],
            [float(p) for p in case["ppl"]],
            rtol=1e-12,
            atol=0.0,
        )
        probes = case["probe_probs"]
        got = [lm.prob(case["probe_context"], int(w)) for w in probes]
        assert_close(
            got, [float(Fraction(p)) for p in probes.values()], rtol=1e-13, atol=0.0
        )


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_probabilities_sum_to_one(n):
    # WHY: sum over v of p(v | ctx) = 1 for every context: the discounted
    #      mass is exactly what gamma hands down, as long as gamma charges
    #      D1, D2, D3 by count bucket and every D_j <= j. Seen, unseen, and
    #      sequence-start contexts all count.
    # KIND: property
    # CATCHES: s02, s03, s07, m02, m04
    # CHAPTER: L2.1 section 2.4
    lines = corpus_lines()
    lm = fitted(lines, n, "modified", 256)
    g = rng(21)
    contexts = [[], list(b"th"), list(b"the c"), [255, 0, 7]]
    for _ in range(10):
        line = lines[g.below(len(lines))]
        t = g.below(len(line))
        contexts.append(line[max(0, t - 5) : t])
    for ctx in contexts:
        s = math.fsum(lm.prob(ctx, w) for w in range(256))
        assert_close(s, 1.0, rtol=0.0, atol=1e-13, msg=f"context {ctx}")


def test_logprobs_agree_with_prob():
    # WHY: logprobs is the vectorized path L8.6's draft model and L6.7's
    #      zoo call; it must give the log of the same numbers prob gives, for
    #      every token at once, with logsumexp 0.
    # KIND: property
    # CATCHES: s10
    # CHAPTER: L2.1 section 4
    lines = corpus_lines()
    lm = fitted(lines, 3, "modified", 256)
    for ctx in ([], list(b"a"), list(b"the "), list(b"zq")):
        lp = lm.logprobs(ctx)
        assert lp.shape == (256,) and lp.dtype == np.float64
        assert_close(
            lp, [math.log(lm.prob(ctx, w)) for w in range(256)], rtol=1e-13, atol=1e-13
        )
        m = float(lp.max())
        assert_close(
            m + math.log(math.fsum(np.exp(lp - m).tolist())), 0.0, rtol=0.0, atol=1e-13
        )


def test_unseen_history_backs_off():
    # WHY: a history never seen in training has no counts at its order;
    #      the model must fall through to the next lower order, not to the
    #      uniform distribution or to 0. With V = 4 (token 3 never seen),
    #      p(. | 3) is the KN unigram: (max(a - 1/2, 0)) / 6 + (1/4)(1/4)
    #      = (23/48, 7/48, 15/48, 3/48).
    # KIND: boundary
    # CATCHES: s05
    # CHAPTER: L2.1 section 2.4
    lm = fitted(HAND, 2, 0.5, 4)
    assert_close(
        [lm.prob([3], w) for w in range(4)],
        [23 / 48, 7 / 48, 15 / 48, 3 / 48],
        rtol=0.0,
        atol=1e-15,
    )


def test_context_length_rules():
    # WHY: a context shorter than n - 1 is a sequence start (<s> goes in
    #      front); a longer one is cut to its LAST n - 1 tokens. So for a
    #      trigram, [b] means "<s> b" while [a, b] and [c, c, a, b] both
    #      mean "a b".
    # KIND: unit
    # CATCHES: s06
    # CHAPTER: L2.1 section 2.2
    lm = fitted(HAND, 3, "modified", 3)
    case = golden()["hand_trigram"]["probs"]
    assert_close(
        [lm.prob([C, C, A, B], w) for w in range(3)],
        [float(Fraction(p)) for p in case["0 1"]],
        rtol=0.0,
        atol=1e-15,
    )
    assert_close(
        [lm.prob([B], w) for w in range(3)],
        [float(Fraction(p)) for p in case["1"]],
        rtol=0.0,
        atol=1e-15,
    )
    assert lm.prob([B], C) != lm.prob([A, B], C)


def test_discount_fallback():
    # WHY: Chen and Goodman's formulas need n1 .. n4 > 0 and give
    #      D_j in (0, j] only on enough data. A tiny corpus has some n_j = 0,
    #      and a unigram corpus with counts 1, 2, 3 and ten tokens seen 4
    #      times gives D3 = 3 - 4 (1/3) 10 < 0: both use (0.5, 1, 1.5).
    # KIND: boundary
    # CATCHES: s04, s11
    # CHAPTER: L2.1 section 2.5
    lm = fitted(HAND, 3, "modified", 3)
    assert all(lm.discounts(k) == FALLBACK_DISCOUNTS for k in (1, 2, 3))
    seq = [0] + [1] * 2 + [2] * 3 + [t for t in range(3, 13) for _ in range(4)]
    uni = fitted([seq], 1, "modified")
    assert uni.discounts(1) == FALLBACK_DISCOUNTS
    assert_close(
        math.fsum(uni.prob([], w) for w in range(13)), 1.0, rtol=0.0, atol=1e-14
    )


def test_nll_and_perplexity():
    # WHY: nll scores one sequence token by token, the first token from <s>;
    #      perplexity is exp(mean nll), the number MS-L2 and the corpus
    #      perplexity filter compare. In the worked example, "a b" costs
    #      -ln(2/3) - ln p(b | a).
    # KIND: unit
    # CATCHES: s01, s08
    # CHAPTER: L2.1 section 2.6
    lm = fitted(HAND, 2, 0.5, 3)
    nll = lm.nll([A, B])
    assert nll.shape == (2,) and nll.dtype == np.float64
    assert_close(
        nll, [-math.log(2 / 3), -math.log(lm.prob([A], B))], rtol=0.0, atol=1e-15
    )
    assert_close(
        lm.perplexity([A, B]), math.exp(float(nll.mean())), rtol=1e-15, atol=0.0
    )
    with pytest.raises(ValueError):
        lm.perplexity([])


def test_save_load_roundtrip(tmp_path):
    # WHY: the model is a file that the zoo (L6.7) and the capstone filter
    #      load later. Reloading must give the same probabilities and
    #      discounts, the header must say tl_arch "ngram" with <s> stored as
    #      -1, and saving the loaded model again must write the same bytes.
    # KIND: unit
    # CATCHES: s09, s13
    # CHAPTER: L2.1 section 4
    lm = fitted(corpus_lines()[:12], 3, 0.75)
    p1, p2 = tmp_path / "kn.safetensors", tmp_path / "again.safetensors"
    lm.save(str(p1))
    raw = p1.read_bytes()
    (hlen,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + hlen])
    meta = header["__metadata__"]
    assert meta == {
        "tl_arch": "ngram",
        "n": "3",
        "discount": "0.75",
        "vocab_size": str(lm.vocab_size),
    }
    assert (
        header["order2.grams"]["dtype"] == "I32"
        and header["order2.counts"]["dtype"] == "I32"
    )
    back = NGramLM.load(str(p1))
    assert back.n == 3 and back.vocab_size == lm.vocab_size
    for k in (1, 2, 3):
        assert back.discounts(k) == lm.discounts(k)
    for ctx in ([], list(b"th"), list(b"in t")):
        assert_close(back.logprobs(ctx), lm.logprobs(ctx), rtol=0.0, atol=0.0)
    back.save(str(p2))
    assert p2.read_bytes() == raw


def test_fit_replaces_previous_counts():
    # WHY: fit is not incremental: fitting again on new data must give the
    #      model of the new data alone, or a reused model object silently
    #      mixes corpora (the capstone fits one filter per source).
    # KIND: boundary
    # CATCHES: s12
    # CHAPTER: L2.1 section 4
    lm = NGramLM(2, 0.5, 3)
    lm.fit([[C, C, C, A]])
    lm.fit(HAND)
    ref = fitted(HAND, 2, 0.5, 3)
    for ctx in ([], [A], [B], [C]):
        assert_close(lm.logprobs(ctx), ref.logprobs(ctx), rtol=0.0, atol=0.0)


def test_input_validation():
    # WHY: ids outside 0 .. V-1 index past the vocabulary; a model queried
    #      before fit has no counts; n < 1 and discounts outside (0, 1]
    #      are config bugs. Without vocab_size, V is 1 + the largest id seen.
    # KIND: boundary
    # CATCHES: s13, m03
    # CHAPTER: L2.1 section 4
    for bad in (0, -1):
        with pytest.raises(ValueError):
            NGramLM(bad)
    for d in (0.0, 1.5, "kneser"):
        with pytest.raises(ValueError):
            NGramLM(2, d)
    with pytest.raises(ValueError):
        NGramLM(2).prob([], 0)
    with pytest.raises(ValueError):
        fitted([[0, 1, -2]], 2)
    with pytest.raises(ValueError):
        fitted([[0, 1, 3]], 2, "modified", 3)
    with pytest.raises(ValueError):
        fitted([[], []], 2)
    lm = fitted(HAND, 2, 0.5)
    assert lm.vocab_size == 3
    with pytest.raises(ValueError):
        lm.prob([A], 3)
    with pytest.raises(ValueError):
        lm.prob([5], A)
