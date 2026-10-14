"""Course tests for L14.5: basic_normalize, edit_ops, wer_stats, wer, cer,
vqa_accuracy, caption_scores, zero_shot_accuracy, bootstrap_ci
(tinyllm/eval/mm.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/L14.5), and the chapter section it comes from.

The chapter's worked example (section 3): the reference "the cat sat on
the mat" against the hypothesis "the bat sat the mat too now" needs one
substitution (cat -> bat), one deletion (on), and two insertions (too,
now): S, D, I, N = 1, 1, 2, 6 and WER = 4 / 6. And "a b" against "b c"
costs 2 either as two substitutions or as a deletion plus an insertion; the
course rule (S before D before I in the backtrace) picks (2, 0, 0).

Golden values in $TINYLLM_FIXTURES/L14.5/metrics_oracle.json come from
course/oracle/L14.5/metrics_oracle.py: jiwer 4.0 (WER, CER, per-item edit
totals), transformers' BasicTextNormalizer, and sacreBLEU 2.6.
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
from tinyllm.eval.mm import (
    basic_normalize,
    bootstrap_ci,
    caption_scores,
    cer,
    edit_ops,
    vqa_accuracy,
    wer,
    wer_stats,
    zero_shot_accuracy,
)

GOLDEN = Path(os.environ.get("TINYLLM_FIXTURES", "")) / "L14.5" / "metrics_oracle.json"


def rng(seq: int) -> PCG32:
    return PCG32(int(os.environ.get("OL_SEED", "0")), seq=seq)


def golden() -> dict:
    return json.loads(GOLDEN.read_text())


def test_hand_example_alignment():
    # WHY: section 3 by hand: one substitution, one deletion, two
    #      insertions against six reference words. WER divides by the
    #      REFERENCE length, so it is 4 / 6, not 4 / 7.
    # KIND: unit, smoke
    # CATCHES: s02, s03, s12, s13
    # CHAPTER: L14.5 section 3
    ref = "the cat sat on the mat"
    hyp = "the bat sat the mat too now"
    assert edit_ops(ref.split(), hyp.split()) == (1, 1, 2, 6)
    assert_close(wer([ref], [hyp]), 4 / 6, rtol=0.0, atol=1e-15)
    assert wer_stats([ref], [hyp]).tolist() == [[1, 1, 2, 6]]


def test_tie_rule_prefers_substitution():
    # WHY: equal-cost alignments exist, and S, D, I counts (not the WER)
    #      depend on which one the backtrace picks. The course rule prefers
    #      the diagonal, then a deletion, then an insertion, so the Rust and
    #      Go ports (and per-group error breakdowns in ethics.07) agree.
    #      jiwer's backtrace prefers deletions: its counts can differ, its
    #      totals cannot.
    # KIND: unit
    # CATCHES: s01
    # CHAPTER: L14.5 section 2.1
    assert edit_ops(["a", "b"], ["b", "c"]) == (2, 0, 0, 2)
    # Shifting by one word costs a deletion and an insertion (2), never 3
    # substitutions: the minimum always comes first.
    assert edit_ops(["a", "b", "c"], ["b", "c", "d"]) == (0, 1, 1, 3)
    assert edit_ops(["a", "b", "c"], ["b", "c"]) == (0, 1, 0, 3)
    assert edit_ops(["x", "a"], ["a"]) == (0, 1, 0, 2)
    assert edit_ops(["a"], ["x", "a"]) == (0, 0, 1, 1)
    assert edit_ops([], ["a", "b"]) == (0, 0, 2, 0)
    assert edit_ops(["a", "b"], []) == (0, 2, 0, 2)


def test_wer_and_cer_match_jiwer():
    # WHY: jiwer is what papers report. Corpus WER and CER must equal its
    #      values, and every item's total edits and reference length must
    #      equal its counts (the split into S, D, I may differ on ties).
    # KIND: golden
    # CATCHES: s02, s06, s11, s12, s13
    # CHAPTER: L14.5 section 2.2
    a = golden()["asr"]
    assert_close(wer(a["refs"], a["hyps"]), a["wer"], rtol=0.0, atol=1e-12)
    assert_close(cer(a["refs"], a["hyps"]), a["cer"], rtol=0.0, atol=1e-12)
    st = wer_stats(a["refs"], a["hyps"])
    want = np.array(a["items_sdin"])
    assert st.dtype == np.int64 and st.shape == want.shape
    assert np.array_equal(st[:, :3].sum(axis=1), want[:, :3].sum(axis=1))
    assert np.array_equal(st[:, 3], want[:, 3])


def test_wer_invariants():
    # WHY: WER(x, x) = 0; an empty hypothesis deletes every reference word
    #      (WER 1); insertions can push WER above 1; swapping reference and
    #      hypothesis swaps D and I but keeps the total; the total is at
    #      least the length difference and at most the longer length.
    # KIND: property
    # CATCHES: s03, s12, s13
    # CHAPTER: L14.5 section 2.2
    r = rng(1)
    vocab = ["a", "b", "c", "d"]
    for _ in range(200):
        x = [vocab[r.below(4)] for _ in range(r.below(7))]
        y = [vocab[r.below(4)] for _ in range(r.below(7))]
        s, d, i, n = edit_ops(x, y)
        s2, d2, i2, n2 = edit_ops(y, x)
        assert (n, n2) == (len(x), len(y))
        assert s + d + i == s2 + d2 + i2
        assert abs(len(x) - len(y)) <= s + d + i <= max(len(x), len(y))
        assert d - i == len(x) - len(y)
        assert edit_ops(x, x) == (0, 0, 0, len(x))
    refs = ["one two three", "four five"]
    assert wer(refs, refs) == 0.0
    assert wer(refs, ["", ""]) == 1.0
    assert wer(["a"], ["b c d"]) == 3.0


def test_corpus_wer_pools_edits():
    # WHY: the corpus rate is total edits over total reference words, not
    #      the mean of per-item rates: a one-word item with one error would
    #      otherwise weigh as much as a long one.
    # KIND: unit
    # CATCHES: s02
    # CHAPTER: L14.5 section 2.2
    refs = ["a", "a b c d e f g h i j"]
    hyps = ["b", "a b c d e f g h i j"]
    assert_close(wer(refs, hyps), 1 / 11, rtol=0.0, atol=1e-15)
    assert_close(cer(["ab", "  cd  "], ["ab", "cd"]), 0.0, rtol=0.0, atol=0.0)
    assert_close(cer(["abcd"], ["abxd"]), 0.25, rtol=0.0, atol=1e-15)
    assert_close(cer(["a b"], ["ab"]), 1 / 3, rtol=0.0, atol=1e-15)


def test_rejects_bad_corpora():
    # WHY: an empty reference set has no rate, mismatched lists are a
    #      caller bug, and a bare string would be scored character by
    #      character as a corpus.
    # KIND: boundary
    # CATCHES: m01
    # CHAPTER: L14.5 section 4
    with pytest.raises(ValueError):
        wer([""], ["x"])
    with pytest.raises(ValueError):
        cer(["  "], ["x"])
    with pytest.raises(ValueError):
        wer(["a"], ["a", "b"])
    with pytest.raises(ValueError):
        wer([], [])
    with pytest.raises(TypeError):
        wer("a b", "a b")
    assert wer(["", "a"], ["", "a"]) == 0.0


def test_basic_normalize_matches_whisper():
    # WHY: WER is only comparable when both sides use the same normalizer;
    #      HF's BasicTextNormalizer is the one multilingual Whisper results
    #      use. Bracketed tags go, symbols and punctuation become spaces,
    #      NFKC folds ligatures and full-width forms, accents stay.
    # KIND: golden
    # CATCHES: s04, s05
    # CHAPTER: L14.5 section 2.3
    for raw, want in golden()["normalize"]:
        assert basic_normalize(raw) == want, raw


def test_vqa_accuracy():
    # WHY: VQA credits an answer by agreement with ten humans: three or
    #      more matching answers count fully, fewer count m / 3. Answers
    #      are compared after normalization ("Two." equals "two").
    # KIND: unit
    # CATCHES: s07, s08, m04
    # CHAPTER: L14.5 section 2.4
    ans = [["two"] * 4 + ["2"] * 6, ["red"] * 2 + ["blue"] * 8, ["yes"] * 10]
    assert_close(
        vqa_accuracy(["Two.", "red", "no"], ans),
        (1 + 2 / 3 + 0) / 3,
        rtol=0.0,
        atol=1e-15,
    )
    assert vqa_accuracy(["2"], [["2"] * 10]) == 1.0
    with pytest.raises(ValueError):
        vqa_accuracy(["a"], [[]])
    with pytest.raises(ValueError):
        vqa_accuracy(["a", "b"], [["a"]])
    with pytest.raises(TypeError):
        vqa_accuracy(["a"], ["a"])


def test_caption_scores_match_sacrebleu():
    # WHY: caption BLEU-4 and chrF are L4.5's corpus metrics with
    #      references per item; they must equal sacreBLEU 2.6 on the same
    #      captions. Attribute recall by hand: words every reference of an
    #      item shares, found in the prediction.
    # KIND: golden
    # CATCHES: s14
    # CHAPTER: L14.5 section 2.4
    c = golden()["captions"]
    got = caption_scores(c["preds"], c["refs"])
    assert_close(got["bleu4"], c["bleu4"], rtol=0.0, atol=1e-9)
    assert_close(got["chrf"], c["chrf"], rtol=0.0, atol=1e-9)
    # Words every reference shares: {red, circle, left}, {blue, squares},
    # {green, triangle, red, circle}, {purple, hexagon}, {small, yellow,
    # star}. Found in the predictions: 3/3, 2/2, 4/4, 0/2, 3/3.
    assert_close(got["attr_recall"], (1 + 1 + 1 + 0 + 1) / 5, rtol=0.0, atol=1e-15)


def test_attribute_recall():
    # WHY: with an attribute vocabulary, the attributes of an item are the
    #      vocabulary words any reference mentions; an item without any
    #      attribute is skipped, and a corpus without any gives nan.
    # KIND: unit
    # CATCHES: s15
    # CHAPTER: L14.5 section 2.4
    attrs = ["red", "blue", "circle", "square"]
    got = caption_scores(
        ["a red square", "a shape", "nothing"],
        [["a red circle", "a blue circle"], ["a square"], ["a thing"]],
        attributes=attrs,
    )
    assert_close(got["attr_recall"], (1 / 3 + 0.0) / 2, rtol=0.0, atol=1e-15)
    none = caption_scores(["x"], [["y z", "q"]])
    assert math.isnan(none["attr_recall"])
    with pytest.raises(TypeError):
        caption_scores(["x"], ["x"])


def test_zero_shot_topk_ties_to_lower_index():
    # WHY: zero-shot accuracy ranks class scores per item; equal scores rank
    #      by the lower class index so the result does not depend on sort
    #      implementation details, and k = classes is always 1.
    # KIND: unit
    # CATCHES: s09, m02, m03
    # CHAPTER: L14.5 section 2.4
    s = np.array([[0.1, 0.9, 0.9], [0.5, 0.2, 0.1], [0.3, 0.3, 0.3]])
    g = np.array([2, 0, 1])
    assert_close(zero_shot_accuracy(s, g), 1 / 3, rtol=0.0, atol=1e-15)
    assert_close(zero_shot_accuracy(s, g, k=2), 1.0, rtol=0.0, atol=0.0)
    assert zero_shot_accuracy(s, np.array([1, 0, 0])) == 1.0
    assert zero_shot_accuracy(s, g, k=3) == 1.0
    with pytest.raises(ValueError):
        zero_shot_accuracy(s, g, k=0)
    with pytest.raises(ValueError):
        zero_shot_accuracy(s, g, k=4)
    with pytest.raises(ValueError):
        zero_shot_accuracy(s, np.array([0, 1, 3]))
    with pytest.raises(ValueError):
        zero_shot_accuracy(s, np.array([0, 1]))


def test_bootstrap_ci_resamples_items():
    # WHY: a WER interval must resample ITEMS (a row of S, D, I, N stays
    #      together) and pool within each resample. The draws follow
    #      M07.4's rule with PCG32(seed), so the same seed gives the same
    #      interval in Python and Go (ag.12), and the point estimate is the
    #      statistic of the full set.
    # KIND: property
    # CATCHES: s10
    # CHAPTER: L14.5 section 2.5
    a = golden()["asr"]
    st = wer_stats(a["refs"], a["hyps"])

    def pooled(x):
        return x[:, :3].sum() / x[:, 3].sum()

    point, lo, hi = bootstrap_ci(st, pooled, n=200, seed=7)
    assert_close(point, wer(a["refs"], a["hyps"]), rtol=0.0, atol=1e-15)
    assert lo <= point <= hi
    # By hand with the frozen PCG32: n * len(st) uniforms in order.
    g = PCG32(7)
    n = len(st)
    thetas = []
    for _ in range(200):
        ix = [min(int(g.uniform() * n), n - 1) for _ in range(n)]
        thetas.append(pooled(st[ix]))
    assert_close(lo, float(np.quantile(thetas, 0.025)), rtol=0.0, atol=1e-12)
    assert_close(hi, float(np.quantile(thetas, 0.975)), rtol=0.0, atol=1e-12)
    assert bootstrap_ci(st, pooled, n=200, seed=7) == (point, lo, hi)
    assert bootstrap_ci(st, pooled, n=200, seed=8) != (point, lo, hi)
