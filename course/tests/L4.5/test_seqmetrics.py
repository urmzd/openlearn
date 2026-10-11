"""Course tests for L4.5: sequence metrics (tinyllm/eval/seqmetrics.py).

Rung R0 reading for this file; your own tests are rung R5 (the chapter's
section 4). Each test names why it exists (WHY), what kind of check it is
(KIND), the planted bugs it kills (CATCHES, mutants in course/mutants/L4.5),
and the chapter section it comes from.

The chapter's worked examples (section 3):
  BLEU  hyp "the cat sat on the mat", ref "the cat is on the mat":
        clipped matches 5/6, 3/5, 1/4, 0/3; the 4-gram precision is smoothed
        to 1 / (2 * 3); BP = 1; BLEU = 100 * (1/48) ** (1/4) = 37.991784.
  chrF  hyp "cat", ref "cats": orders 1..3 have P = 1 and R = 3/4, 2/3, 1/2;
        orders 4..6 have no hypothesis n-gram; P = 1, R = 23/36,
        chrF = 100 * 5 P R / (4 P + R) = 100 * 115/167 = 68.862275.
  EM    "2021-05-03" vs itself, " 2021-05-04\\n" vs "2021-05-04" (strip),
        "2021-5-3" vs "2021-05-03": 2 of 3.

Golden values come from sacreBLEU 2.5.1 (course/fixtures/L4.5/sacrebleu_golden.json,
written by course/oracle/L4.5/sacrebleu_golden.py). Intervals use the frozen
PCG32.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32
from tinyllm.eval.seqmetrics import (
    bleu_from_stats,
    bleu_stats,
    chrf,
    chrf_from_stats,
    chrf_stats,
    corpus_bleu,
    exact_match,
    metric_ci,
    sentence_stats,
    tokenize_13a,
)

FIX = (
    Path(os.environ.get("TINYLLM_FIXTURES", "course/fixtures"))
    / "L4.5"
    / "sacrebleu_golden.json"
)
HYP = "the cat sat on the mat"
REF = "the cat is on the mat"


def seed() -> int:
    return int(os.environ.get("OL_SEED", "0"))


def golden() -> dict:
    return json.loads(FIX.read_text())


def case(section: str, name: str) -> dict:
    return next(c for c in golden()[section] if c["name"] == name)


# --- the worked examples ------------------------------------------------------------------


def test_hand_example_bleu():
    # WHY: the chapter's worked example, count for count: clipped n-gram
    #      matches 5/6, 3/5, 1/4, 0/3, the zero 4-gram precision smoothed to
    #      1 / (2 * 3), no brevity penalty, and the geometric mean of the
    #      four precisions: 100 * (1/48) ** (1/4).
    # KIND: unit
    # CATCHES: s07, s08, m01, m03
    # CHAPTER: L4.5 section 3, Worked example by hand
    assert bleu_stats(HYP, [REF]) == [6, 6, 5, 3, 1, 0, 6, 5, 4, 3]
    assert_close(corpus_bleu([HYP], [[REF]]), 100 * (1 / 48) ** 0.25, dtype="float64")


def test_hand_example_chrf():
    # WHY: chrF of "cat" against "cats": orders 1..3 have precision 1 and
    #      recall 3/4, 2/3, 1/2; orders 4..6 drop out (no hypothesis n-gram,
    #      and the reference has none of orders 5 and 6); the averages give
    #      F_2 = 5 P R / (4 P + R) = 115/167.
    # KIND: unit
    # CATCHES: s12, s13, m05
    # CHAPTER: L4.5 section 3, Worked example by hand
    assert chrf_stats("cat", ["cats"]) == [
        3,
        4,
        3,
        2,
        3,
        2,
        1,
        2,
        1,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    assert_close(chrf(["cat"], [["cats"]]), 100 * 115 / 167, dtype="float64")


def test_hand_example_exact_match():
    # WHY: exact match strips white space at both ends (a decoder's trailing
    #      newline is not an error) and nothing else: "2021-5-3" is not
    #      "2021-05-03". The result is a fraction in [0, 1], not a count.
    # KIND: unit
    # CATCHES: s17, m06
    # CHAPTER: L4.5 section 3, Worked example by hand
    hyps = ["2021-05-03", " 2021-05-04\n", "2021-5-3"]
    refs = ["2021-05-03", "2021-05-04", "2021-05-03"]
    assert_close(exact_match(hyps, refs), 2 / 3, dtype="float64")


# --- the rules, one at a time ----------------------------------------------------------------


def test_clipping():
    # WHY: a hypothesis that repeats a correct word earns it only as often as
    #      the reference has it: "the the the the" against "the cat" matches
    #      1 unigram of 4, not 4 of 4 (the reason BLEU clips at all).
    # KIND: boundary
    # CATCHES: s01, s03
    # CHAPTER: L4.5 section 5, Pitfalls, item 1
    s = bleu_stats("the the the the", ["the cat"])
    assert s[2] == 1 and s[6] == 4
    assert_close(
        corpus_bleu(["the the the the"], [["the cat"]]),
        100 * (1 / 1536) ** 0.25,
        dtype="float64",
    )


def test_closest_reference_length():
    # WHY: the brevity penalty compares the hypothesis with the reference
    #      CLOSEST in length (ties go to the shorter), not the shortest and
    #      not the longest: a 7-token hypothesis with references of 12 and 8
    #      tokens is measured against 8; with 9 and 5 (both 2 away), against
    #      5; with 9 and 4, against 9, the longer one, because it is closer.
    # KIND: boundary
    # CATCHES: s02
    # CHAPTER: L4.5 section 2.2, BLEU
    assert (
        bleu_stats("a b c d e f g", ["a b c d e f g h i j k l", "a b c d e f x y"])[1]
        == 8
    )
    assert bleu_stats("a b c d e f g", ["a b c d e f g h i", "a b c d e"])[1] == 5
    assert bleu_stats("a b c d e f g", ["a b c d e f g h i", "a b c d"])[1] == 9


def test_brevity_penalty():
    # WHY: a short hypothesis is penalized by exactly exp(1 - r / c); a long
    #      one is not penalized at all (its extra words already lowered the
    #      precisions). BLEU without the penalty rewards dropping hard words.
    # KIND: unit
    # CATCHES: s03
    # CHAPTER: L4.5 section 2.2, BLEU
    short = [4, 6, 4, 3, 2, 1, 4, 3, 2, 1]
    assert_close(bleu_from_stats(short), 100 * math.exp(1 - 6 / 4), dtype="float64")
    long = [8, 4, 4, 3, 2, 1, 8, 7, 6, 5]
    want = 100 * math.exp(
        (math.log(4 / 8) + math.log(3 / 7) + math.log(2 / 6) + math.log(1 / 5)) / 4
    )
    assert_close(bleu_from_stats(long), want, dtype="float64")


def test_smoothing_and_zero_orders():
    # WHY: sacreBLEU's default "exp" smoothing gives the j-th order with no
    #      match the precision 1 / (2^j total_n), so one missing 4-gram does
    #      not zero the score; but no match at all is 0, and a corpus with no
    #      4-gram at all (every hypothesis under 4 tokens) is 0.
    # KIND: boundary
    # CATCHES: s08, m03
    # CHAPTER: L4.5 section 2.2, BLEU
    stats = [5, 5, 3, 0, 0, 0, 5, 4, 3, 2]
    want = 100 * math.exp(
        (
            math.log(3 / 5)
            + math.log(1 / (2 * 4))
            + math.log(1 / (4 * 3))
            + math.log(1 / (8 * 2))
        )
        / 4
    )
    assert_close(bleu_from_stats(stats), want, dtype="float64")
    assert corpus_bleu(["xyz qqq"], [["the cat sat"]]) == 0.0
    assert corpus_bleu(["the cat", "a dog ran"], [["the cat"], ["a dog ran"]]) == 0.0


def test_corpus_pools_statistics():
    # WHY: corpus BLEU is computed from the SUM of sentence statistics, not
    #      the mean of sentence scores: the two-sentence corpus scores
    #      31.259718, while the mean of its sentence BLEUs is 36.961. Pooling is
    #      also what makes the bootstrap over sentences cheap.
    # KIND: property
    # CATCHES: s04
    # CHAPTER: L4.5 section 2.4, Corpus statistics
    c = case("bleu", "two-sentences")
    got = corpus_bleu(c["hyps"], c["refs"])
    assert_close(got, c["score"], rtol=1e-12, atol=1e-12)
    summed = [
        sum(col)
        for col in zip(*(bleu_stats(h, r) for h, r in zip(c["hyps"], c["refs"])))
    ]
    assert_close(got, bleu_from_stats(summed), rtol=1e-12, atol=1e-12)
    mean = sum(corpus_bleu([h], [r]) for h, r in zip(c["hyps"], c["refs"])) / 2
    assert abs(got - mean) > 4


def test_references_are_per_hypothesis():
    # WHY: refs[i] lists the references of hyps[i]. sacreBLEU's own API takes
    #      reference streams (the transpose); passing one layout to the other
    #      silently scores the wrong pairs. A bare string in refs[i] is an
    #      error, never a list of one-character references.
    # KIND: boundary
    # CATCHES: s05, s06
    # CHAPTER: L4.5 section 5, Pitfalls, item 4
    hyps = ["the cat sat on the mat", "a dog ran in the park"]
    per_hyp = [
        ["the cat is on the mat", "a cat sat on a mat"],
        ["the dog ran in a park", "a dog ran in the yard"],
    ]
    want = corpus_bleu([hyps[0]], [per_hyp[0]]), corpus_bleu([hyps[1]], [per_hyp[1]])
    both = corpus_bleu(hyps, per_hyp)
    assert min(want) - 1e-9 <= both <= max(want) + 1e-9
    v = case("bleu", "variable-refs")
    assert_close(corpus_bleu(v["hyps"], v["refs"]), v["score"], rtol=1e-12, atol=1e-12)
    with pytest.raises(TypeError):
        corpus_bleu(["the cat"], ["the cat"])
    with pytest.raises(TypeError):
        chrf(["the cat"], ["the cat"])


def test_tokenize_13a_matches_sacrebleu():
    # WHY: BLEU counts tokens, so the tokenizer is part of the metric: 13a
    #      splits punctuation but keeps 3.14 and 1,000 whole, splits a dash
    #      only after a digit, and unescapes &amp; and friends. A different
    #      tokenizer gives a different, incomparable number.
    # KIND: golden
    # CATCHES: s09
    # CHAPTER: L4.5 section 2.1, Tokens
    for line, want in golden()["tokenize_13a"]:
        assert tokenize_13a(line) == want, f"13a({line!r})"


def test_bleu_matches_sacrebleu():
    # WHY: every corpus of the fixture (the worked examples, clipping across
    #      several references, entities and dashes, case, trailing spaces,
    #      variable reference counts, both tokenizers, six random corpora):
    #      your per-sentence statistics and corpus score equal sacreBLEU's.
    # KIND: golden
    # CATCHES: s01, s02, s03, s04, s06, s07, s08, s09, s10, m01, m03, m04, m08
    # CHAPTER: L4.5 section 4, The interface
    for c in golden()["bleu"]:
        got = [bleu_stats(h, r, c["tokenize"]) for h, r in zip(c["hyps"], c["refs"])]
        assert got == c["stats"], c["name"]
        assert_close(
            corpus_bleu(c["hyps"], c["refs"], c["tokenize"]),
            c["score"],
            rtol=1e-12,
            atol=1e-12,
            msg=c["name"],
        )


def test_case_counts():
    # WHY: sacreBLEU's default is case-sensitive: "The Cat" is not "the cat".
    #      Lowercasing silently inflates scores and makes them incomparable.
    # KIND: unit
    # CATCHES: s10
    # CHAPTER: L4.5 section 5, Pitfalls, item 6
    assert corpus_bleu(["The Cat Sat On The Mat"], [["the cat sat on the mat"]]) == 0.0
    assert_close(
        corpus_bleu(["the cat sat on the mat"], [["the cat sat on the mat"]]),
        100.0,
        dtype="float64",
    )


def test_chrf_matches_sacrebleu():
    # WHY: chrF over every fixture corpus: white space ignored, a reference
    #      shorter than the order (its hypothesis count is 0 there), several
    #      references (the best one by sentence chrF), unicode, random corpora.
    # KIND: golden
    # CATCHES: s11, s12, s13, s14, s15, m05
    # CHAPTER: L4.5 section 4, The interface
    for c in golden()["chrf"]:
        got = [chrf_stats(h, r) for h, r in zip(c["hyps"], c["refs"])]
        assert got == c["stats"], c["name"]
        assert_close(
            chrf(c["hyps"], c["refs"]),
            c["score"],
            rtol=1e-12,
            atol=1e-12,
            msg=c["name"],
        )


def test_chrf_ignores_white_space_and_forgives_inflection():
    # WHY: chrF removes white space before counting characters, so "thecat
    #      sat" scores 100 against "the cat sat"; and "cats" shares most of
    #      its character n-grams with "cat", where word BLEU sees a miss.
    # KIND: unit
    # CATCHES: s11
    # CHAPTER: L4.5 section 2.3, chrF
    assert_close(chrf(["thecat  sat"], [["the cat sat"]]), 100.0, dtype="float64")
    assert chrf(["the cats sitting"], [["the cat sat"]]) > 40
    assert corpus_bleu(["the cats sitting"], [["the cat sat"]]) < 20


def test_chrf_beta_weights_recall():
    # WHY: beta = 2 makes recall count four times as much as precision
    #      (F_beta uses beta squared): a hypothesis missing half the reference
    #      is punished harder than one with extra characters.
    # KIND: unit
    # CATCHES: s12
    # CHAPTER: L4.5 section 2.3, chrF
    stats = [4, 8, 4] + [0, 0, 0] * 5  # P = 1, R = 1/2 at order 1 only
    assert_close(chrf_from_stats(stats), 100 * 5 * 0.5 / (4 + 0.5), dtype="float64")
    assert_close(chrf_from_stats(stats, beta=1.0), 100 * 2 * 0.5 / 1.5, dtype="float64")


def test_chrf_uses_the_best_reference():
    # WHY: with several references a sentence keeps the statistics of the
    #      reference with the best sentence chrF, not an average and not the
    #      last one listed.
    # KIND: unit
    # CATCHES: s14
    # CHAPTER: L4.5 section 2.3, chrF
    assert_close(
        chrf(["the quick fox"], [["a slow dog", "the quick fox!"]]),
        chrf(["the quick fox"], [["the quick fox!"]]),
        dtype="float64",
    )
    assert_close(
        chrf(["the quick fox"], [["the quick fox!", "a slow dog"]]),
        chrf(["the quick fox"], [["the quick fox!"]]),
        dtype="float64",
    )


def test_exact_match_is_strict_inside():
    # WHY: only the ends are stripped: case and inner white space count. A
    #      "normalized" exact match (lowercase, collapse spaces) is a
    #      different metric and must say so.
    # KIND: boundary
    # CATCHES: s16, s17
    # CHAPTER: L4.5 section 2.5, Exact match
    assert exact_match(["Paris"], ["paris"]) == 0.0
    assert exact_match(["new  york"], ["new york"]) == 0.0
    assert exact_match(["\tParis \n"], ["Paris"]) == 1.0


def test_metric_ci_matches_independent_bootstrap():
    # WHY: the interval is M07.4's percentile bootstrap over SENTENCES: each
    #      resample keeps a hypothesis with its own references and recomputes
    #      the corpus metric from pooled statistics. With the same PCG32 seed
    #      it equals an independent run that rescored each resample with
    #      sacreBLEU, for exact match, BLEU, and chrF.
    # KIND: golden
    # CATCHES: s18
    # CHAPTER: L4.5 section 4, The interface
    for c in golden()["ci"]:
        got = metric_ci(
            c["metric"],
            c["hyps"],
            c["refs"],
            c["n_boot"],
            c["alpha"],
            PCG32(seed=c["seed"]),
        )
        assert_close(
            got,
            c["want"],
            rtol=1e-10,
            atol=1e-10,
            msg=f"{c['metric']} seed={c['seed']}",
        )


def test_metric_ci_brackets_the_score():
    # WHY: for any seed the point is the corpus score, the interval lies in
    #      the metric's range with lo <= hi, the same seed gives the same
    #      interval (a reported CI is reproducible), and a corpus where every
    #      hypothesis equals its reference has the degenerate interval
    #      [100, 100]: resampling whole sentences never breaks a pair apart.
    # KIND: property
    # CATCHES: s18
    # CHAPTER: L4.5 section 2.4, Corpus statistics
    c = case("bleu", "random-0")
    a = metric_ci("bleu", c["hyps"], c["refs"], 100, 0.1, PCG32(seed=seed()))
    b = metric_ci("bleu", c["hyps"], c["refs"], 100, 0.1, PCG32(seed=seed()))
    assert a == b
    point, lo, hi = a
    assert 0.0 <= lo <= hi <= 100.0
    assert_close(point, c["score"], rtol=1e-12, atol=1e-12)
    rows = sentence_stats("bleu", c["hyps"], c["refs"])
    assert rows.shape == (len(c["hyps"]), 10)
    same = [r[0] for r in c["refs"]]
    for metric, refs in (
        ("bleu", c["refs"]),
        ("chrf", c["refs"]),
        ("exact_match", same),
    ):
        hyps = [r[0] for r in c["refs"]]
        got = metric_ci(metric, hyps, refs, 50, 0.05, PCG32(seed=seed()))
        top = 1.0 if metric == "exact_match" else 100.0
        assert_close(got, [top, top, top], dtype="float64", msg=metric)


def test_validation():
    # WHY: mismatched lengths, an empty corpus, an unknown tokenizer or
    #      metric, an empty reference list, n < 1, and a missing rng are
    #      errors, not a score of 0: a silent 0 hides a broken pipeline.
    # KIND: boundary
    # CATCHES: m07
    # CHAPTER: L4.5 section 4, The interface
    with pytest.raises(ValueError):
        exact_match(["a", "b"], ["a"])
    with pytest.raises(ValueError):
        exact_match(["a"], ["a", "b"])
    with pytest.raises(ValueError):
        corpus_bleu([], [])
    with pytest.raises(ValueError):
        corpus_bleu(["a"], [["a"]], tokenize="intl")
    with pytest.raises(ValueError):
        bleu_stats("a", [])
    with pytest.raises(ValueError):
        chrf(["a"], [["a"]], n=0)
    with pytest.raises(ValueError):
        chrf(["a", "b"], [["a"]])
    with pytest.raises(ValueError):
        bleu_from_stats([1, 2, 3])
    with pytest.raises(ValueError):
        metric_ci("rouge", ["a"], [["a"]], 10, 0.05, PCG32(seed=0))
    with pytest.raises(ValueError):
        metric_ci("bleu", ["a"], [["a"]], 10, 0.05, None)
