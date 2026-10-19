"""My deepened L4.5 suite for craft.07: the rung R5 tests of L4.5, plus one
test per survivor I triaged as killable and per mutant I wrote myself
(triage.toml and mutants/ name them). It imports only the contract."""

import math

import pytest
from tinyllm.eval.seqmetrics import (
    bleu_from_stats,
    bleu_stats,
    chrf,
    chrf_stats,
    corpus_bleu,
    exact_match,
    metric_ci,
    tokenize_13a,
)


class Grid:
    """A deterministic stand-in uniform source: walks the golden-ratio grid."""

    def __init__(self) -> None:
        self.k = 0

    def uniform(self) -> float:
        self.k += 1
        return (self.k * 0.6180339887498949) % 1.0


def test_bleu_by_hand():
    """the cat sat on the mat / the cat is on the mat: 5/6, 3/5, 1/4, 0/3 smoothed to 1/6."""
    assert bleu_stats("the cat sat on the mat", ["the cat is on the mat"]) == [6, 6, 5, 3, 1, 0, 6, 5, 4, 3]
    assert corpus_bleu(["the cat sat on the mat"], [["the cat is on the mat"]]) == pytest.approx(100 * (1 / 48) ** 0.25, rel=1e-12)


def test_clipped_counts():
    """A repeated word counts at most as often as in the reference."""
    assert bleu_stats("the the the the", ["the cat"])[2] == 1
    assert bleu_stats("the the the the", ["the the cat", "the"])[2] == 2


def test_closest_reference_not_shortest():
    """7 tokens against references of 9 and 4: the 9 is closer."""
    assert bleu_stats("a b c d e f g", ["a b c d e f g h i", "a b c d"])[1] == 9


def test_brevity_penalty_only_when_short():
    short = bleu_from_stats([4, 6, 4, 3, 2, 1, 4, 3, 2, 1])
    assert short == pytest.approx(100 * math.exp(1 - 6 / 4), rel=1e-12)
    assert bleu_from_stats([8, 4, 8, 7, 6, 5, 8, 7, 6, 5]) == pytest.approx(100.0, rel=1e-12)


def test_geometric_mean_and_smoothing():
    """p = 1/2, 1/3, then one zero order: 1 / (2 * 2)."""
    stats = [4, 4, 2, 1, 0, 0, 4, 3, 2, 1]
    want = 100 * math.exp((math.log(2 / 4) + math.log(1 / 3) + math.log(1 / (2 * 2)) + math.log(1 / (4 * 1))) / 4)
    assert bleu_from_stats(stats) == pytest.approx(want, rel=1e-12)
    assert corpus_bleu(["x y z w"], [["a b c d"]]) == 0.0


def test_corpus_pools_counts():
    hyps = ["the cat sat on the mat", "a dog ran in the park"]
    refs = [["the cat is on the mat"], ["the dog ran in a park"]]
    pooled = [a + b for a, b in zip(bleu_stats(hyps[0], refs[0]), bleu_stats(hyps[1], refs[1]))]
    assert corpus_bleu(hyps, refs) == pytest.approx(bleu_from_stats(pooled), rel=1e-12)
    mean = (corpus_bleu(hyps[:1], refs[:1]) + corpus_bleu(hyps[1:], refs[1:])) / 2
    assert abs(corpus_bleu(hyps, refs) - mean) > 1


def test_references_per_hypothesis():
    hyps = ["a b c d", "e f g h"]
    refs = [["a b c d", "x y z w"], ["e f g h", "x y z w"]]
    assert corpus_bleu(hyps, refs) == pytest.approx(100.0, rel=1e-12)
    with pytest.raises(TypeError):
        corpus_bleu(["a b c d"], ["a b c d"])


def test_13a_keeps_numbers_whole():
    assert tokenize_13a("It costs 3.14, ok.") == "It costs 3.14 , ok ."
    assert tokenize_13a("1,000 items") == "1,000 items"


def test_case_sensitive():
    assert corpus_bleu(["A B C D"], [["a b c d"]]) == 0.0


def test_chrf_by_hand():
    """cat / cats: P = 1, R = (3/4 + 2/3 + 1/2) / 3, F2 = 5PR / (4P + R)."""
    r = (3 / 4 + 2 / 3 + 1 / 2) / 3
    assert chrf(["cat"], [["cats"]]) == pytest.approx(100 * 5 * r / (4 + r), rel=1e-12)


def test_chrf_orders_and_white_space():
    assert chrf_stats("abcdefg", ["abcdefg"])[15:18] == [2, 2, 2]  # order 6 is counted
    assert chrf(["ab cd"], [["abcd"]]) == pytest.approx(100.0, rel=1e-12)


def test_chrf_best_reference():
    one = chrf(["the quick fox"], [["the quick fox!"]])
    assert chrf(["the quick fox"], [["the quick fox!", "a slow dog"]]) == pytest.approx(one, rel=1e-12)


def test_exact_match():
    assert exact_match(["2021-05-03", " 2021-05-04\n", "2021-5-3"], ["2021-05-03", "2021-05-04", "2021-05-03"]) == pytest.approx(2 / 3)
    assert exact_match(["Paris"], ["paris"]) == 0.0
    with pytest.raises(ValueError):
        exact_match(["a"], [])


def test_metric_ci_pools_resamples():
    """Every pair identical: a degenerate interval at the top; resampled
    sentence BLEUs are never averaged."""
    hyps = ["a b c d", "e f g h i", "j k l m"]
    assert metric_ci("bleu", hyps, [[h] for h in hyps], 50, 0.1, Grid()) == pytest.approx((100.0, 100.0, 100.0))
    mixed_h = ["the cat sat on the mat", "a dog ran in the park"]
    mixed_r = [["the cat is on the mat"], ["the dog ran in a park"]]
    point, lo, hi = metric_ci("bleu", mixed_h, mixed_r, 200, 0.1, Grid())
    assert point == pytest.approx(corpus_bleu(mixed_h, mixed_r), rel=1e-12)
    assert lo <= point <= hi
    # three resamples are possible: sentence 1 twice (lowest: the smoothed
    # 4-gram precision 1 / (2 * 6) halves when the totals double), the mixed
    # pair (the point), and sentence 0 twice (highest)
    assert lo == pytest.approx(corpus_bleu(mixed_h[1:] * 2, mixed_r[1:] * 2), rel=1e-12)
    assert hi == pytest.approx(corpus_bleu(mixed_h[:1] * 2, mixed_r[:1] * 2), rel=1e-12)


# --- craft.07: the survivors my R5 suite missed --------------------------------------------


def test_closest_reference_tie_goes_to_the_shorter():
    """Survivor s02: 7 tokens, references of 9 and 5 tokens, both 2 away: 5."""
    assert bleu_stats("a b c d e f g", ["a b c d e f g h i", "a b c d e"])[1] == 5
    assert bleu_stats("a b c d e f g", ["a b c d e", "a b c d e f g h i"])[1] == 5


def test_13a_unescapes_entities():
    """Survivor s04: &amp; is one & token, &quot; a quote."""
    assert tokenize_13a("a&amp;b &quot;x&quot;") == 'a & b " x "'
    assert bleu_stats("a&amp;b c d", ["a & b c d"])[2] == 5


def test_chrf_best_reference_tie_keeps_the_first():
    """Survivor s06: "aa" and "acab" give "abca" the same sentence chrF with
    different statistics; the first one listed wins, and pooling shows it."""
    first = chrf_stats("abca", ["aa"])
    assert chrf_stats("abca", ["aa", "acab"]) == first
    assert chrf_stats("abca", ["acab", "aa"]) == chrf_stats("abca", ["acab"])
    assert chrf_stats("abca", ["aa"]) != chrf_stats("abca", ["acab"])


def test_13a_sees_the_line_after_rstrip():
    """Survivor s07: rstrip first, so a final "-\\n" keeps its dash."""
    assert bleu_stats("e-mail-\n", ["e-mail-"])[2:3] == [1]


def test_chrf_hypothesis_count_needs_reference_ngrams():
    """Survivor s09: an order the reference has no n-gram of counts 0
    hypothesis n-grams (sacreBLEU's rule), which shows when pooled."""
    stats = chrf_stats("abcdefgh", ["abc"])
    assert stats[9:12] == [0, 0, 0]  # order 4: "abc" has no 4-gram
    assert chrf(["abcdefgh", "abc"], [["abc"], ["abcdefgh"]]) == pytest.approx(
        chrf_from_stats_by_hand(), rel=1e-12
    )


def chrf_from_stats_by_hand() -> float:
    # sentence 1: orders 1..3 [8, 3, 3], [7, 2, 2], [6, 1, 1]; 4..6 zero (no ref n-grams)
    # sentence 2: orders 1..3 [3, 8, 3], [2, 7, 2], [1, 6, 1]; 4..6 [0, 5, 0], [0, 4, 0], [0, 3, 0]
    # pooled orders 1..3: hyp 11, 9, 7; ref 11, 9, 7; match 6, 4, 2; orders 4..6 have hyp 0
    p = (6 / 11 + 4 / 9 + 2 / 7) / 3
    return 100 * 5 * p * p / (4 * p + p)


def test_extra_references_are_an_error():
    """L4.5 m07: more reference entries than hypotheses is a mismatch too."""
    with pytest.raises(ValueError):
        corpus_bleu(["a b"], [["a b"], ["c d"]])


# --- craft.07: the mutants I wrote from L4.5's pitfalls ------------------------------------


def test_13a_splits_a_dash_after_a_digit():
    """My mutant p01: the fourth 13a rule deleted."""
    assert tokenize_13a("pages 10-12") == "pages 10 - 12"


def test_chrf_beta_is_a_parameter():
    """My mutant p02: beta fixed at 2. With beta = 1, F is the harmonic mean."""
    r = (3 / 4 + 2 / 3 + 1 / 2) / 3
    assert chrf(["cat"], [["cats"]], beta=1.0) == pytest.approx(100 * 2 * r / (1 + r), rel=1e-12)
