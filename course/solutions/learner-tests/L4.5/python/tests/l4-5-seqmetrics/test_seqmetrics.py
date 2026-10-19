"""My tests for L4.5 (rung R5: oracles computed by hand, and the laws the
metrics obey). They import only the contract."""

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
