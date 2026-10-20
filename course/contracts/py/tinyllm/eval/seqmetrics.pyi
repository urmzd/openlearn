# contracts/py/tinyllm/eval/seqmetrics.pyi (L4.5): exact match, BLEU, chrF, and their bootstrap intervals
# chapter: content/courses/ai-engineering/10-attention/05-sequence-metrics.md
#
# Metrics that grade generated sequences against references. The numbers
# match sacreBLEU 2.x with its defaults (Post 2018), so a score you report is
# comparable with every paper that reports sacreBLEU:
#
#   BLEU (Papineni et al. 2002), corpus level, 13a tokenization, no
#   lowercasing, n-gram orders 1..4, "exp" smoothing, brevity penalty from
#   the closest reference length. Reported on a 0..100 scale.
#   chrF (Popovic 2015), character n-grams of orders 1..n with white space
#   removed, beta = 2, corpus statistics pooled over sentences, precision and
#   recall averaged over the orders where both sides have n-grams, then the
#   F-beta of the two averages. Reported on a 0..100 scale.
#   exact match: the fraction (0..1) of hypotheses equal to their reference
#   after str.strip() on both; case and inner white space count.
#
# References are PER HYPOTHESIS: refs[i] is the list of references of
# hyps[i] (one or more; sacreBLEU's own API takes reference STREAMS, the
# transpose). A bare string in refs[i] is a TypeError, never a sequence of
# characters.
#
# Sufficient statistics (sacreBLEU's order) make corpus scores additive, so
# a corpus score is computed from the SUM of per-sentence statistics, never
# the mean of sentence scores:
#   BLEU: [hyp_len, ref_len, correct_1..correct_4, total_1..total_4]
#         ref_len is the length of the reference closest to hyp_len (ties to
#         the shorter); correct_n counts hypothesis n-grams clipped by the
#         largest count of that n-gram in any one reference.
#   chrF: per order k = 1..n the triple [hyp_count, ref_count, match]; with
#         several references the triple list of the reference with the best
#         sentence chrF (ties to the first); hyp_count is 0 for an order the
#         reference has no n-gram of (sacreBLEU's rule).
#
# Intervals: metric_ci resamples SENTENCES (hypothesis and its references
# together) with M07.4's bootstrap_ci and recomputes the corpus metric from
# the resampled statistics.
from typing import Any, Sequence

from numpy.typing import NDArray

TOKENIZERS: tuple[str, ...]  # ("13a", "none")

def tokenize_13a(line: str) -> str:
    """sacreBLEU's 13a tokenizer (mteval-v13a): drop "<skipped>", join "-\\n",
    newlines to spaces, unescape &quot; &amp; &lt; &gt;, then split off
    punctuation and symbols, periods and commas except between digits, and a
    dash after a digit; tokens joined by single spaces, no leading or
    trailing space."""

def exact_match(hyps: Sequence[str], refs: Sequence[str]) -> float:
    """Fraction of i with hyps[i].strip() == refs[i].strip(), in [0, 1].
    ValueError for different lengths or no hypotheses."""

def bleu_stats(hyp: str, refs: Sequence[str], tokenize: str = "13a") -> list[int]:
    """The 10 BLEU statistics of one hypothesis against its references, after
    rstrip and the tokenizer ("13a" or "none": white-space split only).
    ValueError for an unknown tokenizer or no references; TypeError when refs
    is a str."""

def bleu_from_stats(stats: Sequence[float]) -> float:
    """BLEU (0..100) from summed statistics: 0.0 when no n-gram matched;
    otherwise bp = 1 if hyp_len >= ref_len else exp(1 - ref_len / hyp_len)
    (0 when hyp_len = 0), p_n = 100 correct_n / total_n, an order with
    correct_n = 0 gets 100 / (2^j total_n) for its j-th such order ("exp"
    smoothing), the first order with total_n = 0 and every later order count
    as log 0 = -9999999999, and BLEU = bp exp(mean of the four log p_n).
    ValueError unless len(stats) == 10."""

def corpus_bleu(hyps: Sequence[str], refs: Sequence[Sequence[str]], tokenize: str = "13a") -> float:
    """bleu_from_stats of the sum of bleu_stats over the corpus. ValueError for
    different lengths or an empty corpus."""

def chrf_stats(hyp: str, refs: Sequence[str], n: int = 6, beta: float = 2.0) -> list[int]:
    """The 3 n chrF statistics of one hypothesis (the best reference's, as
    above). ValueError for n < 1, beta <= 0, or no references; TypeError when
    refs is a str."""

def chrf_from_stats(stats: Sequence[float], n: int = 6, beta: float = 2.0) -> float:
    """chrF (0..100) from summed statistics: average P = match / hyp_count and
    R = match / ref_count over the orders with hyp_count > 0 and ref_count >
    0 (0 when there is none), then 100 (1 + beta^2) P R / (beta^2 P + R), or
    0.0 when P + R = 0. ValueError unless len(stats) == 3 n."""

def chrf(hyps: Sequence[str], refs: Sequence[Sequence[str]], n: int = 6, beta: float = 2.0) -> float:
    """chrf_from_stats of the sum of chrf_stats over the corpus. ValueError for
    different lengths or an empty corpus."""

def sentence_stats(metric: str, hyps: Sequence[str], refs: Sequence[Any]) -> NDArray:
    """float64 [N, S]: one row of statistics per sentence. metric "exact_match"
    (S = 1: 1.0 or 0.0, refs[i] a str), "bleu" (S = 10, 13a), or "chrf"
    (S = 18, n = 6, beta = 2). ValueError for another metric or different
    lengths."""

def score_from_stats(metric: str, summed: NDArray, n_sentences: int) -> float:
    """The corpus metric from the column sums of sentence_stats: exact match is
    summed[0] / n_sentences, BLEU bleu_from_stats, chrF chrf_from_stats."""

def metric_ci(
    metric: str,
    hyps: Sequence[str],
    refs: Sequence[Any],
    n_boot: int = 1000,
    alpha: float = 0.05,
    rng: Any = None,
) -> tuple[float, float, float]:
    """(score, lo, hi): M07.4's bootstrap_ci over the sentence indices
    0 .. N-1 (as float64), with the statistic "column sums of the resampled
    rows of sentence_stats, then score_from_stats". The rng (a UniformSource,
    M07.4) is required: ValueError when it is None, and as bootstrap_ci and
    sentence_stats otherwise."""
