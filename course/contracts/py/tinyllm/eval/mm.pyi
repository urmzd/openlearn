# contracts/py/tinyllm/eval/mm.pyi (L14.5): multimodal eval metrics
# chapter: content/courses/build-an-llm/15-speech/05-multimodal-eval-metrics.md
#
# Speech: word and character error rates from a Levenshtein alignment,
# jiwer 4 semantics. For one pair, the alignment of the reference tokens r
# (length N) against the hypothesis tokens h has the minimum number of
# substitutions S, deletions D (a reference token missing from h), and
# insertions I (an extra token in h), each costing 1. Among alignments of
# equal cost, the backtrace from the end prefers a substitution or match,
# then a deletion, then an insertion (S before D before I). WER pools over
# the corpus: sum(S + D + I) / sum(N). Words are str.split() tokens;
# characters are the characters of str.strip(). The caller normalizes
# first (basic_normalize, or the model's own normalizer).
#
# Vision: VQA accuracy, caption BLEU-4 and chrF (L4.5, sacreBLEU 2
# semantics), attribute recall, zero-shot top-k accuracy. Intervals:
# percentile bootstrap over items with M07.4's bootstrap_ci and a PCG32
# (M06.3) seeded by `seed`.
from typing import Callable, Sequence

from numpy.typing import NDArray

def basic_normalize(text: str) -> str:
    """Whisper's BasicTextNormalizer (spec/asr-decoding.md, normalizer):
    lowercase; drop text in <...> or [...] and in (...); NFKC, then every
    character whose Unicode category starts with M, S, or P becomes a space;
    lowercase again; every run of white space becomes one space. Leading and
    trailing spaces are kept (they do not change a word split)."""

def edit_ops(ref: Sequence[str], hyp: Sequence[str]) -> tuple[int, int, int, int]:
    """(S, D, I, N) of the minimum-cost alignment with the tie rule above;
    N = len(ref). Tokens compare with ==. Either side may be empty."""

def wer_stats(refs: Sequence[str], hyps: Sequence[str]) -> NDArray:
    """int64 [n_items, 4]: edit_ops of the word lists of each pair, one row
    (S, D, I, N) per item. ValueError for different lengths or no items."""

def wer(refs: Sequence[str], hyps: Sequence[str]) -> float:
    """sum(S + D + I) / sum(N) over wer_stats. 0 for identical texts, 1 when
    every hypothesis is empty, and above 1 with many insertions. ValueError
    when every reference is empty (sum N = 0), and as wer_stats."""

def cer(refs: Sequence[str], hyps: Sequence[str]) -> float:
    """wer over characters: edit_ops of list(ref.strip()) and
    list(hyp.strip()), pooled the same way. ValueError as wer."""

def vqa_accuracy(preds: Sequence[str], answers: Sequence[Sequence[str]]) -> float:
    """Mean over items of min(1, m / 3), m = how many of the item's human
    answers equal the prediction after basic_normalize and strip on both
    sides. ValueError for different lengths, no items, or an item with no
    answers; TypeError when answers[i] is a str."""

def caption_scores(preds: Sequence[str], refs: Sequence[Sequence[str]], attributes: Sequence[str] = ()) -> dict[str, float]:
    """{"bleu4": L4.5's corpus_bleu, "chrf": L4.5's chrf (both 0..100),
    "attr_recall": 0..1}. Attribute recall: per item, the attribute words
    (normalized words of the references that are in `attributes`, or, when
    attributes is empty, the words present in every reference of the item);
    the recall is the fraction present among the normalized words of the
    prediction; the result is the mean over items with at least one attribute
    word (nan when no item has one). ValueError for different lengths or no
    items; TypeError when refs[i] is a str."""

def zero_shot_accuracy(pred: NDArray, gold: NDArray, k: int = 1) -> float:
    """Top-k accuracy of scores pred [n, classes] (similarities or logits)
    against int gold [n]: an item counts when gold is among the k highest
    scores, ranking ties by the lower class index. ValueError for shapes that
    do not match, k outside [1, classes], or a gold label outside
    [0, classes)."""

def bootstrap_ci(per_item: NDArray, stat: Callable[[NDArray], float], n: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """(stat(per_item), lo, hi): M07.4's bootstrap_ci over the item indices
    0 .. len(per_item) - 1 with n resamples and rng PCG32(seed), where a
    resample of indices ix is scored as stat(per_item[ix]) (rows of a 2-D
    per_item stay together, so the WER of wer_stats rows is resampled by
    item). ValueError as M07.4's bootstrap_ci."""
