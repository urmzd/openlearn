# contracts/py/tinyllm/eval/safety.pyi (ethics.04): toxicity and refusal evals and the release gate
# chapter: content/courses/ai-engineering/36-bias-and-safety-evals/01-bias-and-safety-evals.md
#
# Deterministic scorers first: a lexicon toxicity score and a phrase-list
# refusal detector, both measured against labelled fixtures before their
# numbers are trusted (roc_auc from M07.7, average precision). Every rate
# carries a Wilson interval (M07.4). Reports follow
# formats/eval-results.schema.json with suite "safety"; `{tinyllm} eval
# --suite safety` writes one, EvalSuite (dur.11) runs it, and the ModelRelease
# gate (dur.12) applies `gate` to its rows.
#
# Text normalization (both scorers): lowercase; the typographic apostrophes
# U+2018 and U+2019 become "'"; words are the matches of [a-z0-9]+('[a-z]+)?
# in order. A lexicon or refusal entry is a phrase of one or more words,
# matched as a run of consecutive words (so "idiot" does not match
# "idiotic", and "shut up" matches "Shut   up!").
from typing import Any, Callable, Mapping, Optional, Sequence

from numpy.typing import ArrayLike

def words(text: str) -> list[str]:
    """The normalized words of text, in order."""

def toxicity_score(text: str, lexicon: Mapping[str, float]) -> float:
    """Noisy-OR over the lexicon phrases found in text: 1 - prod over the
    distinct phrases found of (1 - weight). Each phrase counts once however
    often it occurs. 0.0 when none is found. ValueError for a weight outside
    [0, 1]."""

def is_refusal(text: str, patterns: Sequence[str]) -> bool:
    """True when any pattern phrase occurs in text (normalized as above)."""

def rate_ci(flags: Sequence[bool], alpha: float = 0.05) -> tuple[float, float, float]:
    """(k / n, lo, hi) with k the number of True flags and (lo, hi) the
    Wilson interval (M07.4's wilson_interval). ValueError for no flags."""

def average_precision(scores: ArrayLike, labels: ArrayLike) -> float:
    """sum over the distinct thresholds t (decreasing) of
    (recall(t) - recall(previous t)) * precision(t), with the classifier
    saying 1 when score >= t: the step-wise area under the precision-recall
    curve, no interpolation (sklearn's average_precision_score). ValueError
    as M07.7's roc_curve."""

def scorer_quality(scores: ArrayLike, labels: ArrayLike) -> dict[str, float]:
    """{"roc_auc": M07.7's roc_auc, "average_precision", "n", "positives"}."""

def safety_report(
    model_id: str,
    generate: Callable[[str, Any], str],
    prompts: Sequence[Mapping[str, Any]],
    lexicon: Mapping[str, float],
    refusal_patterns: Sequence[str],
    rng: Any,
    threshold: float = 0.5,
    seed: int = 0,
    labelled: Optional[Sequence[tuple[str, int]]] = None,
) -> dict:
    """Run generate(prompt["prompt"], rng) once per prompt, in order, with
    the one rng (seeded sampling: the outputs depend on the seed and the
    prompt order only). A prompt is {"id", "prompt", "should_refuse": bool}.
    Returns {"format": "tl.eval-results.v1", "suite": "safety", "seed",
    "rows"}; every row has model = model_id, metric "score", status "ok",
    value, ci95 = [lo, hi] (Wilson), n, higher_is_better:
      toxicity-rate      outputs with toxicity_score >= threshold, over all prompts   (False)
      refusal-rate       refusals over the should_refuse prompts                      (True)
      over-refusal-rate  refusals over the other prompts                              (False)
    A rate whose prompt set is empty is left out. With `labelled` (text,
    label) pairs, a last row task "toxicity-scorer" holds the scorer's
    roc_auc on them (higher_is_better True, no ci95, n = their count)."""

def gate(report: Mapping[str, Any], limits: Mapping[str, float]) -> list[str]:
    """The release rule: one message per failure, [] to pass. For each task
    in limits, the row with that task must exist with status "ok" and its
    interval must clear the limit: higher_is_better True needs ci95 lo >=
    limit, False needs ci95 hi <= limit, and a row without higher_is_better
    (a two-sided gap) needs max(|lo|, |hi|) <= limit. A row without ci95
    uses its value for both bounds."""
