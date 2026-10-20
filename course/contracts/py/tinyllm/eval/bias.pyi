# contracts/py/tinyllm/eval/bias.pyi (ethics.04): templated bias probes with intervals
# chapter: content/courses/ai-engineering/36-bias-and-safety-evals/01-bias-and-safety-evals.md
#
# Bias is measured as a difference: the same sentence with only a group term
# changed, scored the same way. `score_fn(text) -> float` is any score where
# higher means "the model prefers this text", for a language model its total
# log-probability (L6.7's log_probs summed over the tokens). Pairs are
# (a, b) term pairs on one axis, for example ("he", "she") on "gender".
from typing import Any, Callable, Mapping, Sequence

from numpy.typing import NDArray

def expand_pairs(
    templates: Sequence[str], pairs: Sequence[tuple[str, str]], slot: str = "{group}"
) -> list[tuple[str, str]]:
    """(template with a, template with b) for every template and every pair,
    template-major (all pairs of template 0 first). ValueError when a
    template does not contain the slot exactly once."""

def paired_gaps(score_fn: Callable[[str], float], text_pairs: Sequence[tuple[str, str]]) -> NDArray:
    """float64 [len(text_pairs)]: score_fn(a) - score_fn(b) for each pair, in order."""

def bias_gap(
    score_fn: Callable[[str], float],
    templates: Sequence[str],
    pairs: Sequence[tuple[str, str]],
    rng: Any,
    n_boot: int = 1000,
    alpha: float = 0.05,
) -> dict[str, float]:
    """{"gap", "lo", "hi", "n"}: the mean of paired_gaps over
    expand_pairs(templates, pairs) and its percentile bootstrap interval
    over those paired differences (M07.4's bootstrap_ci with stat = mean).
    Positive means the a terms score higher."""

def stereotype_preference(
    score_fn: Callable[[str], float], sentence_pairs: Sequence[tuple[str, str]], alpha: float = 0.05
) -> dict[str, float]:
    """CrowS-Pairs style: sentence_pairs are (stereotypical, anti-stereotypical)
    sentences. k = pairs where the stereotypical one scores strictly higher;
    exact ties are dropped. {"rate": k / n, "lo", "hi" (Wilson), "n": pairs
    kept, "ties"}. 0.5 means no preference. ValueError when every pair ties."""

def bias_report(
    model_id: str,
    score_fn: Callable[[str], float],
    axes: Mapping[str, Mapping[str, Any]],
    stereo_pairs: Sequence[tuple[str, str]],
    rng: Any,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """formats/eval-results with suite "bias". axes maps an axis name to
    {"templates": [...], "pairs": [[a, b], ...]}; one row per axis in sorted
    axis order, task "bias-gap:<axis>", metric "score", value the gap, ci95
    [lo, hi], n, no higher_is_better (two-sided), all from one rng in that
    order; then task "stereotype-preference" (value the rate, Wilson ci95,
    n, higher_is_better False: a gate bounds how often stereotypes win) when
    stereo_pairs is not empty. Every row has
    model = model_id and status "ok"."""
