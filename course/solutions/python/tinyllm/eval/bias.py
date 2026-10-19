"""Templated bias probes with intervals (ethics.04).

A probe changes one group term in an otherwise identical sentence and
compares the model's scores; the per-template differences are paired data,
so their mean gets a bootstrap interval over the pairs (M07.4), and the
CrowS-Pairs style preference rate gets a Wilson interval.

Contract: contracts/py/tinyllm/eval/bias.pyi.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from tinyllm.prob.stats import bootstrap_ci, wilson_interval


def expand_pairs(
    templates: Sequence[str], pairs: Sequence[tuple[str, str]], slot: str = "{group}"
) -> list[tuple[str, str]]:
    out = []
    for t in templates:
        if t.count(slot) != 1:
            raise ValueError(f"template {t!r} must contain {slot!r} exactly once")
        for a, b in pairs:
            out.append((t.replace(slot, a), t.replace(slot, b)))
    return out


def paired_gaps(score_fn: Callable[[str], float], text_pairs: Sequence[tuple[str, str]]) -> NDArray:
    return np.array([float(score_fn(a)) - float(score_fn(b)) for a, b in text_pairs], dtype=np.float64)


def bias_gap(
    score_fn: Callable[[str], float],
    templates: Sequence[str],
    pairs: Sequence[tuple[str, str]],
    rng: Any,
    n_boot: int = 1000,
    alpha: float = 0.05,
) -> dict[str, float]:
    d = paired_gaps(score_fn, expand_pairs(templates, pairs))
    # Resample the paired differences, not the two score lists separately:
    # the template's own difficulty cancels inside each pair.
    gap, lo, hi = bootstrap_ci(d, lambda x: float(np.mean(x)), n_boot, alpha, rng)
    return {"gap": float(gap), "lo": float(lo), "hi": float(hi), "n": float(len(d))}


def stereotype_preference(
    score_fn: Callable[[str], float], sentence_pairs: Sequence[tuple[str, str]], alpha: float = 0.05
) -> dict[str, float]:
    d = paired_gaps(score_fn, sentence_pairs)
    ties = int((d == 0).sum())
    n = len(d) - ties
    if n == 0:
        raise ValueError("every pair ties: no preference to measure")
    k = int((d > 0).sum())
    lo, hi = wilson_interval(k, n, alpha)
    return {"rate": k / n, "lo": lo, "hi": hi, "n": float(n), "ties": float(ties)}


def bias_report(
    model_id: str,
    score_fn: Callable[[str], float],
    axes: Mapping[str, Mapping[str, Any]],
    stereo_pairs: Sequence[tuple[str, str]],
    rng: Any,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    rows = []
    for axis in sorted(axes):
        spec = axes[axis]
        pairs = [tuple(p) for p in spec["pairs"]]
        g = bias_gap(score_fn, spec["templates"], pairs, rng, n_boot)
        rows.append(
            {
                "model": model_id,
                "task": f"bias-gap:{axis}",
                "metric": "score",
                "value": g["gap"],
                "ci95": [g["lo"], g["hi"]],
                "n": int(g["n"]),
                "status": "ok",
            }
        )
    if stereo_pairs:
        s = stereotype_preference(score_fn, [tuple(p) for p in stereo_pairs])
        rows.append(
            {
                "model": model_id,
                "task": "stereotype-preference",
                "metric": "score",
                "higher_is_better": False,
                "value": s["rate"],
                "ci95": [s["lo"], s["hi"]],
                "n": int(s["n"]),
                "status": "ok",
            }
        )
    return {"format": "tl.eval-results.v1", "suite": "bias", "seed": int(seed), "rows": rows}
