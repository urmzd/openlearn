"""primers/craft.22/evals.py: model evals as tests (rung R8), the kata.

Your eval library for the course's tiny model (tinymodel.py). Your suite,
test_model_evals.py, uses it to decide whether a model regressed against
the numbers you recorded from the good model in baseline.json. The rules
below are the spec; the course compares your functions with its own.

    heldout_bpb(model, text, seed, windows=16, chars=80)
        rng = np.random.default_rng(seed); starts = rng.integers(0,
        len(text) - chars, size=windows). For each window w = text[s:s+chars]:
        ids = model.encode(w); lp = model.log_probs(ids); bits = -sum over
        t of lp[t, ids[t+1]] / ln 2; bytes = len(w.encode()) - len of the
        first token's text in bytes (it is never predicted). Return the sum
        of bits over the sum of bytes (pooled, not a mean of ratios).
    words(text)             the runs of [a-z] in text.lower()
    repeat_ngram_rate(ws, n=4)
        of the n-grams ws[i:i+n], the fraction equal to an n-gram that starts
        earlier in ws; 0.0 when there are none
    valid_word_rate(text, vocabulary)
        the fraction of words(text) in vocabulary; 0.0 when there are none
    quality_score(model, vocabulary, seed, tokens=48)
        rng = np.random.default_rng(seed); one model.sample(prompt, tokens,
        rng) per prompt of PROMPTS, in order, with the one rng; ws =
        words(" ".join(samples)); valid_word_rate(" ".join(samples)) *
        (1 - repeat_ngram_rate(ws, 4)). Higher is better.
    measure(model, text, vocabulary, seeds=SEEDS)
        {"seeds": [...], "bpb": [heldout_bpb per seed], "quality":
        [quality_score per seed]}
    sign_flip_p(diffs)
        the exact one-sided paired permutation p-value of H1 "the mean of
        diffs is above 0": over all 2^n sign vectors s, the fraction with
        sum(s * diffs) >= sum(diffs) (ties count, within 1e-12).
    regressed(base, cand, margin, alpha=0.05, higher_is_better=False)
        worse_i = cand_i - base_i (or base_i - cand_i when higher is better);
        True when mean(worse) > margin and sign_flip_p(worse) < alpha.

    python primers/craft.22/evals.py --record    writes baseline.json
"""

from __future__ import annotations

import itertools
import json
import math
import re
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

SEEDS = (0, 1, 2, 3, 4)
PROMPTS = ("Once upon a time", "One day", "Mia and the", "At night")
BPB_MARGIN = 0.02
QUALITY_MARGIN = 0.02


def heldout_bpb(model, text: str, seed: int, windows: int = 16, chars: int = 80) -> float:
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(text) - chars, size=windows)
    bits = 0.0
    nbytes = 0
    for s in starts:
        w = text[int(s) : int(s) + chars]
        ids = model.encode(w)
        lp = model.log_probs(ids)
        bits += -sum(float(lp[t, ids[t + 1]]) for t in range(len(ids) - 1)) / math.log(2)
        nbytes += len(w.encode()) - len(model.decode(ids[:1]).encode())
    return bits / nbytes


def words(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def repeat_ngram_rate(ws: Sequence[str], n: int = 4) -> float:
    grams = [tuple(ws[i : i + n]) for i in range(len(ws) - n + 1)]
    if not grams:
        return 0.0
    seen: set[tuple[str, ...]] = set()
    repeats = 0
    for g in grams:
        repeats += g in seen
        seen.add(g)
    return repeats / len(grams)


def valid_word_rate(text: str, vocabulary: set[str]) -> float:
    ws = words(text)
    return sum(w in vocabulary for w in ws) / len(ws) if ws else 0.0


def quality_score(model, vocabulary: set[str], seed: int, tokens: int = 48) -> float:
    rng = np.random.default_rng(seed)
    text = " ".join(model.sample(p, tokens, rng) for p in PROMPTS)
    return valid_word_rate(text, vocabulary) * (1.0 - repeat_ngram_rate(words(text), 4))


def measure(model, text: str, vocabulary: set[str], seeds: Sequence[int] = SEEDS) -> dict:
    return {
        "seeds": list(seeds),
        "bpb": [heldout_bpb(model, text, s) for s in seeds],
        "quality": [quality_score(model, vocabulary, s) for s in seeds],
    }


def sign_flip_p(diffs: Sequence[float]) -> float:
    d = np.asarray(diffs, dtype=np.float64)
    observed = d.sum()
    hits = sum(
        float(np.dot(s, d)) >= observed - 1e-12 for s in itertools.product((1.0, -1.0), repeat=len(d))
    )
    return hits / 2 ** len(d)


def regressed(
    base: Sequence[float],
    cand: Sequence[float],
    margin: float,
    alpha: float = 0.05,
    higher_is_better: bool = False,
) -> bool:
    b, c = np.asarray(base, dtype=np.float64), np.asarray(cand, dtype=np.float64)
    worse = b - c if higher_is_better else c - b
    return bool(worse.mean() > margin and sign_flip_p(worse) < alpha)


def _record() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import tinymodel

    text = tinymodel.fixture("val.txt").read_text()
    vocab = set(tinymodel.fixture("words.txt").read_text().split())
    out = Path(__file__).resolve().parent / "baseline.json"
    out.write_text(json.dumps(measure(tinymodel.load(), text, vocab), indent=1) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__" and sys.argv[1:] == ["--record"]:
    sys.exit(_record())
