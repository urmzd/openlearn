"""tinyllm.eval.mm (L14.5): multimodal eval metrics.

A transcript is graded by how many word edits turn the reference into the
hypothesis: substitutions, deletions, and insertions, counted on the
cheapest alignment (Levenshtein's dynamic program). Word error rate pools
the edits over a corpus and divides by the reference words, so it is a rate
that can exceed 1. The text is normalized first (case, punctuation,
bracketed noise tags), or every comma counts as an error. The vision
metrics are smaller: VQA accuracy credits an answer by how many of the
humans gave it, captions reuse L4.5's BLEU and chrF plus a recall of the
attributes the references agree on, and zero-shot classification is top-k
accuracy. Every metric gets an interval from M07.4's bootstrap over items.

Contract: contracts/py/tinyllm/eval/mm.pyi.
"""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from tinyllm.eval.seqmetrics import chrf, corpus_bleu
from tinyllm.num.rng import PCG32
from tinyllm.prob.stats import bootstrap_ci as _bootstrap


def basic_normalize(text: str) -> str:
    s = text.lower()
    s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # <noise>, [laughter]
    s = re.sub(r"\(([^)]+?)\)", "", s)  # (inaudible)
    # Marks, symbols, and punctuation become spaces; letters and digits stay.
    s = "".join(
        " " if unicodedata.category(c)[0] in "MSP" else c
        for c in unicodedata.normalize("NFKC", s)
    ).lower()
    return re.sub(r"\s+", " ", s)


def edit_ops(ref: Sequence[str], hyp: Sequence[str]) -> tuple[int, int, int, int]:
    r, h = list(ref), list(hyp)
    n, m = len(r), len(h)
    # d[i][j]: edits to turn r[:i] into h[:j].
    d = np.zeros((n + 1, m + 1), dtype=np.int64)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = d[i - 1, j - 1] + (r[i - 1] != h[j - 1])
            d[i, j] = min(sub, d[i - 1, j] + 1, d[i, j - 1] + 1)
    # Backtrace from the end; on equal cost prefer the diagonal, then a
    # deletion, then an insertion.
    S = D = I = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (r[i - 1] != h[j - 1]):
            S += int(r[i - 1] != h[j - 1])
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            D += 1
            i -= 1
        else:
            I += 1
            j -= 1
    return S, D, I, n


def _pairs(refs: Sequence[str], hyps: Sequence[str]) -> None:
    if isinstance(refs, str) or isinstance(hyps, str):
        raise TypeError("refs and hyps are sequences of strings, not one string")
    if len(refs) != len(hyps):
        raise ValueError(f"{len(refs)} references but {len(hyps)} hypotheses")
    if not refs:
        raise ValueError("no items to score")


def wer_stats(refs: Sequence[str], hyps: Sequence[str]) -> NDArray:
    _pairs(refs, hyps)
    return np.array([edit_ops(r.split(), h.split()) for r, h in zip(refs, hyps)], dtype=np.int64)


def _rate(st: NDArray) -> float:
    total = int(st[:, 3].sum())
    if total == 0:
        raise ValueError("every reference is empty: the error rate is undefined")
    return float(st[:, :3].sum()) / total


def wer(refs: Sequence[str], hyps: Sequence[str]) -> float:
    return _rate(wer_stats(refs, hyps))


def cer(refs: Sequence[str], hyps: Sequence[str]) -> float:
    _pairs(refs, hyps)
    st = np.array(
        [edit_ops(list(r.strip()), list(h.strip())) for r, h in zip(refs, hyps)],
        dtype=np.int64,
    )
    return _rate(st)


def _per_item(refs: Sequence, n: int, what: str) -> None:
    if len(refs) != n:
        raise ValueError(f"{n} predictions but {len(refs)} {what} lists")
    if n == 0:
        raise ValueError("no items to score")
    for r in refs:
        if isinstance(r, str):
            raise TypeError(f"each item's {what} is a list of strings, not one string")
        if len(r) == 0:
            raise ValueError(f"an item has no {what}")


def vqa_accuracy(preds: Sequence[str], answers: Sequence[Sequence[str]]) -> float:
    _per_item(answers, len(preds), "answers")
    total = 0.0
    for p, ans in zip(preds, answers):
        key = basic_normalize(p).strip()
        m = sum(basic_normalize(a).strip() == key for a in ans)
        total += min(1.0, m / 3.0)
    return total / len(preds)


def caption_scores(
    preds: Sequence[str], refs: Sequence[Sequence[str]], attributes: Sequence[str] = ()
) -> dict[str, float]:
    _per_item(refs, len(preds), "references")
    vocab = {basic_normalize(a).strip() for a in attributes}
    recalls = []
    for p, rs in zip(preds, refs):
        words = [set(basic_normalize(r).split()) for r in rs]
        if vocab:
            want = set().union(*words) & vocab
        else:
            want = set.intersection(*words)
        if want:
            have = set(basic_normalize(p).split())
            recalls.append(len(want & have) / len(want))
    return {
        "bleu4": corpus_bleu(list(preds), [list(r) for r in refs]),
        "chrf": chrf(list(preds), [list(r) for r in refs]),
        "attr_recall": float(np.mean(recalls)) if recalls else math.nan,
    }


def zero_shot_accuracy(pred: NDArray, gold: NDArray, k: int = 1) -> float:
    s = np.asarray(pred, dtype=np.float64)
    g = np.asarray(gold)
    if s.ndim != 2 or g.shape != (s.shape[0],) or s.shape[0] == 0:
        raise ValueError(f"need scores [n, classes] and gold [n], got {s.shape}, {g.shape}")
    c = s.shape[1]
    if not 1 <= k <= c:
        raise ValueError(f"k = {k} outside [1, {c}]")
    if g.dtype.kind not in "iu" or (g < 0).any() or (g >= c).any():
        raise ValueError(f"gold labels must be integers in [0, {c})")
    # A stable sort of -score ranks ties by the lower class index.
    top = np.argsort(-s, axis=1, kind="stable")[:, :k]
    return float((top == g[:, None]).any(axis=1).mean())


def bootstrap_ci(
    per_item: NDArray,
    stat: Callable[[NDArray], float],
    n: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    a = np.asarray(per_item)
    idx = np.arange(len(a), dtype=np.float64)
    return _bootstrap(idx, lambda ix: float(stat(a[ix.astype(np.int64)])), n, alpha, PCG32(seed))
