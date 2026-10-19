"""Sequence metrics: exact match, BLEU, chrF, and their bootstrap intervals (L4.5).

A translation, a date, a summary: a generated sequence is graded against one
or more references. Exact match asks "identical?"; BLEU counts shared word
n-grams (clipped, so repeating a correct word earns nothing) and penalizes
short output; chrF counts shared character n-grams, which forgives a wrong
inflection that BLEU scores as a miss. Both are computed from additive
sufficient statistics, so a corpus score pools counts over sentences, and a
bootstrap over sentences (M07.4) gives the interval every reported number
carries. The arithmetic follows sacreBLEU 2.x with its defaults.

Contract: contracts/py/tinyllm/eval/seqmetrics.pyi.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from tinyllm.prob.stats import bootstrap_ci

TOKENIZERS = ("13a", "none")
MAX_ORDER = 4  # BLEU n-gram orders 1..4
CHRF_N, CHRF_BETA = 6, 2.0
LOG_ZERO = -9999999999.0  # sacreBLEU's log(0)

_13A = [
    # symbols and punctuation: { | } ~ [ \ ] ^ _ ` space ! " # $ % & ( ) * + : ; < = > ? @ /
    (re.compile(r"([\{-\~\[-\` -\&\(-\+\:-\@\/])"), r" \1 "),
    # period and comma unless preceded by a digit
    (re.compile(r"([^0-9])([\.,])"), r"\1 \2 "),
    # period and comma unless followed by a digit
    (re.compile(r"([\.,])([^0-9])"), r" \1 \2"),
    # dash when preceded by a digit
    (re.compile(r"([0-9])(-)"), r"\1 \2 "),
]


def tokenize_13a(line: str) -> str:
    line = line.replace("<skipped>", "").replace("-\n", "").replace("\n", " ")
    if "&" in line:
        line = line.replace("&quot;", '"').replace("&amp;", "&")
        line = line.replace("&lt;", "<").replace("&gt;", ">")
    line = f" {line} "
    for pattern, repl in _13A:
        line = pattern.sub(repl, line)
    return " ".join(line.split())


def _check_pairs(hyps: Sequence[str], refs: Sequence[Any]) -> None:
    if len(hyps) != len(refs):
        raise ValueError(f"{len(hyps)} hypotheses but {len(refs)} reference entries")
    if len(hyps) == 0:
        raise ValueError("no hypotheses: a corpus metric needs at least one sentence")


def _ref_list(refs: Sequence[str]) -> list[str]:
    """One hypothesis's references as a list; a bare str is a mistake."""
    if isinstance(refs, str):
        raise TypeError(
            "refs[i] must be a sequence of reference strings, not a str "
            "(a str would be read as one reference per character)"
        )
    out = list(refs)
    if not out:
        raise ValueError("a hypothesis needs at least one reference")
    return out


def exact_match(hyps: Sequence[str], refs: Sequence[str]) -> float:
    _check_pairs(hyps, refs)
    hits = sum(1 for h, r in zip(hyps, refs) if h.strip() == r.strip())
    return hits / len(hyps)


# --- BLEU ---------------------------------------------------------------------------


def _tokens(line: str, tokenize: str) -> list[str]:
    if tokenize not in TOKENIZERS:
        raise ValueError(f"tokenize must be one of {TOKENIZERS}, got {tokenize!r}")
    line = line.rstrip()
    return (tokenize_13a(line) if tokenize == "13a" else line).split()


def _word_ngrams(tokens: list[str]) -> Counter:
    """Every n-gram of orders 1..4, as tuples, with counts."""
    grams: Counter = Counter()
    for n in range(1, MAX_ORDER + 1):
        for i in range(len(tokens) - n + 1):
            grams[tuple(tokens[i : i + n])] += 1
    return grams


def _closest_ref_len(hyp_len: int, ref_lens: list[int]) -> int:
    best_diff, best_len = -1, -1
    for r in ref_lens:
        diff = abs(hyp_len - r)
        if best_diff == -1 or diff < best_diff:
            best_diff, best_len = diff, r
        elif diff == best_diff and r < best_len:
            best_len = r
    return best_len


def bleu_stats(hyp: str, refs: Sequence[str], tokenize: str = "13a") -> list[int]:
    refs = _ref_list(refs)
    max_ref: Counter = Counter()
    ref_lens = []
    for r in refs:
        toks = _tokens(r, tokenize)
        ref_lens.append(len(toks))
        for g, c in _word_ngrams(toks).items():
            if c > max_ref[g]:
                max_ref[g] = c
    htoks = _tokens(hyp, tokenize)
    correct = [0] * MAX_ORDER
    total = [0] * MAX_ORDER
    for g, c in _word_ngrams(htoks).items():
        total[len(g) - 1] += c
        correct[len(g) - 1] += min(c, max_ref[g])
    return [len(htoks), _closest_ref_len(len(htoks), ref_lens)] + correct + total


def bleu_from_stats(stats: Sequence[float]) -> float:
    if len(stats) != 2 + 2 * MAX_ORDER:
        raise ValueError(f"BLEU has {2 + 2 * MAX_ORDER} statistics, got {len(stats)}")
    sys_len, ref_len = stats[0], stats[1]
    correct, total = stats[2 : 2 + MAX_ORDER], stats[2 + MAX_ORDER :]
    if not any(correct):
        return 0.0
    bp = 1.0
    if sys_len < ref_len:
        bp = math.exp(1.0 - ref_len / sys_len) if sys_len > 0 else 0.0
    logs = [LOG_ZERO] * MAX_ORDER
    smooth = 1.0
    for n in range(MAX_ORDER):
        if total[n] == 0:
            break
        if correct[n] == 0:
            smooth *= 2.0
            logs[n] = math.log(100.0 / (smooth * total[n]))
        else:
            logs[n] = math.log(100.0 * correct[n] / total[n])
    return bp * math.exp(sum(logs) / MAX_ORDER)


def corpus_bleu(hyps: Sequence[str], refs: Sequence[Sequence[str]], tokenize: str = "13a") -> float:
    _check_pairs(hyps, refs)
    summed = [0] * (2 + 2 * MAX_ORDER)
    for h, rs in zip(hyps, refs):
        for j, v in enumerate(bleu_stats(h, rs, tokenize)):
            summed[j] += v
    return bleu_from_stats(summed)


# --- chrF ---------------------------------------------------------------------------


def _check_chrf(n: int, beta: float) -> None:
    if int(n) != n or n < 1:
        raise ValueError(f"n must be an integer >= 1, got {n!r}")
    if not beta > 0:
        raise ValueError(f"beta must be > 0, got {beta!r}")


def _char_ngrams(line: str, n: int) -> list[Counter]:
    """Character n-gram counts of orders 1..n, white space removed."""
    s = "".join(line.split())
    return [Counter(s[i : i + k] for i in range(len(s) - k + 1)) for k in range(1, n + 1)]


def chrf_from_stats(stats: Sequence[float], n: int = CHRF_N, beta: float = CHRF_BETA) -> float:
    _check_chrf(n, beta)
    if len(stats) != 3 * n:
        raise ValueError(f"chrF with n = {n} has {3 * n} statistics, got {len(stats)}")
    factor = beta**2
    prec = rec = 0.0
    orders = 0
    for k in range(n):
        n_hyp, n_ref, n_match = stats[3 * k : 3 * k + 3]
        if n_hyp > 0 and n_ref > 0:
            prec += n_match / n_hyp
            rec += n_match / n_ref
            orders += 1
    if orders == 0:
        return 0.0
    prec, rec = prec / orders, rec / orders
    if prec + rec == 0:
        return 0.0
    return 100.0 * (1 + factor) * prec * rec / (factor * prec + rec)


def chrf_stats(hyp: str, refs: Sequence[str], n: int = CHRF_N, beta: float = CHRF_BETA) -> list[int]:
    _check_chrf(n, beta)
    refs = _ref_list(refs)
    hyp_grams = _char_ngrams(hyp, n)
    best, best_f = None, -1.0
    for r in refs:
        stats = []
        for h, g in zip(hyp_grams, _char_ngrams(r, n)):
            match = sum(min(c, g[x]) for x, c in h.items() if x in g)
            stats += [sum(h.values()) if g else 0, sum(g.values()), match]
        f = chrf_from_stats(stats, n, beta)
        if f > best_f:
            best, best_f = stats, f
    return best


def chrf(hyps: Sequence[str], refs: Sequence[Sequence[str]], n: int = CHRF_N, beta: float = CHRF_BETA) -> float:
    _check_pairs(hyps, refs)
    summed = [0] * (3 * n)
    for h, rs in zip(hyps, refs):
        for j, v in enumerate(chrf_stats(h, rs, n, beta)):
            summed[j] += v
    return chrf_from_stats(summed, n, beta)


# --- intervals ------------------------------------------------------------------------


METRICS = ("exact_match", "bleu", "chrf")


def sentence_stats(metric: str, hyps: Sequence[str], refs: Sequence[Any]) -> NDArray:
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
    _check_pairs(hyps, refs)
    if metric == "exact_match":
        rows = [[1.0 if h.strip() == r.strip() else 0.0] for h, r in zip(hyps, refs)]
    elif metric == "bleu":
        rows = [bleu_stats(h, rs) for h, rs in zip(hyps, refs)]
    else:
        rows = [chrf_stats(h, rs) for h, rs in zip(hyps, refs)]
    return np.asarray(rows, dtype=np.float64)


def score_from_stats(metric: str, summed: NDArray, n_sentences: int) -> float:
    if metric == "exact_match":
        return float(summed[0]) / n_sentences
    if metric == "bleu":
        return bleu_from_stats([float(v) for v in summed])
    if metric == "chrf":
        return chrf_from_stats([float(v) for v in summed])
    raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")


def metric_ci(
    metric: str,
    hyps: Sequence[str],
    refs: Sequence[Any],
    n_boot: int = 1000,
    alpha: float = 0.05,
    rng: Any = None,
) -> tuple[float, float, float]:
    if rng is None:
        raise ValueError("metric_ci needs an rng (a PCG32): the interval must be reproducible")
    rows = sentence_stats(metric, hyps, refs)

    def stat(idx: NDArray) -> float:
        picked = rows[np.asarray(idx, dtype=np.int64)]
        return score_from_stats(metric, picked.sum(axis=0), len(picked))

    point, lo, hi = bootstrap_ci(np.arange(len(rows), dtype=np.float64), stat, n_boot, alpha, rng)
    return float(point), float(lo), float(hi)
