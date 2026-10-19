"""The language-model evaluation harness (L6.7).

Perplexity of a fixed-context model over a long text, scored with strided
windows so every token gets as much context as the window allows;
multiple-choice tasks scored by log-likelihood (the lm-evaluation-harness
recipe); every metric with a confidence interval (M07.4); and a paired
permutation test (M07.5) to say whether model A beats model B on the same
items.

Contract: contracts/py/tinyllm/eval/lm.pyi.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tinyllm.autograd.mode import no_grad
from tinyllm.autograd.tensor import Tensor
from tinyllm.info.ppl import NLLAccumulator
from tinyllm.num.rng import PCG32
from tinyllm.prob.stats import bootstrap_ci, mean_ci
from tinyllm.prob.tests import paired_permutation_test


@dataclass
class Result:
    value: float
    lo: float
    hi: float
    n: int
    per_item: NDArray


class ByteTokenizer:
    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))


def log_probs(model: Any, ids: ArrayLike) -> NDArray:
    x = np.asarray(ids, dtype=np.int64)
    with no_grad():
        out = model(x[None])
    if isinstance(out, tuple):  # GPT returns (logits, loss)
        out = out[0]
    z = np.asarray(out.data if isinstance(out, Tensor) else out, dtype=np.float64)[0]
    if z.shape[0] != x.size:
        raise ValueError(f"the model returned {z.shape[0]} positions for {x.size} ids")
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def token_nlls(model: Any, ids: ArrayLike, ctx_len: int, stride: int) -> NDArray:
    x = np.asarray(ids, dtype=np.int64)
    n = x.size
    if x.ndim != 1 or n < 2:
        raise ValueError(f"ids must be 1-D with at least 2 tokens, got shape {x.shape}")
    if not 1 <= stride < ctx_len:
        # stride == ctx_len would leave each window's first token unscored.
        raise ValueError(f"need 1 <= stride < ctx_len, got stride {stride}, ctx_len {ctx_len}")
    out = np.empty(n - 1, dtype=np.float64)
    prev = 0
    for s in range(0, n, stride):
        e = min(s + ctx_len, n)
        lp = log_probs(model, x[s:e])
        # Only the tokens this window adds; the overlap was scored by an
        # earlier window. Each new token sees at least ctx_len - stride
        # tokens of context (all of them near the start).
        for t in range(max(prev, 1), e):
            out[t - 1] = -lp[t - 1 - s, x[t]]
        prev = e
        if e == n:
            break
    return out


def eval_ppl(model: Any, ids: ArrayLike, ctx_len: int, stride: int, n_bytes: int = 0) -> dict[str, float]:
    nll = token_nlls(model, ids, ctx_len, stride)
    acc = NLLAccumulator()  # M11.2: compensated sums, bpb
    acc.add(nll, n_bytes=int(n_bytes))
    r = acc.result()
    out = {k: float(r[k]) for k in ("nll_sum", "n_tokens", "nll_mean", "ppl", "bits_per_token")}
    if n_bytes:
        out["bpb"] = float(r["bpb"])
    if nll.size >= 2:
        _, lo, hi = mean_ci(nll)  # M07.4: Student t over the per-token NLLs
        out["ppl_lo"], out["ppl_hi"] = math.exp(lo), math.exp(hi)
        if n_bytes:
            scale = nll.size / (n_bytes * math.log(2.0))
            out["bpb_lo"], out["bpb_hi"] = lo * scale, hi * scale
    return out


def score_choices(
    model: Any, tok: Any, context: str, choices: Sequence[str], normalize: bool = True, max_len: Optional[int] = None
) -> NDArray:
    ctx = list(tok.encode(context))
    if not ctx:
        raise ValueError("score_choices needs a non-empty context")
    scores = np.empty(len(choices), dtype=np.float64)
    for i, ch in enumerate(choices):
        cont = list(tok.encode(ch))
        if not cont:
            raise ValueError(f"choice {i} encodes to no tokens")
        seq = ctx + cont
        if max_len is not None and len(seq) > max_len:
            seq = seq[-max_len:]  # keep the end: the choice and the nearest context
            if len(seq) <= len(cont):
                raise ValueError(f"choice {i} does not fit in max_len {max_len} with any context")
        lp = log_probs(model, seq)
        k = len(seq) - len(cont)
        ll = float(sum(lp[k - 1 + j, cont[j]] for j in range(len(cont))))
        scores[i] = ll / len(ch.encode("utf-8")) if normalize else ll
    return scores


def run_task(
    model: Any, tok: Any, task_jsonl: str, metric: str, rng: Any, n_boot: int = 1000, max_len: Optional[int] = None
) -> Result:
    if metric not in ("acc", "acc_norm"):
        raise ValueError(f"metric must be 'acc' or 'acc_norm', got {metric!r}")
    items = [json.loads(line) for line in Path(task_jsonl).read_text().splitlines() if line.strip()]
    if not items:
        raise ValueError(f"{task_jsonl} has no items")
    right = np.empty(len(items), dtype=np.float64)
    for i, it in enumerate(items):
        s = score_choices(model, tok, it["context"], it["choices"], normalize=metric == "acc_norm", max_len=max_len)
        right[i] = float(int(np.argmax(s)) == int(it["label"]))  # ties to the first choice
    value, lo, hi = bootstrap_ci(right, np.mean, n_boot, 0.05, rng)  # M07.4
    return Result(float(value), float(lo), float(hi), len(items), right)


def compare(a: Result, b: Result, n_perm: int = 10000, rng: Any = None) -> float:
    if a.per_item.shape != b.per_item.shape:
        raise ValueError(f"paired comparison needs the same items: {a.per_item.shape} vs {b.per_item.shape}")
    r = rng if rng is not None else PCG32(0).substream("sample")
    return paired_permutation_test(a.per_item, b.per_item, n_perm, r)  # M07.5
