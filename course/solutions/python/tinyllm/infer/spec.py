"""Speculative decoding: n-gram, prompt-lookup, and model drafts (L8.6).

Contract: contracts/py/tinyllm/infer/spec.pyi. A draft proposes k tokens; the
target scores them all in one forward pass over its KV cache; verify_draft
keeps the longest acceptable prefix plus one token of the target's own; the
cache is truncated back to what was kept. Greedy acceptance is "the draft
equals the target's argmax"; sampled acceptance is M07.6's rejection step,
which makes every emitted token follow the target's distribution exactly.
"""

from __future__ import annotations

import time
from typing import Any, Optional, Protocol, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tinyllm.infer.generate import Generation, cache_dims
from tinyllm.infer.kvcache import KVCache
from tinyllm.infer.sample import (
    SamplingParams,
    request_rng,
    sample,
    sampling_distribution,
    token_logprobs,
)
from tinyllm.lm.ngram import NGramLM
from tinyllm.prob.rejection import speculative_step
from tinyllm.prob.sampling import UniformSource

GREEDY = SamplingParams(temperature=0.0)


class DraftModel(Protocol):
    """What speculative_generate needs from a draft (spec.pyi)."""

    def propose(
        self, ctx: Sequence[int], k: int, rng: Optional[UniformSource]
    ) -> tuple[list[int], Optional[NDArray]]: ...


def _forward_rows(model: Any, ids: Sequence[int], start: int, cache: Any) -> NDArray:
    """float64 [len(ids), V]: the model's logits at positions start .. start
    + len(ids) - 1, appending those positions to cache."""
    pos = np.arange(start, start + len(ids), dtype=np.int64)
    out = model.forward(
        np.asarray([list(ids)], dtype=np.int64), positions=pos, cache=cache
    )
    return np.asarray(getattr(out, "data", out), dtype=np.float64)[0]


def _draft_token(
    logits: NDArray, p: SamplingParams, rng: Optional[UniformSource]
) -> tuple[int, Optional[NDArray]]:
    """One draft id from logits: greedy with no rng, else L8.1's sample under
    p with its exact distribution as the row."""
    if rng is None:
        return sample(logits, GREEDY, [], None)[0], None
    return sample(logits, p, [], rng)[0], sampling_distribution(logits, p)


class NGramDraft:
    def __init__(self, lm: NGramLM) -> None:
        self.lm = lm

    def propose(
        self, ctx: Sequence[int], k: int, rng: Optional[UniformSource]
    ) -> tuple[list[int], Optional[NDArray]]:
        cur = [int(t) for t in ctx]
        ids, rows = [], []
        for _ in range(max(int(k), 0)):
            tok, row = _draft_token(self.lm.logprobs(cur), SamplingParams(), rng)
            ids.append(tok)
            rows.append(row)
            cur.append(tok)
        if rng is None or not ids:
            return ids, None
        return ids, np.stack(rows)


class PromptLookupDraft:
    def __init__(self, max_ngram: int = 3, min_ngram: int = 1) -> None:
        if not 1 <= min_ngram <= max_ngram:
            raise ValueError(
                f"need 1 <= min_ngram <= max_ngram, got {min_ngram}, {max_ngram}"
            )
        self.max_ngram, self.min_ngram = int(max_ngram), int(min_ngram)

    def propose(
        self, ctx: Sequence[int], k: int, rng: Optional[UniformSource]
    ) -> tuple[list[int], Optional[NDArray]]:
        ctx = [int(t) for t in ctx]
        L = len(ctx)
        if k <= 0:
            return [], None
        for n in range(min(self.max_ngram, L - 1), self.min_ngram - 1, -1):
            suffix = ctx[L - n :]
            for i in range(L - n - 1, -1, -1):  # most recent earlier occurrence first
                if ctx[i : i + n] == suffix:
                    cont = ctx[i + n : i + n + k]
                    if cont:
                        return cont, None
        return [], None


class ModelDraft:
    def __init__(self, model: Any, temperature: float = 1.0) -> None:
        if not temperature > 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.model = model
        self.temperature = float(temperature)
        n_layers, n_kv, d_head, max_len = cache_dims(model)
        self._cache = KVCache(n_layers, n_kv, d_head, max_len)
        self._ids: list[int] = []  # what the cache holds, position by position

    def propose(
        self, ctx: Sequence[int], k: int, rng: Optional[UniformSource]
    ) -> tuple[list[int], Optional[NDArray]]:
        ctx = [int(t) for t in ctx]
        if not ctx:
            raise ValueError("ModelDraft needs a non-empty context")
        max_len = self._cache.max_len
        k = max(0, min(int(k), max_len - len(ctx)))
        if k == 0:
            return [], None
        # Sync the cache with ctx: keep the common prefix, but always re-feed
        # at least the last id, whose logits are the ones we need.
        common = 0
        while (
            common < min(len(self._ids), len(ctx) - 1)
            and self._ids[common] == ctx[common]
        ):
            common += 1
        self._cache.truncate(common)
        self._ids = ctx[:common]
        logits = _forward_rows(self.model, ctx[common:], common, self._cache)[-1]
        self._ids = list(ctx)
        p = SamplingParams(temperature=self.temperature)
        ids, rows = [], []
        for i in range(k):
            tok, row = _draft_token(logits, p, rng)
            ids.append(tok)
            rows.append(row)
            if i + 1 < k:
                logits = _forward_rows(self.model, [tok], len(self._ids), self._cache)[
                    -1
                ]
                self._ids.append(tok)
        if rng is None:
            return ids, None
        return ids, np.stack(rows)


def verify_draft(
    target_logits: ArrayLike,
    draft_ids: Sequence[int],
    draft_probs: Optional[ArrayLike],
    p: SamplingParams,
    history: Sequence[int],
    rng: UniformSource,
    prompt: Sequence[int] = (),
) -> tuple[list[int], int]:
    rows = np.asarray(target_logits, dtype=np.float64)
    m = len(draft_ids)
    if rows.ndim != 2 or rows.shape[0] != m + 1:
        raise ValueError(f"target_logits must be [{m + 1}, V], got {rows.shape}")
    V = rows.shape[1]
    q = None if draft_probs is None else np.asarray(draft_probs, dtype=np.float64)
    if q is not None and q.shape != (m, V):
        raise ValueError(f"draft_probs must be [{m}, {V}], got {q.shape}")
    hist = [int(t) for t in history]
    emitted: list[int] = []
    for i, x in enumerate(int(t) for t in draft_ids):
        if p.temperature == 0:
            g = sample(rows[i], p, hist + emitted, rng, prompt)[0]  # greedy: no draw
            emitted.append(g)
            if g != x:
                return emitted, i
            continue
        P = sampling_distribution(rows[i], p, hist + emitted, prompt)
        if q is None:
            Q = np.zeros(V)
            Q[x] = 1.0
        else:
            Q = q[i]
        u_accept = rng.uniform()
        u_resample = rng.uniform()
        y, ok = speculative_step(P, Q, x, u_accept, u_resample)
        emitted.append(int(y))
        if not ok:
            return emitted, i
    emitted.append(sample(rows[m], p, hist + emitted, rng, prompt)[0])
    return emitted, m


def speculative_generate(
    target: Any,
    draft: Any,
    tok: Any,
    prompt: str,
    p: SamplingParams,
    k: int = 4,
    kv_dtype: Any = np.float32,
    eos_ids: Sequence[int] = (),
) -> Generation:
    p.validate()
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    prompt_ids = [int(t) for t in tok.encode(prompt)]
    if not prompt_ids:
        raise ValueError("the prompt encodes to no tokens")
    n_layers, n_kv, d_head, model_max = cache_dims(target)
    if len(prompt_ids) > model_max:
        raise ValueError(
            f"prompt of {len(prompt_ids)} tokens exceeds max_len {model_max}"
        )
    limit = min(model_max, len(prompt_ids) + p.max_tokens + k)
    cache = KVCache(n_layers, n_kv, d_head, limit, batch=1, dtype=kv_dtype)
    rng = request_rng(p.seed if p.seed is not None else 0)
    draft_rng = None if p.temperature == 0 else rng
    eos = {int(e) for e in eos_ids}
    stops = [s for s in p.stop if s]

    t0 = time.perf_counter()
    out: list[int] = []
    lps: list[float] = []
    committed = 0  # positions in the target cache that are kept
    pending = list(prompt_ids)  # ids whose keys and values are not cached yet
    drafted = accepted = calls = 0
    finish, cut = "length", None
    while len(out) < p.max_tokens:
        room = limit - committed - len(pending)  # cache positions left for drafts
        k_eff = max(0, min(k, p.max_tokens - len(out) - 1, room))
        drafts, q = (
            draft.propose(prompt_ids + out, k_eff, draft_rng)
            if k_eff > 0
            else ([], None)
        )
        drafts = [int(t) for t in drafts[:k_eff]]
        if q is not None:
            q = np.asarray(q, dtype=np.float64)[: len(drafts)]
        rows = _forward_rows(target, pending + drafts, committed, cache)[
            len(pending) - 1 :
        ]
        calls += 1
        emitted, n_acc = verify_draft(
            rows, drafts, q if drafts else None, p, out, rng, prompt_ids
        )
        drafted += len(drafts)
        accepted += n_acc
        committed += len(pending) + n_acc
        cache.truncate(committed)  # drop the rejected drafts' keys and values
        pending = [emitted[-1]]  # the extra token is not in the cache yet
        for j, t in enumerate(emitted):
            if t in eos:
                finish = "stop"
                break
            lps.append(float(token_logprobs(rows[j], p, out, prompt_ids)[t]))
            out.append(t)
            if stops:
                text = tok.decode(out)
                hit = min(
                    (i for i in (text.find(s) for s in stops) if i >= 0), default=-1
                )
                if hit >= 0:
                    finish, cut = "stop", text[:hit]
                    break
        if finish == "stop" or committed + len(pending) >= limit:
            break  # done, or no cache position left for the pending token
    text = cut if cut is not None else tok.decode(out)
    t1 = time.perf_counter()
    total = t1 - t0
    return Generation(
        text=text,
        ids=out,
        logprobs=lps,
        timings={"prefill_s": 0.0, "decode_s": total, "total_s": total},
        stats={
            "prompt_tokens": float(len(prompt_ids)),
            "completion_tokens": float(len(out)),
            "drafted": float(drafted),
            "accepted": float(accepted),
            "acceptance_rate": accepted / drafted if drafted else 0.0,
            "target_calls": float(calls),
            "tokens_per_s": len(out) / total if total > 0 else 0.0,
        },
        finish_reason=finish,
    )
