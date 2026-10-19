"""Beam search over any step function (L4.4).

The search holds the beam_size best prefixes by summed log-probability. One
step scores every one-token extension of every live prefix, ranks all of
them together, and keeps the best beam_size; extensions that end in eos
leave the beam as finished hypotheses. The model is only a step function,
so the same code decodes a seq2seq checkpoint (L4.1) and, later, a
transformer from the zoo (L6.7).

Contract: contracts/py/tinyllm/infer/beam.pyi.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tinyllm.num.stable import log_softmax


@dataclass
class Hypothesis:
    tokens: list[int]
    logprob: float
    score: float
    finished: bool


def select_state(state: Any, idx: ArrayLike) -> Any:
    idx = np.asarray(idx, dtype=np.int64)
    if isinstance(state, dict):
        return {k: select_state(v, idx) for k, v in state.items()}
    if isinstance(state, tuple):
        parts = [select_state(v, idx) for v in state]
        # A NamedTuple is rebuilt field by field; a plain tuple from the list.
        return type(state)(*parts) if hasattr(state, "_fields") else tuple(parts)
    if isinstance(state, list):
        return [select_state(v, idx) for v in state]
    if isinstance(state, (set, frozenset)):
        raise TypeError("select_state: a set has no row order")
    shape = getattr(state, "shape", None)
    if shape is None or len(shape) == 0:
        return state
    return state[idx]


def _score(logprob: float, n: int, length_penalty: float) -> float:
    return logprob / (float(n) ** length_penalty)


def beam_search(
    step_fn: Callable[[Any, NDArray], tuple[Any, Any]],
    init_state: Any,
    bos: int,
    eos: int,
    beam_size: int,
    max_len: int,
    length_penalty: float = 1.0,
) -> list[Hypothesis]:
    if beam_size < 1:
        raise ValueError(f"beam_size must be at least 1, got {beam_size}")
    if max_len < 1:
        raise ValueError(f"max_len must be at least 1, got {max_len}")
    if length_penalty < 0:
        raise ValueError(f"length_penalty must be >= 0, got {length_penalty}")
    live_tokens: list[list[int]] = [[]]
    live_lp = np.zeros(1, dtype=np.float64)
    state = init_state
    y_prev = np.array([bos], dtype=np.int64)
    finished: list[Hypothesis] = []
    for t in range(max_len):
        logits, state = step_fn(state, y_prev)
        z = np.asarray(getattr(logits, "data", logits), dtype=np.float64)
        k = len(live_tokens)
        if z.ndim != 2 or z.shape[0] != k:
            raise ValueError(
                f"step_fn must return logits [{k}, V], got shape {z.shape}"
            )
        V = z.shape[1]
        # Normalize each row first: logits are scores, not log-probabilities.
        cand = (live_lp[:, None] + log_softmax(z, axis=-1)).ravel()
        # Highest first; a stable sort of the negated scores sends ties to the
        # lower flat index b * V + v.
        order = np.argsort(-cand, kind="stable")
        # Every finished hypothesis takes one slot for good: the beam narrows.
        n_keep = min(beam_size - len(finished), int(np.isfinite(cand).sum()))
        new_tokens: list[list[int]] = []
        new_lp: list[float] = []
        parents: list[int] = []
        for f in order[:n_keep]:
            b, v = divmod(int(f), V)
            toks = live_tokens[b] + [v]
            lp = float(cand[f])
            if v == eos:
                finished.append(
                    Hypothesis(toks, lp, _score(lp, len(toks), length_penalty), True)
                )
            else:
                new_tokens.append(toks)
                new_lp.append(lp)
                parents.append(b)
        if not new_tokens or len(finished) >= beam_size:
            break
        if t == max_len - 1:
            # Out of length: the survivors are returned as they stand.
            for toks, lp in zip(new_tokens, new_lp):
                finished.append(
                    Hypothesis(toks, lp, _score(lp, len(toks), length_penalty), False)
                )
            break
        state = select_state(state, parents)
        live_tokens = new_tokens
        live_lp = np.array(new_lp, dtype=np.float64)
        y_prev = np.array([toks[-1] for toks in new_tokens], dtype=np.int64)
    # sorted() is stable: equal scores keep the order they were finalized in.
    return sorted(finished, key=lambda h: -h.score)[:beam_size]


def greedy_decode(
    step_fn: Callable[[Any, NDArray], tuple[Any, Any]],
    init_state: Any,
    bos: int,
    eos: int,
    max_len: int,
) -> Hypothesis:
    hyps = beam_search(step_fn, init_state, bos, eos, 1, max_len, length_penalty=0.0)
    if not hyps:
        raise ValueError("greedy_decode: every token is -inf at the first step")
    return hyps[0]
