"""Incremental decoding, an incremental UTF-8 detokenizer, and generate (L8.2).

Prefill runs the model once over the prompt and fills the KV cache; each
decode step then runs it on the one token just sampled. Text streams out
through IncrementalDecoder, which never emits half a character, and is held
back while it could still become a stop string.

Contract: contracts/py/tinyllm/infer/generate.pyi.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol, Sequence

import numpy as np

from tinyllm.infer.kvcache import KVCache
from tinyllm.infer.sample import SamplingParams, request_rng, sample, sampled_entropy
from tinyllm.tok.base import Tokenizer


class CausalLM(Protocol):
    def forward(self, ids: Any, positions: Optional[Any] = None, cache: Optional[Any] = None) -> Any: ...


def cache_dims(model: Any) -> tuple[int, int, int, int]:
    names = ("n_layers", "n_kv_heads", "d_head", "max_len")
    if all(hasattr(model, n) for n in names):
        return tuple(int(getattr(model, n)) for n in names)  # type: ignore[return-value]
    cfg = getattr(model, "config", None) or getattr(model, "cfg", None)
    hf = ("num_hidden_layers", "num_key_value_heads", "head_dim", "max_position_embeddings")
    if cfg is not None and all(hasattr(cfg, n) for n in hf):
        return tuple(int(getattr(cfg, n)) for n in hf)  # type: ignore[return-value]
    raise ValueError("the model exposes neither n_layers/n_kv_heads/d_head/max_len nor an HF-style config")


class IncrementalDecoder:
    def __init__(self, tok: Tokenizer) -> None:
        self.tok = tok
        self.ids: list[int] = []
        self.prefix = 0  # start of the context window (already emitted)
        self.read = 0  # end of what has been emitted

    def push(self, token_id: int) -> str:
        self.ids.append(int(token_id))
        before = self.tok.decode(self.ids[self.prefix : self.read])
        now = self.tok.decode(self.ids[self.prefix :])
        if len(now) > len(before) and not now.endswith("\ufffd"):
            # The window [prefix, read) gives the decoder its left context;
            # emit only what lies past it, then slide the window forward.
            self.prefix, self.read = self.read, len(self.ids)
            return now[len(before) :]
        return ""

    def flush(self) -> str:
        before = self.tok.decode(self.ids[self.prefix : self.read])
        now = self.tok.decode(self.ids[self.prefix :])
        self.ids, self.prefix, self.read = [], 0, 0
        return now[len(before) :]


@dataclass
class Generation:
    text: str
    ids: list[int]
    logprobs: list[float]
    timings: dict[str, float]
    stats: dict[str, float]
    finish_reason: str = "length"


def _logits_last(model: Any, ids: list[int], positions: np.ndarray, cache: Any) -> np.ndarray:
    """The model's float logits for the last position of `ids`."""
    out = model.forward(np.asarray([ids], dtype=np.int64), positions=positions, cache=cache)
    return np.asarray(getattr(out, "data", out))[0, -1]


def _holdback(text: str, stops: Sequence[str]) -> int:
    """Length of the longest suffix of text that is a proper prefix of a
    stop string: text that may still become a stop must not be streamed."""
    best = 0
    for s in stops:
        for n in range(min(len(s) - 1, len(text)), 0, -1):
            if text.endswith(s[:n]):
                best = max(best, n)
                break
    return best


def generate(
    model: Any,
    tok: Tokenizer,
    prompt: str,
    p: SamplingParams,
    cache: Any = "contiguous",
    kv_dtype: Any = np.float32,
    on_text: Optional[Callable[[str], None]] = None,
    eos_ids: Sequence[int] = (),
) -> Generation:
    p.validate()
    prompt_ids = [int(t) for t in tok.encode(prompt)]
    if not prompt_ids:
        raise ValueError("the prompt encodes to no tokens")
    n_layers, n_kv, d_head, model_max = cache_dims(model)
    if len(prompt_ids) > model_max:
        raise ValueError(f"prompt of {len(prompt_ids)} tokens exceeds max_len {model_max}")
    limit = min(model_max, len(prompt_ids) + p.max_tokens)
    if isinstance(cache, str):
        if cache == "none":
            kv = None
        elif cache == "contiguous":
            kv = KVCache(n_layers, n_kv, d_head, limit, batch=1, dtype=kv_dtype)
        elif cache == "paged":
            raise ValueError('cache="paged": pass the paged cache object from L8.3')
        else:
            raise ValueError(f"unknown cache mode {cache!r}")
    else:
        kv = cache
        if kv.seq_len() != 0:
            raise ValueError("a cache passed to generate must be empty")
    rng = request_rng(p.seed if p.seed is not None else 0)
    stops = [s for s in p.stop if s]
    dec = IncrementalDecoder(tok)
    eos = {int(e) for e in eos_ids}

    t0 = time.perf_counter()
    if kv is None:
        logits = _logits_last(model, prompt_ids, np.arange(len(prompt_ids)), None)
    else:
        logits = _logits_last(model, prompt_ids, kv.positions(len(prompt_ids)), kv)
    t1 = time.perf_counter()

    ids: list[int] = []
    lps: list[float] = []
    ents: list[float] = []
    text, sent, finish = "", 0, "length"
    ctx = list(prompt_ids)
    while len(ids) < p.max_tokens:
        tok_id, lp = sample(logits, p, ids, rng, prompt=prompt_ids)
        ents.append(sampled_entropy(logits, p, ids, prompt_ids))
        if tok_id in eos:
            finish = "stop"
            break
        ids.append(tok_id)
        lps.append(lp)
        ctx.append(tok_id)
        text += dec.push(tok_id)
        hit = min((i for i in (text.find(s) for s in stops) if i >= 0), default=-1)
        if hit >= 0:
            text, finish = text[:hit], "stop"
            break
        if on_text is not None:
            ready = len(text) - _holdback(text, stops)
            if ready > sent:
                on_text(text[sent:ready])
                sent = ready
        if len(ids) >= p.max_tokens or len(ctx) >= limit:
            break
        if kv is None:
            logits = _logits_last(model, ctx, np.arange(len(ctx)), None)
        else:
            logits = _logits_last(model, [tok_id], kv.positions(1), kv)
    if finish != "stop":
        text += dec.flush()
        hit = min((i for i in (text.find(s) for s in stops) if i >= 0), default=-1)
        if hit >= 0:
            text, finish = text[:hit], "stop"
    if on_text is not None and len(text) > sent:
        on_text(text[sent:])
    t2 = time.perf_counter()
    decode_s = t2 - t1
    return Generation(
        text=text,
        ids=ids,
        logprobs=lps,
        timings={"prefill_s": t1 - t0, "decode_s": decode_s, "total_s": t2 - t0},
        stats={
            "prompt_tokens": float(len(prompt_ids)),
            "completion_tokens": float(len(ids)),
            "mean_entropy": float(np.mean(ents)) if ents else 0.0,
            "tokens_per_s": len(ids) / decode_s if decode_s > 0 else 0.0,
        },
        finish_reason=finish,
    )
