"""Sliding windows, StreamingLLM sink tokens, and learned sinks (L7.7).

Full causal attention reads every earlier token, so the KV cache grows
without bound. A sliding window reads only the last W tokens; StreamingLLM
adds back the first few tokens, which models use as a place to park
attention they do not want to spend; learned sinks give each head an extra
logit attached to nothing. With a window and sinks, the cache of a stream of
any length holds a constant number of keys.

Contract: contracts/py/tinyllm/modern/window.pyi.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tinyllm.num.stable import logsumexp
from tinyllm.xfmr.masks import causal_mask, sliding_window_mask


def _int(name: str, v: Any, lo: int) -> int:
    if isinstance(v, bool) or int(v) != v or v < lo:
        raise ValueError(f"{name} must be an integer >= {lo}, got {v!r}")
    return int(v)


def sink_window_mask(Tq: int, Tk: int, window: Optional[int], n_sink: int = 0, q_offset: int = 0) -> NDArray:
    Tq, Tk, q_offset = _int("Tq", Tq, 1), _int("Tk", Tk, 1), _int("q_offset", q_offset, 0)
    n_sink = _int("n_sink", n_sink, 0)
    if window is None:
        return causal_mask(Tq, Tk, q_offset)
    window = _int("window", window, 1)
    recent = sliding_window_mask(Tq, Tk, window, q_offset)
    # Sink keys are exempt from the window, never from causality.
    sinks = causal_mask(Tq, Tk, q_offset) & (np.arange(Tk) < n_sink)[None, :]
    return recent | sinks


def windowed_attention(
    q: ArrayLike,
    k: ArrayLike,
    v: ArrayLike,
    window: Optional[int] = None,
    n_sink: int = 0,
    sink_logits: Optional[ArrayLike] = None,
    q_offset: int = 0,
    scale: Optional[float] = None,
) -> tuple[NDArray, NDArray]:
    q, k, v = (np.asarray(a, dtype=np.float64) for a in (q, k, v))
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(f"want 4-D q, k, v, got {q.shape}, {k.shape}, {v.shape}")
    B, H, Tq, dh = q.shape
    Hkv, Tk = k.shape[1], k.shape[2]
    if k.shape[0] != B or v.shape[:3] != k.shape[:3] or k.shape[3] != dh or H % Hkv:
        raise ValueError(f"shapes disagree: q {q.shape}, k {k.shape}, v {v.shape}")
    scale = dh**-0.5 if scale is None else float(scale)
    rep = np.arange(H) // (H // Hkv)  # query head h reads kv head h // n_rep
    s = np.einsum("bhtd,bhkd->bhtk", q, k[:, rep]) * scale
    vis = sink_window_mask(Tq, Tk, window, n_sink, q_offset)
    s = np.where(vis, s, -np.inf)
    if sink_logits is not None:
        sl = np.asarray(sink_logits, dtype=np.float64)
        if sl.shape != (H,):
            raise ValueError(f"sink_logits must be [H] = [{H}], got {sl.shape}")
        col = np.broadcast_to(sl[None, :, None, None], (B, H, Tq, 1))
        lse = logsumexp(np.concatenate([s, col], axis=-1), axis=-1)
    else:
        lse = logsumexp(s, axis=-1)
    # p_j = exp(s_j - lse): the sink takes its share of the denominator and
    # gives back no value.
    with np.errstate(invalid="ignore"):
        p = np.where(np.isfinite(lse)[..., None] & vis, np.exp(s - lse[..., None]), 0.0)
    out = np.einsum("bhtk,bhkd->bhtd", p, v[:, rep])
    return out, lse


class SinkWindowCache:
    def __init__(self, n_sink: int, window: int) -> None:
        self.n_sink = _int("n_sink", n_sink, 0)
        self.window = _int("window", window, 1)
        self._k: dict[int, NDArray] = {}
        self._v: dict[int, NDArray] = {}
        self._pos: dict[int, NDArray] = {}  # absolute positions of what is held
        self._seen: dict[int, int] = {}
        self._last: dict[int, NDArray] = {}

    def update(self, layer: int, k_new: NDArray, v_new: NDArray) -> tuple[NDArray, NDArray]:
        k_new, v_new = np.array(k_new, copy=True), np.array(v_new, copy=True)
        if k_new.ndim != 4 or k_new.shape[:3] != v_new.shape[:3]:
            raise ValueError(f"k and v must be [B, Hkv, T, d], got {k_new.shape} and {v_new.shape}")
        seen, T = self._seen.get(layer, 0), k_new.shape[2]
        new_pos = np.arange(seen, seen + T, dtype=np.int64)
        if layer in self._k:
            old = self._k[layer]
            if old.shape[:2] != k_new.shape[:2] or old.shape[3] != k_new.shape[3] or self._v[layer].shape[3] != v_new.shape[3]:
                raise ValueError(f"layer {layer}: chunk {k_new.shape} does not extend {old.shape}")
            k = np.concatenate([old, k_new], axis=2)
            v = np.concatenate([self._v[layer], v_new], axis=2)
            pos = np.concatenate([self._pos[layer], new_pos])
        else:
            k, v, pos = k_new, v_new, new_pos
        self._seen[layer] = seen + T
        self._last[layer] = pos
        # Evict AFTER returning what this chunk needs: keep the sinks and the
        # window - 1 most recent positions, which is what the next query needs
        # besides itself.
        keep = (pos < self.n_sink) | (pos >= seen + T - (self.window - 1))
        self._k[layer], self._v[layer], self._pos[layer] = k[:, :, keep], v[:, :, keep], pos[keep]
        return k, v

    def seq_len(self, layer: int = 0) -> int:
        return self._seen.get(layer, 0)

    def held(self, layer: int = 0) -> int:
        return int(self._pos[layer].size) if layer in self._pos else 0

    def positions(self, layer: int = 0) -> NDArray:
        return self._last.get(layer, np.zeros(0, dtype=np.int64)).copy()

    def chunk_mask(self, layer: int, T: int) -> NDArray:
        T = _int("T", T, 1)
        seen = self._seen.get(layer, 0)
        held = self._pos.get(layer, np.zeros(0, dtype=np.int64))
        keys = np.concatenate([held, np.arange(seen, seen + T, dtype=np.int64)])
        p = seen + np.arange(T)[:, None]
        j = keys[None, :]
        return (j <= p) & ((j < self.n_sink) | (p - j < self.window))


def attention_mask(cache: Any, layer: int, T: int, window: Optional[int], n_sink: int = 0) -> Optional[NDArray]:
    if window is None:
        return None
    if cache is None:
        return sink_window_mask(T, T, window, n_sink)
    if isinstance(cache, SinkWindowCache):
        return cache.chunk_mask(layer, T)
    if not hasattr(cache, "seq_len"):
        raise ValueError(f"{type(cache).__name__} has no seq_len(layer): cannot place the chunk's queries")
    s = int(cache.seq_len(layer))
    return sink_window_mask(T, s + T, window, n_sink, s)
