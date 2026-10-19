"""KV cache for incremental decoding (L8.2).

Attention at position t needs the keys and values of every position <= t.
Without a cache, generating n tokens re-runs the model over 1, 2, ..., n
tokens: O(n^2) work for O(n) tokens. The cache keeps each layer's K and V
from earlier steps, so a decode step runs the model on one token and
attends to everything stored. Storage is preallocated to max_len (no
reallocation while decoding, the same reason rt.04 uses fixed blocks), each
layer keeps its own fill count, and the committed length is their minimum.

Contract: contracts/py/tinyllm/infer/kvcache.pyi.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from tinyllm.xfmr.masks import causal_mask


def _check_init(n_layers: int, max_len: int, batch: int, dtype: Any, sizes: list[int]) -> np.dtype:
    """ValueError for a size below 1 or a non-floating dtype; the dtype."""
    for name, v in [("n_layers", n_layers), ("max_len", max_len), ("batch", batch)] + [
        (f"size {i}", s) for i, s in enumerate(sizes)
    ]:
        if int(v) < 1:
            raise ValueError(f"{name} must be >= 1, got {v}")
    dt = np.dtype(dtype)
    if not np.issubdtype(dt, np.floating):
        raise ValueError(f"dtype must be a floating type, got {dt}")
    return dt


def _positions(seq_len: int, t_new: int) -> NDArray:
    if t_new < 1:
        raise ValueError(f"t_new must be >= 1, got {t_new}")
    return np.arange(seq_len, seq_len + t_new, dtype=np.int64)


def _mask(seq_len: int, t_new: int) -> NDArray:
    if t_new < 1:
        raise ValueError(f"t_new must be >= 1, got {t_new}")
    return causal_mask(t_new, seq_len + t_new, q_offset=seq_len)


def _append(cache: Any, layer: int, stores: list[NDArray], chunks: list[NDArray], shapes: list[tuple]) -> tuple:
    """Check, convert, and write the chunks at the layer's fill count (the
    time axis is the None in each shape), then return the stored prefix as
    float32. Nothing is written unless every check passes. The caller has
    checked `layer` (it indexed the stores with it)."""
    arrs = [np.asarray(c) for c in chunks]
    T = None
    for a, want in zip(arrs, shapes):
        ax = want.index(None)
        if a.ndim != len(want) or any(w is not None and s != w for s, w in zip(a.shape, want)):
            raise ValueError(f"chunk shape {a.shape} does not match {want} (None = new positions)")
        if T is None:
            T = a.shape[ax]
        elif a.shape[ax] != T:
            raise ValueError("chunks disagree on the number of new positions")
    if T < 1:
        raise ValueError("a chunk needs at least one position")
    start = cache._fill[layer]
    if start + T > cache.max_len:
        raise ValueError(f"layer {layer}: {start} + {T} positions exceed max_len {cache.max_len}")
    out = []
    for store, a, want in zip(stores, arrs, shapes):
        ax = want.index(None)
        idx = [slice(None)] * a.ndim
        idx[ax] = slice(start, start + T)
        store[tuple(idx)] = a.astype(cache.dtype)  # a copy, converted on the way in
        idx[ax] = slice(0, start + T)
        held = store[tuple(idx)]
        out.append(held if held.dtype == np.float32 else held.astype(np.float32))
    cache._fill[layer] = start + T
    return tuple(out)


class KVCache:
    def __init__(
        self,
        n_layers: int,
        n_kv_heads: int,
        d_head: int,
        max_len: int,
        batch: int = 1,
        dtype: Any = np.float32,
    ) -> None:
        self.dtype = _check_init(n_layers, max_len, batch, dtype, [n_kv_heads, d_head])
        self.n_layers, self.max_len, self.batch = int(n_layers), int(max_len), int(batch)
        self._fill = [0] * self.n_layers
        self.n_kv_heads, self.d_head = int(n_kv_heads), int(d_head)
        shape = (self.batch, self.n_kv_heads, self.max_len, self.d_head)
        self._k = [np.zeros(shape, dtype=self.dtype) for _ in range(self.n_layers)]
        self._v = [np.zeros(shape, dtype=self.dtype) for _ in range(self.n_layers)]

    def update(self, layer: int, k_new: NDArray, v_new: NDArray) -> tuple[NDArray, NDArray]:
        want = (self.batch, self.n_kv_heads, None, self.d_head)
        if not 0 <= layer < self.n_layers:
            raise ValueError(f"layer {layer} outside [0, {self.n_layers})")
        return _append(self, layer, [self._k[layer], self._v[layer]], [k_new, v_new], [want, want])

    def seq_len(self, layer: Optional[int] = None) -> int:
        return min(self._fill) if layer is None else self._fill[layer]

    def positions(self, t_new: int) -> NDArray:
        return _positions(self.seq_len(), t_new)

    def mask(self, t_new: int) -> NDArray:
        return _mask(self.seq_len(), t_new)

    def truncate(self, n: int) -> None:
        if not 0 <= n <= self.seq_len():
            raise ValueError(f"truncate({n}) needs 0 <= n <= seq_len = {self.seq_len()}")
        self._fill = [int(n)] * self.n_layers

    def nbytes(self) -> int:
        return 2 * self.n_layers * self.batch * self.n_kv_heads * self.max_len * self.d_head * self.dtype.itemsize


class LatentCache:
    def __init__(
        self,
        n_layers: int,
        kv_lora_rank: int,
        rope_dim: int,
        max_len: int,
        batch: int = 1,
        dtype: Any = np.float32,
    ) -> None:
        self.dtype = _check_init(n_layers, max_len, batch, dtype, [kv_lora_rank, rope_dim])
        self.n_layers, self.max_len, self.batch = int(n_layers), int(max_len), int(batch)
        self._fill = [0] * self.n_layers
        self.kv_lora_rank, self.rope_dim = int(kv_lora_rank), int(rope_dim)
        self._c = [np.zeros((self.batch, self.max_len, self.kv_lora_rank), dtype=self.dtype) for _ in range(self.n_layers)]
        self._kr = [np.zeros((self.batch, self.max_len, self.rope_dim), dtype=self.dtype) for _ in range(self.n_layers)]

    def update(self, layer: int, c_new: NDArray, k_rope_new: NDArray) -> tuple[NDArray, NDArray]:
        if not 0 <= layer < self.n_layers:
            raise ValueError(f"layer {layer} outside [0, {self.n_layers})")
        shapes = [(self.batch, None, self.kv_lora_rank), (self.batch, None, self.rope_dim)]
        return _append(self, layer, [self._c[layer], self._kr[layer]], [c_new, k_rope_new], shapes)

    def seq_len(self, layer: Optional[int] = None) -> int:
        return min(self._fill) if layer is None else self._fill[layer]

    def positions(self, t_new: int) -> NDArray:
        return _positions(self.seq_len(), t_new)

    def mask(self, t_new: int) -> NDArray:
        return _mask(self.seq_len(), t_new)

    def truncate(self, n: int) -> None:
        if not 0 <= n <= self.seq_len():
            raise ValueError(f"truncate({n}) needs 0 <= n <= seq_len = {self.seq_len()}")
        self._fill = [int(n)] * self.n_layers

    def nbytes(self) -> int:
        return self.n_layers * self.batch * self.max_len * (self.kv_lora_rank + self.rope_dim) * self.dtype.itemsize
