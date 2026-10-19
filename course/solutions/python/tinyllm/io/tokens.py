"""The token-stream reader (L0.6): formats/tokens-bin.md, memory-mapped.

A shard is llm.c's `.bin`: a 1024-byte header of 256 little-endian int32
(magic 20240520, version, n_tokens, vocab_size) and then the ids, uint16 for
version 1 and uint32 for version 2. TokenStream cuts training windows of
seq_len + 1 tokens out of the shards: the inputs are the first seq_len, the
targets the same window shifted by one.

Order: the shards are read in turn, and each pass over a shard starts at a
random phase in [0, seq_len) drawn from the PCG32 (so window boundaries move
from pass to pass), then takes non-overlapping windows seq_len tokens apart
until the next one would run past the end. The whole position is
(shard, offset of the next window, generator state): that is the cursor a
checkpoint saves, and restoring it continues the exact same stream.

Contract: contracts/py/tinyllm/io/tokens.pyi.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

MAGIC = 20240520
HEADER_BYTES = 1024
_WIDTH = {1: np.dtype("<u2"), 2: np.dtype("<u4")}


def read_tokens_header(path: str) -> dict[str, int]:
    size = os.path.getsize(path)
    if size < HEADER_BYTES:
        raise ValueError(f"{path}: {size} bytes is shorter than the 1024-byte header")
    head = np.fromfile(path, dtype="<i4", count=256)
    magic, version, n, vocab = (int(x) for x in head[:4])
    if magic != MAGIC:
        raise ValueError(f"{path}: magic {magic}, a tokens .bin starts with {MAGIC}")
    if version not in _WIDTH:
        raise ValueError(f"{path}: unknown version {version} (1: uint16 ids, 2: uint32 ids)")
    if n < 0 or vocab < 0:
        raise ValueError(f"{path}: negative n_tokens {n} or vocab_size {vocab}")
    want = HEADER_BYTES + n * _WIDTH[version].itemsize
    if size != want:
        raise ValueError(f"{path}: {size} bytes, but {n} version-{version} ids need {want}")
    return {"version": version, "n_tokens": n, "vocab_size": vocab}


def open_tokens(path: str) -> NDArray:
    h = read_tokens_header(path)
    if h["n_tokens"] == 0:
        return np.zeros(0, dtype=_WIDTH[h["version"]])
    ids = np.memmap(path, dtype=_WIDTH[h["version"]], mode="r", offset=HEADER_BYTES, shape=(h["n_tokens"],))
    if h["vocab_size"]:
        # The header promises every id is below vocab_size: check it once,
        # here, not as an out-of-range embedding lookup mid-training.
        top = int(ids.max())
        if top >= h["vocab_size"]:
            raise ValueError(f"{path}: id {top} is not below vocab_size {h['vocab_size']}")
    return ids


class TokenStream:
    def __init__(
        self,
        shards: Sequence[str],
        seq_len: int,
        batch: int,
        rng: Any,
        vocab_size: Optional[int] = None,
    ) -> None:
        if isinstance(shards, (str, bytes)) or not shards:
            raise ValueError("shards must be a non-empty list of paths")
        if seq_len < 1 or batch < 1:
            raise ValueError(f"seq_len and batch must be >= 1, got {seq_len} and {batch}")
        self.seq_len, self.batch, self.rng = int(seq_len), int(batch), rng
        self.shards = [str(s) for s in shards]
        self.ids = []
        for s in self.shards:
            h = read_tokens_header(s)
            if vocab_size is not None and h["vocab_size"] not in (0, vocab_size):
                raise ValueError(f"{s}: vocab_size {h['vocab_size']}, the model has {vocab_size}")
            ids = open_tokens(s)
            if ids.size < self.seq_len + 1:
                raise ValueError(f"{s}: {ids.size} tokens cannot hold one window of {self.seq_len + 1}")
            if vocab_size is not None and h["vocab_size"] == 0 and int(ids.max()) >= vocab_size:
                raise ValueError(f"{s}: id {int(ids.max())} is not below vocab_size {vocab_size}")
            self.ids.append(ids)
        self.shard = 0
        self.offset = self._phase(0)

    def _phase(self, shard: int) -> int:
        """A random start in [0, seq_len) for a new pass over `shard`, limited
        to the starts that leave room for one window."""
        starts = self.ids[shard].size - self.seq_len  # valid window starts: 0 .. n - T - 1
        return int(self.rng.below(min(self.seq_len, starts)))

    def next_batch(self) -> tuple[NDArray, NDArray]:
        T = self.seq_len
        rows = np.empty((self.batch, T + 1), dtype=np.int64)
        for b in range(self.batch):
            if self.offset + T + 1 > self.ids[self.shard].size:
                # This pass is done: the next shard (wrapping), at a new phase.
                self.shard = (self.shard + 1) % len(self.ids)
                self.offset = self._phase(self.shard)
            rows[b] = self.ids[self.shard][self.offset : self.offset + T + 1]
            self.offset += T  # windows share their boundary token: target of one, input of the next
        return rows[:, :-1].copy(), rows[:, 1:].copy()

    def cursor(self) -> dict:
        return {"shard": self.shard, "offset": self.offset, "rng": tuple(self.rng.state())}

    def restore(self, cursor: dict) -> None:
        shard, offset = int(cursor["shard"]), int(cursor["offset"])
        if not 0 <= shard < len(self.ids):
            raise ValueError(f"cursor shard {shard} is not one of the {len(self.ids)} shards")
        if not 0 <= offset <= self.ids[shard].size:
            raise ValueError(f"cursor offset {offset} is outside shard {shard} ({self.ids[shard].size} tokens)")
        self.rng.set_state(tuple(cursor["rng"]))
        self.shard, self.offset = shard, offset
