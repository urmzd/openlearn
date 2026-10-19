"""Pure-Python paged KV cache (L8.3).

The page table maps sequence positions to fixed-size NumPy blocks. Blocks
store float16 K and V arrays; forks share references until a write needs a
copy, and allocations are checked before any state changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


class OutOfBlocks(RuntimeError):
    """The cache has too few free blocks for an append or fork."""


@dataclass
class _Block:
    k: NDArray
    v: NDArray
    refs: int = 1


class PagedKVCache:
    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        n_layers: int,
        n_kv_heads: int,
        d_head: int,
    ) -> None:
        dims = (num_blocks, block_size, n_layers, n_kv_heads, d_head)
        if any(int(x) < 1 for x in dims):
            raise ValueError(f"PagedKVCache: every size must be at least 1, got {dims}")
        (
            self.num_blocks,
            self.block_size,
            self.n_layers,
            self.n_kv_heads,
            self.d_head,
        ) = map(int, dims)
        self._blocks: list[_Block | None] = [None] * self.num_blocks
        self._free = list(range(self.num_blocks - 1, -1, -1))
        self._tables: dict[int, list[int]] = {}
        self._lens: dict[int, list[int]] = {}
        self._closed = False

    def add_seq(self, seq_id: int) -> None:
        self._ensure_open()
        if seq_id in self._tables:
            raise ValueError(f"sequence {seq_id} already exists")
        self._tables[seq_id] = []
        self._lens[seq_id] = [0] * self.n_layers

    def fork(self, parent: int, child: int) -> None:
        table = self._table(parent)
        if child in self._tables:
            raise ValueError(f"sequence {child} already exists")
        for block_id in table:
            block = self._block(block_id)
            block.refs += 1
        self._tables[child] = list(table)
        self._lens[child] = list(self._lens[parent])

    def free(self, seq_id: int) -> None:
        table = self._table(seq_id)
        for block_id in table:
            self._unref(block_id)
        del self._tables[seq_id]
        del self._lens[seq_id]

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("PagedKVCache is closed")

    def _table(self, seq_id: int) -> list[int]:
        self._ensure_open()
        try:
            return self._tables[seq_id]
        except KeyError:
            raise KeyError(f"unknown sequence {seq_id}") from None

    def _layer(self, layer: int) -> int:
        if not 0 <= int(layer) < self.n_layers:
            raise ValueError(f"layer {layer} is out of range 0..{self.n_layers - 1}")
        return int(layer)

    def _block(self, block_id: int) -> _Block:
        block = self._blocks[block_id]
        if block is None:
            raise RuntimeError(f"block {block_id} is not allocated")
        return block

    def _alloc_id(self) -> int:
        block_id = self._free.pop()
        shape = (self.n_layers, self.n_kv_heads, self.block_size, self.d_head)
        self._blocks[block_id] = _Block(
            np.zeros(shape, dtype=np.float16), np.zeros(shape, dtype=np.float16)
        )
        return block_id

    def _unref(self, block_id: int) -> None:
        block = self._block(block_id)
        block.refs -= 1
        if block.refs == 0:
            self._blocks[block_id] = None
            self._free.append(block_id)

    def append(self, seq_id: int, layer: int, k: ArrayLike, v: ArrayLike) -> None:
        table = self._table(seq_id)
        layer = self._layer(layer)
        k16 = np.asarray(k, dtype=np.float16)
        v16 = np.asarray(v, dtype=np.float16)
        want = (self.n_kv_heads, k16.shape[1] if k16.ndim == 3 else -1, self.d_head)
        if k16.ndim != 3 or k16.shape != want or v16.shape != want or want[1] < 1:
            raise ValueError(
                f"append: k and v must be [n_kv_heads={self.n_kv_heads}, T >= 1, d_head={self.d_head}], "
                f"got {k16.shape} and {v16.shape}"
            )
        start = self._lens[seq_id][layer]
        end = start + want[1]
        first, last = start // self.block_size, (end - 1) // self.block_size
        shared = [
            i for i in range(first, min(last + 1, len(table)))
            if self._block(table[i]).refs > 1
        ]
        new_count = max(0, last + 1 - len(table))
        if len(self._free) < len(shared) + new_count:
            raise OutOfBlocks("the KV cache has no free block left")

        # Reserve every required block before changing references or tables.
        replacements: dict[int, int] = {}
        for i in shared:
            old_id = table[i]
            old = self._block(old_id)
            new_id = self._alloc_id()
            clone = self._block(new_id)
            clone.k[...] = old.k
            clone.v[...] = old.v
            replacements[i] = new_id
        allocated = [self._alloc_id() for _ in range(new_count)]
        for i, new_id in replacements.items():
            self._unref(table[i])
            table[i] = new_id
        table.extend(allocated)

        pos = start
        while pos < end:
            block_index, slot = divmod(pos, self.block_size)
            count = min(self.block_size - slot, end - pos)
            src = slice(pos - start, pos - start + count)
            block = self._block(table[block_index])
            block.k[layer, :, slot : slot + count, :] = k16[:, src, :]
            block.v[layer, :, slot : slot + count, :] = v16[:, src, :]
            pos += count
        self._lens[seq_id][layer] = end

    def block_table(self, seq_id: int) -> NDArray:
        return np.array(self._table(seq_id), dtype=np.int32)

    def seq_len(self, seq_id: int, layer: int = 0) -> int:
        self._table(seq_id)
        return self._lens[seq_id][self._layer(layer)]

    def gather(self, seq_id: int, layer: int) -> tuple[NDArray, NDArray]:
        table = self._table(seq_id)
        layer = self._layer(layer)
        n = self._lens[seq_id][layer]
        k = np.empty((self.n_kv_heads, n, self.d_head), dtype=np.float16)
        v = np.empty_like(k)
        for block_index in range((n + self.block_size - 1) // self.block_size):
            start = block_index * self.block_size
            count = min(self.block_size, n - start)
            block = self._block(table[block_index])
            k[:, start : start + count] = block.k[layer, :, :count]
            v[:, start : start + count] = block.v[layer, :, :count]
        return k, v

    def stats(self) -> dict[str, int]:
        used = self.num_blocks - len(self._free)
        return {"free": len(self._free), "used": used, "cached": 0, "evictions": 0}

    def num_free_blocks(self) -> int:
        return len(self._free)

    def close(self) -> None:
        if not self._closed:
            for seq_id in list(self._tables):
                self.free(seq_id)
            self._closed = True

    def __enter__(self) -> "PagedKVCache":
        self._ensure_open()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001, interpreter shutdown
            pass
