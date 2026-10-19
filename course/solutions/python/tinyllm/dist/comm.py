"""Collectives over processes: point to point over pipes, ring reduce-scatter,
all-gather, all-reduce, broadcast, and a barrier (L11.2, optional).

The ring all-reduce moves 2 (p - 1) / p of the data per rank whatever the
number of ranks p: each rank sends p - 1 chunks to reduce and p - 1 chunks to
share, each 1/p of the array. Even ranks send first and odd ranks receive
first, so a step never waits on a cycle of blocked sends.

Contract: contracts/py/tinyllm/dist/comm.pyi.
"""

from __future__ import annotations

import multiprocessing as mp
import time
import traceback
from typing import Any, Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray


def chunk_bounds(n: int, world: int) -> list[tuple[int, int]]:
    if n < 0 or world < 1:
        raise ValueError(f"need n >= 0 and world >= 1, got n={n}, world={world}")
    q, extra = divmod(n, world)
    out, at = [], 0
    for r in range(world):
        size = q + (1 if r < extra else 0)
        out.append((at, at + size))
        at += size
    return out


class Comm:
    def __init__(self, rank: int, world: int, conns: dict, barrier: Any) -> None:
        self.rank, self.world = rank, world
        self._conns = conns  # peer rank -> this rank's end of the pipe to it
        self._barrier = barrier
        self.bytes_sent = 0

    def send(self, x: ArrayLike, dst: int) -> None:
        if dst == self.rank or not 0 <= dst < self.world:
            raise ValueError(f"rank {self.rank} cannot send to {dst} (world {self.world})")
        a = np.ascontiguousarray(x)
        self._conns[dst].send((a.dtype.str, a.shape, a.tobytes()))
        self.bytes_sent += a.nbytes

    def recv(self, src: int) -> NDArray:
        dt, shape, raw = self._conns[src].recv()
        return np.frombuffer(raw, dtype=np.dtype(dt)).reshape(shape).copy()

    def _exchange(self, out: NDArray) -> NDArray:
        """Send `out` to the right neighbour and receive from the left one,
        in the deadlock-free order (even ranks send first)."""
        right, left = (self.rank + 1) % self.world, (self.rank - 1) % self.world
        if self.rank % 2 == 0:
            self.send(out, right)
            return self.recv(left)
        got = self.recv(left)
        self.send(out, right)
        return got

    def reduce_scatter(self, x: ArrayLike) -> NDArray:
        flat = np.array(x).reshape(-1)  # a copy: the caller's array is never written
        p, r = self.world, self.rank
        b = chunk_bounds(flat.size, p)
        for s in range(p - 1):
            si, ri = (r - s - 1) % p, (r - s - 2) % p
            got = self._exchange(flat[b[si][0] : b[si][1]])
            flat[b[ri][0] : b[ri][1]] += got
        return flat[b[r][0] : b[r][1]].copy()

    def all_gather(self, x: ArrayLike) -> NDArray:
        p, r = self.world, self.rank
        chunks: list = [None] * p
        chunks[r] = np.array(x).reshape(-1)
        for s in range(p - 1):
            si, ri = (r - s) % p, (r - s - 1) % p
            chunks[ri] = self._exchange(chunks[si])
        return np.concatenate(chunks)

    def all_reduce(self, x: ArrayLike, op: str = "sum") -> NDArray:
        if op not in ("sum", "mean"):
            raise ValueError(f"op must be 'sum' or 'mean', got {op!r}")
        a = np.asarray(x)
        out = self.all_gather(self.reduce_scatter(a)).reshape(a.shape)
        return out / self.world if op == "mean" else out

    def broadcast(self, x: ArrayLike, src: int) -> NDArray:
        p, r = self.world, self.rank
        if p == 1:
            return np.array(x)
        right, left = (r + 1) % p, (r - 1) % p
        data = np.array(x) if r == src else self.recv(left)
        if right != src:
            self.send(data, right)
        return data

    def barrier(self) -> None:
        self._barrier.wait()


def _run(fn: Callable, rank: int, world: int, conns: dict, barrier: Any, results: Any, args: tuple) -> None:
    """The body of each spawned process."""
    try:
        out = fn(Comm(rank, world, conns, barrier), *args)
        results.put((rank, "ok", out))
    except BaseException:
        results.put((rank, "err", traceback.format_exc()))


def spawn(fn: Callable[..., Any], world: int, *args: Any, timeout: float = 60.0) -> list[Any]:
    if world < 1:
        raise ValueError(f"world must be >= 1, got {world}")
    ctx = mp.get_context("spawn")
    ends: list[dict] = [{} for _ in range(world)]
    for i in range(world):
        for j in range(i + 1, world):
            a, b = ctx.Pipe(duplex=True)
            ends[i][j], ends[j][i] = a, b
    barrier = ctx.Barrier(world)
    results = ctx.Queue()
    procs = [
        ctx.Process(target=_run, args=(fn, r, world, ends[r], barrier, results, args), daemon=True)
        for r in range(world)
    ]
    for p in procs:
        p.start()
    out: list[Any] = [None] * world
    deadline = time.monotonic() + timeout
    try:
        for _ in range(world):
            left = deadline - time.monotonic()
            try:
                rank, status, value = results.get(timeout=max(left, 0.01))
            except Exception:  # queue.Empty
                raise TimeoutError(f"spawn: ranks did not finish within {timeout} s (deadlock?)") from None
            if status == "err":
                raise RuntimeError(f"rank {rank} failed:\n{value}")
            out[rank] = value
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=5)
    return out
