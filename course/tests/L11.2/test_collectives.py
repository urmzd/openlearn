"""Course tests for L11.2 (optional): collectives over processes
(tinyllm/dist/comm.py).

Rung R0 reading for the course tests; your own graded tests (rung R5) go in
python/tests/l11-2-comm/. Each test names why it exists (WHY), what kind of
check it is (KIND), the planted bugs it kills (CATCHES, mutants in
course/mutants/L11.2), and the chapter section it comes from.

The worked example of the chapter (section 3): three ranks holding
[1, 2, 3], [10, 20, 30], [100, 200, 300], reduced on a ring to
[111, 222, 333] with 32 bytes sent per rank.

Every test runs real processes through your spawn; the workers are
module-level functions (the "spawn" start method pickles them by name), and
every spawn has a timeout, so a deadlock fails the test instead of hanging.
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32
from tinyllm.dist.comm import chunk_bounds, spawn

T = 20.0  # seconds: a correct run takes well under one


def seed() -> int:
    return int(os.environ.get("OL_SEED", "0"))


def data(rank: int, shape, s: int) -> np.ndarray:
    """Rank-specific inputs from the frozen PCG32, the same in every process."""
    return PCG32(s, 300 + rank).normal_array(shape)


# --- workers (module level: spawn pickles them by name) -----------------------
#
# One "battery" run per world size exercises every collective in a fixed
# order on every rank; the tests below read their parts of its results. One
# spawn per world size keeps the suite fast; a planted bug that breaks the
# battery fails every test that reads it.


def w_battery(comm, s, tmpdir):
    out = {}
    if comm.world == 3:  # section 3's hand example
        x = np.array([1.0, 2.0, 3.0]) * 10.0**comm.rank
        before = comm.bytes_sent
        rs = comm.reduce_scatter(x)
        sent_rs = comm.bytes_sent - before
        y = comm.all_reduce(x)
        out["hand"] = (
            rs.tolist(),
            sent_rs,
            y.tolist(),
            comm.bytes_sent - before - sent_rs,
        )
    for op in ("sum", "mean"):
        x = data(comm.rank, (5, 7), s)
        keep = x.copy()
        out[op] = (comm.all_reduce(x, op=op), bool(np.array_equal(x, keep)))
    before = comm.bytes_sent
    comm.all_reduce(data(comm.rank, (120,), s))
    out["bytes120"] = comm.bytes_sent - before
    sizes = [b - a for a, b in chunk_bounds(11, comm.world)]
    out["rsag"] = (
        comm.reduce_scatter(data(comm.rank, (11,), s)),
        comm.all_gather(np.full(sizes[comm.rank], float(comm.rank))),
    )
    bc = []
    for src in range(comm.world):
        before = comm.bytes_sent
        x = (
            np.arange(5, dtype=np.float64) + 100 * src
            if comm.rank == src
            else np.zeros(5)
        )
        bc.append((comm.broadcast(x, src).tolist(), comm.bytes_sent - before))
    # a collective right after: a stray message from broadcast would corrupt it
    bc.append(comm.all_reduce(np.ones(4) * (comm.rank + 1)).tolist())
    out["bcast"] = bc
    if comm.rank == 0:
        comm.send(np.arange(6, dtype=np.int32).reshape(2, 3), 1)
        comm.send(np.array([2.5]), 1)
        errs = []
        for bad in (0, comm.world, -1):
            try:
                comm.send(np.zeros(1), bad)
            except ValueError:
                errs.append(bad)
        out["p2p"] = errs
    elif comm.rank == 1:
        a, b = comm.recv(0), comm.recv(0)
        a[0, 0] = 99  # the received array is the caller's to write
        out["p2p"] = (a.dtype.str, a.shape, a.tolist(), b.tolist())
    if comm.rank == 0:
        time.sleep(0.15)
    Path(tmpdir, str(comm.rank)).touch()
    comm.barrier()
    out["barrier"] = sorted(os.listdir(tmpdir))
    return out


def w_big(comm, n):
    return float(comm.all_reduce(np.ones(n))[0])


def w_fail(comm):
    if comm.rank == 1:
        raise ArithmeticError("planted failure on rank 1")
    comm.barrier()
    return comm.rank


_RUNS: dict = {}


def battery(world: int) -> list:
    """The battery's per-rank results for this world size, run once per process."""
    if world not in _RUNS:
        d = tempfile.mkdtemp(prefix="l112-")
        atexit.register(shutil.rmtree, d, True)
        try:
            _RUNS[world] = ("ok", spawn(w_battery, world, seed(), d, timeout=T))
        except (
            Exception
        ) as e:  # keep the failure: every test reading it fails the same way
            _RUNS[world] = ("err", e)
    status, value = _RUNS[world]
    if status == "err":
        raise value
    return value


# --- tests --------------------------------------------------------------------


def test_hand_example_ring_allreduce():
    # WHY: section 3 by hand. Three ranks, one entry per chunk. Reduce-scatter
    #      leaves rank r holding chunk r of the sum: [111], [222], [333], after
    #      2 steps of one 8-byte entry each (16 bytes). The all-gather that
    #      completes the all-reduce sends 2 more entries: 32 bytes per rank,
    #      2 (p - 1) / p * n * 8 with p = n = 3.
    # KIND: unit
    # CATCHES: s01, s02, s03, s06, s08, m02
    # CHAPTER: L11.2 section 3, Worked example by hand
    out = [r["hand"] for r in battery(3)]
    assert [o[0] for o in out] == [[111.0], [222.0], [333.0]]
    assert [o[1] for o in out] == [16, 16, 16]
    assert [o[2] for o in out] == [[111.0, 222.0, 333.0]] * 3
    assert [o[3] for o in out] == [32, 32, 32]


def test_chunk_bounds():
    # WHY: every rank must agree on who owns which entries; the split is
    #      numpy.array_split's (the first n % p chunks one larger), so a
    #      ZeRO shard (L11.3) and an all-gather line up.
    # KIND: unit
    # CATCHES: s07, m02
    # CHAPTER: L11.2 section 4, The interface
    assert chunk_bounds(10, 4) == [(0, 3), (3, 6), (6, 8), (8, 10)]
    assert chunk_bounds(2, 3) == [(0, 1), (1, 2), (2, 2)]
    for n in range(0, 23):
        for p in range(1, 6):
            want = [len(c) for c in np.array_split(np.arange(n), p)]
            got = chunk_bounds(n, p)
            assert (
                [b - a for a, b in got] == want and got[0][0] == 0 and got[-1][1] == n
            )
    for bad in ((-1, 2), (3, 0)):
        with pytest.raises(ValueError):
            chunk_bounds(*bad)


@pytest.mark.parametrize("world", [2, 3, 4])
def test_allreduce_matches_numpy_sum(world):
    # WHY: the ring adds in a different order than numpy, so the sum agrees
    #      to float64 rounding, not bit for bit; but all ranks must hold the
    #      same bits (they copy one reduced chunk each), or DDP replicas drift
    #      apart. 35 entries do not divide evenly by 2, 3, or 4. The caller's
    #      array is not written, and "mean" divides by the world size.
    # KIND: differential
    # CATCHES: s01, s02, s03, s04, s06, m01
    # CHAPTER: L11.2 section 2, Principles
    want = sum(data(r, (5, 7), seed()) for r in range(world))
    out = battery(world)
    for op, scale in (("sum", 1.0), ("mean", 1.0 / world)):
        for r in out:
            y, untouched = r[op]
            assert y.shape == (5, 7)
            assert_close(y, want * scale, rtol=1e-12, atol=1e-12)
            assert untouched, "all_reduce wrote into the caller's array"
            assert np.array_equal(y, out[0][op][0]), "ranks disagree bitwise"


def test_bytes_moved_is_2_p_minus_1_over_p():
    # WHY: the ring's selling point: each rank sends 2 (p - 1) / p of the
    #      array whatever p is, so all-reduce time stays flat as ranks are
    #      added (a naive gather to rank 0 sends (p - 1) n into one link).
    #      120 float64 entries divide by 2, 3, and 4.
    # KIND: property
    # CATCHES: s02, s03, s08, m02
    # CHAPTER: L11.2 section 2, Principles
    for world in (2, 3, 4):
        assert [r["bytes120"] for r in battery(world)] == [
            2 * (world - 1) * 120 * 8 // world
        ] * world


def test_reduce_scatter_and_all_gather():
    # WHY: ZeRO (L11.3) uses the halves of all-reduce on their own:
    #      reduce-scatter hands rank r chunk r of the sum, and all-gather
    #      concatenates chunks of unequal sizes in rank order.
    # KIND: unit
    # CATCHES: s01, s02, s03
    # CHAPTER: L11.2 section 4, The interface
    n, world = 11, 3
    total = sum(data(r, (n,), seed()) for r in range(world))
    b = chunk_bounds(n, world)
    want_g = np.concatenate([np.full(e - a, float(r)) for r, (a, e) in enumerate(b)])
    for r, res in enumerate(battery(world)):
        mine, gathered = res["rsag"]
        assert_close(mine, total[b[r][0] : b[r][1]], rtol=1e-12, atol=1e-12)
        assert np.array_equal(gathered, want_g)


def test_broadcast_from_every_src():
    # WHY: DDP starts every replica from rank 0's weights. Each rank but the
    #      one just before the source forwards the array once, and nothing
    #      extra is left in a pipe: the all-reduce right after still works.
    # KIND: unit
    # CATCHES: s08, s09
    # CHAPTER: L11.2 section 4, The interface
    world, n = 4, 5
    for r, res in enumerate(battery(world)):
        bc = res["bcast"]
        for src in range(world):
            vals, sent = bc[src]
            assert vals == (np.arange(n) + 100.0 * src).tolist()
            assert sent == (0 if (r + 1) % world == src else n * 8)
        assert bc[world] == [10.0] * 4


def test_large_messages_do_not_deadlock():
    # WHY: a pipe holds only a few kilobytes; a 1 MiB send blocks until the
    #      neighbour reads. If every rank sends first, every rank waits on a
    #      blocked neighbour forever. Even ranks send first, odd ranks receive
    #      first; a ring of 3 has two even ranks next to each other (2 and
    #      0), the case a naive alternation gets wrong. A correct run takes a
    #      fraction of a second; 8 s only bounds the planted deadlock.
    # KIND: boundary
    # CATCHES: s05
    # CHAPTER: L11.2 section 5, Pitfalls
    assert spawn(w_big, 3, 1 << 17, timeout=8.0) == [3.0] * 3


def test_send_recv_point_to_point():
    # WHY: the layer every collective is built on: messages arrive in order
    #      with their dtype and shape, the receiver owns a writable copy, and
    #      a send to yourself or to a rank that does not exist is an error.
    # KIND: unit
    # CATCHES: s12
    # CHAPTER: L11.2 section 4, The interface
    out = battery(2)
    assert out[0]["p2p"] == [0, 2, -1]
    dt, shape, a, b = out[1]["p2p"]
    assert np.dtype(dt) == np.int32 and tuple(shape) == (2, 3)
    assert a == [[99, 1, 2], [3, 4, 5]] and b == [2.5]


def test_rank_error_propagates():
    # WHY: a rank that crashes must stop the run with its traceback, not
    #      leave the others blocked in a barrier until the timeout.
    # KIND: boundary
    # CATCHES: s10
    # CHAPTER: L11.2 section 5, Pitfalls
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="planted failure on rank 1"):
        spawn(w_fail, 3, timeout=T)
    assert time.monotonic() - t0 < T / 2
    with pytest.raises(ValueError):
        spawn(w_fail, 0)


def test_barrier():
    # WHY: after barrier() returns on any rank, every rank has reached it:
    #      rank 0 arrives 0.15 s late, and still every rank sees all files.
    # KIND: unit
    # CATCHES: s11
    # CHAPTER: L11.2 section 4, The interface
    assert [r["barrier"] for r in battery(3)] == [["0", "1", "2"]] * 3
