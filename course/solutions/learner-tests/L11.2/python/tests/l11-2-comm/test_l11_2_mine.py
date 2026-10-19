"""My tests for L11.2 (rung R5). Oracles: numpy sums of the same inputs,
array_split for the chunks, and byte counts worked out from the ring. They
import only the contract; workers are module-level for the spawn method, and
one run per world size feeds most tests."""

import atexit
import os
import shutil
import tempfile
import time

import numpy as np
import pytest
from tinyllm.dist.comm import chunk_bounds, spawn


def mk(rank, n):
    return np.cos(np.arange(n) * (rank + 1.0)) + rank


def w_all(comm, d):
    out = {}
    x = mk(comm.rank, 24)
    keep = x.copy()
    out["sum"] = comm.all_reduce(x, "sum")
    out["sent"] = comm.bytes_sent
    out["kept"] = bool(np.array_equal(keep, x))
    out["mean"] = comm.all_reduce(x, "mean")
    out["halves"] = (comm.reduce_scatter(mk(comm.rank, 7)), comm.all_gather(np.arange(comm.rank + 1.0)))
    before = comm.bytes_sent
    a = comm.broadcast(np.array([7.0, 8.0]) if comm.rank == comm.world - 1 else np.zeros(2), comm.world - 1)
    out["bcast"] = (a.tolist(), comm.all_reduce(np.array([1.0])).tolist(), comm.bytes_sent - before)
    if comm.rank == 1:
        comm.send(np.array([[1, 2]], dtype=np.int16), 0)
    elif comm.rank == 0:
        got = comm.recv(1)
        got += 1
        out["p2p"] = (got.dtype.str, got.tolist())
    if comm.rank == 1:
        time.sleep(0.15)
    open(os.path.join(d, f"r{comm.rank}"), "w").close()
    comm.barrier()
    out["bar"] = len(os.listdir(d))
    return out


def w_big(comm):
    return comm.all_reduce(np.ones(1 << 17)).sum()


def w_boom(comm):
    if comm.rank == 0:
        raise KeyError("rank zero exploded")
    comm.barrier()


_R = {}


def run(p):
    if p not in _R:
        d = tempfile.mkdtemp(prefix="mine-l112-")
        atexit.register(shutil.rmtree, d, True)
        try:
            _R[p] = (True, spawn(w_all, p, d, timeout=20))
        except Exception as e:
            _R[p] = (False, e)
    ok, v = _R[p]
    if not ok:
        raise v
    return v


@pytest.mark.parametrize("p", [2, 3, 4])
def test_sum_mean_and_bytes(p):
    want = sum(mk(r, 24) for r in range(p))
    out = run(p)
    for o in out:
        np.testing.assert_allclose(o["sum"], want, rtol=1e-12, atol=1e-12)
        assert np.array_equal(o["sum"], out[0]["sum"]) and o["kept"]
        assert o["sent"] == 2 * (p - 1) * 24 * 8 // p
        np.testing.assert_allclose(o["mean"], want / p, rtol=1e-12, atol=1e-12)


def test_three_way_halves():
    tot = sum(mk(r, 7) for r in range(3))
    for r, o in enumerate(run(3)):
        mine, gathered = o["halves"]
        a, b = chunk_bounds(7, 3)[r]
        np.testing.assert_allclose(mine, tot[a:b], rtol=1e-12)
        assert gathered.tolist() == [0.0, 0.0, 1.0, 0.0, 1.0, 2.0]


def test_chunks():
    for n in range(12):
        for p in range(1, 5):
            assert [b - a for a, b in chunk_bounds(n, p)] == [len(c) for c in np.array_split(np.arange(n), p)]


def test_broadcast_then_collective():
    out = run(3)
    assert [o["bcast"][0] for o in out] == [[7.0, 8.0]] * 3
    assert [o["bcast"][1] for o in out] == [[3.0]] * 3
    # broadcast from 2: ranks 2 and 0 send 16 bytes, rank 1 (just before 2)
    # nothing; the 1-entry all-reduce then sends 8, 16, 8 (chunks of 1, 0, 0)
    assert [o["bcast"][2] for o in out] == [24, 16, 24]


def test_big_ring_finishes():
    assert spawn(w_big, 3, timeout=8) == [3.0 * (1 << 17)] * 3


def test_received_array_is_writable():
    assert run(2)[0]["p2p"] == ("<i2", [[2, 3]])


def test_failure_is_raised():
    with pytest.raises(RuntimeError, match="rank zero exploded"):
        spawn(w_boom, 2, timeout=15)


def test_barrier_waits():
    assert [o["bar"] for o in run(3)] == [3, 3, 3]
