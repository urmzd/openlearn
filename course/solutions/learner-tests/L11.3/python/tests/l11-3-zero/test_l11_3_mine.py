"""My tests for L11.3 (rung R5). Oracle: the same model trained in one
process on the whole batch with the same optimizer, plus byte counts worked
out by hand from the chunk sizes. They import only the contract."""

import numpy as np
import pytest
from tinyllm.autograd.tensor import Tensor
from tinyllm.dist.comm import spawn
from tinyllm.dist.zero import DDP, ZeroOptimizer
from tinyllm.nn.module import Module
from tinyllm.optim.adamw import AdamW
from tinyllm.optim.sgd import SGD


class Quad(Module):
    """Two parameters, 7 + 5 = 12 entries; loss sum(c * (w - t)^2) over both."""

    def __init__(self, shift=0.0):
        super().__init__()
        self.u = Tensor(np.linspace(-1, 1, 7) + shift, requires_grad=True, dtype=np.float64)
        self.v = Tensor(np.linspace(2, 0, 5).reshape(5, 1) + shift, requires_grad=True, dtype=np.float64)


def loss(m, k):
    c = np.arange(1.0, 8.0) * (k + 1)
    t = np.full(7, float(k))
    du, dv = m.u - t, m.v - 0.5 * k
    a = (du * du * c)[None, :] @ np.ones((7, 1))
    b = (dv * dv).__getitem__((slice(None), 0))[None, :] @ np.ones((5, 1))
    return a + b


def run_single(steps, opt_cls, world, **kw):
    m = Quad()
    opt = opt_cls(list(m.parameters()), **kw)
    for _ in range(steps):
        opt.zero_grad()
        total = None
        for k in range(world):
            part = loss(m, k) * (1.0 / world)
            total = part if total is None else total + part
        total.backward()
        opt.step()
    return [m.u.data.copy(), m.v.data.copy()]


def w_ddp(comm, steps):
    m = Quad(shift=float(comm.rank))
    d = DDP(m, comm)
    opt = SGD(list(d.parameters()), lr=0.05, momentum=0.5)
    for _ in range(steps):
        opt.zero_grad()
        loss(m, comm.rank).backward()
        d.sync_grads()
        opt.step()
    return [m.u.data.copy(), m.v.data.copy()]


def w_zero(comm, stage, steps):
    m = Quad()
    zo = ZeroOptimizer(AdamW, list(m.parameters()), comm, stage, lr=0.1, weight_decay=0.1)
    held = []
    for _ in range(steps):
        zo.gather()
        zo.zero_grad()
        loss(m, comm.rank).backward()
        zo.step()
        held.append(zo.memory_bytes())
    zo.gather()
    return [m.u.data.copy(), m.v.data.copy()], held[-1]


@pytest.mark.parametrize("world", [2, 3])
def test_ddp_equals_one_process(world):
    want = run_single(4, SGD, world, lr=0.05, momentum=0.5)
    for got in spawn(w_ddp, world, 4, timeout=20):
        for a, b in zip(got, want):
            np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("stage", [1, 2, 3])
def test_zero_equals_one_process(stage):
    want = run_single(3, AdamW, 3, lr=0.1, weight_decay=0.1)
    for got, _ in spawn(w_zero, 3, stage, 3, timeout=20):
        for a, b in zip(got, want):
            np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)


def test_zero_bytes_by_stage():
    # 12 entries on 3 ranks: 4 each; float64
    want = {
        1: {"params": 96, "grads": 96 + 32, "optimizer": 96},
        2: {"params": 96, "grads": 32, "optimizer": 96},
        3: {"params": 32, "grads": 32, "optimizer": 64},
    }
    for stage in (1, 2, 3):
        for _, mem in spawn(w_zero, 3, stage, 1, timeout=20):
            assert mem == want[stage]


def test_bad_stage():
    class C:
        rank, world = 0, 1

    with pytest.raises(ValueError):
        ZeroOptimizer(AdamW, list(Quad().parameters()), C(), 0)
