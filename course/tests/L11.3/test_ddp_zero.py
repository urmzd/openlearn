"""Course tests for L11.3 (optional): DDP and ZeRO stages 1 to 3
(tinyllm/dist/zero.py).

Rung R0 reading for the course tests; your own graded tests (rung R5) go in
python/tests/l11-3-zero/. Each test names why it exists (WHY), what kind of
check it is (KIND), the planted bugs it kills (CATCHES, mutants in
course/mutants/L11.3), and the chapter section it comes from.

The worked example of the chapter (section 3): one 10-entry parameter on 4
ranks (chunks of 3, 3, 2, 2), rank r's loss sum((w - r)^2) / 2, one AdamW
step with lr 0.5, and the bytes each rank holds at each stage.

Every test runs real processes through your L11.2 spawn (workers are
module-level functions). Models are your L0.4 layers on your L0.1 Tensor;
the single-process reference uses M10.3's AdamW and M10.2's SGD directly.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32
from tinyllm.autograd.tensor import Tensor
from tinyllm.dist.comm import spawn
from tinyllm.dist.zero import DDP, ZeroOptimizer
from tinyllm.nn.layers import Linear, Tanh
from tinyllm.nn.module import Module
from tinyllm.optim.adamw import AdamW
from tinyllm.optim.sgd import SGD

T = 30.0
B = 8  # global batch rows


def seed() -> int:
    return int(os.environ.get("OL_SEED", "0"))


class Vec(Module):
    def __init__(self, n: int) -> None:
        super().__init__()
        self.w = Tensor(np.arange(float(n)), requires_grad=True, dtype=np.float64)


class MLP(Module):
    def __init__(self, s: int, init_seq: int = 400) -> None:
        super().__init__()
        rng = PCG32(s, init_seq)
        self.a, self.act, self.b = Linear(4, 6), Tanh(), Linear(6, 2)
        for lin in (self.a, self.b):
            lin.weight.data = rng.normal_array(lin.weight.shape, scale=0.5)
            lin.bias.data = rng.normal_array(lin.bias.shape, scale=0.1)

    def forward(self, x):
        return self.b(self.act(self.a(x)))


def batch(s: int):
    rng = PCG32(s, 401)
    return rng.normal_array((B, 4)), rng.normal_array((B, 2))


def sum_loss(t):
    """The sum of a 2-D tensor's entries as two matmuls (a one-element loss)."""
    ones = np.ones((t.shape[-1], 1))
    return Tensor(np.ones((1, t.shape[0]))) @ (t @ ones)


def mean_sq(model, x, y):
    d = model(Tensor(x, dtype=np.float64)) - y
    return sum_loss(d * d) * (1.0 / d.data.size)


def params_of(model):
    return [p.data.copy() for p in model.parameters()]


# --- workers ----------------------------------------------------------------------------


def w_hand(comm, stage):
    m = Vec(10)
    zo = ZeroOptimizer(
        AdamW, list(m.parameters()), comm, stage, lr=0.5, weight_decay=0.0
    )
    zo.gather()
    d = m.w - float(comm.rank)
    sum_loss((d * d * 0.5)[None, :]).backward()
    zo.step()
    mem = zo.memory_bytes()
    zo.gather()
    return m.w.data.copy(), mem


def w_ddp(comm, s, steps, scramble):
    model = MLP(s + (comm.rank if scramble else 0))
    ddp = DDP(model, comm)
    start = params_of(model)
    opt = SGD(list(ddp.parameters()), lr=0.3, momentum=0.9)
    x, y = batch(s)
    rows = slice(comm.rank * B // comm.world, (comm.rank + 1) * B // comm.world)
    grads = None
    for _ in range(steps):
        opt.zero_grad()
        mean_sq(ddp, x[rows], y[rows]).backward()
        ddp.sync_grads()
        grads = [p.grad.copy() for p in ddp.parameters()]
        opt.step()
    return start, params_of(model), grads


def w_zero(comm, s, stage, steps):
    model = MLP(s)
    zo = ZeroOptimizer(
        AdamW, list(model.parameters()), comm, stage, lr=0.05, weight_decay=0.01
    )
    x, y = batch(s)
    rows = slice(comm.rank * B // comm.world, (comm.rank + 1) * B // comm.world)
    released = []
    for _ in range(steps):
        zo.gather()
        zo.zero_grad()
        mean_sq(model, x[rows], y[rows]).backward()
        zo.step()
        released.append(all(p.data.size == 0 for p in model.parameters()))
    mem = zo.memory_bytes()
    zo.gather()
    return params_of(model), mem, released


def single(s: int, steps: int, opt_name: str):
    model = MLP(s)
    x, y = batch(s)
    if opt_name == "adamw":
        opt = AdamW(list(model.parameters()), lr=0.05, weight_decay=0.01)
    else:
        opt = SGD(list(model.parameters()), lr=0.3, momentum=0.9)
    for _ in range(steps):
        opt.zero_grad()
        mean_sq(model, x, y).backward()
        opt.step()
    return params_of(model), model


# --- tests --------------------------------------------------------------------------------


def test_hand_example_zero_step():
    # WHY: section 3 by hand. w = [0, 1, ..., 9] on 4 ranks; rank r's loss
    #      sum((w - r)^2) / 2 has gradient w - r, so the mean gradient is
    #      w - 1.5. Adam's first step moves each entry by lr against the sign
    #      of its gradient (m / sqrt(v) = +-1 after bias correction): with lr
    #      0.5 the result is [0.5, 1.5, 1.5, 2.5, ..., 8.5] on every rank and
    #      every stage. Rank 0 owns 3 entries (24 bytes), rank 3 owns 2.
    # KIND: unit
    # CATCHES: s04, s06, s09, s10
    # CHAPTER: L11.3 section 3, Worked example by hand
    want = np.array([0.5, 1.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5])
    for stage in (1, 2, 3):
        out = spawn(w_hand, 4, stage, timeout=T)
        for w, _ in out:
            assert_close(w, want, rtol=1e-7, atol=1e-7)


def test_hand_example_memory_by_stage():
    # WHY: section 3's table: what rank 0 (3 entries) and rank 3 (2 entries)
    #      hold after the step, float64. Stage 1: full parameters (80) and
    #      gradients (80 + its chunk's 24), optimizer = 2 moments + master
    #      chunk (72). Stage 2 drops the full gradients (24). Stage 3 keeps
    #      only the chunk between steps: 24 / 24 / 48 on rank 0.
    # KIND: unit
    # CATCHES: s04, s07, s08
    # CHAPTER: L11.3 section 3, Worked example by hand
    want = {
        1: [
            {"params": 80, "grads": 104, "optimizer": 72},
            {"params": 80, "grads": 96, "optimizer": 48},
        ],
        2: [
            {"params": 80, "grads": 24, "optimizer": 72},
            {"params": 80, "grads": 16, "optimizer": 48},
        ],
        3: [
            {"params": 24, "grads": 24, "optimizer": 48},
            {"params": 16, "grads": 16, "optimizer": 32},
        ],
    }
    for stage in (1, 2, 3):
        out = spawn(w_hand, 4, stage, timeout=T)
        assert [out[0][1], out[3][1]] == want[stage], stage


@pytest.mark.parametrize("world", [2, 4])
def test_ddp_matches_single_process(world):
    # WHY: the DDP contract: replicas that average their gradients and step
    #      the same optimizer stay identical, and equal one process training
    #      on the whole batch (equal slices, mean losses) to float64 rounding.
    #      Momentum SGD for 5 steps compounds any error in the average.
    # KIND: differential
    # CATCHES: s02, s03, s09
    # CHAPTER: L11.3 section 2, Principles
    s = seed()
    want, _ = single(s, 5, "sgd")
    out = spawn(w_ddp, world, s, 5, False, timeout=T)
    for _, got, grads in out:
        for g, w in zip(got, want):
            assert_close(g, w, rtol=1e-10, atol=1e-12)
        for a, b in zip(grads, out[0][2]):
            assert np.array_equal(a, b), "ranks disagree on the averaged gradient"


def test_ddp_broadcasts_rank0_weights():
    # WHY: replicas that start from different weights never agree again. DDP
    #      copies rank 0's parameters to every rank in place (each rank here
    #      initializes from a different seed), and training then matches the
    #      single process started from rank 0's weights.
    # KIND: unit
    # CATCHES: s01
    # CHAPTER: L11.3 section 2, Principles
    s = seed()
    want, _ = single(s, 3, "sgd")
    out = spawn(w_ddp, 2, s, 3, True, timeout=T)
    r0 = [p.data for p in MLP(s).parameters()]
    for start, got, _ in out:
        for a, b in zip(start, r0):
            assert np.array_equal(a, b)
        for g, w in zip(got, want):
            assert_close(g, w, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("stage", [1, 2, 3])
@pytest.mark.parametrize("world", [2, 4])
def test_zero_matches_single_process(stage, world):
    # WHY: each stage only changes who stores what; the arithmetic of the
    #      update is the single process's, entry by entry (AdamW is
    #      elementwise). 4 AdamW steps with weight decay agree to 1e-9.
    # KIND: differential
    # CATCHES: s04, s05, s06, s09, s10
    # CHAPTER: L11.3 section 2, Principles
    s = seed()
    want, _ = single(s, 4, "adamw")
    for got, _, _ in spawn(w_zero, world, s, stage, 4, timeout=T):
        for g, w in zip(got, want):
            assert_close(g, w, rtol=1e-9, atol=1e-10)


def test_zero_memory_is_one_over_p():
    # WHY: the point of ZeRO, against M05.1's plan for one process (params
    #      + grads + 2 AdamW moments = 4 values per parameter): stage 3 holds
    #      about 1/p of all of it on each rank, stage 1 about 1/p of the
    #      optimizer state. The model has 44 parameters; with 4 ranks the
    #      chunks are 11 each, so the shares are exact.
    # KIND: property
    # CATCHES: s07, s08
    # CHAPTER: L11.3 section 2, Principles
    s, p = seed(), 4
    n = sum(q.data.size for q in MLP(s).parameters())
    assert n == 44
    single_total = 4 * n * 8
    for stage, mem_want in (
        (1, {"params": n * 8, "grads": n * 8 + 11 * 8, "optimizer": 3 * 11 * 8}),
        (2, {"params": n * 8, "grads": 11 * 8, "optimizer": 3 * 11 * 8}),
        (3, {"params": 11 * 8, "grads": 11 * 8, "optimizer": 2 * 11 * 8}),
    ):
        for _, mem, released in spawn(w_zero, p, s, stage, 2, timeout=T):
            assert mem == mem_want, (stage, mem)
            assert released == [stage == 3] * 2
    assert 4 * 11 * 8 * p == single_total  # stage 3: exactly 1/p per rank


def test_zero_rejects_bad_stage():
    # WHY: a stage outside 1 to 3 is a configuration bug; so is an empty
    #      parameter list. Both fail at construction, before any collective.
    # KIND: boundary
    # CATCHES: m01
    # CHAPTER: L11.3 section 4, The interface
    class FakeComm:
        rank, world = 0, 1

    for bad in (0, 4):
        with pytest.raises(ValueError):
            ZeroOptimizer(AdamW, list(MLP(seed()).parameters()), FakeComm(), bad)
    with pytest.raises(ValueError):
        ZeroOptimizer(AdamW, [], FakeComm(), 1)
