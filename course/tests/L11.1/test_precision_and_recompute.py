"""Course tests for L11.1: emulated mixed precision, loss scaling, gradient
accumulation (tinyllm/train/precision.py) and activation checkpointing
(tinyllm/train/recompute.py).

Rung R0 reading for the course tests; your own graded tests (rung R5) go in
python/tests/l11-1-precision/. Each test names why it exists (WHY), what kind
of check it is (KIND), the planted bugs it kills (CATCHES, mutants in
course/mutants/L11.1), and the chapter section it comes from.

The worked examples of the chapter (section 3): a bf16 matmul whose exact
inputs give 1 + 2^-8, which rounds to 1; a one-weight regression split into
micro-batches of 1 and 3 rows (gradient -15, loss 7.5); a loss-scale run
with interval 2; and a 6-layer stack checkpointed as [0, 3].

Models are built from your L0.4 layers on your L0.1 Tensor, trained with
M10.2's SGD and M10.3's AdamW, with L0.3's losses; every weight is filled
from the frozen PCG32, never from your generator.
"""

from __future__ import annotations

import gc
import math
import os
import tracemalloc

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32
from tinyllm.autograd.losses import mse
from tinyllm.autograd.mode import no_grad
from tinyllm.autograd.tensor import Tensor
from tinyllm.nn.layers import Dropout, Linear, Tanh
from tinyllm.nn.module import Module
from tinyllm.num.fp import round_to_bf16, round_to_fp16
from tinyllm.optim.adamw import AdamW
from tinyllm.optim.sgd import SGD
from tinyllm.train.precision import (
    DynamicLossScaler,
    autocast,
    autocast_bf16,
    autocast_dtype,
    cast,
    grad_accumulate,
    train_step_mixed,
)
from tinyllm.train.recompute import checkpoint, checkpoint_sequential


def seed() -> int:
    return int(os.environ.get("OL_SEED", "0"))


class Stream:
    """The frozen PCG32 with the M06.3 methods Dropout and checkpoint use."""

    def __init__(self, s: int, seq: int = 54) -> None:
        self.g = PCG32(s, seq)

    def uniform(self) -> float:
        return self.g.uniform()

    def uniforms(self, n: int) -> np.ndarray:
        return self.g.uniform_array((n,))

    def state(self) -> tuple[int, int]:
        return (self.g.state, self.g.inc)

    def set_state(self, s) -> None:
        self.g.state, self.g.inc = s


class Block(Module):
    """Linear then tanh (optionally dropout); counts its forward calls."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        rng: PCG32,
        dtype=np.float32,
        p: float = 0.0,
        drop_rng=None,
    ):
        super().__init__()
        self.lin = Linear(d_in, d_out)
        self.lin.weight.data = rng.normal_array(
            (d_out, d_in), scale=1.0 / math.sqrt(d_in)
        ).astype(dtype)
        self.lin.bias.data = rng.normal_array((d_out,), scale=0.1).astype(dtype)
        self.act = Tanh()
        self.drop = Dropout(p, rng=drop_rng) if p > 0 else None
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        y = self.act(self.lin(x))
        return self.drop(y) if self.drop is not None else y


class Stack(Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = list(blocks)
        for i, b in enumerate(self.blocks):
            setattr(self, f"b{i}", b)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


def mlp(rng: PCG32, dims, dtype=np.float32, p=0.0, drop_rng=None) -> Stack:
    return Stack(Block(a, b, rng, dtype, p, drop_rng) for a, b in zip(dims, dims[1:]))


def mse_loss(model, batch):
    return mse(
        model(Tensor(batch["x"], dtype=model.blocks[0].lin.weight.dtype)), batch["y"]
    )


def regression_data(rng: PCG32, n: int, d: int, dtype=np.float32):
    x = rng.normal_array((n, d)).astype(dtype)
    y = np.tanh(x @ np.linspace(-1.0, 1.0, d)[:, None] + 0.3).astype(dtype)
    return {"x": x, "y": y}


def grads(model) -> list[np.ndarray]:
    return [None if p.grad is None else p.grad.copy() for p in model.parameters()]


def representable(a: np.ndarray, rnd) -> bool:
    return bool(np.array_equal(rnd(a).astype(a.dtype), a))


# --- the worked examples --------------------------------------------------------------


def test_hand_example_bf16_matmul():
    # WHY: section 3, number for number. [1, 1] @ [1, 2^-8]^T has exact bf16
    #      inputs and the exact product 1 + 2^-8 = 1.00390625, a tie between
    #      the bf16 neighbours 1 and 1 + 2^-7 that rounds to the even one, 1.
    #      An input of 1.01 is stored as 1.0078125. The gradient of sum(y) is
    #      bf16 too: dA = round(g) round(B)^T = [[1, 2^-8]], dB = [[1], [1]].
    #      Outside the block the same product is the float32 1.00390625.
    # KIND: unit
    # CATCHES: s01
    # CHAPTER: L11.1 section 3, Worked example by hand
    A = Tensor([[1.0, 1.0]], requires_grad=True)
    B = Tensor([[1.0], [2.0**-8]], requires_grad=True)
    with autocast_bf16():
        assert autocast_dtype() == "bf16"
        y = A @ B
        y2 = Tensor([[1.01]]) @ Tensor([[1.0]])
    assert autocast_dtype() is None
    assert y.data.tolist() == [[1.0]] and y.dtype == np.float32
    assert y2.data.tolist() == [[1.0078125]]
    y.backward(np.ones((1, 1), dtype=np.float32))
    assert A.grad.tolist() == [[1.0, 2.0**-8]]
    assert B.grad.tolist() == [[1.0], [1.0]]
    assert (Tensor([[1.0, 1.0]]) @ Tensor([[1.0], [2.0**-8]])).data.tolist() == [
        [1.00390625]
    ]


def test_hand_example_accumulation():
    # WHY: section 3: y = w x with w = 1 on x = [1, 2, 3, 4], targets 2x. One
    #      batch: loss mean((x - 2x)^2) = 7.5, dL/dw = mean(2 (w x - 2x) x) =
    #      -15. Micro-batches of 1 and 3 rows: -2 weighted 1/4 plus -58/3
    #      weighted 3/4 is -15 again. Averaging the two micro-batch gradients
    #      equally gives -10.67; summing them gives -21.3.
    # KIND: unit
    # CATCHES: s07, s08, s09, s10
    # CHAPTER: L11.1 section 3, Worked example by hand
    model = Stack([])
    model.w = Tensor([1.0], requires_grad=True, dtype=np.float64)

    def loss_fn(m, b):
        return mse(Tensor(b["x"], dtype=np.float64) * m.w, b["y"])

    x = np.array([1.0, 2.0, 3.0, 4.0])
    mbs = [{"x": x[:1], "y": 2 * x[:1]}, {"x": x[1:], "y": 2 * x[1:]}]
    loss = grad_accumulate(model, mbs, loss_fn)
    assert_close(loss, 7.5)
    assert_close(model.w.grad, [-15.0])


def test_hand_example_loss_scaler():
    # WHY: section 3's run with S = 8, growth 2, backoff 0.5, interval 2: a
    #      finite step (count 1), an inf (skipped: S halves to 4 and the count
    #      restarts at 0), then two finite steps grow S back to 8, and one more
    #      counts 1 again. Each finite step divides the gradient by S before
    #      the update, so a scaled gradient of 4 at S = 4 is an update of
    #      exactly lr. Without the restart, the second finite step would
    #      already grow S.
    # KIND: unit
    # CATCHES: s11, s12, s14, m02
    # CHAPTER: L11.1 section 3, Worked example by hand
    p = Tensor([0.0], requires_grad=True, dtype=np.float64)
    opt = SGD([p], lr=0.5)
    sc = DynamicLossScaler(init=8.0, growth=2.0, backoff=0.5, interval=2)
    trace = []
    for g in (8.0, math.inf, 4.0, 8.0, 16.0):
        p.grad = np.array([g])
        ok = sc.step(opt, [p])
        trace.append((ok, sc.loss_scale, sc.good_steps, float(p.data[0])))
    assert trace == [
        (True, 8.0, 1, -0.5),
        (False, 4.0, 0, -0.5),
        (True, 4.0, 1, -1.0),
        (True, 8.0, 0, -2.0),
        (True, 8.0, 1, -3.0),
    ]


def test_hand_example_checkpoint_schedule():
    # WHY: section 3's six-layer stack with a budget of 4 units: M08.4 plans
    #      [0, 3], so layers 0 to 2 run forward twice and layers 3 to 5 once,
    #      and the gradients equal the plain run's bit for bit.
    # KIND: unit
    # CATCHES: s18, s24, m01
    # CHAPTER: L11.1 section 3, Worked example by hand
    rng = PCG32(seed(), 111)
    model = mlp(rng, [4] * 7)
    x = Tensor(rng.normal_array((5, 4)).astype(np.float32), requires_grad=True)
    model.zero_grad()
    out = checkpoint_sequential(model.blocks, x, 4)
    assert [b.calls for b in model.blocks] == [1] * 6
    out.backward(np.ones(out.shape, dtype=np.float32))
    assert [b.calls for b in model.blocks] == [2, 2, 2, 1, 1, 1]
    want_x = x.grad.copy()
    want = grads(model)
    x.grad = None
    model.zero_grad()
    model(x).backward(np.ones(out.shape, dtype=np.float32))
    assert np.array_equal(x.grad, want_x)
    for a, b in zip(grads(model), want):
        assert np.array_equal(a, b)


# --- mixed precision ---------------------------------------------------------------


def test_cast_rounds_both_directions():
    # WHY: a cast is an op: forward rounds the values, backward rounds the
    #      gradient (the gradient of storing in bf16 is a bf16 gradient). It
    #      keeps the input's dtype (float64 stays float64, holding bf16
    #      values) and rejects unknown formats.
    # KIND: unit
    # CATCHES: s02, s04
    # CHAPTER: L11.1 section 2, Principles
    rng = PCG32(seed(), 112)
    v = rng.normal_array((3, 4))
    for dt, rnd in (("bf16", round_to_bf16), ("fp16", round_to_fp16)):
        x = Tensor(v, requires_grad=True, dtype=np.float64)
        y = cast(x, dt)
        assert y.dtype == np.float64
        assert np.array_equal(y.data, rnd(v).astype(np.float64))
        g = rng.normal_array((3, 4))
        y.backward(g)
        assert np.array_equal(x.grad, rnd(g).astype(np.float64))
    with pytest.raises(ValueError):
        cast(Tensor([1.0]), "fp8")


def test_autocast_matmul_matches_rounded_reference():
    # WHY: on random [8, 16] @ [16, 5] operands every output equals the
    #      float32 product of the rounded inputs, rounded once more; also for
    #      a reflected ndarray @ Tensor and inside Linear (x @ W^T), fp16 too.
    # KIND: property
    # CATCHES: s01, s03
    # CHAPTER: L11.1 section 2, Principles
    rng = PCG32(seed(), 113)
    a = rng.normal_array((8, 16)).astype(np.float32)
    b = rng.normal_array((16, 5)).astype(np.float32)
    for dt, rnd in (("bf16", round_to_bf16), ("fp16", round_to_fp16)):
        want = rnd(rnd(a) @ rnd(b))
        with autocast(dt):
            got = Tensor(a) @ Tensor(b)
            got_r = a @ Tensor(b)
        assert np.array_equal(got.data, want), dt
        assert np.array_equal(got_r.data, want), dt
    lin = Linear(16, 5)
    lin.weight.data = b.T.copy()
    lin.bias.data = np.zeros(5, dtype=np.float32)
    with autocast("bf16"):
        y = lin(Tensor(a))
    assert np.array_equal(y.data, round_to_bf16(round_to_bf16(a) @ round_to_bf16(b)))


def test_autocast_grads_are_bf16_and_masters_stay_fp32():
    # WHY: under autocast the weight gradients come out of a bf16 backward
    #      matmul, so each is a bf16 value; the weights themselves stay
    #      float32 masters. An update of 1e-3 on a weight of 1 is lost in
    #      bf16 (its spacing at 1 is 2^-7) but kept by the master, which is
    #      why mixed precision keeps one.
    # KIND: property
    # CATCHES: s02, s03
    # CHAPTER: L11.1 section 2, Principles
    rng = PCG32(seed(), 114)
    model = mlp(rng, [6, 16, 1])
    data = regression_data(rng, 32, 6)
    model.zero_grad()
    with autocast_bf16():
        mse_loss(model, data).backward()
    for b in model.blocks:
        assert representable(b.lin.weight.grad, round_to_bf16), (
            "weight gradient is not a bf16 value"
        )
    w = Tensor([1.0], requires_grad=True)
    w.grad = np.array([1.0], dtype=np.float32)
    SGD([w], lr=1e-3).step()
    assert w.dtype == np.float32 and float(w.data[0]) == np.float32(1.0) - np.float32(
        1e-3
    )
    assert not representable(w.data, round_to_bf16)


def test_autocast_restores_matmul():
    # WHY: autocast patches Tensor's matmul for the length of the block. If
    #      the block raises, or blocks nest, the right method must come back,
    #      or every later matmul in the process silently runs in bf16.
    # KIND: boundary
    # CATCHES: s01, s05, s06
    # CHAPTER: L11.1 section 5, Pitfalls
    a, b = Tensor([[1.0, 1.0]]), Tensor([[1.0], [2.0**-8]])
    with pytest.raises(KeyError):
        with autocast_bf16():
            raise KeyError("boom")
    assert autocast_dtype() is None
    assert (a @ b).data.tolist() == [[1.00390625]]
    c = Tensor([[1.0], [2.0**-12]])  # 1 + 2^-12: float32 keeps it, fp16 rounds it to 1
    with autocast("fp16"):
        with autocast("bf16"):
            assert autocast_dtype() == "bf16"
            assert (a @ b).data.tolist() == [[1.0]]
        assert autocast_dtype() == "fp16"
        assert (a @ b).data.tolist() == [[1.00390625]]  # 1 + 2^-8 is an fp16 value
        assert (a @ c).data.tolist() == [[1.0]]
    assert (a @ b).data.tolist() == [[1.00390625]]
    assert (a @ c).data.tolist() == [[1.000244140625]]
    with pytest.raises(ValueError):
        with autocast("int8"):
            pass


def test_fp16_underflow_is_rescued_by_loss_scaling():
    # WHY: fp16's smallest subnormal is 2^-24, about 6e-8. A loss scaled by
    #      1e-6 has gradients near 1e-8: under fp16 they flush to zero and the
    #      model cannot learn. Multiplying the loss by S = 2^16 lifts them into
    #      range; after unscaling they match the float32 gradients to fp16
    #      precision.
    # KIND: differential
    # CATCHES: s01, s02
    # CHAPTER: L11.1 section 2, Principles
    rng = PCG32(seed(), 115)
    model = mlp(rng, [6, 16, 1])
    data = regression_data(rng, 32, 6)

    def tiny(m, b):
        return mse_loss(m, b) * 1e-6

    model.zero_grad()
    grad_accumulate(model, [data], tiny)
    want = grads(model)
    model.zero_grad()
    with autocast("fp16"):
        grad_accumulate(model, [data], tiny)
    w0 = model.blocks[0].lin.weight.grad
    assert np.count_nonzero(w0) < 0.1 * w0.size, (
        "without scaling most fp16 gradients should underflow"
    )
    model.zero_grad()
    sc = DynamicLossScaler(init=2.0**16)
    with autocast("fp16"):
        grad_accumulate(model, [data], tiny, scaler=sc)
    for p, w in zip(model.parameters(), want):
        g = p.grad / sc.loss_scale
        assert_close(g, w, rtol=2e-2, atol=1e-11)


def test_scaler_skips_inf_steps():
    # WHY: a scale too large for fp16 makes some gradient inf. The step must
    #      change no parameter, drop the bad gradients, halve S, and report
    #      False; the next steps then succeed at the smaller scale.
    # KIND: property
    # CATCHES: s02, s12, s13
    # CHAPTER: L11.1 section 5, Pitfalls
    rng = PCG32(seed(), 116)
    model = mlp(rng, [6, 16, 1])
    data = regression_data(rng, 32, 6)
    opt = AdamW(list(model.parameters()), lr=1e-2)
    sc = DynamicLossScaler(init=2.0**40, interval=1000)
    before = [p.data.copy() for p in model.parameters()]
    st = train_step_mixed(model, [data], mse_loss, opt, precision="fp16", scaler=sc)
    assert st["skipped"] == 1.0 and st["scale"] == 2.0**39
    for p, b in zip(model.parameters(), before):
        assert np.array_equal(p.data, b)
        assert p.grad is None
    skipped = 1
    while train_step_mixed(model, [data], mse_loss, opt, precision="fp16", scaler=sc)[
        "skipped"
    ]:
        skipped += 1
        assert skipped < 40
    assert sc.loss_scale == 2.0 ** (40 - skipped) and sc.good_steps == 1
    assert any(
        not np.array_equal(p.data, b) for p, b in zip(model.parameters(), before)
    )


def test_scaler_unscales_before_clipping():
    # WHY: the clip bound is in true gradient units, so unscale first. On a
    #      gradient of true norm 5 scaled by 1024, clipping to 1 must give an
    #      update of norm lr * 1, as the unscaled float32 step does.
    # KIND: differential
    # CATCHES: s11, s15
    # CHAPTER: L11.1 section 5, Pitfalls
    p = Tensor([0.0, 0.0], requires_grad=True, dtype=np.float64)
    q = Tensor([0.0, 0.0], requires_grad=True, dtype=np.float64)
    sc = DynamicLossScaler(init=1024.0)
    p.grad = np.array([3.0, 4.0]) * 1024.0
    assert sc.step(SGD([p], lr=0.1), [p], clip=1.0)
    assert_close(sc.last_grad_norm, 5.0)
    q.grad = np.array([3.0, 4.0])
    from tinyllm.optim.schedule import clip_grad_norm_

    clip_grad_norm_([q], 1.0)
    SGD([q], lr=0.1).step()
    assert_close(p.data, q.data)


def test_scaler_state_dict_roundtrip():
    # WHY: a resumed capstone run must scale and grow exactly as the
    #      uninterrupted one, so the scale and the count of good steps are
    #      part of the checkpoint.
    # KIND: unit
    # CATCHES: s16, m02
    # CHAPTER: L11.1 section 4, The interface
    a = DynamicLossScaler(init=4.0, growth=3.0, backoff=0.25, interval=3)
    p = Tensor([0.0], requires_grad=True, dtype=np.float64)
    for _ in range(2):
        p.grad = np.array([1.0])
        a.step(SGD([p], lr=0.0), [p])
    b = DynamicLossScaler()
    b.load_state_dict(a.state_dict())
    assert b.state_dict() == {
        "loss_scale": 4.0,
        "growth": 3.0,
        "backoff": 0.25,
        "interval": 3,
        "good_steps": 2,
    }
    p.grad = np.array([1.0])
    b.step(SGD([p], lr=0.0), [p])
    assert b.loss_scale == 12.0 and b.good_steps == 0
    for bad in ({"init": 0.0}, {"growth": 1.0}, {"backoff": 1.0}, {"interval": 0}):
        with pytest.raises(ValueError):
            DynamicLossScaler(**bad)


# --- accumulation ----------------------------------------------------------------------


def test_accumulation_equals_one_big_batch():
    # WHY: k micro-batches of unequal sizes (5, 11, 16 of 32 rows) give the
    #      gradients and the loss of one 32-row batch to float32 rounding
    #      (1e-6), which is the whole point: a batch too big for memory,
    #      trained exactly as if it fitted.
    # KIND: differential
    # CATCHES: s07, s08, s09, s10
    # CHAPTER: L11.1 section 2, Principles
    rng = PCG32(seed(), 117)
    model = mlp(rng, [6, 16, 8, 1])
    data = regression_data(rng, 32, 6)
    model.zero_grad()
    big = grad_accumulate(model, [data], mse_loss)
    want = grads(model)
    model.zero_grad()
    cuts = [(0, 5), (5, 16), (16, 32)]
    mbs = [{k: v[a:b] for k, v in data.items()} for a, b in cuts]
    got = grad_accumulate(model, mbs, mse_loss)
    assert_close(got, big, rtol=1e-6, atol=1e-7)
    for g, w in zip(grads(model), want):
        assert_close(g, w, rtol=1e-6, atol=1e-6)
    with pytest.raises(ValueError):
        grad_accumulate(model, [], mse_loss)
    with pytest.raises(ValueError):
        grad_accumulate(model, [{"x": data["x"][:0], "y": data["y"][:0]}], mse_loss)


def test_train_step_mixed_bf16_tracks_fp32():
    # WHY: MS-L11's claim in miniature: 60 AdamW steps on a small regression
    #      with 4 micro-batches under bf16 autocast end within 2% of the same
    #      run in float32. Rounding every matmul to 8 significant bits costs
    #      little because the masters, the loss, and the optimizer stay float32.
    # KIND: differential
    # CATCHES: s09
    # CHAPTER: L11.1 section 2, Principles
    finals = {}
    for prec in ("fp32", "bf16"):
        rng = PCG32(seed(), 118)
        model = mlp(rng, [6, 32, 1])
        data = regression_data(rng, 64, 6)
        opt = AdamW(list(model.parameters()), lr=1e-2, weight_decay=0.0)
        mbs = [{k: v[i : i + 16] for k, v in data.items()} for i in range(0, 64, 16)]
        for _ in range(60):
            st = train_step_mixed(model, mbs, mse_loss, opt, precision=prec, clip=1.0)
        assert "grad_norm" in st and st["skipped"] == 0.0 and st["scale"] == 1.0
        with no_grad():
            finals[prec] = float(mse_loss(model, data).data)
    assert finals["fp32"] < 0.05, finals
    assert abs(finals["bf16"] - finals["fp32"]) <= 0.02 * finals["fp32"] + 1e-4, finals
    with pytest.raises(ValueError):
        train_step_mixed(model, mbs, mse_loss, opt, precision="fp8")


def test_train_step_mixed_rejects_nan_loss():
    # WHY: without a scaler a non-finite loss stops the run before any
    #      parameter changes, exactly as L0.5's train_step does.
    # KIND: boundary
    # CATCHES: s17
    # CHAPTER: L11.1 section 4, The interface
    rng = PCG32(seed(), 119)
    model = mlp(rng, [6, 4, 1])
    data = regression_data(rng, 8, 6)
    opt = SGD(list(model.parameters()), lr=0.1)
    before = [p.data.copy() for p in model.parameters()]
    with pytest.raises(FloatingPointError):
        train_step_mixed(model, [data], lambda m, b: mse_loss(m, b) * math.nan, opt)
    for p, b in zip(model.parameters(), before):
        assert np.array_equal(p.data, b)


# --- checkpointing ---------------------------------------------------------------------


def test_checkpoint_grads_bitwise_equal():
    # WHY: recomputation runs the same operations on the same values, so
    #      every parameter and input gradient equals the plain run's bit for
    #      bit, for every feasible budget of a 7-layer stack, float64 and
    #      float32 alike.
    # KIND: differential
    # CATCHES: s24
    # CHAPTER: L11.1 section 2, Principles
    for dtype in (np.float64, np.float32):
        rng = PCG32(seed(), 120)
        model = mlp(rng, [5] * 8, dtype=dtype)
        x = Tensor(rng.normal_array((3, 5)), requires_grad=True, dtype=dtype)
        g = rng.normal_array((3, 5)).astype(dtype)
        model.zero_grad()
        model(x).backward(g)
        want_x, want = x.grad.copy(), grads(model)
        for budget in (4, 5, 7):
            x.grad = None
            model.zero_grad()
            checkpoint_sequential(model.blocks, x, budget).backward(g)
            assert np.array_equal(x.grad, want_x), (dtype, budget)
            for a, b in zip(grads(model), want):
                assert np.array_equal(a, b), (dtype, budget)


def test_checkpoint_replays_dropout_rng():
    # WHY: dropout draws a fresh mask from its generator on every call. The
    #      rerun must replay the forward pass's draws (or the gradient belongs
    #      to a different mask), and afterwards the generator must continue
    #      where the forward pass left it, as if nothing had been rerun.
    # KIND: differential
    # CATCHES: s21, s22, s24
    # CHAPTER: L11.1 section 5, Pitfalls
    rng = PCG32(seed(), 121)
    w = [rng.normal_array((6, 6)) for _ in range(4)]
    x0 = rng.normal_array((4, 6))
    runs = {}
    for ck in (False, True):
        drop = Stream(seed(), 7)
        blocks = [Block(6, 6, PCG32(0), p=0.5, drop_rng=drop) for _ in range(4)]
        for b, wi in zip(blocks, w):
            b.lin.weight.data = wi.astype(np.float32)
        model = Stack(blocks)
        x = Tensor(x0, requires_grad=True)
        out = checkpoint_sequential(blocks, x, 3, rngs=[drop]) if ck else model(x)
        out.backward(np.ones(out.shape, dtype=np.float32))
        runs[ck] = (x.grad.copy(), grads(model), drop.state())
    assert np.array_equal(runs[True][0], runs[False][0])
    for a, b in zip(runs[True][1], runs[False][1]):
        assert np.array_equal(a, b)
    assert runs[True][2] == runs[False][2], (
        "the generator did not end where the plain run ends"
    )


def test_checkpoint_params_get_grads_from_data_input():
    # WHY: the first segment's input is data, which never requires grad. The
    #      checkpoint node must still be recorded (its parents include the
    #      segment's parameters) or the first layers silently stop learning.
    # KIND: unit
    # CATCHES: s19
    # CHAPTER: L11.1 section 5, Pitfalls
    rng = PCG32(seed(), 122)
    model = mlp(rng, [4] * 5)
    x = Tensor(rng.normal_array((3, 4)))
    model.zero_grad()
    model(x).backward(np.ones((3, 4), dtype=np.float32))
    want = grads(model)
    model.zero_grad()
    out = checkpoint_sequential(model.blocks, x, 3)
    assert out.requires_grad
    out.backward(np.ones((3, 4), dtype=np.float32))
    for a, b in zip(grads(model), want):
        assert a is not None and np.array_equal(a, b)


def test_checkpoint_function_and_no_grad():
    # WHY: checkpoint on its own: two Tensor arguments (one passed twice)
    #      and a plain number get the plain run's gradients; under no_grad it
    #      is fn(*args) with no node and no rerun.
    # KIND: boundary
    # CATCHES: s20, s24
    # CHAPTER: L11.1 section 4, The interface
    rng = PCG32(seed(), 123)
    av, bv = rng.normal_array((3, 3)), rng.normal_array((3, 3))
    calls = []

    def f(a, b, c, k):
        calls.append(1)
        return (a @ b + c * a) * k

    res = {}
    for ck in (False, True):
        a = Tensor(av, requires_grad=True, dtype=np.float64)
        b = Tensor(bv, requires_grad=True, dtype=np.float64)
        y = checkpoint(f, a, b, a, 0.5) if ck else f(a, b, a, 0.5)
        y.backward(np.ones((3, 3)))
        res[ck] = (a.grad.copy(), b.grad.copy())
    assert np.array_equal(res[True][0], res[False][0])
    assert np.array_equal(res[True][1], res[False][1])
    assert len(calls) == 3
    with no_grad():
        y = checkpoint(f, Tensor(av, requires_grad=True), Tensor(bv), Tensor(av), 2.0)
    assert not y.requires_grad and len(calls) == 4


def test_checkpoint_lowers_peak_memory():
    # WHY: the reason to checkpoint at all, measured with tracemalloc on a
    #      12-layer stack (batch 256, width 64). After the forward pass the
    #      plain graph holds all 12 layers' saved values; with budget 5 it
    #      holds 2 segment inputs and the last segment's 3 layers, under half
    #      the bytes. Your L0.1 backward keeps the outer graph until it
    #      returns, so the saving during backward is smaller, but the overall
    #      peak still drops.
    # KIND: property
    # CATCHES: s23, m01
    # CHAPTER: L11.1 section 2, Principles
    rng = PCG32(seed(), 124)
    model = mlp(rng, [64] * 13)
    xv = rng.normal_array((256, 64)).astype(np.float32)

    def peaks(fn) -> tuple[int, int]:
        model.zero_grad()
        gc.collect()
        tracemalloc.start()
        try:
            out = fn(Tensor(xv))
            after_forward = tracemalloc.get_traced_memory()[1]
            out.backward(np.ones(out.shape, dtype=np.float32))
            del out
            return after_forward, tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    plain = peaks(model)
    ck = peaks(lambda x: checkpoint_sequential(model.blocks, x, 5))
    assert ck[0] < 0.5 * plain[0], (
        f"forward: checkpointed {ck[0]} bytes vs plain {plain[0]}"
    )
    assert ck[1] < 0.9 * plain[1], (
        f"overall: checkpointed {ck[1]} bytes vs plain {plain[1]}"
    )
