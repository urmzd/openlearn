"""My tests for L11.1 (rung R5). Oracles: the format tables of M09.1 applied
by hand to numpy products, a big batch against its micro-batches, and the
plain backward pass against the checkpointed one. They import only the
contract."""

import math

import numpy as np
import pytest
from tinyllm.autograd.mode import no_grad
from tinyllm.autograd.tensor import Tensor
from tinyllm.nn.layers import Dropout, Linear
from tinyllm.nn.module import Module
from tinyllm.num.fp import round_to_bf16, round_to_fp16
from tinyllm.num.rng import PCG32
from tinyllm.optim.sgd import SGD
from tinyllm.train.precision import (
    DynamicLossScaler,
    autocast,
    autocast_dtype,
    cast,
    grad_accumulate,
    train_step_mixed,
)
from tinyllm.train.recompute import checkpoint, checkpoint_sequential


def arr(rng, *shape):
    return (rng.uniforms(int(np.prod(shape))) * 2 - 1).reshape(shape)


class Lin(Module):
    def __init__(self, rng, d, dtype=np.float32, p=0.0, drop_rng=None):
        super().__init__()
        self.lin = Linear(d, d)
        self.lin.weight.data = arr(rng, d, d).astype(dtype)
        self.lin.bias.data = arr(rng, d).astype(dtype)
        self.drop = Dropout(p, rng=drop_rng) if p else None
        self.n = 0

    def forward(self, x):
        self.n += 1
        y = self.lin(x)
        y = y * y * 0.5 + y  # a non-linearity built from Tensor ops only
        return self.drop(y) if self.drop is not None else y


class Net(Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = list(layers)
        for i, m in enumerate(self.layers):
            setattr(self, f"l{i}", m)

    def forward(self, x):
        for m in self.layers:
            x = m(x)
        return x


def sq_loss(model, b):
    out = model(Tensor(b["x"], dtype=model.layers[0].lin.weight.dtype))
    d = out - b["y"]
    return _mean(d * d)


def _mean(t):
    """The mean of a 2-D tensor as two matmuls (so autocast touches it too)."""
    ones = np.ones((t.shape[-1], 1), dtype=t.dtype) / t.data.size
    s = t @ ones
    return Tensor(np.ones((1, s.shape[0]), dtype=t.dtype)) @ s


def gr(model):
    return [p.grad.copy() for p in model.parameters()]


def test_bf16_product_rounds_inputs_and_output():
    a = np.array([[1.01, 1.0]], dtype=np.float32)
    b = np.array([[1.0], [2.0**-8]], dtype=np.float32)
    with autocast("bf16"):
        assert autocast_dtype() == "bf16"
        y = Tensor(a) @ Tensor(b)
    assert y.data.tolist() == round_to_bf16(round_to_bf16(a) @ round_to_bf16(b)).tolist()
    assert y.data.tolist() == [[1.015625]]  # 1.0078125 + 2^-8 is a tie: to even


def test_reflected_and_fp16_products():
    rng = PCG32(3)
    a, b = arr(rng, 4, 6).astype(np.float32), arr(rng, 6, 3).astype(np.float32)
    with autocast("fp16"):
        y = a @ Tensor(b)
    assert np.array_equal(y.data, round_to_fp16(round_to_fp16(a) @ round_to_fp16(b)))
    with autocast("bf16"):
        y = Tensor(a) @ Tensor(b)
    assert np.array_equal(y.data, round_to_bf16(round_to_bf16(a) @ round_to_bf16(b)))


def test_cast_gradient_is_rounded_and_dtype_kept():
    rng = PCG32(4)
    x = Tensor(arr(rng, 5), requires_grad=True, dtype=np.float64)
    y = cast(x, "bf16")
    assert y.dtype == np.float64
    g = arr(rng, 5)
    y.backward(g)
    assert np.array_equal(x.grad, round_to_bf16(g).astype(np.float64))


def test_weight_grads_are_bf16_values():
    rng = PCG32(5)
    net = Net([Lin(rng, 6), Lin(rng, 6)])
    with autocast("bf16"):
        sq_loss(net, {"x": arr(rng, 8, 6), "y": arr(rng, 8, 6)}).backward()
    for m in net.layers:
        g = m.lin.weight.grad
        assert np.array_equal(round_to_bf16(g), g)


def test_autocast_undone_after_error_and_nesting():
    a, b = Tensor([[1.0, 1.0]]), Tensor([[1.0], [2.0**-12]])
    with pytest.raises(RuntimeError):
        with autocast("bf16"):
            raise RuntimeError
    assert (a @ b).data.tolist() == [[1.000244140625]]
    with autocast("fp16"):
        with autocast("bf16"):
            pass
        assert (a @ b).data.tolist() == [[1.0]]
    assert autocast_dtype() is None


def test_accumulate_weights_by_rows():
    rng = PCG32(6)
    net = Net([Lin(rng, 4, dtype=np.float64)])
    data = {"x": arr(rng, 12, 4), "y": arr(rng, 12, 4)}
    net.zero_grad()
    big = grad_accumulate(net, [data], sq_loss)
    want = gr(net)
    net.zero_grad()
    parts = [{k: v[a:b] for k, v in data.items()} for a, b in ((0, 2), (2, 9), (9, 12))]
    got = grad_accumulate(net, parts, sq_loss)
    assert abs(got - big) < 1e-12
    for g, w in zip(gr(net), want):
        np.testing.assert_allclose(g, w, rtol=1e-10, atol=1e-12)


def test_scaler_sequence():
    p = Tensor([0.0], requires_grad=True, dtype=np.float64)
    sc = DynamicLossScaler(init=2.0, growth=4.0, backoff=0.25, interval=2)
    seen = []
    for g in (2.0, math.nan, 0.5, 0.5, 2.0):
        p.grad = np.array([g])
        seen.append((sc.step(SGD([p], lr=1.0), [p]), sc.loss_scale, float(p.data[0]), p.grad is None))
    assert seen == [
        (True, 2.0, -1.0, False),
        (False, 0.5, -1.0, True),
        (True, 0.5, -2.0, False),
        (True, 2.0, -3.0, False),
        (True, 2.0, -4.0, False),
    ]


def test_scaler_clip_in_true_units():
    p = Tensor([0.0, 0.0], requires_grad=True, dtype=np.float64)
    sc = DynamicLossScaler(init=100.0)
    p.grad = np.array([600.0, 800.0])
    sc.step(SGD([p], lr=1.0), [p], clip=2.0)
    np.testing.assert_allclose(p.data, [-1.2, -1.6], rtol=1e-5)
    assert abs(sc.last_grad_norm - 10.0) < 1e-9


def test_scaler_resume():
    a = DynamicLossScaler(init=1.0, interval=2)
    p = Tensor([0.0], requires_grad=True, dtype=np.float64)
    p.grad = np.array([1.0])
    a.step(SGD([p], lr=0.0), [p])
    b = DynamicLossScaler()
    b.load_state_dict(a.state_dict())
    p.grad = np.array([1.0])
    b.step(SGD([p], lr=0.0), [p])
    assert b.loss_scale == 2.0


def test_fp16_overflow_skips_and_backs_off():
    rng = PCG32(7)
    net = Net([Lin(rng, 4)])
    data = {"x": arr(rng, 6, 4) * 100, "y": arr(rng, 6, 4)}
    before = [p.data.copy() for p in net.parameters()]
    sc = DynamicLossScaler(init=2.0**30)
    st = train_step_mixed(net, [data], sq_loss, SGD(list(net.parameters()), lr=0.1), "fp16", sc)
    assert st["skipped"] == 1.0 and sc.loss_scale == 2.0**29
    assert all(np.array_equal(p.data, b) and p.grad is None for p, b in zip(net.parameters(), before))


def test_nan_loss_without_scaler_raises():
    rng = PCG32(8)
    net = Net([Lin(rng, 3)])
    with pytest.raises(FloatingPointError):
        train_step_mixed(net, [{"x": arr(rng, 2, 3), "y": arr(rng, 2, 3) * math.nan}], sq_loss,
                         SGD(list(net.parameters()), lr=0.1))


def plain_and_ck(net, x, budget, rngs=()):
    net.zero_grad()
    out = net(x)
    out.backward(np.ones(out.shape, dtype=out.dtype))
    want = gr(net)
    net.zero_grad()
    out = checkpoint_sequential(net.layers, x, budget, rngs=rngs)
    out.backward(np.ones(out.shape, dtype=out.dtype))
    return want, gr(net)


def test_checkpoint_exact_and_recompute_counts():
    rng = PCG32(9)
    net = Net([Lin(rng, 3, dtype=np.float64) for _ in range(6)])
    x = Tensor(arr(rng, 2, 3), dtype=np.float64)
    want, got = plain_and_ck(net, x, 4)
    assert [m.n for m in net.layers] == [3, 3, 3, 2, 2, 2]
    for a, b in zip(got, want):
        assert np.array_equal(a, b)


def test_checkpoint_input_grad_exact():
    rng = PCG32(10)
    net = Net([Lin(rng, 3, dtype=np.float64) for _ in range(5)])
    xs = []
    for ck in (0, 1):
        x = Tensor(arr(PCG32(11), 2, 3), requires_grad=True, dtype=np.float64)
        out = checkpoint_sequential(net.layers, x, 3) if ck else net(x)
        out.backward(np.ones(out.shape))
        xs.append(x.grad)
    assert np.array_equal(xs[0], xs[1])


def test_checkpoint_dropout_same_masks_and_stream():
    res = []
    for ck in (0, 1):
        drop = PCG32(12)
        net = Net([Lin(PCG32(13 + i), 4, p=0.3, drop_rng=drop) for i in range(4)])
        x = Tensor(arr(PCG32(20), 3, 4))
        out = checkpoint_sequential(net.layers, x, 3, rngs=[drop]) if ck else net(x)
        out.backward(np.ones(out.shape, dtype=np.float32))
        res.append((gr(net), drop.state()))
    for a, b in zip(res[0][0], res[1][0]):
        assert np.array_equal(a, b)
    assert res[0][1] == res[1][1]


def test_checkpoint_twice_passed_tensor():
    a = Tensor([[1.0, 2.0]], requires_grad=True, dtype=np.float64)
    y = checkpoint(lambda u, v: u * v, a, a)
    y.backward(np.ones((1, 2)))
    assert a.grad.tolist() == [[2.0, 4.0]]


def test_checkpoint_under_no_grad_runs_once():
    rng = PCG32(14)
    net = Net([Lin(rng, 3) for _ in range(3)])
    with no_grad():
        out = checkpoint_sequential(net.layers, Tensor(arr(rng, 1, 3)), 2)
    assert not out.requires_grad and [m.n for m in net.layers] == [1, 1, 1]


def test_checkpoint_forward_keeps_less():
    import tracemalloc

    rng = PCG32(15)
    net = Net([Lin(rng, 64) for _ in range(10)])
    xv = arr(rng, 128, 64).astype(np.float32)
    held = []
    for ck in (0, 1):
        tracemalloc.start()
        out = checkpoint_sequential(net.layers, Tensor(xv), 4) if ck else net(Tensor(xv))
        held.append(tracemalloc.get_traced_memory()[0])
        tracemalloc.stop()
        del out
    assert held[1] < 0.6 * held[0], held
