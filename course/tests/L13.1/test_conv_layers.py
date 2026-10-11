"""Course tests for L13.1: Conv2d, Conv1d, max_pool2d, avg_pool2d, and LeNet5
(tinyllm/vision/conv.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/L13.1), and the chapter section it comes from.

The chapter's worked example (section 3) pools the 4 x 4 image X_HAND below
with 2 x 2 windows: the maxima are [[4, 2], [3, 5]] and the means
[[2.5, 1.25], [1.5, 2.75]]; the top-right window holds two 2s, and its
gradient goes to the first one in row-major order. The torch golden values
in $TINYLLM_FIXTURES/L13.1/conv_layers_torch.npz come from
course/oracle/L13.1/conv_layers_torch.py (torch.nn.Conv2d, Conv1d,
F.max_pool2d, F.avg_pool2d forward and autograd backward, float64). The
learning test trains on $TINYLLM_FIXTURES/digits/digits.npz, synthetic 8 x 8
digits from course/oracle/L13.1/digits_synth.py.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.gradcheck import gradcheck
from _lib.pcg32 import PCG32
from _lib.thresholds import check
from tinyllm.autograd import functional as F
from tinyllm.autograd.tensor import Tensor
from tinyllm.vision.conv import Conv1d, Conv2d, LeNet5, avg_pool2d, max_pool2d

FIX = Path(os.environ.get("TINYLLM_FIXTURES", ""))
GOLDEN = FIX / "L13.1" / "conv_layers_torch.npz"
DIGITS = FIX / "digits" / "digits.npz"
X_HAND = np.array(
    [
        [1.0, 4.0, 2.0, 2.0],
        [3.0, 2.0, 0.0, 1.0],
        [0.0, 3.0, 5.0, 1.0],
        [1.0, 2.0, 4.0, 1.0],
    ]
)


class Rng:
    """The frozen PCG32 behind the generator API of M06.3."""

    def __init__(self, s: int) -> None:
        self.g = PCG32(seed=s)

    def next_u32(self) -> int:
        return self.g.next_u32()

    def uniform(self) -> float:
        return self.g.uniform()

    def uniforms(self, n: int) -> np.ndarray:
        return np.array([self.g.uniform() for _ in range(n)], dtype=np.float64)

    def below(self, n: int) -> int:
        return self.g.below(n)


def seed() -> int:
    return int(os.environ.get("OL_SEED", "0"))


def t64(a, grad=False) -> Tensor:
    return Tensor(np.asarray(a, dtype=np.float64), requires_grad=grad, dtype=np.float64)


def as64(m) -> list[Tensor]:
    ps = [p for _, p in m.named_parameters()]
    for p in ps:
        p.data = p.data.astype(np.float64)
    return ps


def load_conv(z, name):
    p = [int(v) for v in z[f"{name}.params"]]
    if name.startswith("conv2d"):
        cin, cout, kh, kw, sh, sw, ph, pw, dh, dw, g = p
        m = Conv2d(cin, cout, (kh, kw), (sh, sw), (ph, pw), (dh, dw), g, rng=Rng(1))
    else:
        cin, cout, k, s, pd, d, g = p
        m = Conv1d(cin, cout, k, s, pd, d, g, rng=Rng(1))
    m.weight = t64(z[f"{name}.w"], True)
    m.bias = t64(z[f"{name}.b"], True)
    return m


def test_hand_example_pooling():
    # WHY: section 3's worked example. Max and average pooling with 2 x 2
    #      windows and stride 2; the gradient of the top-right maximum (a tie
    #      of two 2s) goes to the first one, and average pooling spreads
    #      g / 4 over each window.
    # KIND: unit, smoke
    # CATCHES: s01, s04, m01
    # CHAPTER: L13.1 section 3
    x = t64(X_HAND[None, None], True)
    y = max_pool2d(x, 2)
    assert_close(y.data[0, 0], [[4.0, 2.0], [3.0, 5.0]], rtol=0.0, atol=0.0)
    F.sum(y).backward()
    want = np.zeros((4, 4))
    want[0, 1] = want[0, 2] = want[2, 1] = want[2, 2] = 1.0
    assert_close(x.grad[0, 0], want, rtol=0.0, atol=0.0)
    x2 = t64(X_HAND[None, None], True)
    a = avg_pool2d(x2, 2)
    assert_close(a.data[0, 0], [[2.5, 1.25], [1.5, 2.75]], rtol=0.0, atol=0.0)
    F.sum(a).backward()
    assert_close(x2.grad[0, 0], np.full((4, 4), 0.25), rtol=0.0, atol=0.0)


@pytest.mark.parametrize("name", ["conv2d.lenet", "conv2d.grouped", "conv2d.patch"])
def test_conv2d_matches_torch(name):
    # WHY: Conv2d's forward and its hand-written backward must equal
    #      torch.nn.Conv2d and autograd: dx, dW, and db for padding, stride,
    #      dilation, groups, and a ViT patch embedding (kernel = stride).
    #      L13.3 and L13.7 load pretrained conv weights into this layer.
    # KIND: golden
    # CATCHES: s02, s03, m02
    # CHAPTER: L13.1 section 2.1
    z = np.load(GOLDEN)
    m = load_conv(z, name)
    x = t64(z[f"{name}.x"], True)
    y = m(x)
    assert_close(y.data, z[f"{name}.y"], rtol=1e-10, atol=1e-10, msg=f"{name} forward")
    F.sum(y * z[f"{name}.g"]).backward()
    assert_close(x.grad, z[f"{name}.dx"], rtol=1e-10, atol=1e-10, msg=f"{name} dx")
    assert_close(
        m.weight.grad, z[f"{name}.dw"], rtol=1e-10, atol=1e-10, msg=f"{name} dW"
    )
    assert_close(m.bias.grad, z[f"{name}.db"], rtol=1e-10, atol=1e-10, msg=f"{name} db")


@pytest.mark.parametrize("name", ["conv1d.stem1", "conv1d.stem2"])
def test_conv1d_matches_torch(name):
    # WHY: Whisper's stem (L14.2) is Conv1d(k 3, pad 1) then Conv1d(k 3,
    #      stride 2, pad 1): the second halves the frames. Padding belongs
    #      on the time axis only.
    # KIND: golden
    # CATCHES: s05, s03
    # CHAPTER: L13.1 section 2.2
    z = np.load(GOLDEN)
    m = load_conv(z, name)
    x = t64(z[f"{name}.x"], True)
    y = m(x)
    assert_close(y.data, z[f"{name}.y"], rtol=1e-10, atol=1e-10, msg=f"{name} forward")
    F.sum(y * z[f"{name}.g"]).backward()
    assert_close(x.grad, z[f"{name}.dx"], rtol=1e-10, atol=1e-10, msg=f"{name} dx")
    assert_close(
        m.weight.grad, z[f"{name}.dw"], rtol=1e-10, atol=1e-10, msg=f"{name} dW"
    )
    assert_close(m.bias.grad, z[f"{name}.db"], rtol=1e-10, atol=1e-10, msg=f"{name} db")


@pytest.mark.parametrize(
    "name", ["maxpool.k2", "maxpool.k3s2", "maxpool.ties", "avgpool.k2", "avgpool.k3s2"]
)
def test_pooling_matches_torch(name):
    # WHY: pooling's forward and backward equal torch's, including
    #      overlapping windows (k 3, stride 2), where a pixel that is the
    #      maximum of two windows receives both gradients, and integer
    #      inputs full of ties.
    # KIND: golden
    # CATCHES: s01, s04, m01
    # CHAPTER: L13.1 section 2.3
    z = np.load(GOLDEN)
    k, s = (int(v) for v in z[f"{name}.params"])
    fn = max_pool2d if name.startswith("max") else avg_pool2d
    x = t64(z[f"{name}.x"], True)
    y = fn(x, k, s)
    assert_close(y.data, z[f"{name}.y"], rtol=1e-12, atol=1e-12, msg=f"{name} forward")
    F.sum(y * z[f"{name}.g"]).backward()
    assert_close(x.grad, z[f"{name}.dx"], rtol=1e-12, atol=1e-12, msg=f"{name} dx")


@pytest.mark.parametrize(
    "cfg",
    [
        dict(c_in=2, c_out=3, k=(2, 3), stride=1, pad=0, dilation=1, groups=1),
        dict(
            c_in=4, c_out=2, k=2, stride=(2, 1), pad=(1, 0), dilation=(1, 2), groups=2
        ),
    ],
)
def test_gradcheck_conv2d(cfg):
    # WHY: the vector-Jacobian product is written by hand, so it is proven
    #      against the frozen central differences in float64 for x, W, and
    #      b, with stride, padding, dilation, and groups.
    # KIND: gradcheck
    # CATCHES: s02, s03, m02
    # CHAPTER: L13.1 section 2.1
    m = Conv2d(**cfg, rng=Rng(2))
    params = as64(m)
    g = PCG32(seed=3)
    x0 = g.normal_array((2, cfg["c_in"], 4, 5))
    gout = g.normal_array(m(t64(x0)).shape)

    def f(x, w, b):
        m.weight.data, m.bias.data = w, b
        return float(np.sum(m(t64(x)).data * gout))

    start = [p.data.copy() for p in params]
    x = t64(x0, True)
    F.sum(m(x) * gout).backward()
    analytic = [x.grad, m.weight.grad, m.bias.grad]
    gradcheck(f, [x0] + start, analytic, names=["x", "weight", "bias"])


def test_gradcheck_conv1d():
    # WHY: Conv1d reuses the 2-D op on a one-row image; its gradients reach
    #      x, W, and b through the reshapes.
    # KIND: gradcheck
    # CATCHES: s05, s03
    # CHAPTER: L13.1 section 2.2
    m = Conv1d(3, 4, 3, stride=2, pad=1, dilation=1, rng=Rng(4))
    params = as64(m)
    g = PCG32(seed=5)
    x0 = g.normal_array((2, 3, 7))
    gout = g.normal_array(m(t64(x0)).shape)

    def f(x, w, b):
        m.weight.data, m.bias.data = w, b
        return float(np.sum(m(t64(x)).data * gout))

    start = [p.data.copy() for p in params]
    x = t64(x0, True)
    F.sum(m(x) * gout).backward()
    gradcheck(
        f,
        [x0] + start,
        [x.grad, m.weight.grad, m.bias.grad],
        names=["x", "weight", "bias"],
    )


def test_max_pool_ties_route_to_lowest_index():
    # WHY: with ties the "maximum" is ambiguous; the contract (and torch)
    #      sends the gradient to the first element in row-major order, so
    #      gradients are deterministic. A constant image puts the whole
    #      gradient of each window on its top-left pixel.
    # KIND: property
    # CATCHES: s01
    # CHAPTER: L13.1 section 2.3
    x = t64(np.full((1, 2, 4, 6), 7.0), True)
    y = max_pool2d(x, 2)
    gout = np.arange(1.0, 1.0 + y.data.size).reshape(y.shape)
    F.sum(y * gout).backward()
    want = np.zeros((1, 2, 4, 6))
    want[:, :, ::2, ::2] = gout
    assert_close(x.grad, want, rtol=0.0, atol=0.0)


def test_layers_shapes_init_and_dtype():
    # WHY: pretrained checkpoints load by name and shape (weight KCRS with
    #      C / groups input channels, then bias), the init has std
    #      1 / sqrt(fan_in) with fan_in counting one group's channels, and a
    #      float32 input stays float32.
    # KIND: unit
    # CATCHES: s07, m03
    # CHAPTER: L13.1 section 4
    m = Conv2d(8, 16, 3, groups=4, rng=Rng(6))
    assert [n for n, _ in m.named_parameters()] == ["weight", "bias"]
    assert m.weight.shape == (16, 2, 3, 3) and m.bias.shape == (16,)
    std = float(np.std(m.weight.data))
    assert abs(std - 1.0 / math.sqrt(2 * 9)) < 0.05, std
    assert Conv2d(3, 4, 2, bias=False, rng=Rng(6)).bias is None
    c1 = Conv1d(4, 6, 3, groups=2, rng=Rng(7))
    assert c1.weight.shape == (6, 2, 3)
    x = Tensor(PCG32(seed=8).normal_array((2, 8, 5, 5)).astype(np.float32))
    y = m(x)
    assert y.shape == (2, 16, 3, 3) and y.dtype == np.float32
    assert max_pool2d(x, 2).dtype == np.float32
    assert avg_pool2d(x, 3, 1).shape == (2, 8, 3, 3)


def test_lenet_is_the_composition():
    # WHY: LeNet5 wires the layers in the contract's order: conv, ReLU,
    #      pool twice, flatten in (c, h, w) order (torch.flatten), then three
    #      Linear layers. A different flatten order still trains, but a
    #      torch checkpoint would no longer load into it.
    # KIND: unit
    # CATCHES: s06
    # CHAPTER: L13.1 section 2.4
    net = LeNet5(10, 8, rng=Rng(9))
    names = [n for n, _ in net.named_parameters()]
    assert names == [
        "conv1.weight",
        "conv1.bias",
        "conv2.weight",
        "conv2.bias",
        "fc1.weight",
        "fc1.bias",
        "fc2.weight",
        "fc2.bias",
        "fc3.weight",
        "fc3.bias",
    ]
    assert net.fc1.weight.shape == (120, 64)
    x = Tensor(PCG32(seed=10).normal_array((3, 1, 8, 8)).astype(np.float32))
    h = max_pool2d(F.relu(net.conv1(x)), 2)
    h = max_pool2d(F.relu(net.conv2(h)), 2).data
    h = h.reshape(3, -1)
    w = lambda lin: lin.weight.data.T  # noqa: E731
    h = np.maximum(h @ w(net.fc1) + net.fc1.bias.data, 0)
    h = np.maximum(h @ w(net.fc2) + net.fc2.bias.data, 0)
    want = h @ w(net.fc3) + net.fc3.bias.data
    assert_close(net(x).data, want, rtol=1e-5, atol=1e-5)
    assert LeNet5(3, 12, rng=Rng(9)).fc1.weight.shape == (120, 144)


def test_rejects_bad_arguments():
    # WHY: a layer built with impossible sizes, or fed the wrong layout,
    #      must fail loudly at the line that is wrong instead of producing
    #      silently mis-shaped activations.
    # KIND: boundary
    # CATCHES: m03
    # CHAPTER: L13.1 section 4
    for bad in (
        dict(c_in=3, c_out=4, k=3, groups=2),
        dict(c_in=4, c_out=6, k=3, groups=4),
        dict(c_in=3, c_out=4, k=0),
        dict(c_in=3, c_out=4, k=3, pad=-1),
        dict(c_in=3, c_out=4, k=3, stride=0),
    ):
        with pytest.raises(ValueError):
            Conv2d(**bad, rng=Rng(1))
    with pytest.raises(ValueError):
        Conv1d(3, 4, 3, groups=2, rng=Rng(1))
    m = Conv2d(3, 4, 3, rng=Rng(1))
    with pytest.raises(ValueError):
        m(Tensor(np.zeros((1, 2, 5, 5))))
    with pytest.raises(ValueError):
        m(Tensor(np.zeros((3, 5, 5))))
    with pytest.raises(ValueError):
        m(Tensor(np.zeros((1, 3, 2, 2))))
    with pytest.raises(ValueError):
        Conv1d(3, 4, 3, rng=Rng(1))(Tensor(np.zeros((1, 3, 4, 4))))
    with pytest.raises(ValueError):
        max_pool2d(Tensor(np.zeros((1, 1, 2, 2))), 3)
    with pytest.raises(ValueError):
        avg_pool2d(Tensor(np.zeros((1, 2, 2))), 2)
    with pytest.raises(ValueError):
        LeNet5(10, 6, rng=Rng(1))


def test_lenet_learns_digits():
    # WHY: the layers learn together: LeNet-5 on 8 x 8 digits, 400 Adam
    #      steps of batch 32 at lr 3e-3, reaches the reference accuracy on
    #      held-out images (about 0.97). A wrong backward that still passes
    #      shape checks shows up here as a model that does not learn.
    # KIND: learning
    # CATCHES: s02, s03
    # CHAPTER: L13.1 section 2.4
    d = np.load(DIGITS)
    xtr = d["x_train"].astype(np.float32)[:, None] / 255.0
    ytr = d["y_train"]
    net = LeNet5(10, 8, rng=Rng(100 + seed()))
    params = list(net.parameters())
    m1 = [np.zeros_like(p.data) for p in params]
    m2 = [np.zeros_like(p.data) for p in params]
    order = PCG32(seed=200 + seed())
    lr, b1, b2 = 3e-3, 0.9, 0.999
    for step in range(1, 401):
        idx = np.array([order.below(len(ytr)) for _ in range(32)])
        logits = net(Tensor(xtr[idx]))
        logp = F.log_softmax(logits, axis=-1)
        loss = -F.mean(logp[np.arange(32), ytr[idx]])
        net.zero_grad()
        loss.backward()
        for i, p in enumerate(params):  # Adam, the frozen test copy
            m1[i] = b1 * m1[i] + (1 - b1) * p.grad
            m2[i] = b2 * m2[i] + (1 - b2) * p.grad * p.grad
            mh, vh = m1[i] / (1 - b1**step), m2[i] / (1 - b2**step)
            p.data -= (lr * mh / (np.sqrt(vh) + 1e-8)).astype(p.data.dtype)
    xte = d["x_test"].astype(np.float32)[:, None] / 255.0
    acc = float(np.mean(net(Tensor(xte)).data.argmax(-1) == d["y_test"]))
    assert acc >= 0.85, acc
    check("L13.1/test_lenet_learns_digits", "test_acc", acc, direction="min")
