"""My tests for L5.1 (rung R5: the formula written out in numpy is the
forward oracle, and my own central differences check every backward). They
import only the contract."""

import math

import numpy as np
import pytest
from tinyllm.autograd.tensor import Tensor
from tinyllm.xfmr.sdpa import scaled_dot_product_attention, sdpa_backward, sdpa_forward


def oracle(q, k, v, mask=None, scale=None):
    s = (
        q
        @ np.swapaxes(k, -1, -2)
        * (1 / math.sqrt(q.shape[-1]) if scale is None else scale)
    )
    if mask is not None:
        s = np.where(mask, s, -np.inf)
    m = s.max(-1, keepdims=True)
    m = np.where(np.isneginf(m), 0, m)
    e = np.exp(s - m)
    t = e.sum(-1, keepdims=True)
    p = e / np.where(t == 0, 1, t)
    return p @ v, p


def numgrad(f, x, eps=1e-6):
    g = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        old = x[i]
        x[i] = old + eps
        a = f()
        x[i] = old - eps
        b = f()
        x[i] = old
        g[i] = (a - b) / (2 * eps)
    return g


class Rng:
    def __init__(self, seed):
        self.r = np.random.Generator(np.random.PCG64(seed))

    def uniforms(self, n):
        return self.r.uniform(size=n)


def data(seed, tq=3, tk=5, d=4, dv=2):
    r = np.random.Generator(np.random.PCG64(seed))
    return (
        r.normal(size=(2, tq, d)),
        r.normal(size=(2, tk, d)),
        r.normal(size=(2, tk, dv)),
        r.normal(size=(2, tq, dv)),
    )


def test_hand_example():
    q = np.array([[2.0, 0, 0, 0]])
    k = np.array([[1.0, 0, 0, 0], [0, 1, 0, 0], [2, 0, 0, 0]])
    v = np.array([[1.0, 0], [0, 1], [1, 1]])
    out, p = sdpa_forward(q, k, v)
    z = 1 + math.e + math.e**2
    np.testing.assert_allclose(p, [[math.e / z, 1 / z, math.e**2 / z]], rtol=1e-12)
    np.testing.assert_allclose(
        out, [[(math.e + math.e**2) / z, (1 + math.e**2) / z]], rtol=1e-12
    )


@pytest.mark.parametrize("mask", [None, "causal", "pad"])
def test_forward_matches_oracle(mask):
    q, k, v, _ = data(1, tq=5, tk=5)
    m = {
        None: None,
        "causal": np.tril(np.ones((5, 5), bool)),
        "pad": np.array([True] * 3 + [False] * 2),
    }[mask]
    out, p = sdpa_forward(q, k, v, m)
    o2, p2 = oracle(q, k, v, m)
    np.testing.assert_allclose(out, o2, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(p, p2, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(p.sum(-1), 1.0, rtol=1e-12)


def test_scale_argument():
    q, k, v, _ = data(2)
    np.testing.assert_allclose(
        sdpa_forward(q, k, v, scale=0.3)[0], oracle(q, k, v, scale=0.3)[0], rtol=1e-12
    )


@pytest.mark.parametrize("tq,tk", [(3, 5), (4, 4)])
def test_backward_matches_my_gradcheck(tq, tk):
    q, k, v, g = data(3, tq=tq, tk=tk)
    mask = np.tril(np.ones((tq, tk), bool), k=tk - tq)
    tq_, tk_, tv_ = (Tensor(x, requires_grad=True, dtype=np.float64) for x in (q, k, v))
    out, _ = scaled_dot_product_attention(tq_, tk_, tv_, mask=mask)
    out.backward(g)
    for x, t in ((q, tq_), (k, tk_), (v, tv_)):
        num = numgrad(lambda: float(np.sum(oracle(q, k, v, mask)[0] * g)), x)
        np.testing.assert_allclose(t.grad, num, rtol=1e-5, atol=1e-7)


def test_backward_function_directly():
    q, k, v, g = data(4)
    out, p = sdpa_forward(q, k, v)
    dq, dk, dv = sdpa_backward(q, k, v, p, g)
    np.testing.assert_allclose(
        dv,
        numgrad(lambda: float(np.sum(oracle(q, k, v)[0] * g)), v),
        rtol=1e-5,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        dk,
        numgrad(lambda: float(np.sum(oracle(q, k, v)[0] * g)), k),
        rtol=1e-5,
        atol=1e-7,
    )


def test_dropout_forward_and_backward():
    q, k, v, g = data(5)
    tq_, tk_, tv_ = (Tensor(x, requires_grad=True, dtype=np.float64) for x in (q, k, v))
    out, w = scaled_dot_product_attention(tq_, tk_, tv_, dropout_p=0.3, rng=Rng(7))
    _, p = oracle(q, k, v)
    keep = Rng(7).uniforms(p.size).reshape(p.shape) >= 0.3
    np.testing.assert_allclose(w.data, p, rtol=1e-12)
    np.testing.assert_allclose(out.data, (p * keep / 0.7) @ v, rtol=1e-12)
    out.backward(g)

    def f():
        _, pp = oracle(q, k, v)
        return float(np.sum(((pp * keep / 0.7) @ v) * g))

    for x, t in ((q, tq_), (k, tk_), (v, tv_)):
        np.testing.assert_allclose(t.grad, numgrad(f, x), rtol=1e-5, atol=1e-7)


def test_fully_masked_row():
    q, k, v, _ = data(6)
    m = np.array([[True] * 5, [False] * 5, [True, False, False, False, False]])
    out, p = sdpa_forward(q, k, v, m)
    assert np.all(out[:, 1] == 0) and np.all(p[:, 1] == 0)


def test_rejects_float_mask():
    q, k, v, _ = data(7)
    with pytest.raises(ValueError):
        sdpa_forward(q, k, v, np.ones((3, 5)))


def test_float32_stays_float32():
    q, k, v, _ = data(8)
    out, w = scaled_dot_product_attention(
        *(Tensor(x, dtype=np.float32) for x in (q, k, v))
    )
    assert out.dtype == np.float32 and w.dtype == np.float32


def test_weights_carry_no_gradient():
    q, k, v, _ = data(9)
    out, w = scaled_dot_product_attention(
        *(Tensor(x, requires_grad=True, dtype=np.float64) for x in (q, k, v))
    )
    assert out.requires_grad and not w.requires_grad


def test_bad_arguments():
    q, k, v, _ = data(10)
    with pytest.raises(ValueError):
        scaled_dot_product_attention(Tensor(q), Tensor(k), Tensor(v), dropout_p=0.5)
    for s in (0.0, -2.0):
        with pytest.raises(ValueError):
            sdpa_forward(q, k, v, scale=s)
