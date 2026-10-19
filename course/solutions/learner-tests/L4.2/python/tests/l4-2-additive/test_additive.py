"""My tests for L4.2 (rung R5: oracles and my own gradient check). The
oracle is the formula written out in numpy with the module's own weights.
They import only the contract."""

import math

import numpy as np
import pytest
import tinyllm.autograd.functional as F
from tinyllm.autograd.tensor import Tensor
from tinyllm.seq2seq.additive import AdditiveAttention, length_mask


def inputs(seed, B=3, S=5, Dq=4, Dk=6):
    r = np.random.Generator(np.random.PCG64(seed))
    return r.normal(size=(B, Dq)), r.normal(size=(B, S, Dk))


def oracle(att, q, k, lens):
    W, U, b, v = (p.data.astype(np.float64) for p in att.parameters())
    e = np.tanh((q @ W.T)[:, None, :] + k @ U.T + b) @ v[0]
    e = np.where(np.arange(k.shape[1])[None, :] < lens[:, None], e, -np.inf)
    a = np.exp(e - e.max(1, keepdims=True))
    a /= a.sum(1, keepdims=True)
    return np.einsum("bs,bsd->bd", a, k), a


def fd(f, x, eps=1e-6):
    g = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        a, b = x.copy(), x.copy()
        a[i] += eps
        b[i] -= eps
        g[i] = (f(a) - f(b)) / (2 * eps)
    return g


def test_hand_example():
    """W = U = v = 1, b = 0, q = 0, keys (0, 1, 2), third padded: weights (0.318, 0.682, 0)."""
    att = AdditiveAttention(1, 1, 1)
    att.load_state_dict(
        {
            "query.weight": [[1.0]],
            "key.weight": [[1.0]],
            "key.bias": [0.0],
            "v.weight": [[1.0]],
        }
    )
    ctx, a = att(
        Tensor([[0.0]]), Tensor([[[0.0], [1.0], [2.0]]]), [[True, True, False]]
    )
    w = math.exp(math.tanh(1)) / (1 + math.exp(math.tanh(1)))
    np.testing.assert_allclose(a.data, [[1 - w, w, 0.0]], rtol=1e-6)
    np.testing.assert_allclose(ctx.data, [[w]], rtol=1e-6)


def test_matches_the_formula():
    att = AdditiveAttention(4, 6, 5)
    q, k = inputs(1)
    lens = np.array([5, 2, 3])
    ctx, a = att(Tensor(q), Tensor(k), length_mask(lens, 5))
    wc, wa = oracle(att, q, k, lens)
    np.testing.assert_allclose(a.data, wa, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(ctx.data, wc, rtol=1e-4, atol=1e-6)
    proj = att.project_keys(Tensor(k))
    c2, a2 = att(Tensor(q), Tensor(k), length_mask(lens, 5), proj)
    np.testing.assert_array_equal(a2.data, a.data)


def test_gradients_of_query_and_keys():
    att = AdditiveAttention(3, 2, 4)
    for p in att.parameters():
        p.data = p.data.astype(np.float64)
    q, k = inputs(2, B=2, S=4, Dq=3, Dk=2)
    mask = length_mask(np.array([4, 2]), 4)

    def f(qq, kk):
        c, w = att(Tensor(qq, dtype=np.float64), Tensor(kk, dtype=np.float64), mask)
        return float(c.data.sum() + (w.data * np.arange(4)).sum())

    qt, kt = (
        Tensor(q, requires_grad=True, dtype=np.float64),
        Tensor(k, requires_grad=True, dtype=np.float64),
    )
    c, w = att(qt, kt, mask)
    (F.sum(c) + F.sum(w * np.arange(4.0))).backward()
    np.testing.assert_allclose(qt.grad, fd(lambda x: f(x, k), q), rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(kt.grad, fd(lambda x: f(q, x), k), rtol=1e-5, atol=1e-7)


def test_masking():
    att = AdditiveAttention(4, 6, 5)
    q, k = inputs(3)
    lens = np.array([5, 2, 0])
    assert length_mask(lens, 5).tolist()[1] == [True, True, False, False, False]
    kt = Tensor(k, requires_grad=True)
    c, a = att(Tensor(q), kt, length_mask(lens, 5))
    assert (
        np.all(a.data[1, 2:] == 0) and np.all(a.data[2] == 0) and np.all(c.data[2] == 0)
    )
    np.testing.assert_allclose(a.data[:2].sum(1), [1, 1], rtol=1e-6)
    F.sum(c).backward()
    assert np.all(kt.grad[1, 2:] == 0)


def test_parameters():
    att = AdditiveAttention(4, 6, 7)
    assert [(n, v.shape) for n, v in att.state_dict().items()] == [
        ("query.weight", (7, 4)),
        ("key.weight", (7, 6)),
        ("key.bias", (7,)),
        ("v.weight", (1, 7)),
    ]


def test_validation():
    att = AdditiveAttention(4, 6, 5)
    q, k = inputs(4)
    with pytest.raises(ValueError):
        att(Tensor(q), Tensor(k), np.ones((3, 5)))
    with pytest.raises(ValueError):
        length_mask(np.array([6]), 5)
