"""My tests for L4.3 (rung R5: the three scores written out in numpy with the
module's own weights are the oracle, plus my own gradient check). They
import only the contract."""

import math

import numpy as np
import pytest
import tinyllm.autograd.functional as F
from tinyllm.autograd.tensor import Tensor
from tinyllm.seq2seq.additive import length_mask
from tinyllm.seq2seq.luong import LuongAttention


def inputs(seed, B=3, S=4, d=5):
    r = np.random.Generator(np.random.PCG64(seed))
    return r.normal(size=(B, d)), r.normal(size=(B, S, d))


def oracle(att, q, k, lens):
    sd = {n: v.astype(np.float64) for n, v in att.state_dict().items()}
    d = q.shape[1]
    if att.score == "dot":
        e = np.einsum("bsd,bd->bs", k, q)
    elif att.score == "general":
        e = np.einsum("bsd,bd->bs", k @ sd["score_proj.weight"].T, q)
    else:
        W = sd["score_proj.weight"]
        e = np.tanh((q @ W[:, :d].T)[:, None, :] + k @ W[:, d:].T) @ sd["v.weight"][0]
    e = np.where(np.arange(k.shape[1])[None, :] < lens[:, None], e, -np.inf)
    a = np.exp(e - e.max(1, keepdims=True))
    a /= a.sum(1, keepdims=True)
    c = np.einsum("bs,bsd->bd", a, k)
    return c, a, np.tanh(np.concatenate([c, q], 1) @ sd["combine.weight"].T)


def test_hand_example_dot():
    """h = (1, 0), keys (1,0), (0,1), (2,0) padded: weights (0.731, 0.269, 0)."""
    att = LuongAttention(2, "dot")
    att.load_state_dict({"combine.weight": [[1.0, 0, 1, 0], [0, 1, 0, 1]]})
    q = Tensor([[1.0, 0.0]])
    c, a = att(q, Tensor([[[1.0, 0], [0, 1], [2, 0]]]), [[True, True, False]])
    w = math.e / (math.e + 1)
    np.testing.assert_allclose(a.data, [[w, 1 - w, 0]], rtol=1e-6)
    np.testing.assert_allclose(
        att.attentional(q, c).data, [[math.tanh(1 + w), math.tanh(1 - w)]], rtol=1e-6
    )


@pytest.mark.parametrize("score", ["dot", "general", "concat"])
def test_matches_the_formula(score):
    att = LuongAttention(5, score)
    q, k = inputs(1)
    lens = np.array([4, 2, 3])
    qt = Tensor(q)
    c, a = att(qt, Tensor(k), length_mask(lens, 4))
    wc, wa, wh = oracle(att, q, k, lens)
    np.testing.assert_allclose(a.data, wa, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(c.data, wc, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(att.attentional(qt, c).data, wh, rtol=1e-4, atol=1e-6)
    c2, _ = att(qt, Tensor(k), length_mask(lens, 4), att.project_keys(Tensor(k)))
    np.testing.assert_allclose(c2.data, c.data, rtol=1e-6)


def test_gradient_of_keys():
    att = LuongAttention(3, "general")
    for p in att.parameters():
        p.data = p.data.astype(np.float64)
    q, k = inputs(2, B=2, S=3, d=3)
    mask = length_mask(np.array([3, 2]), 3)

    def f(kk):
        c, _ = att(Tensor(q, dtype=np.float64), Tensor(kk, dtype=np.float64), mask)
        return float(c.data.sum())

    kt = Tensor(k, requires_grad=True, dtype=np.float64)
    c, _ = att(Tensor(q, dtype=np.float64), kt, mask)
    F.sum(c).backward()
    num = np.zeros_like(k)
    for i in np.ndindex(k.shape):
        a, b = k.copy(), k.copy()
        a[i] += 1e-6
        b[i] -= 1e-6
        num[i] = (f(a) - f(b)) / 2e-6
    np.testing.assert_allclose(kt.grad, num, rtol=1e-5, atol=1e-7)


def test_fully_masked_row():
    att = LuongAttention(5, "dot")
    q, k = inputs(3)
    c, a = att(Tensor(q), Tensor(k), length_mask(np.array([0, 4, 1]), 4))
    assert (
        np.all(a.data[0] == 0) and np.all(c.data[0] == 0) and np.all(a.data[2, 1:] == 0)
    )


def test_parameters_and_validation():
    assert list(LuongAttention(4, "concat").state_dict()) == [
        "score_proj.weight",
        "v.weight",
        "combine.weight",
    ]
    with pytest.raises(ValueError):
        LuongAttention(4, "scaled")
