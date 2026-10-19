"""My tests for L5.3 (rung R5: multi-head attention written out in numpy
from the module's own weights is the oracle, plus my own gradient check).
They import only the contract."""

import math

import numpy as np
import pytest
import tinyllm.autograd.functional as F
from tinyllm.autograd.tensor import Tensor
from tinyllm.num.gradcheck import gradcheck
from tinyllm.xfmr.mha import MultiHeadAttention


def rnd(seed, *shape):
    return np.random.Generator(np.random.PCG64(seed)).normal(size=shape)


def oracle(m, xq, xkv, mask4):
    sd = {n: v.astype(np.float64) for n, v in m.state_dict().items()}
    H, dh = m.n_heads, m.d_head

    def proj(x, n):
        return x @ sd[n + ".weight"].T + sd.get(n + ".bias", 0.0)

    def split(x):
        B, T, _ = x.shape
        return x.reshape(B, T, H, dh).transpose(0, 2, 1, 3)

    q, k, v = (
        split(proj(xq, "q_proj")),
        split(proj(xkv, "k_proj")),
        split(proj(xkv, "v_proj")),
    )
    e = q @ k.transpose(0, 1, 3, 2) / math.sqrt(dh)
    e = np.where(mask4, e, -np.inf)
    a = np.exp(e - e.max(-1, keepdims=True))
    a /= a.sum(-1, keepdims=True)
    o = (a @ v).transpose(0, 2, 1, 3).reshape(xq.shape[0], xq.shape[1], -1)
    return proj(o, "out_proj"), a


def test_hand_example_two_heads():
    m = MultiHeadAttention(4, 2)
    m.load_state_dict(
        {
            n: (np.eye(4) if n.endswith("weight") else np.zeros(4))
            for n in m.state_dict()
        }
    )
    out, w = m.attend(
        Tensor([[[1.0, 0, 0, 2]]]), Tensor([[[1.0, 0, 0, 0], [0, 1, 0, 2]]])
    )
    s0, s1 = 1 / (1 + math.exp(-1 / math.sqrt(2))), 1 / (1 + math.exp(4 / math.sqrt(2)))
    np.testing.assert_allclose(w.data[0, :, 0], [[s0, 1 - s0], [s1, 1 - s1]], rtol=1e-5)
    np.testing.assert_allclose(
        out.data[0, 0], [s0, 1 - s0, 0, 2 * (1 - s1)], rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize("Tq,Tk", [(3, 3), (3, 5)])
def test_matches_numpy_with_per_sequence_masks(Tq, Tk):
    m = MultiHeadAttention(6, 2)
    xq, xkv = rnd(1, 2, Tq, 6), rnd(2, 2, Tk, 6)
    mask = np.ones((2, Tq, Tk), dtype=bool)
    mask[0] = np.tril(np.ones((Tq, Tk), dtype=bool))
    mask[1, :, 3:] = False
    out, w = m.attend(Tensor(xq), Tensor(xkv), mask)
    want, wa = oracle(m, xq, xkv, mask[:, None])
    np.testing.assert_allclose(out.data, want, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(w.data, wa, rtol=1e-4, atol=1e-5)


def test_split_merge_roundtrip():
    m = MultiHeadAttention(6, 3)
    x = rnd(3, 2, 4, 6)
    s = m.split_heads(Tensor(x))
    np.testing.assert_array_equal(s.data[:, 1], x[:, :, 2:4].astype(np.float32))
    np.testing.assert_array_equal(m.merge_heads(s).data, x.astype(np.float32))


def test_gradcheck():
    m = MultiHeadAttention(4, 2)
    ps = [p for _, p in m.named_parameters()]
    for p in ps:
        p.data = p.data.astype(np.float64)
    xq0, xkv0, g = rnd(4, 1, 2, 4), rnd(5, 1, 3, 4), rnd(6, 1, 2, 4)

    def f(xq, xkv, *vals):
        for p, v in zip(ps, vals):
            p.data = v
        return float(
            np.sum(
                m(Tensor(xq, dtype=np.float64), Tensor(xkv, dtype=np.float64)).data * g
            )
        )

    start = [p.data.copy() for p in ps]
    xq, xkv = (
        Tensor(xq0, requires_grad=True, dtype=np.float64),
        Tensor(xkv0, requires_grad=True, dtype=np.float64),
    )
    F.sum(m(xq, xkv) * g).backward()
    rep = gradcheck(f, [xq0, xkv0] + start, [xq.grad, xkv.grad] + [p.grad for p in ps])
    assert rep.ok, rep


def test_dropout_only_when_training():
    a, b = MultiHeadAttention(4, 2, dropout=0.5), MultiHeadAttention(4, 2)
    xq, xkv = Tensor(rnd(7, 2, 3, 4)), Tensor(rnd(8, 2, 3, 4))
    a.eval()
    np.testing.assert_array_equal(a(xq, xkv).data, b(xq, xkv).data)


def test_packed_in_proj_is_q_k_v():
    m = MultiHeadAttention(2, 1)
    w = np.arange(12.0).reshape(6, 2)
    m.load_packed_in_proj(w, np.arange(6.0))
    np.testing.assert_array_equal(m.q_proj.weight.data, w[:2])
    np.testing.assert_array_equal(m.k_proj.weight.data, w[2:4])


def test_names_and_validation():
    assert list(MultiHeadAttention(4, 2).state_dict()) == [
        f"{p}.{w}"
        for p in ("q_proj", "k_proj", "v_proj", "out_proj")
        for w in ("weight", "bias")
    ]
    assert list(MultiHeadAttention(4, 2, bias=False).state_dict())[1] == "k_proj.weight"
    for bad in (
        lambda: MultiHeadAttention(6, 4),
        lambda: MultiHeadAttention(4, 2, dropout=1.0),
    ):
        with pytest.raises(ValueError):
            bad()
