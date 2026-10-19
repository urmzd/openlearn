"""Model test suite for L3.3: what each test protects. Written test-first,
then the code. They import only the contract."""

import math

import numpy as np
import pytest
from tinyllm.autograd.tensor import Tensor
from tinyllm.rnn.gru import GRU, GRUCell, gru_cell


class Gen:
    def __init__(self, k=0):
        self.k = k

    def uniform(self):
        self.k += 1
        return (self.k * 0.6180339887) % 1.0

    def next_u32(self):
        return int(self.uniform() * 2**32)

    def uniforms(self, n):
        return np.array([self.uniform() for _ in range(n)])


def sig(v):
    return 1 / (1 + np.exp(-v))


def np_cell(x, h, wi, wh, bi, bh):
    a, b = x @ wi.T + bi, h @ wh.T + bh
    H = a.shape[-1] // 3
    r = sig(a[:, :H] + b[:, :H])
    z = sig(a[:, H : 2 * H] + b[:, H : 2 * H])
    n = np.tanh(a[:, 2 * H :] + r * b[:, 2 * H :])
    return (1 - z) * n + z * h


def test_hand_example_gru_cell():
    """r = 0.5, z = 0.75, n = tanh(0.5 * 1.5): h' = 0.25 n + 0.75 * 0.5."""
    w = [Tensor(a, dtype=np.float64) for a in ([[0.0]] * 3, [[0.0], [0.0], [1.0]], [0, math.log(3), 0], [0, 0, 1.0])]
    h2 = gru_cell(Tensor([[1.0]], dtype=np.float64), np.array([[0.5]]), *w)
    np.testing.assert_allclose(h2.data, [[0.25 * np.tanh(0.75) + 0.375]])


def test_cell_matches_numpy_formula():
    """Random weights with b_hn nonzero: r scales h w_hn^T + b_hn (torch's form)."""
    r = np.random.default_rng(0)
    B, D, H = 2, 3, 4
    a = [r.normal(size=s) for s in [(B, D), (B, H), (3 * H, D), (3 * H, H), (3 * H,), (3 * H,)]]
    np.testing.assert_allclose(gru_cell(*(Tensor(v, dtype=np.float64) for v in a)).data, np_cell(*a), rtol=1e-10)


def test_state_dict_names():
    """torch's names and order for two layers."""
    keys = list(GRU(3, 4, num_layers=2, rng=Gen()).state_dict())
    assert keys == [f"{n}_l{k}" for k in (0, 1) for n in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")]


def test_init_orthogonal_w_hh():
    """Each w_hh block is orthogonal."""
    H = 3
    sd = GRU(2, H, rng=Gen()).state_dict()
    for g in range(3):
        blk = sd["weight_hh_l0"][g * H : (g + 1) * H].astype(np.float64)
        np.testing.assert_allclose(blk @ blk.T, np.eye(H), atol=1e-5)


def test_lengths_zero_padding_and_frozen_state():
    """Past its length a sequence outputs 0 and keeps its state."""
    m = GRU(2, 3, rng=Gen())
    x = np.random.default_rng(1).normal(size=(4, 2, 2)).astype(np.float32)
    out, hn = m(x, lengths=np.array([4, 2]))
    assert (out.data[2:, 1] == 0).all()
    _, h1 = m(x[:2, 1:2])
    np.testing.assert_allclose(hn.data[0, 1], h1.data[0, 0], rtol=1e-5, atol=1e-6)


def test_two_layers_match_manual_stacking():
    """Layer 1 reads layer 0's outputs from its own h0."""
    m = GRU(2, 3, num_layers=2, rng=Gen())
    r = np.random.default_rng(2)
    x, h0 = r.normal(size=(3, 1, 2)), r.normal(size=(2, 1, 3))
    out, hn = m(x, h0)
    sd = {k: v.astype(np.float64) for k, v in m.state_dict().items()}
    seq = x
    for k in (0, 1):
        h, outs = h0[k], []
        for t in range(3):
            h = np_cell(seq[t], h, sd[f"weight_ih_l{k}"], sd[f"weight_hh_l{k}"], sd[f"bias_ih_l{k}"], sd[f"bias_hh_l{k}"])
            outs.append(h)
        seq = np.stack(outs)
        np.testing.assert_allclose(hn.data[k], h, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(out.data, seq, rtol=1e-4, atol=1e-5)


def test_dropout_only_between_layers():
    """A one-layer GRU ignores dropout, even in training mode."""
    x = np.random.default_rng(3).normal(size=(3, 2, 2)).astype(np.float32)
    np.testing.assert_allclose(GRU(2, 3, dropout=0.9, rng=Gen())(x)[0].data, GRU(2, 3, rng=Gen())(x)[0].data)


def test_bad_arguments_and_default_state():
    """Wrong shapes and lengths raise; no state means zeros."""
    m = GRU(2, 3, rng=Gen())
    x = np.zeros((3, 1, 2), np.float32)
    for f in (lambda: m(x, lengths=np.array([4])), lambda: GRU(2, 3, dropout=2.0), lambda: m(np.zeros((3, 1, 5)))):
        with pytest.raises(ValueError):
            f()
    cell = GRUCell(2, 3, rng=Gen())
    np.testing.assert_allclose(cell(np.ones((1, 2), np.float32)).data, cell(np.ones((1, 2), np.float32), np.zeros((1, 3), np.float32)).data)
