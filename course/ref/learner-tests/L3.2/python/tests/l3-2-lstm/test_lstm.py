"""My tests for L3.2 (rung R3: tests first with `ol tdd red`, then the code
with `ol tdd green`). They import only the contract."""

import math

import numpy as np
import pytest
import tinyllm.autograd.functional as F
from tinyllm.autograd.tensor import Tensor
from tinyllm.rnn.lstm import LSTM, LSTMCell, lstm_cell


class Gen:
    """A deterministic stand-in generator: uniform() walks a fixed grid."""

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


def np_cell(x, h, c, wi, wh, bi, bh):
    z = x @ wi.T + bi + h @ wh.T + bh
    H = z.shape[-1] // 4
    i, f, g, o = sig(z[:, :H]), sig(z[:, H : 2 * H]), np.tanh(z[:, 2 * H : 3 * H]), sig(z[:, 3 * H :])
    c2 = f * c + i * g
    return o * np.tanh(c2), c2


def test_hand_example_lstm_cell():
    """i = 0.5, f = 0.75, g = 0.6, o = 0.75: c' = 0.675, h' = 0.75 tanh(0.675)."""
    w = [Tensor(a, dtype=np.float64) for a in ([[0.0], [0.0], [math.log(2)], [0.0]], [[0.0]] * 4, [0, math.log(3), 0, math.log(3)], [0.0] * 4)]
    h2, c2 = lstm_cell(Tensor([[1.0]], dtype=np.float64), np.zeros((1, 1)), np.array([[0.5]]), *w)
    np.testing.assert_allclose(c2.data, [[0.675]])
    np.testing.assert_allclose(h2.data, [[0.75 * np.tanh(0.675)]])


def test_cell_matches_numpy_formula():
    """Random weights with both biases nonzero: torch's gate order i, f, g, o."""
    r = np.random.default_rng(0)
    B, D, H = 2, 3, 4
    a = [r.normal(size=s) for s in [(B, D), (B, H), (B, H), (4 * H, D), (4 * H, H), (4 * H,), (4 * H,)]]
    h2, c2 = lstm_cell(*(Tensor(v, dtype=np.float64) for v in a))
    wh, wc = np_cell(*a)
    np.testing.assert_allclose(h2.data, wh, rtol=1e-10)
    np.testing.assert_allclose(c2.data, wc, rtol=1e-10)


def test_state_dict_names():
    """torch's names and order for two layers."""
    keys = list(LSTM(3, 4, num_layers=2, rng=Gen()).state_dict())
    assert keys == [f"{n}_l{k}" for k in (0, 1) for n in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")]


def test_init_forget_bias_and_orthogonal_w_hh():
    """Forget bias 1, other biases 0, each w_hh block orthogonal."""
    H = 3
    sd = LSTM(2, H, rng=Gen()).state_dict()
    want = np.zeros(4 * H)
    want[H : 2 * H] = 1
    np.testing.assert_array_equal(sd["bias_ih_l0"], want)
    for g in range(4):
        blk = sd["weight_hh_l0"][g * H : (g + 1) * H].astype(np.float64)
        np.testing.assert_allclose(blk @ blk.T, np.eye(H), atol=1e-5)


def test_lengths_zero_padding_and_frozen_state():
    """Past its length a sequence outputs 0 and keeps its state."""
    m = LSTM(2, 3, rng=Gen())
    x = np.random.default_rng(1).normal(size=(4, 2, 2)).astype(np.float32)
    out, (hn, cn) = m(x, lengths=np.array([4, 2]))
    assert (out.data[2:, 1] == 0).all()
    alone, (h1, c1) = m(x[:2, 1:2])
    np.testing.assert_allclose(hn.data[0, 1], h1.data[0, 0], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(cn.data[0, 1], c1.data[0, 0], rtol=1e-5, atol=1e-6)


def test_two_layers_match_manual_stacking():
    """Layer 1 reads layer 0's outputs from its own h0, c0; h_n stacks both."""
    m = LSTM(2, 3, num_layers=2, rng=Gen())
    r = np.random.default_rng(2)
    x = r.normal(size=(3, 1, 2))
    h0, c0 = r.normal(size=(2, 1, 3)), r.normal(size=(2, 1, 3))
    out, (hn, cn) = m(x, (h0, c0))
    sd = {k: v.astype(np.float64) for k, v in m.state_dict().items()}
    seq = x
    for k in (0, 1):
        h, c, outs = h0[k], c0[k], []
        for t in range(3):
            h, c = np_cell(seq[t], h, c, sd[f"weight_ih_l{k}"], sd[f"weight_hh_l{k}"], sd[f"bias_ih_l{k}"], sd[f"bias_hh_l{k}"])
            outs.append(h)
        seq = np.stack(outs)
        np.testing.assert_allclose(hn.data[k], h, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(cn.data[k], c, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(out.data, seq, rtol=1e-4, atol=1e-5)


def test_dropout_only_between_layers():
    """A one-layer LSTM ignores dropout, even in training mode."""
    x = np.random.default_rng(3).normal(size=(3, 2, 2)).astype(np.float32)
    a, b = LSTM(2, 3, dropout=0.9, rng=Gen()), LSTM(2, 3, rng=Gen())
    np.testing.assert_allclose(a(x)[0].data, b(x)[0].data)


def test_gradient_reaches_the_input():
    """Autograd through the sequence reaches every real input step."""
    m = LSTM(2, 3, rng=Gen())
    x = Tensor(np.random.default_rng(4).normal(size=(3, 1, 2)), requires_grad=True)
    out, _ = m(x)
    F.sum(out).backward()
    assert np.abs(x.grad).min() > 0


def test_bad_arguments_and_default_state():
    """Wrong shapes and lengths raise; no state means zeros."""
    m = LSTM(2, 3, rng=Gen())
    x = np.zeros((3, 1, 2), np.float32)
    for f in (lambda: m(x, lengths=np.array([4])), lambda: LSTM(2, 3, dropout=2.0), lambda: m(np.zeros((3, 1, 5)))):
        with pytest.raises(ValueError):
            f()
    cell = LSTMCell(2, 3, rng=Gen())
    h, _ = cell(np.ones((1, 2), np.float32))
    h0, _ = cell(np.ones((1, 2), np.float32), (np.zeros((1, 3), np.float32), np.zeros((1, 3), np.float32)))
    np.testing.assert_allclose(h.data, h0.data)
