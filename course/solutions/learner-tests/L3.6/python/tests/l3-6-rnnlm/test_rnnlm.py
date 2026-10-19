"""My tests for L3.6 (rung R4: properties of stateful training, plus the
section 3 windows by hand). They import only the contract."""

import json
import math

import numpy as np
import pytest
import tinyllm.autograd.functional as F
from tinyllm.autograd.losses import cross_entropy
from tinyllm.autograd.tensor import Tensor
from tinyllm.rnn.manual import rnn_backward, rnn_forward
from tinyllm.rnn.rnnlm import (
    RNNLM,
    ElmanRNN,
    load_rnnlm,
    save_rnnlm,
    tbptt_batches,
    train_tbptt,
)


class Spy:
    """An optimizer that changes nothing and copies the gradients at each step."""

    def __init__(self, params):
        self.params, self.grads = list(params), []

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        self.grads.append([p.grad.copy() for p in self.params])


def ids(n, s=0):
    return np.random.Generator(np.random.PCG64(s)).integers(0, 9, size=n)


def lm(cell, layers=1):
    return RNNLM(9, 4, 5, cell, n_layers=layers)


def test_windows_by_hand():
    w = tbptt_batches(np.arange(20), 3, 2)
    assert [x.tolist() for x, _ in w] == [
        [[0, 1, 2], [10, 11, 12]],
        [[3, 4, 5], [13, 14, 15]],
        [[6, 7, 8], [16, 17, 18]],
    ]
    assert w[2][1].tolist() == [[7, 8, 9], [17, 18, 19]]


def test_elman_matches_manual():
    rnn = ElmanRNN(3, 4)
    for p in rnn.parameters():
        p.data = p.data.astype(np.float64) + 0.1
    x0 = np.random.Generator(np.random.PCG64(1)).normal(size=(4, 2, 3))
    x = Tensor(x0, requires_grad=True, dtype=np.float64)
    out, _ = rnn(x)
    w_ih, w_hh, b_ih, b_hh = (p.data for p in rnn.parameters())
    h, cache = rnn_forward(x0, np.zeros((2, 4)), w_ih.T, w_hh.T, b_ih + b_hh)
    np.testing.assert_allclose(out.data, h, rtol=1e-12)
    F.sum(out).backward()
    d = rnn_backward(np.ones_like(h), cache)
    np.testing.assert_allclose(rnn.weight_ih_l0.grad, d["Wxh"].T, rtol=1e-10)
    np.testing.assert_allclose(rnn.bias_hh_l0.grad, d["bh"], rtol=1e-10)


@pytest.mark.parametrize("cell", ["rnn", "lstm", "gru"])
def test_state_carries(cell):
    m = lm(cell, 2)
    x = ids(16).reshape(2, 8)
    whole, _ = m(x)
    a, st = m(x[:, :3])
    b, _ = m(x[:, 3:], st)
    np.testing.assert_allclose(
        np.concatenate([a.data, b.data], 1), whole.data, rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        m.nll(ids(20, 3), chunk=3), m.nll(ids(20, 3)), rtol=1e-5, atol=1e-6
    )


def test_second_window_gradient_is_cut_but_state_flows():
    data = ids(40, 4)
    m = lm("gru")
    spy = Spy(m.parameters())
    losses = train_tbptt(m, data, 4, 2, spy, None, 2)
    (x0, _), (x1, y1) = tbptt_batches(data, 4, 2)[:2]
    _, st = m(x0)
    m.zero_grad()
    logits, _ = m(x1, st.data)
    loss = cross_entropy(F.reshape(logits, (-1, 9)), y1.reshape(-1))
    loss.backward()
    assert math.isclose(float(loss.data), losses[1], rel_tol=1e-6)
    for g, p in zip(spy.grads[1], m.parameters()):
        np.testing.assert_allclose(g, p.grad, rtol=1e-5, atol=1e-6)


def test_epoch_restarts_from_zeros():
    data = ids(30, 5)
    m = lm("lstm")
    n = len(tbptt_batches(data, 4, 3))
    losses = train_tbptt(m, data, 4, 3, Spy(m.parameters()), None, n + 1)
    assert losses[n] == losses[0]


def test_clip():
    m = lm("rnn")
    spy = Spy(m.parameters())
    train_tbptt(m, ids(40, 6), 5, 2, spy, 0.01, 2)
    for gs in spy.grads:
        assert (
            math.sqrt(sum(float((g.astype(np.float64) ** 2).sum()) for g in gs))
            <= 0.0100001
        )


def test_generate_greedy_feeds_back():
    m = lm("lstm")
    got = m.generate([1, 2], 5, 0.0, 0)
    hist = [1, 2]
    for _ in range(5):
        hist.append(int(np.argmax(m(np.array([hist]))[0].data[0, -1])))
    assert got == hist[2:]


def test_save_load(tmp_path):
    m = lm("gru", 2)
    save_rnnlm(m, str(tmp_path), tokenizer="file")
    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg["tl_tokenizer"] == "file" and cfg["num_hidden_layers"] == 2
    x = ids(6).reshape(1, 6)
    np.testing.assert_array_equal(load_rnnlm(str(tmp_path))(x)[0].data, m(x)[0].data)
    with pytest.raises(ValueError):
        RNNLM(9, 4, 5, "cnn")
