"""Course tests for L3.2: the LSTM in torch's gate order (tinyllm/rnn/lstm.py).

Rung R0 for this file: read these before you write code. (Your own graded
tests, rung R3, go in python/tests/l3-2-lstm/; see the chapter, section 4.)
Each test names why it exists (WHY), what kind of check it is (KIND), the
planted bugs it kills (CATCHES, mutants in course/mutants/L3.2), and the
chapter section it comes from.

The chapter's worked example (section 3): one unit, x = 1, h = 0, c = 0.5,
w_ih = [0, 0, ln 2, 0]^T, w_hh = 0, b_ih = [0, ln 3, 0, ln 3], b_hh = 0.
Then i = 0.5, f = 0.75, g = tanh(ln 2) = 0.6, o = 0.75, c' = 0.75 * 0.5 +
0.5 * 0.6 = 0.675, h' = 0.75 * tanh(0.675) = 0.441194, and the gradient of
h' with respect to the old c is o (1 - tanh^2 c') f = 0.367847.

Golden cases (course/fixtures/L3.2/lstm_torch.npz) were recorded from torch
2.14.1 by course/oracle/L3.2/lstm_torch.py. Gradient checks use the frozen
central differences (course/tests/_lib/gradcheck.py).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.gradcheck import gradcheck
from _lib.pcg32 import PCG32
from tinyllm.autograd import functional as F
from tinyllm.autograd.tensor import Tensor
from tinyllm.rnn.lstm import LSTM, LSTMCell, lstm_cell

FIX = (
    Path(os.environ.get("TINYLLM_FIXTURES", "course/fixtures"))
    / "L3.2"
    / "lstm_torch.npz"
)
F32 = dict(rtol=2e-5, atol=2e-6)  # float32 through a few steps of gates


def seed() -> int:
    return int(os.environ.get("OL_SEED", "0"))


class Rng:
    """The frozen PCG32 behind the generator API the initializers use."""

    def __init__(self, s: int) -> None:
        self.g = PCG32(seed=s)

    def next_u32(self) -> int:
        return self.g.next_u32()

    def uniform(self) -> float:
        return self.g.uniform()

    def uniforms(self, n: int) -> np.ndarray:
        return np.array([self.g.uniform() for _ in range(n)], dtype=np.float64)


def golden():
    d = np.load(FIX, allow_pickle=False)
    return d, json.loads(str(d["__meta__"]))


def case_names() -> list[str]:
    try:
        return [c["name"] for c in golden()[1]["cases"] if c["name"] != "cell"]
    except OSError:
        return ["missing-fixture"]


def hand_cell() -> LSTMCell:
    cell = LSTMCell(1, 1, rng=Rng(seed()))
    cell.load_state_dict(
        {
            "weight_ih": [[0.0], [0.0], [math.log(2)], [0.0]],
            "weight_hh": [[0.0]] * 4,
            "bias_ih": [0.0, math.log(3), 0.0, math.log(3)],
            "bias_hh": [0.0] * 4,
        }
    )
    return cell


# --- the worked example ---------------------------------------------------------------


def test_hand_example_lstm_cell():
    # WHY: the chapter's worked example, gate by gate in torch's order
    #      i, f, g, o: the candidate g goes through tanh, the three gates
    #      through sigmoid, c' = f c + i g, h' = o tanh(c'). The gradient into
    #      the old cell state is scaled by f (0.75), the forget gate.
    # KIND: unit
    # CATCHES: s01, s02, s03, s04
    # CHAPTER: L3.2 section 3, Worked example by hand
    cell = hand_cell()
    c = Tensor([[0.5]], requires_grad=True)
    h2, c2 = cell(
        np.array([[1.0]], dtype=np.float32), (np.zeros((1, 1), np.float32), c)
    )
    assert_close(c2.data, [[0.675]], dtype="float32")
    assert_close(h2.data, [[0.44119444229857907]], dtype="float32")
    h2.backward(np.ones((1, 1)))
    assert_close(c.grad, [[0.3678474640848458]], dtype="float32")


# --- against torch -------------------------------------------------------------------------


def _load(m, d, key, params):
    m.load_state_dict({p: d[f"{key}_p_{p}"] for p in params})


def _check_param_grads(m, d, key, params):
    got = dict(m.named_parameters())
    for p in params:
        assert_close(got[p].grad, d[f"{key}_gp_{p}"], **F32, msg=f"{key} grad of {p}")


def test_matches_torch_cell():
    # WHY: with torch's LSTMCell weights loaded by name, one step gives
    #      torch's (h', c') and torch's gradients for x, h, c, and all four
    #      parameters. Gate order, both biases, and the [4H, D] layout all
    #      have to be torch's for this to hold.
    # KIND: golden
    # CATCHES: s01, s02, s03, s04, s09
    # CHAPTER: L3.2 section 4, The interface
    d, meta = golden()
    params = next(c for c in meta["cases"] if c["name"] == "cell")["params"]
    cell = LSTMCell(3, 4, rng=Rng(seed()))
    _load(cell, d, "cell", params)
    x, h, c = (Tensor(d[f"cell_{k}"], requires_grad=True) for k in "xhc")
    h2, c2 = cell(x, (h, c))
    assert_close(h2.data, d["cell_h2"], **F32)
    assert_close(c2.data, d["cell_c2"], **F32)
    (F.sum(h2 * d["cell_uh"]) + F.sum(c2 * d["cell_uc"])).backward()
    for k, t in zip("xhc", (x, h, c)):
        assert_close(t.grad, d[f"cell_g{k}"], **F32, msg=f"grad of {k}")
    _check_param_grads(cell, d, "cell", params)


@pytest.mark.parametrize("name", case_names())
def test_matches_torch_sequence(name):
    # WHY: a whole sequence through torch.nn.LSTM: one layer with a given
    #      (h0, c0), two stacked layers from zeros, and a padded batch run
    #      through pack_padded_sequence. Outputs, final states, and every
    #      gradient match, so a torch-trained LSTM loads and runs here
    #      (L3.6, L4.1).
    # KIND: golden
    # CATCHES: s01, s07, s08, s09, s11, s14
    # CHAPTER: L3.2 section 4, The interface
    d, meta = golden()
    c = next(c for c in meta["cases"] if c["name"] == name)
    m = LSTM(3, 4, num_layers=c["layers"], rng=Rng(seed()))
    _load(m, d, name, c["params"])
    x = Tensor(d[f"{name}_x"], requires_grad=True)
    state = None
    if c["state"]:
        state = (
            Tensor(d[f"{name}_h0"], requires_grad=True),
            Tensor(d[f"{name}_c0"], requires_grad=True),
        )
    lengths = d[f"{name}_lengths"] if c["packed"] else None
    out, (hn, cn) = m(x, state, lengths=lengths)
    assert_close(out.data, d[f"{name}_out"], **F32, msg="out")
    assert_close(hn.data, d[f"{name}_hn"], **F32, msg="h_n")
    assert_close(cn.data, d[f"{name}_cn"], **F32, msg="c_n")
    (
        F.sum(out * d[f"{name}_g"])
        + F.sum(hn * d[f"{name}_uh"])
        + F.sum(cn * d[f"{name}_uc"])
    ).backward()
    assert_close(x.grad, d[f"{name}_gx"], **F32, msg="grad of x")
    if state is not None:
        assert_close(state[0].grad, d[f"{name}_gh0"], **F32, msg="grad of h0")
        assert_close(state[1].grad, d[f"{name}_gc0"], **F32, msg="grad of c0")
    _check_param_grads(m, d, name, c["params"])


def test_state_dict_names_match_torch():
    # WHY: the parameter names and their order are torch's
    #      (weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0, then _l1),
    #      so a checkpoint written by L0.6 from your LSTM and one written by
    #      torch have the same safetensors keys.
    # KIND: golden
    # CATCHES: s12
    # CHAPTER: L3.2 section 2, Principles (torch's layout)
    _, meta = golden()
    assert (
        list(LSTM(3, 4, num_layers=2, rng=Rng(seed())).state_dict())
        == meta["lstm2_state_dict_keys"]
    )
    assert (
        list(LSTMCell(3, 4, rng=Rng(seed())).state_dict())
        == meta["cell_state_dict_keys"]
    )
    shapes = {
        k: v.shape
        for k, v in LSTM(3, 4, num_layers=2, rng=Rng(seed())).state_dict().items()
    }
    assert shapes["weight_ih_l0"] == (16, 3) and shapes["weight_ih_l1"] == (16, 4)
    assert shapes["weight_hh_l1"] == (16, 4) and shapes["bias_hh_l1"] == (16,)


# --- gradients -----------------------------------------------------------------------------


def test_gradcheck_lstm_cell():
    # WHY: autograd through your gate arithmetic, in float64, against
    #      central differences of the forward alone, for all seven inputs.
    # KIND: gradcheck
    # CATCHES: s09
    # CHAPTER: L3.2 section 2, Principles (the cell)
    g = PCG32(seed=seed() + 1)
    B, D, H = 2, 3, 2
    shapes = [(B, D), (B, H), (B, H), (4 * H, D), (4 * H, H), (4 * H,), (4 * H,)]
    inputs = [g.normal_array(s, scale=0.7) for s in shapes]
    uh, uc = g.normal_array((B, H)), g.normal_array((B, H))

    def f(*arrs):
        h2, c2 = lstm_cell(*(Tensor(a, dtype=np.float64) for a in arrs))
        return float((h2.data * uh).sum() + (c2.data * uc).sum())

    ts = [Tensor(a, requires_grad=True, dtype=np.float64) for a in inputs]
    h2, c2 = lstm_cell(*ts)
    (F.sum(h2 * uh) + F.sum(c2 * uc)).backward()
    names = ["x", "h", "c", "w_ih", "w_hh", "b_ih", "b_hh"]
    gradcheck(f, inputs, [t.grad for t in ts], names=names)


def test_cell_state_is_a_gradient_highway():
    # WHY: with w_hh = 0 the only path from c_0 to c_T is the cell state, and
    #      along it dc_T / dc_0 is exactly the product of the forget gates
    #      over the T steps. With f near 0.95 that is 0.36 after 20 steps,
    #      where a tanh RNN's factor would have vanished: this is why the
    #      LSTM remembers.
    # KIND: property
    # CATCHES: s15
    # CHAPTER: L3.2 section 2, Principles (the constant error carousel)
    T, B, H = 20, 1, 3
    m = LSTM(2, H, rng=Rng(seed() + 2))
    sd = m.state_dict()
    sd["weight_hh_l0"] = np.zeros_like(sd["weight_hh_l0"])
    sd["bias_ih_l0"][H : 2 * H] = 3.0
    m.load_state_dict(sd)
    x = PCG32(seed=seed() + 3).normal_array((T, B, 2), scale=0.3).astype(np.float32)
    c0 = Tensor(np.zeros((1, B, H)), requires_grad=True)
    _, (_, cn) = m(x, (np.zeros((1, B, H), np.float32), c0))
    F.sum(cn).backward()
    w, b = sd["weight_ih_l0"].astype(np.float64), sd["bias_ih_l0"].astype(np.float64)
    zf = x.astype(np.float64)[:, 0, :] @ w[H : 2 * H].T + b[H : 2 * H]
    want = np.prod(1.0 / (1.0 + np.exp(-zf)), axis=0)
    assert_close(c0.grad[0, 0], want, rtol=1e-4, atol=1e-7)
    assert (want > 0.2).all()


# --- initialization ------------------------------------------------------------------------


def test_init_forget_bias_and_orthogonal_recurrence():
    # WHY: the contract's initialization: forget-gate bias 1 (and every other
    #      bias 0), each [H, H] block of w_hh orthogonal (singular values all
    #      1, so the recurrence neither blows up nor shrinks the state at the
    #      start), each block of w_ih inside Xavier's bound, and the same seed
    #      giving the same weights (None means PCG32(0).substream("init")).
    # KIND: unit
    # CATCHES: s05, s06, s13
    # CHAPTER: L3.2 section 2, Principles (initialization)
    D, H = 3, 5
    m = LSTM(D, H, num_layers=2, rng=Rng(seed()))
    sd = m.state_dict()
    for k in (0, 1):
        want = np.zeros(4 * H)
        want[H : 2 * H] = 1.0
        assert_close(sd[f"bias_ih_l{k}"], want, dtype="float32")
        assert_close(sd[f"bias_hh_l{k}"], np.zeros(4 * H), dtype="float32")
        for gate in range(4):
            blk = sd[f"weight_hh_l{k}"][gate * H : (gate + 1) * H].astype(np.float64)
            assert_close(blk @ blk.T, np.eye(H), rtol=0, atol=1e-5)
            d_in = D if k == 0 else H
            bound = math.sqrt(6.0 / (H + d_in))
            wi = sd[f"weight_ih_l{k}"][gate * H : (gate + 1) * H]
            assert np.abs(wi).max() <= bound and np.abs(wi).max() > 0.3 * bound
    again = LSTM(D, H, num_layers=2, rng=Rng(seed())).state_dict()
    assert all((again[k] == v).all() for k, v in sd.items())
    other = LSTM(D, H, num_layers=2, rng=Rng(seed() + 1)).state_dict()
    assert not (other["weight_ih_l0"] == sd["weight_ih_l0"]).all()
    d1, d2 = LSTM(D, H).state_dict(), LSTM(D, H).state_dict()
    assert all((d1[k] == d2[k]).all() for k in d1)


# --- lengths, dropout, boundaries --------------------------------------------------------------


def test_lengths_keep_padding_out():
    # WHY: in a padded batch, sequence b stops at lengths[b]: its outputs
    #      past that are 0, its final (h, c) is the state after its own last
    #      step (the same as running it alone), and garbage in the padding
    #      changes nothing, forward or backward.
    # KIND: boundary
    # CATCHES: s07, s08
    # CHAPTER: L3.2 section 5, Pitfalls
    T, B, D, H = 5, 3, 2, 3
    lengths = np.array([5, 2, 3])
    m = LSTM(D, H, rng=Rng(seed() + 4))
    x = PCG32(seed=seed() + 5).normal_array((T, B, D)).astype(np.float32)
    for b, n in enumerate(lengths):
        x[n:, b] = 1e3  # garbage in the padding
    xt = Tensor(x, requires_grad=True)
    out, (hn, cn) = m(xt, lengths=lengths)
    for b, n in enumerate(lengths):
        assert (out.data[n:, b] == 0).all()
        alone, (h1, c1) = m(x[:n, b : b + 1])
        assert_close(out.data[:n, b], alone.data[:, 0], **F32)
        assert_close(hn.data[:, b], h1.data[:, 0], **F32)
        assert_close(cn.data[:, b], c1.data[:, 0], **F32)
    (F.sum(out) + F.sum(hn) + F.sum(cn)).backward()
    for b, n in enumerate(lengths):
        assert (xt.grad[n:, b] == 0).all()


def test_dropout_between_layers_only():
    # WHY: torch applies dropout to each layer's output sequence except the
    #      last one's, and only in training mode: a 2-layer LSTM in eval mode
    #      equals the same weights with dropout 0, and a 1-layer LSTM ignores
    #      its dropout entirely.
    # KIND: unit
    # CATCHES: s10
    # CHAPTER: L3.2 section 4, The interface
    x = PCG32(seed=seed() + 6).normal_array((4, 2, 3)).astype(np.float32)
    a = LSTM(3, 4, num_layers=2, dropout=0.5, rng=Rng(seed()))
    b = LSTM(3, 4, num_layers=2, dropout=0.0, rng=Rng(seed()))
    train_out, _ = a(x)
    assert np.abs(train_out.data - b(x)[0].data).max() > 1e-4
    a.eval()
    assert_close(a(x)[0].data, b(x)[0].data, dtype="float32")
    one = LSTM(3, 4, dropout=0.9, rng=Rng(seed()))
    plain = LSTM(3, 4, rng=Rng(seed()))
    assert_close(one(x)[0].data, plain(x)[0].data, dtype="float32")


def test_default_state_and_bad_arguments():
    # WHY: no state means zeros; a malformed input, state, or lengths is a
    #      bug to report at the call, not to broadcast into a silent wrong
    #      answer.
    # KIND: boundary
    # CATCHES: m01, m02, m03
    # CHAPTER: L3.2 section 4, The interface
    m = LSTM(3, 4, num_layers=2, rng=Rng(seed()))
    x = PCG32(seed=seed() + 7).normal_array((3, 2, 3)).astype(np.float32)
    z = np.zeros((2, 2, 4), np.float32)
    assert_close(m(x)[0].data, m(x, (z, z))[0].data, dtype="float32")
    cell = LSTMCell(3, 4, rng=Rng(seed()))
    h, c = cell(x[0])
    h0, c0 = cell(x[0], (np.zeros((2, 4), np.float32), np.zeros((2, 4), np.float32)))
    assert_close(h.data, h0.data, dtype="float32")
    bad = [
        lambda: m(np.zeros((3, 2, 5), np.float32)),
        lambda: m(np.zeros((2, 3), np.float32)),
        lambda: m(x, (np.zeros((1, 2, 4)), np.zeros((1, 2, 4)))),
        lambda: m(x, lengths=np.array([4, 1])),
        lambda: m(x, lengths=np.array([0, 1])),
        lambda: m(x, lengths=np.array([1, 2, 3])),
        lambda: m(x, lengths=np.array([1.0, 2.0])),
        lambda: LSTM(0, 4),
        lambda: LSTM(3, 4, num_layers=0),
        lambda: LSTM(3, 4, dropout=1.5),
        lambda: LSTMCell(3, 0),
    ]
    for i, f in enumerate(bad):
        with pytest.raises(ValueError):
            f()
