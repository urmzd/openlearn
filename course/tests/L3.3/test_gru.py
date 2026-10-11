"""Course tests for L3.3: the GRU in torch's gate order (tinyllm/rnn/gru.py).

Rung R0 for this file: read these before you write code. (Your own graded
tests, rung R3, go in python/tests/l3-3-gru/; see the chapter, section 4.)
Each test names why it exists (WHY), what kind of check it is (KIND), the
planted bugs it kills (CATCHES, mutants in course/mutants/L3.3), and the
chapter section it comes from.

The chapter's worked example (section 3): one unit, x = 1, h = 0.5,
w_ih = 0, b_ih = [0, ln 3, 0], w_hh = [0, 0, 1]^T, b_hh = [0, 0, 1]. Then
r = 0.5, z = 0.75, the recurrent candidate part is h w_hn + b_hn = 1.5,
n = tanh(0.5 * 1.5) = 0.635149, and h' = 0.25 n + 0.75 h = 0.533787; the
gradient of h' with respect to h is z + (1 - z)(1 - n^2) r w_hn = 0.824573.
Cho's original form, r applied to h before the matmul, would give
n = tanh(0.25 + 1) = 0.848284 instead.

Golden cases (course/fixtures/L3.3/gru_torch.npz) were recorded from torch
2.14.1 by course/oracle/L3.3/gru_torch.py. Gradient checks use the frozen
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
from tinyllm.rnn.gru import GRU, GRUCell, gru_cell

FIX = (
    Path(os.environ.get("TINYLLM_FIXTURES", "course/fixtures"))
    / "L3.3"
    / "gru_torch.npz"
)
F32 = dict(rtol=2e-5, atol=2e-6)


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


# --- the worked example ---------------------------------------------------------------


def test_hand_example_gru_cell():
    # WHY: the chapter's worked example, gate by gate in torch's order
    #      r, z, n: the reset gate scales the recurrent part AFTER its bias
    #      (r * (h w_hn + b_hn)), and z keeps the old state, h' = (1 - z) n +
    #      z h. The gradient into h is z plus the path through n.
    # KIND: unit
    # CATCHES: s01, s02, s03, s05, s12
    # CHAPTER: L3.3 section 3, Worked example by hand
    cell = GRUCell(1, 1, rng=Rng(seed()))
    cell.load_state_dict(
        {
            "weight_ih": [[0.0]] * 3,
            "weight_hh": [[0.0], [0.0], [1.0]],
            "bias_ih": [0.0, math.log(3), 0.0],
            "bias_hh": [0.0, 0.0, 1.0],
        }
    )
    h = Tensor([[0.5]], requires_grad=True)
    h2 = cell(np.array([[1.0]], dtype=np.float32), h)
    assert_close(h2.data, [[0.5337872380968218]], dtype="float32")
    h2.backward(np.ones((1, 1)))
    assert_close(h.grad, [[0.8245732260351665]], dtype="float32")


# --- against torch -------------------------------------------------------------------------


def _load(m, d, key, params):
    m.load_state_dict({p: d[f"{key}_p_{p}"] for p in params})


def _check_param_grads(m, d, key, params):
    got = dict(m.named_parameters())
    for p in params:
        assert_close(got[p].grad, d[f"{key}_gp_{p}"], **F32, msg=f"{key} grad of {p}")


def test_matches_torch_cell():
    # WHY: with torch's GRUCell weights loaded by name, one step gives
    #      torch's h' and torch's gradients for x, h, and all four
    #      parameters. The reset gate's placement (after b_hn) is the
    #      detail that most often breaks this.
    # KIND: golden
    # CATCHES: s01, s02, s03, s04
    # CHAPTER: L3.3 section 4, The interface
    d, meta = golden()
    params = next(c for c in meta["cases"] if c["name"] == "cell")["params"]
    cell = GRUCell(3, 4, rng=Rng(seed()))
    _load(cell, d, "cell", params)
    x, h = (
        Tensor(d["cell_x"], requires_grad=True),
        Tensor(d["cell_h"], requires_grad=True),
    )
    h2 = cell(x, h)
    assert_close(h2.data, d["cell_h2"], **F32)
    F.sum(h2 * d["cell_uh"]).backward()
    assert_close(x.grad, d["cell_gx"], **F32, msg="grad of x")
    assert_close(h.grad, d["cell_gh"], **F32, msg="grad of h")
    _check_param_grads(cell, d, "cell", params)


@pytest.mark.parametrize("name", case_names())
def test_matches_torch_sequence(name):
    # WHY: a whole sequence through torch.nn.GRU: one layer with a given h0,
    #      two stacked layers from zeros, and a padded batch run through
    #      pack_padded_sequence. Outputs, h_n, and every gradient match, so
    #      a torch-trained GRU loads and runs here (L4.1's default encoder).
    # KIND: golden
    # CATCHES: s01, s07, s08, s11
    # CHAPTER: L3.3 section 4, The interface
    d, meta = golden()
    c = next(c for c in meta["cases"] if c["name"] == name)
    m = GRU(3, 4, num_layers=c["layers"], rng=Rng(seed()))
    _load(m, d, name, c["params"])
    x = Tensor(d[f"{name}_x"], requires_grad=True)
    h0 = Tensor(d[f"{name}_h0"], requires_grad=True) if c["state"] else None
    lengths = d[f"{name}_lengths"] if c["packed"] else None
    out, hn = m(x, h0, lengths=lengths)
    assert_close(out.data, d[f"{name}_out"], **F32, msg="out")
    assert_close(hn.data, d[f"{name}_hn"], **F32, msg="h_n")
    (F.sum(out * d[f"{name}_g"]) + F.sum(hn * d[f"{name}_uh"])).backward()
    assert_close(x.grad, d[f"{name}_gx"], **F32, msg="grad of x")
    if h0 is not None:
        assert_close(h0.grad, d[f"{name}_gh0"], **F32, msg="grad of h0")
    _check_param_grads(m, d, name, c["params"])


def test_state_dict_names_match_torch():
    # WHY: torch's names and order (weight_ih_l0, weight_hh_l0, bias_ih_l0,
    #      bias_hh_l0, then _l1) and its [3H, D] shapes: the safetensors key
    #      contract for every GRU checkpoint in the zoo.
    # KIND: golden
    # CATCHES: s10
    # CHAPTER: L3.3 section 2, Principles (torch's layout)
    _, meta = golden()
    assert (
        list(GRU(3, 4, num_layers=2, rng=Rng(seed())).state_dict())
        == meta["gru2_state_dict_keys"]
    )
    assert (
        list(GRUCell(3, 4, rng=Rng(seed())).state_dict())
        == meta["cell_state_dict_keys"]
    )
    shapes = {
        k: v.shape
        for k, v in GRU(3, 4, num_layers=2, rng=Rng(seed())).state_dict().items()
    }
    assert shapes["weight_ih_l0"] == (12, 3) and shapes["weight_ih_l1"] == (12, 4)


# --- gradients -----------------------------------------------------------------------------


def test_gradcheck_gru_cell():
    # WHY: autograd through your gate arithmetic, in float64, against
    #      central differences of the forward alone, for all six inputs.
    # KIND: gradcheck
    # CATCHES: s12
    # CHAPTER: L3.3 section 2, Principles (the cell)
    g = PCG32(seed=seed() + 1)
    B, D, H = 2, 3, 2
    shapes = [(B, D), (B, H), (3 * H, D), (3 * H, H), (3 * H,), (3 * H,)]
    inputs = [g.normal_array(s, scale=0.7) for s in shapes]
    u = g.normal_array((B, H))

    def f(*arrs):
        return float(
            (gru_cell(*(Tensor(a, dtype=np.float64) for a in arrs)).data * u).sum()
        )

    ts = [Tensor(a, requires_grad=True, dtype=np.float64) for a in inputs]
    F.sum(gru_cell(*ts) * u).backward()
    gradcheck(
        f,
        inputs,
        [t.grad for t in ts],
        names=["x", "h", "w_ih", "w_hh", "b_ih", "b_hh"],
    )


def test_update_gate_near_one_keeps_the_state():
    # WHY: with the update gate saturated at 1 (a large z bias), h' = h for
    #      any input and the gradient of h_T with respect to h_0 is the
    #      identity: the GRU's way of carrying information, and gradient,
    #      across many steps.
    # KIND: property
    # CATCHES: s03
    # CHAPTER: L3.3 section 2, Principles (gates as interpolation)
    T, B, H = 15, 2, 3
    m = GRU(2, H, rng=Rng(seed() + 2))
    sd = m.state_dict()
    sd["bias_ih_l0"][H : 2 * H] = 40.0
    m.load_state_dict(sd)
    x = PCG32(seed=seed() + 3).normal_array((T, B, 2)).astype(np.float32)
    h0 = Tensor(
        PCG32(seed=seed() + 4).normal_array((1, B, H), scale=0.5),
        requires_grad=True,
        dtype=np.float64,
    )
    out, hn = m(x, h0)
    assert_close(hn.data, h0.data, rtol=1e-6, atol=1e-7)
    F.sum(hn).backward()
    assert_close(h0.grad, np.ones((1, B, H)), rtol=1e-6, atol=1e-7)


# --- initialization ------------------------------------------------------------------------


def test_init_orthogonal_recurrence_and_zero_bias():
    # WHY: the contract's initialization: both biases 0, each [H, H] block of
    #      w_hh orthogonal, each block of w_ih inside Xavier's bound, and the
    #      same seed giving the same weights (None means
    #      PCG32(0).substream("init")).
    # KIND: unit
    # CATCHES: s06
    # CHAPTER: L3.3 section 2, Principles (initialization)
    D, H = 3, 5
    sd = GRU(D, H, num_layers=2, rng=Rng(seed())).state_dict()
    for k in (0, 1):
        assert_close(sd[f"bias_ih_l{k}"], np.zeros(3 * H), dtype="float32")
        assert_close(sd[f"bias_hh_l{k}"], np.zeros(3 * H), dtype="float32")
        for gate in range(3):
            blk = sd[f"weight_hh_l{k}"][gate * H : (gate + 1) * H].astype(np.float64)
            assert_close(blk @ blk.T, np.eye(H), rtol=0, atol=1e-5)
            bound = math.sqrt(6.0 / (H + (D if k == 0 else H)))
            wi = sd[f"weight_ih_l{k}"][gate * H : (gate + 1) * H]
            assert np.abs(wi).max() <= bound and np.abs(wi).max() > 0.3 * bound
    again = GRU(D, H, num_layers=2, rng=Rng(seed())).state_dict()
    assert all((again[k] == v).all() for k, v in sd.items())
    d1, d2 = GRU(D, H).state_dict(), GRU(D, H).state_dict()
    assert all((d1[k] == d2[k]).all() for k in d1)


# --- lengths, dropout, boundaries --------------------------------------------------------------


def test_lengths_keep_padding_out():
    # WHY: in a padded batch, sequence b stops at lengths[b]: its outputs
    #      past that are 0, h_n is the state after its own last step (the
    #      same as running it alone), and garbage in the padding changes
    #      nothing, forward or backward.
    # KIND: boundary
    # CATCHES: s07, s08
    # CHAPTER: L3.3 section 5, Pitfalls
    T, B, D, H = 5, 3, 2, 3
    lengths = np.array([3, 5, 1])
    m = GRU(D, H, rng=Rng(seed() + 5))
    x = PCG32(seed=seed() + 6).normal_array((T, B, D)).astype(np.float32)
    for b, n in enumerate(lengths):
        x[n:, b] = -1e3
    xt = Tensor(x, requires_grad=True)
    out, hn = m(xt, lengths=lengths)
    for b, n in enumerate(lengths):
        assert (out.data[n:, b] == 0).all()
        alone, h1 = m(x[:n, b : b + 1])
        assert_close(out.data[:n, b], alone.data[:, 0], **F32)
        assert_close(hn.data[:, b], h1.data[:, 0], **F32)
    (F.sum(out) + F.sum(hn)).backward()
    for b, n in enumerate(lengths):
        assert (xt.grad[n:, b] == 0).all()


def test_dropout_between_layers_only():
    # WHY: dropout on each layer's output except the last, in training mode
    #      only, as torch: eval mode equals dropout 0, and a 1-layer GRU
    #      ignores its dropout.
    # KIND: unit
    # CATCHES: s09
    # CHAPTER: L3.3 section 4, The interface
    x = PCG32(seed=seed() + 7).normal_array((4, 2, 3)).astype(np.float32)
    a = GRU(3, 4, num_layers=2, dropout=0.5, rng=Rng(seed()))
    b = GRU(3, 4, num_layers=2, rng=Rng(seed()))
    assert np.abs(a(x)[0].data - b(x)[0].data).max() > 1e-4
    a.eval()
    assert_close(a(x)[0].data, b(x)[0].data, dtype="float32")
    one, plain = GRU(3, 4, dropout=0.9, rng=Rng(seed())), GRU(3, 4, rng=Rng(seed()))
    assert_close(one(x)[0].data, plain(x)[0].data, dtype="float32")


def test_default_state_and_bad_arguments():
    # WHY: no state means zeros; a malformed input, state, or lengths is a
    #      bug to report at the call.
    # KIND: boundary
    # CATCHES: m01, m02, m03
    # CHAPTER: L3.3 section 4, The interface
    m = GRU(3, 4, num_layers=2, rng=Rng(seed()))
    x = PCG32(seed=seed() + 8).normal_array((3, 2, 3)).astype(np.float32)
    assert_close(
        m(x)[0].data, m(x, np.zeros((2, 2, 4), np.float32))[0].data, dtype="float32"
    )
    cell = GRUCell(3, 4, rng=Rng(seed()))
    assert_close(
        cell(x[0]).data, cell(x[0], np.zeros((2, 4), np.float32)).data, dtype="float32"
    )
    bad = [
        lambda: m(np.zeros((3, 2, 5), np.float32)),
        lambda: m(np.zeros((2, 3), np.float32)),
        lambda: m(x, np.zeros((1, 2, 4))),
        lambda: m(x, lengths=np.array([4, 1])),
        lambda: m(x, lengths=np.array([0, 1])),
        lambda: m(x, lengths=np.array([1.0, 2.0])),
        lambda: GRU(0, 4),
        lambda: GRU(3, 4, num_layers=0),
        lambda: GRU(3, 4, dropout=-0.1),
        lambda: GRUCell(0, 4),
    ]
    for f in bad:
        with pytest.raises(ValueError):
            f()
