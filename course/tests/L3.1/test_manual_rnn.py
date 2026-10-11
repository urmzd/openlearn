"""Course tests for L3.1: a vanilla RNN with manual BPTT and truncation
(tinyllm/rnn/manual.py).

Rung R0 for this file: read these before you write code. (Your own graded
tests, rung R3, go in python/tests/l3-1-rnn/; see the chapter, section 4.)
Each test names why it exists (WHY), what kind of check it is (KIND), the
planted bugs it kills (CATCHES, mutants in course/mutants/L3.1), and the
chapter section it comes from.

The chapter's worked example (section 3): D = H = B = 1, Wxh = 0.5,
Whh = 0.8, bh = 0, h0 = 0, x = [1, 0]. Forward: h_0 = tanh(0.5) = 0.462117,
h_1 = tanh(0.8 h_0) = 0.353724. With the loss L = h_1: da_1 = 1 - h_1^2 =
0.874879, the carry back to h_0 is 0.8 da_1 = 0.699904, da_0 = 0.699904 *
(1 - h_0^2) = 0.550438, so dWxh = 0.550438, dWhh = h_0 da_1 = 0.404297,
dbh = 1.425317, dx = [0.275219, 0.437440], dh0 = 0.440350.

The torch golden case (course/fixtures/L3.1/rnn_torch.npz) was recorded from
torch 2.14.1 by course/oracle/L3.1/rnn_torch.py. Gradient checks use the
frozen central differences (course/tests/_lib/gradcheck.py), never yours.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.gradcheck import gradcheck
from _lib.pcg32 import PCG32
from tinyllm.autograd import functional as F
from tinyllm.autograd.tensor import Tensor
from tinyllm.rnn.manual import (
    RNNCache,
    gradient_flow,
    rnn_backward,
    rnn_forward,
    tbptt_grads,
    tbptt_windows,
)

FIX = (
    Path(os.environ.get("TINYLLM_FIXTURES", "course/fixtures"))
    / "L3.1"
    / "rnn_torch.npz"
)
KEYS = ("x", "h0", "Wxh", "Whh", "bh")


def seed() -> int:
    return int(os.environ.get("OL_SEED", "0"))


def problem(s: int, T: int = 4, B: int = 2, D: int = 3, H: int = 3, scale: float = 0.6):
    """Seeded inputs, parameters, and upstream gradients, float64."""
    g = PCG32(seed=s)
    x = g.normal_array((T, B, D))
    h0 = g.normal_array((B, H), scale=0.5)
    Wxh = g.normal_array((D, H), scale=scale)
    Whh = g.normal_array((H, H), scale=scale)
    bh = g.normal_array((H,), scale=0.2)
    G = g.normal_array((T, B, H))
    return x, h0, Wxh, Whh, bh, G


def loss(G, gn=None):
    """L(x, h0, Wxh, Whh, bh) = sum(G * h) (+ sum(gn * h[-1])), by the forward only."""

    def f(x, h0, Wxh, Whh, bh):
        h, _ = rnn_forward(x, h0, Wxh, Whh, bh)
        val = float((G * h).sum())
        if gn is not None:
            val += float((gn * h[-1]).sum())
        return val

    return f


# --- the worked example ---------------------------------------------------------------


def test_hand_example_two_steps():
    # WHY: the chapter's worked example: the gradient reaching h_0 is the sum
    #      of what the loss sends it directly (0 here) and what step 1 sends
    #      back through Whh, and tanh' is read from the saved output, 1 - h^2.
    # KIND: unit
    # CATCHES: s01, s02, s03, s09, m01
    # CHAPTER: L3.1 section 3, Worked example by hand
    x = np.array([[[1.0]], [[0.0]]])
    h, cache = rnn_forward(x, [[0.0]], [[0.5]], [[0.8]], [0.0])
    assert isinstance(cache, RNNCache)
    assert_close(h[:, 0, 0], [0.46211715726000974, 0.3537237885050898], dtype="float64")
    gr = rnn_backward(np.array([[[0.0]], [[1.0]]]), cache)
    assert_close(gr["Wxh"], [[0.5504375878410427]], dtype="float64")
    assert_close(gr["Whh"], [[0.4042968189107551]], dtype="float64")
    assert_close(gr["bh"], [1.425317069286649], dtype="float64")
    assert_close(
        gr["x"][:, 0, 0],
        [0.5504375878410427 * 0.5, 0.8748794814456065 * 0.5],
        dtype="float64",
    )
    assert_close(gr["h0"], [[0.4403500702728342]], dtype="float64")


# --- gradients ----------------------------------------------------------------------------


def test_gradcheck_full_bptt():
    # WHY: every gradient, for every input, against central differences of
    #      the forward alone: x, h0, Wxh, Whh, and bh, with an upstream
    #      gradient at every step (an output layer reading every h_t).
    # KIND: gradcheck
    # CATCHES: s01, s02, s03, s04, s09, s11, m01
    # CHAPTER: L3.1 section 2, Principles (BPTT)
    x, h0, Wxh, Whh, bh, G = problem(seed())
    _, cache = rnn_forward(x, h0, Wxh, Whh, bh)
    gr = rnn_backward(G, cache)
    gradcheck(loss(G), [x, h0, Wxh, Whh, bh], [gr[k] for k in KEYS], names=list(KEYS))


def test_dh_next_enters_at_the_last_step():
    # WHY: dh_next is the gradient arriving at h_{T-1} from after the
    #      sequence (the next window, or the next layer in L3.6). It must be
    #      added to the last step's gradient and flow back from there.
    # KIND: gradcheck
    # CATCHES: s10
    # CHAPTER: L3.1 section 4, The interface
    x, h0, Wxh, Whh, bh, G = problem(seed() + 1, T=3)
    gn = PCG32(seed=seed() + 2).normal_array(h0.shape)
    _, cache = rnn_forward(x, h0, Wxh, Whh, bh)
    gr = rnn_backward(G, cache, dh_next=gn)
    gradcheck(
        loss(G, gn), [x, h0, Wxh, Whh, bh], [gr[k] for k in KEYS], names=list(KEYS)
    )


def test_matches_autograd():
    # WHY: the same forward written with your Tensor and op library (L0.1,
    #      L0.2) and differentiated by autograd gives the same five
    #      gradients. Manual BPTT is what autograd does through time; L3.6
    #      relies on the two agreeing when it fuses this into one op.
    # KIND: differential
    # CATCHES: s01, s03, s04, s11
    # CHAPTER: L3.1 section 2, Principles (BPTT)
    x, h0, Wxh, Whh, bh, G = problem(seed() + 3)
    _, cache = rnn_forward(x, h0, Wxh, Whh, bh)
    want = rnn_backward(G, cache)
    t = {
        k: Tensor(v, requires_grad=True, dtype=np.float64)
        for k, v in zip(KEYS, (x, h0, Wxh, Whh, bh))
    }
    prev, hs = t["h0"], []
    for s in range(x.shape[0]):
        prev = F.tanh(
            F.matmul(t["x"][s], t["Wxh"]) + F.matmul(prev, t["Whh"]) + t["bh"]
        )
        hs.append(prev)
    F.sum(F.stack(hs, axis=0) * G).backward()
    for k in KEYS:
        assert_close(want[k], t[k].grad, dtype="float64", msg=k)


def test_matches_torch_rnn():
    # WHY: torch.nn.RNN computes the same recurrence with W_ih = Wxh^T,
    #      W_hh = Whh^T, and two biases that only ever appear as their sum.
    #      Its outputs and gradients (both biases get dbh) match yours, so
    #      the convention is the standard one and torch RNN weights load.
    # KIND: golden
    # CATCHES: s02, s03, s04, s09
    # CHAPTER: L3.1 section 2, Principles (conventions)
    d = np.load(FIX, allow_pickle=False)
    json.loads(str(d["__meta__"]))
    Wxh, Whh = d["p_weight_ih_l0"].T, d["p_weight_hh_l0"].T
    bh = d["p_bias_ih_l0"] + d["p_bias_hh_l0"]
    h, cache = rnn_forward(d["x"], d["h0"], Wxh, Whh, bh)
    assert_close(h, d["out"], dtype="float64")
    gr = rnn_backward(d["g"], cache, dh_next=d["gn"])
    assert_close(gr["x"], d["gx"], dtype="float64")
    assert_close(gr["h0"], d["gh0"], dtype="float64")
    assert_close(gr["Wxh"].T, d["gp_weight_ih_l0"], dtype="float64")
    assert_close(gr["Whh"].T, d["gp_weight_hh_l0"], dtype="float64")
    assert_close(gr["bh"], d["gp_bias_ih_l0"], dtype="float64")
    assert_close(gr["bh"], d["gp_bias_hh_l0"], dtype="float64")


# --- truncation ---------------------------------------------------------------------------


def test_tbptt_windows_examples():
    # WHY: the cut points of TBPTT(k1, k2): a cut every k1 steps and at T,
    #      each reaching back k2 steps but never before 0. The contract's
    #      example (10, 4, 6) has overlapping windows; k1 = k2 tiles a
    #      multiple of k1; a final partial chunk is not dropped, and it too
    #      reaches back k2 steps.
    # KIND: unit
    # CATCHES: s05, s07
    # CHAPTER: L3.1 section 3, Worked example by hand
    assert tbptt_windows(10, 4, 6) == [(0, 4), (2, 8), (4, 10)]
    assert tbptt_windows(8, 4, 4) == [(0, 4), (4, 8)]
    assert tbptt_windows(10, 4, 4) == [(0, 4), (4, 8), (6, 10)]
    assert tbptt_windows(5, 8, 8) == [(0, 5)]
    assert tbptt_windows(3, 1, 2) == [(0, 1), (0, 2), (1, 3)]
    assert tbptt_windows(0, 2, 2) == []
    for bad in [(5, 0, 1), (5, 3, 2), (-1, 1, 1)]:
        with pytest.raises(ValueError):
            tbptt_windows(*bad)


def test_tbptt_with_long_reach_is_full_bptt():
    # WHY: when every window reaches back to step 0 (k2 >= T), each loss is
    #      differentiated through its whole history exactly once, so the
    #      truncated gradient equals full BPTT for any k1. Counting a loss in
    #      two windows, or dropping the last partial chunk, breaks this.
    # KIND: property
    # CATCHES: s06, s07, m02, m03
    # CHAPTER: L3.1 section 2, Principles (truncation)
    x, h0, Wxh, Whh, bh, G = problem(seed() + 4, T=7)
    _, cache = rnn_forward(x, h0, Wxh, Whh, bh)
    full = rnn_backward(G, cache)
    for k1 in (1, 2, 3, 7):
        got = tbptt_grads(x, h0, Wxh, Whh, bh, G, k1, 7)
        for k in KEYS:
            assert_close(got[k], full[k], rtol=1e-9, atol=1e-12, msg=f"k1={k1} {k}")


def test_tbptt_cuts_long_paths():
    # WHY: with k1 = k2 = 1 each loss reaches only its own step: no gradient
    #      crosses Whh into an earlier state, so dWhh is the sum over t of
    #      h_{t-1}^T (G_t * (1 - h_t^2)) and h0 only hears from step 0. This
    #      is the bias truncation buys its memory savings with.
    # KIND: unit
    # CATCHES: s02, m02, m03
    # CHAPTER: L3.1 section 2, Principles (truncation)
    x, h0, Wxh, Whh, bh, G = problem(seed() + 5, T=5)
    h, _ = rnn_forward(x, h0, Wxh, Whh, bh)
    got = tbptt_grads(x, h0, Wxh, Whh, bh, G, 1, 1)
    da = G * (1 - h * h)
    prev = np.concatenate([h0[None], h[:-1]])
    assert_close(got["Whh"], np.einsum("tbi,tbj->ij", prev, da), dtype="float64")
    assert_close(got["Wxh"], np.einsum("tbi,tbj->ij", x, da), dtype="float64")
    assert_close(got["bh"], da.sum(axis=(0, 1)), dtype="float64")
    assert_close(got["h0"], da[0] @ Whh.T, dtype="float64")
    assert_close(got["x"], da @ Wxh.T, dtype="float64")


# --- exploding and vanishing gradients ----------------------------------------------------


def _non_normal(eigs) -> tuple[np.ndarray, np.ndarray]:
    """P diag(eigs) P^-1 for a fixed non-orthogonal P, and its first right
    eigenvector P[:, 0] (eigenvalue eigs[0])."""
    P = np.eye(len(eigs)) + np.diag([0.8, 0.5, 0.3][: len(eigs) - 1], k=1)
    return P @ np.diag(eigs) @ np.linalg.inv(P), P[:, 0]


def test_gradient_flow_explodes_and_vanishes():
    # WHY: near the linear regime a step back multiplies the row gradient by
    #      Whh^T. A gradient along a right eigenvector of Whh with eigenvalue
    #      rho (the spectral radius) is scaled by exactly rho per step, so the
    #      norms follow rho^(T-1-t): they explode for rho > 1 and vanish for
    #      rho < 1, and any other gradient does the same once the top
    #      direction dominates. Whh here is not symmetric, so its spectral
    #      norm (1.12) is not its radius, and gradient_flow must report rho.
    # KIND: property
    # CATCHES: s08, s12
    # CHAPTER: L3.1 section 2, Principles (exploding and vanishing gradients)
    T, B, H = 10, 2, 4
    S, v = _non_normal([1.0, 0.6, -0.3, 0.1])
    rand = PCG32(seed=seed() + 7).normal_array((B, H))
    for rho in (1.25, 0.8):
        x = np.zeros((T, B, 1))
        _, cache = rnn_forward(
            x, np.full((B, H), 1e-4), np.zeros((1, H)), rho * S, np.zeros(H)
        )
        norms, r = gradient_flow(cache, np.outer([1.0, -2.0], v))
        assert norms.shape == (T,)
        assert_close(r, rho, rtol=1e-6, atol=1e-9)
        assert_close(norms[T - 1], np.sqrt(5.0), dtype="float64")
        assert_close(
            norms / norms[T - 1], rho ** np.arange(T - 1, -1, -1.0), rtol=1e-3, atol=0.0
        )
        _, long = rnn_forward(
            np.zeros((30, B, 1)),
            np.full((B, H), 1e-6),
            np.zeros((1, H)),
            rho * S,
            np.zeros(H),
        )
        norms, _ = gradient_flow(long, rand)
        assert (norms[0] > 10 * norms[-1]) if rho > 1 else (norms[0] < 0.1 * norms[-1])


def test_gradient_flow_uses_whh_transpose():
    # WHY: a step back multiplies the row gradient by Whh^T (from
    #      h_t = tanh(... + h_{t-1} Whh)), and the radius is the largest
    #      eigenvalue modulus, not the largest singular value. A non-normal
    #      Whh tells all three apart: [[0.5, 2], [0, 0.5]] has rho = 0.5 but
    #      spectral norm 2.13.
    # KIND: unit
    # CATCHES: s08, s12
    # CHAPTER: L3.1 section 5, Pitfalls
    Whh = np.array([[0.5, 2.0], [0.0, 0.5]])
    x = np.zeros((3, 1, 1))
    _, cache = rnn_forward(x, np.zeros((1, 2)), np.zeros((1, 2)), Whh, np.zeros(2))
    norms, r = gradient_flow(cache, np.array([[1.0, 0.0]]))
    assert_close(r, 0.5, rtol=1e-6, atol=1e-9)
    # h stays 0, so tanh' = 1: [1, 0] Whh^T = [0.5, 0], then [0.25, 0].
    assert_close(norms, [0.25, 0.5, 1.0], dtype="float64")


# --- boundaries ---------------------------------------------------------------------------


def test_shapes_and_bad_arguments():
    # WHY: T = 0 is a valid empty sequence (h is [0, B, H] and the gradient
    #      for h0 is dh_next); mismatched shapes are bugs to report, not to
    #      broadcast.
    # KIND: boundary
    # CATCHES: m04
    # CHAPTER: L3.1 section 4, The interface
    h, cache = rnn_forward(
        np.zeros((0, 2, 3)),
        np.zeros((2, 4)),
        np.zeros((3, 4)),
        np.zeros((4, 4)),
        np.zeros(4),
    )
    assert h.shape == (0, 2, 4)
    gr = rnn_backward(np.zeros((0, 2, 4)), cache, dh_next=np.ones((2, 4)))
    assert_close(gr["h0"], np.ones((2, 4)), dtype="float64")
    assert gr["x"].shape == (0, 2, 3)
    good = (
        np.zeros((2, 1, 3)),
        np.zeros((1, 4)),
        np.zeros((3, 4)),
        np.zeros((4, 4)),
        np.zeros(4),
    )
    for i, wrong in [
        (2, np.zeros((4, 3))),
        (3, np.zeros((4, 5))),
        (1, np.zeros((2, 4))),
        (4, np.zeros(3)),
        (0, np.zeros((2, 3))),
    ]:
        args = list(good)
        args[i] = wrong
        with pytest.raises(ValueError):
            rnn_forward(*args)
    _, cache = rnn_forward(*good)
    with pytest.raises(ValueError):
        rnn_backward(np.zeros((3, 1, 4)), cache)
    with pytest.raises(ValueError):
        gradient_flow(cache, np.zeros((2, 4)))
