"""Course tests for L3.4: bidirectional RNNs with length-aware reversal
(tinyllm/rnn/bi.py).

Rung R0 for this file: read these before you write code. (Your own graded
tests, rung R4, go in python/tests/l3-4-bi/; see the chapter, section 4.)
Each test names why it exists (WHY), what kind of check it is (KIND), the
planted bugs it kills (CATCHES, mutants in course/mutants/L3.4), and the
chapter section it comes from.

The chapter's worked example (section 3): T = 3, lengths [3, 1, 2], tokens
"abc", "d..", "ef." (a dot is padding). reversal_index is
[[2, 0, 1], [1, 1, 0], [0, 2, 2]], and the reversed batch reads "cba", "d..",
"fe.": the padding stays at the end. With running-sum modules in both
directions, sequence [1, 2, 3] gives forward [1, 3, 6] and backward
[6, 5, 3] (suffix sums), and sequence [4, ., .] gives [4, 0, 0] twice.

Golden cases (course/fixtures/L3.4/bi_torch.npz) were recorded from torch
2.14.1 by course/oracle/L3.4/bi_torch.py.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32
from tinyllm.autograd import functional as F
from tinyllm.autograd.tensor import Tensor
from tinyllm.nn.module import Module
from tinyllm.rnn.bi import bidirectional, reversal_index, reverse_padded
from tinyllm.rnn.gru import GRU
from tinyllm.rnn.lstm import LSTM

FIX = (
    Path(os.environ.get("TINYLLM_FIXTURES", "course/fixtures"))
    / "L3.4"
    / "bi_torch.npz"
)
F32 = dict(rtol=2e-5, atol=2e-6)
HAND_LENGTHS = np.array([3, 1, 2])


def seed() -> int:
    return int(os.environ.get("OL_SEED", "0"))


class Rng:
    def __init__(self, s: int) -> None:
        self.g = PCG32(seed=s)

    def next_u32(self) -> int:
        return self.g.next_u32()

    def uniform(self) -> float:
        return self.g.uniform()

    def uniforms(self, n: int) -> np.ndarray:
        return np.array([self.g.uniform() for _ in range(n)], dtype=np.float64)


class RunningSum(Module):
    """A toy recurrent module: out[t] = x[0] + ... + x[t] within each length, 0 after."""

    def forward(self, x, state=None, lengths=None):
        a = np.asarray(x.data if isinstance(x, Tensor) else x, dtype=np.float64)
        T, B = a.shape[:2]
        n = np.full(B, T) if lengths is None else np.asarray(lengths)
        mask = (np.arange(T)[:, None] < n[None, :])[..., None]
        out = np.cumsum(np.where(mask, a, 0.0), axis=0) * mask
        return Tensor(out, dtype=np.float64), None


def golden():
    d = np.load(FIX, allow_pickle=False)
    return d, json.loads(str(d["__meta__"]))


def case_names() -> list[str]:
    try:
        return [c["name"] for c in golden()[1]["cases"]]
    except OSError:
        return ["missing-fixture"]


# --- the worked example ---------------------------------------------------------------


def test_hand_example_reversal():
    # WHY: the chapter's worked example: each sequence is mirrored inside its
    #      own length (index lengths[b] - 1 - t) and its padding stays where
    #      it was, so "abc", "d..", "ef." read "cba", "d..", "fe.". A plain
    #      x[::-1] would give "cba", "..d", ".fe": the backward RNN would read
    #      padding first.
    # KIND: unit
    # CATCHES: s01, s04, m01
    # CHAPTER: L3.4 section 3, Worked example by hand
    idx = reversal_index(HAND_LENGTHS, 3)
    assert idx.dtype == np.int64
    assert idx.tolist() == [[2, 0, 1], [1, 1, 0], [0, 2, 2]]
    toks = np.array([list("ade"), list("b.f"), list("c..")])  # [T, B]
    out = reverse_padded(toks, HAND_LENGTHS)
    assert ["".join(out[:, b]) for b in range(3)] == ["cba", "d..", "fe."]


def test_hand_example_running_sums():
    # WHY: with running-sum "RNNs" both directions are computable by hand:
    #      the forward half is the prefix sums, the backward half the suffix
    #      sums (the backward module's prefix sums over the reversed
    #      sequence, put back in time order), padding 0 in both halves, and
    #      the forward half comes first.
    # KIND: unit
    # CATCHES: s01, s02, s03, s05, s06
    # CHAPTER: L3.4 section 3, Worked example by hand
    x = np.zeros((3, 2, 1))
    x[:, 0, 0] = [1.0, 2.0, 3.0]
    x[:, 1, 0] = [4.0, 99.0, 99.0]  # padding after the first step
    out = bidirectional(
        RunningSum(), RunningSum(), Tensor(x, dtype=np.float64), np.array([3, 1])
    )
    assert out.shape == (3, 2, 2)
    assert_close(out.data[:, 0, 0], [1.0, 3.0, 6.0], dtype="float64")
    assert_close(out.data[:, 0, 1], [6.0, 5.0, 3.0], dtype="float64")
    assert_close(out.data[:, 1, 0], [4.0, 0.0, 0.0], dtype="float64")
    assert_close(out.data[:, 1, 1], [4.0, 0.0, 0.0], dtype="float64")


# --- reversal properties ----------------------------------------------------------------


def test_reversal_is_an_involution_that_keeps_padding():
    # WHY: reversing twice gives back the batch (so the same call that
    #      reverses the input puts the backward outputs back in order), the
    #      padding positions are fixed points, and on a Tensor the gradient
    #      is the same permutation applied to the upstream gradient.
    # KIND: property
    # CATCHES: s04, m01
    # CHAPTER: L3.4 section 2, Principles (length-aware reversal)
    g = PCG32(seed=seed())
    for T, lengths in [(5, [5, 2, 3, 1]), (4, [4, 4]), (6, [1, 6, 3])]:
        n = np.array(lengths)
        idx = reversal_index(n, T)
        assert (idx[idx, np.arange(len(n))[None, :]] == np.arange(T)[:, None]).all()
        x = g.normal_array((T, len(n), 3))
        assert_close(reverse_padded(reverse_padded(x, n), n), x, dtype="float64")
        for b, nb in enumerate(n):
            assert_close(reverse_padded(x, n)[nb:, b], x[nb:, b], dtype="float64")
        xt = Tensor(x, requires_grad=True, dtype=np.float64)
        w = g.normal_array(x.shape)
        F.sum(reverse_padded(xt, n) * w).backward()
        assert_close(xt.grad, reverse_padded(w, n), dtype="float64")


def test_no_lengths_is_a_plain_flip():
    # WHY: with every sequence full (lengths None), length-aware reversal is
    #      exactly x[::-1] along time, the unpadded case most code starts
    #      from.
    # KIND: unit
    # CATCHES: m03
    # CHAPTER: L3.4 section 2, Principles (length-aware reversal)
    x = PCG32(seed=seed() + 1).normal_array((4, 3, 2))
    assert_close(reverse_padded(x, None), x[::-1], dtype="float64")
    assert_close(reverse_padded(x, np.array([4, 4, 4])), x[::-1], dtype="float64")


# --- against torch and the per-sequence loop ------------------------------------------------


def _split(d, key, params):
    fwd = {
        p[: -len("_l0")] + "_l0": d[f"{key}_p_{p}"] for p in params if p.endswith("_l0")
    }
    bwd = {
        p[: -len("_l0_reverse")] + "_l0": d[f"{key}_p_{p}"]
        for p in params
        if p.endswith("_reverse")
    }
    return fwd, bwd


@pytest.mark.parametrize("name", case_names())
def test_matches_torch_bidirectional(name):
    # WHY: torch's bidirectional=True over a packed batch is two modules: the
    #      *_l0 weights read forward, the *_l0_reverse weights read each
    #      sequence backward within its length, outputs concatenated forward
    #      first, padding 0. Loading the two halves into your LSTMs or GRUs
    #      gives torch's output and every gradient.
    # KIND: golden
    # CATCHES: s01, s02, s03, s05, s06
    # CHAPTER: L3.4 section 4, The interface
    d, meta = golden()
    c = next(c for c in meta["cases"] if c["name"] == name)
    cls = LSTM if c["cell"] == "LSTM" else GRU
    fwd, bwd = cls(3, 4, rng=Rng(seed())), cls(3, 4, rng=Rng(seed() + 1))
    sf, sb = _split(d, name, c["params"])
    fwd.load_state_dict(sf)
    bwd.load_state_dict(sb)
    x = Tensor(d[f"{name}_x"], requires_grad=True)
    out = bidirectional(fwd, bwd, x, d[f"{name}_lengths"])
    assert_close(out.data, d[f"{name}_out"], **F32)
    F.sum(out * d[f"{name}_g"]).backward()
    assert_close(x.grad, d[f"{name}_gx"], **F32, msg="grad of x")
    pf, pb = dict(fwd.named_parameters()), dict(bwd.named_parameters())
    for p in c["params"]:
        got = pb[p.replace("_reverse", "")] if p.endswith("_reverse") else pf[p]
        assert_close(got.grad, d[f"{name}_gp_{p}"], **F32, msg=f"grad of {p}")


def test_matches_per_sequence_loop():
    # WHY: the batch answer equals the obvious slow one: run each sequence
    #      alone, unpadded, forward through fwd and reversed through bwd. The
    #      batched gather is only an optimization of that loop.
    # KIND: differential
    # CATCHES: s01, s02, s05
    # CHAPTER: L3.4 section 2, Principles (bidirectional)
    T, B, D, H = 6, 4, 3, 2
    lengths = np.array([6, 1, 4, 3])
    fwd, bwd = GRU(D, H, rng=Rng(seed() + 2)), GRU(D, H, rng=Rng(seed() + 3))
    x = PCG32(seed=seed() + 4).normal_array((T, B, D)).astype(np.float32)
    out = bidirectional(fwd, bwd, Tensor(x), lengths)
    for b, n in enumerate(lengths):
        seq = x[:n, b : b + 1]
        of, _ = fwd(seq)
        ob, _ = bwd(seq[::-1].copy())
        assert_close(
            out.data[:n, b, :H], of.data[:, 0], **F32, msg=f"forward half, sequence {b}"
        )
        assert_close(
            out.data[:n, b, H:],
            ob.data[::-1, 0],
            **F32,
            msg=f"backward half, sequence {b}",
        )
        assert (out.data[n:, b] == 0).all()


def test_padding_never_reaches_either_direction():
    # WHY: garbage in the padding changes no output and gets no gradient: the
    #      backward direction starts on each sequence's last real token, not on
    #      padding (the bug a plain x[::-1] plants). Final states sit at
    #      out[lengths[b] - 1, b, :H] and out[0, b, H:].
    # KIND: boundary
    # CATCHES: s01, s05, s06
    # CHAPTER: L3.4 section 5, Pitfalls
    T, B, D, H = 5, 3, 2, 3
    lengths = np.array([2, 5, 3])
    fwd, bwd = LSTM(D, H, rng=Rng(seed() + 5)), LSTM(D, H, rng=Rng(seed() + 6))
    x = PCG32(seed=seed() + 7).normal_array((T, B, D)).astype(np.float32)
    dirty = x.copy()
    for b, n in enumerate(lengths):
        x[n:, b] = 0.0
        dirty[n:, b] = 50.0
    clean = bidirectional(fwd, bwd, Tensor(x), lengths)
    xt = Tensor(dirty, requires_grad=True)
    out = bidirectional(fwd, bwd, xt, lengths)
    assert_close(out.data, clean.data, dtype="float32")
    F.sum(out).backward()
    _, (hf, _) = fwd(Tensor(x), lengths=lengths)
    _, (hb, _) = bwd(reverse_padded(Tensor(x), lengths), lengths=lengths)
    for b, n in enumerate(lengths):
        assert (xt.grad[n:, b] == 0).all()
        assert_close(out.data[n - 1, b, :H], hf.data[0, b], **F32)
        assert_close(out.data[0, b, H:], hb.data[0, b], **F32)


def test_bad_lengths():
    # WHY: a length of 0, longer than T, the wrong count, or a float is a
    #      bug in the batcher; reversing with it would silently scramble
    #      tokens.
    # KIND: boundary
    # CATCHES: m02
    # CHAPTER: L3.4 section 4, The interface
    x = np.zeros((3, 2, 1))
    for bad in ([0, 1], [1, 4], [1, 2, 3], [1.0, 2.0]):
        with pytest.raises(ValueError):
            reverse_padded(x, np.array(bad))
    with pytest.raises(ValueError):
        reversal_index(np.array([[1, 2]]), 3)
    with pytest.raises(ValueError):
        reverse_padded(np.zeros(3), None)
