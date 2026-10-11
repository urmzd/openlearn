"""Course tests for L0.3: losses with a fused backward (tinyllm/autograd/losses.py).

Rung R0 for this file: read these before you write code. (Your own graded
tests for this module, rung R2, go in python/tests/l0-3-losses/; see the
chapter, section 4.) Each test names why it exists (WHY), what kind of check
it is (KIND), the planted bugs it kills (CATCHES, mutants in
course/mutants/L0.3), and the chapter section it comes from.

The worked example of the chapter (section 3): logits [0, ln 2, ln 3] with
target 2, so softmax p = [1/6, 1/3, 1/2], loss ln 2, gradient
p - onehot = [1/6, 1/3, -1/2]; a second row whose target is ignore_index
changes nothing. With label smoothing 0.3: q = [0.1, 0.1, 0.8], loss
0.1 ln 6 + 0.1 ln 3 + 0.8 ln 2, gradient p - q = [1/15, 7/30, -3/10].

The golden cases in course/fixtures/L0.3/losses_torch.npz were recorded from
torch 2.14.1 by course/oracle/L0.3/losses_torch.py.
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
from tinyllm.autograd.losses import bce_with_logits, cross_entropy, mse
from tinyllm.autograd.tensor import Tensor
from tinyllm.autograd.vjp import cross_entropy_vjp
from tinyllm.info.entropy import cross_entropy as h_cross

F64 = np.float64
FIX = (
    Path(os.environ.get("TINYLLM_FIXTURES", "course/fixtures"))
    / "L0.3"
    / "losses_torch.npz"
)
LN2, LN3, LN6 = math.log(2.0), math.log(3.0), math.log(6.0)
HAND = [[0.0, LN2, LN3], [5.0, -1.0, 2.0]]


def seed() -> int:
    return int(os.environ.get("OL_SEED", "0"))


def _golden():
    data = np.load(FIX, allow_pickle=False)
    return data, json.loads(str(data["__meta__"]))["cases"]


def _names():
    try:
        return [c["name"] for c in _golden()[1]]
    except OSError:
        return ["missing-fixture"]


# --- the worked example ---------------------------------------------------------------


def test_hand_example_cross_entropy():
    # WHY: the chapter's worked example, number for number. The second row is
    #      ignored, so the mean divides by ONE kept row: loss ln 2, gradient
    #      p - onehot on row 0 and exact zeros on row 1.
    # KIND: unit
    # CATCHES: s01, s02
    # CHAPTER: L0.3 section 3, Worked example by hand
    z = Tensor(HAND, requires_grad=True, dtype=F64)
    loss = cross_entropy(z, np.array([2, -100]))
    assert loss.shape == ()
    assert_close(loss.data, LN2)
    loss.backward()
    assert_close(z.grad, [[1 / 6, 1 / 3, -1 / 2], [0.0, 0.0, 0.0]])


def test_hand_example_label_smoothing():
    # WHY: smoothing 0.3 over V = 3 spreads 0.3 / 3 = 0.1 to every class:
    #      q = [0.1, 0.1, 0.8], loss -sum q log p = 0.1 ln 6 + 0.1 ln 3 + 0.8 ln 2,
    #      gradient p - q.
    # KIND: unit
    # CATCHES: s05, s06
    # CHAPTER: L0.3 section 3, Worked example by hand
    z = Tensor(HAND[:1], requires_grad=True, dtype=F64)
    loss = cross_entropy(z, np.array([2]), label_smoothing=0.3)
    assert_close(loss.data, 0.1 * LN6 + 0.1 * LN3 + 0.8 * LN2)
    loss.backward()
    assert_close(z.grad, [[1 / 6 - 0.1, 1 / 3 - 0.1, 0.5 - 0.8]])


# --- against torch ---------------------------------------------------------------------


@pytest.mark.parametrize("name", _names())
def test_matches_torch(name):
    # WHY: loss and gradient equal torch.nn.functional on every option:
    #      ignore_index (including ignoring class 0), label smoothing, the
    #      three reductions, [B, T, V] logits, logits near 1e3, float32, mse,
    #      and bce with soft targets and pos_weight. Every training loop from
    #      L2 on calls these.
    # KIND: golden
    # CATCHES: s01, s06, s08, s10, s11, s12, m01
    # CHAPTER: L0.3 section 4, The interface
    data, cases = _golden()
    c = next(c for c in cases if c["name"] == name)
    k, dt = c["key"], np.dtype(c["dtype"])
    x = Tensor(data[f"{k}_x"], requires_grad=True, dtype=dt)
    if c["kind"] == "ce":
        y = cross_entropy(x, data[f"{k}_t"], **c["kwargs"])
    elif c["kind"] == "mse":
        y = mse(x, data[f"{k}_t"])
    else:
        y = bce_with_logits(
            x, data[f"{k}_t"], data[f"{k}_pw"] if c["pos_weight"] else None
        )
    tol = {} if dt == F64 else {"dtype": "float32"}
    assert y.dtype == dt
    assert_close(y.data, data[f"{k}_y"], msg=f"{name} loss", **tol)
    y.backward(data[f"{k}_g"])
    assert_close(x.grad, data[f"{k}_gx"], msg=f"{name} gradient", **tol)


# --- against central differences and the earlier modules --------------------------------


@pytest.mark.parametrize(
    "case",
    ["ce_mean", "ce_sum_smooth", "ce_none_ignore", "mse", "bce_pos_weight"],
)
def test_gradcheck_losses(case):
    # WHY: the fused vjps against the frozen central differences in float64:
    #      a closed form that is off by a factor or a sign fails here even
    #      where no golden case exists.
    # KIND: gradcheck
    # CATCHES: s05, s08, s11, s12
    # CHAPTER: L0.3 section 2, Principles (the fused gradient)
    rng = PCG32(seed=seed() + 17 * len(case))
    t = np.array([[1, 3, -100], [0, 2, 3]])
    if case == "ce_mean":
        x = rng.uniform_array((2, 3, 4), -2, 2)
        fn = lambda z: cross_entropy(z, t)  # noqa: E731
    elif case == "ce_sum_smooth":
        x = rng.uniform_array((2, 3, 4), -2, 2)
        fn = lambda z: cross_entropy(z, t, label_smoothing=0.2, reduction="sum")  # noqa: E731
    elif case == "ce_none_ignore":
        x = rng.uniform_array((2, 3, 4), -2, 2)
        fn = lambda z: cross_entropy(z, t, label_smoothing=0.1, reduction="none")  # noqa: E731
    elif case == "mse":
        x = rng.uniform_array((3, 2), -2, 2)
        y = rng.uniform_array((3, 2), -2, 2)
        fn = lambda z: mse(z, y)  # noqa: E731
    else:
        x = rng.uniform_array((3, 2), -3, 3)
        y = rng.uniform_array((3, 2), 0, 1)
        fn = lambda z: bce_with_logits(z, y, pos_weight=np.array([0.5, 4.0]))  # noqa: E731
    out = fn(Tensor(x, dtype=F64))
    w = rng.uniform_array(out.shape, 0.5, 1.5)
    z = Tensor(x, requires_grad=True, dtype=F64)
    fn(z).backward(w)
    gradcheck(lambda a: float((fn(Tensor(a, dtype=F64)).data * w).sum()), [x], [z.grad])


def test_matches_m08_cross_entropy_vjp():
    # WHY: your fused backward is the closed form M08.3 derived; for the plain
    #      mean loss with ignored rows both must give the same [N, V] matrix.
    # KIND: differential
    # CATCHES: s02
    # CHAPTER: L0.3 section 2, Principles (the fused gradient)
    rng = PCG32(seed=seed())
    x = rng.uniform_array((6, 5), -3, 3)
    t = np.array([4, -100, 0, 2, -100, 1])
    z = Tensor(x, requires_grad=True, dtype=F64)
    cross_entropy(z, t).backward()
    assert_close(z.grad, cross_entropy_vjp(x, t, ignore_index=-100))


def test_rows_match_m11_cross_entropy():
    # WHY: a row's smoothed loss is the cross-entropy H(q, p) of M11.1 between
    #      the smoothed target q and the model's softmax p: the definition,
    #      computed the slow way.
    # KIND: differential
    # CATCHES: s06
    # CHAPTER: L0.3 section 2, Principles (label smoothing)
    rng = PCG32(seed=seed() + 1)
    x = rng.uniform_array((4, 5), -2, 2)
    t = np.array([0, 4, 2, 2])
    eps = 0.2
    q = np.full((4, 5), eps / 5)
    q[np.arange(4), t] += 1 - eps
    p = np.exp(x - x.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    rows = cross_entropy(Tensor(x, dtype=F64), t, label_smoothing=eps, reduction="none")
    assert_close(rows.data, h_cross(q, p, axis=-1))


# --- properties and edges --------------------------------------------------------------------


def test_ignore_index_excluded():
    # WHY: padding positions (ignore_index) must neither move the loss nor
    #      receive gradient, whatever their logits hold: appending ignored rows
    #      to a batch leaves the mean loss and the kept rows' gradients equal.
    # KIND: property
    # CATCHES: s01, s02, m01
    # CHAPTER: L0.3 section 5, Pitfalls, item 1
    rng = PCG32(seed=seed())
    x = rng.uniform_array((4, 6), -3, 3)
    t = np.array([5, 0, 3, 3])
    z1 = Tensor(x, requires_grad=True, dtype=F64)
    l1 = cross_entropy(z1, t, ignore_index=7, label_smoothing=0.1)
    l1.backward()
    pad = rng.uniform_array((3, 6), -50, 50)
    z2 = Tensor(np.concatenate([x, pad]), requires_grad=True, dtype=F64)
    l2 = cross_entropy(
        z2, np.concatenate([t, [7, 7, 7]]), ignore_index=7, label_smoothing=0.1
    )
    l2.backward()
    assert_close(l2.data, l1.data)
    assert_close(z2.grad[:4], z1.grad)
    assert_close(z2.grad[4:], np.zeros((3, 6)))


def test_all_ignored_is_zero():
    # WHY: a batch made only of padding has no kept row. The mean over zero
    #      rows is 0 / 0; the course defines it as loss 0 with a zero gradient
    #      (torch gives nan, and one nan step poisons every weight).
    # KIND: boundary
    # CATCHES: s03
    # CHAPTER: L0.3 section 5, Pitfalls, item 3
    z = Tensor(np.ones((2, 3)), requires_grad=True)
    loss = cross_entropy(z, np.array([-100, -100]))
    assert float(loss.data) == 0.0 and loss.dtype == np.float32
    loss.backward()
    assert (z.grad == 0).all()


def test_large_logits_stay_finite():
    # WHY: trained logits reach 1e3 and beyond. -log(softmax(z)) underflows
    #      softmax to 0 and returns inf; log_softmax keeps it finite. The same
    #      for bce: log(sigmoid(-1000)) is -inf, softplus is not.
    # KIND: boundary
    # CATCHES: s04, s09
    # CHAPTER: L0.3 section 5, Pitfalls, item 4
    z = Tensor([[1e4, 0.0, -1e4]], requires_grad=True, dtype=np.float32)
    loss = cross_entropy(z, np.array([2]))
    assert np.isfinite(loss.data) and abs(float(loss.data) - 2e4) < 1.0
    loss.backward()
    assert np.isfinite(z.grad).all()
    assert_close(z.grad, [[1.0, 0.0, -1.0]], dtype="float32")
    b = Tensor([-1000.0, 1000.0], requires_grad=True, dtype=F64)
    lb = bce_with_logits(b, np.array([1.0, 0.0]))
    assert_close(lb.data, 1000.0)
    lb.backward()
    assert_close(b.grad, [-0.5, 0.5])


def test_reductions_agree():
    # WHY: "none" is the per-row loss; "sum" adds the kept rows; "mean"
    #      divides that sum by the number of kept rows. Evaluation (L6.7)
    #      sums "none" over a whole dataset and divides once.
    # KIND: property
    # CATCHES: s01
    # CHAPTER: L0.3 section 4, The interface
    rng = PCG32(seed=seed() + 2)
    x = rng.uniform_array((2, 4, 5), -2, 2)
    t = np.array([[0, 4, -100, 1], [-100, 2, 2, 3]])
    rows = cross_entropy(Tensor(x, dtype=F64), t, reduction="none")
    total = cross_entropy(Tensor(x, dtype=F64), t, reduction="sum")
    avg = cross_entropy(Tensor(x, dtype=F64), t, reduction="mean")
    assert rows.shape == (2, 4) and rows.data[0, 2] == 0.0
    assert_close(total.data, rows.data.sum())
    assert_close(avg.data, rows.data.sum() / 6)


def test_float32_stays_float32():
    # WHY: float32 logits give a float32 loss and gradient: the training loop
    #      updates float32 weights in place with them.
    # KIND: boundary
    # CATCHES: m02
    # CHAPTER: L0.3 section 4, The interface
    z = Tensor(np.zeros((3, 4)), requires_grad=True)
    for loss in (
        cross_entropy(z, np.array([0, 1, 2]), label_smoothing=0.1),
        mse(z, np.ones((3, 4))),
        bce_with_logits(z, np.ones((3, 4))),
    ):
        assert loss.dtype == np.float32
        z.grad = None
        loss.backward()
        assert z.grad.dtype == np.float32


def test_rejects_bad_inputs():
    # WHY: a target outside [0, V) is a data bug (numpy would index from the
    #      end for -1), float targets are a pipeline bug, and an unknown
    #      reduction or smoothing outside [0, 1] is a config bug: all fail here.
    # KIND: boundary
    # CATCHES: s07, m03, m04, m06
    # CHAPTER: L0.3 section 5, Pitfalls, item 2
    z = Tensor(np.zeros((2, 3)))
    for t in (
        np.array([0, 3]),
        np.array([-1, 0]),
        np.array([0.0, 1.0]),
        np.array([0, 1, 2]),
    ):
        with pytest.raises(ValueError):
            cross_entropy(z, t)
    with pytest.raises(ValueError):
        cross_entropy(z, np.array([0, 1]), label_smoothing=1.5)
    with pytest.raises(ValueError):
        cross_entropy(z, np.array([0, 1]), reduction="avg")
    with pytest.raises(ValueError):
        mse(z, np.zeros(3))
    with pytest.raises(ValueError):
        bce_with_logits(z, np.zeros((3, 2)))
