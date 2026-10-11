"""Course tests for L0.5: data loader, train step, eval loop (tinyllm/train/loop.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/L0.5), and the chapter section it comes from.

The worked example of the chapter (section 3): Linear(1, 1) with w = 2,
b = 0, the batch x = [[1], [2]], y = [[3], [5]], mse loss: predictions 2 and
4, loss ((2 - 3)^2 + (4 - 5)^2) / 2 = 1, gradients dL/dw = -3 and
dL/db = -2; one SGD step with lr 0.1 gives w = 2.3, b = 0.2.

The learning tests compare against bars in course/fixtures/ref-thresholds.tsv
(the reference's mean + 3 sd over 5 seeds), recorded by
`ol verify course L0.5 --record-thresholds`.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32
from _lib.thresholds import check
from tinyllm.autograd import functional as F
from tinyllm.autograd.losses import bce_with_logits, cross_entropy, mse
from tinyllm.autograd.tensor import Tensor
from tinyllm.nn.layers import Dropout, Linear, Sequential, Tanh
from tinyllm.num.ema import EMA
from tinyllm.optim.adamw import AdamW
from tinyllm.optim.sgd import SGD
from tinyllm.train.loop import DataLoader, evaluate, train_step


def seed() -> int:
    return int(os.environ.get("OL_SEED", "0"))


class Rng:
    """The frozen PCG32 behind the generator API of M06.3 (next_u32, uniform,
    uniforms, below, shuffle), so no verdict here depends on your PCG32."""

    def __init__(self, s: int) -> None:
        self.g = PCG32(seed=s)

    def next_u32(self) -> int:
        return self.g.next_u32()

    def uniform(self) -> float:
        return self.g.uniform()

    def uniforms(self, n: int) -> np.ndarray:
        return np.array([self.g.uniform() for _ in range(n)], dtype=np.float64)

    def below(self, n: int) -> int:
        return self.g.below(n)

    def shuffle(self, xs) -> None:
        for i in range(len(xs) - 1, 0, -1):  # spec/pcg32.md Fisher-Yates
            j = self.g.below(i + 1)
            xs[i], xs[j] = xs[j], xs[i]


def mse_fn(model, batch):
    return mse(model(Tensor(batch["x"])), batch["y"])


def hand_model() -> Linear:
    m = Linear(1, 1, rng=Rng(0))
    m.load_state_dict({"weight": [[2.0]], "bias": [0.0]})
    return m


HAND = {
    "x": np.array([[1.0], [2.0]], dtype=np.float32),
    "y": np.array([[3.0], [5.0]], dtype=np.float32),
}


# --- the worked example ---------------------------------------------------------------


def test_hand_example_train_step():
    # WHY: the chapter's worked example, number for number: loss 1, gradients
    #      -3 and -2, and one SGD step of lr 0.1 moves w to 2.3 and b to 0.2.
    # KIND: unit
    # CATCHES: s02
    # CHAPTER: L0.5 section 3, Worked example by hand
    m = hand_model()
    stats = train_step(m, HAND, mse_fn, SGD(m.parameters(), lr=0.1))
    assert set(stats) == {"loss"}
    assert_close(stats["loss"], 1.0, dtype="float32")
    assert_close(m.weight.grad, [[-3.0]], dtype="float32")
    assert_close(m.bias.grad, [-2.0], dtype="float32")
    assert_close(m.weight.data, [[2.3]], dtype="float32")
    assert_close(m.bias.data, [0.2], dtype="float32")


def test_zero_grad_every_step():
    # WHY: gradients accumulate across backward calls (L0.1), so a step that
    #      does not zero them first adds the previous step's gradient to this
    #      one. The second step from the hand example must use the gradients
    #      at w = 2.3, b = 0.2 alone: residuals -0.5 and -0.2, so dL/dw = -0.9
    #      and dL/db = -0.7.
    # KIND: unit
    # CATCHES: s01
    # CHAPTER: L0.5 section 5, Pitfalls, item 1
    m = hand_model()
    opt = SGD(m.parameters(), lr=0.1)
    train_step(m, HAND, mse_fn, opt)
    train_step(m, HAND, mse_fn, opt)
    w, b = 2.3, 0.2
    r = [w * 1 + b - 3, w * 2 + b - 5]
    gw = (2 * r[0] * 1 + 2 * r[1] * 2) / 2
    gb = (2 * r[0] + 2 * r[1]) / 2
    assert_close(m.weight.grad, [[gw]], dtype="float32")
    assert_close(m.weight.data, [[w - 0.1 * gw]], dtype="float32")
    assert_close(m.bias.data, [b - 0.1 * gb], dtype="float32")


def test_clip_reports_the_norm_and_clips():
    # WHY: with clip = 1 the global norm sqrt(3^2 + 2^2) = 3.606 is reported
    #      (before clipping) and the update uses gradients scaled to norm 1,
    #      the long-training guard every later loop turns on (M10.4).
    # KIND: unit
    # CATCHES: s03, s04
    # CHAPTER: L0.5 section 4, The interface
    m = hand_model()
    stats = train_step(m, HAND, mse_fn, SGD(m.parameters(), lr=0.1), clip=1.0)
    norm = math.sqrt(13.0)
    assert_close(stats["grad_norm"], norm, rtol=1e-5, atol=0)
    c = 1.0 / (norm + 1e-6)
    assert_close(m.weight.data, [[2.0 + 0.1 * 3.0 * c]], dtype="float32")
    assert_close(m.bias.data, [0.0 + 0.1 * 2.0 * c], dtype="float32")


def test_nonfinite_loss_stops_before_the_update():
    # WHY: one nan loss, stepped, turns every weight into nan and the run is
    #      lost. train_step raises FloatingPointError and leaves every
    #      parameter as it was, so the caller can skip the batch or stop.
    # KIND: boundary
    # CATCHES: s05
    # CHAPTER: L0.5 section 5, Pitfalls, item 4
    m = hand_model()
    bad = {
        "x": np.array([[np.nan]], dtype=np.float32),
        "y": np.array([[1.0]], dtype=np.float32),
    }
    with pytest.raises(FloatingPointError):
        train_step(m, bad, mse_fn, SGD(m.parameters(), lr=0.1))
    assert_close(m.weight.data, [[2.0]], dtype="float32")
    assert_close(m.bias.data, [0.0], dtype="float32")


def test_loss_fn_extras_and_shape():
    # WHY: loss_fn may return (loss, metrics) and train_step reports the
    #      metrics beside the loss; a loss that was never reduced to one
    #      number is a ValueError, not a silent backward of ones.
    # KIND: boundary
    # CATCHES: m01, m04
    # CHAPTER: L0.5 section 4, The interface
    m = hand_model()
    stats = train_step(
        m,
        HAND,
        lambda mod, b: (mse_fn(mod, b), {"acc": 0.5}),
        SGD(m.parameters(), lr=0.1),
    )
    assert stats["acc"] == 0.5 and "loss" in stats
    with pytest.raises(ValueError):
        train_step(
            m,
            HAND,
            lambda mod, b: mod(Tensor(b["x"])) * 1.0,
            SGD(m.parameters(), lr=0.1),
        )


def test_loss_ema_is_the_debiased_average():
    # WHY: a training log plots the EMA of the step losses (M02.2), not the
    #      raw, noisy values. With beta = 0.9 the hand example's losses 1 and
    #      0.145 give m_1 = 0.1 and m_2 = 0.1045; debiased by 1 - 0.9^t that
    #      is 1.0 and 0.55. The biased m_t would start ten times too low. A
    #      non-finite loss never reaches the EMA.
    # KIND: unit
    # CATCHES: s22
    # CHAPTER: L0.5 section 4, The interface
    m = hand_model()
    opt = SGD(m.parameters(), lr=0.1)
    ema = EMA(0.9)
    first = train_step(m, HAND, mse_fn, opt, ema=ema)
    assert_close(first["loss_ema"], 1.0, dtype="float32")
    second = train_step(m, HAND, mse_fn, opt, ema=ema)
    assert_close(second["loss"], 0.145, dtype="float32")
    assert_close(second["loss_ema"], 0.55, dtype="float32")
    bad = {
        "x": np.array([[np.nan]], dtype=np.float32),
        "y": np.array([[1.0]], dtype=np.float32),
    }
    with pytest.raises(FloatingPointError):
        train_step(m, bad, mse_fn, opt, ema=ema)
    assert ema.t == 2


# --- the loader ----------------------------------------------------------------------------


def test_dataloader_batches_in_order():
    # WHY: without shuffle the rows come in order; 10 rows in batches of 4
    #      give 2 batches with drop_last (the partial batch of 2 dropped) and
    #      3 without; every array is sliced with the same rows.
    # KIND: unit
    # CATCHES: s06, m02
    # CHAPTER: L0.5 section 4, The interface
    x = np.arange(10)
    y = np.arange(10) * 10
    dl = DataLoader({"x": x, "y": y}, batch_size=4, shuffle=False, rng=None)
    batches = list(dl)
    assert len(dl) == 2 and len(batches) == 2
    assert batches[1]["x"].tolist() == [4, 5, 6, 7] and batches[1]["y"].tolist() == [
        40,
        50,
        60,
        70,
    ]
    full = DataLoader({"x": x}, batch_size=4, shuffle=False, rng=None, drop_last=False)
    assert len(full) == 3 and [b["x"].tolist() for b in full][-1] == [8, 9]


def test_dataloader_shuffle_is_the_spec_permutation():
    # WHY: shuffling is spec/pcg32.md's Fisher-Yates on the generator you
    #      pass, drawn when each epoch starts: the order is reproducible from
    #      the seed, each epoch is a permutation of all rows, and epoch 2 is a
    #      new permutation (the generator moved on).
    # KIND: property
    # CATCHES: s07, s08
    # CHAPTER: L0.5 section 2, Principles (seeded data order)
    n = 13
    dl = DataLoader(
        {"i": np.arange(n)},
        batch_size=1,
        shuffle=True,
        rng=Rng(seed()),
        drop_last=False,
    )
    e1 = [int(b["i"][0]) for b in dl]
    e2 = [int(b["i"][0]) for b in dl]
    ref = Rng(seed())
    want1 = list(range(n))
    ref.shuffle(want1)
    want2 = list(range(n))
    ref.shuffle(want2)
    assert e1 == want1 and e2 == want2
    assert sorted(e1) == list(range(n)) and e1 != e2


def test_dataloader_rejects_bad_input():
    # WHY: arrays of different lengths would pair x with the wrong y; a
    #      shuffle without a generator would hide an unseeded order.
    # KIND: boundary
    # CATCHES: s09, m03
    # CHAPTER: L0.5 section 4, The interface
    with pytest.raises(ValueError):
        DataLoader(
            {"x": np.zeros(4), "y": np.zeros(5)}, batch_size=2, shuffle=False, rng=None
        )
    with pytest.raises(ValueError):
        DataLoader({"x": np.zeros(4)}, batch_size=2, shuffle=True, rng=None)
    with pytest.raises(ValueError):
        DataLoader({}, batch_size=2, shuffle=False, rng=None)
    with pytest.raises(ValueError):
        DataLoader({"x": np.zeros(4)}, batch_size=0, shuffle=False, rng=None)


# --- evaluation -------------------------------------------------------------------------------


def test_evaluate_weights_by_rows_and_restores_mode():
    # WHY: a last batch of 2 rows must count for 2 rows, not for a full
    #      batch; evaluation runs in eval mode under no_grad and hands the
    #      model back in the mode it found it (a model left in eval mode
    #      trains with dropout silently off).
    # KIND: unit
    # CATCHES: s10, s11, s12, s14
    # CHAPTER: L0.5 section 5, Pitfalls, item 2
    m = Sequential(Linear(1, 1, rng=Rng(0)), Dropout(0.5, rng=Rng(1)))
    m[0].load_state_dict({"weight": [[1.0]], "bias": [0.0]})
    seen = []

    def loss_fn(model, batch):
        out = model(Tensor(batch["x"]))
        seen.append((model.training, out.requires_grad))
        return F.mean(out), {"first": float(batch["x"][0, 0])}

    x = np.array([[1.0], [1.0], [1.0], [1.0], [4.0], [4.0]], dtype=np.float32)
    dl = DataLoader({"x": x}, batch_size=4, shuffle=False, rng=None, drop_last=False)
    r = evaluate(m, dl, loss_fn)
    assert_close(r["loss"], (4 * 1.0 + 2 * 4.0) / 6, dtype="float32")
    assert_close(r["first"], (4 * 1.0 + 2 * 4.0) / 6, dtype="float32")
    assert r["n"] == 6
    assert seen == [(False, False), (False, False)]
    assert m.training and m[1].training
    m.eval()
    evaluate(m, dl, loss_fn)
    assert not m.training


def test_evaluate_restores_mode_after_an_error():
    # WHY: an exception inside evaluation (a bad batch) must still hand the
    #      model back in training mode.
    # KIND: boundary
    # CATCHES: s13
    # CHAPTER: L0.5 section 5, Pitfalls, item 2
    m = Linear(1, 1, rng=Rng(0))

    def boom(model, batch):
        raise KeyError("bad batch")

    with pytest.raises(KeyError):
        evaluate(m, [{"x": np.zeros((1, 1))}], boom)
    assert m.training
    with pytest.raises(ValueError):
        evaluate(m, [], mse_fn)


# --- learning ---------------------------------------------------------------------------------


def two_moons(n: int, noise: float, s: int) -> tuple[np.ndarray, np.ndarray]:
    """n points on two interleaved half circles (sklearn's make_moons), labels 0 and 1."""
    g = PCG32(seed=s)
    xs, ys = [], []
    for i in range(n):
        t = math.pi * g.uniform()
        if i % 2 == 0:
            p = (math.cos(t), math.sin(t))
        else:
            p = (1.0 - math.cos(t), 0.5 - math.sin(t))
        xs.append((p[0] + noise * g.normal(), p[1] + noise * g.normal()))
        ys.append(i % 2)
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.int64)


def run_two_moons(s: int, steps_epochs: int = 15) -> tuple[Sequential, dict]:
    x, y = two_moons(256, 0.1, 1234)
    m = Sequential(
        Linear(2, 16, rng=Rng(s)),
        Tanh(),
        Linear(16, 16, rng=Rng(s + 1)),
        Tanh(),
        Linear(16, 2, rng=Rng(s + 2)),
    )
    opt = AdamW(m.parameters(), lr=0.03, weight_decay=0.0)
    dl = DataLoader({"x": x, "y": y}, batch_size=32, shuffle=True, rng=Rng(s + 3))

    def loss_fn(model, batch):
        logits = model(Tensor(batch["x"]))
        acc = float((logits.data.argmax(-1) == batch["y"]).mean())
        return cross_entropy(logits, batch["y"]), {"acc": acc}

    for _ in range(steps_epochs):
        for batch in dl:
            train_step(m, batch, loss_fn, opt, clip=5.0)
    full = DataLoader({"x": x, "y": y}, batch_size=64, shuffle=False, rng=None)
    return m, evaluate(m, full, loss_fn)


def test_learns_xor():
    # WHY: XOR is the smallest problem a linear model cannot solve: a 2-8-1
    #      tanh MLP trained 300 full-batch AdamW steps on the four points must
    #      classify all four and reach the reference's loss (mean + 3 sd over
    #      5 seeds). Every piece of L0.1 to L0.5 is on this path.
    # KIND: learning
    # CATCHES: s02
    # CHAPTER: L0.5 section 2, Principles (the training loop)
    s = seed()
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([[0], [1], [1], [0]], dtype=np.float32)
    m = Sequential(Linear(2, 8, rng=Rng(s)), Tanh(), Linear(8, 1, rng=Rng(s + 1)))
    opt = AdamW(m.parameters(), lr=0.05, weight_decay=0.0)
    batch = {"x": x, "y": y}

    def loss_fn(model, b):
        return bce_with_logits(model(Tensor(b["x"])), b["y"])

    for _ in range(300):
        stats = train_step(m, batch, loss_fn, opt)
    pred = m(Tensor(x)).data > 0
    assert (pred == (y > 0.5)).all(), pred.ravel()
    check("L0.5/test_learns_xor", "loss", stats["loss"], direction="max")


def test_learns_two_moons():
    # WHY: a real minibatch run: shuffled batches of 32, 15 epochs, AdamW
    #      with clipping, then an evaluation pass. The train loss must reach
    #      the reference bar and the accuracy must pass 95%.
    # KIND: learning
    # CATCHES: s02
    # CHAPTER: L0.5 section 2, Principles (the training loop)
    _, r = run_two_moons(seed())
    assert r["acc"] >= 0.95, r
    check("L0.5/test_learns_two_moons", "loss", r["loss"], direction="max")


def test_same_seed_bitwise_identical():
    # WHY: determinism by construction (P11): two runs with the same seed
    #      give bitwise-identical weights and metrics, and another seed gives
    #      others. Resuming (L0.6) and every regression test rely on it.
    # KIND: property
    # CATCHES: s15
    # CHAPTER: L0.5 section 2, Principles (seeded data order)
    a, ra = run_two_moons(seed(), steps_epochs=2)
    b, rb = run_two_moons(seed(), steps_epochs=2)
    c, rc = run_two_moons(seed() + 100, steps_epochs=2)
    for (na, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert (pa.data == pb.data).all(), na
    assert ra == rb and ra != rc
