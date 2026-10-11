"""Course tests for L0.5's take-over of tinyllm/lm/bigram.py: BigramLogits,
the autograd-trained bigram, and PCG32 sampling.

L0.0's tests still run against this file as L0.5's regression suite (the
count model, the C logits, the safetensors bytes): taking a unit over keeps
its contract.

The worked example of the chapter (section 3, second half): BigramLogits(3)
starts at zeros, so every next token has probability 1/3 and the NLL of any
text is ln 3. On the text "abbacab" the gradient of the mean NLL with
respect to row a (three predictions from a: b, c, b) is
(3 * [1/3, 1/3, 1/3] - [0, 2, 1]) / 6 = [1/6, -1/6, 0].
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32
from tinyllm.autograd.losses import cross_entropy
from tinyllm.lm.bigram import BigramLM, BigramLogits
from tinyllm.optim.adamw import AdamW
from tinyllm.train.loop import train_step

HAND_IDS = np.array([0, 1, 1, 0, 2, 0, 1])  # "abbacab", a=0 b=1 c=2
TEXT = (
    "The lamp keeper climbed the stairs at dusk and wound the clock that turned the light. "
    "Ships passed far out on the water and never saw her, but they saw the beam, and the beam "
    "was enough. In winter the wind came off the sea and shook the glass, and she wrote the "
    "names of the ships in a book: the Heron, the Lark, the Patience, the Morning Star. "
    "When the clock ran down the light stopped, so she never let the clock run down. "
)


def seed() -> int:
    return int(os.environ.get("OL_SEED", "0"))


def nll_loss(model, batch):
    ids = batch["ids"]
    return cross_entropy(model(ids[:-1]), ids[1:])


def test_hand_example_bigram_gradient():
    # WHY: the chapter's worked example: the zero table predicts 1/3 for
    #      everything (NLL ln 3), and row a's gradient is the average of
    #      (softmax - onehot) over the three predictions made from a.
    # KIND: unit
    # CATCHES: s16, s18
    # CHAPTER: L0.5 section 3, Worked example by hand
    m = BigramLogits(3)
    loss = nll_loss(m, {"ids": HAND_IDS})
    assert_close(loss.data, math.log(3.0), dtype="float32")
    loss.backward()
    assert_close(m.weight.grad[0], [1 / 6, -1 / 6, 0.0], dtype="float32")
    assert list(m.state_dict()) == ["weight"]
    assert m.weight.shape == (3, 3) and BigramLogits().weight.shape == (256, 256)


def test_forward_is_a_row_gather():
    # WHY: logits for ids are the rows of the table, the one-hot matmul of
    #      L0.0 without the zeros, for ids of any shape (the trainer feeds
    #      [B, T] windows from L0.6's TokenStream); out-of-range ids are an
    #      error.
    # KIND: unit
    # CATCHES: s17
    # CHAPTER: L0.5 section 2, Principles (the same model, trained)
    m = BigramLogits(4)
    m.load_state_dict({"weight": np.arange(16.0).reshape(4, 4)})
    assert_close(
        m(np.array([3, 0, 3])).data,
        [[12, 13, 14, 15], [0, 1, 2, 3], [12, 13, 14, 15]],
        dtype="float32",
    )
    batch = m(np.array([[3, 0], [1, 1]])).data
    assert batch.shape == (2, 2, 4) and (batch[1, 0] == [4, 5, 6, 7]).all()
    with pytest.raises(ValueError):
        m(np.array([4]))


def test_autograd_bigram_reaches_the_count_mle():
    # WHY: the bigram NLL is convex in the table, and its minimum is the
    #      count model with no smoothing: -1/N sum_ij C_ij ln(C_ij / R_i).
    #      Your autograd, loss, and optimizer must get within 1e-3 nats of it
    #      (and never below it). This is MS-L0's second step in miniature.
    # KIND: property
    # CATCHES: s18
    # CHAPTER: L0.5 section 2, Principles (the same model, trained)
    ids = np.frombuffer(TEXT.encode(), dtype=np.uint8).astype(np.int64)
    counts = np.zeros((256, 256))
    np.add.at(counts, (ids[:-1], ids[1:]), 1.0)
    rows = counts.sum(axis=1, keepdims=True)
    nz = counts > 0
    mle = float(
        -(counts[nz] * np.log((counts / np.where(rows > 0, rows, 1))[nz])).sum()
        / (len(ids) - 1)
    )
    m = BigramLogits(256)
    opt = AdamW(m.parameters(), lr=0.5, weight_decay=0.0)
    for step in range(400):
        opt.lr = 0.5 if step < 300 else 0.05
        stats = train_step(m, {"ids": ids}, nll_loss, opt)
    final = float(nll_loss(m, {"ids": ids}).data)
    assert final >= mle - 1e-4, (final, mle)
    assert final - mle <= 1e-3, (final, mle, stats)


def test_to_lm_serves_the_same_table():
    # WHY: to_lm() hands the trained table to BigramLM, whose logits go
    #      through your C matmul and whose checkpoint the unchanged tracer
    #      engine serves (MS-L0 step 2): same numbers, float32, and a copy
    #      (training on must not change a model already handed off).
    # KIND: unit
    # CATCHES: s19
    # CHAPTER: L0.5 section 4, The interface
    m = BigramLogits(5)
    m.load_state_dict({"weight": PCG32(seed=seed()).uniform_array((5, 5), -2, 2)})
    lm = m.to_lm()
    assert isinstance(lm, BigramLM) and lm.weight.dtype == np.float32
    ids = np.array([4, 0, 2])
    assert_close(lm.logits(ids), m(ids).data, dtype="float32")
    m.weight.data[...] = 0.0
    assert not (lm.weight == 0).all()


def engine_sample(
    w: np.ndarray, prefix: list[int], n: int, temperature: float, s: int
) -> list[int]:
    """The tracer engine's sampler (L10.0, http.rs `sample`) over the frozen
    PCG32: f64 weights exp(z - max) summed in id order, u = uniform() * total,
    the first id whose running sum exceeds u."""
    g = PCG32(seed=s, seq=54)
    cur, out = prefix[-1], []
    for _ in range(n):
        z = [float(x) / temperature for x in w[cur]]
        mx = max(z)
        e = [math.exp(x - mx) for x in z]
        total = 0.0
        for x in e:
            total += x
        u = g.uniform() * total
        cum, pick = 0.0, len(e) - 1
        for i, x in enumerate(e):
            cum += x
            if cum > u:
                pick = i
                break
        out.append(pick)
        cur = pick
    return out


def test_sample_draws_like_the_engine():
    # WHY: from L0.5 the Python sampler draws from PCG32(seed) (stream 54)
    #      exactly as the Rust tracer engine does, so `generate --seed S` and
    #      the engine's completion with seed S produce the same ids.
    # KIND: unit
    # CATCHES: s20, s21
    # CHAPTER: L0.5 section 5, Pitfalls, item 5
    w = PCG32(seed=7).uniform_array((6, 6), -1.5, 1.5).astype(np.float32)
    lm = BigramLM(w)
    for t, s in ((1.0, seed()), (0.7, seed() + 1), (1.6, 99)):
        assert lm.sample([2], 40, t, s) == engine_sample(
            w.astype(np.float64), [2], 40, t, s
        )
