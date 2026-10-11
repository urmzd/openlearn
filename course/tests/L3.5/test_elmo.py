"""Course tests for L3.5 (optional): ELMo's biLM, ScalarMix, and linear probes
(tinyllm/rnn/elmo.py).

Rung R0 for these: read them before you write code. Each test names why it
exists (WHY), what kind of check it is (KIND), the planted bugs it kills
(CATCHES, mutants in course/mutants/L3.5), and the chapter section it comes
from.

The worked example of the chapter (section 3): ScalarMix over two layers
with s = (0, ln 3) and gamma = 2. softmax(s) = (1/4, 3/4), so layers
R^0 = (1, 2) and R^1 = (5, 6) mix to 2 (1/4 (1, 2) + 3/4 (5, 6)) =
2 (4, 5) = (8, 10).

The golden fixture (course/fixtures/L3.5/elmo_torch.npz) holds torch 2.14.1
layers, logits, loss, mix, and gradients for a 2-layer biLM with copied
weights (course/oracle/L3.5/elmo_torch.py).
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.gradcheck import gradcheck
from _lib.pcg32 import PCG32
from _lib.thresholds import check
from tinyllm.autograd import functional as F
from tinyllm.autograd.tensor import Tensor
from tinyllm.optim.adamw import AdamW
from tinyllm.rnn.elmo import (
    BiLM,
    ScalarMix,
    bilm_loss,
    fit_linear_probe,
    probe_accuracy,
)

FIX = os.path.join(os.environ.get("TINYLLM_FIXTURES", ""), "L3.5", "elmo_torch.npz")


def seed() -> int:
    return int(os.environ.get("OL_SEED", "0"))


class Rng:
    """The frozen PCG32 behind the generator API of M06.3."""

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


def ids_batch(s: int, B=3, T=6, V=9):
    g = PCG32(seed=s)
    return np.array([[g.below(V) for _ in range(T)] for _ in range(B)], dtype=np.int64)


# --- the worked example ---------------------------------------------------------------


def test_hand_example_scalar_mix():
    # WHY: the chapter's worked example: s = (0, ln 3) gives weights (1/4,
    #      3/4), and gamma = 2 scales the convex mix: (8, 10).
    # KIND: unit
    # CATCHES: s01, s02
    # CHAPTER: L3.5 section 3, Worked example by hand
    mix = ScalarMix(2)
    mix.load_state_dict({"scalar_parameters": [0.0, math.log(3.0)], "gamma": [2.0]})
    out = mix([Tensor([1.0, 2.0]), Tensor([5.0, 6.0])])
    assert_close(out.data, [8.0, 10.0], dtype="float32")
    assert_close(mix.weights(), [0.25, 0.75], dtype="float32")


def test_scalar_mix_starts_uniform_and_trains():
    # WHY: at init every layer has the same weight 1 / (L + 1) and gamma = 1,
    #      so the mix is the plain average; the gradients reach s and gamma
    #      (the task learns which layer it wants), checked against the frozen
    #      central differences in float64.
    # KIND: gradcheck
    # CATCHES: s01, s02, m01
    # CHAPTER: L3.5 section 2.3, ScalarMix
    mix = ScalarMix(3)
    assert list(mix.state_dict()) == ["scalar_parameters", "gamma"]
    layers = [PCG32(seed=k).normal_array((2, 4)) for k in range(3)]
    avg = mix([Tensor(x, dtype=np.float64) for x in layers])
    assert_close(avg.data, sum(layers) / 3, rtol=1e-6, atol=1e-6)
    for p in mix.parameters():
        p.data = p.data.astype(np.float64) + np.array([0.3, -0.2, 0.5][: p.data.size])
    g = PCG32(seed=9).normal_array((2, 4))

    def f(s, gamma, *xs):
        mix.scalar_parameters.data, mix.gamma.data = s, gamma
        return float(np.sum(mix([Tensor(x, dtype=np.float64) for x in xs]).data * g))

    xs = [Tensor(x, requires_grad=True, dtype=np.float64) for x in layers]
    F.sum(mix(xs) * g).backward()
    start = [mix.scalar_parameters.data.copy(), mix.gamma.data.copy()] + layers
    gradcheck(
        f, start, [mix.scalar_parameters.grad, mix.gamma.grad] + [x.grad for x in xs]
    )


# --- against torch ----------------------------------------------------------------------------


def test_golden_torch():
    # WHY: the biLM built from torch nn.LSTMs over packed sequences (the
    #      backward stack reading each sentence reversed inside its length)
    #      with the same weights gives the same layers R^0..R^2, both
    #      directions' logits, bilm_loss, a ScalarMix, and gradients for every
    #      parameter. Lengths 5 and 3.
    # KIND: golden
    # CATCHES: s01, s02, s03, s04, s05, s06, s07, s08, m02
    # CHAPTER: L3.5 section 2.2, The biLM
    f = np.load(FIX)
    m = BiLM(9, 4, 2, rng=Rng(0))
    m.load_state_dict(
        {n[len("param.") :]: f[n] for n in f.files if n.startswith("param.")}
    )
    ids, lens = f["ids"], f["lengths"]
    reps = m.layers(ids, lens)
    for k, r in enumerate(reps):
        assert_close(r.data, f[f"layer{k}"], rtol=1e-4, atol=1e-6, msg=f"layer {k}")
    fl, bl = m(ids, lens)
    assert_close(fl.data, f["fwd_logits"], rtol=1e-4, atol=1e-5)
    assert_close(bl.data, f["bwd_logits"], rtol=1e-4, atol=1e-5)
    mix = ScalarMix(3)
    mix.load_state_dict({"scalar_parameters": f["s"], "gamma": f["gamma"]})
    out = mix(reps)
    assert_close(out.data, f["mix"], rtol=1e-4, atol=1e-6)
    loss = bilm_loss(m, ids, lens)
    assert_close(loss.data, f["loss"], rtol=1e-5, atol=1e-6)
    (loss + F.sum(out * Tensor(f["g"]))).backward()
    for name, p in m.named_parameters():
        assert_close(p.grad, f["grad." + name], rtol=1e-3, atol=1e-5, msg=name)
    assert_close(mix.scalar_parameters.grad, f["grad.s"], rtol=1e-4, atol=1e-6)
    assert_close(mix.gamma.grad, f["grad.gamma"], rtol=1e-4, atol=1e-6)


# --- directions -------------------------------------------------------------------------------------


def test_forward_layers_never_see_the_future():
    # WHY: the forward LM predicts token t + 1 from tokens 0..t, so f^k_t must
    #      not change when any later token changes (bit for bit), or it would
    #      be trained to copy the answer.
    # KIND: property
    # CATCHES: s03
    # CHAPTER: L3.5 section 5, Pitfalls, item 1
    m = BiLM(9, 4, 2, rng=Rng(1))
    ids = ids_batch(2)
    a = m.layers(ids)
    for t in range(5):
        ids2 = ids.copy()
        ids2[:, t + 1 :] = (ids2[:, t + 1 :] + 1) % 9
        b = m.layers(ids2)
        for k in (1, 2):
            assert np.array_equal(
                a[k].data[:, : t + 1, :4], b[k].data[:, : t + 1, :4]
            ), f"layer {k}, t {t}"


def test_backward_layers_never_see_the_past():
    # WHY: the mirror image: the backward LM predicts token t - 1 from tokens
    #      t..n-1, so b^k_t must not change when an EARLIER token changes.
    #      Reversing the whole padded tensor instead of each sequence inside
    #      its length breaks this for the shorter rows.
    # KIND: property
    # CATCHES: s05, s06
    # CHAPTER: L3.5 section 5, Pitfalls, item 2
    m = BiLM(9, 4, 2, rng=Rng(3))
    ids = ids_batch(4)
    lens = np.array([6, 4, 2])
    a = m.layers(ids, lens)
    for t in range(1, 6):
        ids2 = ids.copy()
        ids2[:, :t] = (ids2[:, :t] + 1) % 9
        b = m.layers(ids2, lens)
        for k in (1, 2):
            assert np.array_equal(a[k].data[:, t:, 4:], b[k].data[:, t:, 4:]), (
                f"layer {k}, t {t}"
            )


def test_padding_never_leaks():
    # WHY: whatever sits past a row's length must not change any real
    #      position of any layer, and the padded positions of R^1..R^L are 0.
    #      The backward LM starts at each row's last REAL token.
    # KIND: property
    # CATCHES: s04, s06
    # CHAPTER: L3.5 section 5, Pitfalls, item 2
    m = BiLM(9, 4, 2, rng=Rng(5))
    ids = ids_batch(6)
    lens = np.array([6, 3, 1])
    ids2 = ids.copy()
    ids2[1, 3:] = 8
    ids2[2, 1:] = 7
    a, b = m.layers(ids, lens), m.layers(ids2, lens)
    real = np.arange(6)[None, :] < lens[:, None]
    for k in range(3):
        assert np.array_equal(a[k].data[real], b[k].data[real])
    for k in (1, 2):
        assert np.all(a[k].data[~real] == 0.0)
    alone = m.layers(ids[1:2, :3], np.array([3]))
    assert_close(alone[2].data[0], a[2].data[1, :3], rtol=1e-5, atol=1e-6)


def test_bilm_loss_targets():
    # WHY: the forward half scores ids[t + 1] at t = 0..len-2 and the backward
    #      half ids[t - 1] at t = 1..len-1, each a mean over its own
    #      predictions, and the loss is their average. A row of length 1 has
    #      nothing to predict.
    # KIND: unit
    # CATCHES: s07, s08
    # CHAPTER: L3.5 section 2.2, The biLM
    m = BiLM(9, 4, 1, rng=Rng(7))
    ids = ids_batch(8)
    lens = np.array([6, 3, 1])
    fl, bl = m(ids, lens)
    fz, bz = fl.data.astype(np.float64), bl.data.astype(np.float64)

    def nll(z, t):
        return -(z[t] - z.max() - np.log(np.exp(z - z.max()).sum()))

    fwd = [nll(fz[b, t], ids[b, t + 1]) for b in range(3) for t in range(lens[b] - 1)]
    bwd = [nll(bz[b, t], ids[b, t - 1]) for b in range(3) for t in range(1, lens[b])]
    assert_close(
        bilm_loss(m, ids, lens).data,
        (np.mean(fwd) + np.mean(bwd)) / 2,
        rtol=1e-5,
        atol=1e-6,
    )
    with pytest.raises(ValueError):
        bilm_loss(m, ids, np.array([1, 1, 1]))


# --- probes -------------------------------------------------------------------------------------------


def test_probe_separates_separable_data():
    # WHY: a linear probe is softmax regression and nothing more: on three
    #      linearly separable clusters it reaches accuracy 1, and the returned
    #      W, b already include the standardization (predict with X @ W + b
    #      on the raw features).
    # KIND: unit
    # CATCHES: s09, s10
    # CHAPTER: L3.5 section 2.4, Linear probes
    g = PCG32(seed=11)
    centers = np.array([[20.0, 0.0], [0.0, 20.0], [-20.0, -20.0]])
    y = np.array([i % 3 for i in range(60)])
    X = centers[y] + g.normal_array((60, 2)) + np.array([1000.0, -500.0])
    W, b = fit_linear_probe(X, y, 3, steps=300)
    assert W.shape == (2, 3) and b.shape == (3,)
    assert probe_accuracy(W, b, X, y) == 1.0
    assert probe_accuracy(
        np.zeros((2, 3)), np.array([0.0, 1.0, 0.0]), X, y
    ) == pytest.approx(1 / 3)


def test_probe_is_deterministic_and_validates():
    # WHY: full-batch gradient descent from zeros has no randomness: the same
    #      data give the same probe bit for bit, so a probe accuracy is a
    #      property of the representation. Wrong shapes or labels fail.
    # KIND: boundary
    # CATCHES: m03
    # CHAPTER: L3.5 section 2.4, Linear probes
    X = PCG32(seed=12).normal_array((30, 4))
    y = np.array([i % 2 for i in range(30)])
    W1, b1 = fit_linear_probe(X, y, 2)
    W2, b2 = fit_linear_probe(X, y, 2)
    assert np.array_equal(W1, W2) and np.array_equal(b1, b2)
    for bad in (
        (X[0], y, 2),
        (X, y[:5], 2),
        (X, y + 5, 2),
        (X, np.zeros(30, dtype=np.int64), 1),
    ):
        with pytest.raises(ValueError):
            fit_linear_probe(*bad)


# --- learning -----------------------------------------------------------------------------------------

THE, THEY, SAW, DOG, CAT, WOOD, CUTS, STOP = range(8)


def sentences(n: int, g: PCG32):
    """'the saw cuts wood .' (saw is a noun) and 'they saw the dog .' (a
    verb): the label of each SAW is 0 after THE and 1 after THEY."""
    rows = []
    for _ in range(n):
        if g.below(2) == 0:
            rows.append(
                [THE, SAW, CUTS]
                + ([WOOD] if g.below(2) else [THE, [DOG, CAT][g.below(2)]])
                + [STOP]
            )
        else:
            rows.append([THEY, SAW, THE, [DOG, CAT, SAW][g.below(3)], STOP])
    return rows


def test_contextual_layers_disambiguate():
    # WHY: the point of ELMo. A biLM trained for 150 steps on two sentence
    #      types lowers its loss to the reference within 3 sd; then a linear
    #      probe on the frozen layers tags "saw" as noun or verb. The
    #      embedding layer R^0 gives every "saw" the same vector, so it can
    #      do no better than the majority; R^1 knows the word before it.
    # KIND: learning
    # CATCHES: s09, s10
    # CHAPTER: L3.5 section 1, Why now
    g = PCG32(seed=500 + seed())
    m = BiLM(8, 8, 2, rng=Rng(600 + seed()))
    opt = AdamW(m.parameters(), lr=0.02, weight_decay=0.0)
    for _ in range(150):
        rows = sentences(16, g)
        lens = np.array([len(r) for r in rows])
        ids = np.zeros((16, 6), dtype=np.int64)
        for i, r in enumerate(rows):
            ids[i, : len(r)] = r
        opt.zero_grad()
        loss = bilm_loss(m, ids, lens)
        loss.backward()
        opt.step()
    rows = sentences(80, PCG32(seed=77))
    lens = np.array([len(r) for r in rows])
    ids = np.zeros((80, 6), dtype=np.int64)
    for i, r in enumerate(rows):
        ids[i, : len(r)] = r
    final = float(bilm_loss(m, ids, lens).data)
    reps = m.layers(ids, lens)
    pos = np.argwhere((ids == SAW) & (np.arange(6)[None, :] < lens[:, None]))
    labels = np.array([0 if ids[b, t - 1] == THE else 1 for b, t in pos])
    acc = []
    for k in (0, 1):
        X = reps[k].data[pos[:, 0], pos[:, 1]].astype(np.float64)
        W, b = fit_linear_probe(X, labels, 2)
        acc.append(probe_accuracy(W, b, X, labels))
    majority = max(labels.mean(), 1 - labels.mean())
    assert acc[0] <= majority + 1e-9, (
        f"R^0 probe {acc[0]:.2f} beat the majority {majority:.2f}"
    )
    assert acc[1] >= 0.95, f"R^1 probe {acc[1]:.2f}"
    check(
        "L3.5/test_contextual_layers_disambiguate", "bilm_loss", final, direction="max"
    )
