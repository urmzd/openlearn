"""Course tests for L2.2: Bengio's NPLM (tinyllm/lm/nplm.py).

Rung R0 for these course tests (your own graded tests are rung R3: write
them first, see the chapter's "How to work this chapter"). Each test names
why it exists (WHY), what kind of check it is (KIND), the planted bugs it
kills (CATCHES, mutants in course/mutants/L2.2), and the chapter section it
comes from.

The chapter's worked example (section 3) is an NPLM with V = 3, a window of
two tokens, one-dimensional embeddings C = (1, 0, -1), one hidden unit
H = (0.5, -0.5), d = 0, U = (1, 0, -1), b = (0, 0.5, 0), and the direct
path W = [[1, 0], [0, 0], [0, 1]]. The window (a, c) gives x = (1, -1),
h = tanh(1), and logits (1 + tanh 1, 0.5, -1 - tanh 1).

The golden values in $TINYLLM_FIXTURES/L2.2/nplm_torch.npz come from the
same architecture written directly in torch (course/oracle/L2.2/nplm_torch.py).
The learning test trains on the first 50 000 bytes of the MS-L2 corpus
(course/oracle/MS-L2/corpus.py) and compares against
course/fixtures/ref-thresholds.tsv (the reference's mean + 3 sd over 5 seeds).
"""

from __future__ import annotations

import json
import math
import os
import struct
from pathlib import Path

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.gradcheck import gradcheck
from _lib.pcg32 import PCG32
from _lib.thresholds import check
from tinyllm.autograd.losses import cross_entropy
from tinyllm.io.tokens import open_tokens
from tinyllm.lm.nplm import NPLM, load_nplm, save_nplm, train_nplm, windows
from tinyllm.nn.layers import Embedding, Linear
from tinyllm.optim.sgd import SGD
from tinyllm.train.loop import DataLoader, train_step

FIX = Path(os.environ.get("TINYLLM_FIXTURES", ""))
TANH1 = math.tanh(1.0)
# exp of the MS-L2 bigram bar: L0.0's add-one bigram, fitted on train.bin,
# scores val.bin at 2.142401587 nats per byte (course/oracle/MS-L2/corpus.py).
BIGRAM_PPL = math.exp(2.142401587)


def seed() -> int:
    return int(os.environ.get("OL_SEED", "0"))


class Rng:
    """The frozen PCG32 behind the generator API of M06.3 (uniform, uniforms,
    below, shuffle), so no verdict here depends on your PCG32."""

    def __init__(self, s: int, seq: int = 54) -> None:
        self.g = PCG32(seed=s, seq=seq)

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


def hand_model(direct: bool = True) -> NPLM:
    m = NPLM(3, 2, 1, 1, direct=direct, rng=Rng(0))
    sd = {
        "emb.weight": [[1.0], [0.0], [-1.0]],
        "hidden.weight": [[0.5, -0.5]],
        "hidden.bias": [0.0],
        "out.weight": [[1.0], [0.0], [-1.0]],
        "out.bias": [0.0, 0.5, 0.0],
    }
    if direct:
        sd["direct.weight"] = [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]
    m.load_state_dict(sd)
    return m


def golden(tag: str) -> dict:
    z = np.load(FIX / "L2.2" / "nplm_torch.npz")
    return {k[len(tag) + 1 :]: z[k] for k in z.files if k.startswith(tag + "/")}


def from_golden(g: dict, direct: bool) -> NPLM:
    meta = json.loads(
        bytes(np.load(FIX / "L2.2" / "nplm_torch.npz")["__meta__"]).decode()
    )
    m = NPLM(
        meta["V"],
        meta["context"],
        meta["d_emb"],
        meta["d_hidden"],
        direct=direct,
        rng=Rng(1),
    )
    m.load_state_dict(
        {k[len("param.") :]: v for k, v in g.items() if k.startswith("param.")}
    )
    return m


def small(direct: bool = True, s: int = 3) -> NPLM:
    return NPLM(7, 2, 3, 4, direct=direct, rng=Rng(s, 1))


def corpus(name: str) -> np.ndarray:
    return np.asarray(open_tokens(str(FIX / "MS-L2" / name)), dtype=np.int64)


# --- the worked example ----------------------------------------------------------


def test_windows_hand_example():
    # WHY: every training example is a window of `context` tokens, oldest
    #      first, and the token right after it. For ids 5 6 7 8 and a window
    #      of 2: (5 6) -> 7 and (6 7) -> 8; the first two tokens are never
    #      targets because they have no full window.
    # KIND: unit, smoke
    # CATCHES: s05
    # CHAPTER: L2.2 section 3
    ctx, tgt = windows(np.array([5, 6, 7, 8]), 2)
    assert ctx.dtype == np.int64 and tgt.dtype == np.int64
    assert ctx.tolist() == [[5, 6], [6, 7]] and tgt.tolist() == [7, 8]
    ctx, tgt = windows([1, 2, 3, 4, 5], 1)
    assert ctx.tolist() == [[1], [2], [3], [4]] and tgt.tolist() == [2, 3, 4, 5]


def test_hand_example_forward():
    # WHY: the chapter's worked example, number for number. The window (a, c)
    #      concatenates C[a] = 1 and C[c] = -1 OLDEST FIRST into x = (1, -1);
    #      H x + d = 1, so h = tanh 1; the direct path adds W x = (1, 0, -1).
    #      Logits (1 + tanh 1, 0.5, -1 - tanh 1); the reversed window (c, a)
    #      gives the mirror image, so the order of the concatenation matters.
    # KIND: unit, smoke
    # CATCHES: s02, s03
    # CHAPTER: L2.2 section 3
    m = hand_model()
    y = m(np.array([[0, 2], [2, 0]]))
    assert y.shape == (2, 3) and y.dtype == np.float32
    assert_close(
        y.data,
        [[1 + TANH1, 0.5, -1 - TANH1], [-1 - TANH1, 0.5, 1 + TANH1]],
        dtype="float32",
    )
    # Without the direct path only b + U h remains.
    y = hand_model(direct=False)(np.array([[0, 2]]))
    assert_close(y.data, [[TANH1, 0.5, -TANH1]], dtype="float32")


def test_hand_example_gradient_reaches_the_bias():
    # WHY: the cross-entropy gradient at the logits is p - onehot(target), and
    #      the output bias receives exactly that. With target a: p = softmax(
    #      1 + tanh 1, 0.5, -1 - tanh 1) = (0.76181, 0.21571, 0.02247).
    # KIND: unit
    # CATCHES: s01, s02, s03
    # CHAPTER: L2.2 section 3
    m = hand_model()
    y = np.array([1 + TANH1, 0.5, -1 - TANH1])
    p = np.exp(y - y.max()) / np.exp(y - y.max()).sum()
    loss = cross_entropy(m(np.array([[0, 2]])), np.array([0]))
    assert_close(float(loss.data), -math.log(p[0]), dtype="float32")
    loss.backward()
    assert_close(m.out.bias.grad, p - np.array([1.0, 0.0, 0.0]), dtype="float32")
    # dL/dC flows back through both paths into the two rows the window used.
    assert np.abs(m.emb.weight.grad[[0, 2]]).sum() > 0 and m.emb.weight.grad[
        1
    ].tolist() == [0.0]


# --- golden and gradients -----------------------------------------------------------


@pytest.mark.parametrize("tag,direct", [("direct", True), ("nodirect", False)])
def test_golden_torch(tag, direct):
    # WHY: the same architecture written directly in torch from the paper's
    #      equation, with the same parameter names: logits, the mean
    #      cross-entropy, and the gradient of every parameter must agree. The
    #      batch repeats an id inside a window and across rows, so the
    #      embedding gradient must add up every use.
    # KIND: golden
    # CATCHES: s01, s02, s03
    # CHAPTER: L2.2 section 2.2
    g = golden(tag)
    m = from_golden(g, direct)
    y = m(g["ids"])
    assert_close(y.data, g["logits"], dtype="float32")
    loss = cross_entropy(y, g["targets"])
    assert_close(float(loss.data), float(g["loss"]), dtype="float32")
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, f"{name} got no gradient"
        assert_close(p.grad, g[f"grad.{name}"], rtol=1e-4, atol=1e-6, msg=name)


def test_gradcheck_every_parameter():
    # WHY: backward must reach every parameter, the embedding table C
    #      included. The model is built from the op library, so autograd does
    #      the work; a forward that reads C's numpy array instead of calling
    #      the embedding op cuts C out of the graph and it never learns. The
    #      frozen central-difference check runs in float64.
    # KIND: gradcheck
    # CATCHES: s01, s03
    # CHAPTER: L2.2 section 5, Pitfalls, item 1
    m = small()
    params = [p for _, p in m.named_parameters()]
    for p in params:
        p.data = p.data.astype(np.float64)
    ctx = np.array([[0, 1], [1, 1], [6, 2], [3, 0]])
    tgt = np.array([2, 0, 5, 3])

    def f(*arrays):
        for p, a in zip(params, arrays):
            p.data = a
        return float(cross_entropy(m(ctx), tgt).data)

    start = [p.data.copy() for p in params]
    m.zero_grad()
    cross_entropy(m(ctx), tgt).backward()
    analytic = [p.grad.copy() for p in params]
    gradcheck(f, start, analytic, names=[n for n, _ in m.named_parameters()])


# --- the parameters ---------------------------------------------------------------------


def test_state_dict_names_and_shapes():
    # WHY: the parameter names ARE the safetensors keys of the model directory
    #      the zoo (L6.7) loads; registration order fixes their order. The
    #      direct path is one optional bias-free matrix.
    # KIND: unit, smoke
    # CHAPTER: L2.2 section 4
    sd = NPLM(7, 3, 4, 5, rng=Rng(0)).state_dict()
    assert [(k, v.shape) for k, v in sd.items()] == [
        ("emb.weight", (7, 4)),
        ("hidden.weight", (5, 12)),
        ("hidden.bias", (5,)),
        ("out.weight", (7, 5)),
        ("out.bias", (7,)),
        ("direct.weight", (7, 12)),
    ]
    assert all(v.dtype == np.float32 for v in sd.values())
    assert list(NPLM(7, 3, 4, 5, direct=False, rng=Rng(0)).state_dict()) == list(sd)[:5]


def test_init_draws_from_one_generator():
    # WHY: one rng feeds every layer in registration order, so a seed fixes
    #      the whole model. Restarting the default generator for each layer
    #      makes hidden.weight a rescaled copy of the embedding draws: the
    #      layers start correlated.
    # KIND: unit
    # CATCHES: s04
    # CHAPTER: L2.2 section 5, Pitfalls, item 4
    r = Rng(9, 1)
    want = [
        Embedding(7, 3, rng=r),
        Linear(6, 4, rng=r),
        Linear(4, 7, rng=r),
        Linear(6, 7, bias=False, rng=r),
    ]
    got = NPLM(7, 2, 3, 4, rng=Rng(9, 1))
    for mod, name in zip(want, ("emb", "hidden", "out", "direct")):
        for k, v in mod.state_dict().items():
            assert_close(
                got.state_dict()[f"{name}.{k}"],
                v,
                rtol=0.0,
                atol=0.0,
                msg=f"{name}.{k}",
            )
    a, b = NPLM(7, 2, 3, 4), NPLM(7, 2, 3, 4)
    for k in a.state_dict():
        assert_close(a.state_dict()[k], b.state_dict()[k], rtol=0.0, atol=0.0, msg=k)
    e = a.emb.weight.data.ravel()[:8].astype(np.float64)
    h = a.hidden.weight.data.ravel()[:8].astype(np.float64) * math.sqrt(6)
    assert np.abs(e - h).max() > 1e-3, "hidden.weight repeats the embedding draws"


# --- scoring -------------------------------------------------------------------------------


def test_nll_matches_cross_entropy_per_window():
    # WHY: nll scores every token that has a full window, one number per
    #      token, in float64; batching is an implementation detail and must
    #      not change the values beyond rounding. Its mean is the loss on the
    #      same windows; perplexity is exp of that mean.
    # KIND: unit
    # CATCHES: s12, m02
    # CHAPTER: L2.2 section 2.4
    m = small()
    ids = np.array([0, 1, 2, 6, 5, 4, 3, 3, 2, 1, 0, 6, 6])
    nll = m.nll(ids)
    assert nll.shape == (11,) and nll.dtype == np.float64
    ctx, tgt = windows(ids, 2)
    z = m(ctx).data.astype(np.float64)
    lse = z.max(axis=1) + np.log(np.exp(z - z.max(axis=1, keepdims=True)).sum(axis=1))
    assert_close(nll, lse - z[np.arange(11), tgt], rtol=1e-12, atol=1e-12)
    # Batches of 3 windows: the same numbers up to float32 rounding (BLAS
    # may block a matmul differently for a different batch size).
    assert_close(m.nll(ids, batch_size=3), nll, rtol=1e-6, atol=1e-7)
    assert_close(float(cross_entropy(m(ctx), tgt).data), nll.mean(), dtype="float32")
    assert_close(
        m.perplexity(ids), math.exp(math.fsum(nll.tolist()) / 11), rtol=1e-12, atol=0.0
    )


# --- generation ---------------------------------------------------------------------------


def expected_sample(
    m: NPLM, prefix: list[int], n: int, temperature: float, s: int
) -> list[int]:
    """The contract's draw, rebuilt here with the frozen PCG32(seed), stream 54."""
    g = PCG32(seed=s)
    hist, out = list(prefix), []
    for _ in range(n):
        row = m(np.array([hist[-m.context :]])).data[0].astype(np.float64)
        w = np.exp(row / temperature - (row / temperature).max()).tolist()
        total = 0.0
        for x in w:
            total += x
        target, cum, pick = g.uniform() * total, 0.0, len(w) - 1
        for i, x in enumerate(w):
            cum += x
            if cum > target:
                pick = i
                break
        out.append(pick)
        hist.append(pick)
    return out


def test_generate_seeded_and_greedy():
    # WHY: generation conditions each step on the last `context` ids and
    #      draws with L0.5's rule (one PCG32(seed) per call, one uniform per
    #      token), so the same seed gives the same text, the MS-L2 check. At
    #      temperature 0 it is the step-by-step argmax.
    # KIND: unit
    # CATCHES: s06, m04
    # CHAPTER: L2.2 section 4
    m = small(s=5)
    for p in m.parameters():
        p.data *= 3.0  # peaked distributions: many different ids get drawn
    prefix = [1, 4, 2]
    got = m.generate(prefix, 12, 0.8, seed=7)
    assert got == expected_sample(m, prefix, 12, 0.8, 7)
    assert got == m.generate(prefix, 12, 0.8, seed=7)
    assert m.generate([3, 6], 0, 1.0, seed=0) == []
    greedy = m.generate(prefix, 5, 0.0, seed=99)
    hist = list(prefix)
    for t in greedy:
        assert t == int(np.argmax(m(np.array([hist[-2:]])).data[0]))
        hist.append(t)


def test_greedy_ties_go_to_the_lowest_id():
    # WHY: D11: at temperature 0 equal logits go to the lowest id, the rule
    #      every engine of the course shares. A model with every weight 0 has
    #      equal logits everywhere, so greedy decoding always picks id 0.
    # KIND: boundary
    # CATCHES: s10
    # CHAPTER: L2.2 section 2.5
    flat = small()
    for p in flat.parameters():
        p.data[...] = 0.0
    assert flat.generate([5, 6], 4, 0.0, seed=0) == [0, 0, 0, 0]


def test_generate_uses_the_last_window():
    # WHY: a prompt longer than the window must condition on its END, the
    #      tokens right before the one being predicted; the start of the
    #      prompt is out of reach of an NPLM.
    # KIND: boundary
    # CATCHES: s06
    # CHAPTER: L2.2 section 5, Pitfalls, item 5
    m = small(s=11)
    for p in m.parameters():
        p.data *= 3.0
    long = [0, 1, 2, 3, 4, 5, 6, 2, 5]
    assert m.generate(long, 6, 0.0, seed=0) == m.generate(long[-2:], 6, 0.0, seed=0)
    assert m.generate(long, 6, 1.0, seed=3) == m.generate(long[-2:], 6, 1.0, seed=3)


# --- training -------------------------------------------------------------------------------


def l05_loop(
    model, ids, steps, batch, lr, rng, momentum=0.0, weight_decay=0.0, clip=None
):
    """train_nplm rebuilt from L0.5's DataLoader and train_step and M10.2's SGD."""
    ctx, tgt = windows(ids, model.context)
    loader = DataLoader(
        {"x": ctx, "y": tgt}, batch, shuffle=True, rng=rng, drop_last=True
    )
    opt = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    out = []
    while len(out) < steps:
        for b in loader:
            out.append(
                train_step(
                    model,
                    b,
                    lambda m, x: cross_entropy(m(x["x"]), x["y"]),
                    opt,
                    clip=clip,
                )["loss"]
            )
            if len(out) == steps:
                break
    return out


def same_run(steps, ids, batch, **kw):
    a, b = small(s=21), small(s=21)
    got = train_nplm(a, ids, steps, batch, 0.2, Rng(4, 3), **kw)
    want = l05_loop(b, ids, steps, batch, 0.2, Rng(4, 3), **kw)
    assert len(got) == steps
    assert_close(got, want, rtol=0.0, atol=0.0)
    for k, v in b.state_dict().items():
        assert_close(a.state_dict()[k], v, rtol=0.0, atol=0.0, msg=k)


IDS38 = np.array([(3 * i + i // 7) % 7 for i in range(38)])


def test_train_nplm_is_the_l05_loop():
    # WHY: train_nplm is exactly L0.5's loop: one shuffled DataLoader that
    #      cycles epochs and one SGD created once, so its momentum buffers
    #      live across epochs. 10 steps of 4 batches per epoch cross two epoch
    #      boundaries; the loop rebuilt here must agree bit for bit.
    # KIND: differential
    # CATCHES: s07, s08
    # CHAPTER: L2.2 section 2.3
    same_run(10, IDS38, 8, momentum=0.9, weight_decay=0.01)


def test_train_nplm_runs_exactly_steps():
    # WHY: `steps` is the number of optimizer steps, however many epochs that
    #      takes. Here the data hold exactly one batch (8 windows), so every
    #      step is a new epoch, and a batch as large as the data is allowed.
    # KIND: boundary
    # CATCHES: s08, m03
    # CHAPTER: L2.2 section 2.3
    same_run(3, IDS38[:10], 8)


def test_train_nplm_passes_the_clip():
    # WHY: the clip reaches train_step: with clip 0.5 the global gradient
    #      norm of these first steps is above 0.5, so a loop that drops it
    #      takes larger steps (one epoch, no momentum).
    # KIND: unit
    # CATCHES: s11
    # CHAPTER: L2.2 section 2.3
    same_run(3, IDS38, 8, clip=0.5)


def test_nplm_learns_the_corpus():
    # WHY: the real check: 300 SGD steps on 50 000 bytes of the MS-L2 corpus
    #      must reach the reference's validation perplexity (its mean + 3 sd
    #      over 5 seeds), far below the count bigram's 8.52, because a window
    #      of six bytes sees whole words.
    # KIND: learning
    # CATCHES: m02
    # CHAPTER: L2.2 section 1
    s = seed()
    m = NPLM(256, 6, 16, 96, rng=Rng(s, 1))
    losses = train_nplm(
        m, corpus("train.bin")[:50_000], 300, 128, 0.1, Rng(s, 3), momentum=0.9
    )
    assert len(losses) == 300 and losses[-1] < losses[0]
    ppl = m.perplexity(corpus("val.bin"))
    assert ppl < BIGRAM_PPL
    check("L2.2/test_nplm_learns_the_corpus", "val_ppl", ppl, direction="max")


# --- the model directory -------------------------------------------------------------------


def header(path: Path) -> dict:
    raw = path.read_bytes()
    (n,) = struct.unpack("<Q", raw[:8])
    return json.loads(raw[8 : 8 + n])


@pytest.mark.parametrize("direct", [True, False])
def test_save_load_roundtrip(tmp_path, direct):
    # WHY: `{tinyllm} lm train nplm` writes this directory and the zoo and
    #      `{tinyllm} generate` load it: config.json names the architecture
    #      and its sizes, model.safetensors holds state_dict() as F32, and the
    #      reloaded model computes the same logits bit for bit.
    # KIND: unit
    # CATCHES: s09
    # CHAPTER: L2.2 section 4
    m = NPLM(9, 3, 2, 5, direct=direct, rng=Rng(6))
    save_nplm(m, str(tmp_path / "nplm"))
    cfg = json.loads((tmp_path / "nplm" / "config.json").read_text())
    assert cfg == {
        "tl_arch": "nplm",
        "tl_tokenizer": "bytes",
        "vocab_size": 9,
        "tl_format": 1,
        "tl_context": 3,
        "tl_d_emb": 2,
        "hidden_size": 5,
        "tl_direct": direct,
    }
    h = header(tmp_path / "nplm" / "model.safetensors")
    assert h.pop("__metadata__") == {"format": "tinyllm", "tl_arch": "nplm"}
    assert sorted(h) == sorted(m.state_dict()) and all(
        v["dtype"] == "F32" for v in h.values()
    )
    back = load_nplm(str(tmp_path / "nplm"))
    assert (back.vocab, back.context, back.d_emb, back.d_hidden, back.use_direct) == (
        9,
        3,
        2,
        5,
        direct,
    )
    x = np.array([[1, 2, 3], [8, 0, 8]])
    assert_close(back(x).data, m(x).data, rtol=0.0, atol=0.0)


# --- input validation ---------------------------------------------------------------------------


def test_input_validation(tmp_path):
    # WHY: sizes below 1, windows of the wrong width, ids outside the
    #      vocabulary, prompts shorter than the window, and too little data
    #      for one batch are caller bugs that must fail at the call, not as a
    #      silently wrapped index (numpy reads row -1) or a shape error later.
    # KIND: boundary
    # CATCHES: s01, m01
    # CHAPTER: L2.2 section 4
    for bad in ((0, 2, 3, 4), (7, 0, 3, 4), (7, 2, 0, 4), (7, 2, 3, 0)):
        with pytest.raises(ValueError):
            NPLM(*bad, rng=Rng(0))
    m = small()
    for x in ([[1, 2, 3]], [1, 2], [[1, 7]], [[-1, 2]]):
        with pytest.raises(ValueError):
            m(np.array(x))
    with pytest.raises(ValueError):
        windows([1, 2], 2)
    with pytest.raises(ValueError):
        windows([1.0, 2.0, 3.0], 1)
    with pytest.raises(ValueError):
        m.generate([1], 3, 1.0, seed=0)
    with pytest.raises(ValueError):
        m.generate([1, 2], -1, 1.0, seed=0)
    with pytest.raises(ValueError):
        m.generate([1, 2], 1, -0.5, seed=0)
    with pytest.raises(ValueError):
        train_nplm(m, np.arange(7), 1, 8, 0.1, Rng(0))
    with pytest.raises(ValueError):
        train_nplm(m, np.arange(7).repeat(4), -1, 8, 0.1, Rng(0))
    with pytest.raises(ValueError):
        save_nplm(m, str(tmp_path / "x"), tokenizer="bpe")
    (tmp_path / "y").mkdir()
    (tmp_path / "y" / "config.json").write_text(
        '{"tl_arch": "bigram", "vocab_size": 256}'
    )
    with pytest.raises(ValueError):
        load_nplm(str(tmp_path / "y"))
