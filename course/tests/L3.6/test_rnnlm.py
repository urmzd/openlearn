"""Course tests for L3.6: an RNN language model with stateful truncated BPTT
(tinyllm/rnn/rnnlm.py).

Rung R0 for these: read them before you write code. Each test names why it
exists (WHY), what kind of check it is (KIND), the planted bugs it kills
(CATCHES, mutants in course/mutants/L3.6), and the chapter section it comes
from.

The worked example of the chapter (section 3): the stream 0, 1, ..., 19 with
batch 2 and k = 3 has two lanes of L = 10 tokens, 0..9 and 10..19. Windows
start at p = 0, 3, 6 (p + k + 1 <= 10): inputs [[0 1 2] [10 11 12]], targets
[[1 2 3] [11 12 13]], then [[3 4 5] [13 14 15]] -> [[4 5 6] [14 15 16]],
then [[6 7 8] [16 17 18]] -> [[7 8 9] [17 18 19]]. Token 9 and 19 are only
targets; nothing is dropped from the middle.

The learning test trains a character LSTM on course/fixtures/L3.6/corpus.txt
(an original synthetic text, course/oracle/L3.6/corpus.py) and compares the
held-out bits per character with a bar in course/fixtures/ref-thresholds.tsv
(the reference's mean + 3 sd over 5 seeds).
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
from tinyllm.autograd.losses import cross_entropy
from tinyllm.autograd.tensor import Tensor
from tinyllm.optim.adamw import AdamW
from tinyllm.rnn.manual import rnn_backward, rnn_forward
from tinyllm.rnn.rnnlm import (
    RNNLM,
    ElmanRNN,
    load_rnnlm,
    save_rnnlm,
    tbptt_batches,
    train_tbptt,
)
from tinyllm.tok.char import CharTokenizer

CORPUS = os.path.join(os.environ.get("TINYLLM_FIXTURES", ""), "L3.6", "corpus.txt")
CELLS = ("rnn", "lstm", "gru")


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


class Capture:
    """An optimizer (the M10.2 protocol) that changes nothing and records a
    copy of every parameter's gradient at each step."""

    def __init__(self, params) -> None:
        self.params = list(params)
        self.grads: list[list[np.ndarray]] = []

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def step(self) -> None:
        self.grads.append(
            [None if p.grad is None else p.grad.copy() for p in self.params]
        )

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, sd: dict) -> None:
        pass


def tiny(cell: str, s: int = 0, n_layers: int = 1) -> RNNLM:
    return RNNLM(11, 5, 6, cell, n_layers=n_layers, rng=Rng(s))


def stream(n: int, V: int = 11, s: int = 1) -> np.ndarray:
    g = PCG32(seed=s)
    return np.array([g.below(V) for _ in range(n)], dtype=np.int64)


# --- the worked example ---------------------------------------------------------------


def test_hand_example_windows():
    # WHY: the chapter's worked example: two contiguous lanes, windows at
    #      p = 0, 3, 6, targets shifted by one inside the lane. Contiguous
    #      lanes are what make carrying the state from one window to the next
    #      meaningful: row b of window j + 1 continues row b of window j.
    # KIND: unit
    # CATCHES: s01
    # CHAPTER: L3.6 section 3, Worked example by hand
    w = tbptt_batches(np.arange(20), k=3, batch=2)
    assert len(w) == 3
    want = [
        ([[0, 1, 2], [10, 11, 12]], [[1, 2, 3], [11, 12, 13]]),
        ([[3, 4, 5], [13, 14, 15]], [[4, 5, 6], [14, 15, 16]]),
        ([[6, 7, 8], [16, 17, 18]], [[7, 8, 9], [17, 18, 19]]),
    ]
    for (x, y), (wx, wy) in zip(w, want):
        assert x.tolist() == wx and y.tolist() == wy
        assert x.dtype == np.int64


def test_windows_drop_the_tail_and_validate():
    # WHY: 21 tokens in 2 lanes is 10 each (the last token is dropped), and
    #      with k = 5 only p = 0 fits (p = 5 would need token 10 of a 10-token
    #      lane); a lane shorter than k + 1 cannot form one window, which is an
    #      error, not an empty epoch that trains forever on nothing.
    # KIND: boundary
    # CATCHES: s01
    # CHAPTER: L3.6 section 2.3, Lanes and windows
    w = tbptt_batches(np.arange(21), k=5, batch=2)
    assert [x.tolist() for x, _ in w] == [[[0, 1, 2, 3, 4], [10, 11, 12, 13, 14]]]
    assert w[0][1].tolist() == [[1, 2, 3, 4, 5], [11, 12, 13, 14, 15]]
    for bad in (
        (np.arange(7), 3, 2),
        (np.arange(20), 0, 2),
        (np.arange(20), 3, 0),
        (np.zeros((4, 4), int), 1, 1),
    ):
        with pytest.raises(ValueError):
            tbptt_batches(*bad)


# --- the fused Elman op ------------------------------------------------------------------------


def test_elman_is_one_fused_op_over_l3_1():
    # WHY: cell = "rnn" runs L3.1's hand-written forward and backward as one
    #      autograd op. Its output and every gradient must equal calling
    #      rnn_forward and rnn_backward directly with Wxh = weight_ih^T,
    #      Whh = weight_hh^T, bh = bias_ih + bias_hh (each bias getting dbh).
    # KIND: differential
    # CATCHES: s03, s04
    # CHAPTER: L3.6 section 2.1, Three cells, one interface
    rnn = ElmanRNN(3, 4, rng=Rng(2))
    for p in rnn.parameters():
        p.data = p.data.astype(np.float64)
    g = PCG32(seed=3)
    x0, h0 = g.normal_array((5, 2, 3)), g.normal_array((1, 2, 4))
    up = g.normal_array((5, 2, 4))
    x = Tensor(x0, requires_grad=True, dtype=np.float64)
    hs = Tensor(h0, requires_grad=True, dtype=np.float64)
    out, h_n = rnn(x, hs)
    w_ih, w_hh, b_ih, b_hh = (p.data for p in rnn.parameters())
    h, cache = rnn_forward(x0, h0[0], w_ih.T, w_hh.T, b_ih + b_hh)
    assert_close(out.data, h, dtype="float64")
    assert_close(h_n.data[0], h[-1], dtype="float64")
    F.sum(out * up).backward()
    d = rnn_backward(up, cache)
    assert_close(x.grad, d["x"], dtype="float64")
    assert_close(hs.grad[0], d["h0"], dtype="float64")
    assert_close(rnn.weight_ih_l0.grad, d["Wxh"].T, dtype="float64")
    assert_close(rnn.weight_hh_l0.grad, d["Whh"].T, dtype="float64")
    assert_close(rnn.bias_ih_l0.grad, d["bh"], dtype="float64")
    assert_close(rnn.bias_hh_l0.grad, d["bh"], dtype="float64")


@pytest.mark.parametrize("cell", ["rnn", "lstm"])
def test_gradcheck_rnnlm(cell):
    # WHY: backward reaches every parameter of the model, two stacked layers
    #      included, through the fused Elman op and an LSTM (L3.3's GRU has its
    #      own gradcheck); checked against the frozen central differences in
    #      float64 on a 2 x 4 batch.
    # KIND: gradcheck
    # CATCHES: s03, s04, m03
    # CHAPTER: L3.6 section 2.1, Three cells, one interface
    m = RNNLM(5, 3, 3, cell, n_layers=2, rng=Rng(4))
    params = [p for _, p in m.named_parameters()]
    for p in params:
        p.data = p.data.astype(np.float64)
    ids = np.array([[0, 3, 1, 4], [2, 2, 0, 1]])
    tgt = np.array([[3, 1, 4, 0], [2, 0, 1, 3]])

    def f(*arrays):
        for p, a in zip(params, arrays):
            p.data = a
        logits, _ = m(ids)
        return float(cross_entropy(F.reshape(logits, (-1, 5)), tgt.reshape(-1)).data)

    start = [p.data.copy() for p in params]
    m.zero_grad()
    logits, _ = m(ids)
    cross_entropy(F.reshape(logits, (-1, 5)), tgt.reshape(-1)).backward()
    gradcheck(
        f,
        start,
        [p.grad.copy() for p in params],
        names=[n for n, _ in m.named_parameters()],
    )


# --- statefulness ---------------------------------------------------------------------------------


@pytest.mark.parametrize("cell", CELLS)
def test_state_carries_across_calls(cell):
    # WHY: reading a sequence in two pieces, passing the state from the first
    #      call into the second, must give the logits of one call over the
    #      whole sequence. That is what "stateful" means, and what lets a
    #      window see context from before it.
    # KIND: property
    # CATCHES: s05, m03
    # CHAPTER: L3.6 section 2.2, Stateful training
    m = tiny(cell, 5, n_layers=2)
    ids = stream(18, s=6).reshape(2, 9)
    whole, _ = m(ids)
    a, st = m(ids[:, :4])
    b, _ = m(ids[:, 4:], st)
    assert whole.shape == (2, 9, 11)
    assert_close(
        np.concatenate([a.data, b.data], axis=1), whole.data, rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize("cell", ["lstm", "gru"])
def test_gradient_is_cut_at_the_window_boundary(cell):
    # WHY: the state entering window 2 is a value, not a path: its gradient
    #      must stop there. The gradients train_tbptt hands the optimizer at
    #      step 2 must equal those of window 2 alone, started from the state
    #      window 1 left (as a constant). Without the detach, backward runs
    #      on into window 1's graph and adds its terms again.
    # KIND: differential
    # CATCHES: s06, s07, m03
    # CHAPTER: L3.6 section 5, Pitfalls, item 1
    data = stream(40, s=7)
    m = tiny(cell, 8)
    opt = Capture(m.parameters())
    train_tbptt(m, data, k=4, batch=2, opt=opt, clip=None, steps=2)
    (x0, _), (x1, y1) = tbptt_batches(data, 4, 2)[:2]
    _, st = m(x0)
    st = tuple(s.data for s in st) if isinstance(st, tuple) else st.data
    m.zero_grad()
    logits, _ = m(x1, st)
    cross_entropy(F.reshape(logits, (-1, 11)), y1.reshape(-1)).backward()
    for p, g in zip(m.parameters(), opt.grads[1]):
        assert_close(g, p.grad, rtol=1e-5, atol=1e-6)


def test_state_flows_into_the_next_window():
    # WHY: the opposite mistake: detaching by resetting the state to zeros at
    #      every window throws away the context the lanes were built to
    #      carry. Step 2's gradients must differ from those of window 2 run
    #      from zeros, and equal those from window 1's state.
    # KIND: differential
    # CATCHES: s05, s07
    # CHAPTER: L3.6 section 5, Pitfalls, item 2
    data = stream(40, s=9)
    m = tiny("gru", 10)
    opt = Capture(m.parameters())
    losses = train_tbptt(m, data, k=4, batch=2, opt=opt, clip=None, steps=2)
    (x0, _), (x1, y1) = tbptt_batches(data, 4, 2)[:2]
    _, st = m(x0)
    for start, same in ((st.data, True), (None, False)):
        m.zero_grad()
        logits, _ = m(x1, start)
        loss = cross_entropy(F.reshape(logits, (-1, 11)), y1.reshape(-1))
        assert (abs(float(loss.data) - losses[1]) < 1e-6) == same


def test_epoch_wraps_and_resets_the_state():
    # WHY: after the last window the lanes start over at p = 0, and so must
    #      the state: carrying the end of lane b into its own beginning
    #      conditions the first window on text that never precedes it. With
    #      an optimizer that changes nothing, step len(windows) repeats step 0
    #      exactly.
    # KIND: unit
    # CATCHES: s08, m03, m04
    # CHAPTER: L3.6 section 2.3, Lanes and windows
    data = stream(30, s=11)
    m = tiny("lstm", 12)
    n = len(tbptt_batches(data, 4, 3))
    losses = train_tbptt(
        m, data, k=4, batch=3, opt=Capture(m.parameters()), clip=None, steps=n + 2
    )
    assert len(losses) == n + 2
    assert losses[n] == losses[0] and losses[n + 1] == losses[1]
    assert losses[1] != losses[0]


def test_clip_is_applied():
    # WHY: exploding gradients are the RNN failure mode (L3.1): train_tbptt
    #      clips the global norm before each step (M10.4) when clip is set.
    #      The gradients the optimizer sees have norm at most clip.
    # KIND: unit
    # CATCHES: s09
    # CHAPTER: L3.6 section 2.4, Clipping
    data = stream(40, s=13)
    m = tiny("rnn", 14)
    opt = Capture(m.parameters())
    train_tbptt(m, data, k=5, batch=2, opt=opt, clip=0.01, steps=3)
    for grads in opt.grads:
        assert math.sqrt(
            sum(float((g.astype(np.float64) ** 2).sum()) for g in grads)
        ) <= 0.01 * (1 + 1e-5)


# --- evaluation and sampling -----------------------------------------------------------------------


@pytest.mark.parametrize("cell", CELLS)
def test_nll_does_not_depend_on_the_chunk(cell):
    # WHY: nll runs a long stream in chunks, carrying the state, so the cost
    #      of evaluation is bounded; the numbers must be those of one pass, and
    #      the first token has no prediction. chunk = 1, 7, and 256 agree.
    # KIND: property
    # CATCHES: s05, s10, m03
    # CHAPTER: L3.6 section 4, The interface
    m = tiny(cell, 15)
    ids = stream(30, s=16)
    full = m.nll(ids, chunk=256)
    assert full.shape == (29,) and full.dtype == np.float64
    logits, _ = m(ids[None, :-1])
    z = logits.data[0].astype(np.float64)
    ls = (
        z
        - np.log(np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True))
        - z.max(1, keepdims=True)
    )
    assert_close(full, -ls[np.arange(29), ids[1:]], rtol=1e-6, atol=1e-6)
    for c in (1, 7):
        assert_close(m.nll(ids, chunk=c), full, rtol=1e-5, atol=1e-6)


def test_generate_is_seeded_and_greedy_is_argmax():
    # WHY: sampling uses one PCG32(seed) per call (the cross-language RNG), so
    #      a seed fixes the text; temperature 0 is the argmax at each step,
    #      fed back as the next input.
    # KIND: unit
    # CATCHES: s05, s11, m03
    # CHAPTER: L3.6 section 4, The interface
    m = tiny("lstm", 17)
    a = m.generate([1, 2, 3], 12, temperature=1.0, seed=5)
    assert a == m.generate([1, 2, 3], 12, temperature=1.0, seed=5)
    assert len(a) == 12 and all(0 <= t < 11 for t in a)
    greedy = m.generate([1, 2, 3], 6, temperature=0.0, seed=0)
    hist = [1, 2, 3]
    for _ in range(6):
        logits, _ = m(np.array([hist]))
        hist.append(int(np.argmax(logits.data[0, -1])))
    assert greedy == hist[3:]
    for bad in (([], 3, 1.0), ([1], -1, 1.0), ([1], 3, -0.5)):
        with pytest.raises(ValueError):
            m.generate(*bad, seed=0)


@pytest.mark.parametrize("cell", CELLS)
def test_save_load_roundtrip(cell, tmp_path):
    # WHY: the zoo (L6.7) loads every rnnlm checkpoint through config.json and
    #      model.safetensors; the reloaded model must give the same logits,
    #      and the keys are the torch names of the cell.
    # KIND: unit
    # CATCHES: s12, m03
    # CHAPTER: L3.6 section 4, The interface
    m = tiny(cell, 18, n_layers=2)
    save_rnnlm(m, str(tmp_path), tokenizer="file")
    import json

    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg == {
        "tl_arch": "rnnlm",
        "tl_tokenizer": "file",
        "vocab_size": 11,
        "tl_format": 1,
        "tl_cell": cell,
        "tl_d_emb": 5,
        "hidden_size": 6,
        "num_hidden_layers": 2,
    }
    back = load_rnnlm(str(tmp_path))
    assert list(back.state_dict()) == list(m.state_dict())
    assert (
        "rnn.weight_ih_l1" in m.state_dict() and list(m.state_dict())[0] == "emb.weight"
    )
    ids = stream(10, s=19).reshape(2, 5)
    assert np.array_equal(back(ids)[0].data, m(ids)[0].data)
    with pytest.raises(ValueError):
        save_rnnlm(m, str(tmp_path), tokenizer="bpe")


def test_validation():
    # WHY: an unknown cell, ids outside the vocabulary, or a 1-D batch are
    #      caller bugs; numpy would silently wrap a negative id.
    # KIND: boundary
    # CATCHES: m06
    # CHAPTER: L3.6 section 4, The interface
    with pytest.raises(ValueError):
        RNNLM(11, 5, 6, "transformer")
    m = tiny("gru", 20)
    for bad in (np.array([[1, 11]]), np.array([[-1, 2]]), np.array([1, 2])):
        with pytest.raises(ValueError):
            m(bad)
    with pytest.raises(ValueError):
        ElmanRNN(3, 4)(Tensor(np.zeros((2, 1, 3))), lengths=np.array([1]))
    with pytest.raises(ValueError):
        train_tbptt(
            m,
            stream(40),
            k=4,
            batch=2,
            opt=Capture(m.parameters()),
            clip=None,
            steps=-1,
        )


# --- learning ---------------------------------------------------------------------------------------


def test_lstm_learns_characters():
    # WHY: the whole module end to end: a character LSTM trained by stateful
    #      TBPTT (AdamW, clip 1, 150 steps) on 90% of the corpus must reach the reference
    #      held-out bits per character within 3 sd, and beat the unigram
    #      entropy of the training text by a wide margin (it learned more than
    #      letter frequencies). Fixed seed and step count.
    # KIND: learning
    # CATCHES: s01, m03
    # CHAPTER: L3.6 section 2.2, Stateful training
    text = open(CORPUS, encoding="utf-8").read()
    tok = CharTokenizer.train([text])
    ids = np.array(tok.encode(text), dtype=np.int64)
    cut = int(0.9 * len(ids))
    train, val = ids[:cut], ids[cut:]
    m = RNNLM(tok.vocab_size, 16, 64, "lstm", rng=Rng(100 + seed()))
    opt = AdamW(m.parameters(), lr=0.01, weight_decay=0.0)
    train_tbptt(m, train, k=16, batch=8, opt=opt, clip=1.0, steps=150)
    bpc = float(m.nll(val[:2000]).mean() / math.log(2))
    counts = np.bincount(train, minlength=tok.vocab_size) / len(train)
    unigram = float(-(counts[counts > 0] * np.log2(counts[counts > 0])).sum())
    assert bpc < unigram - 0.5, f"val bpc {bpc:.3f} vs unigram entropy {unigram:.3f}"
    check("L3.6/test_lstm_learns_characters", "val_bpc", bpc, direction="max")
