"""Course tests for L4.1: an encoder-decoder with teacher forcing
(tinyllm/seq2seq/model.py).

Rung R0 for these: read them before you write code. Each test names why it
exists (WHY), what kind of check it is (KIND), the planted bugs it kills
(CATCHES, mutants in course/mutants/L4.1), and the chapter section it comes
from. Attention comes from L4.2 (AdditiveAttention) and L4.3
(LuongAttention).

The worked example of the chapter (section 3): with every GRU weight and
bias 0, a GRU step gives r = z = sigmoid(0) = 1/2 and n = tanh(0) = 0, so
h' = (1 - z) n + z h = h / 2. The encoder then outputs 0 everywhere, the
bridge gives s_0 = tanh(bridge.bias) = tanh((0.5, -0.5, 1, 0)) =
(0.462117, -0.462117, 0.761594, 0), and each decoder step halves it:
s_1 = (0.231059, -0.231059, 0.380797, 0), s_2 = s_1 / 2. With out = the
first two rows of the identity, logits_1 = (0.231059, -0.231059) and
logits_2 = (0.115529, -0.115529), whatever the tokens.

The golden fixture (course/fixtures/L4.1/seq2seq_torch.npz) holds torch
2.14.1 teacher-forced logits and gradients for four configurations with
copied weights (course/oracle/L4.1/seq2seq_torch.py). The learning test
trains the reversal task and compares exact match with
course/fixtures/ref-thresholds.tsv.
"""

from __future__ import annotations

import functools
import json
import os

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32
from _lib.thresholds import check
from tinyllm.autograd import functional as F
from tinyllm.autograd.losses import cross_entropy
from tinyllm.autograd.tensor import Tensor
from tinyllm.optim.adamw import AdamW
from tinyllm.seq2seq.additive import AdditiveAttention
from tinyllm.seq2seq.luong import LuongAttention
from tinyllm.seq2seq.model import Seq2Seq, load_seq2seq, save_seq2seq

FIX = os.path.join(os.environ.get("TINYLLM_FIXTURES", ""), "L4.1", "seq2seq_torch.npz")
PAD, BOS, EOS = 0, 1, 2


def seed() -> int:
    return int(os.environ.get("OL_SEED", "0"))


class Rng:
    """The frozen PCG32 behind the generator API of M06.3; counts draws."""

    def __init__(self, s: int) -> None:
        self.g = PCG32(seed=s)
        self.draws = 0

    def next_u32(self) -> int:
        return self.g.next_u32()

    def uniform(self) -> float:
        self.draws += 1
        return self.g.uniform()

    def uniforms(self, n: int) -> np.ndarray:
        return np.array([self.g.uniform() for _ in range(n)], dtype=np.float64)

    def below(self, n: int) -> int:
        return self.g.below(n)


def build(
    attn: str, cell: str = "gru", s: int = 0, Vs=7, Vt=8, E=4, H=6, A=5
) -> Seq2Seq:
    att = None
    if attn == "bahdanau":
        att = AdditiveAttention(H, H, A, rng=Rng(s + 1000))
    elif attn.startswith("luong-"):
        att = LuongAttention(H, attn[len("luong-") :], rng=Rng(s + 1000))
    return Seq2Seq(Vs, Vt, E, H, cell=cell, attention=att, rng=Rng(s))


def batch(s: int = 3):
    g = PCG32(seed=s)
    src = np.array([[g.below(7) for _ in range(5)] for _ in range(3)], dtype=np.int64)
    tgt = np.array([[g.below(8) for _ in range(4)] for _ in range(3)], dtype=np.int64)
    return src, np.array([5, 2, 4]), tgt


# --- the worked example ---------------------------------------------------------------


def test_hand_example_zero_weights():
    # WHY: the chapter's worked example: with all-zero GRU weights every step
    #      halves the state, so the decoder's logits are the bridge's tanh
    #      halved once per step, whatever the source and target tokens. It
    #      pins the data path: encoder -> bridge -> decoder cell -> out.
    # KIND: unit
    # CATCHES: s01, m01
    # CHAPTER: L4.1 section 3, Worked example by hand
    m = Seq2Seq(5, 2, 3, 4, rng=Rng(0))
    sd = {n: np.zeros_like(v) for n, v in m.state_dict().items()}
    sd["bridge.bias"] = np.array([0.5, -0.5, 1.0, 0.0])
    sd["out.weight"] = np.eye(2, 4)
    sd["src_emb.weight"] = np.arange(15.0).reshape(5, 3)
    m.load_state_dict(sd)
    logits = m(np.array([[1, 2, 3]]), np.array([3]), np.array([[0, 1]]))
    s0 = np.tanh([0.5, -0.5])
    assert logits.shape == (1, 2, 2)
    assert_close(logits.data[0], [s0 / 2, s0 / 4], dtype="float32")


# --- against torch ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case", ["none.gru", "bahdanau.gru", "luong-concat.gru", "luong-dot.lstm"]
)
def test_golden_torch(case):
    # WHY: the same encoder-decoder built from torch modules (a bidirectional
    #      nn.GRU over packed sequences, nn.GRUCell or nn.LSTMCell) with the
    #      same weights gives the same teacher-forced logits and gradients of
    #      sum(logits * g) for every parameter. Lengths 5, 2, 4.
    # KIND: golden
    # CATCHES: s01, s02, s03, s04, s05, s06, s07, s09, m01, m02
    # CHAPTER: L4.1 section 2.2, The model
    f = np.load(FIX)
    attn, cell = case.split(".")
    m = build(attn, cell)
    pre = f"{case}.param."
    m.load_state_dict({n[len(pre) :]: f[n] for n in f.files if n.startswith(pre)})
    logits = m(f[f"{case}.src"], f[f"{case}.lens"], f[f"{case}.tgt"])
    assert_close(logits.data, f[f"{case}.logits"], rtol=1e-4, atol=1e-5)
    F.sum(logits * Tensor(f[f"{case}.g"])).backward()
    for name, p in m.named_parameters():
        assert_close(p.grad, f[f"{case}.grad.{name}"], rtol=1e-3, atol=1e-5, msg=name)


# --- teacher forcing and the step -------------------------------------------------------------


@pytest.mark.parametrize("attn", ["none", "bahdanau", "luong-general"])
def test_forward_is_the_step_loop(attn):
    # WHY: teacher forcing means step t is fed the TRUE token tgt_in[:, t],
    #      so forward is exactly encode, init_state, and decode_step over the
    #      target tokens, stacked on axis 1. This is the interface beam search
    #      (L4.4) and the zoo drive one step at a time.
    # KIND: differential
    # CATCHES: s02
    # CHAPTER: L4.1 section 2.3, Teacher forcing
    m = build(attn, s=5)
    src, lens, tgt = batch(6)
    whole = m(src, lens, tgt).data
    st = m.init_state(m.encode(src, lens))
    for t in range(tgt.shape[1]):
        logits, st, a = m.decode_step(tgt[:, t], st)
        assert_close(logits.data, whole[:, t], rtol=1e-6, atol=1e-6)
        assert (a is None) == (attn == "none")


def test_teacher_forcing_ratio():
    # WHY: with teacher_forcing < 1 (scheduled sampling) each step after the
    #      first draws ONE uniform for the whole batch and, when it is at
    #      least the ratio, feeds the model's own argmax from the step before.
    #      Ratio 0 is free running: the inputs are the model's guesses, as at
    #      inference. Ratio 1 draws nothing.
    # KIND: unit
    # CATCHES: m03, m04
    # CHAPTER: L4.1 section 2.3, Teacher forcing
    m = build("bahdanau", s=7)
    src, lens, tgt = batch(8)
    r = Rng(9)
    free = m(src, lens, tgt, teacher_forcing=0.0, rng=r).data
    assert r.draws == tgt.shape[1] - 1
    st = m.init_state(m.encode(src, lens))
    y = tgt[:, 0]
    for t in range(tgt.shape[1]):
        logits, st, _ = m.decode_step(y, st)
        assert_close(logits.data, free[:, t], rtol=1e-6, atol=1e-6)
        y = np.argmax(logits.data, axis=-1)
    r1 = Rng(9)
    m(src, lens, tgt, teacher_forcing=1.0, rng=r1)
    assert r1.draws == 0
    with pytest.raises(ValueError):
        m(src, lens, tgt, teacher_forcing=0.5)


def test_bahdanau_reads_with_the_previous_state():
    # WHY: Bahdanau attention scores the source with the decoder state from
    #      BEFORE the step (s_{t-1}), and the context goes INTO the cell. At
    #      the first step that query is s_0, the bridge's output.
    # KIND: unit
    # CATCHES: s09
    # CHAPTER: L4.1 section 2.4, Where attention plugs in
    seen = []

    class Spy(AdditiveAttention):
        def forward(self, query, keys, mask, proj=None):
            seen.append(query.data.copy())
            return super().forward(query, keys, mask, proj)

    m = Seq2Seq(7, 8, 4, 6, attention=Spy(6, 6, 5, rng=Rng(1)), rng=Rng(2))
    src, lens, tgt = batch(10)
    st = m.init_state(m.encode(src, lens))
    h0 = st.h.data.copy()
    _, st1, _ = m.decode_step(tgt[:, 0], st)
    m.decode_step(tgt[:, 1], st1)
    assert np.array_equal(seen[0], h0) and np.array_equal(seen[1], st1.h.data)


def test_luong_reads_with_the_new_state_and_feeds_input():
    # WHY: Luong attention runs the cell first and scores with the NEW state
    #      h_t; its attentional state h~_t is fed into the next step's input
    #      (input feeding), zero before the first step.
    # KIND: unit
    # CATCHES: s04, s05
    # CHAPTER: L4.1 section 2.4, Where attention plugs in
    seen = []

    class Spy(LuongAttention):
        def forward(self, query, keys, mask, proj=None):
            seen.append(query.data.copy())
            return super().forward(query, keys, mask, proj)

    m = Seq2Seq(7, 8, 4, 6, attention=Spy(6, "general", rng=Rng(1)), rng=Rng(2))
    src, lens, tgt = batch(11)
    st = m.init_state(m.encode(src, lens))
    assert np.all(st.feed.data == 0.0)
    logits, st1, _ = m.decode_step(tgt[:, 0], st)
    assert np.array_equal(seen[0], st1.h.data)
    ctx, _ = m.attention(st1.h, st1.keys, st1.mask)
    assert_close(
        st1.feed.data, m.attention.attentional(st1.h, ctx).data, rtol=1e-6, atol=1e-7
    )
    assert_close(logits.data, m.out(st1.feed).data, rtol=1e-6, atol=1e-7)
    # The cell's input at step 2 is [embedding ; feed]: recompute it.
    x = F.concat([m.tgt_emb(tgt[:, 1]), st1.feed], axis=-1)
    _, st2, _ = m.decode_step(tgt[:, 1], st1)
    assert_close(st2.h.data, m.cell(x, st1.h).data, rtol=1e-6, atol=1e-7)


# --- the encoder ---------------------------------------------------------------------------------


@pytest.mark.parametrize("attn", ["none", "bahdanau"])
def test_padding_never_changes_a_sentence(attn):
    # WHY: a sentence's logits must not depend on how it was padded or on what
    #      the padding holds: the bidirectional encoder reverses inside each
    #      length (L3.4), the bridge reads each row's own last state, and
    #      attention never reads padding. Row 1 (length 2) alone, padded to 2,
    #      equals row 1 of the batch padded to 5 with garbage after it.
    # KIND: property
    # CATCHES: s06, m05
    # CHAPTER: L4.1 section 5, Pitfalls, item 3
    m = build(attn, s=12)
    src, lens, tgt = batch(13)
    src2 = src.copy()
    src2[1, 2:] = 99  # outside the vocabulary: padding is never looked up
    a = m(src, lens, tgt).data[1]
    b = m(src2, lens, tgt).data[1]
    c = m(src[1:2, :2], np.array([2]), tgt[1:2]).data[0]
    assert np.array_equal(a, b)
    assert_close(c, a, rtol=1e-5, atol=1e-6)


def test_gradient_reaches_the_encoder():
    # WHY: the decoder's loss must train the encoder too: through the bridge
    #      always, and through the keys when there is attention. Every
    #      parameter of the model gets a nonzero gradient from one batch.
    # KIND: unit
    # CATCHES: s01, s07, s09
    # CHAPTER: L4.1 section 2.2, The model
    for attn in ("none", "bahdanau", "luong-concat"):
        m = build(attn, s=14)
        src, lens, tgt = batch(15)
        logits = m(src, lens, tgt)
        cross_entropy(
            F.reshape(logits, (-1, 8)), np.roll(tgt, -1, axis=1).reshape(-1)
        ).backward()
        for name, p in m.named_parameters():
            assert p.grad is not None and np.abs(p.grad).sum() > 0, (
                f"{attn}: {name} gets no gradient"
            )


# --- decoding, saving, the interface -----------------------------------------------------------


def test_greedy_matches_step_loop_and_stops_at_eos():
    # WHY: greedy decoding feeds back its own argmax and stops a row at its
    #      first eos (kept) or at max_len; rows finish independently.
    # KIND: unit
    # CATCHES: m06
    # CHAPTER: L4.1 section 4, The interface
    m = build("luong-dot", s=16)
    src, lens, _ = batch(17)
    out = m.greedy(src, lens, BOS, EOS, max_len=6)
    st = m.init_state(m.encode(src, lens))
    y = np.full(3, BOS)
    rows = [[] for _ in range(3)]
    for _ in range(6):
        logits, st, _ = m.decode_step(y, st)
        y = np.argmax(logits.data, axis=-1)
        for b in range(3):
            if not rows[b] or rows[b][-1] != EOS:
                rows[b].append(int(y[b]))
    assert out == rows
    assert all(len(r) <= 6 and EOS not in r[:-1] for r in out)


@pytest.mark.parametrize(
    "attn", ["none", "bahdanau", "luong-dot", "luong-general", "luong-concat"]
)
def test_save_load_roundtrip(attn, tmp_path):
    # WHY: the zoo (L6.7) reloads a seq2seq checkpoint, attention included,
    #      from config.json and model.safetensors; the logits must be
    #      identical and the keys the registration order of the contract.
    # KIND: unit
    # CATCHES: m07
    # CHAPTER: L4.1 section 4, The interface
    m = build(attn, cell="lstm" if attn == "luong-dot" else "gru", s=18)
    save_seq2seq(m, str(tmp_path))
    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg["tl_arch"] == "seq2seq" and cfg["tl_seq2seq_attention"] == attn
    assert (
        cfg["vocab_size"] == 8 and cfg["tl_src_vocab"] == 7 and cfg["hidden_size"] == 6
    )
    back = load_seq2seq(str(tmp_path))
    keys = list(m.state_dict())
    assert list(back.state_dict()) == keys
    assert keys[0] == "src_emb.weight" and keys[-2:] == ["out.weight", "out.bias"]
    src, lens, tgt = batch(19)
    assert np.array_equal(back(src, lens, tgt).data, m(src, lens, tgt).data)


def test_validation():
    # WHY: an odd d_h cannot split into two directions, lengths outside
    #      1..S and ids outside the vocabulary are caller bugs, and an object
    #      without query_from is not an attention module.
    # KIND: boundary
    # CATCHES: m08
    # CHAPTER: L4.1 section 4, The interface
    with pytest.raises(ValueError):
        Seq2Seq(7, 8, 4, 5)
    with pytest.raises(ValueError):
        Seq2Seq(7, 8, 4, 6, cell="rnn")
    with pytest.raises(ValueError):
        Seq2Seq(7, 8, 4, 6, attention=object())
    m = build("none", s=20)
    src, lens, tgt = batch(21)
    for bad in (
        (src, np.array([6, 2, 4]), tgt),
        (src, np.array([0, 2, 4]), tgt),
        (src, lens, tgt + 8),
    ):
        with pytest.raises(ValueError):
            m(*bad)
    with pytest.raises(ValueError):
        m.greedy(src, lens, BOS, EOS, max_len=0)


# --- learning ---------------------------------------------------------------------------------------


def reversal(n: int, g: PCG32):
    """Reverse a sequence of 3 to 6 digits; digits are ids 3..10."""
    srcs, tgts = [], []
    for _ in range(n):
        L = 3 + g.below(4)
        d = [3 + g.below(8) for _ in range(L)]
        srcs.append(d)
        tgts.append(d[::-1])
    return srcs, tgts


def pad(rows, width):
    out = np.zeros((len(rows), width), dtype=np.int64)
    for i, r in enumerate(rows):
        out[i, : len(r)] = r
    return out


@functools.lru_cache(maxsize=None)
def train_reversal(attention: bool, steps: int, s: int) -> float:
    """Train a GRU encoder-decoder (Bahdanau attention or none) on the
    reversal task with teacher forcing (AdamW, padding ignored); return the
    greedy exact match on 100 held-out strings."""
    g = PCG32(seed=200 + s)
    att = AdditiveAttention(32, 32, 32, rng=Rng(300 + s)) if attention else None
    m = Seq2Seq(11, 11, 16, 32, attention=att, rng=Rng(400 + s))
    opt = AdamW(m.parameters(), lr=0.01, weight_decay=0.0)
    for _ in range(steps):
        srcs, tgts = reversal(32, g)
        src = pad(srcs, 6)
        lens = np.array([len(x) for x in srcs])
        tgt_in = pad([[BOS] + t for t in tgts], 7)
        tgt_out = pad([t + [EOS] for t in tgts], 7)
        tgt_out[tgt_in == PAD] = -100  # padding positions predict nothing
        opt.zero_grad()
        logits = m(src, lens, tgt_in)
        cross_entropy(F.reshape(logits, (-1, 11)), tgt_out.reshape(-1)).backward()
        opt.step()
    srcs, tgts = reversal(100, PCG32(seed=999))
    out = m.greedy(pad(srcs, 6), np.array([len(x) for x in srcs]), BOS, EOS, max_len=8)
    return float(np.mean([o == t + [EOS] for o, t in zip(out, tgts)]))


def test_attention_learns_to_reverse():
    # WHY: the whole module end to end: a GRU encoder-decoder with Bahdanau
    #      attention, trained with teacher forcing for 150 steps, learns to
    #      reverse digit strings. Exact match by greedy decoding on 100
    #      held-out strings must reach the reference within 3 sd. Fixed seed
    #      and step count.
    # KIND: learning
    # CATCHES: s02, m06
    # CHAPTER: L4.1 section 2.3, Teacher forcing
    em = train_reversal(True, 150, seed())
    assert em > 0.5, f"exact match {em:.2f}"
    check("L4.1/test_attention_learns_to_reverse", "exact_match", em, direction="min")


def test_attention_beats_the_bottleneck():
    # WHY: the lesson of this part. Without attention the decoder sees the
    #      source only through s_0, one fixed-size vector, and after the same
    #      150 steps reverses far fewer strings (about 0.4 against 0.94 for
    #      the reference). The margin must be at least 0.2.
    # KIND: learning
    # CATCHES: s02, s09, m06
    # CHAPTER: L4.1 section 1, Why now
    with_att = train_reversal(True, 150, seed())
    without = train_reversal(False, 150, seed())
    assert with_att - without >= 0.2, f"attention {with_att:.2f} vs none {without:.2f}"
