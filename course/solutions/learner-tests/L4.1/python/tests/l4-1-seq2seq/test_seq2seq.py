"""My tests for L4.1 (rung R5: oracles. The oracles here are the model's
own pieces recombined by hand: the encoder's two GRUs run one sequence at a
time, the decoder step recomputed from its cell and layers). They import
only the contract."""

import json

import numpy as np
import pytest
import tinyllm.autograd.functional as F
from tinyllm.seq2seq.additive import AdditiveAttention
from tinyllm.seq2seq.luong import LuongAttention
from tinyllm.seq2seq.model import Seq2Seq, load_seq2seq, save_seq2seq


class R:
    def __init__(self, s):
        self.g = np.random.Generator(np.random.PCG64(s))
        self.draws = 0

    def uniform(self):
        self.draws += 1
        return float(self.g.random())

    def next_u32(self):
        return int(self.g.integers(0, 2**32))


SRC = np.array([[1, 2, 3, 4, 5], [6, 1, 0, 0, 0], [2, 2, 3, 1, 0]])
LENS = np.array([5, 2, 4])
TGT = np.array([[1, 2, 3, 4], [5, 6, 7, 0], [3, 3, 1, 2]])


def model(attn=None, cell="gru"):
    a = {
        "bahdanau": AdditiveAttention(6, 6, 5),
        "luong": LuongAttention(6, "general"),
    }.get(attn)
    return Seq2Seq(7, 8, 4, 6, cell=cell, attention=a)


def test_hand_example_zero_weights():
    """All-zero GRUs halve the state each step: logits tanh(bias)/2, /4."""
    m = Seq2Seq(5, 2, 3, 4)
    sd = {n: np.zeros_like(v) for n, v in m.state_dict().items()}
    sd["bridge.bias"] = np.array([0.5, -0.5, 1.0, 0.0])
    sd["out.weight"] = np.eye(2, 4)
    m.load_state_dict(sd)
    out = m(np.array([[1, 2, 3]]), np.array([3]), np.array([[0, 1]])).data[0]
    s0 = np.tanh([0.5, -0.5])
    np.testing.assert_allclose(out, [s0 / 2, s0 / 4], rtol=1e-6)


def test_bridge_reads_each_rows_own_final_states():
    m = model()
    enc = m.encode(SRC, LENS)
    for b, n in enumerate(LENS):
        x = m.src_emb(SRC[b : b + 1, :n].T)
        f, _ = m.enc_fwd(x)
        r, _ = m.enc_bwd(m.src_emb(SRC[b : b + 1, :n][:, ::-1].T.copy()))
        want = np.tanh(m.bridge(F.concat([f[n - 1], r[n - 1]], axis=-1)).data)
        np.testing.assert_allclose(enc.init.data[b], want[0], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("attn", [None, "bahdanau", "luong"])
def test_forward_is_teacher_forced_steps(attn):
    m = model(attn)
    whole = m(SRC, LENS, TGT).data
    st = m.init_state(m.encode(SRC, LENS))
    for t in range(4):
        logits, st, _ = m.decode_step(TGT[:, t], st)
        np.testing.assert_allclose(logits.data, whole[:, t], rtol=1e-6, atol=1e-6)


def test_bahdanau_step_by_hand():
    m = model("bahdanau")
    st = m.init_state(m.encode(SRC, LENS))
    logits, st1, _ = m.decode_step(TGT[:, 0], st)
    ctx, _ = m.attention(st.h, st.keys, st.mask)
    h = m.cell(F.concat([m.tgt_emb(TGT[:, 0]), ctx], axis=-1), st.h)
    np.testing.assert_allclose(st1.h.data, h.data, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(
        logits.data, m.out(F.concat([h, ctx], axis=-1)).data, rtol=1e-6, atol=1e-7
    )


def test_luong_step_by_hand():
    m = model("luong", cell="lstm")
    st = m.init_state(m.encode(SRC, LENS))
    _, st1, _ = m.decode_step(TGT[:, 0], st)
    logits, st2, _ = m.decode_step(TGT[:, 1], st1)
    h, c = m.cell(F.concat([m.tgt_emb(TGT[:, 1]), st1.feed], axis=-1), (st1.h, st1.c))
    ctx, _ = m.attention(h, st1.keys, st1.mask)
    feed = m.attention.attentional(h, ctx)
    np.testing.assert_allclose(st2.h.data, h.data, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(logits.data, m.out(feed).data, rtol=1e-6, atol=1e-7)
    assert np.abs(st1.feed.data).sum() > 0


def test_padding_is_never_read():
    m = model("bahdanau")
    src2 = SRC.copy()
    src2[1, 2:] = 99
    np.testing.assert_array_equal(m(SRC, LENS, TGT).data, m(src2, LENS, TGT).data)


def test_teacher_forcing_ratio():
    m = model("bahdanau")
    r = R(1)
    free = m(SRC, LENS, TGT, teacher_forcing=0.0, rng=r).data
    assert r.draws == 3
    st = m.init_state(m.encode(SRC, LENS))
    y = TGT[:, 0]
    for t in range(4):
        logits, st, _ = m.decode_step(y, st)
        np.testing.assert_allclose(logits.data, free[:, t], rtol=1e-6, atol=1e-6)
        y = np.argmax(logits.data, axis=-1)
    r1 = R(2)
    m(SRC, LENS, TGT, teacher_forcing=1.0, rng=r1)
    assert r1.draws == 0


def test_greedy_stops_at_eos():
    m = model("luong")
    out = m.greedy(SRC, LENS, 1, 2, max_len=8)
    assert all(len(r) <= 8 and 2 not in r[:-1] for r in out)
    m2 = model()
    sd = m2.state_dict()
    sd["out.bias"] = np.where(np.arange(8) == 2, 50.0, 0.0)
    m2.load_state_dict(sd)
    assert m2.greedy(SRC, LENS, 1, 2, max_len=8) == [[2], [2], [2]]


def test_save_load(tmp_path):
    m = model("luong", cell="lstm")
    save_seq2seq(m, str(tmp_path))
    assert json.loads((tmp_path / "config.json").read_text())["tl_cell"] == "lstm"
    np.testing.assert_array_equal(
        load_seq2seq(str(tmp_path))(SRC, LENS, TGT).data, m(SRC, LENS, TGT).data
    )


def test_validation():
    with pytest.raises(ValueError):
        Seq2Seq(7, 8, 4, 5)
