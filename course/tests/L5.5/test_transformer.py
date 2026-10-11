"""Course tests for L5.5: the encoder-decoder Transformer, post-LN and pre-LN,
the Noam schedule, label smoothing (tinyllm/xfmr/transformer.py).

Rung R0 for these: read them before you write code. Each test names why it
exists (WHY), what kind of check it is (KIND), the planted bugs it kills
(CATCHES, mutants in course/mutants/L5.5), and the chapter section it comes
from.

The worked example of the chapter (section 3):
  label smoothing, V = 3, logits (1, 0, 0), target 0, eps = 0.1:
    p = (0.576117, 0.211942, 0.211942),
    loss = 0.9 * (-ln 0.576117) + 0.1 * mean(-ln p) = 0.618111,
    gradient p - q with q = (0.933333, 0.033333, 0.033333)
           = (-0.357216, 0.178608, 0.178608); a second row whose target is
    the pad id adds nothing.
  Noam, d_model = 16, warmup = 4: update 0 runs at 16^-1/2 * 4^-3/2 =
    0.03125, update 3 at the peak 16^-1/2 * 4^-1/2 = 0.125, update 15 at
    16^-1/2 * 16^-1/2 = 0.0625.

The golden fixture (course/fixtures/L5.5/transformer_torch.npz) holds torch
2.14.1 TransformerEncoder/Decoder logits, the label-smoothed loss, and every
parameter gradient for norm_first False and True with copied weights, from
course/oracle/L5.5/transformer_torch.py. The learning test compares with a
bar in course/fixtures/ref-thresholds.tsv (the reference's mean + 3 sd over
5 seeds).
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
from tinyllm.autograd.tensor import Tensor
from tinyllm.xfmr.transformer import (
    Transformer,
    TransformerConfig,
    fit,
    label_smoothed_loss,
    load_transformer,
    make_optimizer,
    noam_rate,
    save_transformer,
    translate,
)

FIX = os.path.join(
    os.environ.get("TINYLLM_FIXTURES", ""), "L5.5", "transformer_torch.npz"
)
PAD = 0


def differs(a, b, tol: float = 1e-4) -> bool:
    """True when two arrays differ somewhere by more than tol: the opposite of
    closeness, so it needs no tolerance table."""
    return bool(
        np.max(
            np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
        )
        > tol
    )


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

    def shuffle(self, xs) -> None:
        for i in range(len(xs) - 1, 0, -1):
            j = self.g.below(i + 1)
            xs[i], xs[j] = xs[j], xs[i]


def tiny(norm="post", V=11, Vt=None, layers=2, seed=0, **kw) -> Transformer:
    cfg = TransformerConfig(
        V,
        Vt or V,
        d_model=8,
        n_heads=2,
        d_ff=16,
        n_enc=layers,
        n_dec=layers,
        dropout=0.0,
        norm=norm,
        max_len=16,
        **kw,
    )
    return Transformer(cfg, rng=Rng(seed))


def ids(seed, B, T, V=11, lo=3):
    g = PCG32(seed=seed)
    return np.array(
        [[lo + g.below(V - lo) for _ in range(T)] for _ in range(B)], dtype=np.int64
    )


# --- the worked example --------------------------------------------------------------------


def test_hand_example_label_smoothing():
    # WHY: the chapter's worked example: with eps = 0.1 the target keeps
    #      0.9 + 0.1 / 3 of the mass and every other id gets 0.1 / 3, the
    #      loss is 0.618111, the gradient is p - q, and the pad row counts
    #      for nothing (not even in the mean).
    # KIND: unit
    # CATCHES: s07, m07
    # CHAPTER: L5.5 section 3, Worked example by hand
    x = Tensor(
        [[1.0, 0.0, 0.0], [3.0, -1.0, 2.0]], requires_grad=True, dtype=np.float64
    )
    loss = label_smoothed_loss(x, [0, 2], 0.1, pad_id=2)
    e = math.e / (math.e + 2)
    p = [e, (1 - e) / 2, (1 - e) / 2]
    want = 0.9 * -math.log(p[0]) + 0.1 * sum(-math.log(v) for v in p) / 3
    assert_close(float(loss.data), want, rtol=1e-12, atol=1e-12)
    loss.backward()
    q = [0.9 + 0.1 / 3, 0.1 / 3, 0.1 / 3]
    assert_close(
        x.grad, [[a - b for a, b in zip(p, q)], [0.0, 0.0, 0.0]], rtol=1e-12, atol=1e-12
    )


def test_hand_example_noam():
    # WHY: update number s runs at the paper's step_num = s + 1, so the
    #      first update already learns (0.03125), the peak 0.125 is at
    #      update warmup - 1 = 3, and the rate then decays as step^-1/2.
    # KIND: unit
    # CATCHES: s08, m08
    # CHAPTER: L5.5 section 3, Worked example by hand
    assert_close(
        [noam_rate(s, 16, 4) for s in (0, 3, 15)],
        [0.03125, 0.125, 0.0625],
        rtol=1e-12,
        atol=0.0,
    )
    assert_close(noam_rate(3, 16, 4, factor=2.0), 0.25, rtol=1e-12, atol=0.0)
    rates = [noam_rate(s, 16, 4) for s in range(12)]
    assert max(rates) == rates[3] and rates[:4] == sorted(rates[:4])


# --- against torch -----------------------------------------------------------------------------


@pytest.mark.parametrize("norm", ["post", "pre"])
def test_golden_torch(norm):
    # WHY: torch's TransformerEncoder and TransformerDecoder with the same
    #      weights (norm_first=False is post-LN, True is pre-LN with a final
    #      LayerNorm per stack) must give the same logits, label-smoothed
    #      loss, and gradients for every parameter, with source and target
    #      padding and the causal mask.
    # KIND: golden
    # CATCHES: s01, s02, s03, s04, s05, s06, s07, s09, m01, m02, m03, m05, m07
    # CHAPTER: L5.5 section 2.2, Where the LayerNorm goes
    f = np.load(FIX)
    m = tiny(norm)
    pre = f"{norm}.param."
    m.load_state_dict({k[len(pre) :]: f[k] for k in f.files if k.startswith(pre)})
    src, tin, tout = f["src"], f["tgt_in"], f["tgt_out"]
    logits = m(src, tin, src != PAD, tin != PAD)
    assert_close(logits.data, f[f"{norm}.logits"], rtol=1e-4, atol=1e-5)
    loss = label_smoothed_loss(logits, tout, 0.1, PAD)
    assert_close(float(loss.data), float(f[f"{norm}.loss"]), rtol=1e-5, atol=1e-6)
    F.sum(logits * Tensor(f["g"])).backward()
    for name, p in m.named_parameters():
        assert_close(p.grad, f[f"{norm}.grad.{name}"], rtol=1e-3, atol=1e-5, msg=name)


# --- the laws ------------------------------------------------------------------------------------


@pytest.mark.parametrize("norm", ["post", "pre"])
def test_decoder_is_causal(norm):
    # WHY: position t of the decoder may read targets 0..t only; changing
    #      every target after t leaves the logits of positions 0..t bitwise
    #      unchanged. Without it, teacher forcing lets the model copy the
    #      answer and generation fails.
    # KIND: property
    # CATCHES: s03, m01
    # CHAPTER: L5.5 section 2.3, Three kinds of attention
    m = tiny(norm, seed=1)
    src, tin = ids(2, 2, 5), ids(3, 2, 6)
    a = m(src, tin, src != PAD, tin != PAD).data
    for t in (0, 2, 4):
        t2 = tin.copy()
        t2[:, t + 1 :] = ids(4 + t, 2, 6 - t - 1)
        b = m(src, t2, src != PAD, t2 != PAD).data
        assert np.array_equal(a[:, : t + 1], b[:, : t + 1])


def test_source_padding_is_never_read():
    # WHY: a padded source position is masked in the encoder's
    #      self-attention AND in every cross-attention: whatever id sits
    #      there, the logits do not change.
    # KIND: property
    # CATCHES: s06, m01, m02
    # CHAPTER: L5.5 section 2.3, Three kinds of attention
    m = tiny("post", seed=2)
    src, tin = ids(5, 2, 6), ids(6, 2, 4)
    src[1, 3:] = PAD
    mask = np.ones(src.shape, dtype=bool)
    mask[1, 3:] = False
    a = m(src, tin, mask, tin != PAD).data
    src2 = src.copy()
    src2[1, 3:] = [7, 9, 4]
    assert np.array_equal(m(src2, tin, mask, tin != PAD).data, a)


def test_decoder_reads_the_source():
    # WHY: the decoder's cross-attention takes its keys and values from the
    #      encoder output: a different source changes the logits. A model
    #      that attends to its own targets twice still trains, as a language
    #      model of the targets, and never learns the task.
    # KIND: property
    # CATCHES: s04, m01
    # CHAPTER: L5.5 section 2.3, Three kinds of attention
    m = tiny("pre", seed=3)
    tin = ids(7, 2, 4)
    s1, s2 = ids(8, 2, 5), ids(9, 2, 5)
    a = m(s1, tin, s1 != PAD, tin != PAD).data
    b = m(s2, tin, s2 != PAD, tin != PAD).data
    assert differs(a, b)


def test_decode_step_equals_forward():
    # WHY: decoding one token at a time (decode_step, used by beam search)
    #      must give at each step the logits that the full forward pass gives
    #      at that position; the step returns the LAST position.
    # KIND: differential
    # CATCHES: s03, s10, m01, m04
    # CHAPTER: L5.5 section 4, The interface
    m = tiny("post", seed=4)
    src, tin = ids(10, 2, 5), ids(11, 2, 4)
    smask = src != PAD
    full = m(src, tin, smask, np.ones(tin.shape, dtype=bool)).data
    mem = m.encode(src, smask)
    st = m.init_state(smask)
    for t in range(4):
        logits, st = m.decode_step(tin[:, t], mem, st)
        assert_close(logits.data, full[:, t], rtol=1e-5, atol=1e-6)
    assert np.array_equal(st.prefix, tin)


# --- the interface ----------------------------------------------------------------------------------


def test_embeddings_scaled_and_tied():
    # WHY: section 3.4 of the paper: one table for the source embedding,
    #      the target embedding, and the output layer, and the embedding is
    #      multiplied by sqrt(d_model) before the positions are added. Tied
    #      tensors are one parameter, listed once.
    # KIND: unit
    # CATCHES: s05, s09
    # CHAPTER: L5.5 section 2.1, The stacks
    m = tiny("post", seed=5)
    names = [n for n, _ in m.named_parameters()]
    assert (
        "src_emb.weight" in names
        and "tgt_emb.weight" not in names
        and "out.weight" not in names
    )
    assert m.tgt_emb.weight is m.src_emb.weight and m.out.weight is m.src_emb.weight
    x = m.embed(np.array([[3, 5]]), m.src_emb).data
    from tinyllm.xfmr.pos import sinusoidal_pe

    want = m.src_emb.weight.data[[3, 5]] * math.sqrt(8) + sinusoidal_pe(2, 8)
    assert_close(x[0], want, dtype="float32")
    u = tiny("post", V=11, Vt=7, seed=5)
    un = [n for n, _ in u.named_parameters()]
    assert (
        "tgt_emb.weight" in un
        and "out.weight" not in un
        and u.out.weight is u.tgt_emb.weight
    )


def test_parameter_names_follow_the_norm():
    # WHY: the names are the checkpoint keys the zoo (L6.7) loads; pre-LN
    #      adds exactly enc_norm and dec_norm, the decoder has a third norm
    #      and a cross_attn, and the encoder does not.
    # KIND: unit
    # CATCHES: s09
    # CHAPTER: L5.5 section 4, The interface
    post = list(tiny("post", layers=1).state_dict())
    pre = list(tiny("pre", layers=1).state_dict())
    assert sorted(set(pre) - set(post)) == [
        "dec_norm.bias",
        "dec_norm.weight",
        "enc_norm.bias",
        "enc_norm.weight",
    ]
    assert (
        "decoder.0.norm3.weight" in post
        and "decoder.0.cross_attn.q_proj.weight" in post
    )
    assert not any(
        n.startswith("encoder.0.norm3") or n.startswith("encoder.0.cross_attn")
        for n in post
    )
    assert post[0] == "src_emb.weight" and post[-1].startswith("decoder.0.norm3")


def test_optimizer_is_the_papers_adam():
    # WHY: section 5.3: Adam with beta1 0.9, beta2 0.98, eps 1e-9, no weight
    #      decay; the schedule sets the rate, so it starts at 0.
    # KIND: unit
    # CATCHES: m09
    # CHAPTER: L5.5 section 2.4, Training recipe
    opt = make_optimizer(tiny("post", layers=1))
    assert (
        tuple(opt.betas) == (0.9, 0.98)
        and opt.eps == 1e-9
        and opt.weight_decay == 0.0
        and opt.lr == 0.0
    )


def test_fit_follows_the_schedule_and_is_reproducible():
    # WHY: fit runs exactly `steps` updates at the Noam rate of each update
    #      (or a constant lr), and the same seeds give the same losses bit
    #      for bit: the milestone's divergence demo compares runs.
    # KIND: property
    # CATCHES: s08, m01, m10
    # CHAPTER: L5.5 section 2.4, Training recipe
    src, tgt = (
        ids(12, 32, 5),
        np.concatenate([np.ones((32, 1), dtype=np.int64), ids(13, 32, 4)], axis=1),
    )
    a, b = tiny("pre", seed=6, layers=1), tiny("pre", seed=6, layers=1)
    seen = []
    la = fit(
        a,
        src,
        tgt,
        6,
        8,
        PAD,
        warmup=4,
        rng=Rng(1),
        on_step=lambda i, s: seen.append(i),
    )
    lb = fit(b, src, tgt, 6, 8, PAD, warmup=4, rng=Rng(1))
    assert la == lb and len(la) == 6 and seen == list(range(6))
    c = tiny("pre", seed=6, layers=1)
    before = c.state_dict()
    lc = fit(c, src, tgt, 6, 8, PAD, lr=0.0, rng=Rng(1))
    assert all(
        np.array_equal(before[n], v) for n, v in c.state_dict().items()
    )  # lr 0: nothing moves
    assert lc[0] == la[0] and lc[1] != la[1]


def test_translate_is_greedy_with_beam_one_and_drops_eos(tmp_path):
    # WHY: translate decodes each source with L4.4's beam search over
    #      decode_step; with beam 1 that is the argmax at every step, the
    #      final eos is not part of the output, and the model's mode is
    #      restored afterwards. A saved and reloaded model translates the same.
    # KIND: differential
    # CATCHES: s11, m01
    # CHAPTER: L5.5 section 4, The interface
    m = tiny("post", seed=7, layers=1)
    src = ids(14, 3, 5)
    BOS = 1
    # eos := the first token row 0 picks, so at least one search finishes.
    m.eval()
    first, _ = m.decode_step(
        [BOS], m.encode(src[:1], src[:1] != PAD), m.init_state(src[:1] != PAD)
    )
    EOS = int(np.argmax(first.data[0]))
    m.train()
    out = translate(m, src, BOS, EOS, PAD, 6)
    assert out[0] == []
    assert m.training
    m.eval()
    for row, got in zip(src, out):
        s = row[None]
        mem = m.encode(s, s != PAD)
        st, y, want = m.init_state(s != PAD), BOS, []
        for _ in range(6):
            logits, st = m.decode_step([y], mem, st)
            y = int(np.argmax(logits.data[0]))
            if y == EOS:
                break
            want.append(y)
        assert got == want and EOS not in got
    save_transformer(m, str(tmp_path / "x"))
    back = load_transformer(str(tmp_path / "x"))
    assert translate(back, src, BOS, EOS, PAD, 6) == out
    assert all(
        np.array_equal(a, b)
        for a, b in zip(back.state_dict().values(), m.state_dict().values())
    )


def test_validation():
    # WHY: an unknown LayerNorm placement, a mask that is not bool [B, T],
    #      or a prefix whose batch differs from the memory's is a wiring bug.
    # KIND: boundary
    # CATCHES: m06
    # CHAPTER: L5.5 section 4, The interface
    with pytest.raises(ValueError):
        tiny("middle")
    m = tiny("post", layers=1)
    src, tin = ids(15, 2, 4), ids(16, 2, 3)
    with pytest.raises(ValueError):
        m(src, tin, (src != PAD).astype(np.int64), tin != PAD)
    with pytest.raises(ValueError):
        m(src, tin, src != PAD, (tin != PAD)[:, :2])
    with pytest.raises(ValueError):
        m.decode_step([1, 1, 1], m.encode(src, src != PAD), m.init_state(src != PAD))


# --- learning -------------------------------------------------------------------------------------------


def addition(n, seed):
    """Two-digit reversed addition: '21+43' (12 + 34) -> '64' + '0' (46 -> '640')."""
    g = PCG32(seed=seed)
    S, T = [], []
    for _ in range(n):
        a, b = g.below(100), g.below(100)
        s = f"{a:02d}"[::-1] + "+" + f"{b:02d}"[::-1]
        t = f"{a + b:03d}"[::-1]
        S.append([3 + int(c) if c != "+" else 13 for c in s])
        T.append([1] + [3 + int(c) for c in t] + [2])
    return np.array(S, dtype=np.int64), np.array(T, dtype=np.int64)


def test_learns_reversed_addition():
    # WHY: the whole recipe (pre-LN, Adam, Noam warmup, label smoothing,
    #      tied embeddings) must learn: 600 updates of a one-layer d = 32
    #      model on two-digit reversed addition reach the reference's
    #      held-out loss (mean + 3 sd over 5 seeds).
    # KIND: learning
    # CATCHES: m01
    # CHAPTER: L5.5 section 2.4, Training recipe
    seed = int(os.environ.get("OL_SEED", "0"))
    src, tgt = addition(2000, 100 + seed)
    vs, vt = addition(200, 999)
    cfg = TransformerConfig(
        14,
        14,
        d_model=32,
        n_heads=4,
        d_ff=64,
        n_enc=1,
        n_dec=1,
        dropout=0.0,
        norm="pre",
        max_len=8,
    )
    m = Transformer(cfg, rng=Rng(seed))
    fit(
        m,
        src,
        tgt,
        600,
        32,
        PAD,
        warmup=100,
        factor=0.5,
        smoothing=0.1,
        rng=Rng(seed + 1),
    )
    m.eval()
    logits = m(vs, vt[:, :-1], vs != PAD, vt[:, :-1] != PAD)
    val = float(label_smoothed_loss(logits, vt[:, 1:], 0.0, PAD).data)
    check("L5.5/test_learns_reversed_addition", "val_loss", val, direction="max")
