"""My tests for L5.5 (rung R5: the whole transformer written out in float64
numpy from the model's own state_dict is the oracle, for both LayerNorm
placements; plus the hand examples and the causality and padding laws).
They import only the contract."""

import math

import numpy as np
import pytest
from tinyllm.autograd.tensor import Tensor
from tinyllm.num.rng import PCG32
from tinyllm.xfmr.pos import sinusoidal_pe
from tinyllm.xfmr.transformer import (
    Transformer,
    TransformerConfig,
    fit,
    label_smoothed_loss,
    make_optimizer,
    noam_rate,
    translate,
)

PAD = 0


def model(norm, layers=1, seed=0):
    cfg = TransformerConfig(
        9,
        9,
        d_model=8,
        n_heads=2,
        d_ff=16,
        n_enc=layers,
        n_dec=layers,
        dropout=0.0,
        norm=norm,
        max_len=12,
    )
    return Transformer(cfg, rng=PCG32(seed))


def ids(seed, B, T):
    return np.random.Generator(np.random.PCG64(seed)).integers(3, 9, size=(B, T))


def oracle(m, src, tin):
    sd = {k: v.astype(np.float64) for k, v in m.state_dict().items()}
    d, H = 8, 2
    pre = m.cfg.norm == "pre"
    E = sd["src_emb.weight"]

    def ln(x, p):
        return (x - x.mean(-1, keepdims=True)) / np.sqrt(
            x.var(-1, keepdims=True) + 1e-5
        ) * sd[p + ".weight"] + sd[p + ".bias"]

    def lin(x, p):
        return x @ sd[p + ".weight"].T + sd[p + ".bias"]

    def mha(p, xq, xkv, mask):
        def sp(x):
            return x.reshape(x.shape[0], x.shape[1], H, d // H).transpose(0, 2, 1, 3)

        q, k, v = (
            sp(lin(xq, p + ".q_proj")),
            sp(lin(xkv, p + ".k_proj")),
            sp(lin(xkv, p + ".v_proj")),
        )
        e = np.where(mask, q @ k.transpose(0, 1, 3, 2) / math.sqrt(d // H), -np.inf)
        a = np.exp(e - e.max(-1, keepdims=True))
        a /= a.sum(-1, keepdims=True)
        return lin((a @ v).transpose(0, 2, 1, 3).reshape(xq.shape), p + ".out_proj")

    def ff(p, x):
        return lin(np.maximum(lin(x, p + ".linear1"), 0.0), p + ".linear2")

    smask = (src != PAD)[:, None, None, :]
    T = tin.shape[1]
    tmask = (
        np.tril(np.ones((T, T), dtype=bool))[None, None]
        & (tin != PAD)[:, None, None, :]
    )
    x = E[src] * math.sqrt(d) + sinusoidal_pe(src.shape[1], d)
    p = "encoder.0"
    if pre:
        h = ln(x, p + ".norm1")
        x = x + mha(p + ".self_attn", h, h, smask)
        x = x + ff(p, ln(x, p + ".norm2"))
        x = ln(x, "enc_norm")
    else:
        x = ln(x + mha(p + ".self_attn", x, x, smask), p + ".norm1")
        x = ln(x + ff(p, x), p + ".norm2")
    y = E[tin] * math.sqrt(d) + sinusoidal_pe(T, d)
    p = "decoder.0"
    if pre:
        h = ln(y, p + ".norm1")
        y = y + mha(p + ".self_attn", h, h, tmask)
        y = y + mha(p + ".cross_attn", ln(y, p + ".norm2"), x, smask)
        y = ln(y + ff(p, ln(y, p + ".norm3")), "dec_norm")
    else:
        y = ln(y + mha(p + ".self_attn", y, y, tmask), p + ".norm1")
        y = ln(y + mha(p + ".cross_attn", y, x, smask), p + ".norm2")
        y = ln(y + ff(p, y), p + ".norm3")
    return y @ E.T


@pytest.mark.parametrize("norm", ["post", "pre"])
def test_matches_numpy(norm):
    m = model(norm, seed=1)
    src, tin = ids(2, 2, 5), ids(3, 2, 4)
    src[1, 3:] = PAD
    tin[1, 2:] = PAD
    got = m(src, tin, src != PAD, tin != PAD).data
    np.testing.assert_allclose(got, oracle(m, src, tin), rtol=1e-4, atol=1e-4)


def test_label_smoothing_hand_example():
    x = Tensor([[1.0, 0.0, 0.0], [5.0, 1.0, 2.0]], requires_grad=True, dtype=np.float64)
    loss = label_smoothed_loss(x, [0, 2], 0.1, pad_id=2)
    p = np.exp([1.0, 0, 0]) / np.exp([1.0, 0, 0]).sum()
    want = 0.9 * -np.log(p[0]) + 0.1 * np.mean(-np.log(p))
    assert abs(float(loss.data) - want) < 1e-12
    loss.backward()
    np.testing.assert_allclose(
        x.grad[0], p - [0.9 + 0.1 / 3, 0.1 / 3, 0.1 / 3], atol=1e-12
    )


def test_noam():
    assert noam_rate(0, 16, 4) == pytest.approx(0.03125)
    assert noam_rate(3, 16, 4, factor=2.0) == pytest.approx(0.25)
    opt = make_optimizer(model("pre"))
    assert tuple(opt.betas) == (0.9, 0.98) and opt.eps == 1e-9


def test_causal_and_padding_laws():
    m = model("post", seed=4)
    src, tin = ids(5, 2, 6), ids(6, 2, 5)
    a = m(src, tin, src != PAD, tin != PAD).data
    t2 = tin.copy()
    t2[:, 3:] = 8
    b = m(src, t2, src != PAD, t2 != PAD).data
    np.testing.assert_array_equal(a[:, :3], b[:, :3])
    s2 = src.copy()
    mask = src != PAD
    mask[0, 4:] = False
    s2[0, 4:] = 3
    np.testing.assert_array_equal(
        m(src, tin, mask, tin != PAD).data, m(s2, tin, mask, tin != PAD).data
    )


def test_decode_step_and_translate():
    m = model("pre", seed=7)
    src, tin = ids(8, 1, 5), ids(9, 1, 4)
    full = m(src, tin, src != PAD, np.ones(tin.shape, dtype=bool)).data
    mem, st = m.encode(src, src != PAD), m.init_state(src != PAD)
    for t in range(4):
        logits, st = m.decode_step(tin[:, t], mem, st)
        np.testing.assert_allclose(logits.data, full[:, t], rtol=1e-5, atol=1e-6)
    first, _ = m.decode_step([1], mem, m.init_state(src != PAD))
    eos = int(np.argmax(first.data[0]))
    assert translate(m, src, 1, eos, PAD, 5) == [[]]


def test_fit_is_reproducible_and_tied():
    m = model("pre", seed=10)
    names = [n for n, _ in m.named_parameters()]
    assert (
        "out.weight" not in names
        and "tgt_emb.weight" not in names
        and "enc_norm.weight" in names
    )
    src = ids(11, 16, 4)
    tgt = np.concatenate([np.ones((16, 1), dtype=np.int64), ids(12, 16, 3)], axis=1)
    a = fit(m, src, tgt, 3, 4, PAD, warmup=2, rng=PCG32(1))
    b = fit(model("pre", seed=10), src, tgt, 3, 4, PAD, warmup=2, rng=PCG32(1))
    assert a == b
