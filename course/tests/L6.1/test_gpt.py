"""Course tests for L6.1: GPT decoder-only, the causal LM loss, and GPT-2
weight loading (tinyllm/obj/gpt.py).

Rung R0 for these: read them before you write code. Each test names why it
exists (WHY), what kind of check it is (KIND), the planted bugs it kills
(CATCHES, mutants in course/mutants/L6.1), and the chapter section it comes
from.

The worked example of the chapter (section 3): ids (0, 1, 1), V = 2, and
logits rows (0, ln 3), (0, 0), (5, -5). Position 0 predicts id 1 with
p = 3 / 4, loss -ln 0.75 = 0.287682; position 1 predicts id 1 with p = 1/2,
loss 0.693147; position 2 has no next token and is dropped. The causal LM
loss is the mean, 0.490415. With the third token marked as padding, only
position 0 counts: 0.287682.

The golden fixture (course/fixtures/L6.1/gpt2_tiny_hf.npz) holds a random
tiny Hugging Face GPT2LMHeadModel (transformers 5.19.0, torch 2.14.1): its
state dict, logits, loss, and gradients, from course/oracle/L6.1/gpt2_hf.py.
The learning test trains on course/fixtures/MS-L2/train.bin (byte tokens,
read with L0.6's TokenStream) and compares the held-out loss on
course/fixtures/MS-L2/val.bin with a bar in ref-thresholds.tsv.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32
from _lib.thresholds import check
from tinyllm.autograd.mode import no_grad
from tinyllm.autograd.tensor import Tensor
from tinyllm.io.tokens import TokenStream, open_tokens
from tinyllm.obj.gpt import (
    GPT,
    GPTConfig,
    clm_loss,
    fit_gpt,
    load_gpt,
    load_hf_gpt2,
    save_gpt,
)

FIXDIR = os.environ.get("TINYLLM_FIXTURES", "")
FIX = os.path.join(FIXDIR, "L6.1", "gpt2_tiny_hf.npz")
CFG = GPTConfig(vocab=50, n_ctx=16, d_model=16, n_heads=2, n_layers=2, d_ff=64)


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


def hf_sd():
    f = np.load(FIX)
    return f, {k[3:]: f[k] for k in f.files if k.startswith("sd.")}


def ids(seed, B, T, V=50):
    g = PCG32(seed=seed)
    return np.array([[g.below(V) for _ in range(T)] for _ in range(B)], dtype=np.int64)


# --- the worked example --------------------------------------------------------------------


def test_hand_example_clm_loss():
    # WHY: the chapter's worked example: position t predicts token t + 1, so
    #      the last position has no target, and a padded target counts for
    #      nothing.
    # KIND: unit
    # CATCHES: s04, s09, m01
    # CHAPTER: L6.1 section 3, Worked example by hand
    logits = Tensor(
        [[[0.0, math.log(3.0)], [0.0, 0.0], [5.0, -5.0]]],
        requires_grad=True,
        dtype=np.float64,
    )
    x = np.array([[0, 1, 1]])
    want = (-math.log(0.75) + math.log(2.0)) / 2
    assert_close(float(clm_loss(logits, x).data), want, rtol=1e-12, atol=1e-12)
    padded = clm_loss(logits, x, mask=np.array([[True, True, False]]))
    assert_close(float(padded.data), -math.log(0.75), rtol=1e-12, atol=1e-12)
    padded.backward()
    assert_close(logits.grad[0, 0], [0.25, -0.25], rtol=1e-12, atol=1e-12)
    assert np.all(logits.grad[0, 1:] == 0.0)


# --- against Hugging Face ----------------------------------------------------------------------------


def test_golden_hf_gpt2():
    # WHY: a random tiny GPT-2 from Hugging Face, loaded through
    #      load_hf_gpt2, must give HF's logits and loss, and the gradient of
    #      that loss for every HF tensor (our q, k, v, and Linear weights
    #      mapped back to HF's packed, transposed Conv1D layout).
    # KIND: golden
    # CATCHES: s01, s02, s03, s04, s06, s07, s10, m01, m02, m03, m04, m05, m06, m07
    # CHAPTER: L6.1 section 2.4, Loading GPT-2
    f, sd = hf_sd()
    m = GPT(CFG, rng=Rng(0))
    load_hf_gpt2(m, sd)
    x = f["ids"]
    logits, _ = m(x)
    assert_close(logits.data, f["logits"], rtol=1e-4, atol=1e-5)
    loss = clm_loss(logits, x)
    assert_close(float(loss.data), float(f["loss"]), rtol=1e-5, atol=1e-6)
    loss.backward()
    d = CFG.d_model
    g = {n: p.grad for n, p in m.named_parameters()}
    assert_close(
        g["wte.weight"],
        f["grad.transformer.wte.weight"],
        rtol=1e-4,
        atol=1e-6,
        msg="wte",
    )
    assert_close(
        g["wpe.weight"],
        f["grad.transformer.wpe.weight"],
        rtol=1e-4,
        atol=1e-6,
        msg="wpe",
    )
    for i in range(CFG.n_layers):
        p, h = f"h.{i}.", f"grad.transformer.h.{i}."
        qkv = np.concatenate(
            [g[f"{p}attn.{n}.weight"].T for n in ("q_proj", "k_proj", "v_proj")], axis=1
        )
        assert_close(
            qkv, f[h + "attn.c_attn.weight"], rtol=1e-4, atol=1e-6, msg=f"{p}c_attn"
        )
        bqkv = np.concatenate(
            [g[f"{p}attn.{n}.bias"] for n in ("q_proj", "k_proj", "v_proj")]
        )
        assert_close(
            bqkv, f[h + "attn.c_attn.bias"], rtol=1e-4, atol=1e-6, msg=f"{p}c_attn.bias"
        )
        assert_close(
            g[p + "attn.out_proj.weight"].T,
            f[h + "attn.c_proj.weight"],
            rtol=1e-4,
            atol=1e-6,
        )
        assert_close(
            g[p + "mlp_fc.weight"].T, f[h + "mlp.c_fc.weight"], rtol=1e-4, atol=1e-6
        )
        assert_close(
            g[p + "mlp_proj.weight"].T, f[h + "mlp.c_proj.weight"], rtol=1e-4, atol=1e-6
        )
        assert_close(g[p + "ln_1.weight"], f[h + "ln_1.weight"], rtol=1e-4, atol=1e-6)
    assert_close(g["ln_f.bias"], f["grad.transformer.ln_f.bias"], rtol=1e-4, atol=1e-6)
    assert qkv.shape == (d, 3 * d)


def test_load_hf_gpt2_keys():
    # WHY: GPT2Model checkpoints have no "transformer." prefix and older
    #      ones carry attn.bias / attn.masked_bias buffers; both must load to
    #      the same model. A missing key is named; a wrong config is a shape
    #      error, not a silent broadcast.
    # KIND: boundary
    # CATCHES: m03, m08
    # CHAPTER: L6.1 section 2.4, Loading GPT-2
    f, sd = hf_sd()
    a, b = GPT(CFG, rng=Rng(1)), GPT(CFG, rng=Rng(2))
    load_hf_gpt2(a, sd)
    bare = {k.removeprefix("transformer."): v for k, v in sd.items()}
    bare["h.0.attn.bias"] = np.tril(np.ones((1, 1, 16, 16)))
    bare["h.0.attn.masked_bias"] = np.array(-1e4)
    load_hf_gpt2(b, bare)
    assert all(
        np.array_equal(x, y)
        for x, y in zip(a.state_dict().values(), b.state_dict().values())
    )
    broken = dict(sd)
    del broken["transformer.h.1.mlp.c_fc.bias"]
    with pytest.raises(KeyError, match="h.1.mlp.c_fc.bias"):
        load_hf_gpt2(GPT(CFG), broken)
    with pytest.raises(ValueError):
        load_hf_gpt2(GPT(GPTConfig(50, 16, 8, 2, 2, 32)), sd)


# --- the laws ---------------------------------------------------------------------------------------


def test_logits_are_causal():
    # WHY: position t reads tokens 0..t only: changing the tokens after t
    #      leaves the logits of 0..t bitwise unchanged, so training on every
    #      position at once is honest and generation needs no future.
    # KIND: property
    # CATCHES: s03, m02
    # CHAPTER: L6.1 section 2.1, One stack, one mask
    m = GPT(CFG, rng=Rng(3))
    x = ids(4, 2, 10)
    a = m(x)[0].data
    for t in (0, 4, 8):
        y = x.copy()
        y[:, t + 1 :] = ids(5 + t, 2, 10 - t - 1)
        assert np.array_equal(m(y)[0].data[:, : t + 1], a[:, : t + 1])


def test_block_matches_the_formula():
    # WHY: one GPT-2 block written out in float64 numpy from the block's own
    #      weights (pre-LN, causal attention, tanh-approximated GELU, two
    #      residual adds), with weights large enough that the exact GELU
    #      differs from the tanh form by far more than the tolerance.
    # KIND: differential
    # CATCHES: s05, m05, m06, m07
    # CHAPTER: L6.1 section 2.3, The GPT-2 block
    from tinyllm.obj.gpt import Block

    cfg = GPTConfig(vocab=10, n_ctx=8, d_model=8, n_heads=2, n_layers=1, d_ff=16)
    b = Block(cfg, Rng(20))
    g = PCG32(seed=21)
    b.load_state_dict(
        {
            n: g.normal_array(v.shape) * 0.7
            + (1.0 if n.endswith("ln_1.weight") or n.endswith("ln_2.weight") else 0.0)
            for n, v in b.state_dict().items()
        }
    )
    x = g.normal_array((2, 5, 8)) * 2.0
    sd = {n: v.astype(np.float64) for n, v in b.state_dict().items()}

    def ln(z, p):
        mu, var = z.mean(-1, keepdims=True), z.var(-1, keepdims=True)
        return (z - mu) / np.sqrt(var + 1e-5) * sd[p + ".weight"] + sd[p + ".bias"]

    def lin(z, p):
        return z @ sd[p + ".weight"].T + sd[p + ".bias"]

    h = ln(x, "ln_1")
    q, k, v = (
        lin(h, f"attn.{n}_proj").reshape(2, 5, 2, 4).transpose(0, 2, 1, 3)
        for n in "qkv"
    )
    e = q @ k.transpose(0, 1, 3, 2) / 2.0 + np.where(
        np.tril(np.ones((5, 5))) > 0, 0.0, -np.inf
    )
    a = np.exp(e - e.max(-1, keepdims=True))
    a /= a.sum(-1, keepdims=True)
    y = x + lin((a @ v).transpose(0, 2, 1, 3).reshape(2, 5, 8), "attn.out_proj")
    u = lin(ln(y, "ln_2"), "mlp_fc")
    gelu = 0.5 * u * (1 + np.tanh(math.sqrt(2 / math.pi) * (u + 0.044715 * u**3)))
    want = y + lin(gelu, "mlp_proj")
    from tinyllm.xfmr.masks import causal_mask

    assert_close(b(Tensor(x), causal_mask(5)).data, want, rtol=1e-4, atol=1e-4)


def test_forward_targets_equals_clm_loss():
    # WHY: forward(ids[:, :-1], targets=ids[:, 1:]) is the same loss as
    #      clm_loss on the full window: the shift happens once, either in the
    #      data (L0.6's TokenStream gives inputs and targets) or in the loss.
    # KIND: property
    # CATCHES: s03, s04, m01, m02
    # CHAPTER: L6.1 section 2.2, The causal LM loss
    m = GPT(CFG, rng=Rng(6))
    x = ids(7, 3, 9)
    _, a = m(x[:, :-1], x[:, 1:])
    logits, _ = m(x)
    assert_close(float(a.data), float(clm_loss(logits, x).data), rtol=1e-6, atol=1e-7)


def test_head_is_tied_to_the_embedding():
    # WHY: the output layer is the token table itself (logits = h E^T), one
    #      parameter with two jobs: it is listed once and its gradient sums
    #      both uses. tie=False adds a separate lm_head.
    # KIND: unit
    # CATCHES: m02, m09
    # CHAPTER: L6.1 section 2.3, The GPT-2 block
    m = GPT(CFG, rng=Rng(8))
    names = [n for n, _ in m.named_parameters()]
    assert "lm_head.weight" not in names and names[:2] == ["wte.weight", "wpe.weight"]
    x = ids(9, 1, 6)
    with no_grad():
        h = m.hidden(x).data
        assert_close(m(x)[0].data, h @ m.wte.weight.data.T, rtol=1e-5, atol=1e-6)
    u = GPT(GPTConfig(50, 16, 16, 2, 2, 64, tie=False), rng=Rng(8))
    assert [n for n, _ in u.named_parameters()][-1] == "lm_head.weight"


def test_gpt2_init():
    # WHY: GPT-2's init: N(0, 0.02^2) everywhere except the two projections
    #      that write into the residual stream, which get
    #      0.02 / sqrt(2 n_layers) so the stream's variance does not grow with
    #      depth; biases 0, LayerNorms (1, 0); a seed fixes every weight.
    # KIND: statistical
    # CATCHES: s08, m10
    # CHAPTER: L6.1 section 2.3, The GPT-2 block
    cfg = GPTConfig(vocab=64, n_ctx=32, d_model=64, n_heads=4, n_layers=8, d_ff=256)
    m = GPT(cfg, rng=Rng(10))
    sd = m.state_dict()
    resid = 0.02 / math.sqrt(16)
    for name, want in (
        ("h.3.attn.out_proj.weight", resid),
        ("h.3.mlp_proj.weight", resid),
        ("h.3.attn.q_proj.weight", 0.02),
        ("h.3.mlp_fc.weight", 0.02),
        ("wte.weight", 0.02),
    ):
        s = float(sd[name].astype(np.float64).std())
        assert abs(s - want) < 0.08 * want, f"{name}: std {s:.5f}, want {want:.5f}"
    assert np.all(sd["h.0.mlp_fc.bias"] == 0) and np.all(sd["ln_f.weight"] == 1)
    m2 = GPT(cfg, rng=Rng(10))
    assert all(
        np.array_equal(a, b) for a, b in zip(sd.values(), m2.state_dict().values())
    )


def test_save_load_roundtrip(tmp_path):
    # WHY: the model directory is how the zoo (L6.7) and later passes load a
    #      GPT: config.json says tl_arch "gpt" and the sizes, the weights come
    #      back exactly, and the logits match.
    # KIND: unit
    # CATCHES: m02, m11
    # CHAPTER: L6.1 section 4, The interface
    m = GPT(CFG, rng=Rng(11))
    save_gpt(m, str(tmp_path / "g"))
    back = load_gpt(str(tmp_path / "g"))
    assert back.cfg == CFG and not back.training
    x = ids(12, 2, 7)
    assert np.array_equal(back(x)[0].data, m(x)[0].data)


def test_validation():
    # WHY: a window longer than the learned position table has no position
    #      embedding; the loss needs at least two tokens.
    # KIND: boundary
    # CATCHES: m12
    # CHAPTER: L6.1 section 4, The interface
    m = GPT(CFG, rng=Rng(13))
    with pytest.raises(ValueError):
        m(ids(14, 1, 17))
    with pytest.raises(ValueError):
        clm_loss(m(ids(15, 1, 1))[0], ids(15, 1, 1))
    with pytest.raises(ValueError):
        m(np.zeros((2, 3)))


# --- learning -------------------------------------------------------------------------------------------


def test_learns_byte_stories():
    # WHY: the whole model learns language: 150 AdamW updates with cosine
    #      decay of a 2-layer d = 64 byte-level GPT, batches of 8 windows of
    #      64 bytes from L0.6's TokenStream, reach the reference's held-out
    #      loss (mean + 3 sd over 5 seeds) on the validation shard.
    # KIND: learning
    # CATCHES: m02
    # CHAPTER: L6.1 section 2.2, The causal LM loss
    seed = int(os.environ.get("OL_SEED", "0"))
    cfg = GPTConfig(vocab=256, n_ctx=64, d_model=64, n_heads=4, n_layers=2, d_ff=256)
    m = GPT(cfg, rng=Rng(seed))
    stream = TokenStream(
        [os.path.join(FIXDIR, "MS-L2", "train.bin")],
        64,
        8,
        PCG32(seed=seed + 1),
        vocab_size=256,
    )
    losses = fit_gpt(m, stream, 150, lr=3e-3, warmup=15)
    assert len(losses) == 150 and losses[-1] < losses[0]
    val = np.asarray(
        open_tokens(os.path.join(FIXDIR, "MS-L2", "val.bin")), dtype=np.int64
    )
    n = (val.size - 1) // 64
    windows = np.stack([val[i * 64 : i * 64 + 65] for i in range(0, min(n - 1, 64), 4)])
    m.eval()
    with no_grad():
        z = m(windows[:, :64])[0].data.astype(np.float64)
    z -= z.max(axis=-1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
    nll = float(-np.take_along_axis(logp, windows[:, 1:, None], axis=-1).mean())
    check("L6.1/test_learns_byte_stories", "val_loss", nll, direction="max")
