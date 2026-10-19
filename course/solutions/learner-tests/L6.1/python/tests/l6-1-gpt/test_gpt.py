"""My tests for L6.1 (rung R5: a GPT-2 forward written out in float64 numpy
from Hugging Face-layout weights I make myself is the oracle for loading and
for the model; the causal law; the loss by hand). They import only the
contract."""

import math

import numpy as np
import pytest
from tinyllm.autograd.tensor import Tensor
from tinyllm.num.rng import PCG32
from tinyllm.obj.gpt import GPT, GPTConfig, clm_loss, load_hf_gpt2

CFG = GPTConfig(vocab=12, n_ctx=8, d_model=8, n_heads=2, n_layers=1, d_ff=16)


def hf_weights(seed):
    r = np.random.Generator(np.random.PCG64(seed))
    d, f = 8, 16
    n = lambda *s: r.normal(size=s) * 0.6  # noqa: E731
    sd = {
        "transformer.wte.weight": n(12, d),
        "transformer.wpe.weight": n(8, d),
        "transformer.ln_f.weight": 1 + n(d),
        "transformer.ln_f.bias": n(d),
    }
    p = "transformer.h.0."
    sd.update(
        {
            p + "ln_1.weight": 1 + n(d),
            p + "ln_1.bias": n(d),
            p + "ln_2.weight": 1 + n(d),
            p + "ln_2.bias": n(d),
            p + "attn.c_attn.weight": n(d, 3 * d),
            p + "attn.c_attn.bias": n(3 * d),
            p + "attn.c_proj.weight": n(d, d),
            p + "attn.c_proj.bias": n(d),
            p + "mlp.c_fc.weight": n(d, f),
            p + "mlp.c_fc.bias": n(f),
            p + "mlp.c_proj.weight": n(f, d),
            p + "mlp.c_proj.bias": n(d),
        }
    )
    return sd


def hf_forward(sd, ids):
    d, H = 8, 2
    p = "transformer.h.0."

    def ln(x, k):
        return (x - x.mean(-1, keepdims=True)) / np.sqrt(
            x.var(-1, keepdims=True) + 1e-5
        ) * sd[k + ".weight"] + sd[k + ".bias"]

    def conv(x, k):  # HF Conv1D: x @ W + b, W is [in, out]
        return x @ sd[k + ".weight"] + sd[k + ".bias"]

    T = ids.shape[1]
    x = sd["transformer.wte.weight"][ids] + sd["transformer.wpe.weight"][:T]
    qkv = conv(ln(x, p + "ln_1"), p + "attn.c_attn")
    q, k, v = (
        qkv[..., i * d : (i + 1) * d].reshape(-1, T, H, d // H).transpose(0, 2, 1, 3)
        for i in range(3)
    )
    e = np.where(
        np.tril(np.ones((T, T), dtype=bool)),
        q @ k.transpose(0, 1, 3, 2) / math.sqrt(d // H),
        -np.inf,
    )
    a = np.exp(e - e.max(-1, keepdims=True))
    a /= a.sum(-1, keepdims=True)
    x = x + conv((a @ v).transpose(0, 2, 1, 3).reshape(x.shape), p + "attn.c_proj")
    u = conv(ln(x, p + "ln_2"), p + "mlp.c_fc")
    x = x + conv(
        0.5 * u * (1 + np.tanh(math.sqrt(2 / math.pi) * (u + 0.044715 * u**3))),
        p + "mlp.c_proj",
    )
    return ln(x, "transformer.ln_f") @ sd["transformer.wte.weight"].T


def test_loads_hf_layout_and_matches_numpy():
    sd = hf_weights(1)
    m = GPT(CFG, rng=PCG32(0))
    load_hf_gpt2(m, sd)
    ids = np.random.Generator(np.random.PCG64(2)).integers(0, 12, size=(2, 6))
    np.testing.assert_allclose(
        m(ids)[0].data, hf_forward(sd, ids), rtol=1e-4, atol=1e-4
    )


def test_causal():
    m = GPT(CFG, rng=PCG32(3))
    x = np.array([[1, 2, 3, 4, 5]])
    y = x.copy()
    y[0, 3:] = 9
    np.testing.assert_array_equal(m(x)[0].data[:, :3], m(y)[0].data[:, :3])


def test_clm_loss_hand_example():
    logits = Tensor([[[0.0, math.log(3.0)], [0.0, 0.0], [5.0, -5.0]]], dtype=np.float64)
    ids = np.array([[0, 1, 1]])
    assert float(clm_loss(logits, ids).data) == pytest.approx(
        (-math.log(0.75) + math.log(2)) / 2
    )
    assert float(
        clm_loss(logits, ids, np.array([[1, 1, 0]], dtype=bool)).data
    ) == pytest.approx(-math.log(0.75))


def test_init_scales_residual_projections():
    m = GPT(
        GPTConfig(vocab=32, n_ctx=8, d_model=64, n_heads=4, n_layers=8, d_ff=256),
        rng=PCG32(4),
    )
    sd = m.state_dict()
    assert abs(sd["h.0.mlp_proj.weight"].std() - 0.02 / 4) < 0.0004
    assert abs(sd["h.0.mlp_fc.weight"].std() - 0.02) < 0.0016
    assert "lm_head.weight" not in sd


def test_tied_head_gets_both_gradients_and_bare_keys_load():
    import tinyllm.autograd.functional as F

    m = GPT(CFG, rng=PCG32(5))
    F.sum(m(np.array([[1, 2]]))[0]).backward()
    assert np.any(
        m.wte.weight.grad[5] != 0
    )  # token 5 is never read: only the head reaches it
    sd = hf_weights(6)
    a, b = GPT(CFG, rng=PCG32(7)), GPT(CFG, rng=PCG32(8))
    load_hf_gpt2(a, sd)
    load_hf_gpt2(b, {k.removeprefix("transformer."): v for k, v in sd.items()})
    assert all(
        np.array_equal(x, y)
        for x, y in zip(a.state_dict().values(), b.state_dict().values())
    )


def test_untied_and_validation():
    u = GPT(
        GPTConfig(
            vocab=12, n_ctx=8, d_model=8, n_heads=2, n_layers=1, d_ff=16, tie=False
        ),
        rng=PCG32(9),
    )
    assert "lm_head.weight" in u.state_dict()
    with pytest.raises(ValueError):
        clm_loss(GPT(CFG)(np.array([[1]]))[0], np.array([[1]]))
