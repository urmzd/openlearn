"""My tests for L7.6 (rung R5). The oracle is MLA written out in numpy
float64 from the chapter's equations, with RoPE as complex multiplication;
the absorbed form is checked against the naive one. They import only the
contract."""

import numpy as np
import pytest
import tinyllm.autograd.functional as F
from tinyllm.autograd.tensor import Tensor
from tinyllm.modern.mla import ConcatLatentCache, MLAttention
from tinyllm.modern.rope import RopeSpec

D, H, R, DN, DR, DV = 12, 3, 5, 4, 4, 3


def ladder(r, base=10000.0):
    return base ** (-np.arange(0, r, 2) / r)


def rope_np(x, pos, layout):
    ang = pos[:, None] * ladder(x.shape[-1])
    if layout == "half":
        h = x.shape[-1] // 2
        z = (x[..., :h] + 1j * x[..., h:]) * np.exp(1j * ang)
        return np.concatenate([z.real, z.imag], axis=-1)
    z = (x[..., 0::2] + 1j * x[..., 1::2]) * np.exp(1j * ang)
    return np.stack([z.real, z.imag], axis=-1).reshape(x.shape)


def rms(x, w):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6) * w


def oracle(sd, x, pos, layout, rq):
    B, T, _ = x.shape
    if rq is None:
        q = x @ sd["q_proj.weight"].T
    else:
        a = x @ sd["q_a_proj.weight"].T + sd.get("q_a_proj.bias", 0)
        q = rms(a, sd["q_a_layernorm.weight"]) @ sd["q_b_proj.weight"].T
    q = q.reshape(B, T, H, DN + DR).transpose(0, 2, 1, 3)
    ckv = x @ sd["kv_a_proj_with_mqa.weight"].T + sd.get("kv_a_proj_with_mqa.bias", 0)
    c = rms(ckv[..., :R], sd["kv_a_layernorm.weight"])
    kr = rope_np(ckv[..., R:], pos, layout)
    qr = rope_np(q[..., DN:], pos, layout)
    kv = (c @ sd["kv_b_proj.weight"].T).reshape(B, T, H, DN + DV).transpose(0, 2, 1, 3)
    s = (
        q[..., :DN] @ kv[..., :DN].transpose(0, 1, 3, 2)
        + qr @ kr[:, None].transpose(0, 1, 3, 2)
    ) / np.sqrt(DN + DR)
    s = np.where(np.tril(np.ones((T, T), bool)), s, -np.inf)
    p = np.exp(s - s.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    o = (p @ kv[..., DN:]).transpose(0, 2, 1, 3).reshape(B, T, H * DV)
    return o @ sd["o_proj.weight"].T + sd.get("o_proj.bias", 0)


def model(rq, layout, bias, seed=0):
    m = MLAttention(
        D,
        H,
        rq,
        R,
        DN,
        DR,
        DV,
        RopeSpec(ladder(DR), 1.0, layout, DR),
        attention_bias=bias,
    )
    rng = np.random.Generator(np.random.PCG64(seed))
    sd = {
        k: rng.normal(size=v.shape) * 0.5 + (1.0 if "layernorm" in k else 0.0)
        for k, v in m.state_dict().items()
    }
    for k, p in m.named_parameters():
        p.data = sd[k].copy()
    return m, sd


@pytest.mark.parametrize(
    "rq,layout,bias", [(None, "interleaved", False), (6, "half", True)]
)
def test_matches_numpy_oracle(rq, layout, bias):
    m, sd = model(rq, layout, bias)
    x = np.random.Generator(np.random.PCG64(1)).normal(size=(2, 5, D))
    got = m(Tensor(x, dtype=np.float64), np.arange(5)).data
    np.testing.assert_allclose(
        got, oracle(sd, x, np.arange(5.0), layout, rq), atol=1e-6
    )  # float32 rope tables


def test_absorbed_equals_naive_while_decoding():
    m, _ = model(6, "half", True, seed=2)
    ab = m.absorb_weights()
    x = np.random.Generator(np.random.PCG64(3)).normal(size=(1, 6, D))
    c1, c2 = ConcatLatentCache(), ConcatLatentCache()
    for a, b in [(0, 3), (3, 4), (4, 5), (5, 6)]:
        want = m(Tensor(x[:, a:b], dtype=np.float64), np.arange(a, b), cache=c1).data
        np.testing.assert_allclose(
            ab.forward(x[:, a:b], np.arange(a, b), c2), want, atol=1e-9
        )
    np.testing.assert_allclose(
        want, m(Tensor(x, dtype=np.float64), np.arange(6)).data[:, 5:], atol=1e-9
    )


def test_absorbed_respects_padding_mask():
    m, _ = model(None, "half", False, seed=4)
    x = np.random.Generator(np.random.PCG64(5)).normal(size=(1, 4, D))
    keep = np.array([False, True, True, True])[None, None, None, :]
    want = m(Tensor(x, dtype=np.float64), np.arange(4), mask=keep).data
    got = m.absorb_weights().forward(x, np.arange(4), ConcatLatentCache(), mask=keep)
    np.testing.assert_allclose(got, want, atol=1e-9)
    assert np.allclose(want[0, 0], 0)


def test_gradient_reaches_the_latent_projection():
    m, sd = model(None, "half", False, seed=6)
    x = np.random.Generator(np.random.PCG64(7)).normal(size=(1, 3, D))
    gy = np.random.Generator(np.random.PCG64(8)).normal(size=(1, 3, D))
    F.sum(m(Tensor(x, dtype=np.float64), np.arange(3)) * gy).backward()
    w = m.kv_a_proj_with_mqa.weight
    g = w.grad.copy()
    for idx in [(0, 0), (2, 5), (7, 11)]:
        old = w.data[idx]
        w.data[idx] = old + 1e-6
        hi = np.sum(m(Tensor(x, dtype=np.float64), np.arange(3)).data * gy)
        w.data[idx] = old - 1e-6
        lo = np.sum(m(Tensor(x, dtype=np.float64), np.arange(3)).data * gy)
        w.data[idx] = old
        assert abs((hi - lo) / 2e-6 - g[idx]) < 1e-6


def test_cache_keeps_a_copy():
    cache = ConcatLatentCache()
    c = np.ones((1, 2, R))
    cache.update(0, c, np.ones((1, 2, DR)))
    c[:] = 5
    held, _ = cache.update(0, np.ones((1, 1, R)), np.ones((1, 1, DR)))
    assert held.max() == 1 and cache.seq_len(0) == 3


def test_rope_width_must_match():
    with pytest.raises(ValueError):
        MLAttention(D, H, None, R, DN, DR, DV, RopeSpec(ladder(2), 1.0, "half", 2))


def test_state_dict_names_follow_deepseek():
    m, _ = model(6, "half", True)
    assert list(m.state_dict()) == [
        "q_a_proj.weight",
        "q_a_proj.bias",
        "q_a_layernorm.weight",
        "q_b_proj.weight",
        "kv_a_proj_with_mqa.weight",
        "kv_a_proj_with_mqa.bias",
        "kv_a_layernorm.weight",
        "kv_b_proj.weight",
        "o_proj.weight",
        "o_proj.bias",
    ]
