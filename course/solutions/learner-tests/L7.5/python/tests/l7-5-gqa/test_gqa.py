"""My tests for L7.5 (rung R5). The oracle is grouped-query attention written
out in numpy float64 with the module's own weights: projections, RoPE as
complex multiplication (half layout), masked softmax with an optional sink
column, weighted values, output projection. They import only the contract."""

import numpy as np
import pytest
import tinyllm.autograd.functional as F
from tinyllm.autograd.tensor import Tensor
from tinyllm.modern.gqa import ConcatKVCache, GQAttention, repeat_kv
from tinyllm.modern.rope import RopeSpec

D, H, HKV, DH = 12, 6, 2, 4


def inv():
    return 1e4 ** (-np.arange(0, DH, 2) / DH)


def rot(x, pos):
    ang = pos[:, None] * inv()
    z = (x[..., : DH // 2] + 1j * x[..., DH // 2 :]) * np.exp(1j * ang)
    return np.concatenate([z.real, z.imag], axis=-1)


def oracle(a, x, pos, window=None):
    sd = {k: v.astype(np.float64) for k, v in a.state_dict().items()}
    B, T, _ = x.shape
    q = (x @ sd["q_proj.weight"].T).reshape(B, T, H, DH).transpose(0, 2, 1, 3)
    k = (x @ sd["k_proj.weight"].T).reshape(B, T, HKV, DH).transpose(0, 2, 1, 3)
    v = (x @ sd["v_proj.weight"].T).reshape(B, T, HKV, DH).transpose(0, 2, 1, 3)
    q, k = rot(q, pos), rot(k, pos)
    out = np.zeros((B, H, T, DH))
    for h in range(H):
        kv = h // (H // HKV)
        s = q[:, h] @ k[:, kv].transpose(0, 2, 1) / np.sqrt(DH)
        i, j = np.arange(T)[:, None], np.arange(T)[None, :]
        vis = (j <= i) & (((i - j) < window) if window else True)
        s = np.where(vis, s, -np.inf)
        if "sinks" in sd:
            s = np.concatenate([s, np.full((B, T, 1), sd["sinks"][h])], axis=-1)
        e = np.exp(s - s.max(-1, keepdims=True))
        p = (e / e.sum(-1, keepdims=True))[..., :T]
        out[:, h] = p @ v[:, kv]
    return out.transpose(0, 2, 1, 3).reshape(B, T, H * DH) @ sd["o_proj.weight"].T


def make(**kw):
    a = GQAttention(D, H, HKV, DH, RopeSpec(inv(), 1.0, "half", DH), **kw)
    r = np.random.Generator(np.random.PCG64(7))
    for p in a.parameters():
        p.data = r.normal(scale=0.5, size=p.data.shape).astype(np.float32)
    return a


@pytest.mark.parametrize("kw", [{}, {"window": 2}, {"sinks": True}, {"window": 3, "sinks": True}])
def test_matches_the_formula(kw):
    a = make(**kw)
    x = np.random.Generator(np.random.PCG64(1)).normal(size=(2, 5, D))
    pos = np.arange(5)
    np.testing.assert_allclose(a(Tensor(x), pos).data, oracle(a, x, pos, kw.get("window")), rtol=1e-4, atol=1e-5)


def test_decode_with_cache_matches_full():
    a = make(window=3)
    x = np.random.Generator(np.random.PCG64(2)).normal(size=(1, 6, D))
    full = a(Tensor(x), np.arange(6)).data
    c = ConcatKVCache()
    out = [a(Tensor(x[:, :2]), np.arange(2), cache=c).data]
    out += [a(Tensor(x[:, t:t + 1]), np.array([t]), cache=c).data for t in range(2, 6)]
    np.testing.assert_allclose(np.concatenate(out, 1), full, rtol=1e-4, atol=1e-5)


def test_repeat_kv_order():
    x = Tensor(np.arange(2.0).reshape(1, 2, 1, 1))
    assert repeat_kv(x, 3).data.ravel().tolist() == [0, 0, 0, 1, 1, 1]


def test_sinks_gradient():
    a = make(sinks=True)
    x = np.random.Generator(np.random.PCG64(3)).normal(size=(1, 4, D))
    F.sum(a(Tensor(x), np.arange(4))).backward()
    assert np.abs(a.sinks.grad).max() > 0


def test_extra_mask_hides_keys():
    a = make()
    x = np.random.Generator(np.random.PCG64(4)).normal(size=(1, 4, D))
    m = np.ones((1, 1, 4, 4), dtype=bool)
    m[..., 1] = False
    x2 = x.copy()
    x2[:, 1] += 3.0
    y1, y2 = a(Tensor(x), np.arange(4), m).data, a(Tensor(x2), np.arange(4), m).data
    np.testing.assert_allclose(y1[:, 2:], y2[:, 2:], rtol=1e-6)


def test_per_row_positions():
    a = make()
    x = np.random.Generator(np.random.PCG64(5)).normal(size=(2, 4, D))
    rows = np.array([[0, 1, 2, 3], [0, 3, 4, 9]])
    y = a(Tensor(x), rows).data
    np.testing.assert_allclose(y[1], a(Tensor(x[1:]), rows[1]).data[0], rtol=1e-5, atol=1e-6)


def test_names_and_defaults():
    a = GQAttention(D, H, HKV, None, RopeSpec(1e4 ** -np.arange(0, 2, 2), 1.0, "half", 2), qkv_bias=True)
    names = [n for n, _ in a.named_parameters()]
    assert "o_proj.bias" not in names and a.d_head == D // H


def test_cache_keeps_kv_heads():
    a = make()
    c = ConcatKVCache()
    a(Tensor(np.ones((1, 3, D))), np.arange(3), cache=c)
    k, _ = c.update(0, np.zeros((1, HKV, 1, DH)), np.zeros((1, HKV, 1, DH)))
    assert k.shape == (1, HKV, 4, DH)


def test_attention_scaling():
    x = np.random.Generator(np.random.PCG64(6)).normal(size=(1, 4, D))
    a = make()
    b = GQAttention(D, H, HKV, DH, RopeSpec(inv(), 2.0, "half", DH))
    b.load_state_dict(a.state_dict())
    assert not np.allclose(a(Tensor(x), np.arange(4)).data, b(Tensor(x), np.arange(4)).data)


def test_cache_copies_and_checks():
    c = ConcatKVCache()
    k = np.ones((1, 1, 2, 2))
    c.update(0, k, k)
    k[:] = 5
    assert c.update(0, k[:, :, :1], k[:, :, :1])[0][0, 0, 0, 0] == 1
    with pytest.raises(ValueError):
        c.update(1, np.ones((1, 1, 1, 2)), np.ones((1, 1, 1, 3)))


@pytest.mark.parametrize("kw", [dict(n_kv_heads=4), dict(window=0)])
def test_rejects(kw):
    args = dict(d=D, n_heads=H, n_kv_heads=HKV, d_head=DH, rope=RopeSpec(inv(), 1.0, "half", DH))
    args.update(kw)
    with pytest.raises(ValueError):
        GQAttention(**args)
