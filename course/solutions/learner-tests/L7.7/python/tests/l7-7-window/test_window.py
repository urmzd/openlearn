"""My tests for L7.7 (rung R5). Oracles: the visibility rule written as a
double loop, attention with a sink written row by row in numpy float64, and
full attention under the mask as the oracle for the bounded cache. They
import only the contract."""

import math

import numpy as np
import pytest
from tinyllm.autograd.tensor import Tensor
from tinyllm.modern.gqa import ConcatKVCache, GQAttention
from tinyllm.modern.rope import RopeSpec
from tinyllm.modern.window import (
    SinkWindowCache,
    attention_mask,
    sink_window_mask,
    windowed_attention,
)


def rule(Tq, Tk, w, s, off):
    return np.array(
        [
            [j <= off + i and (j < s or off + i - j < w) for j in range(Tk)]
            for i in range(Tq)
        ]
    )


@pytest.mark.parametrize(
    "Tq,Tk,w,s,off",
    [(6, 6, 2, 1, 0), (3, 9, 3, 2, 6), (1, 7, 1, 0, 6), (4, 4, 9, 2, 0)],
)
def test_mask_matches_the_rule(Tq, Tk, w, s, off):
    assert np.array_equal(sink_window_mask(Tq, Tk, w, s, off), rule(Tq, Tk, w, s, off))


def attend_np(q, k, v, vis, sink):
    out = np.zeros(q.shape[:-1] + (v.shape[-1],))
    H, Hkv = q.shape[1], k.shape[1]
    for b in range(q.shape[0]):
        for h in range(H):
            kv = h // (H // Hkv)
            for i in range(q.shape[2]):
                s = [
                    q[b, h, i] @ k[b, kv, j] / math.sqrt(q.shape[-1])
                    for j in range(k.shape[2])
                    if vis[i, j]
                ]
                z = sum(math.exp(x) for x in s) + (
                    math.exp(sink[h]) if sink is not None else 0
                )
                js = [j for j in range(k.shape[2]) if vis[i, j]]
                out[b, h, i] = sum(math.exp(x) / z * v[b, kv, j] for x, j in zip(s, js))
    return out


@pytest.mark.parametrize("sink", [None, np.array([0.3, -0.7, 1.1, 0.0])])
def test_attention_matches_loops(sink):
    rng = np.random.Generator(np.random.PCG64(1))
    q, k, v = (
        rng.normal(size=(1, 4, 3, 5)),
        rng.normal(size=(1, 2, 7, 5)),
        rng.normal(size=(1, 2, 7, 2)),
    )
    out, lse = windowed_attention(
        q, k, v, window=3, n_sink=1, sink_logits=sink, q_offset=4
    )
    np.testing.assert_allclose(
        out, attend_np(q, k, v, rule(3, 7, 3, 1, 4), sink), atol=1e-12
    )


def test_hand_sink_row():
    out, lse = windowed_attention(
        [[[[1.0]]]],
        [[[[0.0], [math.log(2)]]]],
        [[[[4.0], [8.0]]]],
        sink_logits=[0.0],
        q_offset=1,
    )
    assert out[0, 0, 0, 0] == pytest.approx(5.0) and lse[0, 0, 0] == pytest.approx(
        math.log(4)
    )


def model():
    m = GQAttention(
        8, 2, 1, 4, RopeSpec(10000.0 ** (-np.arange(0, 4, 2) / 4), 1.0, "half", 4)
    )
    rng = np.random.Generator(np.random.PCG64(2))
    for _, p in m.named_parameters():
        p.data = rng.normal(size=p.shape) * 0.5
    return m


@pytest.mark.parametrize("chunks", [[1] * 9, [4, 2, 3]])
def test_bounded_cache_equals_full_attention(chunks):
    m = model()
    x = np.random.Generator(np.random.PCG64(3)).normal(size=(1, 9, 8))
    full = m(Tensor(x, dtype=np.float64), np.arange(9), mask=rule(9, 9, 3, 2, 0)).data
    cache, outs, t = SinkWindowCache(2, 3), [], 0
    for n in chunks:
        outs.append(
            m(
                Tensor(x[:, t : t + n], dtype=np.float64),
                np.arange(t, t + n),
                mask=attention_mask(cache, 0, n, 3, 2),
                cache=cache,
            ).data
        )
        t += n
        assert cache.held(0) <= 2 + 3 - 1
    np.testing.assert_allclose(np.concatenate(outs, axis=1), full, atol=1e-10)


def test_full_history_cache_gets_offset_mask():
    m = model()
    x = np.random.Generator(np.random.PCG64(4)).normal(size=(1, 6, 8))
    full = m(Tensor(x, dtype=np.float64), np.arange(6), mask=rule(6, 6, 2, 1, 0)).data
    cache = ConcatKVCache()
    a = m(
        Tensor(x[:, :4], dtype=np.float64),
        np.arange(4),
        mask=attention_mask(cache, 0, 4, 2, 1),
        cache=cache,
    ).data
    b = m(
        Tensor(x[:, 4:], dtype=np.float64),
        np.arange(4, 6),
        mask=attention_mask(cache, 0, 2, 2, 1),
        cache=cache,
    ).data
    np.testing.assert_allclose(np.concatenate([a, b], axis=1), full, atol=1e-10)


def test_positions_are_what_was_returned():
    cache = SinkWindowCache(1, 2)
    for t in range(5):
        cache.update(0, np.zeros((1, 1, 1, 1)), np.zeros((1, 1, 1, 1)))
    assert cache.positions(0).tolist() == [0, 3, 4] and cache.held(0) == 2


def test_rejects_bad_sizes():
    with pytest.raises(ValueError):
        SinkWindowCache(-1, 3)
    with pytest.raises(ValueError):
        windowed_attention(
            np.zeros((1, 3, 1, 2)), np.zeros((1, 2, 1, 2)), np.zeros((1, 2, 1, 2))
        )
