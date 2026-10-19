"""My tests for L8.2 (rung R5). The oracle for the cache is the same model
run without it (full recompute); the oracle for the detokenizer is
decode(all ids); the oracle for seeded generation is L8.1's sample replayed
by hand. My model is a one-layer attention LM with rotary positions written
here in numpy. They import only the contract."""

import numpy as np
import pytest
from tinyllm.infer.generate import IncrementalDecoder, cache_dims, generate
from tinyllm.infer.kvcache import KVCache, LatentCache
from tinyllm.infer.sample import SamplingParams, request_rng, sample, sampled_entropy
from tinyllm.num.rng import PCG32
from tinyllm.tok.bpe import BPETokenizer

TOK = BPETokenizer.train(
    [
        "héllo wörld " * 8,
        "日本語のテキスト " * 8,
        "🙂 smile " * 8,
        "plain text here " * 8,
    ],
    vocab_size=320,
)
B2U = {}
_bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
_cs, _n = _bs[:], 0
for _b in range(256):
    if _b not in _bs:
        _bs.append(_b)
        _cs.append(256 + _n)
        _n += 1
B2U = dict(zip(_bs, map(chr, _cs)))


def byte_id(b):
    return TOK.token_to_id(B2U[b])


class LM:
    n_layers, n_kv_heads, d_head = 2, 1, 4

    def __init__(self, vocab, max_len=64, seed=3):
        g = PCG32(seed, 9)
        r = lambda *s: (
            np.array([g.uniform() for _ in range(int(np.prod(s)))]).reshape(s) - 0.5
        ).astype(np.float32)
        self.max_len, self.E = max_len, r(vocab, 8) * 2
        self.W = [(r(8, 4), r(8, 4), r(8, 4), r(4, 8)) for _ in range(self.n_layers)]

    def forward(self, ids, positions=None, cache=None):
        ids = np.asarray(ids)
        T = ids.shape[1]
        pos = np.asarray(
            positions if positions is not None else np.arange(T), dtype=np.float32
        )
        mask = cache.mask(T) if cache is not None else np.tril(np.ones((T, T), bool))
        x = self.E[ids]
        for li, (wq, wk, wv, wo) in enumerate(self.W):
            q, k, v = x @ wq, x @ wk, x @ wv
            c, s = np.cos(pos * 0.7)[:, None], np.sin(pos * 0.7)[:, None]
            rot = lambda z: np.concatenate(
                [z[..., :2] * c - z[..., 2:] * s, z[..., :2] * s + z[..., 2:] * c], -1
            )
            q, k = rot(q)[:, None], rot(k)[:, None]
            v = v[:, None]
            if cache is not None:
                k, v = cache.update(li, k, v)
            a = np.where(mask, q @ np.swapaxes(k, -1, -2) / 2, -np.inf)
            a = np.exp(a - a.max(-1, keepdims=True))
            x = x + ((a / a.sum(-1, keepdims=True)) @ v)[:, 0] @ wo
        return (x @ self.E.T).astype(np.float32)


class Script:
    n_layers, n_kv_heads, d_head = 1, 1, 1

    def __init__(self, ids, max_len=200):
        self.ids, self.max_len, self.V = ids, max_len, TOK.vocab_size

    def forward(self, ids, positions=None, cache=None):
        T = np.asarray(ids).shape[1]
        pos = np.asarray(positions if positions is not None else np.arange(T))
        if cache is not None:
            cache.update(0, np.zeros((1, 1, T, 1)), np.zeros((1, 1, T, 1)))
        out = np.zeros((1, T, self.V), np.float32)
        for t, j in enumerate(pos.tolist()):
            if j + 1 < len(self.ids):
                out[0, t, self.ids[j + 1]] = 9.0
        return out


class Rec:
    def __init__(self, m):
        self.m, self.rows = m, []
        self.n_layers, self.n_kv_heads, self.d_head, self.max_len = (
            m.n_layers,
            m.n_kv_heads,
            m.d_head,
            m.max_len,
        )

    def forward(self, ids, positions=None, cache=None):
        out = self.m.forward(ids, positions, cache)
        self.rows.append(out[0, -1].copy())
        return out


def test_cache_equals_recompute_each_step():
    a, b = Rec(LM(TOK.vocab_size)), Rec(LM(TOK.vocab_size))
    ga = generate(
        a, TOK, "héllo", SamplingParams(temperature=0.0, max_tokens=12), cache="none"
    )
    gb = generate(b, TOK, "héllo", SamplingParams(temperature=0.0, max_tokens=12))
    assert ga.ids == gb.ids
    for x, y in zip(a.rows, b.rows):
        assert np.abs(x - y).max() < 1e-5


def test_chunks_and_commit_rules():
    m = LM(50)
    ids = [3, 9, 1, 4, 4, 7, 2]
    full = m.forward(np.array([ids]))[0]
    c = KVCache(2, 1, 4, 7)
    out = [m.forward(np.array([ids[:4]]), c.positions(4), c)[0]]
    assert c.seq_len() == 4
    for i in range(4, 7):
        out.append(m.forward(np.array([[ids[i]]]), c.positions(1), c)[0])
    assert np.abs(np.concatenate(out) - full).max() < 1e-5
    d = KVCache(2, 1, 1, 9)
    d.update(0, np.ones((1, 1, 2, 1)), np.ones((1, 1, 2, 1)))
    assert d.seq_len() == 0 and d.positions(1).tolist() == [0]
    assert d.mask(2).shape == (2, 2)


def test_update_contract():
    c = KVCache(1, 2, 2, 5)
    x = np.arange(8, dtype=np.float32).reshape(1, 2, 2, 2)
    k, v = c.update(0, x, x + 1)
    x[:] = -1
    k, v = c.update(0, np.zeros((1, 2, 2, 2)), np.zeros((1, 2, 2, 2)))
    assert k.shape == (1, 2, 4, 2) and k[0, 0, 0].tolist() == [0.0, 1.0]
    assert c.seq_len() == 4
    for bad in (np.zeros((1, 1, 1, 2)), np.zeros((1, 2, 0, 2)), np.zeros((2, 2, 1, 2))):
        with pytest.raises(ValueError):
            c.update(0, bad, bad)
    with pytest.raises(ValueError):
        c.update(1, np.zeros((1, 2, 1, 2)), np.zeros((1, 2, 1, 2)))
    c.update(0, np.zeros((1, 2, 1, 2)), np.zeros((1, 2, 1, 2)))  # fills it exactly
    with pytest.raises(ValueError):
        c.update(0, np.zeros((1, 2, 1, 2)), np.zeros((1, 2, 1, 2)))  # full


def test_truncate_and_float16():
    c = KVCache(2, 1, 1, 6, dtype=np.float16)
    for layer in (0, 1):
        c.update(layer, np.full((1, 1, 3, 1), 0.1, np.float32), np.zeros((1, 1, 3, 1)))
    c.truncate(1)
    for layer in (0, 1):
        k, _ = c.update(
            layer, np.full((1, 1, 1, 1), 0.3, np.float32), np.zeros((1, 1, 1, 1))
        )
        assert k.shape == (1, 1, 2, 1) and k[0, 0, 1, 0] == np.float32(np.float16(0.3))
        assert k[0, 0, 0, 0] == np.float32(np.float16(0.1))
    with pytest.raises(ValueError):
        c.truncate(5)
    lc = LatentCache(1, 3, 1, 4)
    a, b = lc.update(0, np.ones((1, 2, 3)), np.ones((1, 2, 1)))
    assert a.shape == (1, 2, 3) and lc.seq_len() == 2
    with pytest.raises(ValueError):
        c.positions(0)


def test_detokenizer_streams_only_whole_characters():
    for text in ["héllo wörld", "日本語", "🙂🙂", "plain"]:
        ids = [byte_id(b) for b in text.encode()]
        d = IncrementalDecoder(TOK)
        out = [d.push(i) for i in ids]
        assert "".join(out) + d.flush() == text
        assert all("�" not in o for o in out)
    d = IncrementalDecoder(TOK)
    assert d.push(byte_id(0xE6)) == "" and d.flush() == "�" and d.flush() == ""
    g = PCG32(5, 5)
    for _ in range(100):
        ids = [g.below(TOK.vocab_size) for _ in range(12)]
        d = IncrementalDecoder(TOK)
        assert "".join(d.push(i) for i in ids) + d.flush() == TOK.decode(ids)


def test_generate_stops_and_streams():
    prompt = TOK.encode("plain ")
    cont = TOK.encode("text") + TOK.encode(" here") + TOK.encode(" héllo")
    m = Script(prompt + cont)
    seen = []
    g = generate(
        m,
        TOK,
        "plain ",
        SamplingParams(temperature=0.0, max_tokens=len(cont), stop=["here!"]),
        on_text=seen.append,
    )
    assert g.text == "text here héllo" == "".join(seen) and g.ids == cont
    acc = ""
    for s in seen[:-1]:
        acc += s
        assert not any(acc.endswith("here!"[:n]) for n in range(1, 5))
    g = generate(
        m, TOK, "plain ", SamplingParams(temperature=0.0, max_tokens=40, stop=[" he"])
    )
    assert g.text == "text" and g.finish_reason == "stop"
    eos = cont[1]
    g = generate(
        m, TOK, "plain ", SamplingParams(temperature=0.0, max_tokens=40), eos_ids=[eos]
    )
    assert g.ids == cont[:1] and g.finish_reason == "stop"
    short = Script(prompt + cont, max_len=len(prompt) + 2)
    assert (
        len(
            generate(
                short, TOK, "plain ", SamplingParams(temperature=0.0, max_tokens=40)
            ).ids
        )
        == 2
    )
    with pytest.raises(ValueError):
        generate(m, TOK, "", SamplingParams(), cache="none")


def test_seeded_generation_replays_l81():
    m = LM(TOK.vocab_size, seed=8)
    p = SamplingParams(
        temperature=1.1,
        repetition_penalty=1.4,
        presence_penalty=0.5,
        seed=21,
        max_tokens=10,
    )
    g = generate(m, TOK, "日本", p)
    prompt, ids, rng, ent = TOK.encode("日本"), [], request_rng(21), []
    for _ in range(10):
        logits = m.forward(np.array([prompt + ids]))[0, -1]
        ent.append(sampled_entropy(logits, p, ids, prompt))
        ids.append(sample(logits, p, ids, rng, prompt=prompt)[0])
    assert g.ids == ids
    assert abs(g.stats["mean_entropy"] - np.mean(ent)) < 1e-6


def test_cache_objects_and_modes():
    m = LM(TOK.vocab_size)
    c = KVCache(2, 1, 4, 40)
    g = generate(
        m, TOK, "héllo", SamplingParams(temperature=0.0, max_tokens=4), cache=c
    )
    assert c.seq_len() == len(TOK.encode("héllo")) + 3
    with pytest.raises(ValueError):
        generate(m, TOK, "héllo", SamplingParams(max_tokens=2), cache="paged")

    class Cfg:
        num_hidden_layers, num_key_value_heads, head_dim, max_position_embeddings = (
            4,
            2,
            16,
            99,
        )

    class M:
        cfg = Cfg()

    assert cache_dims(M()) == (4, 2, 16, 99)
