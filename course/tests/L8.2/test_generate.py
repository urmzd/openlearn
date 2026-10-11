"""Course tests for L8.2: the KV cache, incremental decode, the incremental
UTF-8 detokenizer, and generate (tinyllm/infer/kvcache.py, generate.py).

Rung R0 for the course suite: read these before you write code. Each test
names why it exists (WHY), what kind of check it is (KIND), the planted bugs
it kills (CATCHES, mutants in course/mutants/L8.2), and the chapter section
it comes from. You write your own graded tests too (rung R5, see the
chapter).

The models are test helpers in _tinylm.py (a two-layer GQA decoder with
RoPE, and a scripted model), so a failure here is about the cache and the
loop, never about your L7 modules. One test runs L7.5's attention through
your cache, the seam L7.9's model uses. The tokenizer is your L1.2
byte-level BPE trained on text with multi-byte characters, so tokens end in
the middle of characters.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32
from _tinylm import ScriptLM, TinyLM
from tinyllm.autograd.tensor import Tensor
from tinyllm.infer.generate import IncrementalDecoder, cache_dims, generate
from tinyllm.infer.kvcache import KVCache, LatentCache
from tinyllm.infer.sample import SamplingParams, request_rng, sample, sampled_entropy
from tinyllm.modern.gqa import GQAttention
from tinyllm.modern.llama import LlamaConfig, LlamaForCausalLM
from tinyllm.modern.rope import RopeSpec
from tinyllm.tok.bpe import BPETokenizer
from tinyllm.xfmr.masks import causal_mask

CORPUS = [
    "naïve café crème brûlée " * 6,
    "日本語のテキストと漢字 " * 6,
    "emoji 🙂🙃 and ✓ marks " * 6,
    "plain ascii words here " * 4,
]
_TOK = None


def seed() -> int:
    return int(os.environ.get("OL_SEED", "0"))


def tok() -> BPETokenizer:
    global _TOK
    if _TOK is None:
        _TOK = BPETokenizer.train(CORPUS, vocab_size=330, min_freq=2)
    return _TOK


def gpt2_byte_chars() -> dict[int, str]:
    """GPT-2's byte -> character map (written out here, independent of M05.2):
    printable Latin-1 bytes map to themselves, the rest to 256 + n."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def byte_ids(data: bytes) -> list[int]:
    """The single-byte token id of each byte (byte-level BPE has all 256)."""
    m = gpt2_byte_chars()
    return [tok().token_to_id(m[b]) for b in data]


def greedy(**kw) -> SamplingParams:
    return SamplingParams(temperature=0.0, **kw)


class Recorder:
    """Wraps a model and keeps the last-position logits of every forward."""

    def __init__(self, model):
        self.model, self.last = model, []
        for n in ("n_layers", "n_kv_heads", "d_head", "max_len"):
            if hasattr(model, n):
                setattr(self, n, getattr(model, n))
        if hasattr(model, "config"):
            self.config = model.config

    def forward(self, ids, positions=None, cache=None):
        out = self.model.forward(ids, positions=positions, cache=cache)
        self.last.append(np.asarray(getattr(out, "data", out))[0, -1].copy())
        return out


# --- the worked example -------------------------------------------------------


def test_hand_example():
    # WHY: section 3 by hand: one layer, one kv head of width 2, max_len 4.
    #      A prefill chunk of 2 positions comes back as [1, 1, 2, 2]; a decode
    #      chunk of 1 comes back as all 3 positions held; the committed length
    #      is 3, the next position is 3, and its mask row sees keys 0..3.
    # KIND: unit, smoke
    # CATCHES: s04, s05
    # CHAPTER: L8.2 section 3
    c = KVCache(1, 1, 2, 4)
    k0 = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
    k, v = c.update(0, k0, -k0)
    assert k.shape == (1, 1, 2, 2) and k.dtype == np.float32
    assert k.tolist() == k0.tolist() and v.tolist() == (-k0).tolist()
    assert c.seq_len() == 2 and c.positions(1).tolist() == [2]
    k1 = np.array([[[[5.0, 6.0]]]], dtype=np.float32)
    k, v = c.update(0, k1, k1)
    assert k.tolist() == [[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]]
    assert c.seq_len() == 3 and c.positions(1).tolist() == [3]
    assert c.mask(1).tolist() == [[True, True, True, True]]
    assert c.nbytes() == 2 * 1 * 1 * 1 * 4 * 2 * 4


def test_hand_example_detokenizer():
    # WHY: section 3's stream: "é" is the two bytes C3 A9. Pushed one byte
    #      token at a time, the first push must emit nothing (decoding C3 alone
    #      gives U+FFFD) and the second emits "é"; then "x" comes out at once.
    # KIND: unit, smoke
    # CATCHES: s10, s11
    # CHAPTER: L8.2 section 3
    t = tok()
    c3, a9, x = byte_ids("éx".encode("utf-8"))
    assert t.decode([c3]) == "\ufffd" and t.decode([c3, a9]) == "é"
    d = IncrementalDecoder(t)
    assert d.push(c3) == ""
    assert d.push(a9) == "é"
    assert d.push(x) == "x"
    assert d.flush() == ""


# --- the cache --------------------------------------------------------------------


def test_cached_logits_equal_full_recompute():
    # WHY: the whole point of the cache: at every one of 16 decode steps the
    #      last-position logits with the cache equal a full recompute of the
    #      sequence (atol 1e-5), and greedy ids agree. A wrong position, a
    #      stale key, or a mask that hides the newest key shows up here.
    # KIND: differential
    # CATCHES: s01, s04, s05, s06, s13
    # CHAPTER: L8.2 section 2.2
    t = tok()
    m1, m2 = (
        Recorder(TinyLM(t.vocab_size, seed=1)),
        Recorder(TinyLM(t.vocab_size, seed=1)),
    )
    a = generate(m1, t, "naïve café", greedy(max_tokens=16), cache="none")
    b = generate(m2, t, "naïve café", greedy(max_tokens=16), cache="contiguous")
    assert a.ids == b.ids and len(a.ids) == 16
    assert len(m1.last) == len(m2.last) == 16
    for x, y in zip(m1.last, m2.last):
        assert_close(y, x, rtol=0, atol=1e-5)
    # prefill once over the prompt, then one token per step
    n = len(t.encode("naïve café"))
    assert m2.model.calls[0] == (n, list(range(n)))
    assert m2.model.calls[1:] == [(1, [n + i]) for i in range(15)]


def test_chunked_prefill_equals_whole():
    # WHY: a prompt fed in chunks of 3, 1, 5, and 2 positions through the
    #      cache gives the same last logits as one forward over all 11 (atol
    #      1e-5): each chunk's queries sit at the right absolute positions and
    #      see exactly the keys before them.
    # KIND: differential
    # CATCHES: s04, s05
    # CHAPTER: L8.2 section 2.2
    m = TinyLM(64, seed=2)
    rng = PCG32(seed(), 21)
    ids = [rng.below(64) for _ in range(11)]
    whole = m.forward(np.array([ids]))[0]
    c = KVCache(m.n_layers, m.n_kv_heads, m.d_head, 16)
    pos = 0
    for n in (3, 1, 5, 2):
        out = m.forward(np.array([ids[pos : pos + n]]), cache=c)[0]
        assert_close(out, whole[pos : pos + n], rtol=0, atol=1e-5)
        pos += n
    assert c.seq_len() == 11


def test_l75_attention_through_the_cache():
    # WHY: the seam L7.9's model uses: L7.5's grouped-query attention with
    #      your KVCache as its hook, fed one token at a time after a prefill
    #      of 3, equals the full forward (atol 1e-5).
    # KIND: differential
    # CATCHES: s05, s08
    # CHAPTER: L8.2 section 4
    rope = RopeSpec(
        inv_freq=(10000.0 ** (-np.arange(0, 8, 2) / 8)).astype(np.float32),
        attention_scaling=1.0,
        layout="half",
        rotary_dim=8,
    )
    attn = GQAttention(16, 4, 2, 8, rope, rng=PCG32(seed(), 22))
    x = PCG32(seed(), 23).normal_array((1, 7, 16)).astype(np.float32)
    full = attn(Tensor(x), np.arange(7)).data
    c = KVCache(5, 2, 8, 7)
    parts = [attn(Tensor(x[:, :3]), c.positions(3), cache=c, layer=4).data]
    for t in range(3, 7):
        parts.append(
            attn(Tensor(x[:, t : t + 1]), np.array([t]), cache=c, layer=4).data
        )
    assert_close(np.concatenate(parts, axis=1), full, rtol=0, atol=1e-5)


def test_llama_generates_the_same_with_and_without_cache():
    # WHY: the model generate drives in MS-L8 is L7.9's Llama: a tiny random
    #      LlamaForCausalLM (GQA, 2 layers) generates the same greedy ids
    #      through your KVCache as by full recompute, with every step's logits
    #      within 1e-5, and the cache ends holding prompt + new - 1 positions.
    # KIND: differential
    # CATCHES: s01, s05, s06, s13, s24
    # CHAPTER: L8.2 section 2.2
    t = tok()
    cfg = LlamaConfig(
        vocab_size=t.vocab_size,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        rms_norm_eps=1e-5,
        tie_word_embeddings=True,
        max_position_embeddings=64,
    )
    m = LlamaForCausalLM(cfg, rng=PCG32(seed(), 27))
    assert cache_dims(m) == (2, 2, 4, 64)
    a, b = Recorder(m), Recorder(m)
    ga = generate(a, t, "naïve", greedy(max_tokens=8), cache="none")
    c = KVCache(2, 2, 4, 64)
    gb = generate(b, t, "naïve", greedy(max_tokens=8), cache=c)
    assert ga.ids == gb.ids
    for x, y in zip(a.last, b.last):
        assert_close(y, x, rtol=0, atol=1e-5)
    assert c.seq_len() == len(t.encode("naïve")) + 7


def test_update_copies_and_checks():
    # WHY: update must copy what it is given (a caller that reuses its buffer
    #      must not change the cache), keep earlier positions intact, and
    #      refuse a wrong shape or an append past max_len without changing
    #      anything.
    # KIND: boundary
    # CATCHES: s05, s07, s08, m01, m02
    # CHAPTER: L8.2 section 4
    c = KVCache(2, 2, 3, 5, batch=2)
    buf = np.ones((2, 2, 2, 3), dtype=np.float32)
    c.update(0, buf, buf)
    buf[:] = 7.0
    k, _ = c.update(0, buf[:, :, :1], buf[:, :, :1])
    assert (k[:, :, :2] == 1.0).all() and (k[:, :, 2] == 7.0).all()
    for bad in [
        np.ones((1, 2, 1, 3)),
        np.ones((2, 1, 1, 3)),
        np.ones((2, 2, 1, 4)),
        np.ones((2, 2, 0, 3)),
        np.ones((2, 2, 3)),
    ]:
        with pytest.raises(ValueError):
            c.update(1, bad, bad)
    with pytest.raises(ValueError):
        c.update(0, np.ones((2, 2, 3, 3)), np.ones((2, 2, 3, 3)))  # 3 + 3 > max_len 5
    with pytest.raises(ValueError):
        c.update(2, buf, buf)
    k, _ = c.update(
        0, np.zeros((2, 2, 2, 3)), np.zeros((2, 2, 2, 3))
    )  # exactly fills it
    assert k.shape == (2, 2, 5, 3)
    with pytest.raises(ValueError):
        KVCache(0, 1, 1, 1)
    with pytest.raises(ValueError):
        KVCache(1, 1, 1, 1, dtype=np.int32)


def test_seq_len_commits_after_every_layer():
    # WHY: seq_len() is the minimum over layers: while a forward pass has
    #      appended to layer 0 but not yet to layer 1, positions() and mask()
    #      must still describe the same chunk for layer 1; seq_len(layer) is
    #      that layer's own count, which L7.7's mask and L7.9 read.
    # KIND: boundary
    # CATCHES: s02, s03
    # CHAPTER: L8.2 section 2.3
    c = KVCache(2, 1, 2, 8)
    z = np.zeros((1, 1, 3, 2), dtype=np.float32)
    c.update(0, z, z)
    assert c.seq_len() == 0 and c.positions(3).tolist() == [0, 1, 2]
    assert c.seq_len(0) == 3 and c.seq_len(1) == 0  # one layer's own count
    c.update(1, z, z)
    assert c.seq_len() == 3 and c.positions(2).tolist() == [3, 4]


def test_mask_is_l52_causal_with_offset():
    # WHY: a decode chunk at offset s attends to every cached key and to the
    #      chunk's own past: L5.2's causal_mask(T, s + T, q_offset=s), the
    #      bottom-right-aligned triangle.
    # KIND: unit
    # CATCHES: s04, m03
    # CHAPTER: L8.2 section 2.3
    c = KVCache(1, 1, 2, 9)
    z = np.zeros((1, 1, 4, 2), dtype=np.float32)
    c.update(0, z, z)
    for t in (1, 3):
        assert np.array_equal(c.mask(t), causal_mask(t, 4 + t, q_offset=4))
    assert c.mask(2).tolist() == [[True] * 5 + [False], [True] * 6]
    for bad in (0, -1):
        with pytest.raises(ValueError):
            c.mask(bad)
        with pytest.raises(ValueError):
            c.positions(bad)


def test_truncate_rolls_back():
    # WHY: speculative decoding (L8.6) appends draft positions and rolls them
    #      back when the target rejects them; after truncate(n), new appends
    #      overwrite from n and the result equals a cache that never saw the
    #      rejected positions.
    # KIND: property
    # CATCHES: s09, m04
    # CHAPTER: L8.2 section 2.4
    rng = PCG32(seed(), 24)
    a, b = KVCache(2, 2, 4, 12), KVCache(2, 2, 4, 12)
    chunks = [rng.normal_array((1, 2, n, 4)).astype(np.float32) for n in (3, 2, 4)]
    for layer in range(2):
        a.update(layer, chunks[0], chunks[0])
        a.update(layer, chunks[1], chunks[1])
        b.update(layer, chunks[0], chunks[0])
    a.truncate(3)
    assert a.seq_len() == 3
    for layer in range(2):
        ka, va = a.update(layer, chunks[2], -chunks[2])
        kb, vb = b.update(layer, chunks[2], -chunks[2])
        assert np.array_equal(ka, kb) and np.array_equal(va, vb)
    for bad in (-1, 8):
        with pytest.raises(ValueError):
            a.truncate(bad)
    a.truncate(0)
    assert a.seq_len() == 0


def test_float16_cache_rounds_what_it_stores():
    # WHY: a float16 cache halves the bytes and must return what it stores,
    #      the float16-rounded keys, for the new chunk too: L8.3 compares its
    #      paged float16 cache with this one, so both must round identically.
    # KIND: unit
    # CATCHES: s05, s12
    # CHAPTER: L8.2 section 2.5
    rng = PCG32(seed(), 25)
    c16, c32 = KVCache(1, 2, 4, 6, dtype=np.float16), KVCache(1, 2, 4, 6)
    assert c16.nbytes() * 2 == c32.nbytes()
    x = (rng.normal_array((1, 2, 3, 4)) * 3).astype(np.float32)
    k, v = c16.update(0, x, x)
    assert k.dtype == np.float32
    assert np.array_equal(k, x.astype(np.float16).astype(np.float32))
    k, _ = c16.update(0, x[:, :, :1] + 1e-3, x[:, :, :1])
    assert np.array_equal(
        k[:, :, 3],
        (x[:, :, 0] + np.float32(1e-3)).astype(np.float16).astype(np.float32),
    )


def test_float16_generation_stays_close():
    # WHY: generating with a float16 cache keeps every step's logits within
    #      the rounding of the stored keys (atol 2e-2 on this model) and the
    #      greedy ids of the float32 run.
    # KIND: differential
    # CATCHES: s05, s06
    # CHAPTER: L8.2 section 2.5
    t = tok()
    m1, m2 = (
        Recorder(TinyLM(t.vocab_size, seed=3)),
        Recorder(TinyLM(t.vocab_size, seed=3)),
    )
    a = generate(m1, t, "crème", greedy(max_tokens=10), cache="contiguous")
    b = generate(
        m2, t, "crème", greedy(max_tokens=10), cache="contiguous", kv_dtype=np.float16
    )
    assert a.ids == b.ids
    for x, y in zip(m1.last, m2.last):
        assert_close(y, x, rtol=0, atol=2e-2)
    assert any(np.abs(x - y).max() > 0 for x, y in zip(m1.last[1:], m2.last[1:]))


def test_latent_cache():
    # WHY: MLA (L7.6) caches one latent [B, T, r] and one RoPE key [B, T, dr]
    #      per position instead of per-head K and V; the same append, commit,
    #      and rollback rules apply, with far fewer bytes.
    # KIND: unit
    # CATCHES: s04, s05
    # CHAPTER: L8.2 section 2.6
    c = LatentCache(2, 6, 2, 10)
    assert c.nbytes() == 2 * 1 * 10 * (6 + 2) * 4
    cn, kr = np.ones((1, 3, 6), np.float32), np.full((1, 3, 2), 2.0, np.float32)
    for layer in range(2):
        a, b = c.update(layer, cn, kr)
    assert a.shape == (1, 3, 6) and b.shape == (1, 3, 2) and c.seq_len() == 3
    a, b = c.update(0, 3 * cn[:, :1], kr[:, :1])
    assert a.shape == (1, 4, 6) and (a[:, 3] == 3.0).all() and c.seq_len() == 3
    with pytest.raises(ValueError):
        c.update(1, np.ones((1, 1, 5)), kr[:, :1])
    c.truncate(1)
    assert c.positions(2).tolist() == [1, 2]
    assert np.array_equal(c.mask(1), causal_mask(1, 2, q_offset=1))


# --- the detokenizer ---------------------------------------------------------------------


def test_incremental_decode_concat_equals_decode():
    # WHY: the invariant of streaming: whatever the token boundaries, the
    #      pushed pieces plus the final flush equal decode(all ids), and every
    #      piece is final (a prefix of the full decode, never a U+FFFD that a
    #      later byte would have completed). Checked on the corpus encodings
    #      and on 300 random id sequences.
    # KIND: property
    # CATCHES: s10, s11
    # CHAPTER: L8.2 section 2.7
    t = tok()
    rng = PCG32(seed(), 26)
    seqs = [t.encode(s) for s in CORPUS] + [
        [rng.below(t.vocab_size) for _ in range(1 + rng.below(30))] for _ in range(300)
    ]
    for ids in seqs:
        d = IncrementalDecoder(t)
        full = t.decode(ids)
        got = ""
        for i in ids:
            got += d.push(i)
            assert full.startswith(got)
        got += d.flush()
        assert got == full


def test_detokenizer_waits_for_whole_characters():
    # WHY: a 4-byte emoji pushed as four byte tokens emits nothing until the
    #      last byte arrives, then the whole character; a byte that can never
    #      complete (a lone continuation byte) is emitted as U+FFFD once the
    #      next character proves it is invalid, and a sequence cut off at the
    #      end comes out of flush() as U+FFFD.
    # KIND: boundary
    # CATCHES: s10, s11, s14
    # CHAPTER: L8.2 section 2.7
    t = tok()
    d = IncrementalDecoder(t)
    assert [d.push(i) for i in byte_ids("🙂".encode("utf-8"))] == ["", "", "", "🙂"]
    assert d.flush() == ""
    d = IncrementalDecoder(t)
    lone, a = byte_ids(b"\xa9A")
    assert d.push(lone) == ""
    assert d.push(a) == "\ufffdA"
    assert d.flush() == ""
    d = IncrementalDecoder(t)
    assert [d.push(i) for i in byte_ids("a日".encode("utf-8")[:-1])] == ["a", "", ""]
    assert d.flush() == "\ufffd"
    assert d.flush() == ""  # flush resets the window


def test_generate_greedy_follows_the_script():
    # WHY: generate end to end with a model whose greedy output is known: the
    #      ids, the text (decode of the ids), one logprob per id, the token
    #      counts, and finish_reason "length" at max_tokens.
    # KIND: unit
    # CATCHES: s01, s06, s13
    # CHAPTER: L8.2 section 4
    t = tok()
    prompt = t.encode("plain ")
    cont = t.encode("ascii words here")
    m = ScriptLM(t.vocab_size, prompt + cont + t.encode(" more"))
    g = generate(m, t, "plain ", greedy(max_tokens=len(cont)))
    assert g.ids == cont and g.text == "ascii words here"
    assert len(g.logprobs) == len(cont) and all(lp <= 0 for lp in g.logprobs)
    assert g.finish_reason == "length"
    assert g.stats["prompt_tokens"] == len(prompt) and g.stats[
        "completion_tokens"
    ] == len(cont)
    assert set(g.timings) >= {"prefill_s", "decode_s", "total_s"}


def test_eos_stops_and_is_not_emitted():
    # WHY: an EOS id ends generation with finish_reason "stop" and is part of
    #      neither the ids nor the text (OpenAI semantics).
    # KIND: unit
    # CATCHES: s01, s06, s13, s16
    # CHAPTER: L8.2 section 2.8
    t = tok()
    prompt = t.encode("plain ")
    eos = t.vocab_size - 1
    m = ScriptLM(t.vocab_size, prompt + t.encode("ascii") + [eos] + t.encode(" words"))
    g = generate(m, t, "plain ", greedy(max_tokens=30), eos_ids=[eos])
    assert g.text == "ascii" and eos not in g.ids and g.finish_reason == "stop"


def test_stop_strings_cut_and_stream_safely():
    # WHY: a stop string ends generation and is cut from the text, even when
    #      it spans tokens; streaming must never send a piece of it: on_text
    #      holds back any suffix that could still become a stop string, and the
    #      pieces it sends concatenate to the final text.
    # KIND: boundary
    # CATCHES: s01, s06, s13, s17, s18, s19
    # CHAPTER: L8.2 section 2.8
    t = tok()
    prompt = t.encode("plain ")
    cont = t.encode("words here café crème ascii plain")
    m = ScriptLM(t.vocab_size, prompt + cont)
    pieces = []
    g = generate(
        m,
        t,
        "plain ",
        greedy(max_tokens=40, stop=["café cr", "zzz"]),
        on_text=pieces.append,
    )
    assert g.text == "words here " and g.finish_reason == "stop"
    assert "".join(pieces) == g.text
    pieces = []
    g = generate(
        m,
        t,
        "plain ",
        greedy(max_tokens=len(cont), stop=["nope"]),
        on_text=pieces.append,
    )
    assert "".join(pieces) == g.text == "words here café crème ascii plain"
    # "here!" never completes, but while the text ends in "here" (4 of its 5
    # characters) that tail must not be sent: every piece but the last ends
    # where no stop string could still begin.
    cont = t.encode("words") + t.encode(" here") + t.encode(" ascii")
    m = ScriptLM(t.vocab_size, prompt + cont)
    pieces = []
    g = generate(
        m,
        t,
        "plain ",
        greedy(max_tokens=len(cont), stop=["here!"]),
        on_text=pieces.append,
    )
    assert "".join(pieces) == g.text == "words here ascii"
    sent = ""
    for piece in pieces[:-1]:
        sent += piece
        assert not any(sent.endswith("here!"[:n]) for n in range(1, 5)), pieces


def test_max_len_limits_generation():
    # WHY: the context cannot grow past the model's max_len: generation stops
    #      with finish_reason "length" when it is full, and a prompt longer
    #      than max_len, or one that encodes to no tokens, is an error.
    # KIND: boundary
    # CATCHES: s06, s15, m05
    # CHAPTER: L8.2 section 4
    t = tok()
    prompt = t.encode("plain ")
    m = ScriptLM(
        t.vocab_size,
        prompt + t.encode("ascii words here and more"),
        max_len=len(prompt) + 4,
    )
    g = generate(m, t, "plain ", greedy(max_tokens=50))
    assert len(g.ids) == 4 and g.finish_reason == "length"
    small = ScriptLM(t.vocab_size, prompt, max_len=len(prompt) - 1)
    with pytest.raises(ValueError):
        generate(small, t, "plain ", greedy())
    with pytest.raises(ValueError):
        generate(m, t, "", greedy(), cache="none")  # nothing to condition on


def test_seeded_sampling_uses_the_request_stream():
    # WHY: a seeded request draws from request_rng(seed) (L8.1), one uniform
    #      per token, with history = the ids generated so far and the prompt
    #      ids passed for the repetition penalty: replaying L8.1's sample by
    #      hand over full recomputes gives the same ids, logprobs, and entropy.
    # KIND: differential
    # CATCHES: s01, s04, s05, s06, s13, s20, s21, s22, s23
    # CHAPTER: L8.2 section 2.1
    t = tok()
    m = TinyLM(t.vocab_size, seed=4)
    p = SamplingParams(
        temperature=0.9,
        top_k=40,
        repetition_penalty=1.3,
        frequency_penalty=0.2,
        seed=11,
        max_tokens=12,
    )
    g = generate(m, t, "日本語", p)
    prompt = t.encode("日本語")
    rng, ids, lps, ents = request_rng(11), [], [], []
    for _ in range(12):
        logits = m.forward(np.array([prompt + ids]))[0, -1]
        ents.append(sampled_entropy(logits, p, ids, prompt))
        tok_id, lp = sample(logits, p, ids, rng, prompt=prompt)
        ids.append(tok_id)
        lps.append(lp)
    assert g.ids == ids
    assert_close(g.logprobs, lps, rtol=0, atol=1e-4)
    assert_close(g.stats["mean_entropy"], float(np.mean(ents)), rtol=1e-3, atol=1e-6)
    assert generate(m, t, "日本語", p).ids == ids  # reproducible
    assert (
        generate(m, t, "日本語", SamplingParams(temperature=0.9, max_tokens=12)).ids
        == generate(
            m, t, "日本語", SamplingParams(temperature=0.9, max_tokens=12, seed=0)
        ).ids
    )


def test_cache_modes_and_dims():
    # WHY: "paged" is L8.3's object, not a string; an unknown mode is an
    #      error; a cache object passed in is used as is (L8.3's hook);
    #      cache_dims reads a model's attributes or L7.9's HF config names.
    # KIND: boundary
    # CATCHES: s05, s06, s24, s25
    # CHAPTER: L8.2 section 4
    t = tok()
    m = TinyLM(t.vocab_size, seed=5)
    for bad in ("paged", "ring"):
        with pytest.raises(ValueError):
            generate(m, t, "café", greedy(max_tokens=2), cache=bad)
    c = KVCache(2, 2, 8, 64)
    g = generate(m, t, "café", greedy(max_tokens=5), cache=c)
    assert c.seq_len() == len(t.encode("café")) + 4
    assert g.ids == generate(m, t, "café", greedy(max_tokens=5)).ids
    with pytest.raises(ValueError):
        generate(m, t, "café", greedy(max_tokens=5), cache=c)  # not empty
    assert cache_dims(m) == (2, 2, 8, 128)

    class Cfg:
        num_hidden_layers, num_key_value_heads, head_dim, max_position_embeddings = (
            30,
            3,
            64,
            8192,
        )

    class HF:
        cfg = Cfg()

    assert cache_dims(HF()) == (30, 3, 64, 8192)
    with pytest.raises(ValueError):
        cache_dims(object())
