"""Course tests for L5.2: masks, causal, padding, sliding window, additive
(tinyllm/xfmr/masks.py).

Rung R0 for these (your own graded tests are rung R5, see the chapter): read
them before you write code. Each test names why it exists (WHY), what kind of
check it is (KIND), the planted bugs it kills (CATCHES, mutants in
course/mutants/L5.2), and the chapter section it comes from.

The worked example of the chapter (section 3): causal_mask(3) is the lower
triangle with the diagonal; a decode chunk of 2 queries after 3 cached keys,
causal_mask(2, 5, q_offset=3), opens 4 then 5 keys; sliding_window_mask(4, 4,
2) is a band of width 2; padding_mask([3, 1], 3) opens 3 then 1 key; scores
(2, 1, 5) with the third key blocked give weights (e/(e+1), 1/(e+1), 0) =
(0.731059, 0.268941, 0).

Attention in these tests is a few lines of numpy below (`attend`), not L5.1,
so this module needs no other module's code. The golden fixture
(course/fixtures/L5.2/masks_torch.npz) holds torch 2.14.1's causal biases,
scaled_dot_product_attention outputs, and a flex_attention sliding window,
from course/oracle/L5.2/masks_torch.py.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32
from tinyllm.xfmr.masks import (
    causal_mask,
    combine,
    padding_mask,
    sliding_window_mask,
    to_additive,
)

FIX = os.path.join(os.environ.get("TINYLLM_FIXTURES", ""), "L5.2", "masks_torch.npz")
SEED = int(os.environ.get("OL_SEED", "0"))


def attend(q, k, v, add):
    """softmax(q k^T / sqrt(d) + add) v, float64, a fully blocked row -> zeros."""
    s = q @ np.swapaxes(k, -1, -2) / math.sqrt(q.shape[-1]) + add
    m = np.max(s, axis=-1, keepdims=True)
    m = np.where(np.isneginf(m), 0.0, m)
    e = np.exp(s - m)
    tot = np.sum(e, axis=-1, keepdims=True)
    return (e / np.where(tot == 0, 1.0, tot)) @ v


def B(rows):
    return np.array(rows, dtype=bool)


# --- the worked example ----------------------------------------------------------------


def test_hand_example_masks():
    # WHY: section 3, every mask written out: the causal triangle, a decode
    #      chunk aligned bottom right by q_offset, a window of 2, padding of
    #      lengths 3 and 1, and causal AND padding for the short sequence
    #      (only key 0 open in every row).
    # KIND: unit, smoke
    # CATCHES: s01, s02, s03, s04, s05, s08, m01
    # CHAPTER: L5.2 section 3, Worked example by hand
    assert causal_mask(3).tolist() == B([[1, 0, 0], [1, 1, 0], [1, 1, 1]]).tolist()
    assert (
        causal_mask(2, 5, q_offset=3).tolist()
        == B([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]]).tolist()
    )
    assert (
        sliding_window_mask(4, 4, 2).tolist()
        == B([[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]]).tolist()
    )
    pad = padding_mask(np.array([3, 1]), 3)
    assert pad.tolist() == B([[1, 1, 1], [1, 0, 0]]).tolist()
    both = combine(causal_mask(3), pad[:, None, :])
    assert both.shape == (2, 3, 3)
    assert both[1].tolist() == B([[1, 0, 0], [1, 0, 0], [1, 0, 0]]).tolist()
    assert both[0].tolist() == causal_mask(3).tolist()


def test_hand_example_additive_softmax():
    # WHY: section 3: adding to_additive([T, T, F]) = (0, 0, -inf) to the
    #      scores (2, 1, 5) and taking the softmax gives (0.731059, 0.268941,
    #      0): the blocked key gets exactly 0, though its score was the largest.
    # KIND: unit, smoke
    # CATCHES: s06, s07
    # CHAPTER: L5.2 section 3, Worked example by hand
    add = to_additive(B([True, True, False]), np.float64)
    assert add[0] == 0.0 and add[1] == 0.0 and add[2] == -np.inf
    s = np.array([2.0, 1.0, 5.0]) + add
    w = np.exp(s - s.max())
    w /= w.sum()
    e = math.e
    assert_close(w, [e / (e + 1), 1 / (e + 1), 0.0], rtol=1e-15, atol=0)
    assert w[2] == 0.0


# --- against torch -----------------------------------------------------------------------


def test_causal_matches_torch_biases():
    # WHY: torch has two causal conventions for Tq != Tk: upper left
    #      (is_causal=True: query i sees keys 0..i) and lower right (the last
    #      query sees every key, what a KV-cache chunk needs). Ours is one
    #      function: q_offset = 0 is upper left, q_offset = Tk - Tq lower right.
    # KIND: golden
    # CATCHES: s01, s02, m01
    # CHAPTER: L5.2 section 2.2, Causal masks and q_offset
    f = np.load(FIX)
    for tq, tk in [(4, 4), (3, 7), (1, 5), (5, 3)]:
        assert causal_mask(tq, tk).tolist() == f[f"upper_left.{tq}x{tk}"].tolist(), (
            tq,
            tk,
        )
        if tq <= tk:
            got = causal_mask(tq, tk, q_offset=tk - tq)
            assert got.tolist() == f[f"lower_right.{tq}x{tk}"].tolist(), (tq, tk)


def test_attention_with_masks_matches_torch_sdpa():
    # WHY: masks are only right if attention through them is: torch's
    #      scaled_dot_product_attention with is_causal, with the lower-right
    #      bias of a 3-query chunk over 7 keys, and with key padding (lengths
    #      5 and 3) AND causal, against numpy attention with our additive masks.
    # KIND: golden
    # CATCHES: s01, s02, s05, s07, s08, m01
    # CHAPTER: L5.2 section 2.5, Additive masks
    f = np.load(FIX)
    q, k, v = f["sdpa.causal.q"], f["sdpa.causal.k"], f["sdpa.causal.v"]
    got = attend(q, k, v, to_additive(causal_mask(5), np.float64))
    assert_close(got, f["sdpa.causal.out"], rtol=1e-12, atol=1e-13, msg="causal")
    q, k, v = f["sdpa.decode.q"], f["sdpa.decode.k"], f["sdpa.decode.v"]
    got = attend(q, k, v, to_additive(causal_mask(3, 7, q_offset=4), np.float64))
    assert_close(got, f["sdpa.decode.out"], rtol=1e-12, atol=1e-13, msg="decode chunk")
    q, k, v = f["sdpa.padded.q"], f["sdpa.padded.k"], f["sdpa.padded.v"]
    pad = padding_mask(f["sdpa.padded.lengths"], 5)
    m = combine(causal_mask(5), pad[:, None, None, :])
    assert m.shape == (2, 1, 5, 5)
    got = attend(q, k, v, to_additive(m, np.float64))
    assert_close(got, f["sdpa.padded.out"], rtol=1e-12, atol=1e-13, msg="padded")


def test_sliding_window_matches_flex_attention():
    # WHY: Mistral's sliding window as torch's flex_attention builds it from
    #      the rule "kv <= q and q - kv < window": each query sees at most
    #      `window` keys, itself included.
    # KIND: golden
    # CATCHES: s03, s04
    # CHAPTER: L5.2 section 2.4, Sliding windows
    f = np.load(FIX)
    assert sliding_window_mask(6, 6, 3).tolist() == f["window.6x6.w3"].tolist()


# --- properties ---------------------------------------------------------------------------


def test_causality_is_bitwise():
    # WHY: the defining property of a causal language model: changing tokens
    #      after position t must leave every output at positions <= t
    #      bitwise unchanged (exp(-inf) is exactly 0, and adding exact zeros
    #      changes no sum). Any leak of the future, however small, breaks
    #      next-token training: the model reads the answer.
    # KIND: property
    # CATCHES: s04, s07, m01
    # CHAPTER: L5.2 section 2.2, Causal masks and q_offset
    g = PCG32(seed=SEED)
    T, d = 7, 4
    for case in range(6):
        x = g.normal_array((T, d))
        wq, wk, wv = (g.normal_array((d, d)) for _ in range(3))
        add = to_additive(causal_mask(T), np.float64)
        if case % 2:
            add = to_additive(sliding_window_mask(T, T, 3), np.float64)
        base = attend(x @ wq, x @ wk, x @ wv, add)
        t = 1 + g.below(T - 1)
        y = x.copy()
        y[t + 1 :] += 10.0 * g.normal_array((T - t - 1, d))
        out = attend(y @ wq, y @ wk, y @ wv, add)
        assert np.array_equal(out[: t + 1], base[: t + 1]), (case, t)
        if t + 1 < T:
            assert not np.array_equal(out[t + 1 :], base[t + 1 :])


def test_chunked_decode_masks_are_rows_of_the_full_mask():
    # WHY: decoding with a KV cache computes the last c queries of a length-T
    #      sequence in one step; their mask must be exactly the last c rows of
    #      the full mask, causal and sliding alike, or cached decoding and
    #      full recomputation disagree (L8.2's differential test).
    # KIND: property
    # CATCHES: s02, s09, m02
    # CHAPTER: L5.2 section 2.2, Causal masks and q_offset
    for T in (1, 4, 9):
        for c in range(1, T + 1):
            off = T - c
            assert (
                causal_mask(c, T, q_offset=off).tolist()
                == causal_mask(T)[off:].tolist()
            )
            assert (
                causal_mask(c, q_offset=off).tolist() == causal_mask(T)[off:].tolist()
            )
            for w in (1, 2, 5):
                full = sliding_window_mask(T, T, w)[off:]
                assert (
                    sliding_window_mask(c, T, w, q_offset=off).tolist() == full.tolist()
                )


def test_window_counts():
    # WHY: query at position p sees min(window, p + 1) keys, all at
    #      positions p - window + 1 .. p; a window at least the sequence
    #      length is the plain causal mask.
    # KIND: property
    # CATCHES: s03, s04
    # CHAPTER: L5.2 section 2.4, Sliding windows
    for T in (3, 8):
        for w in (1, 2, 4, 8, 20):
            m = sliding_window_mask(T, T, w)
            assert m.sum(axis=1).tolist() == [min(w, p + 1) for p in range(T)]
            if w >= T:
                assert m.tolist() == causal_mask(T).tolist()


def test_combine_is_and_with_broadcasting():
    # WHY: masks compose by AND (a key is open only if every rule opens it)
    #      with numpy broadcasting, so a [T, T] causal mask and a
    #      [B, 1, 1, T] key-padding mask make one [B, 1, T, T] mask for every
    #      head. Order does not matter, and one mask combines to itself.
    # KIND: property
    # CATCHES: s08
    # CHAPTER: L5.2 section 2.3, Padding and combining
    g = PCG32(seed=SEED + 1)
    for _ in range(10):
        T = 2 + g.below(6)
        lengths = np.array([g.below(T + 1) for _ in range(3)])
        c, p = causal_mask(T), padding_mask(lengths, T)[:, None, None, :]
        m = combine(c, p)
        assert m.dtype == np.bool_ and m.shape == (3, 1, T, T)
        assert m.tolist() == np.logical_and(c, p).tolist()
        assert combine(p, c).tolist() == m.tolist()
        w = sliding_window_mask(T, T, 2)
        assert (
            combine(c, w, p).tolist()
            == np.logical_and(np.logical_and(c, w), p).tolist()
        )
    assert combine(causal_mask(3)).tolist() == causal_mask(3).tolist()


def test_padding_mask_edges():
    # WHY: a sequence of length 0 (an empty prompt slot in a batch) opens no
    #      key, one of length T opens every key, and the result is [B, T]
    #      with True on the real tokens.
    # KIND: boundary
    # CATCHES: s05
    # CHAPTER: L5.2 section 2.3, Padding and combining
    m = padding_mask(np.array([0, 4, 2]), 4)
    assert m.shape == (3, 4) and m.dtype == np.bool_
    assert m.tolist() == B([[0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 0, 0]]).tolist()


def test_fully_masked_row_is_all_minus_inf():
    # WHY: a query whose keys are all blocked (a padded query row in a batch
    #      whose sequence has length 0) must get an all -inf row, which the
    #      stable softmax (M09.2) turns into zero weights. A large finite
    #      negative such as -1e9 turns it into a uniform average over keys
    #      the query was forbidden to read.
    # KIND: boundary
    # CATCHES: s05, s06, s08
    # CHAPTER: L5.2 section 5, Pitfalls
    m = combine(causal_mask(3), padding_mask(np.array([0]), 3)[:, None, :])
    add = to_additive(m)
    assert np.all(np.isneginf(add))
    out = attend(
        np.ones((1, 3, 2)), np.ones((1, 3, 2)), np.arange(6.0).reshape(1, 3, 2), add
    )
    assert np.array_equal(out, np.zeros((1, 3, 2)))


def test_additive_dtypes_and_values():
    # WHY: the additive mask is added to scores in their dtype: float32 by
    #      default, float16 and float64 on request, with exactly 0.0 and -inf
    #      (adding any finite "almost 0" would shift every open score).
    # KIND: unit
    # CATCHES: s06, s07
    # CHAPTER: L5.2 section 2.5, Additive masks
    m = causal_mask(3)
    assert to_additive(m).dtype == np.float32
    for dt in (np.float16, np.float32, np.float64):
        a = to_additive(m, dt)
        assert a.dtype == dt and a.shape == (3, 3)
        assert np.all(a[m] == 0.0) and np.all(np.isneginf(a[~m]))
        assert not np.any(np.signbit(a[m]))


def test_default_tk_follows_q_offset():
    # WHY: with no Tk, the keys run up to the last query: q_offset + Tq. A
    #      default of Tk = Tq silently drops the cached keys of a decode step.
    # KIND: unit
    # CATCHES: s09
    # CHAPTER: L5.2 section 4, The interface
    assert causal_mask(2, q_offset=3).shape == (2, 5)
    assert causal_mask(1, q_offset=6).tolist() == [[True] * 7]
    assert causal_mask(4).shape == (4, 4)


def test_rejects_bad_arguments():
    # WHY: a float or integer 0/1 "mask" is the classic convention mix-up
    #      (torch's nn.MultiheadAttention uses True = blocked); sizes below 1,
    #      negative offsets, lengths outside [0, T], unbroadcastable masks,
    #      and integer dtypes raise instead of producing a silently wrong mask.
    # KIND: boundary
    # CATCHES: m03, m04, m05
    # CHAPTER: L5.2 section 5, Pitfalls
    for args in ((0,), (2, 0), (2, 3, -1), (2.5,)):
        with pytest.raises(ValueError):
            causal_mask(*args)
    for args in ((0, 3, 1), (3, 3, 0), (3, 3, 2, -1)):
        with pytest.raises(ValueError):
            sliding_window_mask(*args)
    for lengths, T in (([3, 5], 4), ([-1], 3), ([[1]], 3), ([1.5], 3), ([1], 0)):
        with pytest.raises(ValueError):
            padding_mask(np.array(lengths), T)
    with pytest.raises(ValueError):
        combine()
    with pytest.raises(ValueError):
        combine(causal_mask(3), causal_mask(4))
    with pytest.raises(ValueError):
        combine(causal_mask(3).astype(np.float32))
    with pytest.raises(ValueError):
        to_additive(np.array([[1, 0]]))
    with pytest.raises(ValueError):
        to_additive(causal_mask(2), np.int32)
