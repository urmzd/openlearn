"""Course tests for M11.2: perplexity, bits per byte, and the NLL accumulator
(tinyllm/info/ppl.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/M11.2), and the chapter section it comes from.

The chapter's worked example (section 3) scores "the cat sat." as the four
tokens "the", " cat", " sat", "." with probabilities 1/4, 1/2, 1/8, 1/2:
7 ln 2 nats in all over 4 tokens and 12 bytes.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32
from tinyllm.info.ppl import NLLAccumulator, perplexity

GOLDEN = Path(os.environ.get("TINYLLM_FIXTURES", "")) / "M11.2" / "golden.json"
LN2 = math.log(2.0)
HAND = [math.log(4), math.log(2), math.log(8), math.log(2)]  # -ln of 1/4, 1/2, 1/8, 1/2
KEYS = {"nll_sum", "n_tokens", "n_bytes", "nll_mean", "ppl", "bits_per_token", "bpb"}


def seed() -> int:
    return int(os.environ.get("OL_SEED", "0"))


def batches(seed_: int, sizes, scale: float, masked: float):
    """The golden cases' batches. Must stay identical to batches() in
    course/oracle/M11.2/ppl_golden.py."""
    rng = PCG32(seed=seed_)
    out = []
    for n in sizes:
        x = (rng.uniform_array((n,)) * scale).astype(np.float32)
        keep = rng.uniform_array((n,)) >= masked
        x[~keep] = np.nan  # padding holds garbage
        n_bytes = 3 * int(keep.sum()) + rng.below(50)
        out.append((x, keep, n_bytes))
    return out


# --- the worked example -------------------------------------------------------


def test_hand_example_sentence():
    # WHY: the chapter's worked example. 7 ln 2 nats over 4 tokens is 1.75
    #      bits per token, perplexity 2^1.75 = 3.3636 (the model is as unsure
    #      as a fair choice among 3.36 tokens), and 7 bits over 12 bytes is
    #      0.5833 bits per byte.
    # KIND: unit
    # CATCHES: s07, s08, m03
    # CHAPTER: M11.2 section 3, Worked example by hand
    acc = NLLAccumulator()
    acc.add(np.array(HAND), n_bytes=12)
    r = acc.result()
    assert set(r) == KEYS
    assert r["n_tokens"] == 4 and r["n_bytes"] == 12
    assert_close(r["nll_sum"], 7 * LN2, dtype="float64")
    assert_close(r["nll_mean"], 1.75 * LN2, dtype="float64")
    assert_close(r["ppl"], 2**1.75, dtype="float64")
    assert_close(r["bits_per_token"], 1.75, dtype="float64")
    assert_close(r["bpb"], 7 / 12, dtype="float64")


def test_hand_example_two_batches_with_padding():
    # WHY: the same sentence split as [the] and [cat, sat, ., PAD]. The
    #      batch means are 2 ln 2 and 5/3 ln 2; their average, 11/6 ln 2, is
    #      wrong. The right mean weights every token equally: 7/4 ln 2. The
    #      padding slot holds NaN and must not count.
    # KIND: unit
    # CATCHES: s01, s02, s03
    # CHAPTER: M11.2 section 3, Worked example by hand
    acc = NLLAccumulator()
    acc.add(np.array([HAND[0]]), n_bytes=3)
    acc.add(np.array(HAND[1:] + [math.nan]), mask=np.array([1, 1, 1, 0]), n_bytes=9)
    r = acc.result()
    assert r["n_tokens"] == 4
    assert_close(r["nll_mean"], 1.75 * LN2, dtype="float64")
    assert_close(r["bpb"], 7 / 12, dtype="float64")


# --- perplexity -----------------------------------------------------------------


def test_perplexity_of_a_uniform_model_is_k():
    # WHY: a model that spreads its mass evenly over k tokens pays ln k per
    #      token, and its perplexity is exactly k: perplexity is the
    #      effective number of choices. A perfect model (0 nats) scores 1.
    # KIND: property
    # CATCHES: m01
    # CHAPTER: M11.2 section 2, Principles (perplexity)
    for k in (1, 2, 3, 256, 50_257):
        for n in (1, 7, 1000):
            assert_close(perplexity(n * math.log(k), n), float(k), dtype="float64")
    assert perplexity(0.0, 5) == 1.0


def test_perplexity_overflow_is_inf():
    # WHY: an untrained or broken model can average more than 709.78 nats
    #      per token, where exp overflows float64. math.exp raises
    #      OverflowError there; an evaluation report needs inf instead.
    # KIND: boundary
    # CATCHES: s06, m01
    # CHAPTER: M11.2 section 5, Pitfalls, item 5
    assert perplexity(800.0, 1) == math.inf
    assert perplexity(math.inf, 3) == math.inf
    assert_close(perplexity(709.0, 1), math.exp(709.0), dtype="float64")


def test_perplexity_rejects_bad_arguments():
    # WHY: zero tokens has no perplexity, and a negative or NaN total is a
    #      bug upstream (a log-probability added with the wrong sign).
    # KIND: boundary
    # CATCHES: m01
    # CHAPTER: M11.2 section 4, The interface
    for nll, n in ((1.0, 0), (1.0, -1), (-0.5, 3), (math.nan, 3)):
        with pytest.raises(ValueError):
            perplexity(nll, n)


# --- the accumulator ------------------------------------------------------------


def test_masked_positions_are_ignored_whatever_they_hold():
    # WHY: padded batches carry garbage in the padding: NaN, inf, or a loss
    #      computed against a pad id. Multiplying by the mask gives
    #      NaN * 0 = NaN; the mask must select. A 0/1 integer mask works too.
    # KIND: boundary
    # CATCHES: s02, s03
    # CHAPTER: M11.2 section 5, Pitfalls, item 2
    acc = NLLAccumulator()
    nll = np.array([[1.0, math.nan], [2.0, math.inf]])
    acc.add(nll, mask=np.array([[1, 0], [1, 0]]), n_bytes=2)
    acc.add(np.array([-5.0, 3.0]), mask=np.array([False, True]))
    r = acc.result()
    assert r["n_tokens"] == 3
    assert_close(r["nll_sum"], 6.0, dtype="float64")


def test_rejects_bad_batches():
    # WHY: a mask of another shape, a counted NaN or negative NLL, and a
    #      negative byte count are bugs in the caller. A counted +inf is not:
    #      the model gave a real token probability 0, and the honest
    #      perplexity is inf.
    # KIND: boundary
    # CATCHES: s11, m04
    # CHAPTER: M11.2 section 4, The interface
    acc = NLLAccumulator()
    with pytest.raises(ValueError):
        acc.add(np.ones(3), mask=np.ones(4, dtype=bool))
    with pytest.raises(ValueError):
        acc.add(np.array([1.0, math.nan]))
    with pytest.raises(ValueError):
        acc.add(np.array([1.0, -0.1]))
    with pytest.raises(ValueError):
        acc.add(np.array([1.0]), n_bytes=-1)
    acc.add(np.array([1.0, math.inf]), n_bytes=4)
    r = acc.result()
    assert r["nll_sum"] == math.inf and r["ppl"] == math.inf


def test_result_needs_tokens_and_bpb_needs_bytes():
    # WHY: no tokens means no mean; no bytes means bits per byte is
    #      undefined, reported as NaN so a table shows it as missing rather
    #      than as 0 (a perfect score) or a crash.
    # KIND: boundary
    # CATCHES: s12
    # CHAPTER: M11.2 section 4, The interface
    acc = NLLAccumulator()
    with pytest.raises(ValueError):
        acc.result()
    acc.add(np.zeros(3), mask=np.zeros(3, dtype=bool))
    with pytest.raises(ValueError):
        acc.result()
    acc.add(np.array([LN2, LN2]))
    r = acc.result()
    assert r["n_tokens"] == 2 and math.isnan(r["bpb"])
    assert_close(r["ppl"], 2.0, dtype="float64")


def test_streaming_equals_one_batch():
    # WHY: L6.7 scores 10^7 tokens one batch at a time. Any split of the
    #      same tokens into batches, in any order, must give the same result
    #      as one batch (to float64 rounding), or a number depends on the
    #      batch size of the run that produced it.
    # KIND: property
    # CATCHES: s01, s03, s04
    # CHAPTER: M11.2 section 2, Principles (one accumulator, many batches)
    rng = PCG32(seed=seed())
    for _ in range(10):
        n = 200 + rng.below(800)
        x = rng.uniform_array((n,)) * 8.0
        keep = rng.uniform_array((n,)) >= 0.25
        whole = NLLAccumulator()
        whole.add(x, mask=keep, n_bytes=n)
        parts = NLLAccumulator()
        cuts = sorted({0, n} | {rng.below(n) for _ in range(rng.below(12))})
        pieces = list(zip(cuts[:-1], cuts[1:]))
        for a, b in reversed(pieces):
            parts.add(x[a:b], mask=keep[a:b], n_bytes=b - a)
        rw, rp = whole.result(), parts.result()
        assert rw["n_tokens"] == rp["n_tokens"] == int(keep.sum())
        for k in ("nll_sum", "nll_mean", "ppl", "bpb"):
            assert_close(rp[k], rw[k], rtol=1e-13, atol=0.0)


def test_compensated_sum_across_many_adds():
    # WHY: a total near 1.0 cannot absorb 1e-15 with plain float64 addition
    #      (each add rounds to a multiple of 2.2e-16), so 100 000 such adds
    #      drift by about 1e-11. Compensated summation carries the lost low
    #      bits and lands on 1 + 1e-10 to within a few ulps. Over 10^7 real
    #      tokens the same drift moves the last digits of a reported ppl.
    # KIND: boundary
    # CATCHES: s04, m02
    # CHAPTER: M11.2 section 5, Pitfalls, item 3
    acc = NLLAccumulator()
    acc.add(np.array([1.0]))
    tiny = np.array([1e-15])
    for _ in range(100_000):
        acc.add(tiny)
    want = math.fsum([1.0] + [1e-15] * 100_000)
    assert_close(acc.result()["nll_sum"], want, rtol=0.0, atol=4e-16)


def test_float32_batches_are_widened():
    # WHY: losses arrive as float32. Summing 100 000 of them in float32
    #      loses about seven digits; the batch total must be taken in
    #      float64 (exactly, with fsum) from the float32 values.
    # KIND: boundary
    # CATCHES: s05
    # CHAPTER: M11.2 section 5, Pitfalls, item 4
    x = np.full(100_000, 0.1, dtype=np.float32) + np.linspace(
        0, 1, 100_000, dtype=np.float32
    )
    acc = NLLAccumulator()
    acc.add(x)
    want = math.fsum(x.astype(np.float64).tolist())
    assert_close(acc.result()["nll_sum"], want, rtol=1e-15, atol=0.0)


def test_merge_equals_a_single_accumulator():
    # WHY: shards scored by separate workers (dur.11's EvalSuite) are
    #      combined with merge. Merging must carry the other side's tokens,
    #      bytes, and compensated total, low bits included, and leave the
    #      other side unchanged.
    # KIND: property
    # CATCHES: s09, s10
    # CHAPTER: M11.2 section 2, Principles (one accumulator, many batches)
    single, a, b = NLLAccumulator(), NLLAccumulator(), NLLAccumulator()
    for tgt in (a, b):  # both shards start large, so both carry low bits
        for acc in (single, tgt):
            acc.add(np.array([1.0]), n_bytes=1)
    for i in range(20_000):
        tgt = a if i % 2 else b
        for acc in (single, tgt):
            acc.add(np.array([1e-15]), n_bytes=1)
    before = b.result()
    a.merge(b)
    assert b.result() == before
    ra, rs = a.result(), single.result()
    assert ra["n_tokens"] == rs["n_tokens"] and ra["n_bytes"] == rs["n_bytes"]
    assert_close(ra["nll_sum"], rs["nll_sum"], rtol=0.0, atol=4e-16)
    a.merge(NLLAccumulator())
    assert a.result() == ra


def test_bits_per_byte_compares_tokenizers():
    # WHY: perplexity depends on what a token is. A byte-level model scoring
    #      the same 12 bytes with 8.4 bits has per-token perplexity 2^0.7 =
    #      1.62, which looks better than 3.36, and is worse: 0.7 bits per
    #      byte against 0.5833. Bits per byte is the one number that compares
    #      models with different tokenizers (L1.6, the C1 ablation).
    # KIND: unit
    # CATCHES: s07, s08
    # CHAPTER: M11.2 section 2, Principles (bits per byte)
    word = NLLAccumulator()
    word.add(np.array(HAND), n_bytes=12)
    byte = NLLAccumulator()
    byte.add(np.full(12, 0.7 * LN2), n_bytes=12)
    rw, rb = word.result(), byte.result()
    assert rb["ppl"] < rw["ppl"]
    assert rb["bpb"] > rw["bpb"]
    assert_close(rb["bpb"], 0.7, dtype="float64")


def test_golden_cases():
    # WHY: three long cases (3473 masked tokens; 400 batches of 16; losses
    #      near 20 nats) against totals computed with 50-digit arithmetic by
    #      course/oracle/M11.2, independent of float64 rounding.
    # KIND: golden
    # CATCHES: s01, s02, s03, s05, s07
    # CHAPTER: M11.2 section 4, What the tests check
    doc = json.loads(GOLDEN.read_text())
    for name, case in doc["cases"].items():
        acc = NLLAccumulator()
        for x, keep, nb in batches(
            case["seed"], case["sizes"], case["scale"], case["masked"]
        ):
            acc.add(x, mask=keep, n_bytes=nb)
        r = acc.result()
        assert r["n_tokens"] == case["n_tokens"], name
        assert r["n_bytes"] == case["n_bytes"], name
        for k in ("nll_sum", "nll_mean", "ppl", "bits_per_token", "bpb"):
            assert_close(r[k], float(case[k]), rtol=1e-14, atol=0.0, msg=f"{name} {k}")
