"""Course tests for M06.3: tinyllm/num/rng.py against
spec/pcg32.md.

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/M06.3), and the chapter section it comes from.

The reference vectors are a copy of contracts/spec/pcg32.vectors.json
(fixture course/fixtures/M06.3/pcg32.vectors.json, rebuilt from the spec by
course/oracle/M06.3/vectors.py). Rust and Go ports consume the same frozen
vectors through their own process-based conformance drivers.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from _lib.pcg32 import PCG32 as FrozenPCG32

# The module under test. (Other modules' course tests never import your
# generator: they draw from the frozen _lib copy, D35.)
from tinyllm.num import rng as learner_rng

PCG32 = learner_rng.PCG32
VECTORS = json.loads(
    (
        Path(os.environ.get("TINYLLM_FIXTURES", "")) / "M06.3" / "pcg32.vectors.json"
    ).read_text()
)
SEEDS = ["0", "1", str(2**63)]
M64 = (1 << 64) - 1


def test_worked_example_first_output_of_seed_0():
    # WHY: the chapter's worked example (and spec/pcg32.md's): seeding with
    #      (0, 54) gives state 0x9AE4F7499BA72696, and the first output is
    #      rotr32(0x5C9A3E14, 19) = 0x47C28B93.
    # KIND: unit, smoke
    # CATCHES: s01, s02
    # CHAPTER: M06.3 section 3
    g = PCG32(0)
    assert g.state() == (0x9AE4F7499BA72696, 109)
    assert g.next_u32() == 0x47C28B93


def test_oneill_demo_line():
    # WHY: the published first line of O'Neill's pcg32-demo, pcg32_srandom_r
    #      (42, 54): an oracle outside this course entirely.
    # KIND: golden
    # CATCHES: s01, s02, s03, s05
    # CHAPTER: M06.3 section 2.3
    g = PCG32(42, seq=54)
    want = [0xA15C02B7, 0x7B47F409, 0xBA1D3330, 0x83D2F293, 0xBFA4784B, 0xCBED606E]
    assert [g.next_u32() for _ in range(6)] == want


@pytest.mark.parametrize("seed", SEEDS)
def test_first_1024_outputs_match_spec_vectors(seed):
    # WHY: every language's PCG32 must produce this exact stream (D10): the
    #      Rust sampler and Go load generator are both
    #      checked against the same file. 1024 outputs walk the state far past
    #      the first wrap mod 2^64, so a missing mask shows up.
    # KIND: golden
    # CATCHES: s01, s02, s03, s05
    # CHAPTER: M06.3 section 2.3
    g = PCG32(int(seed))
    assert [g.next_u32() for _ in range(1024)] == VECTORS["next_u32"][seed]


@pytest.mark.parametrize("seed", SEEDS)
def test_uniform_bit_exact(seed):
    # WHY: uniform() is the one float every sampler consumes (L8.1 draws
    #      exactly one per token). It is built from integers only, so it is
    #      bit-exact across languages; any rounding step would break parity.
    # KIND: golden
    # CATCHES: s04
    # CHAPTER: M06.3 section 2.4
    g = PCG32(int(seed))
    assert [g.uniform() for _ in range(8)] == VECTORS["uniform_f64"][seed]
    g = PCG32(int(seed))
    u = g.uniforms(8)
    assert u.dtype == np.float64 and u.tolist() == VECTORS["uniform_f64"][seed]


def test_uniform_uses_53_bits_and_stays_below_one():
    # WHY: (a >> 5) * 2^26 + (b >> 6) is a 53-bit integer, so u * 2^53 is an
    #      integer in [0, 2^53): every double in the grid is reachable and
    #      1.0 never is. A one-draw u32 / 2^32 has only 32 bits (u * 2^32 is
    #      always an integer), which the sampler's tail probabilities notice.
    # KIND: property
    # CATCHES: s04
    # CHAPTER: M06.3 section 2.4
    g = PCG32(7)
    us = [g.uniform() for _ in range(2000)]
    assert all(0.0 <= u < 1.0 for u in us)
    scaled = [u * 2.0**53 for u in us]
    assert all(s == int(s) for s in scaled)
    assert any((u * 2.0**32) != int(u * 2.0**32) for u in us), "only 32 random bits"


@pytest.mark.parametrize("seed", SEEDS)
def test_below_and_shuffle_match_spec_vectors(seed):
    # WHY: L0.5 shuffles training windows with these two, and the order is
    #      saved in trainer_state.json; resuming a run must replay it exactly.
    # KIND: golden
    # CATCHES: s06
    # CHAPTER: M06.3 section 2.5
    g = PCG32(int(seed))
    assert [g.below(10) for _ in range(16)] == VECTORS["below_10"][seed]
    xs = list(range(10))
    PCG32(int(seed)).shuffle(xs)
    assert xs == VECTORS["shuffle_10"][seed]


def test_below_has_no_modulo_bias():
    # WHY: with n = 3 * 2^30, r mod n maps the 2^32 raw values onto [0, n)
    #      so that [0, 2^30) is hit twice as often (probability 1/2 instead of
    #      1/3). Rejecting r < 2^32 mod n removes the bias. 3000 draws put the
    #      biased share 15 standard errors away from 1/3.
    # KIND: statistical
    # CATCHES: s07
    # CHAPTER: M06.3 section 2.5
    n = 3 << 30
    g = PCG32(11)
    draws = [g.below(n) for _ in range(3000)]
    assert all(0 <= d < n for d in draws)
    share = sum(d < (1 << 30) for d in draws) / len(draws)
    se = (1 / 3 * 2 / 3 / len(draws)) ** 0.5
    assert abs(share - 1 / 3) < 4 * se, f"share below 2^30 is {share:.3f}, want 1/3"


def test_below_rejects_bad_n():
    # WHY: n = 0 has no valid output (and would divide by zero); n > 2^32
    #      cannot be reached from one 32-bit draw.
    # KIND: boundary
    # CATCHES: m02
    # CHAPTER: M06.3 section 4
    g = PCG32(0)
    for bad in (0, -1, (1 << 32) + 1):
        with pytest.raises(ValueError):
            g.below(bad)
    assert 0 <= g.below(1 << 32) < (1 << 32)
    assert g.below(1) == 0


@pytest.mark.parametrize("seed", SEEDS)
def test_substreams_match_spec_vectors(seed):
    # WHY: stream(seed, purpose) gives init, dropout, shuffle, sample, and
    #      mutation their own generators from one user seed, so turning on
    #      dropout does not change the initialization. The child seed is
    #      SplitMix64's p-th output: mix64(seed + p * 0x9E3779B97F4A7C15).
    # KIND: golden
    # CATCHES: s08, s09
    # CHAPTER: M06.3 section 2.6
    s = int(seed)
    for purpose, pid in VECTORS["purposes"].items():
        assert learner_rng.PURPOSES[purpose] == pid
        assert learner_rng.child_seed(s, pid) == VECTORS["child_seed"][seed][purpose]
        g = PCG32(s).substream(purpose)
        assert [g.next_u32() for _ in range(4)] == VECTORS["stream_next_u32"][seed][
            purpose
        ]


def test_substream_ignores_draws_already_made():
    # WHY: a sub-stream depends on the seed and the purpose only. Deriving it
    #      from the parent's current state would make the dropout masks depend
    #      on how many init draws happened first.
    # KIND: property
    # CATCHES: s08
    # CHAPTER: M06.3 section 2.6
    g = PCG32(5)
    first = g.substream("dropout").next_u32()
    for _ in range(17):
        g.next_u32()
    assert g.substream("dropout").next_u32() == first
    with pytest.raises(KeyError):
        g.substream("no-such-purpose")


def test_splitmix64_published_outputs():
    # WHY: SplitMix64 (Steele, Lea, and Flood 2014; Vigna's splitmix64.c)
    #      seeded with 0 starts 0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4,
    #      0x06C45D188009454F, and seeded with 1234567 starts
    #      6457827717110365317, 3203168211198807973, 9817491932198370423.
    #      The k-th output is splitmix64(seed + k * GOLDEN).
    # KIND: golden
    # CATCHES: s09
    # CHAPTER: M06.3 section 2.6
    g = learner_rng.GOLDEN
    assert g == 0x9E3779B97F4A7C15
    out0 = [learner_rng.splitmix64((k * g) & M64) for k in (1, 2, 3)]
    assert out0 == [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F]
    out1 = [learner_rng.splitmix64((1234567 + k * g) & M64) for k in (1, 2, 3)]
    assert out1 == [6457827717110365317, 3203168211198807973, 9817491932198370423]


def test_fnv1a64_published_vectors():
    # WHY: FNV-1a 64 test vectors from the FNV reference test suite (Noll).
    #      rt.04 chains this hash over KV blocks, and the Rust engine computes
    #      the same block hash; a byte order or xor-then-multiply slip makes
    #      every prefix-cache lookup miss.
    # KIND: golden, smoke
    # CATCHES: s10
    # CHAPTER: M06.3 section 2.7
    f = learner_rng.fnv1a64
    assert f(b"") == 0xCBF29CE484222325 == learner_rng.FNV_OFFSET
    assert f(b"a") == 0xAF63DC4C8601EC8C
    assert f(b"b") == 0xAF63DF4C8601F1A5
    assert f(b"foobar") == 0x85944171F73967E8
    assert f(b"chongo was here!\n") == 0x46810940EFF5F915


def test_fnv1a64_chains():
    # WHY: the KV block hash continues from the parent block's hash:
    #      fnv1a64(b, fnv1a64(a)) == fnv1a64(a + b). The h argument is how a
    #      hash is extended without concatenating buffers.
    # KIND: property
    # CATCHES: m03
    # CHAPTER: M06.3 section 2.7
    f = learner_rng.fnv1a64
    rng = FrozenPCG32(int(os.environ.get("OL_SEED", "0")), seq=63)
    for _ in range(50):
        a = bytes(rng.below(256) for _ in range(rng.below(20)))
        b = bytes(rng.below(256) for _ in range(rng.below(20)))
        assert f(b, f(a)) == f(a + b)
        assert 0 <= f(a + b) <= M64


def test_universal_hash_hand_example():
    # WHY: Carter-Wegman h(x) = ((a x + b) mod p) mod m, the chapter's worked
    #      example: a = 3, b = 7, p = 13, m = 5, x = 10 gives
    #      (37 mod 13) mod 5 = 11 mod 5 = 1. Reducing mod m before mod p is a
    #      different (and not universal) function.
    # KIND: unit
    # CATCHES: s11
    # CHAPTER: M06.3 section 3
    h = learner_rng.universal_hash
    assert h(10, 3, 7, 13, 5) == 1
    assert [h(x, 3, 7, 13, 5) for x in range(13)] == [
        ((3 * x + 7) % 13) % 5 for x in range(13)
    ]
    with pytest.raises(ValueError):
        h(1, 0, 7, 13, 5)  # a = 0 sends everything to b
    with pytest.raises(ValueError):
        h(1, 3, 7, 13, 0)


def test_universal_hash_collision_rate():
    # WHY: universality: for x != y, the share of (a, b) pairs with
    #      h(x) == h(y) is at most about 1/m. Exhaustive over all a, b for
    #      p = 101, m = 10: no pair of keys collides more than 1/m + 1/p.
    # KIND: property
    # CATCHES: s11
    # CHAPTER: M06.3 section 2.2
    h = learner_rng.universal_hash
    p, m = 101, 10
    for x, y in [(0, 1), (3, 50), (17, 99), (20, 30)]:
        same = sum(
            h(x, a, b, p, m) == h(y, a, b, p, m) for a in range(1, p) for b in range(p)
        )
        assert same / ((p - 1) * p) <= 1 / m + 1 / p


def test_state_roundtrip_resumes_the_stream():
    # WHY: checkpoints store (state, inc) in trainer_state.json; set_state on
    #      a fresh generator must continue exactly where the old one stopped,
    #      and an even inc is not a PCG stream.
    # KIND: property
    # CATCHES: s03, s12
    # CHAPTER: M06.3 section 4
    g = PCG32(123, seq=9)
    for _ in range(100):
        g.next_u32()
    st = g.state()
    assert 0 <= st[0] <= M64 and st[1] % 2 == 1
    ahead = [g.next_u32() for _ in range(10)]
    h = PCG32(0)
    h.set_state(st)
    assert [h.next_u32() for _ in range(10)] == ahead
    with pytest.raises(ValueError):
        h.set_state((5, 4))


def test_uniforms_rejects_negative_n():
    # WHY: an array of -1 draws is a caller bug, not an empty array.
    # KIND: boundary
    # CATCHES: m01
    # CHAPTER: M06.3 section 4
    g = PCG32(0)
    assert g.uniforms(0).shape == (0,)
    with pytest.raises(ValueError):
        g.uniforms(-1)
