"""Course tests for L8.1: sampling and logit processors
(tinyllm/infer/sample.py, spec/sampling.md).

Rung R0 for the course suite: read these before you write code. Each test
names why it exists (WHY), what kind of check it is (KIND), the planted bugs
it kills (CATCHES, mutants in course/mutants/L8.1), and the chapter section
it comes from. You write your own graded tests too (rung R5, see the
chapter).

The worked example is the spec's: V = 5, x = [1, 3, 2, 3, -1], top_k = 3,
top_p = 0.8, seed 0, which samples id 3 with logprob -0.92487. The golden
fixture course/fixtures/L8.1/sampler_golden.json comes from an independent
stdlib transcription of the spec (course/oracle/L8.1/sampler_golden.py) and
is the same file the Rust port (L10.1) is held to. Expected generators are
built from the frozen PCG32.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32, splitmix64
from tinyllm.infer.sample import (
    SamplingParams,
    apply_penalties,
    process_logits,
    request_rng,
    sample,
    sampled_entropy,
    sampling_distribution,
    token_logprobs,
)

FIX = Path(os.environ.get("TINYLLM_FIXTURES", "")) / "L8.1" / "sampler_golden.json"
HAND_X = [1.0, 3.0, 2.0, 3.0, -1.0]


def seed() -> int:
    return int(os.environ.get("OL_SEED", "0"))


def frozen_stream(s: int) -> PCG32:
    """stream(s, "sample") from the frozen helpers: PCG32(child_seed, seq 4),
    child_seed = mix64(s + 4 * GOLDEN) (frozen splitmix64 adds GOLDEN once)."""
    _, child = splitmix64((s + 3 * 0x9E3779B97F4A7C15) & ((1 << 64) - 1))
    return PCG32(child, 4)


class Script:
    """A uniform source that returns fixed values and counts the calls; it
    fails the test if asked for more than it holds."""

    def __init__(self, values=()):
        self.values = list(values)
        self.calls = 0

    def uniform(self) -> float:
        assert self.calls < len(self.values), (
            "the sampler drew more uniforms than expected"
        )
        v = self.values[self.calls]
        self.calls += 1
        return v


def spec_q(l: np.ndarray, keep) -> np.ndarray:
    """Step 9 written in the test: ascending ids, math.exp, a running sum."""
    ids = sorted(keep)
    m = max(l[i] for i in ids)
    e = [math.exp(l[i] - m) for i in ids]
    z = 0.0
    for v in e:
        z += v
    q = np.zeros(len(l))
    for i, v in zip(ids, e):
        q[i] = v / z
    return q


# --- the worked example -------------------------------------------------------


def test_hand_example():
    # WHY: the spec's worked example end to end: top-k keeps ids 1, 3, 2;
    #      top-p over them reaches 0.8 at id 3 and keeps {1, 3}; the request
    #      stream of seed 0 draws u = 0.80209, which is past c = 0.5, so the
    #      token is id 3 with logprob -ln(2.52153) = -0.92487.
    # KIND: unit, smoke
    # CATCHES: s04, s13, s14, s15, s17, s20, s22
    # CHAPTER: L8.1 section 3
    p = SamplingParams(top_k=3, top_p=0.8, seed=0)
    tok, lp = sample(HAND_X, p, [], request_rng(0))
    assert tok == 3
    assert_close(lp, -0.92487, rtol=0, atol=5e-6)
    assert lp == -math.log(math.exp(-2) + 1 + math.exp(-1) + 1 + math.exp(-4))
    pl = process_logits(HAND_X, p)
    assert pl.tolist() == [-np.inf, 3.0, -np.inf, 3.0, -np.inf]
    assert sampling_distribution(HAND_X, p).tolist() == [0.0, 0.5, 0.0, 0.5, 0.0]


def test_hand_example_intermediates():
    # WHY: the numbers of section 3 one step at a time: top-k alone keeps
    #      {1, 2, 3} with q = [0.42232, 0.15536, 0.42232]; logprobs are taken
    #      before temperature and filtering, so a filter never changes them.
    # KIND: unit, smoke
    # CATCHES: s01, s17
    # CHAPTER: L8.1 section 3
    q = sampling_distribution(HAND_X, SamplingParams(top_k=3))
    assert_close(q, [0.0, 0.42232, 0.15536, 0.42232, 0.0], rtol=0, atol=5e-6)
    lp = token_logprobs(HAND_X, SamplingParams(temperature=0.3, top_k=1))
    assert_close(lp, token_logprobs(HAND_X, SamplingParams()), rtol=0, atol=0)
    assert_close(lp[1], -0.92487, rtol=0, atol=5e-6)


def test_request_rng_is_the_sample_stream():
    # WHY: the request generator is stream(seed, "sample") of spec/pcg32.md:
    #      child_seed(0, 4) = 0xF88BB8A8724C81EC, seq 4; its first uniform is
    #      the spec's u = 0.80209. PCG32(seed) itself would give different
    #      tokens than the Rust engine for the same seed.
    # KIND: golden
    # CATCHES: s15
    # CHAPTER: L8.1 section 2.6
    r = request_rng(0)
    want = frozen_stream(0)
    assert want.state == PCG32(0xF88BB8A8724C81EC, 4).state
    assert_close(r.uniform(), 0.80209, rtol=0, atol=5e-6)
    r, want = request_rng(123456789), frozen_stream(123456789)
    assert [r.uniform() for _ in range(5)] == [want.uniform() for _ in range(5)]


# --- penalties -------------------------------------------------------------------


def test_repetition_penalty_hf_semantics():
    # WHY: HF's rule divides a positive logit by r and multiplies a negative
    #      one by r, so both move away from being chosen; dividing a negative
    #      logit would make a repeated unlikely token MORE likely. It applies
    #      once per distinct id seen in the prompt or the output.
    # KIND: unit
    # CATCHES: s05, s16
    # CHAPTER: L8.1 section 2.2
    x = [2.0, -2.0, 1.0, 0.0, 4.0]
    l = apply_penalties(
        x, SamplingParams(repetition_penalty=2.0), history=[0, 0, 0], prompt=[1, 3]
    )
    assert l.dtype == np.float64
    assert l.tolist() == [1.0, -4.0, 1.0, 0.0, 4.0]
    assert x == [2.0, -2.0, 1.0, 0.0, 4.0]  # the input is not modified


def test_presence_and_frequency_openai_semantics():
    # WHY: OpenAI's rule subtracts a_f * count + a_p from each generated id,
    #      prompt ids excluded; the frequency term grows with the count.
    # KIND: unit
    # CATCHES: s06, s07
    # CHAPTER: L8.1 section 2.2
    p = SamplingParams(presence_penalty=0.5, frequency_penalty=0.25)
    l = apply_penalties([1.0, 1.0, 1.0, 1.0], p, history=[2, 2, 2, 0], prompt=[1, 1])
    assert l.tolist() == [1.0 - 0.25 - 0.5, 1.0, 1.0 - 0.75 - 0.5, 1.0]


def test_penalty_order():
    # WHY: repetition first, then presence/frequency (spec steps 2 and 3);
    #      the other order gives (3 - 1) / 2 = 1 instead of 3 / 2 - 1 = 0.5.
    # KIND: unit
    # CATCHES: s21
    # CHAPTER: L8.1 section 2.2
    p = SamplingParams(repetition_penalty=2.0, presence_penalty=1.0)
    assert apply_penalties([3.0, 0.0], p, history=[0]).tolist() == [0.5, 0.0]


# --- filters ------------------------------------------------------------------------


def test_top_p_keeps_the_crossing_token():
    # WHY: nucleus sampling keeps the smallest prefix of sorted probs whose mass
    #      reaches p, and that prefix INCLUDES the token that crosses p.
    #      Four equal logits give q = 1/4 each, exactly, so the running sums
    #      0.25, 0.5, 0.75 hit p = 0.5 exactly: ">=" keeps two tokens, ">"
    #      keeps three; p = 0.6 is crossed by the third token, which stays.
    # KIND: boundary
    # CATCHES: s03, s20
    # CHAPTER: L8.1 section 5, Pitfalls, item 2
    x = [1.0, 1.0, 1.0, 1.0]
    for top_p, n in [(0.25, 1), (0.5, 2), (0.6, 3), (0.75, 3), (0.76, 4)]:
        q = sampling_distribution(x, SamplingParams(top_p=top_p))
        assert int((q > 0).sum()) == n, top_p
    q = sampling_distribution(np.log([0.5, 0.3, 0.2]), SamplingParams(top_p=0.79))
    assert (q > 0).tolist() == [True, True, False]


def test_top_k_ties_go_to_the_lowest_id():
    # WHY: ordering by (logit desc, id asc) makes top-k deterministic when
    #      logits tie, which is what lets Rust and Python agree; ids 1 and 3
    #      tie for the last slot of top_k = 2 below, and 1 must win.
    # KIND: boundary
    # CATCHES: s02, s17
    # CHAPTER: L8.1 section 2.3
    x = [5.0, 2.0, 1.0, 2.0, 0.0]
    q = sampling_distribution(x, SamplingParams(top_k=2))
    assert (q > 0).tolist() == [True, True, False, False, False]
    q = sampling_distribution(x, SamplingParams(top_k=3))
    assert (q > 0).tolist() == [True, True, False, True, False]


def test_min_p_is_relative_to_the_top():
    # WHY: min-p keeps the tokens with q >= min_p * max(q): with q = [0.6,
    #      0.3, 0.07, 0.03] and min_p = 0.1 the cut is 0.06, so 0.07 stays
    #      and 0.03 goes; an absolute cut at 0.1 would drop both. Two tokens
    #      tied for the top both survive min_p = 1 (the comparison is >=).
    # KIND: unit
    # CATCHES: s08, m02
    # CHAPTER: L8.1 section 2.3
    x = np.log([0.6, 0.3, 0.07, 0.03])
    q = sampling_distribution(x, SamplingParams(min_p=0.1))
    assert (q > 0).tolist() == [True, True, True, False]
    q = sampling_distribution([2.0, 2.0, 1.0], SamplingParams(min_p=1.0))
    assert q.tolist() == [0.5, 0.5, 0.0]


def test_filters_compose_in_spec_order():
    # WHY: top-p sees the distribution AFTER top-k (renormalized), and min-p
    #      the one after top-p. On q = [0.4, 0.3, 0.2, 0.1], top_k = 2 makes
    #      id 0 worth 0.4 / 0.7 = 0.57, which alone crosses top_p = 0.55; on
    #      the full distribution it would not (0.4). With top_k = 3, top_p =
    #      0.5, min_p = 0.6, top-p keeps {0, 1} and min-p keeps both; min-p
    #      first would drop id 2, and top-p would then keep only id 0.
    # KIND: unit
    # CATCHES: s08, s13, s17, s18, s20
    # CHAPTER: L8.1 section 2.3
    x = np.log([0.4, 0.3, 0.2, 0.1])
    q = sampling_distribution(x, SamplingParams(top_k=2, top_p=0.55))
    assert (q > 0).tolist() == [True, False, False, False]
    q = sampling_distribution(x, SamplingParams(top_k=3, top_p=0.5, min_p=0.6))
    assert (q > 0).tolist() == [True, True, False, False]


def test_masked_logits_are_never_sampled():
    # WHY: constrained decoding (L8.7) writes -inf on forbidden ids; they must
    #      have probability 0 under every filter, even when top_k is larger
    #      than the number of allowed ids, and every logit -inf is an error.
    # KIND: property
    # CATCHES: s22
    # CHAPTER: L8.1 section 2.3
    rng = PCG32(seed(), 81)
    for _ in range(200):
        x = rng.normal_array((12,))
        x[rng.uniform_array((12,)) < 0.6] = -np.inf
        x[rng.below(12)] = 0.5
        p = SamplingParams(top_k=int(rng.below(15)), top_p=0.5 + 0.5 * rng.uniform())
        q = sampling_distribution(x, p)
        assert (q[np.isneginf(x)] == 0).all()
        assert np.isneginf(
            process_logits(x, p)[q == 0]
        ).all()  # removed ids are -inf, not -1e9
        tok, _ = sample(x, p, [], rng)
        assert np.isfinite(x[tok])
    with pytest.raises(ValueError):
        sample([-np.inf, -np.inf], SamplingParams(), [], Script([0.5]))


# --- the draw --------------------------------------------------------------------------


def test_inverse_cdf_boundaries():
    # WHY: exact enumeration of the draw: for every configuration of the
    #      processors, u just below the cumulative sum c_i of
    #      sampling_distribution gives id i and u = c_i gives the next kept id
    #      (u < c, strictly), so P(id i) = q_i exactly.
    # KIND: statistical
    # CATCHES: s14
    # CHAPTER: L8.1 section 2.5
    rng = PCG32(seed(), 82)
    configs = [
        SamplingParams(),
        SamplingParams(temperature=0.6),
        SamplingParams(top_k=4),
        SamplingParams(top_p=0.7),
        SamplingParams(min_p=0.15),
        SamplingParams(
            temperature=1.4,
            top_k=6,
            top_p=0.9,
            min_p=0.02,
            repetition_penalty=1.2,
            frequency_penalty=0.3,
        ),
    ]
    for p in configs:
        x = (rng.normal_array((10,)) * 2).astype(np.float32)
        hist = [int(rng.below(10)) for _ in range(4)]
        q = sampling_distribution(x, p, hist)
        kept = [int(i) for i in np.flatnonzero(q > 0)]
        c = 0.0
        for n, i in enumerate(kept):
            c += q[i]
            if n + 1 < len(kept):
                below = np.nextafter(c, 0.0)
                assert sample(x, p, hist, Script([below]))[0] == i
                assert sample(x, p, hist, Script([c]))[0] == kept[n + 1]
        assert sample(x, p, hist, Script([0.0]))[0] == kept[0]
        assert sample(x, p, hist, Script([1.0 - 2.0**-53]))[0] == kept[-1]


def test_chi_square_against_the_distribution():
    # WHY: drawn with the request stream, 20000 samples follow
    #      sampling_distribution (chi-square, p-value above 1e-3 at a fixed
    #      seed): the sampler and the distribution it reports agree.
    # KIND: statistical
    # CATCHES: s17
    # CHAPTER: L8.1 section 2.5
    x = np.array([1.2, 0.3, -0.5, 2.0, 1.9, -3.0, 0.0])
    p = SamplingParams(temperature=0.9, top_k=5, min_p=0.05)
    q = sampling_distribution(x, p)
    rng = frozen_stream(seed())
    counts = np.zeros(7)
    for _ in range(20000):
        counts[sample(x, p, [], rng)[0]] += 1
    keep = q > 0
    stat = float(((counts[keep] - 20000 * q[keep]) ** 2 / (20000 * q[keep])).sum())
    assert (counts[~keep] == 0).all()
    assert stat < {1: 10.828, 2: 13.816, 3: 16.266, 4: 18.467}[int(keep.sum()) - 1]


def test_one_draw_per_token_none_for_greedy():
    # WHY: draw accounting (spec): exactly one uniform per sampled token and
    #      none per greedy token, so the generator's position after n tokens
    #      is known (disaggregated serving and speculative decoding rely on
    #      it). Script fails the test on any extra draw.
    # KIND: unit
    # CATCHES: s04, s09, s10
    # CHAPTER: L8.1 section 2.5
    src = Script([0.3])
    sample(HAND_X, SamplingParams(top_k=1), [], src)
    assert src.calls == 1
    src = Script([])
    tok, lp = sample(HAND_X, SamplingParams(temperature=0.0), [], src)
    assert src.calls == 0 and tok == 1
    assert lp == token_logprobs(HAND_X, SamplingParams())[1]


def test_greedy_and_its_limits():
    # WHY: temperature 0 is greedy with ties to the lowest id; top_k = 1 and a
    #      tiny temperature give the same token for any u, so a seeded run and
    #      a greedy run agree when the margin is clear (the near-tie rule of
    #      MS-L8 compares greedily).
    # KIND: property
    # CATCHES: s02, s09, s10, s17
    # CHAPTER: L8.1 section 2.4
    rng = PCG32(seed(), 83)
    for _ in range(100):
        x = rng.normal_array((9,))
        x[rng.below(9)] = x.max()  # an exact tie for the maximum
        best = int(np.flatnonzero(x == x.max())[0])
        assert sample(x, SamplingParams(temperature=0.0), [], Script())[0] == best
        assert (
            sample(x, SamplingParams(top_k=1), [], Script([rng.uniform()]))[0] == best
        )
        y = x.copy()
        y[best] += 1.0  # a clear margin
        assert (
            sample(y, SamplingParams(temperature=1e-3), [], Script([rng.uniform()]))[0]
            == best
        )
    q = sampling_distribution(HAND_X, SamplingParams(temperature=0.0))
    assert q.tolist() == [0.0, 1.0, 0.0, 0.0, 0.0]
    assert process_logits(HAND_X, SamplingParams(temperature=0.0)).tolist() == HAND_X


def test_sums_are_sequential_in_ascending_ids():
    # WHY: the normalizer Z is a left-to-right float64 loop over ascending
    #      ids; Python's sum() (compensated since 3.12) and numpy's pairwise
    #      np.sum differ from it in the last bit, which flips a sampled token
    #      when u lands within an ulp of a boundary and breaks parity with
    #      Rust. 300 random 500-id distributions must match bitwise.
    # KIND: unit
    # CATCHES: s11, s12
    # CHAPTER: L8.1 section 2.5
    rng = PCG32(seed(), 84)
    for _ in range(300):
        x = rng.normal_array((500,)) * 3
        q = sampling_distribution(x, SamplingParams())
        assert q.tolist() == spec_q(x, range(500)).tolist()
        l = x.astype(np.float64)
        m = l.max()
        z = 0.0
        for v in l:
            z += math.exp(v - m)
        assert (
            token_logprobs(x, SamplingParams()).tolist()
            == (l - m - math.log(z)).tolist()
        )


def test_golden_ids_and_logprobs():
    # WHY: 12 cases x 12 sampled tokens from an independent transcription of
    #      the spec: every processor, penalties that change as the history
    #      grows, masks, ties, and greedy. The Rust engine (L10.1) is held to
    #      the same file, so passing here is what makes the two agree.
    # KIND: golden
    # CATCHES: s01, s04, s05, s06, s07, s08, s11, s12, s13, s14, s15, s16, s20, s21
    # CHAPTER: L8.1 section 4
    doc = json.loads(FIX.read_text())
    for case in doc["cases"]:
        x = np.array([float(v) for v in case["logits"]], dtype=np.float32)
        p = SamplingParams(**case["params"], seed=case["seed"])
        rng = request_rng(case["seed"])
        out, lps = [], []
        for _ in range(len(case["ids"])):
            tok, lp = sample(x, p, out, rng, prompt=case["prompt"])
            out.append(tok)
            lps.append(lp)
        assert out == case["ids"], case["name"]
        assert lps == case["logprobs"], case["name"]


def test_entropy_of_the_sampling_distribution():
    # WHY: generate (L8.2) logs the entropy of the distribution it actually
    #      sampled from (after the filters), in nats: 0 when greedy, ln 2 for
    #      two equally likely kept tokens.
    # KIND: unit
    # CATCHES: s13, s17, s19, s20
    # CHAPTER: L8.1 section 4
    assert sampled_entropy(HAND_X, SamplingParams(temperature=0.0)) == 0.0
    assert_close(
        sampled_entropy(HAND_X, SamplingParams(top_k=3, top_p=0.8)),
        math.log(2),
        rtol=1e-12,
        atol=0,
    )
    q = sampling_distribution(HAND_X, SamplingParams(top_k=3))
    want = -sum(v * math.log(v) for v in q if v > 0)
    assert_close(
        sampled_entropy(HAND_X, SamplingParams(top_k=3)), want, rtol=1e-12, atol=0
    )


def test_params_are_validated():
    # WHY: a temperature below 0, top_p of 0, min_p above 1, a zero
    #      repetition penalty, NaN logits, or an id outside the vocabulary
    #      are caller bugs; the sampler must refuse them instead of returning
    #      a token. The defaults turn every processor off.
    # KIND: boundary
    # CATCHES: m01, m02, m03, m04
    # CHAPTER: L8.1 section 4
    d = SamplingParams()
    assert (d.temperature, d.top_k, d.top_p, d.min_p, d.repetition_penalty) == (
        1.0,
        0,
        1.0,
        0.0,
        1.0,
    )
    assert d.stop == [] and d.stop is not SamplingParams().stop
    assert apply_penalties(HAND_X, d, history=[1, 3], prompt=[0]).tolist() == HAND_X
    for kw in [
        dict(temperature=-0.1),
        dict(top_p=0.0),
        dict(top_p=1.5),
        dict(min_p=1.5),
        dict(min_p=-0.1),
        dict(repetition_penalty=0.0),
        dict(top_k=-1),
        dict(presence_penalty=math.inf),
        dict(max_tokens=0),
        dict(seed=-1),
        dict(temperature=math.nan),
    ]:
        with pytest.raises(ValueError):
            sample(HAND_X, SamplingParams(**kw), [], Script([0.5]))
    for x, hist in [
        ([1.0, math.nan], []),
        ([1.0, math.inf], []),
        ([[1.0, 2.0]], []),
        (HAND_X, [5]),
        (HAND_X, [-1]),
    ]:
        with pytest.raises(ValueError):
            sample(x, SamplingParams(), hist, Script([0.5]))
    SamplingParams(top_p=1.0, min_p=1.0, temperature=0.0).validate()
