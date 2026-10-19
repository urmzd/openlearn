"""My tests for L8.1 (rung R5). The oracle is spec/sampling.md transcribed
again in plain Python (lists, math.exp, explicit loops), fed the same
uniforms as the code under test through a scripted source, on random logits
with ties, masks, and every processor. They import only the contract."""

import math
from collections import Counter

import numpy as np
import pytest
from tinyllm.infer.sample import (
    SamplingParams,
    process_logits,
    request_rng,
    sample,
    sampled_entropy,
    sampling_distribution,
    token_logprobs,
)
from tinyllm.num.rng import PCG32


class Fixed:
    def __init__(self, us):
        self.us, self.n = list(us), 0

    def uniform(self):
        assert self.n < len(self.us), "drew too many uniforms"
        self.n += 1
        return self.us[self.n - 1]


def loop_sum(xs):
    s = 0.0
    for v in xs:
        s += v
    return s


def soft(l, ids):
    ids = sorted(ids)
    m = max(l[i] for i in ids)
    e = [math.exp(l[i] - m) for i in ids]
    z = loop_sum(e)
    return {i: v / z for i, v in zip(ids, e)}


def oracle(x, p, prompt, out, u):
    """Returns (id, logprob, q as a dict over the kept ids)."""
    l = [float(v) for v in x]
    if p.repetition_penalty != 1.0:
        for i in sorted(set(prompt) | set(out)):
            l[i] = (
                l[i] / p.repetition_penalty if l[i] > 0 else l[i] * p.repetition_penalty
            )
    for i, c in sorted(Counter(out).items()):
        l[i] = l[i] - p.frequency_penalty * c - p.presence_penalty
    m = max(l)
    z = loop_sum([math.exp(v - m) for v in l])
    lp = [v - m - math.log(z) for v in l]
    if p.temperature == 0:
        best = max(range(len(l)), key=lambda i: (l[i], -i))
        return best, lp[best], {best: 1.0}
    t = [v / p.temperature for v in l]
    order = sorted(
        (i for i in range(len(t)) if t[i] != -math.inf), key=lambda i: (-t[i], i)
    )
    if 0 < p.top_k < len(order):
        order = order[: p.top_k]
    if p.top_p < 1:
        q = soft(t, order)
        s, cut = 0.0, len(order)
        for n, i in enumerate(order):
            s += q[i]
            if s >= p.top_p:
                cut = n + 1
                break
        order = order[:cut]
    if p.min_p > 0:
        q = soft(t, order)
        top = max(q.values())
        order = [i for i in order if q[i] >= p.min_p * top]
    q = soft(t, order)
    c = 0.0
    for i in sorted(order):
        c += q[i]
        if u < c:
            return i, lp[i], q
    return max(order), lp[max(order)], q


def random_case(g, V):
    x = [float(np.float32((g.uniform() * 2 - 1) * 5)) for _ in range(V)]
    for _ in range(3):  # exact ties
        x[g.below(V)] = x[g.below(V)]
    if g.below(3) == 0:
        for i in range(V):
            if g.below(4) == 0:
                x[i] = -math.inf
        x[g.below(V)] = 1.0
    p = SamplingParams(
        temperature=[0.0, 0.5, 1.0, 1.7][g.below(4)],
        top_k=[0, 1, 3, 8, 50][g.below(5)],
        top_p=[1.0, 0.9, 0.6, 0.3][g.below(4)],
        min_p=[0.0, 0.05, 0.3][g.below(3)],
        repetition_penalty=[1.0, 1.3, 0.8][g.below(3)],
        presence_penalty=[0.0, 0.4, -0.3][g.below(3)],
        frequency_penalty=[0.0, 0.25][g.below(2)],
    )
    prompt = [g.below(V) for _ in range(g.below(5))]
    out = [g.below(V) for _ in range(g.below(6))]
    return x, p, prompt, out


def test_matches_the_oracle_on_random_cases():
    g = PCG32(2026, 1)
    for _ in range(400):
        V = 2 + g.below(30)
        x, p, prompt, out = random_case(g, V)
        u = g.uniform()
        want_id, want_lp, want_q = oracle(x, p, prompt, out, u)
        src = Fixed([u])
        tok, lp = sample(x, p, out, src, prompt=prompt)
        assert (tok, lp) == (want_id, want_lp)
        assert src.n == (0 if p.temperature == 0 else 1)
        q = sampling_distribution(x, p, out, prompt)
        assert {i: v for i, v in enumerate(q.tolist()) if v > 0} == {
            i: v for i, v in want_q.items() if v > 0
        }
        assert token_logprobs(x, p, out, prompt).tolist()[tok] == want_lp


def test_removed_ids_are_minus_inf_in_processed_logits():
    g = PCG32(7, 2)
    for _ in range(100):
        x, p, prompt, out = random_case(g, 12)
        if p.temperature == 0:
            continue
        pl = process_logits(x, p, out, prompt)
        q = sampling_distribution(x, p, out, prompt)
        assert np.isneginf(pl[q == 0]).all() and np.isfinite(pl[q > 0]).all()


def test_sequential_sums_on_large_vocabularies():
    g = PCG32(11, 3)
    for _ in range(100):
        x = [g.uniform() * 8 for _ in range(400)]
        q = sampling_distribution(x, SamplingParams())
        want = soft(x, range(400))
        assert q.tolist() == [want[i] for i in range(400)]


def test_nucleus_boundary_and_crossing_token():
    x = [0.0] * 4
    assert [
        int((sampling_distribution(x, SamplingParams(top_p=tp)) > 0).sum())
        for tp in (0.25, 0.5, 0.6, 1.0)
    ] == [1, 2, 3, 4]


def test_request_stream_and_greedy_draws():
    for s in (0, 1, 99, 2**40):
        r, want = request_rng(s), PCG32(s).substream("sample")
        assert [r.uniform() for _ in range(3)] == [want.uniform() for _ in range(3)]
    tok, _ = sample([0.0, 2.0, 2.0], SamplingParams(temperature=0.0), [], Fixed([]))
    assert tok == 1


def test_entropy_matches_the_kept_distribution():
    x = [1.0, 2.0, 3.0, 0.5]
    p = SamplingParams(top_k=2)
    q = sampling_distribution(x, p)
    assert sampled_entropy(x, p) == pytest.approx(
        -sum(v * math.log(v) for v in q if v > 0), rel=1e-12
    )
    assert sampled_entropy(x, SamplingParams(temperature=0.0)) == 0.0


def test_bad_parameters_are_refused():
    for kw in (
        dict(top_p=0.0),
        dict(min_p=1.01),
        dict(repetition_penalty=0.0),
        dict(temperature=-1.0),
    ):
        with pytest.raises(ValueError):
            sample([1.0, 2.0], SamplingParams(**kw), [], Fixed([0.5]))
    with pytest.raises(ValueError):
        sample([1.0, 2.0], SamplingParams(), [7], Fixed([0.5]))
    SamplingParams(min_p=1.0).validate()
