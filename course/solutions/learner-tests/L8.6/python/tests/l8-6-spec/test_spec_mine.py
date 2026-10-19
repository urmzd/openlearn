"""My tests for L8.6 (rung R5). Exactness is checked by enumerating every
uniform on a grid (the toy distributions are multiples of 1/N, so the sum
is exact); greedy equivalence on a toy model whose logits cannot change
with chunking. They import only the contract."""

from __future__ import annotations

import math
from collections import defaultdict
from fractions import Fraction

import numpy as np
import pytest

from tinyllm.infer.generate import generate
from tinyllm.infer.sample import SamplingParams, request_rng
from tinyllm.infer.spec import (
    ModelDraft,
    NGramDraft,
    PromptLookupDraft,
    speculative_generate,
    verify_draft,
)
from tinyllm.lm.ngram import NGramLM
from tinyllm.num.rng import PCG32

GREEDY = SamplingParams(temperature=0.0, max_tokens=40)


class BagLM:
    def __init__(
        self,
        vocab: int = 8,
        d: int = 6,
        max_len: int = 96,
        seed: int = 0,
        scale: float = 3.0,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.vocab, self.d = vocab, d
        self.n_layers, self.n_kv_heads, self.d_head, self.max_len = 1, 1, d, max_len
        self.emb = rng.standard_normal((vocab, d))
        self.pos = 0.5 * rng.standard_normal((max_len, d))
        self.W = scale * rng.standard_normal((vocab, d))
        self.calls: list[
            tuple[list[int], list[int]]
        ] = []  # (positions, ids) per forward

    def forward(self, ids, positions=None, cache=None):
        ids = np.asarray(ids, dtype=np.int64)
        assert ids.ndim == 2 and ids.shape[0] == 1
        T = ids.shape[1]
        pos = (
            np.arange(T) if positions is None else np.asarray(positions, dtype=np.int64)
        )
        self.calls.append((pos.tolist(), ids[0].tolist()))
        k = self.emb[ids[0]].astype(np.float32)[None, None]  # [1, 1, T, d]
        if cache is not None:
            K, _ = cache.update(0, k, k)
            base = int(pos[0])
            assert K.shape[2] == base + T, (
                "the cache must hold exactly the positions before this chunk"
            )
        else:
            K, base = k, 0
            assert int(pos[0]) == 0
        keys = np.asarray(K, dtype=np.float64)[0, 0]
        out = np.empty((1, T, self.vocab))
        for j in range(T):
            t = base + j
            h = np.tanh(keys[: t + 1].mean(axis=0) + self.pos[t])
            out[0, j] = [float(np.dot(w, h)) for w in self.W]
        return out


class Letters:
    """ids 0..V-1 are the letters a, b, c, ...; no special tokens."""

    def __init__(self, vocab: int = 8) -> None:
        self.vocab_size = vocab
        self.special_ids: dict[str, int] = {}
        self.unk_id = None

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        return [ord(c) - 97 for c in text]

    def decode(self, ids, skip_special: bool = False) -> str:
        return "".join(chr(97 + int(i)) for i in ids)


def greedy_reference(model: BagLM, prompt: list[int], n: int) -> list[int]:
    """Plain greedy decoding without any cache: argmax, ties to the lowest id."""
    ids = list(prompt)
    for _ in range(n):
        if len(ids) >= model.max_len:
            break
        logits = model.forward(np.asarray([ids]))[0, -1]
        ids.append(int(np.argmax(logits)))
    return ids[len(prompt) :]


class NeedMore(Exception):
    pass


class Script:
    """Returns fixed uniforms in order; asking past the end raises NeedMore,
    which the enumerator turns into a branch."""

    def __init__(self, values=()):
        self.values = list(values)
        self.calls = 0

    def uniform(self) -> float:
        if self.calls >= len(self.values):
            raise NeedMore
        v = self.values[self.calls]
        self.calls += 1
        return v


def enumerate_draws(fn, n_grid: int) -> dict:
    """{outcome: probability as a Fraction} of fn(source) over independent
    uniforms, each taking the values (j + 1/2) / n_grid with weight 1/n_grid."""
    out: dict = defaultdict(Fraction)
    todo = [()]
    while todo:
        prefix = todo.pop()
        src = Script([(j + 0.5) / n_grid for j in prefix])
        try:
            res = fn(src)
        except NeedMore:
            todo += [prefix + (j,) for j in range(n_grid)]
            continue
        assert src.calls == len(prefix)
        out[res] += Fraction(1, n_grid ** len(prefix))
    return dict(out)


def logits_of(p) -> np.ndarray:
    """Logits whose softmax is p (zeros become -inf, a token never drawn)."""
    p = np.asarray(p, dtype=np.float64)
    with np.errstate(divide="ignore"):
        return np.log(p)


T1 = SamplingParams(temperature=1.0)


# --- the worked example ----------------------------------------------------------------


def test_hand_example():
    P = [0.5, 0.25, 0.25]
    rows = np.stack([logits_of(P), logits_of([0.0, 0.0, 1.0])])
    Q = np.array([[0.25, 0.5, 0.25]])
    assert verify_draft(rows, [1], Q, T1, [], Script([0.7, 0.4])) == ([0], 0)
    src = Script([0.3, 0.9, 0.1])
    assert verify_draft(rows, [1], Q, T1, [], src) == ([1, 2], 1)
    assert src.calls == 3  # two per draft position, one for the bonus


def test_hand_example_greedy():
    rows = np.array([[0.0, 1.0, 3.0], [2.0, 1.0, 0.0], [0.0, 5.0, 1.0]])
    src = Script([])
    assert verify_draft(rows, [2, 1], None, GREEDY, [], src) == ([2, 0], 1)
    assert verify_draft(rows, [2, 0], None, GREEDY, [], src) == ([2, 0, 1], 2)
    assert verify_draft(rows[:1], [], None, GREEDY, [], src) == ([2], 0)
    assert src.calls == 0


# --- exactness by enumeration ------------------------------------------------------------


P5 = [Fraction(3, 8), Fraction(1, 4), Fraction(0), Fraction(1, 4), Fraction(1, 8)]
Q5 = [Fraction(1, 8), Fraction(1, 2), Fraction(1, 4), Fraction(0), Fraction(1, 8)]


def test_one_token_output_is_the_target_exactly():
    rows = np.stack([logits_of([float(x) for x in P5])] * 2)
    Qf = np.array([[float(x) for x in Q5]])
    dist: dict = defaultdict(Fraction)
    for x, qx in enumerate(Q5):
        if qx == 0:
            continue
        for (first,), w in enumerate_draws(
            lambda s, x=x: tuple(verify_draft(rows, [x], Qf, T1, [], s)[0][:1]), 8
        ).items():
            dist[first] += qx * w
    assert [dist[i] for i in range(5)] == P5


TARGET2 = {  # P(y2 | y1) for the second position, multiples of 1/4
    0: [Fraction(1, 4), Fraction(1, 4), Fraction(1, 2), Fraction(0), Fraction(0)],
    1: [Fraction(0), Fraction(3, 4), Fraction(0), Fraction(0), Fraction(1, 4)],
    3: [Fraction(1, 2), Fraction(1, 4), Fraction(1, 4), Fraction(0), Fraction(0)],
}
P1 = [Fraction(1, 4), Fraction(1, 2), Fraction(0), Fraction(1, 4), Fraction(0)]
Q1 = [Fraction(1, 2), Fraction(1, 4), Fraction(0), Fraction(1, 4), Fraction(0)]
Q2 = [Fraction(0), Fraction(1, 2), Fraction(1, 4), Fraction(0), Fraction(1, 4)]
# Every acceptance ratio and every residual's cumulative sum above is a
# multiple of 1/4, so a grid of 4 midpoints per uniform integrates exactly.


def test_two_token_draft_is_exact_at_each_position():
    first: dict = defaultdict(Fraction)
    second: dict = defaultdict(lambda: defaultdict(Fraction))
    for x1, q1 in enumerate(Q1):
        if q1 == 0 or x1 not in TARGET2:
            continue
        for x2, q2 in enumerate(Q2):
            if q2 == 0:
                continue
            rows = np.stack(
                [
                    logits_of([float(v) for v in P1]),
                    logits_of([float(v) for v in TARGET2[x1]]),
                    logits_of([1.0, 0, 0, 0, 0]),
                ]
            )
            Q = np.array([[float(v) for v in Q1], [float(v) for v in Q2]])
            res = enumerate_draws(
                lambda s, x1=x1, x2=x2: tuple(
                    verify_draft(rows, [x1, x2], Q, T1, [], s)[0]
                ),
                4,
            )
            for em, w in res.items():
                first[em[0]] += q1 * q2 * w
                if len(em) >= 2:
                    second[em[0]][em[1]] += q1 * q2 * w
    assert [first[i] for i in range(5)] == P1
    for y1, cond in second.items():
        total = sum(cond.values())
        assert [cond[i] / total for i in range(5)] == TARGET2[y1], f"after y1 = {y1}"


def test_deterministic_draft_is_exact():
    PD = [Fraction(1, 4)] * 4 + [Fraction(0)]  # residuals in thirds: grid of 24
    rows = np.stack([logits_of([float(x) for x in PD])] * 2)
    for x in range(5):
        res = enumerate_draws(
            lambda s, x=x: tuple(verify_draft(rows, [x], None, T1, [], s)[0][:1]), 24
        )
        assert [res.get((i,), Fraction(0)) for i in range(5)] == PD, f"draft {x}"


def test_penalties_see_the_accepted_drafts():
    rows = np.array([[3.0, 2.9, 0.0], [3.0, 2.9, 0.0]] + [[0.0, 0.0, 1.0]])
    p = SamplingParams(temperature=0.0, repetition_penalty=2.0)
    emitted, n = verify_draft(rows, [0, 0], None, p, [], Script([]))
    assert (emitted, n) == ([0, 1], 1)


def test_verify_checks_shapes():
    rows = np.zeros((2, 3))
    with pytest.raises(ValueError):
        verify_draft(rows, [0, 1], None, GREEDY, [], Script([]))
    with pytest.raises(ValueError):
        verify_draft(rows, [0], np.full((2, 3), 1 / 3), T1, [], Script([0.1, 0.1, 0.1]))


# --- drafts ----------------------------------------------------------------------------------


def test_prompt_lookup_hand_example():
    d = PromptLookupDraft(max_ngram=3)
    assert d.propose([1, 2, 3, 9, 1, 2], 3, None) == ([3, 9, 1], None)
    assert d.propose([5, 1, 7, 1, 8, 1], 2, None) == ([8, 1], None)
    assert d.propose([4, 4, 4], 5, None) == (
        [4],
        None,
    )  # the continuation may reach the suffix itself
    assert d.propose([1, 2, 3], 2, None) == ([], None)
    assert d.propose([1, 2, 1, 2], 0, None) == ([], None)
    assert d.propose([1, 2, 3, 7, 2, 3, 1, 2, 3], 2, None) == (
        [7, 2],
        None,
    )  # the longest n-gram first
    with pytest.raises(ValueError):
        PromptLookupDraft(max_ngram=1, min_ngram=2)


def test_ngram_draft_follows_the_model():
    rng = PCG32(3)
    seqs = [[rng.below(6) for _ in range(30)] for _ in range(20)]
    lm = NGramLM(3, vocab_size=6)
    lm.fit(seqs)
    d = NGramDraft(lm)
    ctx = [1, 4, 2]
    ids, probs = d.propose(ctx, 4, None)
    cur = list(ctx)
    for t in ids:
        lp = lm.logprobs(cur)
        assert t == int(np.argmax(lp))
        cur.append(t)
    assert probs is None and len(ids) == 4
    ids, probs = d.propose(ctx, 3, request_rng(0))
    cur = list(ctx)
    for t, row in zip(ids, probs):
        np.testing.assert_allclose(row, np.exp(lm.logprobs(cur)), rtol=0, atol=1e-12)
        assert row[t] > 0
        cur.append(t)


def test_model_draft_syncs_its_cache():
    m = BagLM(seed=5)
    d = ModelDraft(m)
    for ctx in ([1, 2, 3], [1, 2, 3], [1, 2, 3, 4, 4], [1, 2, 5], [6], [6, 0, 2, 2, 7]):
        ids, probs = d.propose(ctx, 4, None)
        assert ids == greedy_reference(m, ctx, 4) and probs is None
    ids, probs = d.propose([3, 1], 2, request_rng(1))
    assert probs.shape == (2, 8)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(2), rtol=0, atol=1e-12)
    with pytest.raises(ValueError):
        ModelDraft(m, temperature=0.0)


# --- the decoding loop ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "draft_kind", ["prompt-lookup", "ngram", "model-same", "model-other"]
)
def test_speculative_greedy_equals_greedy(draft_kind):
    target, tok = BagLM(seed=1), Letters()
    prompt = "abcabcabdabc"
    if draft_kind == "prompt-lookup":
        draft = PromptLookupDraft()
    elif draft_kind == "ngram":
        lm = NGramLM(2, vocab_size=8)
        lm.fit([tok.encode(prompt)])
        draft = NGramDraft(lm)
    else:
        draft = ModelDraft(target if draft_kind == "model-same" else BagLM(seed=9))
    g = speculative_generate(target, draft, tok, prompt, GREEDY, k=4)
    want = greedy_reference(BagLM(seed=1), tok.encode(prompt), 40)
    assert g.ids == want
    assert generate(BagLM(seed=1), tok, prompt, GREEDY).ids == want
    assert g.text == tok.decode(want) and g.finish_reason == "length"
    assert len(g.logprobs) == 40


def test_rejected_drafts_are_rolled_back():
    target, tok = BagLM(seed=1), Letters()
    g = speculative_generate(
        target, ModelDraft(BagLM(seed=9)), tok, "abcab", GREEDY, k=3
    )
    full = tok.encode("abcab") + g.ids
    assert (
        target.calls[0][0][0] == 0 and target.calls[0][1][:5] == full[:5]
    )  # the prompt first
    last = -1
    for pos, ids in target.calls:
        assert pos == list(range(pos[0], pos[0] + len(ids)))
        assert pos[0] > last
        assert ids[0] == full[pos[0]], (
            "a round must start by feeding the last kept token"
        )
        last = pos[0]
    assert 0 < g.stats["acceptance_rate"] < 1, "the test wants some rejections"


def test_stats_and_one_pass_per_round():
    tok = Letters()
    t = BagLM(seed=1)
    g = speculative_generate(t, ModelDraft(BagLM(seed=1)), tok, "abc", GREEDY, k=4)
    assert g.stats["target_calls"] == 8 and len(t.calls) == 8
    assert (
        g.stats["acceptance_rate"] == 1.0
        and g.stats["drafted"] == g.stats["accepted"] == 32
    )
    assert g.stats["completion_tokens"] == 40 and g.stats["prompt_tokens"] == 3
    g0 = speculative_generate(
        BagLM(seed=1), PromptLookupDraft(), tok, "abc", GREEDY, k=0
    )
    assert (
        g0.stats["target_calls"] == 40
        and g0.stats["drafted"] == 0
        and g0.stats["acceptance_rate"] == 0.0
    )


def test_eos_stop_and_budget():
    tok = Letters()
    want = greedy_reference(BagLM(seed=1), tok.encode("abc"), 40)
    eos = want[7]
    g = speculative_generate(
        BagLM(seed=1), ModelDraft(BagLM(seed=1)), tok, "abc", GREEDY, k=4, eos_ids=[eos]
    )
    assert g.ids == want[: want.index(eos)] and g.finish_reason == "stop"
    stop = tok.decode(want[5:7])
    p = SamplingParams(temperature=0.0, max_tokens=40, stop=[stop])
    g = speculative_generate(
        BagLM(seed=1), ModelDraft(BagLM(seed=1)), tok, "abc", p, k=4
    )
    full = tok.decode(want)
    assert g.text == full[: full.index(stop)] and g.finish_reason == "stop"
    for n in (1, 3, 7):
        g = speculative_generate(
            BagLM(seed=1),
            ModelDraft(BagLM(seed=1)),
            tok,
            "abc",
            SamplingParams(temperature=0.0, max_tokens=n),
            k=4,
        )
        assert g.ids == want[:n]


def test_bad_arguments():
    tok = Letters()
    with pytest.raises(ValueError):
        speculative_generate(BagLM(), PromptLookupDraft(), tok, "abc", GREEDY, k=-1)
    with pytest.raises(ValueError):
        speculative_generate(BagLM(), PromptLookupDraft(), tok, "", GREEDY)
    with pytest.raises(ValueError):
        speculative_generate(
            BagLM(max_len=4), PromptLookupDraft(), tok, "abcde", GREEDY
        )


def test_logprobs_are_the_targets():
    tok = Letters()
    g = speculative_generate(
        BagLM(seed=1), ModelDraft(BagLM(seed=9)), tok, "abcab", GREEDY, k=3
    )
    ref = generate(BagLM(seed=1), tok, "abcab", GREEDY)
    np.testing.assert_allclose(
        np.array(g.logprobs), np.array(ref.logprobs), rtol=0, atol=1e-12
    )
    assert all(math.isfinite(x) and x <= 0 for x in g.logprobs)
