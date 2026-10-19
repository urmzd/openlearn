"""My tests for L4.4 (rung R5: oracles. The oracle here is brute force:
every output of a tiny model, scored by my own log-softmax). They import
only the contract."""

import itertools
import math
from typing import NamedTuple

import numpy as np
import pytest
from tinyllm.infer.beam import beam_search, greedy_decode, select_state

EOS, A, B, BOS = 0, 1, 2, 9


def lsm(z):
    z = np.asarray(z, dtype=np.float64)
    m = z.max()
    if not np.isfinite(m):
        m = 0.0
    return z - m - np.log(np.exp(z - m).sum())


class LM:
    """Logits from a table keyed by the prefix (uniform when missing), or
    pseudo-random ones from a hash of the prefix."""

    def __init__(self, V, table=None, seed=None, eos_bias=0.0, masked=()):
        self.V, self.table, self.seed, self.eos_bias, self.masked = (
            V,
            table,
            seed,
            eos_bias,
            masked,
        )

    def logits(self, prefix):
        if self.table is not None:
            p = self.table.get(prefix)
            if p is None:
                return np.zeros(self.V)
            with np.errstate(divide="ignore"):
                return np.log(np.array(p, dtype=np.float64))
        h = (hash((self.seed,) + prefix) * 2654435761) % (2**32)
        z = np.cos(np.arange(self.V) * 1.7 + h * 1e-6) * 3.0 + np.sin(
            h % 97 + np.arange(self.V)
        )
        z[EOS] += self.eos_bias
        for v in self.masked:
            z[v] = -np.inf
        return z


def stepper(lm, limit=None):
    calls = []

    def step(state, y):
        y = np.asarray(y)
        assert len(state["p"]) == len(y)
        if limit is not None:
            assert len(y) <= limit
        calls.append(len(y))
        new = np.empty(
            len(y), dtype=object
        )  # an array, so select_state gathers its rows
        for i, (p, t) in enumerate(zip(state["p"], y)):
            new[i] = p + ((int(t),) if state["started"] else ())
        return np.stack([lm.logits(p) for p in new]), {"p": new, "started": True}

    first = np.empty(1, dtype=object)
    first[0] = ()
    return step, {"p": first, "started": False}, calls


def lp_of(lm, toks):
    return sum(lsm(lm.logits(tuple(toks[:i])))[t] for i, t in enumerate(toks))


HAND = LM(
    3, table={(): [0.1, 0.5, 0.4], (A,): [0.2, 0.1, 0.7], (B,): [0.9, 0.05, 0.05]}
)


def test_hand_example():
    """Beam 2 finds b eos (0.36); greedy commits to a b (0.35)."""
    step, init, calls = stepper(HAND, 2)
    hyps = beam_search(step, init, BOS, EOS, 2, 2, length_penalty=0.0)
    assert [h.tokens for h in hyps] == [[B, EOS], [A, B]]
    assert [h.finished for h in hyps] == [True, False]
    np.testing.assert_allclose(
        [h.logprob for h in hyps], [math.log(0.36), math.log(0.35)]
    )
    np.testing.assert_allclose(
        [h.score for h in hyps], [math.log(0.36), math.log(0.35)]
    )
    assert calls == [1, 2]
    step, init, _ = stepper(HAND)
    g = greedy_decode(step, init, BOS, EOS, 2)
    assert g.tokens == [A, B] and math.isclose(g.score, g.logprob)


def test_length_penalty_divides_by_length_including_eos():
    step, init, _ = stepper(HAND)
    hyps = beam_search(step, init, BOS, EOS, 2, 2, length_penalty=1.0)
    np.testing.assert_allclose(
        [h.score for h in hyps], [math.log(0.36) / 2, math.log(0.35) / 2]
    )


@pytest.mark.parametrize("alpha", [0.0, 1.0])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_exhaustive_equals_brute_force(seed, alpha):
    V, L = 3, 4
    lm = LM(V, seed=seed)
    outs = []
    for n in range(1, L + 1):
        for body in itertools.product([1, 2], repeat=n - 1):
            outs.append((list(body) + [EOS], True))
    for body in itertools.product([1, 2], repeat=L):
        outs.append((list(body), False))
    want = sorted(
        ((lp_of(lm, t) / len(t) ** alpha, t, f) for t, f in outs), key=lambda r: -r[0]
    )
    step, init, _ = stepper(lm, V**L)
    got = beam_search(step, init, BOS, EOS, V**L, L, length_penalty=alpha)
    assert [h.tokens for h in got] == [w[1] for w in want]
    assert [h.finished for h in got] == [w[2] for w in want]
    np.testing.assert_allclose([h.score for h in got], [w[0] for w in want], rtol=1e-12)


@pytest.mark.parametrize("seed", [4, 5, 6])
def test_beam_one_is_greedy(seed):
    lm = LM(5, seed=seed, eos_bias=-0.5)
    step, init, _ = stepper(lm, 1)
    got = beam_search(step, init, BOS, EOS, 1, 6, length_penalty=0.0)[0]
    toks = []
    for _ in range(6):
        v = int(np.argmax(lsm(lm.logits(tuple(toks)))))
        toks.append(v)
        if v == EOS:
            break
    assert got.tokens == toks


def test_never_more_rows_than_beam_and_beam_narrows():
    lm = LM(6, seed=7, eos_bias=2.0)
    step, init, calls = stepper(lm, 3)
    hyps = beam_search(step, init, BOS, EOS, 3, 20)
    assert len(hyps) <= 3 and len(calls) < 20
    assert all(b <= a for a, b in zip(calls[1:], calls[2:]))
    for h in hyps:
        assert h.tokens.count(EOS) == (1 if h.finished else 0)
        assert math.isclose(h.logprob, lp_of(lm, h.tokens), rel_tol=1e-12)


def test_stops_when_all_finished():
    lm = LM(3, table={(): [0.0, 0.6, 0.4], (A,): [1.0, 0, 0], (B,): [1.0, 0, 0]})
    step, init, calls = stepper(lm)
    hyps = beam_search(step, init, BOS, EOS, 2, 10, length_penalty=0.0)
    assert calls == [1, 2] and [h.tokens for h in hyps] == [[A, EOS], [B, EOS]]


@pytest.mark.parametrize("L", [1, 3])
def test_max_len_returns_unfinished(L):
    lm = LM(4, seed=8, masked=[EOS])
    step, init, calls = stepper(lm, 3)
    hyps = beam_search(step, init, BOS, EOS, 3, L)
    assert len(hyps) == 3 and all(len(h.tokens) == L and not h.finished for h in hyps)
    assert len(calls) == L


def test_score_ranks_and_length_penalty_can_flip_the_winner():
    lm = LM(
        3,
        table={
            (): [0.0, 0.3, 0.7],
            (A,): [1, 0, 0],
            (B,): [0, 0.1, 0.9],
            (B, B): [0, 0.1, 0.9],
            (B, B, B): [0.5, 0.25, 0.25],
            (B, A): [1, 0, 0],
            (B, B, A): [1, 0, 0],
        },
    )
    for alpha, first in ((0.0, [A, EOS]), (1.0, [B, B, B, EOS])):
        step, init, _ = stepper(lm)
        hyps = beam_search(step, init, BOS, EOS, 2, 4, length_penalty=alpha)
        assert hyps[0].tokens == first


def test_ties_lowest_id_and_unnormalized_logits():
    step, init, _ = stepper(LM(4, table={}))
    assert [
        h.tokens for h in beam_search(step, init, BOS, 3, 2, 2, length_penalty=0.0)
    ] == [[0, 0], [0, 1]]
    lm = LM(4, seed=9)
    step, init, _ = stepper(lm)
    want = beam_search(step, init, BOS, EOS, 3, 4)
    step2, init2, _ = stepper(lm)
    shifted = lambda s, y: (
        lambda z, st: (z + 7.0 * np.arange(1, len(z) + 1)[:, None], st)
    )(*step2(s, y))
    got = beam_search(shifted, init2, BOS, EOS, 3, 4)
    assert [h.tokens for h in got] == [h.tokens for h in want]


def test_masked_never_chosen():
    lm = LM(4, table={(): [0, 1.0, 0, 0], (1,): [1.0, 0, 0, 0]})
    step, init, calls = stepper(lm)
    hyps = beam_search(step, init, BOS, EOS, 3, 4)
    assert [h.tokens for h in hyps] == [[1, EOS]] and calls == [1, 1]


class P(NamedTuple):
    h: np.ndarray
    c: list


def test_select_state():
    h = np.arange(6.0).reshape(3, 2)
    got = select_state({"s": P(h, [h * 2, None]), "k": 4}, [2, 0, 2])
    assert type(got["s"]) is P and isinstance(got["s"].c, list)
    np.testing.assert_array_equal(got["s"].h, h[[2, 0, 2]])
    np.testing.assert_array_equal(got["s"].c[0], (h * 2)[[2, 0, 2]])
    assert got["s"].c[1] is None and got["k"] == 4


def test_validation():
    step, init, _ = stepper(HAND)
    with pytest.raises(ValueError):
        beam_search(step, init, BOS, EOS, 0, 3)
    with pytest.raises(ValueError):
        greedy_decode(
            lambda s, y: (np.full((len(y), 3), -np.inf), s), None, BOS, EOS, 3
        )
