"""Model test suite for L2.3: what each test protects. Written test-first;
the chapter's section 3 numbers are the first test.
They import only contract names."""

import math

import numpy as np
import pytest
from tinyllm.lm.word2vec import (
    SkipGramNS,
    analogy,
    cooccurrence,
    ppmi_svd_embeddings,
    sgns_loss,
    spearman,
    word_similarity,
    word_vocab,
)
from tinyllm.num.rng import PCG32

S1 = 1.0 / (1.0 + math.e)  # sigma(-1)


class Fixed:
    """uniform() returns u (after `skip` init draws of 0.5), below() returns b."""

    def __init__(self, u, b=(), skip=0):
        self.u, self.b = [0.5] * skip + list(u), list(b)

    def uniform(self):
        return self.u.pop(0)

    def below(self, n):
        return self.b.pop(0)


def test_hand_example_loss_and_gradients():
    """v = (0.5, -1), u_o = (2, 1), u_k = (0, 1): s = 0, t = -1."""
    loss, gv, gp, gn = sgns_loss([[0.5, -1.0]], [[2.0, 1.0]], [[[0.0, 1.0]]])
    np.testing.assert_allclose(
        loss, [math.log(2) + math.log1p(math.exp(-1))], rtol=1e-12
    )
    np.testing.assert_allclose(gv, [[-1.0, -0.5 + S1]], rtol=1e-12)
    np.testing.assert_allclose(gp, [[-0.25, 0.5]], rtol=1e-12)
    np.testing.assert_allclose(gn, [[[0.5 * S1, -S1]]], rtol=1e-12)


def test_gradients_by_finite_differences():
    g = PCG32(0)

    def arr(*shape):
        return np.array(
            [2 * g.uniform() - 1 for _ in range(int(np.prod(shape)))]
        ).reshape(shape)

    v, up, un = arr(3, 4), arr(3, 4), arr(3, 2, 4)
    _, gv, gp, gn = sgns_loss(v, up, un)
    for x, gx in ((v, gv), (up, gp), (un, gn)):
        for idx in [(0, 0), (2, 3)] if x.ndim == 2 else [(0, 1, 2), (2, 0, 0)]:
            old = x[idx]
            x[idx] = old + 1e-6
            hi = sgns_loss(v, up, un)[0].sum()
            x[idx] = old - 1e-6
            lo = sgns_loss(v, up, un)[0].sum()
            x[idx] = old
            assert abs((hi - lo) / 2e-6 - gx[idx]) < 1e-6


def test_large_scores_stay_finite():
    loss, gv, *_ = sgns_loss([[10.0, 0.0]], [[-100.0, 0.0]], [[[0.0, 1.0]]])
    np.testing.assert_allclose(loss, [1000 + math.log(2)], rtol=1e-12)
    assert np.isfinite(gv).all()


def test_noise_and_keep():
    m = SkipGramNS(4, 2, subsample_t=0.01, rng=PCG32(0))
    m.set_counts([16, 0, 4, 12])
    w = np.array([16, 0, 4, 12]) ** 0.75
    np.testing.assert_allclose(m.noise_probs(), w / w.sum(), rtol=1e-12)
    np.testing.assert_allclose(m.keep_probs()[0], math.sqrt(0.01 / 0.5), rtol=1e-12)
    assert (m.negatives(500) != 1).all()


def test_pairs_dynamic_window_and_subsampling():
    m = SkipGramNS(
        4, 2, window=2, subsample_t=1.0, rng=Fixed([0.0] * 4, [1, 0, 0, 1], skip=8)
    )
    m.set_counts([1, 1, 1, 1])
    c, o = m.pairs([0, 1, 2, 3])
    assert list(zip(c.tolist(), o.tolist())) == [
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 2),
        (2, 1),
        (2, 3),
        (3, 1),
        (3, 2),
    ]
    m = SkipGramNS(
        3, 1, window=1, subsample_t=0.01, rng=Fixed([0.2] * 3, [0, 0], skip=3)
    )
    m.set_counts([18, 1, 1])
    c, o = m.pairs([1, 0, 2])
    assert list(zip(c.tolist(), o.tolist())) == [(1, 2), (2, 1)]


def test_step_adds_repeated_rows():
    a, b = (
        SkipGramNS(5, 3, n_neg=1, rng=PCG32(2)),
        SkipGramNS(5, 3, n_neg=1, rng=PCG32(2)),
    )
    for m in (a, b):
        m.set_counts([3, 3, 3, 3, 3])
        m.W_out[:] = 0.2
    c, o = np.array([1, 1]), np.array([2, 3])
    negs = a.negatives(2).reshape(2, 1)
    _, gv, gp, gn = sgns_loss(a.W_in[c], a.W_out[o], a.W_out[negs])
    b.step(c, o, 0.5)
    np.testing.assert_allclose(b.W_in[1], a.W_in[1] - 0.5 * (gv[0] + gv[1]), rtol=1e-12)
    np.testing.assert_array_equal(b.W_in[0], a.W_in[0])


def test_init_and_embeddings_are_the_input_table():
    m = SkipGramNS(4, 2, rng=PCG32(5))
    g = PCG32(5)
    want = (np.array([g.uniform() for _ in range(8)]) - 0.5) / 2
    np.testing.assert_array_equal(m.embeddings(), want.reshape(4, 2))
    assert (m.W_out == 0).all()


def test_fit_decays_the_learning_rate():
    ids = np.array([(i * 7 + i // 5) % 9 for i in range(60)])
    a = SkipGramNS(9, 4, n_neg=3, window=2, subsample_t=0.05, rng=PCG32(1))
    a.fit(ids, 2, 16, 0.3)
    b = SkipGramNS(9, 4, n_neg=3, window=2, subsample_t=0.05, rng=PCG32(1))
    b.set_counts(np.bincount(ids, minlength=9))
    for e in range(2):
        c, o = b.pairs(ids)
        starts = list(range(0, c.size, 16))
        for k, i in enumerate(starts):
            b.step(
                c[i : i + 16],
                o[i : i + 16],
                0.3 * max(1e-4, 1 - (e + k / len(starts)) / 2),
            )
    np.testing.assert_array_equal(a.W_in, b.W_in)


def test_cooccurrence_and_ppmi_svd():
    c = cooccurrence(np.array([0, 1, 2, 0]), 3, 1)
    assert c.tolist() == [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    cooc = np.array(
        [[0, 4, 1, 0], [4, 0, 0, 1], [1, 0, 0, 5], [0, 1, 5, 0]], dtype=float
    )
    e = ppmi_svd_embeddings(cooc, 2)
    cc = cooc.sum(0) ** 0.75
    with np.errstate(divide="ignore"):
        pmi = (
            np.log(cooc / cooc.sum())
            - np.log(cooc.sum(1) / cooc.sum())[:, None]
            - np.log(cc / cc.sum())[None, :]
        )
    u, s, _ = np.linalg.svd(np.maximum(np.nan_to_num(pmi, neginf=0.0), 0))
    want = u[:, :2] * np.sqrt(s[:2])
    np.testing.assert_allclose(np.abs(e @ e.T), np.abs(want @ want.T), atol=1e-9)


def test_analogy_excludes_the_query():
    vocab = ["man", "woman", "king", "queen", "apple"]
    e = np.array(
        [[1.0, 0, 0], [0, 1.0, 0], [1.0, 0, 1.0], [0.5, 1.0, 0.5], [0, 0, -1.0]]
    )
    assert analogy(e, vocab, "man", "king", "woman", k=2) == ["queen", "apple"]


def test_spearman_and_word_similarity():
    assert spearman([1, 2, 2, 3], [1, 3, 2, 4]) == pytest.approx(
        4.5 / math.sqrt(22.5), rel=1e-12
    )
    e = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    pairs = [("a", "b", 3), ("a", "c", 0), ("b", "c", 1), ("a", "zzz", 3)]
    assert word_similarity(e, ["a", "b", "c"], pairs) == pytest.approx(1.0)


def test_word_vocab():
    vocab, ids = word_vocab("b a d a b c a".split())
    assert vocab == ["a", "b", "c", "d"] and ids.tolist() == [1, 0, 3, 0, 1, 2, 0]
