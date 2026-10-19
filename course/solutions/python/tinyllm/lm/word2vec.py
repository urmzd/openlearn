"""tinyllm.lm.word2vec (L2.3): skip-gram with negative sampling, and PPMI-SVD.

"You shall know a word by the company it keeps" (Firth). Two routes from
co-occurrence to dense word vectors:

- Skip-gram with negative sampling (SGNS, Mikolov et al. 2013) slides a
  window over the text and, for every (center, context) pair, nudges the
  center's vector toward the context's output vector and away from a few
  random "negative" words drawn from the unigram distribution raised to
  0.75. The gradient is three lines of algebra, so training is manual
  sparse SGD on two tables, with no autograd.
- PPMI-SVD (Levy and Goldberg 2014) counts the same co-occurrences, turns
  them into positive pointwise mutual information (M11.4), and keeps the
  top singular directions (M03.5). Levy and Goldberg showed SGNS implicitly
  factorizes a shifted PMI matrix, so the explicit factorization is the
  baseline the learned vectors are compared with.

The model zoo (L6.7) scores both by word similarity (Spearman against gold
scores), and L3.6 can start its embedding table from these vectors.

Contract: contracts/py/tinyllm/lm/word2vec.pyi.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tinyllm.info.pmi import ppmi
from tinyllm.linalg.inner import cosine_sim, normalize, topk_cosine
from tinyllm.linalg.svd import svd
from tinyllm.num.rng import PCG32
from tinyllm.prob.sampling import AliasTable

NOISE_POWER = 0.75
LR_FLOOR = 1e-4


def word_vocab(words: Sequence[str], min_count: int = 1) -> tuple[list[str], NDArray]:
    if min_count < 1:
        raise ValueError(f"word_vocab: min_count must be >= 1, got {min_count}")
    counts = Counter(words)
    vocab = sorted(
        (w for w, c in counts.items() if c >= min_count), key=lambda w: (-counts[w], w)
    )
    index = {w: i for i, w in enumerate(vocab)}
    ids = np.array([index[w] for w in words if w in index], dtype=np.int64)
    return vocab, ids


def _neg_log_sigmoid(x: NDArray) -> NDArray:
    """-log sigma(x) = softplus(-x), finite for every finite x."""
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(-x, 0.0)


def _sigmoid(x: NDArray) -> NDArray:
    """sigma(x) without overflow: exp of a non-positive number only."""
    e = np.exp(-np.abs(x))
    return np.where(x >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def sgns_loss(
    v: ArrayLike, u_pos: ArrayLike, u_neg: ArrayLike
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    v = np.asarray(v, dtype=np.float64)
    up = np.asarray(u_pos, dtype=np.float64)
    un = np.asarray(u_neg, dtype=np.float64)
    if (
        v.ndim != 2
        or up.shape != v.shape
        or un.ndim != 3
        or un.shape[0] != v.shape[0]
        or un.shape[2] != v.shape[1]
    ):
        raise ValueError(
            f"sgns_loss: want v, u_pos [B, d] and u_neg [B, K, d], got {v.shape}, {up.shape}, {un.shape}"
        )
    s = np.einsum("bd,bd->b", up, v)  # positive scores u_o . v_c
    t = np.einsum("bkd,bd->bk", un, v)  # negative scores u_k . v_c
    loss = _neg_log_sigmoid(s) + _neg_log_sigmoid(-t).sum(axis=1)
    a = _sigmoid(s) - 1.0  # d loss / d s
    b = _sigmoid(t)  # d loss / d t_j
    g_v = a[:, None] * up + np.einsum("bk,bkd->bd", b, un)
    g_pos = a[:, None] * v
    g_neg = b[:, :, None] * v[:, None, :]
    return loss, g_v, g_pos, g_neg


class SkipGramNS:
    def __init__(
        self,
        vocab: int,
        dim: int,
        n_neg: int = 5,
        window: int = 5,
        subsample_t: float = 1e-5,
        rng: Any = None,
    ) -> None:
        for name, x in (
            ("vocab", vocab),
            ("dim", dim),
            ("n_neg", n_neg),
            ("window", window),
        ):
            if int(x) < 1:
                raise ValueError(f"SkipGramNS: {name} must be >= 1, got {x}")
        if not subsample_t > 0:
            raise ValueError(f"SkipGramNS: subsample_t must be > 0, got {subsample_t}")
        self.vocab, self.dim = int(vocab), int(dim)
        self.n_neg, self.window = int(n_neg), int(window)
        self.subsample_t = float(subsample_t)
        self._rng = rng if rng is not None else PCG32(0).substream("init")
        # word2vec.c: input vectors uniform in [-0.5/dim, 0.5/dim), output vectors 0.
        u = np.array(
            [self._rng.uniform() for _ in range(self.vocab * self.dim)],
            dtype=np.float64,
        )
        self.W_in = ((u - 0.5) / self.dim).reshape(self.vocab, self.dim)
        self.W_out = np.zeros((self.vocab, self.dim), dtype=np.float64)
        self._counts: NDArray | None = None
        self._alias: AliasTable | None = None

    def set_counts(self, counts: ArrayLike) -> None:
        c = np.asarray(counts, dtype=np.float64)
        if c.shape != (self.vocab,):
            raise ValueError(
                f"set_counts: want counts [{self.vocab}], got shape {c.shape}"
            )
        if not np.isfinite(c).all() or (c < 0).any() or c.sum() <= 0:
            raise ValueError(
                "set_counts: counts must be finite, non-negative, with a positive sum"
            )
        self._counts = c.copy()
        self._alias = AliasTable(self.noise_probs())

    def _need_counts(self) -> NDArray:
        if self._counts is None:
            raise RuntimeError("SkipGramNS: call set_counts (or fit) first")
        return self._counts

    def noise_probs(self) -> NDArray:
        w = self._need_counts() ** NOISE_POWER
        return w / w.sum()

    def keep_probs(self) -> NDArray:
        c = self._need_counts()
        f = c / c.sum()  # a frequency, not a raw count
        keep = np.ones(self.vocab, dtype=np.float64)
        seen = f > 0
        keep[seen] = np.minimum(1.0, np.sqrt(self.subsample_t / f[seen]))
        return keep

    def _ids(self, ids: ArrayLike, what: str) -> NDArray:
        a = np.asarray(ids)
        if a.ndim != 1 or (a.size and not np.issubdtype(a.dtype, np.integer)):
            raise ValueError(f"{what}: want 1-D integer ids, got {a.dtype} {a.shape}")
        a = a.astype(np.int64)
        if a.size and (a.min() < 0 or a.max() >= self.vocab):
            raise ValueError(
                f"{what}: ids must be in [0, {self.vocab}), got {a.min()} .. {a.max()}"
            )
        return a

    def pairs(self, ids: ArrayLike) -> tuple[NDArray, NDArray]:
        a = self._ids(ids, "pairs")
        keep = self.keep_probs()
        kept = [int(w) for w in a.tolist() if self._rng.uniform() < keep[w]]
        centers: list[int] = []
        contexts: list[int] = []
        n = len(kept)
        for j in range(n):
            # dynamic window: near words count more
            r = 1 + self._rng.below(self.window)
            for k in range(max(0, j - r), min(n, j + r + 1)):
                if k != j:
                    centers.append(kept[j])
                    contexts.append(kept[k])
        return np.array(centers, dtype=np.int64), np.array(contexts, dtype=np.int64)

    def negatives(self, n: int) -> NDArray:
        if n < 0:
            raise ValueError(f"negatives: n must be >= 0, got {n}")
        self._need_counts()
        return np.asarray(self._alias.sample(self._rng, n), dtype=np.int64)

    def step(self, centers: ArrayLike, contexts: ArrayLike, lr: float) -> float:
        c = self._ids(centers, "step")
        o = self._ids(contexts, "step")
        if c.size == 0 or c.shape != o.shape:
            raise ValueError(
                f"step: want two non-empty id arrays of one length, got {c.shape} and {o.shape}"
            )
        if not lr >= 0:
            raise ValueError(f"step: lr must be >= 0, got {lr}")
        negs = self.negatives(c.size * self.n_neg).reshape(c.size, self.n_neg)
        loss, g_v, g_pos, g_neg = sgns_loss(
            self.W_in[c], self.W_out[o], self.W_out[negs]
        )
        # Scatter-add: a word that appears twice in the batch gets both updates.
        np.add.at(self.W_in, c, -lr * g_v)
        np.add.at(self.W_out, o, -lr * g_pos)
        np.add.at(self.W_out, negs, -lr * g_neg)
        return float(loss.mean())

    def fit(
        self, ids: ArrayLike, epochs: int, batch_size: int, lr: float
    ) -> list[float]:
        if epochs < 1 or batch_size < 1 or not lr > 0:
            raise ValueError(
                f"fit: want epochs >= 1, batch_size >= 1, lr > 0, got {epochs}, {batch_size}, {lr}"
            )
        a = self._ids(ids, "fit")
        self.set_counts(np.bincount(a, minlength=self.vocab))
        out: list[float] = []
        for e in range(epochs):
            cen, ctx = self.pairs(a)
            if cen.size == 0:
                raise ValueError("fit: the sequence yields no (center, context) pairs")
            starts = range(0, cen.size, batch_size)
            n = len(starts)
            losses = []
            for b, i in enumerate(starts):
                rate = lr * max(LR_FLOOR, 1.0 - (e + b / n) / epochs)
                losses.append(
                    self.step(cen[i : i + batch_size], ctx[i : i + batch_size], rate)
                )
            out.append(float(np.mean(losses)))
        return out

    def embeddings(self) -> NDArray:
        return self.W_in.copy()


def cooccurrence(ids: ArrayLike, vocab: int, window: int) -> NDArray:
    a = np.asarray(ids)
    if a.ndim != 1 or (a.size and not np.issubdtype(a.dtype, np.integer)):
        raise ValueError(f"cooccurrence: want 1-D integer ids, got {a.dtype} {a.shape}")
    if window < 1:
        raise ValueError(f"cooccurrence: window must be >= 1, got {window}")
    a = a.astype(np.int64)
    if a.size and (a.min() < 0 or a.max() >= vocab):
        raise ValueError(f"cooccurrence: ids must be in [0, {vocab})")
    m = np.zeros((vocab, vocab), dtype=np.float64)
    for d in range(1, window + 1):  # every pair at distance d, in both directions
        np.add.at(m, (a[:-d], a[d:]), 1.0)
        np.add.at(m, (a[d:], a[:-d]), 1.0)
    return m


def ppmi_svd_embeddings(cooc: ArrayLike, dim: int) -> NDArray:
    m = ppmi(cooc)
    if not 1 <= dim <= m.shape[0]:
        raise ValueError(
            f"ppmi_svd_embeddings: dim must be in 1 .. {m.shape[0]}, got {dim}"
        )
    u, s, _ = svd(m)
    return u[:, :dim] * np.sqrt(s[:dim])


def analogy(
    emb: ArrayLike, vocab: Sequence[str], a: str, b: str, c: str, k: int = 1
) -> list[str]:
    e = np.asarray(emb, dtype=np.float64)
    index = {w: i for i, w in enumerate(vocab)}
    for w in (a, b, c):
        if w not in index:
            raise ValueError(f"analogy: {w!r} is not in the vocabulary")
    if not 1 <= k <= len(vocab) - 3:
        raise ValueError(f"analogy: k must be in 1 .. {len(vocab) - 3}, got {k}")
    ia, ib, ic = index[a], index[b], index[c]
    q = normalize(e[ib]) - normalize(e[ia]) + normalize(e[ic])
    idx, _ = topk_cosine(q, e, min(len(vocab), k + 3))
    out = [vocab[int(i)] for i in idx if int(i) not in (ia, ib, ic)]
    return out[:k]


def _ranks(x: NDArray) -> NDArray:
    """1-based ranks, ties sharing their average rank."""
    order = np.argsort(x, kind="stable")
    r = np.empty(x.size, dtype=np.float64)
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and x[order[j + 1]] == x[order[i]]:
            j += 1
        r[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def spearman(x: ArrayLike, y: ArrayLike) -> float:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.ndim != 1 or a.shape != b.shape or a.size < 2:
        raise ValueError(
            f"spearman: want two 1-D arrays of one length >= 2, got {a.shape} and {b.shape}"
        )
    ra, rb = _ranks(a), _ranks(b)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    if den == 0:
        raise ValueError("spearman: an input is constant")
    return float((ra * rb).sum() / den)


def word_similarity(
    emb: ArrayLike, vocab: Sequence[str], pairs: Sequence[tuple[str, str, float]]
) -> float:
    e = np.asarray(emb, dtype=np.float64)
    index = {w: i for i, w in enumerate(vocab)}
    sims, gold = [], []
    for a, b, score in pairs:
        if a in index and b in index:
            sims.append(float(cosine_sim(e[index[a]], e[index[b]])))
            gold.append(float(score))
    if len(sims) < 2:
        raise ValueError(
            f"word_similarity: {len(sims)} pair(s) covered by the vocabulary, need 2"
        )
    return spearman(sims, gold)
