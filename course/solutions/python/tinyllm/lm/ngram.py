"""tinyllm.lm.ngram (L2.1): an n-gram language model with interpolated
modified Kneser-Ney smoothing.

Counting n-grams and dividing gives the maximum-likelihood model, which puts
probability 0 on everything it has not seen and so has infinite perplexity
on any new text. Kneser-Ney fixes this in two moves. Absolute discounting
takes a little mass D from every seen count and hands it to a lower-order
model, interpolated at every context. And the lower-order model is not
built from how often a word occurs but from how many different words it
follows (its continuation count): "Francisco" is frequent but only ever
follows "San", so as a fresh continuation it deserves little mass. The
modified variant (Chen and Goodman 1998) uses three discounts, for counts 1,
2, and 3 or more, estimated from the counts of counts.

The model is the baseline every neural model in the zoo must beat (L6.7),
the draft model of speculative decoding (L8.6), and the corpus pipeline's
perplexity filter in the capstone.

Contract: contracts/py/tinyllm/lm/ngram.pyi.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Literal, Union

import numpy as np
from numpy.typing import NDArray

from tinyllm.info.ppl import NLLAccumulator
from tinyllm.io.safetensors import load_safetensors, save_safetensors
from tinyllm.prob.mle import ney_discount

BOS = -1  # the start symbol <s>: a context symbol, never a token
FALLBACK_DISCOUNTS = (0.5, 1.0, 1.5)


class NGramLM:
    """Interpolated (modified) Kneser-Ney n-gram model over ids 0 .. V-1."""

    def __init__(
        self,
        n: int,
        discount: Union[float, Literal["modified"]] = "modified",
        vocab_size: int | None = None,
    ) -> None:
        if not isinstance(n, int) or n < 1:
            raise ValueError(f"NGramLM: the order n must be an int >= 1, got {n!r}")
        if discount != "modified":
            if isinstance(discount, str) or not 0.0 < float(discount) <= 1.0:
                raise ValueError(
                    f"NGramLM: discount must be 'modified' or 0 < d <= 1, got {discount!r}"
                )
            discount = float(discount)
        if vocab_size is not None and vocab_size < 1:
            raise ValueError(f"NGramLM: vocab_size must be >= 1, got {vocab_size}")
        self.n = n
        self.discount = discount
        self._fixed_vocab = vocab_size
        self.vocab_size = vocab_size or 0
        self._raw: dict[int, Counter] = {}
        self._fitted = False

    # ---- fitting -----------------------------------------------------------

    def fit(self, sequences: Iterable[Sequence[int]]) -> None:
        raw: dict[int, Counter] = {k: Counter() for k in range(1, self.n + 1)}
        biggest = -1
        for seq in sequences:
            ids = [int(t) for t in seq]
            if not ids:
                continue
            lo, hi = min(ids), max(ids)
            if lo < 0 or (self._fixed_vocab is not None and hi >= self._fixed_vocab):
                raise ValueError(
                    f"NGramLM.fit: token ids must be in 0 .. {self._fixed_vocab or 'V'} - 1, got {lo} .. {hi}"
                )
            biggest = max(biggest, hi)
            padded = [BOS] + ids
            # Every k-gram that ends at a token: position i of padded (i >= 1)
            # ends grams of every length k <= i + 1, <s> only ever first.
            for i in range(1, len(padded)):
                for k in range(1, min(self.n, i + 1) + 1):
                    raw[k][tuple(padded[i - k + 1 : i + 1])] += 1
        if biggest < 0:
            raise ValueError("NGramLM.fit: no sequence holds a token")
        self.vocab_size = self._fixed_vocab or biggest + 1
        self._build(raw)

    def _build(self, raw: dict[int, Counter]) -> None:
        """Adjusted counts, discounts, and per-history totals from raw counts."""
        self._raw = raw
        n = self.n
        adj: dict[int, dict[tuple, int]] = {}
        for k in range(n, 0, -1):
            if k == n:
                adj[k] = dict(raw[k])
                continue
            # Continuation counts: how many distinct symbols precede g.
            cont: Counter = Counter()
            for g in raw[k + 1]:
                cont[g[1:]] += 1
            adj[k] = {g: (c if g[0] == BOS else cont[g]) for g, c in raw[k].items()}
        self._disc: dict[int, tuple[float, float, float]] = {}
        self._hist: dict[int, dict[tuple, tuple[dict[int, int], float, float]]] = {}
        for k in range(1, n + 1):
            D = self._estimate(adj[k])
            self._disc[k] = D
            by_hist: dict[tuple, dict[int, int]] = {}
            for g, a in adj[k].items():
                by_hist.setdefault(g[:-1], {})[g[-1]] = a
            table = {}
            for h, nxt in by_hist.items():
                total = float(sum(nxt.values()))
                n1 = sum(1 for a in nxt.values() if a == 1)
                n2 = sum(1 for a in nxt.values() if a == 2)
                n3 = sum(1 for a in nxt.values() if a >= 3)
                gamma = (D[0] * n1 + D[1] * n2 + D[2] * n3) / total
                table[h] = (nxt, total, gamma)
            self._hist[k] = table
        self._fitted = True

    def _estimate(self, adj_k: dict[tuple, int]) -> tuple[float, float, float]:
        """Chen and Goodman's three discounts from the counts of counts."""
        if self.discount != "modified":
            d = float(self.discount)
            return (d, d, d)
        nj = Counter(a for a in adj_k.values() if 1 <= a <= 4)
        n1, n2, n3, n4 = nj[1], nj[2], nj[3], nj[4]
        if min(n1, n2, n3, n4) == 0:
            return FALLBACK_DISCOUNTS
        Y = ney_discount(adj_k)  # n1 / (n1 + 2 n2)
        D = (1.0 - 2.0 * Y * n2 / n1, 2.0 - 3.0 * Y * n3 / n2, 3.0 - 4.0 * Y * n4 / n3)
        if not all(0.0 < d <= j for d, j in zip(D, (1, 2, 3))):
            return FALLBACK_DISCOUNTS
        return D

    # ---- queries -----------------------------------------------------------

    def _ready(self) -> None:
        if not self._fitted:
            raise ValueError("NGramLM: call fit (or load) first")

    def _padded(self, context: Sequence[int]) -> tuple:
        """The symbols the model conditions on: <s> + context at a sequence
        start, else the last n - 1 tokens."""
        ctx = tuple(int(t) for t in context)
        for t in ctx:
            self._check_id(t)
        if len(ctx) < self.n - 1:
            return (BOS,) + ctx
        return ctx[len(ctx) - (self.n - 1) :]

    def _check_id(self, t: int) -> None:
        if not 0 <= t < self.vocab_size:
            raise ValueError(
                f"NGramLM: token id {t} is outside 0 .. {self.vocab_size - 1}"
            )

    def discounts(self, order: int) -> tuple[float, float, float]:
        self._ready()
        if order not in self._disc:
            raise ValueError(
                f"NGramLM.discounts: order must be 1 .. {self.n}, got {order}"
            )
        return self._disc[order]

    def prob(self, context: Sequence[int], token: int) -> float:
        self._ready()
        self._check_id(int(token))
        w = int(token)
        hist = self._padded(context)
        K = min(self.n, len(hist) + 1)
        p = 1.0 / self.vocab_size
        for k in range(1, K + 1):
            h = hist[len(hist) - (k - 1) :] if k > 1 else ()
            entry = self._hist[k].get(h)
            if entry is None:
                continue  # unseen history: keep the lower-order estimate
            nxt, total, gamma = entry
            D = self._disc[k]
            a = nxt.get(w, 0)
            d = 0.0 if a == 0 else D[min(a, 3) - 1]
            p = max(a - d, 0.0) / total + gamma * p
        return p

    def logprobs(self, context: Sequence[int]) -> NDArray:
        self._ready()
        hist = self._padded(context)
        V = self.vocab_size
        K = min(self.n, len(hist) + 1)
        p = np.full(V, 1.0 / V)
        for k in range(1, K + 1):
            h = hist[len(hist) - (k - 1) :] if k > 1 else ()
            entry = self._hist[k].get(h)
            if entry is None:
                continue
            nxt, total, gamma = entry
            D = self._disc[k]
            a = np.zeros(V)
            idx = np.fromiter(nxt.keys(), dtype=np.int64, count=len(nxt))
            a[idx] = np.fromiter(nxt.values(), dtype=np.float64, count=len(nxt))
            d = np.select([a == 0, a == 1, a == 2], [0.0, D[0], D[1]], default=D[2])
            p = np.maximum(a - d, 0.0) / total + gamma * p
        with np.errstate(divide="ignore"):
            return np.log(p)

    def nll(self, ids: Sequence[int]) -> NDArray:
        seq = [int(t) for t in ids]
        m = self.n - 1
        # seq[max(0, t - m) : t] is the whole prefix while it is shorter than
        # n - 1 (a sequence start, so <s> is added) and the last n - 1 tokens
        # after that: the same history as seq[:t], without the O(t) copy.
        return np.array(
            [
                -math.log(self.prob(seq[max(0, t - m) : t], seq[t]))
                for t in range(len(seq))
            ],
            dtype=np.float64,
        )

    def perplexity(self, ids: Sequence[int]) -> float:
        if len(ids) == 0:
            raise ValueError("NGramLM.perplexity: empty sequence")
        acc = NLLAccumulator()
        acc.add(self.nll(ids))
        return acc.result()["ppl"]

    # ---- persistence -------------------------------------------------------

    def save(self, path: str) -> None:
        self._ready()
        tensors = {}
        for k in range(1, self.n + 1):
            grams = sorted(self._raw[k])
            tensors[f"order{k}.grams"] = np.array(grams, dtype=np.int32).reshape(
                len(grams), k
            )
            tensors[f"order{k}.counts"] = np.array(
                [self._raw[k][g] for g in grams], dtype=np.int32
            )
        meta = {
            "tl_arch": "ngram",
            "n": str(self.n),
            "discount": "modified"
            if self.discount == "modified"
            else repr(float(self.discount)),
            "vocab_size": str(self.vocab_size),
        }
        save_safetensors(path, tensors, meta)

    @classmethod
    def load(cls, path: str) -> NGramLM:
        tensors, meta = load_safetensors(path)
        if meta.get("tl_arch") != "ngram":
            raise ValueError(f"NGramLM.load: {path} is not an n-gram model")
        disc = meta["discount"]
        lm = cls(
            int(meta["n"]),
            "modified" if disc == "modified" else float(disc),
            int(meta["vocab_size"]),
        )
        raw = {}
        for k in range(1, lm.n + 1):
            grams = tensors[f"order{k}.grams"].reshape(-1, k)
            counts = tensors[f"order{k}.counts"]
            raw[k] = Counter(
                {tuple(int(x) for x in g): int(c) for g, c in zip(grams, counts)}
            )
        lm._build(raw)
        return lm
