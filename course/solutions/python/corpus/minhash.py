"""Near-duplicate detection and decontamination (data.04).

Two documents are near duplicates when their sets of word shingles overlap
a lot (Jaccard similarity). Comparing every pair is quadratic, so:

  signature   minhash: num_perm hash functions, keep each one's minimum over
              the shingles; equal entries estimate the Jaccard similarity
  LSH         split the signature into bands; documents that agree on a
              whole band become candidates (an S-curve in the similarity)
  confirm     keep candidates whose estimate reaches the threshold
  union-find  merge confirmed pairs into clusters; the smallest id is root

Decontamination drops every document that shares a word 13-gram with a
protected eval or validation set, so the model never trains on its tests.

Contract: contracts/py/corpus/minhash.pyi.
"""

from __future__ import annotations

import dataclasses
import json
import multiprocessing
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import AbstractSet, Iterable, Iterator, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from corpus.stage import Doc
from tinyllm.num.rng import PCG32

MERSENNE31 = (1 << 31) - 1
# FNV-1a 64 constants, as M06.3 defines them (fnv1a64 there hashes one string;
# shingle_hashes runs the same steps on many at once).
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
_WORD = re.compile(r"\w+")
_SKIP_KEYS = frozenset({"case_id", "id", "tags", "scorer_args"})


def words(text: str) -> list[str]:
    """re.findall(r"\\w+", text.lower())."""
    return _WORD.findall(text.lower())


def shingles(text: str, k: int = 5) -> set[str]:
    """The set of word k-shingles; one shingle for 1..k-1 words; empty for
    no words. ValueError for k < 1."""
    if k < 1:
        raise ValueError(f"shingle length k must be >= 1, got {k}")
    w = words(text)
    if not w:
        return set()
    if len(w) < k:
        return {" ".join(w)}
    return {" ".join(w[i : i + k]) for i in range(len(w) - k + 1)}


def jaccard(a: AbstractSet[str], b: AbstractSet[str]) -> float:
    """|a & b| / |a | b|; 0.0 when both are empty."""
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def shingle_hashes(shingles: Iterable[str]) -> NDArray:
    """uint64 [n]: fnv1a64(s.encode("utf-8")) mod MERSENNE31 per shingle.

    FNV-1a runs byte by byte, but every shingle runs the same steps, so the
    loop goes over byte positions with all shingles at once: numpy's uint64
    multiply wraps modulo 2^64, exactly what FNV needs."""
    data = [s.encode("utf-8") for s in shingles]
    if not data:
        return np.zeros(0, dtype=np.uint64)
    lengths = np.fromiter((len(b) for b in data), dtype=np.int64, count=len(data))
    width = int(lengths.max())
    grid = np.zeros((len(data), width), dtype=np.uint64)
    for i, b in enumerate(data):
        grid[i, : len(b)] = np.frombuffer(b, dtype=np.uint8)
    h = np.full(len(data), FNV_OFFSET, dtype=np.uint64)
    prime = np.uint64(FNV_PRIME)
    for j in range(width):
        live = lengths > j
        h[live] = (h[live] ^ grid[live, j]) * prime
    return h % np.uint64(MERSENNE31)


def perm_params(num_perm: int, seed: int) -> tuple[NDArray, NDArray]:
    """(a, b), uint64 [num_perm] each: a_i = 1 + below(P - 1), then
    b_i = below(P), interleaved, from one PCG32(seed)."""
    if num_perm < 1:
        raise ValueError(f"num_perm must be >= 1, got {num_perm}")
    rng = PCG32(seed)
    a = np.empty(num_perm, dtype=np.uint64)
    b = np.empty(num_perm, dtype=np.uint64)
    for i in range(num_perm):
        a[i] = 1 + rng.below(MERSENNE31 - 1)
        b[i] = rng.below(MERSENNE31)
    return a, b


def _signature(x: NDArray, a: NDArray, b: NDArray) -> NDArray:
    """min over shingle hashes x of (a_i * x + b_i) mod P, all P when empty."""
    if x.size == 0:
        return np.full(a.shape[0], MERSENNE31, dtype=np.uint64)
    h = (a[:, None] * x[None, :] + b[:, None]) % np.uint64(MERSENNE31)
    return h.min(axis=1)


def minhash(shingles: Iterable[str], num_perm: int = 128, seed: int = 0) -> NDArray:
    """uint64 [num_perm]: sig_i = min over shingles of (a_i x + b_i) mod P."""
    a, b = perm_params(num_perm, seed)
    return _signature(shingle_hashes(shingles), a, b)


def estimate(sig_a: NDArray, sig_b: NDArray) -> float:
    """Share of equal entries. ValueError when the shapes differ."""
    sig_a, sig_b = np.asarray(sig_a), np.asarray(sig_b)
    if sig_a.shape != sig_b.shape:
        raise ValueError(f"signature shapes differ: {sig_a.shape} vs {sig_b.shape}")
    return float(np.mean(sig_a == sig_b))


class LSH:
    """Banded locality-sensitive hashing over minhash signatures."""

    def __init__(self, bands: int = 16, rows: int = 8) -> None:
        """ValueError unless bands >= 1 and rows >= 1."""
        if bands < 1 or rows < 1:
            raise ValueError(f"bands and rows must be >= 1, got {bands} and {rows}")
        self.bands = bands
        self.rows = rows
        self._buckets: dict[tuple[int, bytes], list[str]] = {}
        self._keys: set[str] = set()

    @property
    def threshold(self) -> float:
        """(1 / bands) ** (1 / rows)."""
        return (1.0 / self.bands) ** (1.0 / self.rows)

    def probability(self, s: float) -> float:
        """1 - (1 - s**rows) ** bands."""
        return 1.0 - (1.0 - s**self.rows) ** self.bands

    def insert(self, key: str, sig: NDArray) -> None:
        """Index sig under key, one bucket per band."""
        sig = np.asarray(sig, dtype=np.uint64)
        if sig.shape != (self.bands * self.rows,):
            raise ValueError(
                f"signature has {sig.size} entries, LSH needs bands * rows = {self.bands * self.rows}"
            )
        if key in self._keys:
            raise ValueError(f"key {key!r} was inserted before")
        self._keys.add(key)
        if bool(np.all(sig == MERSENNE31)):
            return  # an empty text has no shingles: nobody's near duplicate
        for band in range(self.bands):
            part = sig[band * self.rows : (band + 1) * self.rows]
            self._buckets.setdefault((band, part.tobytes()), []).append(key)

    def candidates(self) -> list[tuple[str, str]]:
        """Sorted unique pairs (x, y), x < y, that share a band bucket."""
        pairs: set[tuple[str, str]] = set()
        for keys in self._buckets.values():
            if len(keys) < 2:
                continue
            ks = sorted(keys)
            for i in range(len(ks)):
                for j in range(i + 1, len(ks)):
                    pairs.add((ks[i], ks[j]))
        return sorted(pairs)


class UnionFind:
    """Disjoint sets of string keys whose root is always the smallest member."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        """The root of x's set, adding x when new; compresses the path."""
        parent = self._parent
        if x not in parent:
            parent[x] = x
            return x
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(self, x: str, y: str) -> str:
        """Merge the sets of x and y; the smaller root becomes the root."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return rx
        if ry < rx:
            rx, ry = ry, rx
        self._parent[ry] = rx
        return rx

    def groups(self) -> list[set[str]]:
        """Every set with at least two members, sorted by smallest member."""
        by_root: dict[str, set[str]] = {}
        for x in self._parent:
            by_root.setdefault(self.find(x), set()).add(x)
        return sorted((g for g in by_root.values() if len(g) > 1), key=min)


def _signature_chunk(args: tuple[list[str], int, int, int]) -> NDArray:
    """Signatures of one contiguous chunk (runs in a worker process)."""
    texts, num_perm, seed, k = args
    a, b = perm_params(num_perm, seed)
    out = np.empty((len(texts), num_perm), dtype=np.uint64)
    for i, t in enumerate(texts):
        out[i] = _signature(shingle_hashes(shingles(t, k)), a, b)
    return out


def signatures(
    texts: Sequence[str],
    *,
    num_perm: int = 128,
    seed: int = 0,
    k: int = 5,
    workers: int = 1,
) -> NDArray:
    """uint64 [len(texts), num_perm], identical for every workers value."""
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    texts = list(texts)
    if workers == 1 or len(texts) < 2:
        return _signature_chunk((texts, num_perm, seed, k))
    size = -(-len(texts) // workers)
    chunks = [
        (texts[i : i + size], num_perm, seed, k) for i in range(0, len(texts), size)
    ]
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        parts = list(pool.map(_signature_chunk, chunks))
    return np.vstack(parts)


def clusters(
    docs: Mapping[str, str],
    *,
    num_perm: int = 128,
    bands: int = 16,
    threshold: float = 0.8,
    k: int = 5,
    seed: int = 0,
    workers: int = 1,
) -> list[set[str]]:
    """Near-duplicate clusters of {id: text}, sorted by smallest id."""
    if bands < 1 or num_perm % bands:
        raise ValueError(f"bands ({bands}) must divide num_perm ({num_perm})")
    keys = sorted(docs)
    sigs = signatures(
        [docs[x] for x in keys], num_perm=num_perm, seed=seed, k=k, workers=workers
    )
    lsh = LSH(bands, num_perm // bands)
    row = {}
    for i, key in enumerate(keys):
        lsh.insert(key, sigs[i])
        row[key] = i
    uf = UnionFind()
    for x, y in lsh.candidates():
        if estimate(sigs[row[x]], sigs[row[y]]) >= threshold:
            uf.union(x, y)
    return uf.groups()


# fmt: off
def near_dedup(
    docs: Iterable[Doc], *, num_perm: int = 128, bands: int = 16, threshold: float = 0.8,
    k: int = 5, seed: int = 0, workers: int = 1, drop: bool = True,
) -> Iterator[Doc]:
    """Tag every document with meta["minhash_cluster"] (its cluster's
    smallest id) and, with drop=True, keep only the cluster roots."""
    items = list(docs)
    texts: dict[str, str] = {}
    for d in items:
        if d.id in texts:
            raise ValueError(f"document id {d.id!r} appears twice")
        texts[d.id] = d.text
    root = {}
    for group in clusters(
        texts,
        num_perm=num_perm,
        bands=bands,
        threshold=threshold,
        k=k,
        seed=seed,
        workers=workers,
    ):
        r = min(group)
        for x in group:
            root[x] = r
    for d in items:
        r = root.get(d.id, d.id)
        if drop and r != d.id:
            continue
        yield dataclasses.replace(d, meta={**d.meta, "minhash_cluster": r})

# fmt: on


def _strings(value: object, out: list[str]) -> None:
    """Every string inside a JSON value, skipping the id-like keys."""
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, list):
        for v in value:
            _strings(v, out)
    elif isinstance(value, dict):
        for key, v in value.items():
            if key not in _SKIP_KEYS:
                _strings(v, out)


def protected_ngrams(paths: Sequence[Path], n: int = 13) -> set[str]:
    """Every word n-gram of every protected text."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    grams: set[str] = set()
    for p in paths:
        p = Path(p)
        raw = p.read_text(encoding="utf-8")
        texts: list[str] = []
        if p.name.endswith(".jsonl"):
            for i, line in enumerate(raw.splitlines(), 1):
                if not line.strip():
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError(
                        f"{p}:{i}: a protected JSON line must be an object"
                    )
                _strings(obj, texts)
        else:
            texts.append(raw)
        for t in texts:
            w = words(t)
            grams.update(" ".join(w[i : i + n]) for i in range(len(w) - n + 1))
    return grams


def contaminated(text: str, grams: AbstractSet[str], n: int = 13) -> bool:
    """True when some word n-gram of text is in grams."""
    w = words(text)
    return any(" ".join(w[i : i + n]) in grams for i in range(len(w) - n + 1))


def decontaminate(
    docs: Iterable[Doc], protected: Sequence[Path], n: int = 13
) -> Iterator[Doc]:
    """Drop every document that shares a word n-gram with a protected text."""
    grams = protected_ngrams(protected, n)
    for d in docs:
        if not contaminated(d.text, grams, n):
            yield d
