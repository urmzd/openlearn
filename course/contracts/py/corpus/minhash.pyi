# contracts/py/corpus/minhash.pyi (data.04): near-duplicate detection and
# decontamination
# chapter: content/courses/data-engineering/05-corpus-pipeline/04-near-dedup-minhash-lsh.md
#
# Shingles. words(text) is re.findall(r"\w+", text.lower()): Unicode word
# runs, lower-cased. A k-shingle is k consecutive words joined by one space.
# A text with 1 to k-1 words has one shingle (all its words); a text with no
# words has none, and is never anyone's near duplicate.
#
# Hashing (M06.3). P = MERSENNE31 = 2^31 - 1, a prime. A shingle s hashes to
# x(s) = fnv1a64(utf8(s)) mod P. Permutation i is h_i(x) = (a_i * x + b_i)
# mod P with, from one PCG32(seed) in this order, a_0 = 1 + below(P - 1),
# b_0 = below(P), a_1, b_1, ... (a and b interleaved). Every product fits in
# uint64 (a, x < 2^31), so numpy computes it exactly.
#
# Signature. sig_i = min over the shingles of h_i(x); a text without
# shingles has sig_i = P for every i (P is never a hash value).
# Pr[sig_i(A) = sig_i(B)] is the Jaccard similarity |A n B| / |A u B|, so
# the share of equal entries estimates it without bias.
#
# LSH. b bands of r rows (b * r = num_perm). Two keys are candidates when
# all r entries of at least one band are equal: probability
# 1 - (1 - s^r)^b at similarity s, an S-curve whose steep part sits near
# (1/b)^(1/r). Candidates are confirmed by the estimate (>= threshold) and
# merged by union-find; the root of a cluster is its smallest id (byte
# order), so the output never depends on the order of unions.
#
# Determinism. Results do not depend on `workers` (signatures are computed
# in contiguous chunks on a process pool, every chunk with the same
# parameters) or on the order of the input.
#
# Decontamination. The protected n-grams are the word n-grams (n = 13 by
# default, words() as above) of every protected text. A protected file is
# JSON Lines when its name ends in .jsonl (each line an object; its texts
# are every string value, recursively, except under the keys "case_id",
# "id", "tags", and "scorer_args"), otherwise one UTF-8 text. A document is
# contaminated when one of its word n-grams is protected; texts with fewer
# than n words have no n-grams.
from pathlib import Path
from typing import AbstractSet, Iterable, Iterator, Mapping, Sequence

from numpy.typing import NDArray

from corpus.stage import Doc

MERSENNE31: int  # 2**31 - 1
FNV_OFFSET: int  # 0xCBF29CE484222325, FNV-1a 64 offset basis (as M06.3)
FNV_PRIME: int  # 0x100000001B3, FNV-1a 64 prime (as M06.3)

def words(text: str) -> list[str]:
    """re.findall(r"\\w+", text.lower())."""

def shingles(text: str, k: int = 5) -> set[str]:
    """The set of word k-shingles; one shingle for 1..k-1 words; empty for
    no words. ValueError for k < 1."""

def jaccard(a: AbstractSet[str], b: AbstractSet[str]) -> float:
    """|a & b| / |a | b|; 0.0 when both are empty."""

def shingle_hashes(shingles: Iterable[str]) -> NDArray:
    """uint64 [n]: fnv1a64(s.encode("utf-8")) mod MERSENNE31 for each
    shingle, in iteration order."""

def perm_params(num_perm: int, seed: int) -> tuple[NDArray, NDArray]:
    """(a, b), uint64 [num_perm] each, drawn from PCG32(seed) as above.
    ValueError for num_perm < 1."""

def minhash(shingles: Iterable[str], num_perm: int = 128, seed: int = 0) -> NDArray:
    """uint64 [num_perm]: the signature defined above."""

def estimate(sig_a: NDArray, sig_b: NDArray) -> float:
    """Share of equal entries. ValueError when the shapes differ."""

class LSH:
    bands: int
    rows: int

    def __init__(self, bands: int = 16, rows: int = 8) -> None:
        """ValueError unless bands >= 1 and rows >= 1."""

    @property
    def threshold(self) -> float:
        """(1 / bands) ** (1 / rows)."""

    def probability(self, s: float) -> float:
        """1 - (1 - s**rows) ** bands: the chance that two keys with
        similarity s become candidates."""

    def insert(self, key: str, sig: NDArray) -> None:
        """Index sig under key. ValueError when len(sig) != bands * rows or
        key was inserted before. A signature of an empty text (all entries
        MERSENNE31) is not indexed."""

    def candidates(self) -> list[tuple[str, str]]:
        """Every pair (x, y), x < y, whose signatures agree on all rows of at
        least one band (band i covers entries i*rows to (i+1)*rows - 1;
        equal values in different bands do not collide). Sorted, no
        duplicates."""

class UnionFind:
    def __init__(self) -> None: ...
    def find(self, x: str) -> str:
        """The root of x's set (adding x as a singleton when new). The root
        is always the smallest member."""

    def union(self, x: str, y: str) -> str:
        """Merge the sets of x and y; return the root of the merged set."""

    def groups(self) -> list[set[str]]:
        """Every set with at least two members, sorted by smallest member."""

def signatures(
    texts: Sequence[str], *, num_perm: int = 128, seed: int = 0, k: int = 5, workers: int = 1
) -> NDArray:
    """uint64 [len(texts), num_perm]: minhash(shingles(t, k)) per text. With
    workers > 1 the rows are computed on a process pool in contiguous
    chunks; the result is identical for every workers value. ValueError for
    workers < 1."""

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
    """Near-duplicate clusters of {id: text}: LSH(bands, num_perm // bands)
    candidates whose estimate is >= threshold, merged by union-find; every
    cluster with at least two ids, sorted by smallest id. ValueError when
    bands does not divide num_perm."""

def near_dedup(
    docs: Iterable[Doc],
    *,
    num_perm: int = 128,
    bands: int = 16,
    threshold: float = 0.8,
    k: int = 5,
    seed: int = 0,
    workers: int = 1,
    drop: bool = True,
) -> Iterator[Doc]:
    """A Stage over the whole input (it reads every document before
    yielding). Each output Doc carries meta["minhash_cluster"]: the smallest
    id of its cluster, its own id when it has no near duplicate. drop=True
    keeps only cluster roots; drop=False keeps every document. Output order
    is input order; other meta keys are kept. ValueError for a repeated id."""

def protected_ngrams(paths: Sequence[Path], n: int = 13) -> set[str]:
    """Every word n-gram (words joined by one space) of every protected
    text. OSError for a missing file; ValueError for a .jsonl line that is
    not a JSON object, or n < 1."""

def contaminated(text: str, grams: AbstractSet[str], n: int = 13) -> bool:
    """True when some word n-gram of text is in grams."""

def decontaminate(docs: Iterable[Doc], protected: Sequence[Path], n: int = 13) -> Iterator[Doc]:
    """A Stage: drop every contaminated document, keep the rest unchanged
    and in order. Streams: reads the protected files once, then one
    document at a time."""
