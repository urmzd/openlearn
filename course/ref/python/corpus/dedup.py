"""corpus.dedup (data.03): exact dedup by paragraph hashes, a Bloom screen,
and a sort-merge confirm.

A corpus repeats itself: mirrored pages, boilerplate, the same story posted
twice. Exact dedup keeps the first occurrence of every paragraph and drops
the rest. Holding every paragraph's hash in a set costs 32 bytes plus Python
overhead per paragraph; a small Python Bloom screen answers
"maybe seen" for about bloom_bytes_per_item bytes each. A Bloom filter has
false positives, never false negatives, so its positives are only
candidates: a sort-merge over the candidates' exact hashes confirms which
are real duplicates, and no unique paragraph is ever dropped.

    spool    write the input to a temporary JSON Lines file, count paragraphs
    screen   hash every paragraph; a Bloom hit makes its hash a candidate
    confirm  collect (hash, doc, paragraph) for candidates, sort, merge
    emit     reread the spool, drop confirmed repeats, drop emptied docs

Chapter: content/tracks/data-engineering/05-corpus-pipeline/03-exact-dedup-bloom-and-sort-merge.md.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from tinyllm.num.rng import fnv1a64, splitmix64

from corpus.stage import Doc

_BLANK = re.compile(r"\n[ \t]*\n")
STATS_KEYS = (
    "docs_in",
    "paragraphs",
    "bloom_bits",
    "candidates",
    "false_positives",
    "duplicate_paragraphs",
    "docs_dropped",
    "docs_out",
)


class _Bloom:
    """Small data.03 Bloom screen matching contracts/formats/bloom.md."""

    def __init__(self, n: int, rate: float) -> None:
        if n < 1 or not 0 < rate < 1:
            raise ValueError("Bloom needs n >= 1 and 0 < rate < 1")
        self.m = math.ceil(-n * math.log(rate) / math.log(2) ** 2)
        self.k = max(1, math.floor(self.m / n * math.log(2) + 0.5))
        self.bits = bytearray((self.m + 7) // 8)

    def _positions(self, item: bytes):
        h1 = fnv1a64(item)
        h2 = splitmix64(h1) | 1
        for i in range(self.k):
            yield ((h1 + i * h2) & ((1 << 64) - 1)) % self.m

    def contains(self, item: bytes) -> bool:
        return all(self.bits[pos // 8] & (1 << (pos % 8)) for pos in self._positions(item))

    def insert(self, item: bytes) -> None:
        for pos in self._positions(item):
            self.bits[pos // 8] |= 1 << (pos % 8)


def paragraphs(text: str) -> list[str]:
    """The paragraphs of a text: pieces between blank lines (a line that is
    empty or only spaces and tabs), each stripped; empty pieces are skipped."""
    # SOLUTION-BEGIN data.03
    return [p.strip() for p in _BLANK.split(text) if p.strip()]
    # SOLUTION-END


def paragraph_hash(p: str) -> bytes:
    """SHA-256 of the paragraph's UTF-8 bytes (32 bytes)."""
    # SOLUTION-BEGIN data.03
    return hashlib.sha256(p.encode("utf-8")).digest()
    # SOLUTION-END


def bloom_rate(bloom_bytes_per_item: float) -> float:
    """The false-positive rate p whose optimal filter spends
    bloom_bytes_per_item bytes per item: m / n = 8 B = -ln p / (ln 2)^2, so
    p = exp(-8 B (ln 2)^2). ValueError unless B > 0."""
    # SOLUTION-BEGIN data.03
    if not bloom_bytes_per_item > 0:
        raise ValueError("bloom_bytes_per_item must be positive")
    return math.exp(-8.0 * bloom_bytes_per_item * math.log(2) ** 2)
    # SOLUTION-END


def _doc_line(doc: Doc) -> str:
    # SOLUTION-BEGIN data.03
    return json.dumps(
        {"id": doc.id, "source_id": doc.source_id, "text": doc.text, "meta": dict(doc.meta)},
        ensure_ascii=False,
    )
    # SOLUTION-END


def _read_spool(path: Path) -> Iterator[Doc]:
    # SOLUTION-BEGIN data.03
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            yield Doc(r["id"], r["source_id"], r["text"], r["meta"])
    # SOLUTION-END


def exact_dedup(
    docs: Iterable[Doc],
    bloom_bytes_per_item: float = 10,
    *,
    spool_dir: str | Path | None = None,
    stats: dict[str, Any] | None = None,
) -> Iterator[Doc]:
    """Exact paragraph dedup, as a stage (it reads all of its input before it
    yields the first document, holding it on disk, not in memory).

    Keeps the first occurrence, in input order, of every distinct paragraph
    (paragraphs, compared by paragraph_hash) and removes every later one. A
    document that loses no paragraph is yielded unchanged; one that loses some
    is yielded with its remaining paragraphs joined by "\\n\\n"; one that loses
    all is dropped. Output order is input order.

    The screen is a Python Bloom filter sized for N paragraphs at bloom_rate(B);
    a paragraph whose hash the filter already
    contains becomes a candidate. Only candidates are confirmed, by sorting
    their (hash, document index, paragraph index) occurrences and merging
    equal hashes. If `stats` is given it receives STATS_KEYS once the output
    is exhausted. The spool lives in a temporary directory (under spool_dir
    when given) that is removed when the generator finishes or is closed."""
    # SOLUTION-BEGIN data.03
    rate = bloom_rate(bloom_bytes_per_item)
    with tempfile.TemporaryDirectory(prefix="dedup-", dir=spool_dir) as td:
        spool = Path(td) / "spool.jsonl"
        n_docs = n_paras = 0
        with spool.open("w", encoding="utf-8") as fh:
            for doc in docs:
                fh.write(_doc_line(doc) + "\n")
                n_docs += 1
                n_paras += len(paragraphs(doc.text))

        # Screen: one Bloom filter sized for every paragraph of the input.
            bloom = _Bloom(max(n_paras, 1), rate)
        candidates: set[bytes] = set()
        for doc in _read_spool(spool):
            for p in paragraphs(doc.text):
                h = paragraph_hash(p)
                if bloom.contains(h):
                    candidates.add(h)
                else:
                    bloom.insert(h)

        # Confirm: sort-merge the occurrences of candidate hashes.
        occ: list[tuple[bytes, int, int]] = []
        for i, doc in enumerate(_read_spool(spool)):
            for j, p in enumerate(paragraphs(doc.text)):
                h = paragraph_hash(p)
                if h in candidates:
                    occ.append((h, i, j))
        occ.sort()
        drop: set[tuple[int, int]] = set()
        false_pos = 0
        k = 0
        while k < len(occ):
            end = k
            while end + 1 < len(occ) and occ[end + 1][0] == occ[k][0]:
                end += 1
            if end == k:
                false_pos += 1  # the filter said "seen", but this hash occurs once
            for _, i, j in occ[k + 1 : end + 1]:
                drop.add((i, j))  # every occurrence after the first
            k = end + 1

        # Emit, in input order.
        dropped_docs = out_docs = 0
        for i, doc in enumerate(_read_spool(spool)):
            paras = paragraphs(doc.text)
            keep = [p for j, p in enumerate(paras) if (i, j) not in drop]
            if not keep:
                dropped_docs += 1
                continue
            out_docs += 1
            if len(keep) == len(paras):
                yield doc
            else:
                yield dataclasses.replace(doc, text="\n\n".join(keep))
        if stats is not None:
            stats.update(
                docs_in=n_docs,
                paragraphs=n_paras,
                bloom_bits=bloom.m,
                candidates=len(candidates),
                false_positives=false_pos,
                duplicate_paragraphs=len(drop),
                docs_dropped=dropped_docs,
                docs_out=out_docs,
            )
    # SOLUTION-END
