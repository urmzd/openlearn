# contracts/py/corpus/dedup.pyi (data.03): exact dedup by paragraph hashes, Bloom screen, sort-merge confirm
# chapter: content/courses/data-engineering/05-corpus-pipeline/03-exact-dedup-bloom-and-sort-merge.md
#
# Words used below:
#   paragraph   a piece of a text between blank lines (a line that is empty
#               or holds only spaces and tabs), stripped; empty pieces skipped
#   repeat      an occurrence of a paragraph whose exact text (its
#               paragraph_hash) occurred earlier: in an earlier document, or
#               earlier in the same document
#
# data.03 uses a private Python Bloom screen. Its positives are candidates
# only: a paragraph is removed only when a second occurrence of its exact
# hash is confirmed.
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from corpus.stage import Doc

STATS_KEYS: tuple[str, ...]
# ("docs_in", "paragraphs", "bloom_bits", "candidates", "false_positives",
#  "duplicate_paragraphs", "docs_dropped", "docs_out")
#   bloom_bits         m of the filter (formats/bloom.md bytes 8..16)
#   candidates         distinct hashes the filter reported as already present
#   false_positives    candidates whose hash occurs exactly once in the input
#   duplicate_paragraphs  repeats removed

def paragraphs(text: str) -> list[str]:
    """The paragraphs of text, in order."""

def paragraph_hash(p: str) -> bytes:
    """hashlib.sha256(p.encode("utf-8")).digest()."""

def bloom_rate(bloom_bytes_per_item: float) -> float:
    """exp(-8 * B * (ln 2)^2): the false-positive rate whose optimal filter
    (formats/bloom.md sizing) spends B bytes per item. ValueError unless B > 0."""

def exact_dedup(
    docs: Iterable[Doc],
    bloom_bytes_per_item: float = 10,
    *,
    spool_dir: str | Path | None = None,
    stats: dict[str, Any] | None = None,
) -> Iterator[Doc]:
    """A Stage (bind the keyword arguments with functools.partial): every
    repeat removed. A document that loses nothing is yielded unchanged; one
    that loses some paragraphs is yielded with the rest joined by "\\n\\n";
    one that loses all is dropped; order is input order. The screen is one
    Python Bloom filter sized for max(N, 1) and bloom_rate(B) over the N
    paragraphs of the input. The input is spooled to a temporary directory
    (under spool_dir when given, removed afterwards), so memory does not
    grow with the corpus; nothing is yielded before the input is exhausted.
    `stats`, when given, receives every STATS_KEYS entry once the output is
    exhausted."""
