"""corpus.stage (data.02): the Doc type, streaming stages, and extraction.

Every corpus stage is a function from an iterator of documents to an
iterator of documents, so a pipeline is just function composition and
memory stays flat whatever the corpus size: a stage pulls one document,
yields zero or one (or more), and never holds the stream.

    pipeline = compose(normalize_unicode, lang_filter(0.65), gopher_rules())
    for doc in pipeline(extract(manifest)): ...

Chapter: content/tracks/data-engineering/05-corpus-pipeline/02-extract-normalize-and-quality-filters.md.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from corpus.fetch import Manifest, read_raw


@dataclass(frozen=True, slots=True)
class Doc:
    id: str  # "<source_id>:<k>", k the 0-based position in the source's raw files
    source_id: str
    text: str
    meta: Mapping[str, Any] = field(default_factory=dict)


Stage = Callable[[Iterator[Doc]], Iterator[Doc]]


def compose(*stages: Stage) -> Stage:
    """One stage that runs `stages` left to right: compose(a, b)(docs) is
    b(a(docs)). compose() is the identity. Nothing is pulled until the
    result is iterated."""
    def run(docs: Iterator[Doc]) -> Iterator[Doc]:
        out: Iterator[Doc] = iter(docs)
        for s in stages:
            out = s(out)
        return out

    return run


def extract(manifest: Manifest) -> Iterator[Doc]:
    """The raw documents of every fetched or cached source of a fetch
    manifest, as Docs, source by source in manifest order. Doc.id is
    "<source_id>:<k>" with k counting from 0 within the source; meta holds
    the raw line's url, fetched_at, and license_spdx. Quarantined sources
    have no documents."""
    for entry in manifest.entries:
        if entry.status == "quarantined":
            continue
        sid = entry.source.id
        for k, rec in enumerate(read_raw(entry.raw_dir)):
            yield Doc(
                id=f"{sid}:{k}",
                source_id=sid,
                text=rec["text"],
                meta={
                    "url": rec["url"],
                    "fetched_at": rec["fetched_at"],
                    "license_spdx": rec["license_spdx"],
                },
            )
