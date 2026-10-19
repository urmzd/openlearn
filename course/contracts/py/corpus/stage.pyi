# contracts/py/corpus/stage.pyi (data.02): the Doc type, streaming stages, extraction
# chapter: content/courses/data-engineering/05-corpus-pipeline/02-extract-normalize-and-quality-filters.md
#
# Every corpus stage (data.02 to data.05) is a Stage: a function from an
# iterator of Docs to an iterator of Docs. A stage pulls documents one at a
# time and yields as it goes; it never collects the stream (a list, a sort,
# a len), so a pipeline runs in constant memory on any corpus size. Doc is
# immutable: a stage that changes a document yields a new one
# (dataclasses.replace) and never mutates `meta` in place.
#
# meta keys set so far: url, fetched_at, license_spdx (extract); lang and
# lang_conf (corpus.filter.lang_filter); ppl (corpus.filter.ppl_filter).
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from corpus.fetch import Manifest

@dataclass(frozen=True, slots=True)
class Doc:
    id: str  # "<source_id>:<k>", k the 0-based position in the source's raw files (formats/corpus-shard.md)
    source_id: str
    text: str
    meta: Mapping[str, Any] = field(default_factory=dict)

Stage = Callable[[Iterator[Doc]], Iterator[Doc]]

def compose(*stages: Stage) -> Stage:
    """One Stage running `stages` left to right: compose(a, b)(docs) is
    b(a(docs)); compose() is the identity. Building or calling the composed
    stage pulls nothing; documents flow only as the result is iterated."""

def extract(manifest: Manifest) -> Iterator[Doc]:
    """The raw documents (corpus.fetch.read_raw) of every "fetched" or
    "cached" manifest entry, entry by entry in manifest order, as Docs with
    id "<source_id>:<k>" and meta {url, fetched_at, license_spdx} from the
    raw line. Quarantined entries contribute nothing."""
