# contracts/py/corpus/fetch.pyi (data.01): async fetch with resume, checksums, license capture
# chapter: content/tracks/data-engineering/05-corpus-pipeline/01-async-fetch-resume-checksums.md
#
# Input: the `sources` of a corpus config (formats/corpus-config.schema.json).
# Output under `dest` (the corpus directory, /artifacts/corpus):
#
#   raw/<source_id>/<yyyymmdd>/part-<nnnnn>.jsonl.zst   raw documents (formats/corpus-shard.md)
#   LEDGER.jsonl                                         one row per fetched source (formats/ledger.schema.json)
#   quarantine/<source_id>/<name>                        bytes whose sha256 is not the pinned one
#   downloads/<source_id>/<name>[.part]                  downloads, kept so a cut one resumes
#
# Words used below:
#   pinned     the sha256 the config gives for a source; the only checksum
#              trusted (a server's own checksum header is not)
#   retryable  a failure a later attempt may not see: HTTP 5xx and 429, a body
#              cut short, a reset or refused connection, no byte for
#              `timeout` seconds. Anything else (other 4xx, a malformed
#              response, a file that is not UTF-8 text or JSON Lines) is not.
#   <line>     the 1-based line where a document starts in its source file
#
# A raw document line is {"url": "<source url>#<line>", "fetched_at":
# <retrieved_at>, "license_spdx": <source license>, "text": <document>},
# written as compact JSON in that key order, non-ASCII kept as UTF-8. Parts
# hold at most `part_docs` documents and are zstd frames (zstandard, level 3).
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LEDGER_NAME: str  # "LEDGER.jsonl"

@dataclass(frozen=True)
class Source:
    id: str
    url: str
    sha256: str  # pinned: 64 lowercase hex digits
    license_spdx: str
    allowed_uses: tuple[str, ...] = ("train", "eval")
    format: str = "jsonl"  # "text" | "jsonl" | "parquet"
    pii_policy: str = "scrub"  # the ledger's pii_policy ("scrub" | "drop" | "none")

@dataclass(frozen=True)
class Fetched:
    source: Source
    status: str  # "fetched" | "cached" | "quarantined"
    raw_dir: str  # absolute raw/<source_id>/<yyyymmdd>; "" when quarantined
    sha256: str  # of the bytes received (the pinned value when cached)
    bytes: int  # bytes received by this call (0 when cached)
    n_docs: int  # documents in the raw parts (0 when quarantined)
    retrieved_at: str  # RFC 3339 UTC with a Z: "2026-01-01T00:00:00Z"
    requests: int  # HTTP requests this call made for the source (0 when cached)

@dataclass(frozen=True)
class Manifest:
    entries: tuple[Fetched, ...]  # one per source, in the order the sources were given

    @property
    def ok(self) -> bool:
        """True when no source was quarantined."""

class FetchError(Exception):
    source_id: str
    retryable: bool  # True: exit 75 under spec/subprocess-activity.md; False: exit 65

    def __init__(self, source_id: str, message: str, retryable: bool) -> None:
        """str(err) is "<source_id>: <message>"."""

def sources_from_config(config: Mapping[str, Any]) -> list[Source]:
    """config["sources"] as Source values with the schema defaults filled in.
    ValueError for a missing id, url, sha256, or license_spdx; a sha256 that
    is not 64 lowercase hex digits; a format outside text, jsonl, parquet; or
    two sources with the same id."""

def split_documents(data: bytes, fmt: str) -> Iterator[tuple[int, str]]:
    """The documents of a source file as (<line>, text), in file order.
    "text": one document per block of non-blank lines (a blank line is empty
    or white space only), its lines joined by "\\n". "jsonl": the string
    "text" field of each non-blank line. ValueError naming the line for bytes
    that are not UTF-8, a line that is not a JSON object with a string
    "text", and for "parquet" (data.01 reads text and jsonl only)."""

def read_raw(raw_dir: str | Path) -> Iterator[dict[str, str]]:
    """Every raw document in raw_dir: part files in name order, lines in order."""

def ledger_row(f: Fetched) -> dict[str, Any]:
    """The LEDGER.jsonl row of a fetched source: every field of
    formats/ledger.schema.json, with filters_applied [], kept = n_docs,
    dropped 0, notes ""."""

async def fetch(
    srcs: Sequence[Source],
    dest: str | Path,
    *,
    concurrency: int = 8,
    attempts: int = 4,
    timeout: float = 30.0,
    base_delay: float = 0.5,
    part_docs: int = 100_000,
    clock: Any = None,
    sleep: Callable[[float], Awaitable[None]] = ...,
) -> Manifest:
    """Fetch every source with at most `concurrency` downloads in flight.

    Cached: LEDGER.jsonl has a row with the source's id and pinned sha256 and
    that row's raw directory exists; no request is made. Otherwise the
    download resumes a partial file with `Range: bytes=<size>-` (a 200 reply
    restarts it from byte 0; a 416 means the partial file is whole). A
    retryable failure waits base_delay * 2 ** (n - 1) seconds through `sleep`
    after attempt n, for at most `attempts` attempts. A sha256 that is not the
    pinned one moves the file to quarantine/ ("quarantined"; never retried).
    Verified bytes become raw parts, written to a temporary directory renamed
    into place when complete ("fetched"). retrieved_at comes from
    clock.now() (Unix seconds, UTC; wall time when clock is None).

    LEDGER.jsonl gains ledger_row() of each "fetched" source, in input order,
    also when another source failed. A source that fails for good raises
    FetchError after every other download is cancelled. ValueError when
    concurrency, attempts, or part_docs is below 1, or two sources share an id."""
