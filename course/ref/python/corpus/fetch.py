"""corpus.fetch (data.01): async fetch with resume, checksums, license capture.

Every source in a corpus config is one file at a URL with a pinned sha256 and
an SPDX license. `fetch` downloads them concurrently (at most `concurrency`
at once), resumes a cut download with an HTTP Range request, verifies the
pinned checksum, quarantines a file whose bytes do not match, splits each
verified file into documents, writes them as raw-document parts
(formats/corpus-shard.md), and appends one ledger row per source to
LEDGER.jsonl (formats/ledger.schema.json). A rerun downloads nothing whose
checksum the ledger already records.

Layout under `dest` (the corpus directory, /artifacts/corpus):

    downloads/<source_id>/<name>.part     a download in progress (resumable)
    downloads/<source_id>/<name>          a verified download
    quarantine/<source_id>/<name>         bytes whose sha256 is not the pinned one
    raw/<source_id>/<yyyymmdd>/part-<nnnnn>.jsonl.zst
    LEDGER.jsonl

Standard library plus zstandard (contracts/allowed-deps.toml, [python.corpus]).
Chapter: content/tracks/data-engineering/05-corpus-pipeline/01-async-fetch-resume-checksums.md.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import hashlib
import json
import os
import shutil
import ssl
import time
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit

import zstandard

LEDGER_NAME = "LEDGER.jsonl"
CHUNK = 64 * 1024
MAX_REDIRECTS = 5
FORMATS = ("text", "jsonl", "parquet")

Sleep = Callable[[float], Awaitable[None]]


@dataclass(frozen=True)
class Source:
    """One `[[sources]]` entry of a corpus config, plus the ledger's pii_policy."""

    id: str
    url: str
    sha256: str
    license_spdx: str
    allowed_uses: tuple[str, ...] = ("train", "eval")
    format: str = "jsonl"
    pii_policy: str = "scrub"


@dataclass(frozen=True)
class Fetched:
    """What happened to one source."""

    source: Source
    status: str  # "fetched" | "cached" | "quarantined"
    raw_dir: str  # absolute raw/<id>/<yyyymmdd> directory; "" when quarantined
    sha256: str  # of the bytes received
    bytes: int  # size of the bytes received
    n_docs: int  # documents written to raw parts (0 when quarantined)
    retrieved_at: str  # RFC 3339 UTC, "2026-01-01T00:00:00Z"
    requests: int  # HTTP requests this call made for the source (0 when cached)


@dataclass(frozen=True)
class Manifest:
    entries: tuple[Fetched, ...]  # one per source, in the order the sources were given

    @property
    def ok(self) -> bool:
        """True when no source was quarantined."""
        # SOLUTION-BEGIN data.01
        return all(e.status != "quarantined" for e in self.entries)
        # SOLUTION-END


class FetchError(Exception):
    """A source that could not be fetched. `retryable` follows the subprocess
    activity contract: True means a later attempt may succeed (exit 75),
    False means it will not (exit 65)."""

    def __init__(self, source_id: str, message: str, retryable: bool) -> None:
        # SOLUTION-BEGIN data.01
        super().__init__(f"{source_id}: {message}")
        self.source_id = source_id
        self.retryable = retryable
        # SOLUTION-END


class _WallClock:
    def now(self) -> float:
        # SOLUTION-BEGIN data.01
        return time.time()
        # SOLUTION-END


# ---------------------------------------------------------------------------
# config and formats


def sources_from_config(config: Mapping[str, Any]) -> list[Source]:
    """The `sources` of a parsed corpus config (corpus-config.schema.json) as
    Source values, defaults filled in. ValueError for a missing field, a
    format outside text/jsonl/parquet, a sha256 that is not 64 lowercase hex
    digits, or two sources with the same id."""
    # SOLUTION-BEGIN data.01
    out: list[Source] = []
    seen: set[str] = set()
    for i, s in enumerate(config.get("sources", [])):
        for key in ("id", "url", "sha256", "license_spdx"):
            if not s.get(key):
                raise ValueError(f"sources[{i}]: missing {key!r}")
        if s["id"] in seen:
            raise ValueError(f"sources[{i}]: duplicate source id {s['id']!r}")
        seen.add(s["id"])
        sha = s["sha256"]
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError(f"sources[{i}]: sha256 must be 64 lowercase hex digits")
        fmt = s.get("format", "jsonl")
        if fmt not in FORMATS:
            raise ValueError(f"sources[{i}]: format {fmt!r} not in {FORMATS}")
        out.append(
            Source(
                id=s["id"],
                url=s["url"],
                sha256=sha,
                license_spdx=s["license_spdx"],
                allowed_uses=tuple(s.get("allowed_uses", ("train", "eval"))),
                format=fmt,
            )
        )
    return out
    # SOLUTION-END


def split_documents(data: bytes, fmt: str) -> Iterator[tuple[int, str]]:
    """The documents of a fetched file as (line, text), in file order.

    `line` is the 1-based line number where the document starts. `text`: one
    document per block of non-blank lines, the block's lines joined by "\\n";
    `jsonl`: the "text" field of each non-blank line. ValueError, naming the
    line, for bytes that are not UTF-8, a line that is not a JSON object with
    a string "text", or the `parquet` format (data.06 owns pyarrow)."""
    # SOLUTION-BEGIN data.01
    if fmt == "parquet":
        raise ValueError("format parquet: data.01 reads text and jsonl sources only")
    if fmt not in ("text", "jsonl"):
        raise ValueError(f"unknown format {fmt!r}")
    try:
        content = data.decode("utf-8")
    except UnicodeDecodeError as e:
        line = data[: e.start].count(b"\n") + 1
        raise ValueError(f"line {line}: not UTF-8 (byte {e.start})") from None
    lines = content.splitlines()
    if fmt == "jsonl":
        for n, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"line {n}: not JSON ({e.msg})") from None
            if not isinstance(obj, dict) or not isinstance(obj.get("text"), str):
                raise ValueError(f'line {n}: no string "text" field')
            yield n, obj["text"]
        return
    start, block = 0, []
    for n, line in enumerate(lines, 1):
        if line.strip():
            if not block:
                start = n
            block.append(line)
        elif block:
            yield start, "\n".join(block)
            block = []
    if block:
        yield start, "\n".join(block)
    # SOLUTION-END


def read_raw(raw_dir: str | Path) -> Iterator[dict[str, str]]:
    """Every raw document under raw_dir: its part-*.jsonl.zst files in name
    order, lines in order, each a dict with url, fetched_at, license_spdx,
    text."""
    # SOLUTION-BEGIN data.01
    dctx = zstandard.ZstdDecompressor()
    for part in sorted(Path(raw_dir).glob("part-*.jsonl.zst")):
        with part.open("rb") as fh, dctx.stream_reader(fh) as reader:
            data = reader.read()
        for line in data.decode("utf-8").splitlines():
            if line:
                yield json.loads(line)
    # SOLUTION-END


def ledger_row(f: Fetched) -> dict[str, Any]:
    """The LEDGER.jsonl row for a fetched source (formats/ledger.schema.json):
    nothing is filtered yet, so kept = n_docs and dropped = 0."""
    # SOLUTION-BEGIN data.01
    s = f.source
    return {
        "source_id": s.id,
        "url": s.url,
        "license_spdx": s.license_spdx,
        "retrieved_at": f.retrieved_at,
        "sha256": f.sha256,
        "n_docs": f.n_docs,
        "allowed_uses": list(s.allowed_uses),
        "pii_policy": s.pii_policy,
        "filters_applied": [],
        "kept": f.n_docs,
        "dropped": 0,
        "notes": "",
    }
    # SOLUTION-END


# ---------------------------------------------------------------------------
# HTTP over asyncio streams


class _Retry(Exception):
    """A failure the next attempt may not see."""


@dataclass
class _Head:
    status: int
    headers: dict[str, str]


async def _open(url: str, headers: dict[str, str], timeout: float):
    """Connect, send a GET, read the status line and headers. Follows up to
    MAX_REDIRECTS redirects. Returns (head, reader, writer, final_url)."""
    # SOLUTION-BEGIN data.01
    for _ in range(MAX_REDIRECTS + 1):
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError(f"unsupported URL {url!r}")
        tls = parts.scheme == "https"
        port = parts.port or (443 if tls else 80)
        target = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(
                parts.hostname, port, ssl=ssl.create_default_context() if tls else None
            ),
            timeout,
        )
        try:
            lines = [
                f"GET {target} HTTP/1.1",
                f"Host: {parts.netloc}",
                "Connection: close",
                "Accept-Encoding: identity",
            ] + [f"{k}: {v}" for k, v in headers.items()]
            writer.write(("\r\n".join(lines) + "\r\n\r\n").encode("latin-1"))
            await writer.drain()
            status_line = await asyncio.wait_for(reader.readline(), timeout)
            fields = status_line.decode("latin-1").split(None, 2)
            if len(fields) < 2 or not fields[1].isdigit():
                if not status_line:
                    raise _Retry("connection closed before a response")
                raise ValueError(f"malformed status line {status_line!r}")
            got: dict[str, str] = {}
            while True:
                line = await asyncio.wait_for(reader.readline(), timeout)
                if line in (b"\r\n", b"\n", b""):
                    break
                name, _, value = line.decode("latin-1").partition(":")
                got[name.strip().lower()] = value.strip()
        except BaseException:
            writer.close()
            raise
        status = int(fields[1])
        if status in (301, 302, 303, 307, 308) and "location" in got:
            writer.close()
            url = urljoin(url, got["location"])
            continue
        return _Head(status, got), reader, writer, url
    raise ValueError(f"more than {MAX_REDIRECTS} redirects")
    # SOLUTION-END


async def _attempt(url: str, part: Path, timeout: float) -> tuple[int, int | None]:
    """One GET of url into the partial file `part`, resuming from its size.

    Returns (bytes in part, total size if the server said it). Raises _Retry
    for a failure worth retrying (5xx, 429, a cut body, a reset or refused
    connection, a timeout) and ValueError for one that is not (other 4xx,
    chunked transfer coding)."""
    # SOLUTION-BEGIN data.01
    offset = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    try:
        head, reader, writer, _ = await _open(url, headers, timeout)
    except (OSError, asyncio.TimeoutError, asyncio.IncompleteReadError) as e:
        raise _Retry(f"connect: {type(e).__name__}: {e}") from None
    try:
        h = head.headers
        if head.status == 416 and offset:
            # Nothing at or after `offset`: the part is whole, or it is longer
            # than the file (the file changed upstream) and must start over.
            tail = h.get("content-range", "").rpartition("/")[2]
            if tail.isdigit() and int(tail) != offset:
                part.unlink()
                raise _Retry(f"part has {offset} bytes, the file {tail}")
            return offset, int(tail) if tail.isdigit() else None
        if head.status >= 500 or head.status == 429:
            raise _Retry(f"HTTP {head.status}")
        if head.status not in (200, 206):
            raise ValueError(f"HTTP {head.status}")
        if "chunked" in h.get("transfer-encoding", "").lower():
            raise ValueError("chunked transfer coding is not supported")
        total: int | None = None
        if head.status == 206:
            # Content-Range: bytes <first>-<last>/<total>
            spec = h.get("content-range", "")
            first = spec.removeprefix("bytes ").partition("-")[0]
            if not first.isdigit() or int(first) != offset:
                part.unlink(missing_ok=True)
                raise _Retry(f"server resumed at {spec!r}, not byte {offset}")
            tail = spec.rpartition("/")[2]
            total = int(tail) if tail.isdigit() else None
            mode = "ab"
        else:
            # 200: the server sent the whole file, whatever we asked for.
            offset, mode = 0, "wb"
        length = int(h["content-length"]) if "content-length" in h else None
        if total is None and length is not None:
            total = offset + length
        got = 0
        with part.open(mode) as fh:
            while length is None or got < length:
                want = CHUNK if length is None else min(CHUNK, length - got)
                try:
                    chunk = await asyncio.wait_for(reader.read(want), timeout)
                except (OSError, asyncio.TimeoutError) as e:
                    raise _Retry(f"body: {type(e).__name__}") from None
                if not chunk:
                    if length is None:
                        break
                    raise _Retry(f"body cut at {offset + got} of {total} bytes")
                fh.write(chunk)
                got += len(chunk)
        return offset + got, total
    finally:
        writer.close()
    # SOLUTION-END


def _sha256_file(path: Path) -> str:
    # SOLUTION-BEGIN data.01
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
    # SOLUTION-END


def _rfc3339(t: float) -> str:
    # SOLUTION-BEGIN data.01
    return (
        _dt.datetime.fromtimestamp(int(t), _dt.timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    # SOLUTION-END


def _write_parts(
    src: Source, data: bytes, raw_dir: Path, retrieved_at: str, part_docs: int
) -> int:
    """Split `data` into documents and write them under raw_dir atomically:
    parts go to a sibling .tmp directory that is renamed into place last."""
    # SOLUTION-BEGIN data.01
    tmp = raw_dir.with_name(raw_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    cctx = zstandard.ZstdCompressor(level=3)
    n, buf, part_no = 0, [], 0

    def flush() -> None:
        nonlocal buf, part_no
        (tmp / f"part-{part_no:05d}.jsonl.zst").write_bytes(
            cctx.compress("".join(buf).encode("utf-8"))
        )
        buf, part_no = [], part_no + 1

    for line, text in split_documents(data, src.format):
        doc = {
            "url": f"{src.url}#{line}",
            "fetched_at": retrieved_at,
            "license_spdx": src.license_spdx,
            "text": text,
        }
        buf.append(json.dumps(doc, ensure_ascii=False, separators=(",", ":")) + "\n")
        n += 1
        if len(buf) == part_docs:
            flush()
    if buf or part_no == 0:
        flush()
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    os.replace(tmp, raw_dir)
    return n
    # SOLUTION-END


def _read_ledger(dest: Path) -> list[dict[str, Any]]:
    # SOLUTION-BEGIN data.01
    p = dest / LEDGER_NAME
    if not p.is_file():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    # SOLUTION-END


def _raw_dir(dest: Path, source_id: str, retrieved_at: str) -> Path:
    # SOLUTION-BEGIN data.01
    return dest / "raw" / source_id / retrieved_at[:10].replace("-", "")
    # SOLUTION-END


async def _one(
    src: Source, dest: Path, ledger: list[dict[str, Any]], *, attempts: int,
    timeout: float, base_delay: float, part_docs: int, clock: Any, sleep: Sleep,
) -> Fetched:
    """Fetch, verify, and convert one source (the caller holds the semaphore)."""
    # SOLUTION-BEGIN data.01
    for row in ledger:
        if row.get("source_id") == src.id and row.get("sha256") == src.sha256:
            raw = _raw_dir(dest, src.id, row["retrieved_at"])
            if raw.is_dir():
                return Fetched(
                    src, "cached", str(raw), src.sha256, 0, row["n_docs"], row["retrieved_at"], 0
                )
    retrieved_at = _rfc3339(clock.now())
    name = urlsplit(src.url).path.rstrip("/").rpartition("/")[2] or "index"
    ddir = dest / "downloads" / src.id
    ddir.mkdir(parents=True, exist_ok=True)
    done, part = ddir / name, ddir / (name + ".part")
    requests = 0
    got = ""
    if done.is_file():
        # A verified download whose conversion did not finish (a crash).
        got = await asyncio.to_thread(_sha256_file, done)
        if got != src.sha256:
            done.unlink()
            got = ""
    if not got:
        for n in range(1, attempts + 1):
            requests += 1
            try:
                size, total = await _attempt(src.url, part, timeout)
                if total is not None and size != total:
                    raise _Retry(f"have {size} of {total} bytes")
                break
            except _Retry as e:
                if n == attempts:
                    raise FetchError(
                        src.id, f"gave up after {attempts} attempts: {e}", True
                    ) from None
            except ValueError as e:
                raise FetchError(src.id, str(e), False) from None
            await sleep(base_delay * 2 ** (n - 1))
        os.replace(part, done)
        got = await asyncio.to_thread(_sha256_file, done)
    size = done.stat().st_size
    if got != src.sha256:
        q = dest / "quarantine" / src.id
        q.mkdir(parents=True, exist_ok=True)
        os.replace(done, q / name)
        return Fetched(src, "quarantined", "", got, size, 0, retrieved_at, requests)
    raw = _raw_dir(dest, src.id, retrieved_at)
    try:
        n_docs = await asyncio.to_thread(
            _write_parts, src, done.read_bytes(), raw, retrieved_at, part_docs
        )
    except ValueError as e:
        raise FetchError(src.id, str(e), False) from None
    return Fetched(src, "fetched", str(raw), got, size, n_docs, retrieved_at, requests)
    # SOLUTION-END


async def fetch(
    srcs: Sequence[Source], dest: str | Path, *, concurrency: int = 8, attempts: int = 4,
    timeout: float = 30.0, base_delay: float = 0.5, part_docs: int = 100_000,
    clock: Any = None, sleep: Sleep = asyncio.sleep,
) -> Manifest:
    """Fetch every source into `dest` with at most `concurrency` downloads in
    flight, and return what happened to each, in input order.

    Per source: skip it when LEDGER.jsonl has a row with its id and pinned
    sha256 and that row's raw directory exists ("cached", no request).
    Otherwise download to downloads/<id>/<name>.part, resuming a partial file
    with `Range: bytes=<size>-` (a 200 reply restarts it from byte 0, a 416
    means the part is already whole). A retryable failure (5xx, 429, a cut
    body, a reset or refused connection, no byte for `timeout` seconds) waits
    base_delay * 2 ** (n - 1) through `sleep` after attempt n, for at most
    `attempts` attempts. Then compare the bytes' sha256 with the pinned one:
    on a mismatch move the file to quarantine/<id>/ ("quarantined", never
    retried); on a match write raw/<id>/<yyyymmdd>/part-<nnnnn>.jsonl.zst with
    at most part_docs documents each, renamed into place when complete
    ("fetched"). `clock.now()` (Unix seconds) gives retrieved_at.

    LEDGER.jsonl gains one row (ledger_row) per "fetched" source, in input
    order, also when another source failed. A source that fails for good
    raises FetchError after the other downloads are cancelled; their partial
    files stay for the next run to resume. ValueError when concurrency or
    attempts is below 1 or two sources share an id."""
    # SOLUTION-BEGIN data.01
    if concurrency < 1 or attempts < 1 or part_docs < 1:
        raise ValueError("concurrency, attempts, and part_docs must be at least 1")
    if len({s.id for s in srcs}) != len(srcs):
        raise ValueError("two sources share an id")
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    clock = clock or _WallClock()
    ledger = _read_ledger(dest)
    sem = asyncio.Semaphore(concurrency)
    results: dict[str, Fetched] = {}

    async def run(s: Source) -> None:
        async with sem:
            results[s.id] = await _one(
                s,
                dest,
                ledger,
                attempts=attempts,
                timeout=timeout,
                base_delay=base_delay,
                part_docs=part_docs,
                clock=clock,
                sleep=sleep,
            )

    try:
        async with asyncio.TaskGroup() as tg:
            for s in srcs:
                tg.create_task(run(s))
    except BaseExceptionGroup as eg:
        first = eg.exceptions[0]
        while isinstance(first, BaseExceptionGroup):
            first = first.exceptions[0]
        raise first from None
    finally:
        rows = [ledger_row(results[s.id]) for s in srcs if s.id in results and results[s.id].status == "fetched"]
        if rows:
            with (dest / LEDGER_NAME).open("a", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
    return Manifest(tuple(results[s.id] for s in srcs))
    # SOLUTION-END
