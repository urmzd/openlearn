# Async fetch with resume, checksums, and license capture

The tokenizers in [Tokenizers](../../build-an-llm/p01-tokenizers/README.md) from *Build Your Own LLM System* are trained on whatever text they are handed, and so far that text has been a file copied by hand. The corpus pipeline starts here. A config names the sources (a URL, the sha256 you reviewed, the license), and the pipeline must turn that list into documents on disk: the same documents every time, with a record of where each came from and under what terms.

The network makes this hard. A 300 MB download from a dataset mirror fails at 280 MB more often than you would like; mirrors answer `503` under load; a URL you pinned last month now serves different bytes; and the durable `CorpusBuild` workflow, built in [CorpusBuild as a durable workflow](09-corpusbuild-as-a-durable-workflow.md), will rerun this step after every crash. Without resume, every cut restarts from zero. Without the checksum, a changed file silently changes the corpus. Without the ledger, nobody can later answer "may we train on this?".

This section builds the fetch stage, the first stage of the corpus build: every later stage reads what it writes. It leans on three primers. [Python asyncio](../../programming-languages/02-language-and-tool-primers/08-python-asyncio.md) and [HTTP/1.1 from the wire up](../../programming-languages/02-language-and-tool-primers/05-http-and-sse.md) in *Programming Languages* supply the event loop, bounded concurrency, and the protocol; [Data licensing and the ledger](../../responsible-ai/01-data-licensing/01-data-licensing-and-the-ledger.md) in *Responsible AI* explains what the ledger is for. The complete solution is [`python/corpus/fetch.py`](../../../../course/solutions/python/corpus/fetch.py), its contract is [`course/contracts/py/corpus/fetch.pyi`](../../../../course/contracts/py/corpus/fetch.pyi), and it uses only the standard library plus `zstandard` (allowed for `corpus` in `contracts/allowed-deps.toml`), because the raw parts are zstd files. As you read the implementation, try to predict each step before you see it.

> **Learning objectives**
> - Resume a cut download with an HTTP `Range` request and interpret `206`, `200`, and `416` correctly.
> - Explain why only the checksum pinned in the config counts, and why a mismatch is quarantined rather than retried.
> - Classify fetch errors as retryable or not and map them to exit codes 75 and 65.
> - Capture one ledger row per source at fetch time, written deterministically in config order.
> - Make a rerun download nothing the ledger already records.

## What fetch produces

Start with the output, because everything else is in service of it. For each source of the config, `fetch` writes under `dest` (the corpus directory, `/artifacts/corpus`) the following paths:

| Path | Content |
|---|---|
| `downloads/<id>/<name>.part` | a download in progress; its size is how many bytes have arrived |
| `downloads/<id>/<name>` | a complete download (renamed from `.part`) |
| `quarantine/<id>/<name>` | a complete download whose sha256 is not the pinned one |
| `raw/<id>/<yyyymmdd>/part-<nnnnn>.jsonl.zst` | the source split into documents ([`formats/corpus-shard.md`](../../../../course/contracts/formats/corpus-shard.md)) |
| `LEDGER.jsonl` | one row per fetched source ([`formats/ledger.schema.json`](../../../../course/contracts/formats/ledger.schema.json)) |

`<name>` is the last segment of the URL's path; `<yyyymmdd>` is the UTC date of the fetch. The config itself follows [`corpus-config.schema.json`](../../../../course/contracts/formats/corpus-config.schema.json). A raw document line is compact JSON with keys in this order:

```json
{"url":"http://127.0.0.1:8765/tiny.jsonl#1","fetched_at":"2026-01-01T00:00:00Z","license_spdx":"CC-BY-4.0","text":"Tom has a red ball."}
```

The `#<line>` suffix is the 1-based line where the document starts in the source file, so every document points back into the checksummed bytes. A source's `format` says how to split it. `jsonl` takes the string `"text"` field of each non-blank line. `text` takes each block of non-blank lines (a line of only spaces counts as blank too), joined by `\n`. A line that is not JSON, or has no string `text`, is a data error named by its line number.

## Resuming with Range

With the outputs fixed, the first hard problem is the cut connection. HTTP lets a client ask for part of a file with a **Range** header (RFC 9110, section 14). The notation used below:

| Symbol | Meaning | Type |
|---|---|---|
| $s$ | bytes already in `<name>.part` (its size) | integer |
| $T$ | the file's total size, from `Content-Range` or `Content-Length` | integer |
| `Range: bytes=s-` | "send me the bytes from offset $s$ to the end" | request header |
| `Content-Range: bytes a-b/T` | "these are bytes $a$ to $b$ inclusive of a $T$-byte file" | response header |

The server's answer decides what the partial file means, and there are five cases to distinguish:

| Status | Meaning | What fetch does |
|---|---|---|
| `206 Partial Content`, with `a = s` | the server honoured the range | append the body to the `.part` file |
| `206` with `a ≠ s` | the server sent some other range | delete the `.part` file, retry from 0 |
| `200 OK` | the server ignored the range and sent the whole file | truncate the `.part` file and write from 0 |
| `416 Range Not Satisfiable`, `Content-Range: bytes */T` with $T = s$ | there is nothing after byte $s$: the part is whole | stop and verify |
| `416` with $T \ne s$ | the part is longer than the file (it changed upstream) | delete the part, retry from 0 |

The body is written to the `.part` file *as it arrives*, chunk by chunk, so a connection cut at byte 280,000,000 leaves 280,000,000 bytes on disk for the next attempt. Fetch reads until `Content-Length` bytes have arrived; an empty read before that means the connection was cut, which is retryable. The download is complete when $s = T$, and only then is `.part` renamed to `<name>`. A rename within one directory is atomic, so `<name>` exists only when it is whole.

## Checksums and quarantine

A complete file is not yet a trusted file. SHA-256 maps any byte string to a 32-byte digest, written as 64 hex digits, and no two different files with the same digest have ever been found. The config pins the digest you reviewed. After the download, fetch hashes the file and compares.

On a **match**, the bytes are the reviewed ones, and fetch splits them into raw parts. On a **mismatch**, the server is serving something else: a new version, a corrupted mirror, or an attacker. Fetch moves the file to `quarantine/<id>/` for a human, records the status `"quarantined"` in the manifest, writes nothing to `raw/` or the ledger, and *does not retry*, because the next download will hash the same way. The run continues with the other sources; the manifest's `ok` is false, and the CLI exits 65 (non-retryable).

Some servers send their own checksum header (the test fixture server sends `X-Content-Sha256`). It is not evidence: it comes from the same place as the bytes. Hashing is CPU work, so it runs in a thread (`await asyncio.to_thread(sha256_file, path)`) where it does not stall the event loop.

## Retries, timeouts, and what an error means

Quarantine is one kind of failure; the network produces many others, and the important question about each is whether trying again can help. Fetch reuses the policy from [Python asyncio](../../programming-languages/02-language-and-tool-primers/08-python-asyncio.md). HTTP 5xx and 429, a cut body, a refused or reset connection, and no byte for `timeout` seconds are all retryable. Other 4xx statuses (`404`, `403`), a malformed response, and a source file that is not UTF-8 text or valid JSON Lines are not.

After a retryable failure of attempt $n$, fetch waits $b \cdot 2^{n-1}$ seconds through the injected `sleep` (here $b$ is `base_delay`, default 0.5 s), for at most `attempts` attempts. The timeout bounds the wait for *each read*, not the whole download: a healthy 300 MB download can take minutes, but a server that sends nothing for 30 s is dead.

A source that fails for good raises `FetchError(source_id, message, retryable)`. The [subprocess activity contract](../../../../course/contracts/spec/subprocess-activity.md) turns `retryable=True` into exit 75 (the durable engine retries the activity later) and `retryable=False` into exit 65 (it does not). Redirects (`301`, `302`, `303`, `307`, `308` with a `Location` header) are followed up to 5 times, because dataset hosts answer file URLs with a redirect to a CDN; the ledger keeps the configured URL.

## Concurrency, order, and idempotence

One source at a time would be correct but slow. Downloads run in an `asyncio.TaskGroup`, each inside one shared `asyncio.Semaphore(concurrency)`, exactly as in the asyncio primer. When one source fails for good, the task group cancels the others, and their `.part` files stay on disk for the next run. Three rules keep reruns cheap and outputs comparable.

First, a source is **cached**, skipped with no request at all, when `LEDGER.jsonl` has a row with its id *and its pinned sha256* and that row's raw directory exists. A new pinned sha256 for the same id is a new version and is fetched into a new dated directory. Second, **raw parts are written atomically**: they go into `raw/<id>/<yyyymmdd>.tmp/` and the directory is renamed into place last, so a crash never leaves a half-written raw directory that looks complete. Third, **the ledger is deterministic**: rows are appended once the downloads end, *in the order of the config*, not in the order downloads finished, and they are appended also when another source failed (in a `finally:`), so a finished source is never fetched twice. The manifest lists entries in input order too.

The license is captured here, at fetch time, because this is the only moment when the bytes, the URL, the license you reviewed, and the time are all in one place. `ledger_row(f)` writes every field of the schema: `source_id`, `url`, `license_spdx`, `retrieved_at` (UTC, `YYYY-MM-DDTHH:MM:SSZ`), `sha256` of the bytes, `n_docs`, `allowed_uses`, `pii_policy` (default `scrub`, as set out in [Privacy and PII policy](../../responsible-ai/02-privacy-and-pii/01-privacy-and-pii-policy.md)), and, because nothing is filtered yet, `filters_applied = []`, `kept = n_docs`, `dropped = 0`, `notes = ""`. Time comes from an injected `clock` (`clock.now()` in Unix seconds), so a test or a replay can fix it.

## Working it by hand

Now trace one source through a cut and a resume. The source is `tiny`, license `CC-BY-4.0`, format `jsonl`, two documents in 54 bytes:

```text
{"text": "Tom has a red ball."}\n{"text": "Mia naps."}\n
```

Its sha256 is `e9d5b783f9fbf6ec94f8144919e20e8fea345376df3ad331cc01df785a7bd42b`, pinned in the config. The clock reads 1767225600, which is 2026-01-01T00:00:00Z. The server cuts the first response after 20 bytes. The timeline:

| Step | Request | Response | On disk |
|---|---|---|---|
| 1 | `GET /tiny.jsonl` (no Range: no `.part` yet) | `200`, `Content-Length: 54`, connection cut after 20 bytes | `downloads/tiny/tiny.jsonl.part`: the 20 bytes `{"text": "Tom has a ` |
| 2 | (retryable: wait $0.5 \cdot 2^0 = 0.5$ s) | | |
| 3 | `GET /tiny.jsonl` with `Range: bytes=20-` | `206`, `Content-Range: bytes 20-53/54`, 34 bytes | `.part` has $20 + 34 = 54 = T$ bytes: renamed to `tiny.jsonl` |
| 4 | | | sha256 of the 54 bytes equals the pinned digest |
| 5 | | | `raw/tiny/20260101/part-00000.jsonl.zst` holds 2 lines, `url` ending `#1` and `#2` |

`fetch` returns one `Fetched` with status `fetched`, 2 requests, 54 bytes, and `n_docs = 2`, and `LEDGER.jsonl` gains this row:

```json
{"source_id":"tiny","url":"http://127.0.0.1:8765/tiny.jsonl","license_spdx":"CC-BY-4.0","retrieved_at":"2026-01-01T00:00:00Z","sha256":"e9d5b783f9fbf6ec94f8144919e20e8fea345376df3ad331cc01df785a7bd42b","n_docs":2,"allowed_uses":["train","eval"],"pii_policy":"scrub","filters_applied":[],"kept":2,"dropped":0,"notes":""}
```

Run `fetch` again. The ledger has `tiny` with that sha256 and `raw/tiny/20260101/` exists, so the source is `cached`, the server sees no request, and the ledger is unchanged. Had the config pinned `e9d5...` while the server sent different bytes, step 4 would have moved the file to `quarantine/tiny/tiny.jsonl` and stopped there.

## Designing the interface

The public surface is small: three frozen dataclasses, one exception, three helpers, and the async driver.

```python
@dataclass(frozen=True)
class Source:  id, url, sha256, license_spdx, allowed_uses=("train", "eval"), format="jsonl", pii_policy="scrub"
@dataclass(frozen=True)
class Fetched: source, status, raw_dir, sha256, bytes, n_docs, retrieved_at, requests
@dataclass(frozen=True)
class Manifest: entries          # property ok: no source quarantined
class FetchError(Exception):     # source_id, retryable

def sources_from_config(config: Mapping[str, Any]) -> list[Source]: ...
def split_documents(data: bytes, fmt: str) -> Iterator[tuple[int, str]]: ...
def read_raw(raw_dir: str | Path) -> Iterator[dict[str, str]]: ...
def ledger_row(f: Fetched) -> dict[str, Any]: ...
async def fetch(srcs, dest, *, concurrency=8, attempts=4, timeout=30.0, base_delay=0.5,
                part_docs=100_000, clock=None, sleep=asyncio.sleep) -> Manifest: ...
```

The full contract, word by word, is [`fetch.pyi`](../../../../course/contracts/py/corpus/fetch.pyi). The solution uses `asyncio.open_connection` for HTTP (with `ssl=ssl.create_default_context()` for `https://`), `hashlib` for SHA-256, and `zstandard.ZstdCompressor(level=3)` for the parts. Besides the public names, [`fetch.py`](../../../../course/solutions/python/corpus/fetch.py) has four private helpers: `_open` (HTTP/1.1 over asyncio streams), `_attempt` (one resumable GET), `_one` (one source end to end), and `_write_parts`.

The solution's CLI, [`corpus/__main__.py`](../../../../course/solutions/entry/python/corpus/__main__.py), runs fetch as the first stage of `corpus run --config C` (stop after it with `--until fetch`). It parses the TOML config, calls `sources_from_config` and `asyncio.run(fetch(...))`, and exits 0, 65 (a config error, a quarantined source, or `FetchError` with `retryable=False`), or 75 (`FetchError` with `retryable=True`).

## Building it step by step

Before the code, here is the chain of reasoning that produces it. Each step follows from a requirement.

1. **The output must be a function of the config, not of the network's mood.** So every byte that reaches `raw/` must be the pinned bytes: verify with SHA-256 against the config's digest, and treat anything else as a different file (quarantine, never retry, never trust a header from the same server).
2. **A cut must not cost the bytes already received.** So the body is written to `.part` chunk by chunk as it arrives, and the next attempt asks for `Range: bytes=<size>-`. The server's answer decides what the part file means: append on a matching 206, restart on 200 or a mismatched 206, accept on a 416 whose total equals the size.
3. **"Complete" must be observable and atomic.** A download is complete when its size equals the total; only then is `.part` renamed. Raw parts go into a `.tmp` directory renamed into place last. A crash at any point leaves either a resumable `.part`, a verified file not yet split, or a complete raw directory; each has a recovery path, and nothing looks complete that is not.
4. **An error must say whether retrying can help,** because the workflow engine above will act on it. Transport failures, 5xx, 429, and timeouts are `_Retry`; other 4xx, malformed responses, and bad data are `ValueError`. At the source level they become `FetchError(retryable=True/False)`, which the CLI maps to exit 75 or 65.
5. **Timeouts bound silence, not total time.** Each connect, header read, and body read has its own `wait_for(timeout)`, so a slow but steady 300 MB download survives and a silent server does not.
6. **Reruns must be free.** The ledger already records what was fetched under which digest, so a source with a matching row and an existing raw directory is skipped before any request. Rows are written in config order in a `finally:`, so they are deterministic and survive another source's failure.
7. **Concurrency must be bounded** (politeness, file descriptors): one semaphore around each source's work inside a task group, which also cancels the rest on a hard failure, leaving their `.part` files for the next run.

### One attempt

`_attempt` is the Range table in code. Notice that the offset comes from the part file itself, so there is no separate state to get out of sync.

```python
async def _attempt(url: str, part: Path, timeout: float) -> tuple[int, int | None]:
    offset = part.stat().st_size if part.exists() else 0     # bytes already on disk
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    try:
        head, reader, writer, _ = await _open(url, headers, timeout)  # follows redirects
    except (OSError, asyncio.TimeoutError, asyncio.IncompleteReadError) as e:
        raise _Retry(f"connect: {type(e).__name__}: {e}") from None   # refused/reset/timeout
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
            raise _Retry(f"HTTP {head.status}")                # the server may recover
        if head.status not in (200, 206):
            raise ValueError(f"HTTP {head.status}")            # 404, 403: retrying cannot help
        # ... chunked transfer coding is refused (ValueError) ...
        total: int | None = None
        if head.status == 206:
            spec = h.get("content-range", "")                  # bytes <first>-<last>/<total>
            first = spec.removeprefix("bytes ").partition("-")[0]
            if not first.isdigit() or int(first) != offset:
                part.unlink(missing_ok=True)                   # some other range: start over
                raise _Retry(f"server resumed at {spec!r}, not byte {offset}")
            tail = spec.rpartition("/")[2]
            total = int(tail) if tail.isdigit() else None
            mode = "ab"                                        # continue the file
        else:
            # 200: the server sent the whole file, whatever we asked for.
            offset, mode = 0, "wb"                             # truncate, never append
        length = int(h["content-length"]) if "content-length" in h else None
        if total is None and length is not None:
            total = offset + length
        got = 0
        with part.open(mode) as fh:
            while length is None or got < length:
                want = CHUNK if length is None else min(CHUNK, length - got)
                try:
                    chunk = await asyncio.wait_for(reader.read(want), timeout)  # per-read timeout
                except (OSError, asyncio.TimeoutError) as e:
                    raise _Retry(f"body: {type(e).__name__}") from None
                if not chunk:
                    if length is None:
                        break                                  # no length: EOF ends the body
                    raise _Retry(f"body cut at {offset + got} of {total} bytes")
                fh.write(chunk)                                # on disk as it arrives
                got += len(chunk)
        return offset + got, total
    finally:
        writer.close()
```

`_open` speaks just enough HTTP/1.1. It sends a GET with `Connection: close` and `Accept-Encoding: identity` (so byte offsets are offsets into the file, not into a compressed stream), reads the status line and headers each under `wait_for(timeout)`, and follows up to `MAX_REDIRECTS = 5` redirects with `urljoin`. A connection closed before a status line is `_Retry`; a malformed one is `ValueError`.

### One source

`_one` wraps attempts into the life of a source. It checks the ledger, recovers a verified download left by a crash, retries attempts with backoff, verifies, and either quarantines or writes parts.

```python
async def _one(src, dest, ledger, *, attempts, timeout, base_delay, part_docs, clock, sleep) -> Fetched:
    for row in ledger:
        if row.get("source_id") == src.id and row.get("sha256") == src.sha256:
            raw = _raw_dir(dest, src.id, row["retrieved_at"])
            if raw.is_dir():                                   # recorded AND on disk
                return Fetched(src, "cached", str(raw), src.sha256, 0,
                               row["n_docs"], row["retrieved_at"], 0)   # zero requests
    retrieved_at = _rfc3339(clock.now())                       # UTC, ...Z
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
                break                                          # whole
            except _Retry as e:
                if n == attempts:
                    raise FetchError(src.id, f"gave up after {attempts} attempts: {e}", True) from None
            except ValueError as e:
                raise FetchError(src.id, str(e), False) from None   # no retry, no wait
            await sleep(base_delay * 2 ** (n - 1))             # 0.5, 1.0, 2.0, ... via injected sleep
        os.replace(part, done)                                 # atomic: `done` exists only when whole
        got = await asyncio.to_thread(_sha256_file, done)      # CPU work off the event loop
    size = done.stat().st_size
    if got != src.sha256:                                      # only the PINNED digest counts
        q = dest / "quarantine" / src.id
        q.mkdir(parents=True, exist_ok=True)
        os.replace(done, q / name)
        return Fetched(src, "quarantined", "", got, size, 0, retrieved_at, requests)
    raw = _raw_dir(dest, src.id, retrieved_at)                 # raw/<id>/<yyyymmdd>
    try:
        n_docs = await asyncio.to_thread(_write_parts, src, done.read_bytes(), raw, retrieved_at, part_docs)
    except ValueError as e:
        raise FetchError(src.id, str(e), False) from None      # bad data: not retryable
    return Fetched(src, "fetched", str(raw), got, size, n_docs, retrieved_at, requests)
```

### Raw parts

Documents are compact JSON with the contract's key order (`url`, `fetched_at`, `license_spdx`, `text`), and each `url` carries `#<line>` back into the checksummed file. Everything goes into a `.tmp` sibling that replaces the final directory last.

```python
def _write_parts(src, data, raw_dir, retrieved_at, part_docs) -> int:
    tmp = raw_dir.with_name(raw_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)                                     # leftovers of a crashed run
    tmp.mkdir(parents=True)
    cctx = zstandard.ZstdCompressor(level=3)
    n, buf, part_no = 0, [], 0
    # flush() writes buf to tmp/part-<part_no:05d>.jsonl.zst and starts the next part
    for line, text in split_documents(data, src.format):
        doc = {"url": f"{src.url}#{line}", "fetched_at": retrieved_at,
               "license_spdx": src.license_spdx, "text": text}
        buf.append(json.dumps(doc, ensure_ascii=False, separators=(",", ":")) + "\n")
        n += 1
        if len(buf) == part_docs:
            flush()                                            # bounded parts for later stages
    if buf or part_no == 0:
        flush()                                                # at least one part, even if empty
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    os.replace(tmp, raw_dir)                                   # the directory appears complete
    return n
```

### The driver

The driver is one semaphore, one task group, results keyed by id, and the ledger written in a `finally:` in config order. The first underlying exception is re-raised so the caller sees a plain `FetchError`, not an exception group.

```python
async def fetch(srcs, dest, *, concurrency=8, attempts=4, timeout=30.0, base_delay=0.5,
                part_docs=100_000, clock=None, sleep=asyncio.sleep) -> Manifest:
    # ... validate concurrency/attempts/part_docs >= 1 and unique ids ...
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    clock = clock or _WallClock()
    ledger = _read_ledger(dest)                                # read once, before any task runs
    sem = asyncio.Semaphore(concurrency)
    results: dict[str, Fetched] = {}

    async def run(s: Source) -> None:
        async with sem:                                        # at most `concurrency` in flight
            results[s.id] = await _one(s, dest, ledger, attempts=attempts, timeout=timeout,
                                       base_delay=base_delay, part_docs=part_docs,
                                       clock=clock, sleep=sleep)

    try:
        async with asyncio.TaskGroup() as tg:                  # one hard failure cancels the rest
            for s in srcs:
                tg.create_task(run(s))
    except BaseExceptionGroup as eg:
        first = eg.exceptions[0]
        while isinstance(first, BaseExceptionGroup):
            first = first.exceptions[0]
        raise first from None
    finally:
        # config order, not completion order; also when another source failed
        rows = [ledger_row(results[s.id]) for s in srcs
                if s.id in results and results[s.id].status == "fetched"]
        if rows:
            with (dest / LEDGER_NAME).open("a", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
                fh.flush()
                os.fsync(fh.fileno())                          # durable before we report success
    return Manifest(tuple(results[s.id] for s in srcs))
```

`ledger_row` is a pure mapping from `Fetched` to the schema's fields (nothing filtered yet: `filters_applied = []`, `kept = n_docs`, `dropped = 0`), and `split_documents` implements the `jsonl` and `text` rules described at the start of the section, raising `ValueError` with the line number for bad data. The full solution is [`fetch.py`](../../../../course/solutions/python/corpus/fetch.py).

### Why this design

The cost is network-bound: each source is downloaded at most once across any number of runs and cuts, plus one SHA-256 pass (in a thread) and one compression pass. The body streams in 64 KiB chunks, but `_write_parts` reads the verified file whole (`done.read_bytes()`). That is fine for the corpus sizes here and would be the first thing to stream for multi-gigabyte sources.

You could use `urllib` or `requests` in threads. They work, but the asyncio policy this book follows wants one event loop with a semaphore and per-read timeouts, and a hand-written minimal HTTP/1.1 client over `asyncio.open_connection` keeps the dependency list to `zstandard` and makes every status and header decision visible. `aiohttp` or `httpx` are production-grade and preferable for a real crawler (chunked encoding, keep-alive, HTTP/2), but they are a dependency the contract does not allow, and they would hide exactly the Range and status logic this section is about. Trusting `ETag` or a server checksum header is tempting, but those come from the same server as the bytes; only a digest reviewed out of band proves the bytes are the reviewed ones. Writing ledger rows as downloads finish is simpler, but the ledger's line order (and the corpus build's output hash) would then depend on network timing.

Three tradeoffs are accepted deliberately. Chunked transfer coding is refused rather than implemented, since static dataset files are served with a length. Parquet sources are rejected here and left to [Parquet shards and the manifest](06-parquet-shards-and-manifest.md). And a quarantined source makes `ok` false but does not stop the other sources, so one bad mirror does not waste a whole run.

## What a correct fetch must handle

None of these cases needs the real network: a local fixture server that can cut, lie, redirect, and stall is enough. The table lists each case, what is checked, and why it matters downstream.

| Case | Checks | Why it matters |
|---|---|---|
| The hand example: resume after a cut | the timeline above: `Range: bytes=20-`, 2 requests, 54 bytes, the raw lines, the exact ledger row | every byte is pinned |
| Raw parts are zstd JSON lines | parts decoded with `zstandard` itself: magic bytes, key order, compact JSON, UTF-8 | extraction and any auditor read them with stock tools |
| Ledger rows validate against the schema | every row against `formats/ledger.schema.json`; UTC with `Z` | ledger verification and model release read these rows |
| `ledger_row` is a pure function of `Fetched` | `ledger_row` maps each field | the row's shape lives in one place |
| A server that ignores Range restarts the file | a 200 to a Range request restarts the file | a good source is not quarantined for a server quirk |
| Cut bodies resume until whole | 3 cuts, 4 attempts, ranges 500, 1000, 1500 | long downloads survive flaky networks |
| A checksum mismatch is quarantined, not retried | 1 request, file in `quarantine/`, no raw parts, no ledger row, `ok` false | changed bytes never reach the corpus |
| The server's checksum header is not trusted | a lying `X-Content-Sha256` header changes nothing | only the pinned digest counts |
| A rerun downloads nothing | second run: no requests, `cached`, ledger byte-identical | `CorpusBuild` reruns cost nothing |
| A new pinned checksum fetches again | a new pinned sha256 fetches into a new dated directory | versions are tracked, not overwritten |
| A whole partial file gets 416 and is kept | a complete `.part` plus `416` is verified, not failed | crash between last byte and rename |
| A verified download left by a crash is not downloaded again | a complete download with no raw parts is converted without a request | crash between verify and split |
| 5xx is retried with exponential backoff | 503, 503, 200: waits `[0.5, 1.0]` | backoff spares a struggling mirror |
| A 404 is not retried | 1 request, `FetchError.retryable` false | the workflow does not retry a wrong URL |
| Exhausted retries raise a retryable error | 3 attempts, waits `[0.1, 0.2]`, `retryable` true | the workflow retries later |
| A silent server times out | silent for 1 s with `timeout=0.1` fails, retryable | a dead mirror cannot hang the pipeline |
| Concurrency is bounded | 8 sources, `concurrency=3`: exactly 3 in flight | politeness and throughput |
| Ledger rows follow the input order | rows in config order although the first source finishes last | the build's output hash is deterministic |
| A failure keeps the rows of finished sources | a 404 still leaves the finished source's row | no download is wasted |
| Redirects are followed | 302 to a CDN path; the ledger keeps the configured URL | real dataset hosts redirect |
| Text sources split on blank lines | blocks, blank lines of spaces, start lines | `text` sources such as books |
| JSONL errors name the line | bad lines raise `ValueError` naming the line; `FetchError` not retryable | data errors are reported, not retried |
| Parts hold at most `part_docs` documents | 5 docs with `part_docs=2`: 3 parts, read in order | later stages read bounded files |
| `sources_from_config` fills defaults and rejects bad entries | schema defaults; duplicate ids and malformed digests rejected | the config fails before any download |

## What can go wrong

Most fetch bugs corrupt the file rather than crash, which makes them expensive. The commonest is retrying a cut download without a Range header while appending to the part: you get prefix plus whole file, a checksum mismatch, and a good source quarantined. `_attempt` avoids it by sending `Range: bytes=<part size>-` whenever the part is non-empty. Its twin is appending a `200` reply to the partial file, which produces the same corrupt file whenever a server ignores ranges; a `200` therefore opens the part with `"wb"` and resets the offset to 0.

> **Warning:** Never retry after a checksum mismatch, and never verify against the server's own checksum header. Retrying downloads the same wrong bytes again and again; trusting the header lets a swapped file pass and fails a good file against a lying header. In the solution, verification happens after the retry loop, a mismatch moves the file to `quarantine/` and returns, and `_one` compares only with `src.sha256`.

Retrying every HTTP error is the next trap: a 404 then costs `attempts` requests and the workflow retries it forever. Only 5xx, 429, and transport failures are `_Retry`; other statuses are `ValueError`, so the result is `FetchError(retryable=False)` at once.

Several pitfalls cost reruns rather than correctness. Not consulting the ledger before downloading means every crash recovery downloads the whole corpus again; `_one` returns `cached` for a row with the id, the pinned digest, and an existing raw directory. Writing the ledger only when every source succeeded means finished sources are fetched again after any failure; the `finally:` block prevents that. Appending ledger rows as downloads finish gives a different ledger, and a different output hash, on every run; rows are built from `results` in the order of `srcs`.

The remaining three are about resources and time. With no semaphore around the downloads, every source starts at once and you are throttled by the mirror or run out of file descriptors; each task runs `_one` inside `async with sem`. With no timeout on reads, a silent server hangs the run, and with one timeout for the whole download a large healthy download is killed; every connect, header line, and body read has its own `wait_for(timeout)`. And local time in `retrieved_at` produces rows that fail the schema pattern and dates that depend on the machine's time zone; `_rfc3339` formats in UTC with a trailing `Z`.

## Where this leads

Every later stage reads what fetch wrote. [Extract, normalize, and quality filters](02-extract-normalize-and-quality-filters.md) calls `extract(manifest)`, which reads each source's raw parts with `read_raw` and numbers the documents `<source_id>:<k>`. [Ledger verification and the datasheet](08-ledger-and-datasheet.md) verifies every ledger row against the schema and the license allowlist from [Data licensing and the ledger](../../responsible-ai/01-data-licensing/01-data-licensing-and-the-ledger.md) before a shard may be used. And [CorpusBuild as a durable workflow](09-corpusbuild-as-a-durable-workflow.md) runs `{corpus} run --stage fetch` as a subprocess activity, where exit 75 retries it and exit 65 fails it. Looking back, the semaphore, task group, per-attempt timeout, and backoff are unchanged from [Python asyncio](../../programming-languages/02-language-and-tool-primers/08-python-asyncio.md).

### Further reading

- RFC 9110, sections 14 (Range requests) and 15.3.7 / 15.5.17 (206, 416).
- Penedo et al., *The FineWeb Datasets* (2024), section 3: how a web-scale corpus is collected.
- Gebru et al., *Datasheets for Datasets* (2021), "Collection process".
- `huggingface_hub.hf_hub_download` (`huggingface_hub/file_download.py`): the production equivalent of `fetch`, adding resume, ETag-based caching, symlinked snapshots per revision, and parallel chunked downloads.
- `aria2` (manual, "Segmented download") and `curl --continue-at -`: many connections per file, mirror lists, rate limits.
- datatrove's `JsonlWriter` and readers (`datatrove/pipeline/writers/jsonl.py`) and Dolma's source metadata: sharded writers and per-document provenance at web scale.
- Longpre et al. (2023), the Data Provenance Initiative's provenance cards: license and lineage per dataset, aggregated across collections.

## Key terms

- **Range request**: an HTTP request with `Range: bytes=s-` asking for the bytes from offset $s$ to the end of a file.
- **Part file**: `<name>.part`, a download in progress whose size is the resume offset.
- **Pinned checksum**: the sha256 recorded in the config after review; the only digest fetch trusts.
- **Quarantine**: the directory where a complete download with the wrong digest is moved for a human, never reaching `raw/` or the ledger.
- **Retryable error**: a failure that a later attempt can fix (5xx, 429, cut, timeout); mapped to exit 75.
- **Ledger row**: one JSON record per fetched source with its URL, license, digest, time, and document count.

## Summary

- A cut download resumes with `Range: bytes=<size>-`: a `206` continues the file, a `200` means the server ignored the range and the file starts over, a `416` means the partial file is already whole.
- The checksum pinned in the config is the only one to trust. A mismatch is quarantined, not retried, and never reaches the corpus.
- The license is captured at fetch time, one ledger row per source, tied to the sha256 of the exact bytes.
- A rerun downloads nothing the ledger already records, so a crashed or retried pipeline costs nothing twice.
- Errors say whether a retry can help: `FetchError.retryable` maps to exit 75 or 65 under the subprocess activity contract.

## Review questions

1. A Range request gets a `200`. Why does the solution truncate the part file instead of appending?

   <details><summary>Answer</summary>

   A `200` means the server ignored the range and is sending the whole file from byte 0. Appending would produce prefix plus whole file, whose SHA-256 cannot match, so a perfectly good source would be quarantined. Opening with `"wb"` and resetting `offset` to 0 makes the part file hold exactly what the server is sending.

   </details>

2. Why is a checksum mismatch quarantined and not retried, when a cut body is retried?

   <details><summary>Answer</summary>

   A cut is a transport failure: the next attempt can get the remaining bytes. A mismatch on a complete file means the server is serving different bytes (a new version, a corrupt mirror, an attacker); downloading again would hash the same way. Retrying wastes bandwidth and, worse, invites the workflow to retry forever. Quarantine hands the file to a human and keeps it out of `raw/` and the ledger.

   </details>

3. Why is the ledger written in a `finally:` block, in config order?

   <details><summary>Answer</summary>

   In a `finally:` so that when one source fails for good, the sources that already finished still get their rows and are `cached` on the next run instead of being downloaded again. In config order so the ledger's content is a function of the config, not of which download happened to finish first; the corpus build hashes its outputs and that hash must be reproducible.

   </details>

4. Why does the timeout wrap each read rather than the whole download?

   <details><summary>Answer</summary>

   The failure to detect is a dead or stalled server, which shows up as no bytes for a while. A whole-download timeout would have to be long enough for the largest healthy file (minutes), so it would let a dead server hang the run for that long, or kill healthy large downloads if set shorter. A per-read timeout of 30 s catches silence quickly and never limits a steady transfer.

   </details>

5. A crash happens after `os.replace(part, done)` but before the raw directory is renamed into place. What does the next run do?

   <details><summary>Answer</summary>

   There is no ledger row yet (rows are written at the end), so the source is not cached. `_one` finds `downloads/<id>/<name>` as a file, hashes it, sees the pinned digest, and skips the download entirely (zero requests), then writes the parts again into a fresh `.tmp` directory (removing any leftover) and renames it into place.

   </details>
