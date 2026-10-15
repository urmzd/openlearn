<!-- ol:module data.01 -->
# Async fetch with resume, checksums, license capture

## Overview

| | |
|---|---|
| **Module** | `data.01` · build · Python · Pass 3 · 4 to 6 h |
| **You build** | `python/corpus/fetch.py`: `Source`, `Fetched`, `Manifest`, `FetchError`, `sources_from_config`, `split_documents`, `read_raw`, `ledger_row`, and `async def fetch` |
| **Contract** | [`course/contracts/py/corpus/fetch.pyi`](../../../../course/contracts/py/corpus/fetch.pyi) · formats: [`corpus-shard.md`](../../../../course/contracts/formats/corpus-shard.md) (raw documents), [`ledger.schema.json`](../../../../course/contracts/formats/ledger.schema.json), [`corpus-config.schema.json`](../../../../course/contracts/formats/corpus-config.schema.json) · exit codes: [`spec/subprocess-activity.md`](../../../../course/contracts/spec/subprocess-activity.md) |
| **Tests** | `course/tests/data.01/` (what they check: section 4); no test touches the network |
| **Needs** | nothing to call: reading `lang.08` (asyncio), `lang.05` (HTTP/1.1), `ethics.01` (what the ledger is for) |
| **Used by** | `data.02` extracts documents from your raw parts with `read_raw` · later: the `CorpusBuild` workflow runs `fetch` as its first activity (`data.09`), and `data.08` verifies the ledger rows it writes |
| **Milestone** | `MS-corpus` |
| **Optional depth** | RFC 9110 sections 14 (Range requests) and 15.3.7 / 15.5.17 (206, 416); Penedo et al., *The FineWeb Datasets* (2024), section 3 (how a web-scale corpus is collected); Gebru et al., *Datasheets for Datasets* (2021), "Collection process" |

## Key Takeaways

- A cut download resumes with `Range: bytes=<size>-`: a `206` continues the file, a `200` means the server ignored the range and the file starts over, a `416` means the partial file is already whole (`test_hand_example_resume_after_a_cut`, `test_a_server_that_ignores_range_restarts_the_file`, `test_a_whole_partial_file_gets_416_and_is_kept`).
- The checksum you pinned in the config is the only one you trust. A mismatch is quarantined, not retried, and never reaches the corpus (`test_a_checksum_mismatch_is_quarantined_not_retried`, `test_the_servers_checksum_header_is_not_trusted`).
- The license is captured at fetch time, one ledger row per source, tied to the sha256 of the exact bytes (`test_ledger_rows_validate_against_the_schema`).
- A rerun downloads nothing the ledger already records, so a crashed or retried pipeline costs nothing twice (`test_a_rerun_downloads_nothing`).
- Errors say whether a retry can help: `FetchError.retryable` maps to exit 75 or 65 under the subprocess activity contract (`test_a_404_is_not_retried`, `test_exhausted_retries_raise_a_retryable_error`).

## How to work this chapter

```bash
ol start data.01              # stubs python/corpus/fetch.py; contract alongside
ol tests data.01              # read the test catalog first
ol check data.01              # exit code is the verdict
ol diff  data.01              # after passing: your code against the reference
```

Your `python/pyproject.toml` must list `zstandard` (it is in `contracts/allowed-deps.toml` for `corpus`): the raw parts are zstd files. `ol start` writes a `pyproject.toml` that does when you have none.

---

## 1. Why now

Your tokenizers (Pass 3, `L1.*`) are trained on whatever text you hand them, and so far that text has been a file you copied by hand. The corpus pipeline starts here: a config names the sources (a URL, the sha256 you reviewed, the license), and the pipeline must turn that list into documents on disk, the same documents every time, with a record of where each came from and under what terms. The network makes this hard. A 300 MB download from a dataset mirror fails at 280 MB more often than you would like; mirrors answer `503` under load; a URL you pinned last month now serves different bytes; and the `CorpusBuild` workflow (`data.09`) will rerun this step after every crash. Without resume, every cut restarts from zero; without the checksum, a changed file silently changes your corpus; without the ledger, nobody can later answer "may we train on this?".

## 2. Principles

### 2.1 What fetch produces

For each source of the config, `fetch` writes under `dest` (the corpus directory, `/artifacts/corpus`):

| Path | Content |
|---|---|
| `downloads/<id>/<name>.part` | a download in progress; its size is how many bytes have arrived |
| `downloads/<id>/<name>` | a complete download (renamed from `.part`) |
| `quarantine/<id>/<name>` | a complete download whose sha256 is not the pinned one |
| `raw/<id>/<yyyymmdd>/part-<nnnnn>.jsonl.zst` | the source split into documents (`formats/corpus-shard.md`) |
| `LEDGER.jsonl` | one row per fetched source (`formats/ledger.schema.json`) |

`<name>` is the last segment of the URL's path; `<yyyymmdd>` is the UTC date of the fetch. A raw document line is compact JSON with keys in this order:

```json
{"url":"http://127.0.0.1:8765/tiny.jsonl#1","fetched_at":"2026-01-01T00:00:00Z","license_spdx":"CC-BY-4.0","text":"Tom has a red ball."}
```

`#<line>` is the 1-based line where the document starts in the source file, so every document points back into the bytes you checksummed. A source's `format` says how to split it: `jsonl` takes the string `"text"` field of each non-blank line, and `text` takes each block of non-blank lines (a line of only spaces is blank too), joined by `\n`. A line that is not JSON, or has no string `text`, is a data error named by its line number.

### 2.2 Resuming with Range

HTTP lets a client ask for part of a file with a **Range** header (RFC 9110, section 14):

| Symbol | Meaning | Type |
|---|---|---|
| $s$ | bytes already in `<name>.part` (its size) | integer |
| $T$ | the file's total size, from `Content-Range` or `Content-Length` | integer |
| `Range: bytes=s-` | "send me the bytes from offset $s$ to the end" | request header |
| `Content-Range: bytes a-b/T` | "these are bytes $a$ to $b$ inclusive of a $T$-byte file" | response header |

The answer decides what to do with the partial file:

| Status | Meaning | What fetch does |
|---|---|---|
| `206 Partial Content`, with `a = s` | the server honoured the range | append the body to the `.part` file |
| `206` with `a ≠ s` | the server sent some other range | delete the `.part` file, retry from 0 |
| `200 OK` | the server ignored the range and sent the whole file | truncate the `.part` file and write from 0 |
| `416 Range Not Satisfiable`, `Content-Range: bytes */T` with $T = s$ | there is nothing after byte $s$: the part is whole | stop and verify |
| `416` with $T \ne s$ | the part is longer than the file (it changed upstream) | delete the part, retry from 0 |

The body is written to the `.part` file **as it arrives**, chunk by chunk, so a connection cut at byte 280,000,000 leaves 280,000,000 bytes on disk for the next attempt. Read until `Content-Length` bytes have arrived; an empty read before that means the connection was cut, which is retryable. The download is complete when $s = T$; only then is `.part` renamed to `<name>`. A rename within one directory is atomic, so `<name>` exists only when it is whole.

### 2.3 Checksums and quarantine

SHA-256 maps any byte string to a 32-byte digest, written as 64 hex digits; two different files with the same digest have never been found. The config pins the digest you reviewed. After the download, hash the file and compare:

- **Match**: the bytes are the ones you reviewed. Split them into raw parts.
- **Mismatch**: the server is serving something else (a new version, a corrupted mirror, an attacker). Move the file to `quarantine/<id>/` for a human, record the status `"quarantined"` in the manifest, write nothing to `raw/` or the ledger, and **do not retry**: the next download will hash the same way. The run continues with the other sources; the manifest's `ok` is false, and the CLI exits 65 (non-retryable).

Some servers send their own checksum header (the fixture server sends `X-Content-Sha256`). It is not evidence: it comes from the same place as the bytes. Hashing is CPU work, so it runs in a thread (`await asyncio.to_thread(sha256_file, path)`) where it does not stall the event loop.

### 2.4 Retries, timeouts, and what an error means

`fetch` reuses `lang.08`'s policy. Retryable: HTTP 5xx and 429, a cut body, a refused or reset connection, no byte for `timeout` seconds. Not retryable: other 4xx (`404`, `403`), a malformed response, a source file that is not UTF-8 text or valid JSON Lines. After a retryable failure of attempt $n$ wait $b \cdot 2^{n-1}$ seconds through the injected `sleep` (here $b$ is `base_delay`, default 0.5 s), for at most `attempts` attempts. The timeout bounds the wait for **each read**, not the whole download: a healthy 300 MB download can take minutes, a server that sends nothing for 30 s is dead.

A source that fails for good raises `FetchError(source_id, message, retryable)`. The subprocess activity contract turns `retryable=True` into exit 75 (the durable engine retries the activity later) and `retryable=False` into exit 65 (it does not). Redirects (`301`, `302`, `303`, `307`, `308` with a `Location` header) are followed up to 5 times; dataset hosts answer file URLs with a redirect to a CDN, and the ledger keeps the URL you configured.

### 2.5 Concurrency, order, and idempotence

Downloads run in an `asyncio.TaskGroup`, each inside one shared `asyncio.Semaphore(concurrency)`, exactly as in `lang.08`. When one source fails for good, the task group cancels the others; their `.part` files stay on disk for the next run. Three rules keep reruns cheap and outputs comparable:

- **Cached**: a source is skipped, with no request at all, when `LEDGER.jsonl` has a row with its id **and its pinned sha256** and that row's raw directory exists. A new pinned sha256 for the same id is a new version and is fetched into a new dated directory.
- **Atomic raw parts**: parts are written into `raw/<id>/<yyyymmdd>.tmp/` and the directory is renamed into place last, so a crash never leaves a half-written raw directory that looks complete.
- **Deterministic ledger**: rows are appended once the downloads end, **in the order of the config**, not in the order downloads finished, and also when another source failed (in a `finally:`), so a finished source is never fetched twice. The manifest lists entries in input order too.

The license is captured here, at fetch time, because this is the only moment the bytes, the URL, the license you reviewed, and the time are all in one place. `ledger_row(f)` writes every field of the schema: `source_id`, `url`, `license_spdx`, `retrieved_at` (UTC, `YYYY-MM-DDTHH:MM:SSZ`), `sha256` of the bytes, `n_docs`, `allowed_uses`, `pii_policy` (default `scrub`, `ethics.02`), and, because nothing is filtered yet, `filters_applied = []`, `kept = n_docs`, `dropped = 0`, `notes = ""`. Time comes from an injected `clock` (`clock.now()` in Unix seconds), so tests fix it.

## 3. Worked example by hand

One source, `tiny`, license `CC-BY-4.0`, format `jsonl`, two documents in 54 bytes:

```text
{"text": "Tom has a red ball."}\n{"text": "Mia naps."}\n
```

Its sha256 is `e9d5b783f9fbf6ec94f8144919e20e8fea345376df3ad331cc01df785a7bd42b`, pinned in the config. The clock reads 1767225600, which is 2026-01-01T00:00:00Z. The server cuts the first response after 20 bytes.

| Step | Request | Response | On disk |
|---|---|---|---|
| 1 | `GET /tiny.jsonl` (no Range: no `.part` yet) | `200`, `Content-Length: 54`, connection cut after 20 bytes | `downloads/tiny/tiny.jsonl.part`: the 20 bytes `{"text": "Tom has a ` |
| 2 | (retryable: wait $0.5 \cdot 2^0 = 0.5$ s) | | |
| 3 | `GET /tiny.jsonl` with `Range: bytes=20-` | `206`, `Content-Range: bytes 20-53/54`, 34 bytes | `.part` has $20 + 34 = 54 = T$ bytes: renamed to `tiny.jsonl` |
| 4 | | | sha256 of the 54 bytes equals the pinned digest |
| 5 | | | `raw/tiny/20260101/part-00000.jsonl.zst` holds 2 lines, `url` ending `#1` and `#2` |

`fetch` returns one `Fetched` with status `fetched`, 2 requests, 54 bytes, `n_docs = 2`, and `LEDGER.jsonl` gains:

```json
{"source_id":"tiny","url":"http://127.0.0.1:8765/tiny.jsonl","license_spdx":"CC-BY-4.0","retrieved_at":"2026-01-01T00:00:00Z","sha256":"e9d5b783f9fbf6ec94f8144919e20e8fea345376df3ad331cc01df785a7bd42b","n_docs":2,"allowed_uses":["train","eval"],"pii_policy":"scrub","filters_applied":[],"kept":2,"dropped":0,"notes":""}
```

Run `fetch` again: the ledger has `tiny` with that sha256 and `raw/tiny/20260101/` exists, so the source is `cached`, the server sees no request, and the ledger is unchanged. Had the config pinned `e9d5...` while the server sent different bytes, step 4 would move the file to `quarantine/tiny/tiny.jsonl` and stop there.

## 4. The interface

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

The full contract, word by word, is `course/contracts/py/corpus/fetch.pyi`. Use `asyncio.open_connection` for HTTP (with `ssl=ssl.create_default_context()` for `https://`), `hashlib` for SHA-256, and `zstandard.ZstdCompressor(level=3)` for the parts. Your CLI (`{corpus} fetch --config ...`, learner territory) reads the TOML config, calls `sources_from_config` and `asyncio.run(fetch(...))`, and exits 0, 65 (a quarantined source, or `FetchError` with `retryable=False`), or 75.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_resume_after_a_cut` | unit | the section 3 timeline: `Range: bytes=20-`, 2 requests, 54 bytes, the raw lines, the exact ledger row | you and the test agree on every byte |
| `test_raw_parts_are_zstd_json_lines` | conformance | the parts decoded with `zstandard` itself: magic bytes, key order, compact JSON, UTF-8 | data.02 and any auditor read them with stock tools |
| `test_ledger_rows_validate_against_the_schema` | conformance | every row against `formats/ledger.schema.json`; UTC with `Z` | data.08 and dur.12 read these rows |
| `test_ledger_row_is_a_pure_function_of_fetched` | unit | `ledger_row` maps each field | the row's shape lives in one place |
| `test_a_server_that_ignores_range_restarts_the_file` | fault | a 200 to a Range request restarts the file | a good source is not quarantined for a server quirk |
| `test_cut_bodies_resume_until_whole` | fault | 3 cuts, 4 attempts, ranges 500, 1000, 1500 | long downloads survive flaky networks |
| `test_a_checksum_mismatch_is_quarantined_not_retried` | fault | 1 request, file in `quarantine/`, no raw parts, no ledger row, `ok` false | changed bytes never reach the corpus |
| `test_the_servers_checksum_header_is_not_trusted` | fault | a lying `X-Content-Sha256` header changes nothing | only the pinned digest counts |
| `test_a_rerun_downloads_nothing` | fault | second run: no requests, `cached`, ledger byte-identical | `CorpusBuild` reruns cost nothing |
| `test_a_new_pinned_checksum_fetches_again` | boundary | a new pinned sha256 fetches into a new dated directory | versions are tracked, not overwritten |
| `test_a_whole_partial_file_gets_416_and_is_kept` | fault | a complete `.part` plus `416` is verified, not failed | crash between last byte and rename |
| `test_a_verified_download_left_by_a_crash_is_not_downloaded_again` | fault | a complete download with no raw parts is converted without a request | crash between verify and split |
| `test_5xx_is_retried_with_exponential_backoff` | unit | 503, 503, 200: waits `[0.5, 1.0]` | backoff spares a struggling mirror |
| `test_a_404_is_not_retried` | boundary | 1 request, `FetchError.retryable` false | the workflow does not retry a wrong URL |
| `test_exhausted_retries_raise_a_retryable_error` | boundary | 3 attempts, waits `[0.1, 0.2]`, `retryable` true | the workflow retries later |
| `test_a_silent_server_times_out` | fault | a server silent for 1 s with `timeout=0.1` fails, retryable | a dead mirror cannot hang the pipeline |
| `test_concurrency_is_bounded` | unit | 8 sources, `concurrency=3`: exactly 3 in flight | politeness and throughput |
| `test_ledger_rows_follow_the_input_order` | property | rows in config order although the first source finishes last | deterministic output hash (MS-corpus) |
| `test_a_failure_keeps_the_rows_of_finished_sources` | fault | a 404 still leaves the finished source's row | no download is wasted |
| `test_redirects_are_followed` | unit | 302 to a CDN path; the ledger keeps the configured URL | real dataset hosts redirect |
| `test_text_sources_split_on_blank_lines` | unit | blocks, blank lines of spaces, start lines | `text` sources such as books |
| `test_jsonl_errors_name_the_line` | boundary | bad lines raise `ValueError` naming the line; `FetchError` not retryable | data errors are reported, not retried |
| `test_parts_hold_at_most_part_docs_documents` | boundary | 5 docs with `part_docs=2`: 3 parts, read in order | later stages read bounded files |
| `test_sources_from_config_fills_defaults_and_rejects_bad_entries` | unit | schema defaults; duplicate ids and malformed digests rejected | the config fails before any download |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. Retrying a cut download without a Range header, appending to the part | prefix plus whole file: a checksum mismatch and a good source quarantined | `test_hand_example_resume_after_a_cut` (mutant `s08`) |
| 2. Appending a `200` reply to the partial file | the same corrupt file whenever a server ignores ranges | `test_a_server_that_ignores_range_restarts_the_file` (mutant `s01`) |
| 3. Retrying after a checksum mismatch | the same wrong bytes downloaded again and again | `test_a_checksum_mismatch_is_quarantined_not_retried` (mutant `s03`) |
| 4. Verifying against the server's checksum header | a swapped or changed file passes; a good file fails against a lying header | `test_the_servers_checksum_header_is_not_trusted` (mutant `s02`) |
| 5. Retrying every HTTP error | a 404 costs `attempts` requests and the workflow retries it forever | `test_a_404_is_not_retried` (mutant `s04`) |
| 6. Not consulting the ledger before downloading | every crash recovery downloads the whole corpus again | `test_a_rerun_downloads_nothing` (mutant `s05`) |
| 7. Appending ledger rows as downloads finish | a different ledger, and a different output hash, on every run | `test_ledger_rows_follow_the_input_order` (mutant `s06`) |
| 8. No semaphore around the downloads | every source at once: throttled by the mirror, out of file descriptors | `test_concurrency_is_bounded` (mutant `s10`) |
| 9. No timeout on reads, or one timeout for the whole download | a silent server hangs the run; a large healthy download is killed | `test_a_silent_server_times_out` (mutant `s22`) |
| 10. Writing the ledger only when every source succeeded | finished sources are fetched again after any failure | `test_a_failure_keeps_the_rows_of_finished_sources` (mutant `s15`) |
| 11. Local time in `retrieved_at` | rows that fail the schema pattern; dates that depend on the machine's time zone | `test_ledger_rows_validate_against_the_schema` (mutant `s18`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Forward | `data.02` | `extract(manifest)` reads each source's raw parts with `read_raw` and numbers the documents `<source_id>:<k>` |
| Forward | `data.08` | verifies every ledger row against the schema and your license allowlist (`ethics.01`) before a shard may be used |
| Forward | `data.09` | the `CorpusBuild` workflow runs `{corpus} run --stage fetch` as a subprocess activity; exit 75 retries it, exit 65 fails it |
| Back | `lang.08` | the semaphore, task group, per-attempt timeout, and backoff, unchanged |

This module calls no other module's code, so nothing blocks `ol check data.01`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `fetch` | `huggingface_hub.hf_hub_download` | resume, ETag-based caching, symlinked snapshots per revision, parallel chunked downloads | `huggingface_hub/file_download.py` |
| `fetch` | `aria2`, `curl --continue-at -` | many connections per file, mirror lists, rate limits | aria2 manual, "Segmented download" |
| raw parts + ledger | datatrove `JsonlWriter` and readers; Dolma's source metadata | sharded writers, per-document provenance at web scale | `datatrove/pipeline/writers/jsonl.py` |
| the ledger | the Data Provenance Initiative's provenance cards | license and lineage per dataset, aggregated across collections | Longpre et al. 2023 |
