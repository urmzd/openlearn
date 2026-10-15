<!-- ol:module lang.08 -->
# Python asyncio: event loop, tasks, cancellation, bounded concurrency

## Overview

| | |
|---|---|
| **Module** | `lang.08` · practice · Python · Pass 3 · 3 to 4 h |
| **You build** | `primers/lang.08/downloader.py`: `http_get` (one HTTP/1.1 GET over asyncio streams), `is_retryable`, `fetch` (retries, exponential backoff, a timeout per attempt), `download_all` (many fetches at once, at most `limit` in flight, cancelling the rest on a failure). Standard library only |
| **Contract** | none: a primer exercise, not part of the system. The signatures are in the stubs `ol start lang.08` writes and in section 4 |
| **Tests** | `course/tests/lang.08/check` runs `test_lang08_downloader.py` against your file, with the course's `flakyhttp` fixture server standing in for the network (what each test checks: section 4) |
| **Needs** | reading: `lang.01` (a uv-run Python), `lang.05` (the HTTP/1.1 request and response you write by hand here) |
| **Used by** | no code call site (a primer): `data.01` applies it next, where the corpus fetcher downloads every source concurrently with these exact pieces plus resume and checksums |
| **Milestone** | `MS-P3` |
| **Optional depth** | Python docs, "asyncio: Coroutines and Tasks", "Streams", "Synchronization Primitives" (docs.python.org, free); Nathaniel J. Smith, "Notes on structured concurrency, or: Go statement considered harmful" (vorpus.org, free); Yury Selivanov, PEP 654 "Exception Groups and except*" (peps.python.org, free) |

## Key Takeaways

- An `async def` function returns a **coroutine**; nothing runs until the **event loop** drives it, and it gives the loop back only at an `await`. One blocking call stalls every coroutine at once (section 2.1), and awaiting downloads one after another runs them one at a time (`test_download_all_keeps_limit_requests_in_flight`).
- `readexactly(n)` either returns `n` bytes or raises; `read(n)` returns whatever arrived. A download that can be cut must use the first (`test_http_get_never_returns_a_short_body`).
- Retry only what a later attempt may not see (5xx, 429, a cut or refused connection, a timeout), wait `base * 2^(n-1)` between attempts, and give every attempt its own deadline (`test_is_retryable_classifies_failures`, `test_fetch_retries_5xx_with_exponential_backoff`, `test_fetch_times_out_each_attempt`).
- An `asyncio.Semaphore(limit)` bounds how many coroutines are inside a section at once; `limit` downloads in flight, never more, never fewer while work remains (`test_download_all_keeps_limit_requests_in_flight`).
- `asyncio.TaskGroup` cancels the remaining tasks when one fails and waits for them; `asyncio.gather` does not, and leaves them running with nobody waiting (`test_a_failure_cancels_the_other_downloads`).

## How to work this chapter

```bash
ol start lang.08              # writes primers/lang.08/downloader.py: signatures, docstrings, stub bodies
ol tests lang.08              # read the test catalog first
ol check lang.08              # exit code is the verdict
```

The exercise is standard library only, so `primers/lang.08/` needs no uv project: the check runs `uv run --no-project --python ">=3.11" --with pytest` and puts your directory on `PYTHONPATH`. Python 3.11 is the floor because `asyncio.TaskGroup` and `asyncio.timeout` arrived in it.

---

## 1. Why now

Your corpus starts as files on other people's servers. In `data.01` the pipeline downloads every source in the corpus config, and a source can be hundreds of megabytes on a mirror that answers slowly, fails with `503 Service Unavailable` under load, or drops the connection halfway. Downloading one file after another wastes the time each request spends waiting on the network; starting every download at once gets your address throttled and fills the mirror's queue. You want a fixed number of downloads in flight, each retried with backoff when the failure is temporary, each with a deadline, and the whole run stopped cleanly when one source fails for good. Python's answer to "many things waiting on the network at once, on one thread" is `asyncio`, and the corpus fetcher is built from exactly the four functions of this primer.

## 2. Principles

### 2.1 Coroutines and the event loop

| Term | Meaning |
|---|---|
| **coroutine function** | a function defined with `async def`. Calling it runs none of its body; it returns a coroutine object |
| **coroutine** | a paused computation. It runs only when something drives it, and it can stop part way at an `await` and continue later |
| **`await x`** | "pause me until `x` has a result". `x` is a coroutine, a task, or a future. While this coroutine is paused, others run |
| **event loop** | the scheduler. It keeps a queue of coroutines that are ready to run, runs one until it reaches an `await` that must wait, then picks the next. It also watches sockets and timers and makes the matching coroutines ready again when their data arrives or their time comes |
| **`asyncio.run(main())`** | creates an event loop, runs the coroutine `main()` to completion, closes the loop |

There is one thread. Only one coroutine runs at any instant, and it runs until it reaches an `await` that has to wait. This is **cooperative** multitasking: a coroutine is never interrupted, so between two `await`s it sees no other coroutine's changes. It also means a coroutine that never awaits, for example one stuck in `time.sleep(5)` or in `requests.get(url)`, holds the whole loop for that long and every other download stops. Inside a coroutine, waiting is spelled `await asyncio.sleep(5)` or `await reader.read(n)`.

### 2.2 Streams: talking HTTP/1.1 yourself

`reader, writer = await asyncio.open_connection(host, port)` opens a TCP connection and returns a `StreamReader` and a `StreamWriter`. You send bytes with `writer.write(data)` followed by `await writer.drain()` (which waits until the operating system has accepted them), and close with `writer.close()` then `await writer.wait_closed()`. Reading has three calls that differ in what happens when the peer stops early:

| Call | Returns | When the connection ends first |
|---|---|---|
| `await reader.readline()` | one line including its `\n` | the bytes so far, possibly `b""` |
| `await reader.read(n)` | **up to** `n` bytes, as soon as any arrive | the bytes so far, or `b""` at the end |
| `await reader.readexactly(n)` | exactly `n` bytes | raises `asyncio.IncompleteReadError` (its `.partial` holds what arrived) |

The HTTP exchange is the one from `lang.05`: a request line, headers, an empty line; a status line, headers, an empty line, then exactly `Content-Length` body bytes. Sending `Connection: close` asks the server to close the connection after the response, so one connection carries one request and the end is unambiguous. Header names are case-insensitive, so store them lower-cased.

### 2.3 Tasks and task groups

`await` runs one thing at a time. To run several at once you wrap each coroutine in a **task**: `asyncio.create_task(coro)` schedules it on the loop immediately and returns a `Task`, which you can await later for its result. A task whose result nobody awaits is a **background task**: if it fails, nobody hears about it.

`asyncio.TaskGroup` is the structured way to run tasks:

```python
async with asyncio.TaskGroup() as tg:
    for u in urls:
        tg.create_task(one(u))
# here every task has finished
```

The `async with` block does not end until every task created in it has finished. If one task raises, the group **cancels** every other task, waits for them to finish cancelling, and then raises a `BaseExceptionGroup` holding the failures. So a task group never leaves work running behind your back. `asyncio.gather(*coros)` looks similar but is not: when one of its coroutines raises, `gather` passes that exception to you at once and the others **keep running**, unwatched.

### 2.4 Cancellation

Cancelling a task (`task.cancel()`, or a task group doing it for you) makes the `await` the task is paused at raise `asyncio.CancelledError` inside it. The task can run cleanup (`finally:` blocks, `async with` exits) and then the error propagates out. Two rules follow:

- Put cleanup that must happen, such as closing a connection, in `finally:`. That is how a cancelled download closes its socket instead of leaking it.
- Do not swallow `CancelledError`. It derives from `BaseException`, not `Exception`, precisely so that `except Exception:` does not catch it by accident.

An `ExceptionGroup` wraps one or more exceptions. A caller of `download_all` wants to handle "the download failed with HTTP 404", not "a group containing an HTTPError", so this primer unwraps the group and re-raises its first exception with `raise first from None`.

### 2.5 Bounded concurrency: a semaphore

An `asyncio.Semaphore(k)` holds `k` permits. `async with sem:` takes one before entering its block, waiting if none is free, and gives it back on the way out, even when the block raises or is cancelled. Wrapping each download's body in `async with sem:` therefore guarantees at most `k` downloads inside at once, and, because a waiting task gets a permit the moment one is returned, exactly `k` while at least `k` are waiting. Create **one** semaphore and share it: a semaphore created inside each task bounds nothing.

`Semaphore(0)` never grants a permit, so `limit = 0` would hang forever. Reject `limit < 1` before starting.

### 2.6 Timeouts, retries, and backoff

`async with asyncio.timeout(seconds):` cancels the block if it has not finished in time and raises `TimeoutError` outside it. Put it **inside** the retry loop, around one attempt: a timeout around the whole loop lets one slow attempt eat every retry's time.

Not every failure deserves a retry. A failure is **retryable** when a later attempt may not see it:

| Failure | Retryable | Why |
|---|---|---|
| HTTP 500 to 599 | yes | the server is overloaded or broken right now |
| HTTP 429 Too Many Requests | yes | the server asks you to come back later |
| HTTP 400 to 499 except 429 (404, 403, 400) | no | the request itself is wrong; it will be wrong next time |
| `asyncio.IncompleteReadError` (body cut short) | yes | the connection dropped |
| `ConnectionRefusedError`, `ConnectionResetError`, other `OSError` | yes | the network or the server restarted |
| `TimeoutError` | yes | the server was slow this time |
| `ValueError` (a malformed status line), any other bug | no | retrying a bug repeats it |

Between attempts, wait longer each time: **exponential backoff**. After attempt $n$ (counting from 1) wait

$$w_n = b \cdot 2^{\,n-1}$$

| Symbol | Meaning | Type |
|---|---|---|
| $b$ | the base delay, `base_delay` | seconds, float |
| $n$ | the attempt that just failed, $1 \le n < A$ | integer |
| $A$ | the number of attempts allowed, `attempts` | integer, at least 1 |
| $w_n$ | the wait before attempt $n + 1$ | seconds |

So with $b = 0.1$ the waits are 0.1, 0.2, 0.4 seconds, and the total wait over $A$ attempts is $b(2^{A-1} - 1)$. There is no wait after the last attempt: its failure is raised. Waits go through a `sleep` parameter that defaults to `asyncio.sleep`, so the tests pass a fake that only records the delays and the suite never sleeps for real.

## 3. Worked example by hand

**One exchange.** `http_get("http://127.0.0.1:8765/hello.txt")` sends

```text
GET /hello.txt HTTP/1.1\r\n
Host: 127.0.0.1:8765\r\n
Connection: close\r\n
\r\n
```

and the fixture server answers

```text
HTTP/1.1 200 OK\r\n
Server: BaseHTTP/0.6 Python/3.12.12\r\n
Date: Fri, 09 Oct 2026 12:00:00 GMT\r\n
Content-Length: 11\r\n
X-Content-Sha256: b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9\r\n
Accept-Ranges: bytes\r\n
\r\n
hello world
```

The status line splits on white space into `HTTP/1.1`, `200`, `OK`; the headers become a dict with names lower-cased, among them `"content-length": "11"` and `"accept-ranges": "bytes"` (the `Server` and `Date` lines are added by Python's `http.server` to every response); the body is `readexactly(11)`, the 11 bytes `hello world`. With `headers={"Range": "bytes=6-"}` the server answers `206 Partial Content` with the 5 bytes `world`.

**A retry timeline.** The server fails twice with 503, then succeeds; `attempts = 3`, `base_delay = 0.1`:

| Attempt | Response | `is_retryable` | Then |
|---|---|---|---|
| 1 | 503 | yes | wait $0.1 \cdot 2^0 = 0.1$ s |
| 2 | 503 | yes | wait $0.1 \cdot 2^1 = 0.2$ s |
| 3 | 200, 11 bytes | (no failure) | return the body |

The server saw 3 requests and the fake `sleep` recorded `[0.1, 0.2]`. Had the third answer also been 503, `fetch` would raise that `HTTPError(503)` with no third wait.

**A bounded run.** Twelve URLs, `limit = 3`, each answer taking 0.05 s:

| Time (s) | In flight | Event |
|---|---|---|
| 0.00 | 3 | tasks 1 to 3 take the 3 permits; tasks 4 to 12 wait in `async with sem` |
| 0.05 | 3 | 1 to 3 finish and return their permits; 4 to 6 take them |
| 0.10 | 3 | 7 to 9 |
| 0.15 | 3 | 10 to 12 |
| 0.20 | 0 | done |

The run takes about 4 rounds of 0.05 s, 0.2 s, against 0.6 s one at a time; the server never sees more than 3 requests at once. The test server counts exactly that maximum.

## 4. The artifact and its check

`ol start lang.08` writes `primers/lang.08/downloader.py` with every body replaced by `raise NotImplementedError("lang.08")`. You replace the bodies, never the signatures:

```python
class HTTPError(Exception):            # given: .url, .status
@dataclass(frozen=True)
class Response:                        # given: status, headers (names lower-cased), body

async def http_get(url: str, headers: dict[str, str] | None = None) -> Response: ...
    # one GET over a fresh connection with Connection: close; status line, headers, exactly
    # Content-Length body bytes (read to the end when there is none); IncompleteReadError for
    # a short body, ValueError for a malformed status line; the connection closed on every path out
def is_retryable(exc: BaseException) -> bool: ...
    # the table of section 2.6
async def fetch(url: str, *, attempts: int = 3, timeout: float = 5.0,
                base_delay: float = 0.1, sleep = asyncio.sleep) -> bytes: ...
    # the body of a 200; HTTPError for any other status; each attempt under its own timeout;
    # retryable failures wait base_delay * 2 ** (n - 1) through `sleep`; the last failure is raised
async def download_all(urls, *, limit: int = 4, attempts: int = 3, timeout: float = 5.0,
                       base_delay: float = 0.1, sleep = asyncio.sleep) -> dict[str, bytes]: ...
    # fetch every distinct url, at most `limit` in flight; {url: body} in first-given order;
    # on a failure cancel the rest and raise that failure itself; ValueError when limit < 1
```

Rules: standard library only (no `aiohttp`, no `httpx`, no `requests`: they hide the event loop this primer is about), and HTTP only (`http://`; `data.01` adds `https://`).

`ol check lang.08` runs `course/tests/lang.08/check` in your repo. It refuses early if the file is missing, then runs the annotated tests with `primers/lang.08` and the course testkit on `PYTHONPATH`. The tests start their own servers on `127.0.0.1`: the course fixture `flakyhttp` (it fails, cuts, and delays on request) and a small asyncio server that counts requests in flight.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_http_get_reads_status_headers_and_body` | unit | the section 3 exchange: status 200, lower-cased headers, 11 body bytes | you and the test agree on the wire format |
| `test_http_get_sends_extra_headers` | unit | a `Range` header reaches the server and the 206 comes back | `data.01` resumes cut downloads with Range |
| `test_http_get_returns_an_error_status_as_a_response` | boundary | a 404 is a `Response`, not an exception | deciding what a status means is `fetch`'s job |
| `test_http_get_never_returns_a_short_body` | boundary | a body cut after 4 bytes raises `IncompleteReadError` | a short file must never pass as a whole one |
| `test_is_retryable_classifies_failures` | unit | the section 2.6 table, row by row | the retry policy of every later download |
| `test_fetch_retries_5xx_with_exponential_backoff` | unit | the section 3 timeline: 3 requests, waits `[0.1, 0.2]` | a struggling mirror sees fewer requests |
| `test_fetch_does_not_retry_a_404` | boundary | 1 request, no wait, `HTTPError.status == 404` | a wrong URL fails fast |
| `test_fetch_gives_up_after_its_attempts` | boundary | exactly `attempts` requests, no wait after the last, the last error raised | retries are bounded |
| `test_fetch_retries_a_cut_body` | fault | a connection cut mid-body is retried and the full body returned | the most common failure on long downloads |
| `test_fetch_times_out_each_attempt` | fault | two 0.05 s attempts against a 0.6 s server end in well under 0.5 s | a silent server cannot hang the run |
| `test_download_all_keeps_limit_requests_in_flight` | unit | 12 downloads with limit 3: the server sees exactly 3 at once | the section 3 bounded run |
| `test_download_all_keeps_order_and_fetches_each_url_once` | unit | results in first-given order; a repeated URL fetched once | callers can zip results with inputs |
| `test_a_failure_cancels_the_other_downloads` | fault | a 404 cancels three held downloads, closes their connections, leaves no task running, and raises `HTTPError` itself | no orphan work after a failure |
| `test_limit_must_be_positive` | boundary | `limit=0` raises `ValueError` instead of hanging | a semaphore of 0 never opens |
| `test_module_imports_only_the_standard_library` | unit | no import outside the standard library | the primer is about asyncio itself |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. `for u in urls: await fetch(u)` | correct results, one download at a time, 4 times slower here | `test_download_all_keeps_limit_requests_in_flight` |
| 2. A semaphore created inside each task | nothing is bounded: all 12 requests at once | `test_download_all_keeps_limit_requests_in_flight` |
| 3. `await reader.read(n)` for the body | a cut connection returns a short body as if it were the whole file | `test_http_get_never_returns_a_short_body` |
| 4. Retrying every error | a 404 is requested `attempts` times with waits in between | `test_fetch_does_not_retry_a_404` |
| 5. One timeout around all attempts, or none | a silent server hangs the run, or one slow attempt uses up every retry's time | `test_fetch_times_out_each_attempt` |
| 6. `asyncio.gather` instead of a `TaskGroup` | the failure is raised but the other downloads keep running, holding connections open | `test_a_failure_cancels_the_other_downloads` |
| 7. Sleeping after the last attempt | every failed download ends with one wasted wait | `test_fetch_gives_up_after_its_attempts` |
| 8. `base * 2 ** n` instead of `base * 2 ** (n - 1)` | every wait twice as long as specified | `test_fetch_retries_5xx_with_exponential_backoff` |
| 9. Letting the `ExceptionGroup` escape | callers must write `except*` to see an HTTP 404 | `test_a_failure_cancels_the_other_downloads` |
| 10. Accepting `limit = 0` | `Semaphore(0)` never opens; the run hangs forever | `test_limit_must_be_positive` |

A blocking call inside a coroutine (`time.sleep`, `requests.get`, a plain `socket`) is the pitfall behind most slow asyncio code, and no test here can see it reliably: the requests still overlap on the server, they only start late. Find it by reading: every call in an `async def` that waits must be awaited.

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `lang.05` | the HTTP/1.1 request and response you parse here by hand |
| Back | `lang.01` | `uv run` and a Python of a known version |
| Forward | `data.01` | `corpus.fetch.fetch` downloads every source of the corpus config with a semaphore of `concurrency`, a task group, a timeout per read, and backoff through an injected `sleep`; it adds resume with `Range`, checksums, quarantine, and ledger rows |
| Forward | `lang.09` | the same ideas in Rust: futures, tokio tasks, cancellation, bounded channels |

A primer has no code call site, so no module's `ol check` blocks on it. MS-P3 requires a fresh pass of `lang.08`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `http_get` | `aiohttp`, `httpx` | connection pooling and keep-alive, chunked transfer coding, TLS, HTTP/2 | `httpx/_transports/default.py` |
| `fetch` | `tenacity`, `backoff` | jittered backoff (randomized waits so many clients do not retry in lockstep), retry budgets | tenacity `wait_random_exponential` |
| `download_all` | `asyncio.Semaphore` plus `TaskGroup`, Trio nurseries, `anyio` | structured concurrency as a language-level rule; cancel scopes with deadlines | Trio docs, "Tasks let you do multiple things at once" |
| the timeout per attempt | `asyncio.timeout`, Go's `context.WithTimeout` | deadlines that propagate through every call below them | `lang.06` section 2.5 |
