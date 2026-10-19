# Python asyncio: event loop, tasks, cancellation, bounded concurrency

A training corpus starts as files on other people's servers. A source can be hundreds of megabytes on a mirror that answers slowly, fails with `503 Service Unavailable` under load, or drops the connection halfway. Downloading one file after another wastes the time each request spends waiting on the network; starting every download at once gets the client's address throttled and fills the mirror's queue. What you need is a fixed number of downloads in flight, each retried with backoff when the failure is temporary, each with a deadline, and the whole run stopped cleanly when one source fails for good.

Python's answer to "many things waiting on the network at once, on one thread" is `asyncio`. In this section you build a small downloader from four functions: `http_get` (one HTTP/1.1 GET over asyncio streams), `is_retryable`, `fetch` (retries, exponential backoff, a timeout per attempt), and `download_all` (many fetches at once, at most `limit` in flight, cancelling the rest on a failure). The corpus fetcher of [Async fetch with resume, checksums, license capture](../../data-engineering/05-corpus-pipeline/01-async-fetch-resume-checksums.md) in *Data Engineering* is built from exactly these four pieces, plus resume and checksums.

This section builds on [Python and numpy](01-python-and-numpy.md) for running a Python of a known version with `uv`, and on [HTTP/1.1, JSON, and Server-Sent Events from the wire up](05-http-and-sse.md), whose request and response you write by hand here. The solution is standard library only, in [`course/solutions/primers/lang.08/downloader.py`](../../../../course/solutions/primers/lang.08/downloader.py), so `primers/lang.08/` needs no uv project: `uv run --no-project --python ">=3.11"` runs it. Python 3.11 is the floor because `asyncio.TaskGroup` and `asyncio.timeout` arrived in it. The exercise is not part of the system and has no separate contract; the signatures in "Designing the API" below are the contract.

> **Learning objectives**
> - Explain how coroutines and the event loop share one thread, and why one blocking call stalls every coroutine.
> - Read an HTTP/1.1 response over asyncio streams so that a short body is an error, never a smaller file.
> - Classify failures as retryable or not, and compute exponential backoff waits $b \cdot 2^{n-1}$.
> - Bound concurrency with one shared `asyncio.Semaphore` and run tasks in an `asyncio.TaskGroup` that cancels the rest on a failure.
> - Make cleanup run on every path out, including cancellation.

## Coroutines and the event loop

Everything in asyncio follows from five terms:

| Term | Meaning |
|---|---|
| **coroutine function** | a function defined with `async def`. Calling it runs none of its body; it returns a coroutine object |
| **coroutine** | a paused computation. It runs only when something drives it, and it can stop part way at an `await` and continue later |
| **`await x`** | "pause me until `x` has a result". `x` is a coroutine, a task, or a future. While this coroutine is paused, others run |
| **event loop** | the scheduler. It keeps a queue of coroutines that are ready to run, runs one until it reaches an `await` that must wait, then picks the next. It also watches sockets and timers and makes the matching coroutines ready again when their data arrives or their time comes |
| **`asyncio.run(main())`** | creates an event loop, runs the coroutine `main()` to completion, closes the loop |

There is one thread. Only one coroutine runs at any instant, and it runs until it reaches an `await` that has to wait. This is **cooperative** multitasking: a coroutine is never interrupted, so between two `await`s it sees no other coroutine's changes. It also means a coroutine that never awaits, for example one stuck in `time.sleep(5)` or in `requests.get(url)`, holds the whole loop for that long, and every other download stops. Inside a coroutine, waiting is spelled `await asyncio.sleep(5)` or `await reader.read(n)`. Note too that awaiting downloads one after another runs them one at a time; overlapping them takes tasks, which come shortly.

## Streams: talking HTTP/1.1 yourself

To wait on the network you need asyncio's socket API. `reader, writer = await asyncio.open_connection(host, port)` opens a TCP connection and returns a `StreamReader` and a `StreamWriter`. You send bytes with `writer.write(data)` followed by `await writer.drain()` (which waits until the operating system has accepted them), and close with `writer.close()` then `await writer.wait_closed()`.

Reading has three calls, and they differ in exactly the case that matters for downloads, when the peer stops early:

| Call | Returns | When the connection ends first |
|---|---|---|
| `await reader.readline()` | one line including its `\n` | the bytes so far, possibly `b""` |
| `await reader.read(n)` | **up to** `n` bytes, as soon as any arrive | the bytes so far, or `b""` at the end |
| `await reader.readexactly(n)` | exactly `n` bytes | raises `asyncio.IncompleteReadError` (its `.partial` holds what arrived) |

So `readexactly(n)` either returns `n` bytes or raises, while `read(n)` returns whatever arrived. A download that can be cut must use the first.

The HTTP exchange is the one from [HTTP/1.1, JSON, and Server-Sent Events from the wire up](05-http-and-sse.md): a request line, headers, an empty line; then a status line, headers, an empty line, and exactly `Content-Length` body bytes. Sending `Connection: close` asks the server to close the connection after the response, so one connection carries one request and the end is unambiguous. Header names are case-insensitive, so store them lower-cased.

## Tasks and task groups

`await` runs one thing at a time. To run several at once you wrap each coroutine in a **task**: `asyncio.create_task(coro)` schedules it on the loop immediately and returns a `Task`, which you can await later for its result. A task whose result nobody awaits is a **background task**: if it fails, nobody hears about it.

`asyncio.TaskGroup` is the structured way to run tasks:

```python
async with asyncio.TaskGroup() as tg:
    for u in urls:
        tg.create_task(one(u))
# here every task has finished
```

The `async with` block does not end until every task created in it has finished. If one task raises, the group **cancels** every other task, waits for them to finish cancelling, and then raises a `BaseExceptionGroup` holding the failures. So a task group never leaves work running behind your back. `asyncio.gather(*coros)` looks similar but is not: when one of its coroutines raises, `gather` passes that exception to you at once and the others **keep running**, unwatched.

## Cancellation

The task group's guarantee depends on what cancellation does inside a task. Cancelling a task (`task.cancel()`, or a task group doing it for you) makes the `await` the task is paused at raise `asyncio.CancelledError` inside it. The task can run cleanup (`finally:` blocks, `async with` exits) and then the error propagates out. Two rules follow. Put cleanup that must happen, such as closing a connection, in `finally:`; that is how a cancelled download closes its socket instead of leaking it. And do not swallow `CancelledError`: it derives from `BaseException`, not `Exception`, precisely so that `except Exception:` does not catch it by accident.

An `ExceptionGroup` wraps one or more exceptions. A caller of `download_all` wants to handle "the download failed with HTTP 404", not "a group containing an HTTPError", so the solution unwraps the group and re-raises its first exception with `raise first from None`.

## Bounded concurrency: a semaphore

A task group starts everything at once. To keep a fixed number in flight you need a counter of permits. An `asyncio.Semaphore(k)` holds `k` permits. `async with sem:` takes one before entering its block, waiting if none is free, and gives it back on the way out, even when the block raises or is cancelled. Wrapping each download's body in `async with sem:` therefore guarantees at most `k` downloads inside at once, and, because a waiting task gets a permit the moment one is returned, exactly `k` while at least `k` are waiting: never more, never fewer while work remains. Create **one** semaphore and share it, because a semaphore created inside each task bounds nothing.

`Semaphore(0)` never grants a permit, so `limit = 0` would hang forever. Reject `limit < 1` before starting.

## Timeouts, retries, and backoff

The last ingredients decide how long to wait and when to try again. `async with asyncio.timeout(seconds):` cancels the block if it has not finished in time and raises `TimeoutError` outside it. Put it **inside** the retry loop, around one attempt: a timeout around the whole loop lets one slow attempt eat every retry's time.

Not every failure deserves a retry. A failure is **retryable** when a later attempt may not see it, which gives this classification:

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

where the symbols are:

| Symbol | Meaning | Type |
|---|---|---|
| $b$ | the base delay, `base_delay` | seconds, float |
| $n$ | the attempt that just failed, $1 \le n < A$ | integer |
| $A$ | the number of attempts allowed, `attempts` | integer, at least 1 |
| $w_n$ | the wait before attempt $n + 1$ | seconds |

So with $b = 0.1$ the waits are 0.1, 0.2, 0.4 seconds, and the total wait over $A$ attempts is $b(2^{A-1} - 1)$. There is no wait after the last attempt: its failure is raised. Waits go through a `sleep` parameter that defaults to `asyncio.sleep`, so a test can pass a fake that only records the delays and never sleeps for real.

## Working it by hand

Three small traces make the rules concrete: one exchange on the wire, one retry timeline, and one bounded run.

### One exchange

`http_get("http://127.0.0.1:8765/hello.txt")` sends

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

The status line splits on white space into `HTTP/1.1`, `200`, `OK`. The headers become a dict with names lower-cased, among them `"content-length": "11"` and `"accept-ranges": "bytes"` (the `Server` and `Date` lines are added by Python's `http.server` to every response). The body is `readexactly(11)`, the 11 bytes `hello world`. With `headers={"Range": "bytes=6-"}` the server answers `206 Partial Content` with the 5 bytes `world`.

### A retry timeline

Now suppose the server fails twice with 503, then succeeds, with `attempts = 3` and `base_delay = 0.1`:

| Attempt | Response | `is_retryable` | Then |
|---|---|---|---|
| 1 | 503 | yes | wait $0.1 \cdot 2^0 = 0.1$ s |
| 2 | 503 | yes | wait $0.1 \cdot 2^1 = 0.2$ s |
| 3 | 200, 11 bytes | (no failure) | return the body |

The server saw 3 requests and the fake `sleep` recorded `[0.1, 0.2]`. Had the third answer also been 503, `fetch` would raise that `HTTPError(503)` with no third wait.

### A bounded run

Finally, twelve URLs with `limit = 3`, each answer taking 0.05 s:

| Time (s) | In flight | Event |
|---|---|---|
| 0.00 | 3 | tasks 1 to 3 take the 3 permits; tasks 4 to 12 wait in `async with sem` |
| 0.05 | 3 | 1 to 3 finish and return their permits; 4 to 6 take them |
| 0.10 | 3 | 7 to 9 |
| 0.15 | 3 | 10 to 12 |
| 0.20 | 0 | done |

The run takes about 4 rounds of 0.05 s, 0.2 s, against 0.6 s one at a time, and the server never sees more than 3 requests at once. A server that counts requests in flight sees exactly that maximum.

## Designing the API

`primers/lang.08/downloader.py` has these signatures:

```python
class HTTPError(Exception):            # given: .url, .status
@dataclass(frozen=True)
class Response:                        # given: status, headers (names lower-cased), body

async def http_get(url: str, headers: dict[str, str] | None = None) -> Response: ...
    # one GET over a fresh connection with Connection: close; status line, headers, exactly
    # Content-Length body bytes (read to the end when there is none); IncompleteReadError for
    # a short body, ValueError for a malformed status line; the connection closed on every path out
def is_retryable(exc: BaseException) -> bool: ...
    # the retry classification table above
async def fetch(url: str, *, attempts: int = 3, timeout: float = 5.0,
                base_delay: float = 0.1, sleep = asyncio.sleep) -> bytes: ...
    # the body of a 200; HTTPError for any other status; each attempt under its own timeout;
    # retryable failures wait base_delay * 2 ** (n - 1) through `sleep`; the last failure is raised
async def download_all(urls, *, limit: int = 4, attempts: int = 3, timeout: float = 5.0,
                       base_delay: float = 0.1, sleep = asyncio.sleep) -> dict[str, bytes]: ...
    # fetch every distinct url, at most `limit` in flight; {url: body} in first-given order;
    # on a failure cancel the rest and raise that failure itself; ValueError when limit < 1
```

Two rules frame the implementation. It uses the standard library only: no `aiohttp`, no `httpx`, no `requests`, because they hide the event loop this section is about. And it speaks HTTP only (`http://`); the corpus fetcher in *Data Engineering* adds `https://`.

## Building it step by step

### Reasoning from first principles

Each design decision follows from a principle above. Try to predict each function from this chain before reading its code.

1. **One thread, many waits.** Each download spends almost all its time waiting on a socket. A coroutine gives the loop back at every `await`, so one thread can hold many downloads open at once, provided nothing in them blocks. So every network call is an asyncio stream call, awaited.
2. **A short body must be an error, not a smaller file.** `Content-Length` says how many bytes the body has. `readexactly` is the only read that turns "the connection ended first" into an exception, so it is the body read whenever the length is known.
3. **The socket must close on every path out**, including cancellation, which arrives as `CancelledError` at whatever `await` the coroutine is paused at. Only `finally:` runs on all of them.
4. **What a status means is policy, not transport.** `http_get` returns a 404 as a `Response`; `fetch` decides that anything but 200 is an `HTTPError`. That keeps `http_get` reusable for a `Range` request that wants a 206.
5. **Retry only what can change, a bounded number of times, waiting longer each time.** The classification table becomes `is_retryable`; attempts are counted from 1 so the wait after attempt $n$ is $b \cdot 2^{n-1}$; the last failure is raised rather than waited on.
6. **The deadline belongs to one attempt.** A timeout around the loop would let one slow attempt use every retry's budget, so `asyncio.timeout` wraps a single `http_get`.
7. **Bound concurrency with one shared counter of permits.** A single `Semaphore(limit)`, shared by every task, admits exactly `limit` at once while work is waiting. `Semaphore(0)` would hang, so `limit < 1` is rejected first.
8. **One failure must stop the rest and be seen as itself.** A `TaskGroup` cancels the siblings and waits for them; the group's `ExceptionGroup` is unwrapped so the caller sees the `HTTPError` it can handle.

### `http_get`: the wire exchange

The function parses the URL and connects; everything after the connection is inside `try`/`finally`.

```python
async def http_get(url: str, headers: dict[str, str] | None = None) -> Response:
    parts = urlsplit(url)
    if parts.scheme != "http" or not parts.hostname:
        raise ValueError(f"only http:// URLs are supported, got {url!r}")
    host, port = parts.hostname, parts.port or 80
    target = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
    reader, writer = await asyncio.open_connection(host, port)
    try:
        # Connection: close -> one request per connection, and the end of the
        # response is unambiguous.
        lines = [f"GET {target} HTTP/1.1", f"Host: {host}:{port}", "Connection: close"]
        lines += [f"{k}: {v}" for k, v in (headers or {}).items()]  # e.g. Range
        writer.write(("\r\n".join(lines) + "\r\n\r\n").encode("latin-1"))
        await writer.drain()  # wait until the OS has taken the bytes

        status_line = await reader.readline()
        fields = status_line.decode("latin-1").split(None, 2)  # "HTTP/1.1", "200", "OK"
        if len(fields) < 2 or not fields[0].startswith("HTTP/") or not fields[1].isdigit():
            raise ValueError(f"malformed status line {status_line!r}")  # not retryable: a bug
        status = int(fields[1])

        got: dict[str, str] = {}
        while True:
            line = await reader.readline()
            if line in (b"\r\n", b"\n", b""):  # the empty line, or the peer closed early
                break
            name, _, value = line.decode("latin-1").partition(":")
            got[name.strip().lower()] = value.strip()  # header names are case-insensitive

        if "content-length" in got:
            body = await reader.readexactly(int(got["content-length"]))  # short -> IncompleteReadError
        else:
            body = await reader.read()  # no length: the body ends when the server closes
        return Response(status, got, body)
    finally:
        # Runs on return, on every exception, and on CancelledError.
        writer.close()
        try:
            await writer.wait_closed()
        except (ConnectionError, OSError):
            pass  # the peer may already have reset the connection; closing is best effort
```

Headers are encoded and decoded as Latin-1 because HTTP/1.1 header bytes are octets, not UTF-8; Latin-1 maps every byte to one character and back. `split(None, 2)` keeps a reason phrase with spaces in it (`Not Found`) as one field.

### `is_retryable`: the classification table as code

```python
def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, HTTPError):
        return exc.status >= 500 or exc.status == 429  # 404, 403, 400: the request is wrong
    return isinstance(
        exc, (asyncio.IncompleteReadError, ConnectionError, TimeoutError, OSError)
    )  # cut body, refused or reset, slow; ValueError and other bugs fall through to False
```

`ConnectionRefusedError` and `ConnectionResetError` are `ConnectionError`s, and `ConnectionError` and `TimeoutError` are both `OSError`s; naming them anyway keeps the table readable. `IncompleteReadError` is an `EOFError`, not an `OSError`, so it must be named.

### `fetch`: bounded retries with backoff

```python
async def fetch(url: str, *, attempts: int = 3, timeout: float = 5.0,
                base_delay: float = 0.1, sleep: Sleep = asyncio.sleep) -> bytes:
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    for n in range(1, attempts + 1):  # n counts attempts from 1
        try:
            async with asyncio.timeout(timeout):  # this attempt's deadline, not the loop's
                resp = await http_get(url)
            if resp.status != 200:
                raise HTTPError(url, resp.status)  # status policy lives here, not in http_get
            return resp.body
        except Exception as e:  # CancelledError is a BaseException: never caught here
            if n == attempts or not is_retryable(e):
                raise  # the last failure, or one that would only repeat
        await sleep(base_delay * 2 ** (n - 1))  # 0.1, 0.2, 0.4, ...; never after the last
    raise AssertionError("unreachable")
```

`except Exception` is safe precisely because cancellation is a `BaseException`: a cancelled `fetch` stops at once instead of being retried. The `sleep` happens outside the `try`, so it is reached only after a retryable, non-final failure. The retry timeline you worked by hand falls out directly: two 503s, waits `[0.1, 0.2]`, then the body.

### `download_all`: one semaphore, one task group

```python
async def download_all(urls: Iterable[str], *, limit: int = 4, attempts: int = 3, timeout: float = 5.0,
                       base_delay: float = 0.1, sleep: Sleep = asyncio.sleep) -> dict[str, bytes]:
    if limit < 1:
        raise ValueError("limit must be at least 1")  # Semaphore(0) would never open
    order = list(dict.fromkeys(urls))  # distinct, in first-given order
    sem = asyncio.Semaphore(limit)     # ONE semaphore, shared by every task
    got: dict[str, bytes] = {}

    async def one(u: str) -> None:
        async with sem:  # at most `limit` inside; the permit is returned even on cancel
            got[u] = await fetch(
                u, attempts=attempts, timeout=timeout, base_delay=base_delay, sleep=sleep
            )

    try:
        async with asyncio.TaskGroup() as tg:  # exits only when every task has finished
            for u in order:
                tg.create_task(one(u))
    except BaseExceptionGroup as eg:  # one failed: the rest are already cancelled and awaited
        first = eg.exceptions[0]
        while isinstance(first, BaseExceptionGroup):  # unwrap nested groups too
            first = first.exceptions[0]
        raise first from None  # the caller sees HTTPError(404), not a group
    return {u: got[u] for u in order}  # rebuild in input order: tasks finish in any order
```

`dict.fromkeys` dedupes while keeping order, so a repeated URL is fetched once. Writing into a shared dict from many tasks is safe because there is one thread and the assignment has no `await` in the middle. All tasks are created at once, but the semaphore admits them `limit` at a time, which is the bounded run you traced by hand.

The full solution is [`downloader.py`](../../../../course/solutions/primers/lang.08/downloader.py).

### Why this design

With $N$ URLs, each taking time $t$, the run takes about $\lceil N / \text{limit} \rceil \cdot t$ instead of $N t$, using one thread and at most `limit` sockets. Memory is the bodies plus $N$ small task objects.

Threads (`concurrent.futures` with blocking sockets) would also overlap the waits, but each costs an OS thread and a stack, cancellation is cooperative only by convention, and a stuck `recv` cannot be interrupted by a timeout from outside. asyncio's cancellation reaches any `await`.

A worker pool (`limit` workers pulling from an `asyncio.Queue`) bounds concurrency too, and is the better choice when the inputs arrive as a stream. With a known list, a semaphore plus one task per URL is shorter and gives each URL its own result and failure.

`gather(..., return_exceptions=True)` would collect every failure instead of stopping at the first. That is a different policy (best effort), and it still does not cancel anything. A corpus build that must be complete wants the first hard failure to stop everything.

The design accepts three tradeoffs: a new connection per request (no keep-alive), no jitter in the backoff (see the production equivalents below), and HTTP only.

## What can go wrong

A correct solution is checked against servers on `127.0.0.1`: one that fails, cuts, and delays on request, and a small asyncio server that counts requests in flight. These are the cases it must handle:

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| Reads status, headers, and body | unit | the hand-worked exchange: status 200, lower-cased headers, 11 body bytes | the wire format is the one of the HTTP section |
| Sends extra headers | unit | a `Range` header reaches the server and the 206 comes back | the corpus fetcher resumes cut downloads with Range |
| An error status is a response | boundary | a 404 is a `Response`, not an exception | deciding what a status means is `fetch`'s job |
| Never returns a short body | boundary | a body cut after 4 bytes raises `IncompleteReadError` | a short file must never pass as a whole one |
| Classifies failures | unit | the classification table, row by row | the retry policy of every later download |
| Retries 5xx with exponential backoff | unit | the hand-worked timeline: 3 requests, waits `[0.1, 0.2]` | a struggling mirror sees fewer requests |
| Does not retry a 404 | boundary | 1 request, no wait, `HTTPError.status == 404` | a wrong URL fails fast |
| Gives up after its attempts | boundary | exactly `attempts` requests, no wait after the last, the last error raised | retries are bounded |
| Retries a cut body | fault | a connection cut mid-body is retried and the full body returned | the most common failure on long downloads |
| Times out each attempt | fault | two 0.05 s attempts against a 0.6 s server end in well under 0.5 s | a silent server cannot hang the run |
| Keeps `limit` requests in flight | unit | 12 downloads with limit 3: the server sees exactly 3 at once | the hand-worked bounded run |
| Keeps order, fetches each URL once | unit | results in first-given order; a repeated URL fetched once | callers can zip results with inputs |
| A failure cancels the other downloads | fault | a 404 cancels three held downloads, closes their connections, leaves no task running, and raises `HTTPError` itself | no orphan work after a failure |
| `limit` must be positive | boundary | `limit=0` raises `ValueError` instead of hanging | a semaphore of 0 never opens |
| Standard library only | unit | no import outside the standard library | the section is about asyncio itself |

The common mistakes each have a recognizable symptom and a specific line in the solution that prevents them:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| `for u in urls: await fetch(u)` | correct results, one download at a time, 4 times slower here | one task per URL in a `TaskGroup` (step 8) |
| A semaphore created inside each task | nothing is bounded: all 12 requests at once | one `Semaphore(limit)` created in `download_all`, shared by every `one` |
| `await reader.read(n)` for the body | a cut connection returns a short body as if it were the whole file | `readexactly(content-length)` (step 2) |
| Retrying every error | a 404 is requested `attempts` times with waits in between | `is_retryable` returns False for 4xx except 429 |
| One timeout around all attempts, or none | a silent server hangs the run, or one slow attempt uses up every retry's time | `asyncio.timeout` wraps one `http_get`, inside the loop |
| `asyncio.gather` instead of a `TaskGroup` | the failure is raised but the other downloads keep running, holding connections open | `TaskGroup`, which cancels and awaits the siblings |
| Sleeping after the last attempt | every failed download ends with one wasted wait | `n == attempts` re-raises before the `sleep` |
| `base * 2 ** n` instead of `base * 2 ** (n - 1)` | every wait twice as long as specified | `n` counts from 1 and the exponent is `n - 1` |
| Letting the `ExceptionGroup` escape | callers must write `except*` to see an HTTP 404 | the group is unwrapped and `raise first from None` |
| Accepting `limit = 0` | `Semaphore(0)` never opens; the run hangs forever | `ValueError` when `limit < 1`, before any task starts |

> **Warning:** A blocking call inside a coroutine (`time.sleep`, `requests.get`, a plain `socket`) is the pitfall behind most slow asyncio code, and it is hard to see from outside: the requests still overlap on the server, they only start late. Find it by reading: every call in an `async def` that waits must be awaited. The solution has none: every wait is `open_connection`, `drain`, `readline`, `readexactly`, `read`, `wait_closed`, or the injected `sleep`.

## Where this leads

This section used the HTTP/1.1 request and response from [HTTP/1.1, JSON, and Server-Sent Events from the wire up](05-http-and-sse.md), parsed here by hand, and `uv run` with a Python of a known version from [Python and numpy](01-python-and-numpy.md).

Next, in [Async fetch with resume, checksums, license capture](../../data-engineering/05-corpus-pipeline/01-async-fetch-resume-checksums.md) in *Data Engineering*, `corpus.fetch.fetch` downloads every source of the corpus config with a semaphore of `concurrency`, a task group, a timeout per read, and backoff through an injected `sleep`; it adds resume with `Range`, checksums, quarantine, and ledger rows. And [Async Rust and tokio](09-async-rust-and-tokio.md) carries the same ideas into Rust: futures, tokio tasks, cancellation, bounded channels.

Each piece has a production equivalent worth knowing:

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `http_get` | `aiohttp`, `httpx` | connection pooling and keep-alive, chunked transfer coding, TLS, HTTP/2 | `httpx/_transports/default.py` |
| `fetch` | `tenacity`, `backoff` | jittered backoff (randomized waits so many clients do not retry in lockstep), retry budgets | tenacity `wait_random_exponential` |
| `download_all` | `asyncio.Semaphore` plus `TaskGroup`, Trio nurseries, `anyio` | structured concurrency as a language-level rule; cancel scopes with deadlines | Trio docs, "Tasks let you do multiple things at once" |
| the timeout per attempt | `asyncio.timeout`, Go's `context.WithTimeout` | deadlines that propagate through every call below them | the `context` discussion in [Go](06-go.md) |

### Further reading

- Python docs, "asyncio: Coroutines and Tasks", "Streams", and "Synchronization Primitives" (docs.python.org, free).
- Nathaniel J. Smith, "Notes on structured concurrency, or: Go statement considered harmful" (vorpus.org, free).
- Yury Selivanov, PEP 654, "Exception Groups and except*" (peps.python.org, free).

## Key terms

- **coroutine**: a paused computation returned by calling an `async def` function; it runs only when driven and yields at `await`.
- **event loop**: the single-threaded scheduler that runs ready coroutines and wakes them when their sockets or timers are ready.
- **cooperative multitasking**: scheduling in which a coroutine is never interrupted and gives up control only at an `await`.
- **task**: a coroutine scheduled on the loop with `asyncio.create_task`, whose result can be awaited later.
- **TaskGroup**: a structured block that waits for all its tasks and cancels the rest when one fails.
- **cancellation**: raising `CancelledError` at the `await` a task is paused at, so its cleanup runs and the error propagates.
- **semaphore**: a counter of permits that bounds how many coroutines are inside a section at once.
- **exponential backoff**: waiting $b \cdot 2^{n-1}$ after failed attempt $n$ before trying again.
- **retryable failure**: a failure a later attempt may not see, such as a 5xx, 429, cut connection, or timeout.

## Summary

- An `async def` function returns a coroutine; nothing runs until the event loop drives it, and it gives the loop back only at an `await`. One blocking call stalls every coroutine at once, and awaiting downloads one after another runs them one at a time.
- `readexactly(n)` either returns `n` bytes or raises; `read(n)` returns whatever arrived. A download that can be cut must use the first, inside a `try`/`finally` that closes the socket on every path, including cancellation.
- Retry only what a later attempt may not see (5xx, 429, a cut or refused connection, a timeout), wait `base * 2^(n-1)` between attempts, never after the last, and give every attempt its own deadline.
- One shared `asyncio.Semaphore(limit)` keeps `limit` downloads in flight, never more, never fewer while work remains.
- `asyncio.TaskGroup` cancels the remaining tasks when one fails and waits for them; `asyncio.gather` does not, and leaves them running with nobody waiting.

## Review questions

1. Why does `fetch` catch `Exception` and not `BaseException`?
   <details><summary>Answer</summary>
   `asyncio.CancelledError` (and `KeyboardInterrupt`) derive from `BaseException`, not `Exception`, so `except Exception` never sees a cancellation: a cancelled `fetch` stops at the `await` it was paused at, runs `http_get`'s `finally:`, and propagates. With `except BaseException` the code would at best re-raise it after a pointless classification, and any later change that retried "everything" would turn a cancel into another attempt, so the task group could never finish cancelling.
   </details>
2. Why is the semaphore created in `download_all` and not inside `one`?
   <details><summary>Answer</summary>
   A semaphore bounds only the tasks that share it. One per task gives each task its own `limit` permits, so nothing is bounded and all twelve requests start at once.
   </details>
3. In the hand-worked retry timeline, why are the recorded waits `[0.1, 0.2]` and not `[0.1, 0.2, 0.4]` or `[0.2, 0.4]`?
   <details><summary>Answer</summary>
   The wait after attempt $n$ is `base_delay * 2 ** (n - 1)`, with $n$ starting at 1, so attempt 1 waits 0.1 and attempt 2 waits 0.2. Attempt 3 succeeds, and even if it had failed there is no wait after the last attempt: `n == attempts` raises before the `sleep`. `[0.2, 0.4]` is the off-by-one `2 ** n`.
   </details>
4. What guarantees that a download cancelled by the task group closes its socket?
   <details><summary>Answer</summary>
   Cancellation raises `CancelledError` at the `await` inside `http_get` (a `readline`, `readexactly`, or `drain`). That propagates through the `finally:` block, which closes the writer. The `async with sem` exit returns the permit on the same path.
   </details>
5. Why does `http_get` use `readexactly` for the body but `readline` for headers?
   <details><summary>Answer</summary>
   For the body the length is known and a shortfall must be an error, which only `readexactly` gives. Header lines have no known length; they end at `\r\n`. A header read that returns `b""` means the peer closed early, and the loop stops; the body read then fails if a length was promised.
   </details>
6. Twelve URLs, `limit = 3`, each taking 0.05 s: how long does `download_all` take, and how long would a plain `for` loop of awaits take?
   <details><summary>Answer</summary>
   About $\lceil 12 / 3 \rceil \cdot 0.05 = 0.2$ s, with exactly 3 requests in flight at every moment until the end. One at a time it takes $12 \cdot 0.05 = 0.6$ s.
   </details>
