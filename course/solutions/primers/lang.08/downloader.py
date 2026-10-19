"""lang.08 bounded concurrent downloader (a primer exercise, not part of the system).

Four pieces, each one asyncio idea:

    http_get       one HTTP/1.1 GET over asyncio streams (coroutines, await)
    fetch          retries with exponential backoff and a per-attempt timeout
    download_all   many fetches at once, at most `limit` in flight (tasks,
                   TaskGroup, Semaphore), cancelling the rest on a failure

Standard library only. The corpus fetcher (data.01) is built from the same
pieces, plus resume and checksums.

Chapter: content/courses/programming-languages/02-language-and-tool-primers/08-python-asyncio.md.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from urllib.parse import urlsplit

Sleep = Callable[[float], Awaitable[None]]


class HTTPError(Exception):
    """A response whose status is not 200 (or 206 for a Range request)."""

    def __init__(self, url: str, status: int) -> None:
        super().__init__(f"GET {url}: HTTP {status}")
        self.url = url
        self.status = status


@dataclass(frozen=True)
class Response:
    status: int
    headers: dict[str, str]  # names lower-cased
    body: bytes


async def http_get(url: str, headers: dict[str, str] | None = None) -> Response:
    """One GET over a fresh connection, with `Connection: close`.

    Reads the status line, the headers, and exactly Content-Length body bytes.
    A body cut short raises asyncio.IncompleteReadError; a malformed status
    line raises ValueError. The connection is closed on every path out,
    including cancellation.
    """
    parts = urlsplit(url)
    if parts.scheme != "http" or not parts.hostname:
        raise ValueError(f"only http:// URLs are supported, got {url!r}")
    host, port = parts.hostname, parts.port or 80
    target = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
    reader, writer = await asyncio.open_connection(host, port)
    try:
        lines = [f"GET {target} HTTP/1.1", f"Host: {host}:{port}", "Connection: close"]
        lines += [f"{k}: {v}" for k, v in (headers or {}).items()]
        writer.write(("\r\n".join(lines) + "\r\n\r\n").encode("latin-1"))
        await writer.drain()

        status_line = await reader.readline()
        fields = status_line.decode("latin-1").split(None, 2)
        if len(fields) < 2 or not fields[0].startswith("HTTP/") or not fields[1].isdigit():
            raise ValueError(f"malformed status line {status_line!r}")
        status = int(fields[1])

        got: dict[str, str] = {}
        while True:
            line = await reader.readline()
            if line in (b"\r\n", b"\n", b""):
                break
            name, _, value = line.decode("latin-1").partition(":")
            got[name.strip().lower()] = value.strip()

        if "content-length" in got:
            body = await reader.readexactly(int(got["content-length"]))
        else:
            body = await reader.read()  # no length: the body ends when the server closes
        return Response(status, got, body)
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except (ConnectionError, OSError):
            pass


def is_retryable(exc: BaseException) -> bool:
    """True for failures the next attempt may not see: HTTP 5xx and 429, a
    body cut short, a refused or reset connection, a timeout. False for
    everything else (HTTP 4xx, a malformed response, a bug)."""
    if isinstance(exc, HTTPError):
        return exc.status >= 500 or exc.status == 429
    return isinstance(
        exc, (asyncio.IncompleteReadError, ConnectionError, TimeoutError, OSError)
    )


async def fetch(
    url: str,
    *,
    attempts: int = 3,
    timeout: float = 5.0,
    base_delay: float = 0.1,
    sleep: Sleep = asyncio.sleep,
) -> bytes:
    """The body of a 200 response to GET url.

    Each attempt runs under its own `timeout` (seconds). A retryable failure
    (is_retryable) waits base_delay * 2 ** (n - 1) seconds through `sleep`
    after attempt n and tries again, up to `attempts` attempts in all; the
    last failure is raised. A non-retryable failure is raised at once.
    """
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    for n in range(1, attempts + 1):
        try:
            async with asyncio.timeout(timeout):
                resp = await http_get(url)
            if resp.status != 200:
                raise HTTPError(url, resp.status)
            return resp.body
        except Exception as e:  # noqa: BLE001  classified just below
            if n == attempts or not is_retryable(e):
                raise
        await sleep(base_delay * 2 ** (n - 1))
    raise AssertionError("unreachable")


async def download_all(
    urls: Iterable[str],
    *,
    limit: int = 4,
    attempts: int = 3,
    timeout: float = 5.0,
    base_delay: float = 0.1,
    sleep: Sleep = asyncio.sleep,
) -> dict[str, bytes]:
    """fetch() every distinct url concurrently, with at most `limit` fetches
    in flight at any moment. Returns {url: body} in the order the urls were
    first given.

    When any fetch fails for good, every other fetch is cancelled (none is
    left running) and that first failure is raised as itself, not wrapped in
    an ExceptionGroup. ValueError when limit < 1.
    """
    if limit < 1:
        raise ValueError("limit must be at least 1")
    order = list(dict.fromkeys(urls))
    sem = asyncio.Semaphore(limit)
    got: dict[str, bytes] = {}

    async def one(u: str) -> None:
        async with sem:
            got[u] = await fetch(
                u, attempts=attempts, timeout=timeout, base_delay=base_delay, sleep=sleep
            )

    try:
        async with asyncio.TaskGroup() as tg:
            for u in order:
                tg.create_task(one(u))
    except BaseExceptionGroup as eg:
        first = eg.exceptions[0]
        while isinstance(first, BaseExceptionGroup):
            first = first.exceptions[0]
        raise first from None
    return {u: got[u] for u in order}
