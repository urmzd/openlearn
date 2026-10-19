"""flakyhttp: a local HTTP server that fails on purpose (B5).

The corpus fetcher (data.01) must survive what real mirrors do: transient
5xx, connections cut mid-body, servers that ignore or honour Range, and
published checksums that do not match the bytes. No test touches the
network; they point the fetcher at this server.

    with FlakyHTTP({"/shard-0.parquet": data}) as srv:
        srv.fail("/shard-0.parquet", status=503, times=2)    # then succeed
        srv.truncate("/shard-0.parquet", after=1000, times=1)
        url = srv.url("/shard-0.parquet")
        ...
        assert srv.hits("/shard-0.parquet") == 4

Every response carries `X-Content-Sha256` (the true sha256 unless
`lie_checksum` is set) and `Accept-Ranges: bytes` (unless `no_ranges`).
A Range request `bytes=a-` or `bytes=a-b` gets 206 with Content-Range.
"""

from __future__ import annotations

import hashlib
import http.server
import threading
from dataclasses import dataclass, field


@dataclass
class _Plan:
    fail_status: int = 0
    fail_times: int = 0
    truncate_after: int = -1
    truncate_times: int = 0
    lie: bool = False
    no_ranges: bool = False
    delay_s: float = 0.0
    hits: int = 0
    log: list = field(default_factory=list)


class FlakyHTTP:
    def __init__(self, files: dict[str, bytes], poll_interval: float = 0.5):
        self.poll_interval = (
            poll_interval  # serve_forever's shutdown poll; tests may pass 0.01
        )
        self.files = dict(files)
        self.plans: dict[str, _Plan] = {p: _Plan() for p in files}
        self._lock = threading.Lock()
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_HEAD(self):
                self._serve(head=True)

            def do_GET(self):
                self._serve(head=False)

            def _serve(self, head: bool):
                path = self.path.split("?", 1)[0]
                with outer._lock:
                    plan = outer.plans.get(path)
                    data = outer.files.get(path)
                    if plan is not None:
                        plan.hits += 1
                        plan.log.append(dict(self.headers))
                if plan is None or data is None:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if plan.delay_s:
                    import time

                    time.sleep(plan.delay_s)
                with outer._lock:
                    failing = plan.fail_times > 0
                    if failing:
                        plan.fail_times -= 1
                    cut = plan.truncate_after if plan.truncate_times > 0 else -1
                    if cut >= 0:
                        plan.truncate_times -= 1
                if failing:
                    body = b'{"error": "flaky"}'
                    self.send_response(plan.fail_status)
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Retry-After", "0")
                    self.end_headers()
                    if not head:
                        self.wfile.write(body)
                    return
                sha = hashlib.sha256(data).hexdigest()
                if plan.lie:
                    sha = hashlib.sha256(data + b"lie").hexdigest()
                status, start, end = 200, 0, len(data)
                rng = self.headers.get("Range")
                if rng and not plan.no_ranges and rng.startswith("bytes="):
                    a, _, b = rng[len("bytes=") :].partition("-")
                    start = int(a) if a else 0
                    end = int(b) + 1 if b else len(data)
                    if start >= len(data):
                        self.send_response(416)
                        self.send_header("Content-Range", f"bytes */{len(data)}")
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    end = min(end, len(data))
                    status = 206
                body = data[start:end]
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("X-Content-Sha256", sha)
                if not plan.no_ranges:
                    self.send_header("Accept-Ranges", "bytes")
                if status == 206:
                    self.send_header(
                        "Content-Range", f"bytes {start}-{end - 1}/{len(data)}"
                    )
                self.end_headers()
                if head:
                    return
                if cut >= 0:
                    self.wfile.write(body[:cut])
                    self.wfile.flush()
                    self.close_connection = True
                    try:
                        self.connection.shutdown(2)
                    except OSError:
                        pass
                    return
                self.wfile.write(body)

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            kwargs={"poll_interval": self.poll_interval},
            daemon=True,
        )

    # -- lifecycle ------------------------------------------------------------

    def __enter__(self) -> "FlakyHTTP":
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def url(self, path: str) -> str:
        return self.base + path

    # -- faults -----------------------------------------------------------------

    def _plan(self, path: str) -> _Plan:
        if path not in self.plans:
            raise KeyError(f"flakyhttp serves no {path!r}")
        return self.plans[path]

    def fail(self, path: str, status: int = 503, times: int = 1) -> None:
        with self._lock:
            p = self._plan(path)
            p.fail_status, p.fail_times = status, times

    def truncate(self, path: str, after: int, times: int = 1) -> None:
        with self._lock:
            p = self._plan(path)
            p.truncate_after, p.truncate_times = after, times

    def lie_checksum(self, path: str, lie: bool = True) -> None:
        with self._lock:
            self._plan(path).lie = lie

    def no_ranges(self, path: str, off: bool = True) -> None:
        with self._lock:
            self._plan(path).no_ranges = off

    def delay(self, path: str, seconds: float) -> None:
        with self._lock:
            self._plan(path).delay_s = seconds

    def hits(self, path: str) -> int:
        with self._lock:
            return self._plan(path).hits

    def headers(self, path: str) -> list[dict]:
        with self._lock:
            return list(self._plan(path).log)
