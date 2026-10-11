"""The effects sink a milestone step names with `sink = "effects"` (DESIGN 4.4).

The same wire protocol as course/testkit/go/effects, served in-process so a
milestone needs no Go toolchain of its own:

    POST /effects   {"key": "...", "value": "..."}   200 {"applied": true|false}
    GET  /effects   every record in arrival order of its first delivery

The worker's `--test-activities` registers `tl.test.Append`, which POSTs its
idempotency key ("<workflow id>/<activity id>") to $TL_TEST_EFFECTS_URL. The
sink dedupes by key, as a correct external system would, and counts every
delivery, so the `effects` matcher sees both the state and the retries.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import HarnessError


@dataclass
class Record:
    key: str
    value: str
    deliveries: int


class Sink:
    def __init__(self) -> None:
        self._mu = threading.Lock()
        self._order: list[str] = []
        self._recs: dict[str, Record] = {}
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, status: int, body) -> None:
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                if self.path.split("?")[0] != "/effects":
                    return self._send(404, {"error": "not found"})
                n = int(self.headers.get("Content-Length") or 0)
                try:
                    doc = json.loads(self.rfile.read(n) or b"{}")
                except json.JSONDecodeError:
                    doc = None
                if not isinstance(doc, dict) or not doc.get("key"):
                    return self._send(400, {"error": 'want {"key": ..., "value": ...}'})
                self._send(200, {"applied": outer.apply(str(doc["key"]), doc)})

            def do_GET(self):
                if self.path.split("?")[0] != "/effects":
                    return self._send(404, {"error": "not found"})
                self._send(200, [r.__dict__ for r in outer.records()])

        self._srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self._srv.daemon_threads = True
        self._thread = threading.Thread(target=self._srv.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._srv.server_address[1]}/effects"

    def apply(self, key: str, doc: dict) -> bool:
        with self._mu:
            rec = self._recs.get(key)
            first = rec is None
            if first:
                rec = Record(key, str(doc.get("value", "")), 0)
                self._recs[key] = rec
                self._order.append(key)
            rec.deliveries += 1
            return first

    def records(self) -> list[Record]:
        with self._mu:
            return [Record(**self._recs[k].__dict__) for k in self._order]

    def close(self) -> None:
        self._srv.shutdown()
        self._srv.server_close()


def parse_keys(spec, where: str) -> list[str]:
    """`keys = "0..499"` (inclusive) or a list of key suffixes."""
    if isinstance(spec, list):
        return [str(k) for k in spec]
    m = re.fullmatch(r"\s*(\d+)\s*\.\.\s*(\d+)\s*", str(spec))
    if not m or int(m.group(1)) > int(m.group(2)):
        raise HarnessError(
            f'{where}: effects keys {spec!r} is not "a..b" (inclusive) or a list'
        )
    return [str(i) for i in range(int(m.group(1)), int(m.group(2)) + 1)]


def check(
    records: list[Record], keys: list[str], prefix: str, delivered_once: bool
) -> tuple[list[str], dict]:
    """Problems with the records under `prefix` against prefix+key for every
    key: missing, unexpected, and (with delivered_once) redelivered. The same
    rules as testkit effects.Sink.Check."""
    want = {prefix + k for k in keys}
    mine = {r.key: r for r in records if r.key.startswith(prefix)}
    problems = []
    for k in sorted(want - set(mine)):
        problems.append(f"missing {k}")
    for k in sorted(set(mine) - want):
        problems.append(f"unexpected {k}")
    redelivered = sorted(k for k, r in mine.items() if r.deliveries > 1)
    if delivered_once:
        problems += [f"{k} delivered {mine[k].deliveries} times" for k in redelivered]
    stats = {
        "applied": len(want & set(mine)),
        "want": len(want),
        "redelivered": len(redelivered),
        "deliveries": sum(r.deliveries for r in mine.values()),
    }
    return problems, stats
