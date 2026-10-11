"""Tiny HTTP client (stdlib http.client) for milestones, conformance, drills.

Unlike urllib it never raises on 4xx/5xx: the status is data. Bodies are read
to the end, so an SSE stream is returned whole once the server closes it.
"""

from __future__ import annotations

import http.client
import json
import socket
import time
import urllib.parse
from dataclasses import dataclass, field


@dataclass
class Response:
    status: int
    headers: dict[str, str] = field(default_factory=dict)  # lower-cased names
    body: bytes = b""
    error: str = ""  # connection-level failure; status is 0

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")

    def json(self):
        return json.loads(self.body.decode("utf-8"))

    @property
    def content_type(self) -> str:
        return self.headers.get("content-type", "").split(";")[0].strip().lower()


def request(
    method: str,
    url: str,
    *,
    json_body=None,
    headers: dict | None = None,
    timeout: float = 30.0,
    data: bytes | None = None,
) -> Response:
    u = urllib.parse.urlsplit(url)
    if u.scheme not in ("http", "https"):
        return Response(0, error=f"unsupported URL {url!r}")
    conn_cls = (
        http.client.HTTPSConnection
        if u.scheme == "https"
        else http.client.HTTPConnection
    )
    hdrs = dict(headers or {})
    if json_body is not None:
        data = json.dumps(json_body).encode()
        hdrs.setdefault("Content-Type", "application/json")
    path = u.path or "/"
    if u.query:
        path += "?" + u.query
    try:
        conn = conn_cls(u.hostname, u.port, timeout=timeout)
        conn.request(method, path, body=data, headers=hdrs)
        r = conn.getresponse()
        body = r.read()
        out = Response(r.status, {k.lower(): v for k, v in r.getheaders()}, body)
        conn.close()
        return out
    except (OSError, http.client.HTTPException, socket.timeout) as e:
        return Response(0, error=f"{method} {url}: {type(e).__name__}: {e}")


def get(url: str, params: dict | None = None, **kw) -> Response:
    if params:
        url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
    return request("GET", url, **kw)


def wait_ok(
    url: str, timeout_s: float, alive=lambda: True, interval: float = 0.1
) -> tuple[bool, str]:
    """Poll until the URL answers 2xx. `alive()` returning False stops early."""
    deadline = time.monotonic() + timeout_s
    last = ""
    while time.monotonic() < deadline:
        if not alive():
            return False, "the process exited before it was healthy"
        r = request("GET", url, timeout=2.0)
        if 200 <= r.status < 300:
            return True, ""
        last = r.error or f"HTTP {r.status}"
        time.sleep(interval)
    return False, f"not healthy after {timeout_s}s ({last})"


def sse_events(body: bytes) -> tuple[list[str], list[str]]:
    """Split an SSE body into `data:` payloads, byte for byte per DESIGN 2.6.
    Returns (payloads, framing errors). Comment lines (`: ping`) are allowed."""
    errs: list[str] = []
    text = body.decode("utf-8", errors="replace")
    if not text.endswith("\n\n"):
        errs.append("stream does not end with a blank line")
    payloads: list[str] = []
    for i, block in enumerate(text.split("\n\n")):
        if block == "":
            continue
        lines = block.split("\n")
        if all(x.startswith(":") for x in lines):
            continue
        if len(lines) != 1 or not lines[0].startswith("data: "):
            errs.append(f"event {i}: want one `data: <json>` line, got {block[:80]!r}")
            continue
        payloads.append(lines[0][len("data: ") :])
    if not payloads or payloads[-1] != "[DONE]":
        errs.append("stream does not end with `data: [DONE]`")
    elif "[DONE]" in payloads[:-1]:
        errs.append("`data: [DONE]` appears before the end")
    return payloads, errs
