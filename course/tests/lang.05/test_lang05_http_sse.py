"""lang.05 course tests: your HTTP/1.1, JSON, and SSE server in primers/lang.05.

Annotated exemplars (DESIGN 5.12). The session builds your crate once
(`cargo build`), starts `wire --port 0`, reads the port from its first
stdout line, and speaks raw HTTP/1.1 to it over TCP: the bytes each test
sends are written out in the test, so you can replay any of them with
`printf '...' | nc 127.0.0.1 <port>`. Nothing here edits your files.
"""

import json
import os
import re
import select
import shutil
import socket
import subprocess
import time
from pathlib import Path

import pytest

PRIMER = Path(os.environ.get("OL_PRIMER_DIR", "primers/lang.05")).resolve()
TARGET = Path(os.environ.get("CARGO_TARGET_DIR", str(PRIMER / "target"))).resolve()
TIMEOUT = 5.0


def tail(text: str, n: int = 30) -> str:
    """The last n lines: compiler errors and test failures come last."""
    lines = text.rstrip().splitlines()
    return "\n".join((["..."] if len(lines) > n else []) + lines[-n:])


def cargo(*args: str, timeout: float = 300) -> subprocess.CompletedProcess:
    env = dict(os.environ, CARGO_TARGET_DIR=str(TARGET), CARGO_TERM_COLOR="never")
    return subprocess.run(
        ["cargo", *args, "--offline", "--manifest-path", str(PRIMER / "Cargo.toml")],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


@pytest.fixture(scope="session")
def build():
    return cargo("build", "-q")


@pytest.fixture(scope="session")
def server(build):
    assert build.returncode == 0, f"cargo build failed:\n{tail(build.stderr)}"
    proc = subprocess.Popen(
        [str(TARGET / "debug" / "wire"), "--port", "0"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    ready, _, _ = select.select([proc.stdout], [], [], 10)
    first = proc.stdout.readline().decode(errors="replace") if ready else ""
    m = re.match(r"listening on 127\.0\.0\.1:(\d+)\s*$", first)
    if not m:
        proc.kill()
        pytest.fail(
            f"wire --port 0 must print `listening on 127.0.0.1:<port>` first; got {first!r}"
        )
    yield ("127.0.0.1", int(m.group(1)))
    proc.kill()
    proc.wait(timeout=5)


class Reply:
    def __init__(self, raw: bytes):
        self.raw = raw
        head, sep, self.body = raw.partition(b"\r\n\r\n")
        assert sep, (
            f"no blank line (\\r\\n\\r\\n) after the response headers: {raw[:200]!r}"
        )
        lines = head.decode("latin-1").split("\r\n")
        m = re.match(r"HTTP/1\.1 (\d{3}) (.+)$", lines[0])
        assert m, f"bad status line {lines[0]!r}; want `HTTP/1.1 <code> <reason>`"
        self.status = int(m.group(1))
        self.headers = {}
        for line in lines[1:]:
            k, _, v = line.partition(":")
            self.headers[k.strip().lower()] = v.strip()

    def json(self):
        return json.loads(self.body.decode())


def exchange(addr, *parts: bytes, pause: float = 0.0) -> Reply:
    """Send the parts (sleeping `pause` between them), read until the server
    closes the connection, and parse the response."""
    s = socket.create_connection(addr, timeout=TIMEOUT)
    for i, p in enumerate(parts):
        if i and pause:
            time.sleep(pause)
        s.sendall(p)
    data = b""
    while True:
        chunk = s.recv(65536)
        if not chunk:
            break
        data += chunk
    s.close()
    return Reply(data)


def post_echo(addr, body: bytes, name: str = "Content-Length") -> Reply:
    head = f"POST /echo HTTP/1.1\r\nHost: test\r\n{name}: {len(body)}\r\n\r\n".encode()
    return exchange(addr, head + body)


def test_crate_builds(build):
    # WHY: primers/lang.05 is one package with a library and a `wire`
    #      binary, std only: no HTTP, JSON, or async crate does the work.
    # KIND: conformance
    # CHAPTER: lang.05 section 4
    assert build.returncode == 0, f"cargo build failed:\n{tail(build.stderr)}"
    assert (TARGET / "debug" / "wire").is_file(), (
        "the package must build a binary named wire"
    )


def test_your_cargo_tests_pass(build):
    # WHY: your own unit tests of the parser and the writers are part of
    #      the deliverable.
    # KIND: unit
    # CHAPTER: lang.05 section 2
    r = cargo("test", "-q")
    assert r.returncode == 0, (
        f"cargo test failed:\n{tail(r.stdout)}\n{tail(r.stderr, 12)}"
    )


def test_health_response_bytes(server):
    # WHY: the chapter's worked example, byte for byte: status line, the
    #      headers with a Content-Length that counts the body's bytes, the
    #      blank line, the body. A client trusts Content-Length, not the close.
    # KIND: unit
    # CHAPTER: lang.05 section 3
    r = exchange(server, b"GET /health HTTP/1.1\r\nHost: test\r\n\r\n")
    assert r.raw.startswith(b"HTTP/1.1 200 OK\r\n")
    assert r.headers.get("content-type") == "application/json"
    assert r.headers.get("content-length") == "11"
    assert r.body == b'{"ok":true}'


def test_echo_round_trips_json_text(server):
    # WHY: the body is parsed as JSON and the reply is built with a JSON
    #      writer: quotes, backslashes, newlines, and non-ASCII text survive
    #      the trip, and "bytes" counts UTF-8 bytes (é is two).
    # KIND: unit
    # CHAPTER: lang.05 section 3
    text = 'hé "q" \\ \n\ttab'
    r = post_echo(server, json.dumps({"text": text}).encode())
    assert r.status == 200, r.raw
    assert r.json() == {"text": text, "bytes": len(text.encode())}
    assert int(r.headers["content-length"]) == len(r.body), (
        "Content-Length counts bytes, not characters"
    )


def test_json_unicode_escapes_and_surrogate_pairs(server):
    # WHY: JSON may write any character as \uXXXX, and a character outside
    #      the Basic Multilingual Plane as TWO escapes (a surrogate pair):
    #      😀 is one emoji, four UTF-8 bytes, not two broken ones.
    # KIND: boundary
    # CHAPTER: lang.05 section 5, pitfalls
    r = post_echo(
        server,
        b'{"text": "\\u00e9\\ud83d\\ude00", "extra": [1, 2.5e-1, null, {"a": false}]}',
    )
    assert r.status == 200, r.raw
    assert r.json() == {"text": "é\U0001f600", "bytes": 6}


def test_header_names_are_case_insensitive(server):
    # WHY: HTTP header names ignore case: curl sends Content-Length, some
    #      proxies send content-length. A parser that compares exactly
    #      misses the body and answers 411.
    # KIND: boundary
    # CHAPTER: lang.05 section 5, pitfalls
    r = post_echo(server, b'{"text":"ok"}', name="content-length")
    assert r.status == 200, r.raw
    r = post_echo(server, b'{"text":"ok"}', name="CONTENT-LENGTH")
    assert r.status == 200, r.raw


def test_body_split_across_packets(server):
    # WHY: TCP may deliver the head and the body in separate reads, and the
    #      body itself in pieces. The server must read exactly Content-Length
    #      bytes, however they arrive.
    # KIND: boundary
    # CHAPTER: lang.05 section 5, pitfalls
    body = b'{"text":"slow body"}'
    head = f"POST /echo HTTP/1.1\r\nHost: t\r\nContent-Length: {len(body)}\r\n\r\n".encode()
    r = exchange(server, head, body[:7], body[7:], pause=0.15)
    assert r.status == 200, r.raw
    assert r.json()["text"] == "slow body"


def test_bad_json_is_a_400_with_a_json_error(server):
    # WHY: a syntax error in the body is the client's fault (400), reported
    #      as JSON so the client can parse it, and never a crash.
    # KIND: boundary
    # CHAPTER: lang.05 section 5, pitfalls
    for body in (
        b'{"text": }',
        b'{"text": "unterminated',
        b'{"text": 1}',
        b"[1, 2",
        b'{"text":"a"} trailing',
    ):
        r = post_echo(server, body)
        assert r.status == 400, (body, r.raw)
        assert "error" in r.json()


def test_post_without_content_length_is_411(server):
    # WHY: without Content-Length (and without chunked encoding, which this
    #      server does not accept) the server cannot know where the body
    #      ends; HTTP's answer is 411 Length Required.
    # KIND: boundary
    # CHAPTER: lang.05 section 2
    r = exchange(server, b'POST /echo HTTP/1.1\r\nHost: t\r\n\r\n{"text":"x"}')
    assert r.status == 411, r.raw


def test_malformed_request_line_is_400(server):
    # WHY: the request line is exactly `METHOD SP target SP HTTP/1.x`;
    #      anything else is answered 400, not guessed at.
    # KIND: boundary
    # CHAPTER: lang.05 section 2
    for raw in (
        b"GARBAGE\r\n\r\n",
        b"GET /health\r\n\r\n",
        b"GET health HTTP/1.1\r\n\r\n",
        b"GET /health HTTP/9\r\n\r\n",
    ):
        r = exchange(server, raw)
        assert r.status == 400, (raw, r.raw)


def test_unknown_path_and_wrong_method(server):
    # WHY: 404 means "no such resource", 405 means "the resource exists but
    #      not for this method", and a 405 names the allowed method in Allow.
    # KIND: unit
    # CHAPTER: lang.05 section 2
    r = exchange(server, b"GET /nope HTTP/1.1\r\nHost: t\r\n\r\n")
    assert r.status == 404, r.raw
    r = exchange(server, b"GET /echo HTTP/1.1\r\nHost: t\r\n\r\n")
    assert r.status == 405, r.raw
    assert r.headers.get("allow") == "POST"


def test_sse_stream_framing(server):
    # WHY: the SSE worked example: text/event-stream, each event is one
    #      `data: <json>` line plus a blank line, and `data: [DONE]` ends the
    #      stream. The tracer engine (L10.0) streams completions exactly so.
    # KIND: conformance
    # CHAPTER: lang.05 section 3
    r = exchange(server, b"GET /count?n=3 HTTP/1.1\r\nHost: t\r\n\r\n")
    assert r.status == 200, r.raw
    assert r.headers.get("content-type") == "text/event-stream"
    assert "content-length" not in r.headers, (
        "a stream has no length: it ends when the server closes"
    )
    assert (
        r.body == b'data: {"i":0}\n\ndata: {"i":1}\n\ndata: {"i":2}\n\ndata: [DONE]\n\n'
    )


def test_sse_events_leave_as_they_happen(server):
    # WHY: streaming means the first event reaches the client before the
    #      last one is produced. Buffering the response (a BufWriter that is
    #      never flushed, Nagle on tiny writes) delivers everything at the end.
    # KIND: fault
    # CHAPTER: lang.05 section 5, pitfalls
    s = socket.create_connection(server, timeout=TIMEOUT)
    t0 = time.monotonic()
    s.sendall(b"GET /count?n=2&delay_ms=1000 HTTP/1.1\r\nHost: t\r\n\r\n")
    data = b""
    first = None
    while True:
        chunk = s.recv(65536)
        if not chunk:
            break
        data += chunk
        if first is None and b'data: {"i":0}\n\n' in data:
            first = time.monotonic() - t0
    total = time.monotonic() - t0
    s.close()
    assert first is not None, data
    assert first < 0.7, (
        f"the first event arrived after {first:.2f}s; it must leave before the 1s pause"
    )
    assert total >= 0.95, (
        f"the stream ended after {total:.2f}s; the pause between events is 1s"
    )


def test_curl_n_reads_the_stream(server):
    # WHY: the exercise's done-when: an off-the-shelf client (curl -N, no
    #      buffering) reads your stream, so your framing is the standard one.
    # KIND: conformance
    # CHAPTER: lang.05 section 4
    curl = shutil.which("curl")
    if curl is None:
        pytest.skip("curl is not installed")
    host, port = server
    r = subprocess.run(
        [curl, "-sN", "--max-time", "5", f"http://{host}:{port}/count?n=2"],
        capture_output=True,
        timeout=10,
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout == b'data: {"i":0}\n\ndata: {"i":1}\n\ndata: [DONE]\n\n'
