"""lang.09 course tests: your async ticker server in primers/lang.09.

Annotated exemplars (DESIGN 5.12). The session builds your crate once
(`cargo build`), starts `ticker --port 0`, reads the port from its first
stdout line, and speaks raw HTTP/1.1 to it over TCP with `Connection:
close`. Streamed bodies arrive with chunked transfer encoding (hyper's
choice for a body of unknown length), which `dechunk` undoes. Nothing here
edits your files.
"""

import json
import os
import re
import select
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path

import pytest

PRIMER = Path(os.environ.get("OL_PRIMER_DIR", "primers/lang.09")).resolve()
TARGET = Path(os.environ.get("CARGO_TARGET_DIR", str(PRIMER / "target"))).resolve()
TIMEOUT = 10.0


def tail(text: str, n: int = 30) -> str:
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


def launch():
    proc = subprocess.Popen(
        [str(TARGET / "debug" / "ticker"), "--port", "0"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    ready, _, _ = select.select([proc.stdout], [], [], 10)
    first = proc.stdout.readline().decode(errors="replace") if ready else ""
    m = re.match(r"listening on 127\.0\.0\.1:(\d+)\s*$", first)
    if not m:
        proc.kill()
        pytest.fail(
            f"ticker --port 0 must print `listening on 127.0.0.1:<port>` first; got {first!r}"
        )
    return proc, ("127.0.0.1", int(m.group(1)))


@pytest.fixture(scope="session")
def server(build):
    assert build.returncode == 0, f"cargo build failed:\n{tail(build.stderr)}"
    proc, addr = launch()
    yield addr
    proc.kill()
    proc.wait(timeout=5)


def dechunk(body: bytes) -> bytes:
    out = b""
    while True:
        size_line, _, body = body.partition(b"\r\n")
        size = int(size_line.strip() or b"0", 16)
        if size == 0:
            return out
        out += body[:size]
        body = body[size + 2 :]


class Reply:
    def __init__(self, raw: bytes):
        self.raw = raw
        head, sep, body = raw.partition(b"\r\n\r\n")
        assert sep, f"no blank line after the response headers: {raw[:200]!r}"
        lines = head.decode("latin-1").split("\r\n")
        m = re.match(r"HTTP/1\.1 (\d{3}) ", lines[0])
        assert m, f"bad status line {lines[0]!r}"
        self.status = int(m.group(1))
        self.headers = {}
        for line in lines[1:]:
            k, _, v = line.partition(":")
            self.headers[k.strip().lower()] = v.strip()
        chunked = self.headers.get("transfer-encoding", "").lower() == "chunked"
        self.body = dechunk(body) if chunked else body

    def json(self):
        return json.loads(self.body.decode())


def exchange(addr, raw: bytes) -> Reply:
    s = socket.create_connection(addr, timeout=TIMEOUT)
    s.sendall(raw)
    data = b""
    while True:
        chunk = s.recv(65536)
        if not chunk:
            break
        data += chunk
    s.close()
    return Reply(data)


def get(addr, path: str) -> Reply:
    return exchange(
        addr, f"GET {path} HTTP/1.1\r\nHost: t\r\nConnection: close\r\n\r\n".encode()
    )


def post(addr, path: str, body: bytes) -> Reply:
    head = f"POST {path} HTTP/1.1\r\nHost: t\r\nConnection: close\r\nContent-Length: {len(body)}\r\n\r\n"
    return exchange(addr, head.encode() + body)


def stats(addr) -> dict:
    return get(addr, "/stats").json()


def test_crate_builds(build):
    # WHY: primers/lang.09 is one package, a library and a `ticker` binary,
    #      on tokio and hyper (the crates L10.5's server uses).
    # KIND: conformance
    # CHAPTER: lang.09 section 4
    assert build.returncode == 0, f"cargo build failed:\n{tail(build.stderr)}"
    assert (TARGET / "debug" / "ticker").is_file(), (
        "the package must build a binary named ticker"
    )


def test_your_cargo_tests_pass(build):
    # WHY: your own async tests (#[tokio::test]) of the producer and the
    #      helpers are part of the deliverable.
    # KIND: unit
    # CHAPTER: lang.09 section 2
    r = cargo("test", "-q")
    assert r.returncode == 0, (
        f"cargo test failed:\n{tail(r.stdout)}\n{tail(r.stderr, 12)}"
    )


def test_hand_example_ticks(server):
    # WHY: the chapter's worked example: /ticks?n=3 is a text/event-stream
    #      of {"i":0}, {"i":1}, {"i":2}, then [DONE], each `data: ...\n\n`.
    # KIND: unit
    # CHAPTER: lang.09 section 3
    r = get(server, "/ticks?n=3&interval_ms=10")
    assert r.status == 200, r.raw[:300]
    assert r.headers.get("content-type") == "text/event-stream"
    assert (
        r.body == b'data: {"i":0}\n\ndata: {"i":1}\n\ndata: {"i":2}\n\ndata: [DONE]\n\n'
    )


def test_health(server):
    # WHY: a handler that answers without touching shared state.
    # KIND: unit
    # CHAPTER: lang.09 section 3
    r = get(server, "/health")
    assert r.status == 200 and r.json() == {"ok": True}


def test_events_leave_as_produced(server):
    # WHY: a stream is written as it is produced: with 5 events 150 ms apart,
    #      the first arrives long before the producer is done, and the gaps
    #      between arrivals follow the producer's pace.
    # KIND: unit
    # CHAPTER: lang.09 section 2
    s = socket.create_connection(server, timeout=TIMEOUT)
    s.sendall(
        b"GET /ticks?n=5&interval_ms=150 HTTP/1.1\r\nHost: t\r\nConnection: close\r\n\r\n"
    )
    t0 = time.monotonic()
    times, data = [], b""
    while True:
        chunk = s.recv(65536)
        if not chunk:
            break
        data += chunk
        while len(times) < data.count(b"data: "):
            times.append(time.monotonic() - t0)
    s.close()
    assert len(times) == 6, data
    assert times[0] < 0.4, (
        f"the first event took {times[0]:.2f} s: is the whole stream buffered?"
    )
    assert times[-1] - times[0] > 0.5, f"events arrived together ({times})"


def test_disconnect_cancels_the_producer(server):
    # WHY: when the client goes away, hyper drops the body, the channel's
    #      receiver goes with it, and the producer must stop (not tick on for
    #      1000 events): /stats shows it cancelled and no producer active.
    # KIND: fault
    # CHAPTER: lang.09 section 5, pitfalls
    before = stats(server)["cancelled"]
    s = socket.create_connection(server, timeout=TIMEOUT)
    s.sendall(
        b"GET /ticks?n=1000&interval_ms=20 HTTP/1.1\r\nHost: t\r\nConnection: close\r\n\r\n"
    )
    data = b""
    while data.count(b"data: ") < 2:
        data += s.recv(65536)
    s.close()
    deadline = time.monotonic() + 3
    while True:
        st = stats(server)
        if st["cancelled"] == before + 1 and st["active"] == 0:
            break
        assert time.monotonic() < deadline, (
            f"producer still running after the client left: {st}"
        )
        time.sleep(0.05)


def test_slow_reader_bounds_the_producer(server):
    # WHY: backpressure: the producer awaits a bounded channel, so a client
    #      that reads nothing for a while makes it wait instead of queueing
    #      every event in memory; it is still active (not finished) after
    #      200 zero-delay events would have been produced.
    # KIND: unit
    # CHAPTER: lang.09 section 2
    s = socket.create_connection(server, timeout=TIMEOUT)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    s.sendall(
        b"GET /ticks?n=200000&interval_ms=0 HTTP/1.1\r\nHost: t\r\nConnection: close\r\n\r\n"
    )
    time.sleep(0.5)
    st = stats(server)
    assert st["active"] >= 1, (
        f"the producer finished 200000 events without a reader: {st}"
    )
    s.close()


def test_bounded_admission_answers_429(server):
    # WHY: two job slots: of three concurrent 600 ms jobs, two run and the
    #      third is refused at once with 429 and Retry-After, instead of
    #      waiting in an unbounded queue.
    # KIND: fault
    # CHAPTER: lang.09 section 2
    results = []

    def run():
        results.append(post(server, "/jobs", b'{"ms": 600}'))

    threads = [threading.Thread(target=run) for _ in range(2)]
    for t in threads:
        t.start()
    time.sleep(0.2)
    t0 = time.monotonic()
    third = post(server, "/jobs", b'{"ms": 600}')
    assert third.status == 429, third.raw[:200]
    assert time.monotonic() - t0 < 0.3, "a refusal must be immediate"
    assert third.headers.get("retry-after") == "1"
    for t in threads:
        t.join()
    assert sorted(r.status for r in results) == [200, 200]
    assert post(server, "/jobs", b'{"ms": 1}').status == 200, (
        "slots come back when jobs end"
    )


def test_sigterm_drains_and_exits_zero(build):
    # WHY: SIGTERM (what Kubernetes sends) stops new connections, lets the
    #      request in flight finish, and exits 0 (spec/cli-roles.md).
    # KIND: fault
    # CHAPTER: lang.09 section 2
    assert build.returncode == 0
    proc, addr = launch()
    try:
        out = []
        t = threading.Thread(
            target=lambda: out.append(post(addr, "/jobs", b'{"ms": 400}'))
        )
        t.start()
        time.sleep(0.15)
        proc.send_signal(signal.SIGTERM)
        t.join()
        assert out and out[0].status == 200, "the job in flight must finish"
        assert proc.wait(timeout=5) == 0, "exit 0 after a drain"
    finally:
        if proc.poll() is None:
            proc.kill()
