"""lang.04 course tests: your line-protocol echo server in primers/lang.04.

Annotated exemplars (DESIGN 5.12). The session builds your workspace once
(`cargo build`), starts `echo --port 0`, reads the port from its first
stdout line, and talks to it over real TCP sockets. Nothing here edits your
files; build products go to $CARGO_TARGET_DIR (under .ol/ when `ol check`
runs this).
"""

import os
import re
import select
import socket
import subprocess
import time
from pathlib import Path

import pytest

PRIMER = Path(os.environ.get("OL_PRIMER_DIR", "primers/lang.04")).resolve()
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
    exe = TARGET / "debug" / "echo"
    proc = subprocess.Popen(
        [str(exe), "--port", "0"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    ready, _, _ = select.select([proc.stdout], [], [], 10)
    first = proc.stdout.readline().decode(errors="replace") if ready else ""
    m = re.match(r"listening on 127\.0\.0\.1:(\d+)\s*$", first)
    if not m:
        proc.kill()
        err = (
            proc.stderr.read().decode(errors="replace")
            if proc.poll() is not None
            else ""
        )
        pytest.fail(
            f"echo --port 0 must print `listening on 127.0.0.1:<port>` first; got {first!r} {err}"
        )
    yield ("127.0.0.1", int(m.group(1)))
    proc.kill()
    proc.wait(timeout=5)


class Client:
    """One TCP connection with a line reader on top of recv()."""

    def __init__(self, addr):
        self.sock = socket.create_connection(addr, timeout=TIMEOUT)
        self.buf = b""

    def send(self, data: bytes) -> None:
        self.sock.sendall(data)

    def line(self) -> bytes:
        while b"\n" not in self.buf:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise AssertionError(
                    f"connection closed before a full reply line (got {self.buf!r})"
                )
            self.buf += chunk
        out, self.buf = self.buf.split(b"\n", 1)
        return out

    def ask(self, request: bytes) -> bytes:
        self.send(request)
        return self.line()

    def closed(self) -> bool:
        return self.buf == b"" and self.sock.recv(1) == b""

    def close(self) -> None:
        self.sock.close()


def test_workspace_builds(build):
    # WHY: primers/lang.04 is a Cargo workspace with a library (lineproto)
    #      and a binary (echo) that depends on it by path; `cargo build` at
    #      the workspace root builds both, std only, with no network.
    # KIND: conformance
    # CHAPTER: lang.04 section 4
    assert build.returncode == 0, f"cargo build failed:\n{tail(build.stderr)}"
    assert (TARGET / "debug" / "echo").is_file(), (
        "the workspace must build a binary named echo"
    )


def test_your_cargo_tests_pass(build):
    # WHY: your own #[test] functions are part of the deliverable; a primer
    #      whose own tests fail is not done, whatever the server does.
    # KIND: unit
    # CHAPTER: lang.04 section 2, testing
    r = cargo("test", "-q")
    assert r.returncode == 0, (
        f"cargo test failed:\n{tail(r.stdout)}\n{tail(r.stderr, 12)}"
    )


def test_ping_pong(server):
    # WHY: the chapter's worked example, first exchange: one request line in,
    #      one reply line out, `\n`-terminated.
    # KIND: unit
    # CHAPTER: lang.04 section 3
    c = Client(server)
    assert c.ask(b"PING\n") == b"PONG"
    c.close()


def test_echo_returns_the_text_byte_for_byte(server):
    # WHY: the text is everything after the FIRST space, so inner and
    #      leading spaces survive (splitting on whitespace would lose them),
    #      and non-ASCII text comes back as the same bytes.
    # KIND: unit
    # CHAPTER: lang.04 section 3
    c = Client(server)
    assert c.ask(b"ECHO hello world\n") == b"hello world"
    assert c.ask(b"ECHO  two  spaces\n") == b" two  spaces"
    assert c.ask("ECHO héllo\n".encode()) == "héllo".encode()
    c.close()


def test_crlf_line_endings_are_accepted(server):
    # WHY: nc and telnet end lines with \r\n; the \r is part of the line
    #      ending, not of the command (else `PING\r` is an unknown command).
    # KIND: boundary
    # CHAPTER: lang.04 section 5, pitfalls
    c = Client(server)
    assert c.ask(b"PING\r\n") == b"PONG"
    assert c.ask(b"ECHO tail\r\n") == b"tail"
    c.close()


def test_len_counts_utf8_bytes(server):
    # WHY: Rust's str.len() counts UTF-8 bytes, so "héllo" (é is two bytes)
    #      is 6, not 5 Unicode scalar values.
    # KIND: unit
    # CHAPTER: lang.04 section 3
    c = Client(server)
    assert c.ask("LEN héllo\n".encode()) == b"6"
    assert c.ask(b"LEN \n") == b"0"
    c.close()


def test_len_counts_nul_as_a_byte(server):
    # WHY: Rust strings carry NUL as ordinary content; LEN counts every UTF-8
    #      byte, including the NUL, instead of treating it as a terminator.
    # KIND: boundary
    # CHAPTER: lang.04 section 5, pitfalls
    c = Client(server)
    assert c.ask(b"LEN a\x00b\n") == b"3"
    assert c.ask(b"PING\n") == b"PONG", "an error must leave the connection usable"
    c.close()


def test_errors_are_replies_not_crashes(server):
    # WHY: bad input is answered with `ERR <reason>` on the same connection:
    #      an unknown command, an empty line, bytes that are not UTF-8.
    #      `.unwrap()` on any of these would kill the connection's thread.
    # KIND: boundary
    # CHAPTER: lang.04 section 5, pitfalls
    c = Client(server)
    assert c.ask(b"JUMP high\n") == b"ERR unknown command JUMP"
    assert c.ask(b"\n") == b"ERR empty line"
    assert c.ask(b"ECHO \xff\xfe\n") == b"ERR invalid utf-8"
    assert c.ask(b"ping\n") == b"ERR unknown command ping", (
        "commands are case-sensitive"
    )
    assert c.ask(b"PING\n") == b"PONG"
    c.close()


def test_quit_says_bye_and_closes(server):
    # WHY: QUIT is the one request after which the SERVER closes the
    #      connection: the client reads BYE, then end of stream.
    # KIND: unit
    # CHAPTER: lang.04 section 3
    c = Client(server)
    assert c.ask(b"QUIT\n") == b"BYE"
    assert c.closed(), "after BYE the server must close the connection"
    c.close()


def test_several_lines_in_one_packet(server):
    # WHY: TCP is a byte stream, not a message stream: three requests sent in
    #      one write may arrive in one read(). Each still gets its own reply,
    #      in order (a server that answers one line per read() loses two).
    # KIND: boundary
    # CHAPTER: lang.04 section 5, pitfalls
    c = Client(server)
    c.send(b"PING\nECHO a\nLEN abc\n")
    assert [c.line(), c.line(), c.line()] == [b"PONG", b"a", b"3"]
    c.close()


def test_a_line_split_across_packets(server):
    # WHY: the other half of the byte-stream rule: one request may arrive in
    #      several reads. The server must wait for the \n, not answer a
    #      fragment.
    # KIND: boundary
    # CHAPTER: lang.04 section 5, pitfalls
    c = Client(server)
    c.send(b"EC")
    time.sleep(0.1)
    c.send(b"HO split")
    time.sleep(0.1)
    c.send(b" line\n")
    assert c.line() == b"split line"
    c.close()


def test_two_clients_at_once(server):
    # WHY: the stage's done-when: client A holds a half-written line while
    #      client B is served. A server that handles one connection at a
    #      time leaves B waiting on A forever; a thread per connection does not.
    # KIND: fault
    # CHAPTER: lang.04 section 2, threads
    a = Client(server)
    a.send(b"PI")  # A stalls mid-line
    time.sleep(0.1)
    b = Client(server)
    t0 = time.monotonic()
    assert b.ask(b"PING\n") == b"PONG"
    assert time.monotonic() - t0 < 2.0
    a.send(b"NG\n")
    assert a.line() == b"PONG"
    a.close()
    b.close()


def test_a_client_vanishing_mid_line_leaves_the_server_up(server):
    # WHY: a client that disconnects without finishing its line is the
    #      normal case on the internet; that connection ends, the server
    #      keeps accepting.
    # KIND: fault
    # CHAPTER: lang.04 section 5, pitfalls
    gone = Client(server)
    gone.send(b"ECHO never finish")
    gone.sock.shutdown(socket.SHUT_RDWR)
    gone.close()
    time.sleep(0.1)
    c = Client(server)
    assert c.ask(b"PING\n") == b"PONG"
    c.close()
