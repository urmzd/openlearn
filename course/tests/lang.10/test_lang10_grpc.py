"""lang.10 course tests: your Protocol Buffers and gRPC primer in primers/lang.10.

Annotated exemplars (DESIGN 5.12). The session builds your Go module
(the `kvstore` server and the `evolve` decoder, over the vendored stubs in
contracts/go) and your Rust package (the `kvpush` client over
contracts/rust/tl-proto), offline. Wire-format tests feed `evolve` bytes
written out by hand in each test; RPC tests start `kvstore --port 0` and
drive it with `kvpush`, so your Go server and your Rust client must agree
through the contract alone. Nothing here edits your files.
"""

import json
import os
import re
import select
import signal
import subprocess
from pathlib import Path

import pytest

PRIMER = Path(os.environ.get("OL_PRIMER_DIR", "primers/lang.10")).resolve()
TARGET = Path(
    os.environ.get("CARGO_TARGET_DIR", str(PRIMER / "rust" / "target"))
).resolve()
BIN = Path(os.environ.get("OL_PRIMER_BIN", str(PRIMER / "bin"))).resolve()
GO_ENV = dict(
    os.environ, GOPROXY="off", GOWORK="off", GOTOOLCHAIN="local", GOFLAGS="-mod=mod"
)


def tail(text: str, n: int = 30) -> str:
    lines = text.rstrip().splitlines()
    return "\n".join((["..."] if len(lines) > n else []) + lines[-n:])


def run(cmd, cwd, env=None, timeout=300, stdin=None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        input=stdin,
    )


@pytest.fixture(scope="session")
def go_build():
    BIN.mkdir(parents=True, exist_ok=True)
    return run(
        ["go", "build", "-o", str(BIN) + "/", "./cmd/kvstore", "./cmd/evolve"],
        PRIMER / "go",
        GO_ENV,
    )


@pytest.fixture(scope="session")
def rust_build():
    env = dict(os.environ, CARGO_TARGET_DIR=str(TARGET), CARGO_TERM_COLOR="never")
    return run(
        [
            "cargo",
            "build",
            "-q",
            "--offline",
            "--manifest-path",
            str(PRIMER / "rust" / "Cargo.toml"),
        ],
        PRIMER,
        env,
    )


def evolve(hexwire: str) -> dict:
    r = run([str(BIN / "evolve")], PRIMER, stdin=hexwire + "\n", timeout=10)
    assert r.returncode == 0, f"evolve failed:\n{r.stderr}"
    return json.loads(r.stdout.strip().splitlines()[-1])


def launch():
    proc = subprocess.Popen(
        [str(BIN / "kvstore"), "--port", "0"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    ready, _, _ = select.select([proc.stdout], [], [], 10)
    first = proc.stdout.readline().decode(errors="replace") if ready else ""
    m = re.match(r"listening on 127\.0\.0\.1:(\d+)\s*$", first)
    if not m:
        proc.kill()
        pytest.fail(
            f"kvstore --port 0 must print `listening on 127.0.0.1:<port>` first; got {first!r}"
        )
    return proc, f"http://127.0.0.1:{m.group(1)}"


@pytest.fixture(scope="session")
def server(go_build, rust_build):
    assert go_build.returncode == 0, f"go build failed:\n{tail(go_build.stderr)}"
    assert rust_build.returncode == 0, f"cargo build failed:\n{tail(rust_build.stderr)}"
    proc, addr = launch()
    yield addr
    proc.kill()
    proc.wait(timeout=5)


def kvpush(*args: str) -> tuple[int, dict]:
    r = run([str(TARGET / "debug" / "kvpush"), *args], PRIMER, timeout=20)
    lines = r.stdout.strip().splitlines()
    assert lines, f"kvpush printed nothing (exit {r.returncode}):\n{r.stderr}"
    return r.returncode, json.loads(lines[-1])


def test_go_builds_and_your_go_tests_pass(go_build):
    # WHY: the Go half builds against the vendored generated stubs (no
    #      protoc on your machine) and your own `go test` passes.
    # KIND: conformance
    # CHAPTER: lang.10 section 4
    assert go_build.returncode == 0, f"go build failed:\n{tail(go_build.stderr)}"
    r = run(["go", "test", "./..."], PRIMER / "go", GO_ENV)
    assert r.returncode == 0, f"go test failed:\n{tail(r.stdout)}\n{tail(r.stderr, 12)}"


def test_rust_builds_and_your_cargo_tests_pass(rust_build):
    # WHY: the Rust half builds against tl-proto (prost and tonic output,
    #      committed) and your own `cargo test` passes.
    # KIND: conformance
    # CHAPTER: lang.10 section 4
    assert rust_build.returncode == 0, f"cargo build failed:\n{tail(rust_build.stderr)}"
    env = dict(os.environ, CARGO_TARGET_DIR=str(TARGET), CARGO_TERM_COLOR="never")
    r = run(
        [
            "cargo",
            "test",
            "-q",
            "--offline",
            "--manifest-path",
            str(PRIMER / "rust" / "Cargo.toml"),
        ],
        PRIMER,
        env,
    )
    assert r.returncode == 0, (
        f"cargo test failed:\n{tail(r.stdout)}\n{tail(r.stderr, 12)}"
    )


def test_hand_example_wire_bytes(go_build):
    # WHY: the chapter's worked example: KvChunk{handle_id: "h",
    #      block_index: 2} is 0a 01 68 18 02 (tag 1 LEN, length 1, 'h'; tag 3
    #      VARINT, 2); every other field is at its default and takes no bytes.
    # KIND: unit
    # CHAPTER: lang.10 section 3
    assert go_build.returncode == 0
    f = evolve("0a01681802")
    assert f["handle_id"] == "h" and f["block_index"] == 2
    assert f["kv_format"] == 0 and f["block_hash"] == 0 and f["payload_len"] == 0
    assert f["reencoded"] == "0a01681802"


def test_unknown_fields_survive_a_round_trip(go_build):
    # WHY: forward compatibility: a newer writer may add fields this reader
    #      does not know (here field 99, a varint, and field 8, bytes); the
    #      reader skips them by wire type and keeps them, so re-encoding
    #      loses nothing.
    # KIND: unit
    # CHAPTER: lang.10 section 2
    assert go_build.returncode == 0
    for wire in ["0a01681802980607", "0a0168180242026869"]:
        f = evolve(wire)
        assert f["handle_id"] == "h" and f["block_index"] == 2, f
        assert f["reencoded"] == wire


def test_varints_and_bytes_decode(go_build):
    # WHY: varints are little-endian groups of 7 bits with a continuation
    #      bit: block_hash 300 is ac 02; bytes fields carry a length prefix;
    #      fixed numbers of fields in any order decode the same.
    # KIND: unit
    # CHAPTER: lang.10 section 2
    assert go_build.returncode == 0
    f = evolve("10ac02" + "28" + "01" + "3203616263" + "0a0178")
    assert f["block_hash"] == 300
    assert f["kv_format"] == 1
    assert f["payload_len"] == 3
    assert f["handle_id"] == "x"
    assert f["reencoded"] == "0a017810ac0228013203616263", (
        "known fields come back in field-number order"
    )


def test_push_then_dedup(server):
    # WHY: unary then client streaming: the first push of 3 blocks finds
    #      none present and sends all 3; the second finds all 3 and sends
    #      them empty (deduplicated), which is the bandwidth L10.6 saves.
    # KIND: conformance
    # CHAPTER: lang.10 section 2
    rc, out = kvpush(
        "push", "--addr", server, "--handle", "a", "--blocks", "3", "--seed", "11"
    )
    assert rc == 0, out
    assert out == {"present": [False, False, False], "received": 3, "deduped": 0}
    rc, out = kvpush(
        "push", "--addr", server, "--handle", "b", "--blocks", "5", "--seed", "11"
    )
    assert rc == 0, out
    assert out == {
        "present": [True, True, True, False, False],
        "received": 2,
        "deduped": 3,
    }


def test_crc_mismatch_is_data_loss(server):
    # WHY: a payload whose CRC-32C does not match is refused with DATA_LOSS
    #      (the status kv.proto names), and the client reports the code.
    # KIND: fault
    # CHAPTER: lang.10 section 5, pitfalls
    rc, out = kvpush(
        "push",
        "--addr",
        server,
        "--handle",
        "c",
        "--blocks",
        "2",
        "--seed",
        "22",
        "--corrupt",
    )
    assert rc == 1 and out["error"] == "DATA_LOSS", out


def test_kv_format_mismatch_is_failed_precondition(server):
    # WHY: a reader refuses a format it cannot read with FAILED_PRECONDITION
    #      before reading any block (the craft.13 migration relies on it).
    # KIND: fault
    # CHAPTER: lang.10 section 5, pitfalls
    rc, out = kvpush(
        "push",
        "--addr",
        server,
        "--handle",
        "d",
        "--blocks",
        "1",
        "--seed",
        "33",
        "--kv-format",
        "2",
    )
    assert rc == 1 and out["error"] == "FAILED_PRECONDITION", out


def test_release_is_idempotent(server):
    # WHY: Release of a handle, and of the same or an unknown handle again,
    #      succeeds: cleanup paths may run twice.
    # KIND: unit
    # CHAPTER: lang.10 section 2
    for h in ["a", "a", "never-pushed"]:
        rc, out = kvpush("release", "--addr", server, "--handle", h)
        assert rc == 0 and out == {"released": True}, out


def test_sigterm_stops_gracefully(go_build):
    # WHY: SIGTERM makes the gRPC server stop accepting, finish in-flight
    #      calls (GracefulStop), and exit 0.
    # KIND: fault
    # CHAPTER: lang.10 section 2
    assert go_build.returncode == 0
    proc, _ = launch()
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=10) == 0
