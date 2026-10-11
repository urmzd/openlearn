"""ol fetch (DESIGN 5.11): pinned assets from a local HTTP server, file://,
and a local build; size and sha256 checks; nothing half-written lands in the
cache; a second fetch is a no-op; verify check 12 lints ASSETS.tsv."""

import hashlib
import http.server
import threading
from pathlib import Path

import pytest

BLOB = b"tiny weights " * 1000
TOK = b'{"model": {"type": "BPE"}}\n'


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@pytest.fixture
def server(tmp_path):
    root = tmp_path / "www"
    root.mkdir()
    (root / "model.bin").write_bytes(BLOB)
    (root / "tok.json").write_bytes(TOK)
    hits = []

    class H(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=str(root), **k)

        def log_message(self, *a):
            hits.append(self.path)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", root, hits
    srv.shutdown()
    srv.server_close()


def _assets(ss, rows: list[list[str]]) -> None:
    text = "# asset\tpath\turl\tsha256\tbytes\tlicense\trevision\n" + "".join(
        "\t".join(r) + "\n" for r in rows
    )
    (ss.course / "fixtures/ASSETS.tsv").write_text(text)


def test_fetch_verifies_and_is_idempotent(ss, server, tmp_path):
    base, root, hits = server
    gen = ss.site / "course/oracle/tiny/build.py"
    gen.parent.mkdir(parents=True)
    gen.write_text(
        "import sys, pathlib\npathlib.Path(sys.argv[1], 'made.txt').write_text('built here\\n')\n"
    )
    local = tmp_path / "local.txt"
    local.write_bytes(b"from disk\n")
    _assets(
        ss,
        [
            [
                "tiny-model",
                "weights/model.bin",
                f"{base}/model.bin",
                sha(BLOB),
                str(len(BLOB)),
                "Apache-2.0",
                "r1",
            ],
            [
                "tiny-model",
                "tokenizer.json",
                f"{base}/tok.json",
                sha(TOK),
                str(len(TOK)),
                "Apache-2.0",
                "r1",
            ],
            [
                "tiny-local",
                "local.txt",
                f"file://{local}",
                sha(b"from disk\n"),
                "10",
                "CC0-1.0",
                "-",
            ],
            [
                "tiny-built",
                "made.txt",
                "build:course/oracle/tiny/build.py",
                sha(b"built here\n"),
                "11",
                "Apache-2.0",
                "-",
            ],
        ],
    )
    out = ss("fetch", "tiny-model", "tiny-local", "tiny-built", rc=0).out
    assert "downloaded weights/model.bin" not in out  # the line names asset/path
    assert (
        "tiny-model/weights/model.bin" in out and "sha256 ok" in out and "built" in out
    )
    cache = Path(ss.env["OL_CACHE"]) / "assets"
    assert (cache / "tiny-model/weights/model.bin").read_bytes() == BLOB
    assert (cache / "tiny-built/made.txt").read_text() == "built here\n"
    n = len(hits)
    out = ss("fetch", "tiny-model", rc=0).out
    assert "0 fetched, 2 already cached" in out and len(hits) == n  # no second download
    assert "tiny-model" in ss("fetch", "--list", rc=0).out


def test_fetch_rejects_a_wrong_hash_and_leaves_nothing(ss, server):
    base, root, hits = server
    _assets(
        ss,
        [
            [
                "tiny-model",
                "model.bin",
                f"{base}/model.bin",
                "0" * 64,
                str(len(BLOB)),
                "Apache-2.0",
                "r1",
            ]
        ],
    )
    out = ss("fetch", "tiny-model", rc=5).out
    assert "sha256" in out and "ASSETS.tsv pins 0000" in out
    d = Path(ss.env["OL_CACHE"]) / "assets/tiny-model"
    assert not (d / "model.bin").exists() and not list(d.glob(".part-*"))
    _assets(
        ss,
        [
            [
                "tiny-model",
                "model.bin",
                f"{base}/model.bin",
                sha(BLOB),
                "5",
                "Apache-2.0",
                "r1",
            ]
        ],
    )
    assert "bytes, ASSETS.tsv pins 5" in ss("fetch", "tiny-model", rc=5).out
    _assets(
        ss,
        [
            [
                "tiny-model",
                "model.bin",
                f"{base}/missing.bin",
                sha(BLOB),
                str(len(BLOB)),
                "Apache-2.0",
                "r1",
            ]
        ],
    )
    assert "download failed" in ss("fetch", "tiny-model", rc=5).out


def test_verify_lints_assets_and_milestone_asset_ids(ss):
    _assets(ss, [["Bad_Id", "../x", "ftp://x", "abc", "1", "-", "-"]])
    out = ss("verify", "course", "--global").out
    assert (
        "asset id must be lowercase" in out
        and "sha256 must be 64" in out
        and "license is required" in out
    )
