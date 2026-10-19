"""Model test suite for data.07 (properties): token
conservation, split purity, whole documents per file, determinism, and the
header, mostly with the byte tokenizer so a document's ids are its UTF-8
bytes. They import only names in contracts/py/corpus/{tokenize,shard,stage}.pyi;
`ol mutate data.07` runs them against the reference with one planted bug at
a time."""

import hashlib
import json
import struct

import pytest
from corpus.shard import read_shards, write_shards
from corpus.stage import Doc
from corpus.tokenize import read_bin, tokenize_shards, write_bin
from hypothesis import given, settings
from hypothesis import strategies as st


def build(tmp, texts, permille=300, name="c"):
    docs = [
        Doc(
            id=f"a:{i:04d}",
            source_id="a",
            text=t,
            meta={"url": "u", "license_spdx": "MIT"},
        )
        for i, t in enumerate(texts)
    ]
    out = tmp / name
    write_shards(
        docs, out, dataset="d", version="v1", shard_rows=16, val_permille=permille
    )
    return out / "_MANIFEST.json"


def test_hand_example(tmp_path):
    m = build(tmp_path, ["Hi", "é", "ok"], permille=100)
    tm = tokenize_shards(m, None, tmp_path / "t", tokenizer_id="bytes")
    raw = (tmp_path / "t" / "train-00000.bin").read_bytes()
    assert struct.unpack("<4i", raw[:16]) == (20240520, 1, 4, 256)
    assert raw[1024:] == bytes([72, 0, 105, 0, 111, 0, 107, 0])
    assert tm.val_bytes_per_token == 1.0


texts = st.lists(st.text(min_size=0, max_size=20), min_size=0, max_size=25)


@settings(max_examples=40)
@given(texts)
def test_bytes_are_conserved_per_split(tmp_path_factory, ts):
    tmp = tmp_path_factory.mktemp("p")
    m = build(tmp, ts)
    tm = tokenize_shards(m, None, tmp / "t", tokenizer_id="bytes", max_file_tokens=50)
    for split in ("train", "val"):
        want = b"".join(d.text.encode() for d in read_shards(m.parent, split=split))
        got = b""
        for f in tm.files:
            if f["split"] == split:
                h, ids = read_bin(tmp / "t" / f["name"])
                assert h["n_tokens"] == f["n_tokens"] == len(ids)
                got += bytes(ids.tolist())
        assert got == want


def test_files_end_at_document_boundaries(tmp_path):
    m = build(tmp_path, ["a" * 4, "b" * 6, "c" * 2, "d" * 15], permille=0)
    tm = tokenize_shards(
        m, None, tmp_path / "t", tokenizer_id="bytes", max_file_tokens=10
    )
    assert [f["n_tokens"] for f in tm.files if f["split"] == "train"] == [10, 2, 15]
    m = build(tmp_path, ["x" * 12, "y"], permille=0, name="c2")
    tm = tokenize_shards(
        m, None, tmp_path / "t2", tokenizer_id="bytes", max_file_tokens=10
    )
    assert [f["n_tokens"] for f in tm.files if f["split"] == "train"] == [12, 1]


def test_large_vocabularies_use_uint32(tmp_path):
    write_bin(tmp_path / "a.bin", [70000, 1], 70001)
    h, ids = read_bin(tmp_path / "a.bin")
    assert h["version"] == 2 and ids.tolist() == [70000, 1]
    with pytest.raises(ValueError):
        write_bin(tmp_path / "b.bin", [-1], 10)


def test_separator_precedes_each_document(tmp_path, monkeypatch):
    import os
    from pathlib import Path

    tok = Path(os.environ["TINYLLM_FIXTURES"]) / "tok-gpt2" / "tokenizer.json"
    m = build(tmp_path, ["Hello world", "Hi"], permille=0)
    tokenize_shards(m, tok, tmp_path / "t", tokenizer_id="gpt2", doc_sep_id=50256)
    _, ids = read_bin(tmp_path / "t" / "train-00000.bin")
    assert ids.tolist() == [50256, 15496, 995, 50256, 17250]


def test_manifest_and_empty_val(tmp_path):
    m = build(tmp_path, ["one", "two"], permille=0)
    tm = tokenize_shards(m, None, tmp_path / "t", tokenizer_id="bytes")
    j = json.loads((tmp_path / "t" / "_MANIFEST.json").read_text())
    assert j["corpus_manifest_sha256"] == hashlib.sha256(m.read_bytes()).hexdigest()
    assert [f["name"] for f in j["files"]] == ["train-00000.bin", "val-00000.bin"]
    assert j["files"][1]["n_tokens"] == 0 and tm.val_bytes_per_token is None


def test_rerun_replaces_and_clears_tmp(tmp_path):
    m = build(tmp_path, ["one"], permille=0)
    (tmp_path / "t.tmp").mkdir()
    tokenize_shards(m, None, tmp_path / "t", tokenizer_id="bytes")
    assert not (tmp_path / "t.tmp").exists()
    first = (tmp_path / "t" / "_MANIFEST.json").read_bytes()
    tokenize_shards(m, None, tmp_path / "t", tokenizer_id="bytes")
    assert (tmp_path / "t" / "_MANIFEST.json").read_bytes() == first


def test_header_details_and_reader_checks(tmp_path):
    write_bin(tmp_path / "a.bin", [65535], 65536)
    raw = (tmp_path / "a.bin").read_bytes()
    assert struct.unpack("<i", raw[4:8])[0] == 1 and raw[16:1024] == b"\0" * 1008
    for bad in [b"\1" + raw[1:], raw + b"\0\0"]:
        (tmp_path / "b.bin").write_bytes(bad)
        with pytest.raises(ValueError):
            read_bin(tmp_path / "b.bin")


def test_manifest_fields(tmp_path):
    m = build(tmp_path, ["one", "two", "three"], permille=0)
    tm = tokenize_shards(m, None, tmp_path / "t", tokenizer_id="bytes")
    j = json.loads((tmp_path / "t" / "_MANIFEST.json").read_text())
    assert list(j)[:4] == [
        "tokenizer_id",
        "tokenizer_sha256",
        "vocab_size",
        "doc_sep_id",
    ]
    assert j["tokenizer_sha256"] == hashlib.sha256(b"").hexdigest()
    assert [f["n_docs"] for f in tm.files] == [3, 0]


def test_generation_config_and_val_metric(tmp_path):
    import os
    import shutil
    from pathlib import Path

    d = tmp_path / "tok"
    d.mkdir()
    shutil.copy(
        Path(os.environ["TINYLLM_FIXTURES"]) / "tok-gpt2" / "tokenizer.json",
        d / "tokenizer.json",
    )
    (d / "generation_config.json").write_text('{"eos_token_id": 50256}')
    m = build(tmp_path, ["Hello world", "Hi"], permille=1000)
    tm = tokenize_shards(m, d / "tokenizer.json", tmp_path / "t", tokenizer_id="gpt2")
    assert tm.doc_sep_id == 50256
    assert tm.val_bytes_per_token == (11 + 2) / 3  # 3 tokens, separators not counted
    assert (
        tokenize_shards(
            m, d / "tokenizer.json", tmp_path / "u", tokenizer_id="gpt2", doc_sep_id=0
        ).doc_sep_id
        == 0
    )
