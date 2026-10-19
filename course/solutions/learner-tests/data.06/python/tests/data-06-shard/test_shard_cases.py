"""Model test suite for data.06 (red then green): written before the
writer, from formats/corpus-shard.md and the chapter's pitfalls. They
import only names in contracts/py/corpus/{shard,stage}.pyi (and read the
files back with pyarrow)."""

import hashlib
import json

import pyarrow.parquet as pq
import pytest
from corpus.shard import read_shards, shard_name, split_of, write_shards
from corpus.stage import Doc


def doc(i, text, **meta):
    m = {"url": "u", "license_spdx": "CC0-1.0"}
    m.update(meta)
    return Doc(id=i, source_id=i.split(":")[0], text=text, meta=m)


def rows(out):
    m = json.loads((out / "_MANIFEST.json").read_text())
    out_rows = []
    for s in m["shards"]:
        out_rows += pq.read_table(out / s["name"]).to_pylist()
    return m, out_rows


def test_hand_example(tmp_path):
    docs = [
        doc("web:2", "Tom has a red ball.", minhash_cluster="web:10"),
        doc("web:10", "Tom has a red ball!", minhash_cluster="web:10"),
        doc("books:0", "Mia naps."),
    ]
    m, r = rows_after(tmp_path, docs, shard_rows=2, val_permille=300)
    assert [x["id"] for x in r] == ["books:0", "web:10", "web:2"]
    assert [x["split"] for x in r] == ["val", "train", "train"]
    assert [x["minhash_cluster"] for x in r] == [0, 1, 1]
    assert [s["rows"] for s in m["shards"]] == [2, 1]


def rows_after(tmp_path, docs, **kw):
    write_shards(docs, tmp_path / "c", dataset="d", version="v1", **kw)
    return rows(tmp_path / "c")


def test_split_is_a_function_of_the_text(tmp_path):
    docs = [doc(f"a:{i}", f"text {i % 50}") for i in range(200)]
    _, r = rows_after(tmp_path, docs, shard_rows=64, val_permille=300)
    side = {}
    for x in r:
        assert side.setdefault(x["text"], x["split"]) == x["split"]
        v = (
            int.from_bytes(hashlib.sha256(x["text"].encode()).digest()[:8], "big")
            % 1000
        )
        assert x["split"] == ("val" if v < 300 else "train")


def test_split_boundary():
    sha = hashlib.sha256(b"Mia naps.").digest()  # mod 1000 = 248
    assert split_of(sha, 249) == "val" and split_of(sha, 248) == "train"


def test_shard_counts_and_names(tmp_path):
    m, _ = rows_after(
        tmp_path, [doc(f"a:{i:02d}", "t") for i in range(23)], shard_rows=10
    )
    assert [s["rows"] for s in m["shards"]] == [10, 10, 3]
    assert shard_name(2, 3) == "shard-00002-of-00003.parquet"
    m, _ = rows_after(tmp_path, [], shard_rows=10)
    assert m["n_shards"] == 1 and m["shards"][0]["rows"] == 0


def test_rows_sorted_by_id_string(tmp_path):
    _, r = rows_after(
        tmp_path, [doc(i, "t") for i in ["a:2", "a:10", "a:1"]], shard_rows=2
    )
    assert [x["id"] for x in r] == ["a:1", "a:10", "a:2"]


def test_missing_license_is_an_error(tmp_path):
    with pytest.raises(ValueError):
        write_shards(
            [Doc(id="a:0", source_id="a", text="t", meta={"url": "u"})],
            tmp_path / "c",
            dataset="d",
            version="v1",
            shard_rows=1,
        )


def test_failed_write_keeps_the_old_output(tmp_path):
    write_shards(
        [doc("a:0", "t")], tmp_path / "c", dataset="d", version="v1", shard_rows=1
    )
    (tmp_path / "c.tmp").mkdir()
    write_shards(
        [doc("a:1", "u")], tmp_path / "c", dataset="d", version="v1", shard_rows=1
    )
    assert not (tmp_path / "c.tmp").exists()
    assert [d.id for d in read_shards(tmp_path / "c")] == ["a:1"]


def test_pii_totals_and_manifest_hashes(tmp_path):
    docs = [
        doc(
            "a:0",
            "x",
            pii={"email": 2, "phone": 0, "card": 0, "ip": 1, "key": 0},
            pii_redactions=3,
        )
    ]
    m, r = rows_after(tmp_path, docs + [doc("a:1", "y")], shard_rows=5)
    assert m["pii"] == {"emails": 2, "phones": 0, "cards": 0, "ips": 1, "keys": 0}
    assert [x["pii_redactions"] for x in r] == [3, 0]
    for s in m["shards"]:
        assert (
            hashlib.sha256((tmp_path / "c" / s["name"]).read_bytes()).hexdigest()
            == s["sha256"]
        )


def test_same_input_same_bytes(tmp_path):
    docs = [doc(f"a:{i}", f"t{i}") for i in range(30)]
    write_shards(docs, tmp_path / "a", dataset="d", version="v1", shard_rows=8)
    write_shards(docs[::-1], tmp_path / "b", dataset="d", version="v1", shard_rows=8)
    for p in (tmp_path / "a").iterdir():
        assert p.read_bytes() == (tmp_path / "b" / p.name).read_bytes()


def test_columns_and_defaults(tmp_path):
    docs = [
        doc("a:0", "é!"),
        doc("a:1", "x", lang="fr"),
        Doc(
            id="a:2", source_id="a", text="y", meta={"url": "u", "license_spdx": "MIT"}
        ),
    ]
    _, r = rows_after(tmp_path, docs, shard_rows=5)
    assert [x["n_chars"] for x in r] == [2, 1, 1]
    assert [x["lang"] for x in r] == ["und", "fr", "und"]
    assert [x["minhash_cluster"] for x in r] == [0, 1, 2]


def test_counts_are_recorded(tmp_path):
    write_shards(
        [doc("a:0", "t", pii={"email": 1}), doc("a:1", "u", pii={"email": 2})],
        tmp_path / "c",
        dataset="d",
        version="v1",
        shard_rows=1,
        filters={"lang": 3},
        dedup={"near_dropped": 4},
    )
    m = json.loads((tmp_path / "c" / "_MANIFEST.json").read_text())
    assert m["filters"] == {"lang": 3} and m["pii"]["emails"] == 3
    assert (
        m["dedup"]["near_dropped"] == 4
        and m["dedup"]["num_perm"] == 128
        and m["dedup"]["ngram"] == 13
    )


def test_repeated_ids_and_tampering(tmp_path):
    with pytest.raises(ValueError):
        write_shards(
            [doc("a:0", "t"), doc("a:0", "u")],
            tmp_path / "x",
            dataset="d",
            version="v1",
            shard_rows=1,
        )
    write_shards(
        [doc("a:0", "t"), doc("a:1", "Mia naps.")],
        tmp_path / "c",
        dataset="d",
        version="v1",
        shard_rows=5,
        val_permille=300,
    )
    assert [d.id for d in read_shards(tmp_path / "c", split="val")] == ["a:1"]
    p = tmp_path / "c" / "shard-00000-of-00001.parquet"
    t = pq.read_table(p)
    pq.write_table(t.slice(0, 1), p)
    with pytest.raises(ValueError):
        list(read_shards(tmp_path / "c"))
