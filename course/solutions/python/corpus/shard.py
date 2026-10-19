"""Parquet shards, the manifest, and the document-hash split (data.06).

The cleaned corpus leaves Python's memory here: sorted by id, cut into
fixed-size Parquet shards (columnar, zstd), and described by a manifest
that records every file's hash and every count the pipeline dropped. A
document goes to val when its content hash says so, so the split never
depends on the order documents arrived in, and exact duplicates can never
straddle it.

Contract: contracts/py/corpus/shard.pyi. Files: formats/corpus-shard.md.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional

import pyarrow as pa
import pyarrow.parquet as pq

from corpus.pii import manifest_counts
from corpus.stage import Doc

SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("text", pa.string()),
        ("source_id", pa.string()),
        ("url", pa.string()),
        ("license_spdx", pa.string()),
        ("lang", pa.string()),
        ("n_chars", pa.int32()),
        ("sha256", pa.binary(32)),
        ("minhash_cluster", pa.int64()),
        ("pii_redactions", pa.int32()),
        ("split", pa.string()),
    ]
)
DEFAULT_ROW_GROUP_BYTES = 64 * 1024 * 1024
DEDUP_DEFAULTS: dict[str, Any] = {
    "exact_dropped": 0,
    "near_dropped": 0,
    "decontaminated": 0,
    "jaccard_threshold": 0.8,
    "num_perm": 128,
    "bands": 16,
    "ngram": 13,
}
MANIFEST = "_MANIFEST.json"


def split_of(sha256: bytes, val_permille: int) -> str:
    """ "val" when the first 8 bytes, big-endian, mod 1000 < val_permille."""
    if not 0 <= val_permille <= 1000:
        raise ValueError(f"val_permille must be in 0..1000, got {val_permille}")
    if len(sha256) != 32:
        raise ValueError(f"a sha256 digest has 32 bytes, got {len(sha256)}")
    return "val" if int.from_bytes(sha256[:8], "big") % 1000 < val_permille else "train"


def shard_name(i: int, n: int) -> str:
    """f"shard-{i:05d}-of-{n:05d}.parquet"."""
    if not 0 <= i < n:
        raise ValueError(f"shard index {i} is not in 0..{n - 1}")
    return f"shard-{i:05d}-of-{n:05d}.parquet"


def _file_sha256(path: Path) -> str:
    """Hex SHA-256 of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _rows(
    docs: Iterable[Doc], val_permille: int
) -> tuple[dict[str, list], dict[str, int]]:
    """Columns of the sorted rows, and the PII totals for the manifest."""
    items = sorted(docs, key=lambda d: d.id)
    for a, b in zip(items, items[1:]):
        if a.id == b.id:
            raise ValueError(f"document id {a.id!r} appears twice")
    cols: dict[str, list] = {name: [] for name in SCHEMA.names}
    pii = manifest_counts({})
    first_row: dict[str, int] = {}
    for row, d in enumerate(items):
        try:
            url, lic = d.meta["url"], d.meta["license_spdx"]
        except KeyError as e:
            raise ValueError(f"document {d.id!r} has no meta[{e.args[0]!r}]") from None
        digest = hashlib.sha256(d.text.encode("utf-8")).digest()
        cluster = d.meta.get("minhash_cluster", d.id)
        first_row.setdefault(cluster, row)
        cols["id"].append(d.id)
        cols["text"].append(d.text)
        cols["source_id"].append(d.source_id)
        cols["url"].append(url)
        cols["license_spdx"].append(lic)
        cols["lang"].append(d.meta.get("lang", "und"))
        cols["n_chars"].append(len(d.text))
        cols["sha256"].append(digest)
        cols["minhash_cluster"].append(first_row[cluster])
        cols["pii_redactions"].append(int(d.meta.get("pii_redactions", 0)))
        cols["split"].append(split_of(digest, val_permille))
        for k, n in manifest_counts(d.meta.get("pii")).items():
            pii[k] += n
    return cols, pii


def _write_one(
    path: Path, cols: dict[str, list], lo: int, hi: int, row_group_bytes: int
) -> None:
    """Rows lo..hi-1 into one Parquet file, row groups cut by text bytes."""
    table = pa.table({k: v[lo:hi] for k, v in cols.items()}, schema=SCHEMA)
    groups: list[tuple[int, int]] = []
    start, size = 0, 0
    for i, text in enumerate(cols["text"][lo:hi]):
        n = len(text.encode("utf-8"))
        if i > start and size + n > row_group_bytes:
            groups.append((start, i - start))
            start, size = i, 0
        size += n
    groups.append((start, hi - lo - start))
    with pq.ParquetWriter(path, SCHEMA, compression="zstd") as w:
        for s, length in groups:
            w.write_table(table.slice(s, length), row_group_size=max(1, length))


# fmt: off
def write_shards(
    docs: Iterable[Doc], out: Path, *, dataset: str, version: str, shard_rows: int,
    val_permille: int = 5, row_group_bytes: int = DEFAULT_ROW_GROUP_BYTES,
    filters: Optional[Mapping[str, int]] = None, dedup: Optional[Mapping[str, Any]] = None,
    ledger_ref: str = "corpus/LEDGER.jsonl", config_sha256: str = "0" * 64,
) -> dict[str, Any]:
    """Write the shards and _MANIFEST.json into out; return the manifest."""
    if shard_rows < 1 or row_group_bytes < 1:
        raise ValueError("shard_rows and row_group_bytes must be >= 1")
    cols, pii = _rows(docs, val_permille)
    n_docs = len(cols["id"])
    n = max(1, -(-n_docs // shard_rows))
    out = Path(out)
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    shards = []
    for i in range(n):
        lo, hi = i * shard_rows, min(n_docs, (i + 1) * shard_rows)
        name = shard_name(i, n)
        _write_one(tmp / name, cols, lo, hi, row_group_bytes)
        shards.append(
            {"name": name, "sha256": _file_sha256(tmp / name), "rows": hi - lo}
        )
    manifest = {
        "dataset": dataset,
        "version": version,
        "n_docs": n_docs,
        "n_shards": n,
        "shards": shards,
        "filters": dict(filters or {}),
        "dedup": {**DEDUP_DEFAULTS, **dict(dedup or {})},
        "pii": pii,
        "ledger_ref": ledger_ref,
        "config_sha256": config_sha256,
    }
    (tmp / MANIFEST).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    if out.exists():
        shutil.rmtree(out)
    os.rename(tmp, out)
    return manifest

# fmt: on


def read_shards(out: Path, split: Optional[str] = None) -> Iterator[Doc]:
    """The documents of the manifest's shards, in manifest and row order."""
    out = Path(out)
    manifest = json.loads((out / MANIFEST).read_text())
    for s in manifest["shards"]:
        path = out / s["name"]
        if _file_sha256(path) != s["sha256"]:
            raise ValueError(f"{path}: sha256 differs from the manifest")
        table = pq.read_table(path, schema=SCHEMA)
        if table.num_rows != s["rows"]:
            raise ValueError(
                f"{path}: {table.num_rows} rows, the manifest says {s['rows']}"
            )
        for r, row in enumerate(table.to_pylist()):
            if split is not None and row["split"] != split:
                continue
            meta = {
                k: row[k]
                for k in (
                    "url",
                    "license_spdx",
                    "lang",
                    "n_chars",
                    "minhash_cluster",
                    "pii_redactions",
                    "split",
                )
            }
            meta.update(sha256=row["sha256"].hex(), shard=s["name"], row=r)
            yield Doc(
                id=row["id"], source_id=row["source_id"], text=row["text"], meta=meta
            )
