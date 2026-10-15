# contracts/py/corpus/shard.pyi (data.06): Parquet shards, the manifest, and
# the document-hash train/val split
# chapter: content/tracks/data-engineering/05-corpus-pipeline/06-parquet-shards-and-manifest.md
#
# The files are fixed by formats/corpus-shard.md and
# formats/corpus-manifest.schema.json; this module writes and reads them
# with pyarrow (allowed for `corpus` in allowed-deps.toml).
#
# Rows. One row per document, columns in SCHEMA order:
#   id, text, source_id      the Doc's fields
#   url, license_spdx        meta["url"], meta["license_spdx"] (KeyError
#                            surfaces as ValueError naming the document)
#   lang                     meta.get("lang", "und")
#   n_chars                  len(text) in code points
#   sha256                   SHA-256 of text's UTF-8 bytes (32 raw bytes)
#   minhash_cluster          rows sharing meta["minhash_cluster"] (data.04;
#                            a document without it is its own cluster) form
#                            one cluster; its value is the smallest row
#                            index (global, 0-based, in output order) of
#                            the cluster's rows
#   pii_redactions           meta.get("pii_redactions", 0) (data.05)
#   split                    split_of(sha256, val_permille)
#
# Order and files. Rows are sorted by id (Python str order, which is UTF-8
# byte order) and cut into shards of shard_rows rows; n = max(1,
# ceil(n_docs / shard_rows)) shards named shard_name(i, n), so an empty
# corpus still has one (empty) shard. Inside a shard, a row group holds
# consecutive rows whose UTF-8 text bytes sum to at most row_group_bytes (a
# longer document gets a group of its own). Compression zstd.
#
# Manifest. _MANIFEST.json, keys in this order: dataset, version, n_docs,
# n_shards, shards [{name, sha256 (of the file bytes), rows}], filters,
# dedup {exact_dropped, near_dropped, decontaminated, jaccard_threshold,
# num_perm, bands, ngram}, pii {emails, phones, cards, ips, keys} (summed
# from each document's data.05 manifest_counts(meta["pii"])), ledger_ref,
# config_sha256; written as json.dumps(m, indent=2, ensure_ascii=False)
# plus "\n". Every file is byte-identical for the same input, whatever the
# input order.
#
# Atomicity. Everything is written under out + ".tmp" (removed first if a
# crashed run left it) and renamed to out at the end, replacing an older out.
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional

import pyarrow as pa

from corpus.stage import Doc

SCHEMA: pa.Schema
DEFAULT_ROW_GROUP_BYTES: int  # 64 MiB
DEDUP_DEFAULTS: dict[str, Any]  # {"exact_dropped": 0, "near_dropped": 0, "decontaminated": 0, "jaccard_threshold": 0.8, "num_perm": 128, "bands": 16, "ngram": 13}

def split_of(sha256: bytes, val_permille: int) -> str:
    """"val" when int.from_bytes(sha256[:8], "big") % 1000 < val_permille,
    else "train". ValueError unless 0 <= val_permille <= 1000 and
    len(sha256) == 32."""

def shard_name(i: int, n: int) -> str:
    """f"shard-{i:05d}-of-{n:05d}.parquet". ValueError unless 0 <= i < n."""

def write_shards(
    docs: Iterable[Doc],
    out: Path,
    *,
    dataset: str,
    version: str,
    shard_rows: int,
    val_permille: int = 5,
    row_group_bytes: int = ...,
    filters: Optional[Mapping[str, int]] = None,
    dedup: Optional[Mapping[str, Any]] = None,
    ledger_ref: str = "corpus/LEDGER.jsonl",
    config_sha256: str = ...,
) -> dict[str, Any]:
    """Write the shards and _MANIFEST.json into out; return the manifest.
    filters: dropped counts by filter name (default {}); dedup: overrides of
    DEDUP_DEFAULTS. ValueError for shard_rows < 1, row_group_bytes < 1, a
    repeated id, or a document without url or license_spdx."""

def read_shards(out: Path, split: Optional[str] = None) -> Iterator[Doc]:
    """The documents of the shards listed in out/_MANIFEST.json, in manifest
    and row order (only split's rows when split is given). meta holds url,
    license_spdx, lang, n_chars, sha256 (hex), minhash_cluster,
    pii_redactions, split, shard (file name), and row (index in the shard).
    ValueError when a shard's sha256 or row count differs from the manifest."""
