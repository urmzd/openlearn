# Parquet shards, manifest, and the document-hash split

## Overview

| | |
|---|---|
| **Module** | `data.06` · build · Python · Pass 3 · 3 to 4 h |
| **Solution** | [`python/corpus/shard.py`](../../../../course/solutions/python/corpus/shard.py): `split_of`, `shard_name`, `write_shards`, `read_shards`, and the tables `SCHEMA`, `DEDUP_DEFAULTS`, `DEFAULT_ROW_GROUP_BYTES` |
| **Contract** | [`course/contracts/py/corpus/shard.pyi`](../../../../course/contracts/py/corpus/shard.pyi) · files: [`formats/corpus-shard.md`](../../../../course/contracts/formats/corpus-shard.md), [`formats/corpus-manifest.schema.json`](../../../../course/contracts/formats/corpus-manifest.schema.json) · allowed library: `pyarrow` ([`allowed-deps.toml`](../../../../course/contracts/allowed-deps.toml)) |
| **Needs** | `data.02` `Doc` · `data.04` the `minhash_cluster` tag · `data.05` the PII counts and `manifest_counts` · reading: [Storage & Warehousing](../02-storage-warehousing/) (columnar files) |
| **Used by** | `data.07` tokenizes the shards through `read_shards` · `data.08` joins every row with the ledger |
| **Optional depth** | Apache Parquet [file format specification](https://parquet.apache.org/docs/file-format/) (free); Abadi et al., *The Design and Implementation of Modern Column-Oriented Database Systems* (2013), chapters 1 to 3 (free); the FineWeb datatrove writers (going further) |

## Key Takeaways

- The corpus becomes files with a contract: Parquet shards in a fixed column order and type, sorted by id, cut every `shard_rows` rows, and a manifest that records each file's hash and every count the pipeline dropped.
- A document's split is a function of its text alone, the first 8 bytes of its SHA-256 modulo 1000, so exact copies can never straddle train and val and the split never moves when documents are added or reordered.
- The same documents in any order give byte-identical shards and manifest, and a failed write never damages the previous output.

## How to read this chapter

Sections 1 to 3 build the shard layout, the document-hash split, and the manifest from first principles and work three documents by hand. Section 4 fixes the interface and lists the cases a correct solution must handle, including the model test suite. Section 5 walks through `shard.py` and explains why each choice is the right one. Try to predict each step of the solution before reading it.

The solution depends on `pyarrow` (`dependencies = ["numpy>=2", "pyarrow>=18", "zstandard>=0.23"]` in `python/pyproject.toml`); [`allowed-deps.toml`](../../../../course/contracts/allowed-deps.toml) allows it for `corpus` units only.

---

## 1. Why now

After `data.05` the cleaned corpus exists only as a Python iterator. Nothing downstream can use it: the tokenizer (`data.07`) needs files it can read in a fixed order, the training loader needs a validation split that never overlaps training, the ledger check (`data.08`) needs every row's source and license, and the durable `CorpusBuild` (`data.09`) needs an output whose hash says whether a retried run produced the same thing. A split made by `random()` would move documents between train and val on every run, and an exact copy that sits in both makes the validation loss lie. This module writes the corpus as sorted, hash-split Parquet shards with a manifest, atomically.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $N$ | number of documents | `int` |
| $R$ | rows per shard (`shard_rows`) | `int` |
| $n = \max(1, \lceil N / R \rceil)$ | number of shards | `int` |
| $h(d)$ | SHA-256 of the UTF-8 bytes of document $d$'s text | 32 bytes |
| $u(d)$ | the first 8 bytes of $h(d)$ as a big-endian unsigned integer | `int` in $[0, 2^{64})$ |
| $v$ | `val_permille`: the share of val documents in thousandths | `int` in $[0, 1000]$ |
| $B$ | `row_group_bytes`, the text budget of a row group | `int` |

**Columnar files.** A Parquet file stores each column contiguously, compressed (zstd here), in **row groups**: blocks of rows that a reader loads together. Reading only `source_id` and `license_spdx` (what `data.08` needs) touches a fraction of the bytes; reading `text` in order (what `data.07` needs) streams one row group at a time, so $B$ bounds a reader's memory. The schema is fixed in `formats/corpus-shard.md`: `id, text, source_id, url, license_spdx, lang, n_chars (int32), sha256 (fixed_size_binary[32]), minhash_cluster (int64), pii_redactions (int32), split`, in that order.

**Sort, then cut.** All rows are sorted by `id` as strings (Python's `str` order, which equals UTF-8 byte order): `"a:10"` comes before `"a:2"`. Shard $i$ holds rows $iR$ to $(i+1)R - 1$ and is named `shard-{i:05d}-of-{n:05d}.parquet`; an empty corpus still has one empty shard, so a reader never finds a manifest without a file. Inside a shard, consecutive rows go into one row group while their text bytes sum to at most $B$; a longer document gets a group of its own.

**The document-hash split.** A document is val when

$$u(d) \bmod 1000 < v,$$

else train. The hash depends only on the text, so the decision is the same in every run, on every machine, in any order, and for every copy of the same text. Over many documents $u \bmod 1000$ is close to uniform, so the val share is close to $v / 1000$: with 4000 documents and $v = 100$, the val count is Binomial(4000, 0.1), mean 400, standard deviation 19.

**Columns from upstream.** `minhash_cluster`: rows sharing `meta["minhash_cluster"]` (the root id `data.04` wrote; a document without it is its own cluster) form a cluster, and its value is the **smallest row index** of the cluster in output order, a number to group by without the ids. `pii_redactions` is `meta["pii_redactions"]`; the manifest's `pii` object sums `data.05`'s `manifest_counts(meta["pii"])` over the documents. `lang` defaults to `und` (undetermined), `n_chars` counts code points. A document without `url` or `license_spdx` is an error: the ledger could not trace it.

**The manifest.** `_MANIFEST.json` lists every shard with its file's SHA-256 and row count, the dropped counts of every stage (`filters` as given, `dedup` as `DEDUP_DEFAULTS` overridden by the caller), the `pii` totals, the ledger path, and the config's hash, keys in the format's order, written as `json.dumps(m, indent=2, ensure_ascii=False)` plus a newline. Two writers that follow the contract write the same bytes.

**Atomic output.** Write everything into `out.tmp` (removing a stale one first), then rename it to `out`, replacing an older `out`. A crash before the rename leaves the previous output untouched; a crash after it leaves the new one complete. A rerun of the durable activity (`data.09`) sees either the old or the new version, never half of each.

## 3. Worked example by hand

Three documents, `val_permille` 300, `shard_rows` 2. `web:2` is a near duplicate of `web:10`, kept with `drop=False`, so both carry `minhash_cluster = "web:10"`:

| id | text | first 8 bytes of SHA-256 | as an integer, mod 1000 | split |
|---|---|---|---|---|
| `web:2` | `Tom has a red ball.` | `ba7f5216fe451c81` | 13438550071805549697 → 697 | train |
| `web:10` | `Tom has a red ball!` | `39777282a7bff242` | 4140904287876149826 → 826 | train |
| `books:0` | `Mia naps.` | `fcd7f3a1b48b4158` | 18219298693394940248 → 248 | val (248 < 300) |

**Sort.** By id as strings: `books:0` < `web:10` < `web:2` (`b` before `w`; then `1` before `2`, character by character). Rows 0, 1, 2.

**Clusters.** `web:10` and `web:2` share the tag, and their smallest row is 1; `books:0` is alone at row 0. Column: `[0, 1, 1]`. `n_chars`: `[9, 19, 19]`.

**Shards.** $n = \lceil 3 / 2 \rceil = 2$: `shard-00000-of-00002.parquet` holds rows 0 and 1, `shard-00001-of-00002.parquet` holds row 2.

**The boundary.** `split_of(h("Mia naps."), 248)` is train: $248 < 248$ is false. That strict inequality is what makes $v = 0$ an empty val split and $v = 1000$ all val.

`read_shards` gives the same three rows back, in that order.

## 4. The interface

```python
# python/corpus/shard.py (the full contract is contracts/py/corpus/shard.pyi)
SCHEMA: pa.Schema
DEFAULT_ROW_GROUP_BYTES: int                        # 64 MiB
DEDUP_DEFAULTS: dict[str, Any]

def split_of(sha256: bytes, val_permille: int) -> str
def shard_name(i: int, n: int) -> str
def write_shards(docs, out: Path, *, dataset, version, shard_rows, val_permille=5,
                 row_group_bytes=DEFAULT_ROW_GROUP_BYTES, filters=None, dedup=None,
                 ledger_ref="corpus/LEDGER.jsonl", config_sha256="0" * 64) -> dict
def read_shards(out: Path, split: str | None = None) -> Iterator[Doc]
```

`write_shards` has to read every document before writing (it sorts), so it is the end of the streaming part of the pipeline. The solution writes with `pq.ParquetWriter(path, SCHEMA, compression="zstd")` and one `write_table` per row group (`row_group_size` set to the group's length, so pyarrow does not split it again).

### What a correct solution must handle

| Case | Kind of check | What must hold | Why it matters downstream |
|---|---|---|---|
| The hand example | unit | the section 3 order, splits, clusters, character counts, and shards | the definitions are exactly the ones above |
| Schema, compression, manifest | conformance | exact Arrow schema, zstd, manifest valid against the schema, file hashes and row counts true, no stray files | every reader relies on the contract |
| Sort order | unit | string order, not numeric; `n_chars` in code points; `und` default; untagged rows are their own cluster | rows land in the same shard on every machine |
| The split follows the hash | property | the big-endian rule on every row; exact copies on one side; split stable under reordering and subsets | no val text leaks into training |
| The val share | statistical | 4000 documents at 100 permille: val within 4 standard deviations of 400; bad arguments raise | the configured share is what you get |
| Shard counts and names | boundary | 23 rows at 10 per shard: 10, 10, 3; names; an empty corpus has one empty shard | readers can always open shard 0 |
| Row groups by text bytes | boundary | groups fit the byte budget; a long document gets its own group, also first in a shard | bounded reader memory |
| Determinism | property | shuffled input gives byte-identical files | `data.09` compares output hashes across retries |
| Cluster column | unit | `data.04`'s tags become the smallest row index | the column analysts group by |
| PII totals | unit | `data.05`'s counts summed into the manifest; the per-row column | the datasheet's numbers |
| Counts and key order | unit | filters as given, dedup over the defaults, the format's key order and serialization | two writers, one manifest |
| Bad input | boundary | missing url or license, repeated id, `shard_rows` 0: errors, nothing written | the ledger can trace every row |
| Atomic replace | fault | a failed run leaves the old output; a stale `.tmp` is cleared | retried activities are safe |
| Reading back | unit | manifest and row order, split filter, meta columns, a rewritten shard raises | `data.07` and `data.08` read through it |

**The model test suite.** [`test_shard_cases.py`](../../../../course/solutions/learner-tests/data.06/python/tests/data-06-shard/test_shard_cases.py) covers the hand example; the split as a function of the text (with the big-endian rule written out); the 248 boundary; shard counts and names, empty corpus included; string order of ids; a missing license raising; a failed write keeping the old output; PII totals and file hashes; same input, same bytes; column defaults; recorded counts; repeated ids and a tampered shard. The split case is the one that catches pitfall 2.

## 5. The solution, step by step

### 5.1 From first principles

1. **Downstream needs reproducible files, so the output must be a function of the set of documents, not their arrival order.** That forces a total sort before cutting. Sorting by `id` as Python strings (code-point order, equal to UTF-8 byte order) is the one order every implementation can agree on without parsing ids.
2. **The split must be stable under reordering, subsets, and duplication.** Any function of position or a random draw fails at least one of these. A function of the text alone passes all three, and a cryptographic hash makes it close to uniform: $u(d) \bmod 1000 < v$. Big-endian is fixed by the contract so ports agree.
3. **The writer has to sort, so it must hold every document's columns anyway.** It builds the columns in one pass (`_rows`), validating as it goes, so a bad document fails before any file is written.
4. **Reader memory is bounded by row-group size, so cut groups by text bytes, before overflow.** A group closes when adding the next document would exceed $B$; a single document larger than $B$ gets a group to itself.
5. **The manifest is the output's identity.** It records each shard's file hash and row count, and contains nothing that changes between runs (no timestamps, fixed key order, fixed serialization), so its own hash is a retry check.
6. **A crash must never leave half an output that looks whole.** Everything goes into `out.tmp` and is swapped in by a rename at the end.
7. **The reader should not trust bytes it did not verify.** `read_shards` rehashes each file against the manifest before yielding a row.

### 5.2 The code, annotated

The split and names ([`shard.py`](../../../../course/solutions/python/corpus/shard.py)):

```python
def split_of(sha256: bytes, val_permille: int) -> str:
    if not 0 <= val_permille <= 1000:
        raise ValueError(f"val_permille must be in 0..1000, got {val_permille}")
    if len(sha256) != 32:
        raise ValueError(f"a sha256 digest has 32 bytes, got {len(sha256)}")
    # strict <: v = 0 is no val at all, v = 1000 is all val
    return "val" if int.from_bytes(sha256[:8], "big") % 1000 < val_permille else "train"


def shard_name(i: int, n: int) -> str:
    if not 0 <= i < n:
        raise ValueError(f"shard index {i} is not in 0..{n - 1}")
    return f"shard-{i:05d}-of-{n:05d}.parquet"   # zero-padded: lexical order = numeric order
```

Building the columns: sort, reject repeats, require provenance, assign clusters by first row:

```python
def _rows(docs, val_permille):
    items = sorted(docs, key=lambda d: d.id)          # str order: "a:10" < "a:2"
    for a, b in zip(items, items[1:]):
        if a.id == b.id:                              # after sorting, repeats are adjacent
            raise ValueError(f"document id {a.id!r} appears twice")
    cols: dict[str, list] = {name: [] for name in SCHEMA.names}
    pii = manifest_counts({})                         # all kinds at 0
    first_row: dict[str, int] = {}
    for row, d in enumerate(items):
        try:
            url, lic = d.meta["url"], d.meta["license_spdx"]   # no default: untraceable rows are errors
        except KeyError as e:
            raise ValueError(f"document {d.id!r} has no meta[{e.args[0]!r}]") from None
        digest = hashlib.sha256(d.text.encode("utf-8")).digest()
        cluster = d.meta.get("minhash_cluster", d.id)  # untagged: its own cluster
        first_row.setdefault(cluster, row)             # rows are in output order, so the first seen is the smallest
        # ... append id, text, source_id, url, license_spdx ...
        cols["lang"].append(d.meta.get("lang", "und"))
        cols["n_chars"].append(len(d.text))            # code points, not bytes
        cols["sha256"].append(digest)
        cols["minhash_cluster"].append(first_row[cluster])
        cols["pii_redactions"].append(int(d.meta.get("pii_redactions", 0)))
        cols["split"].append(split_of(digest, val_permille))
        for k, n in manifest_counts(d.meta.get("pii")).items():
            pii[k] += n
    return cols, pii
```

One shard file, row groups cut before they overflow:

```python
def _write_one(path, cols, lo, hi, row_group_bytes):
    table = pa.table({k: v[lo:hi] for k, v in cols.items()}, schema=SCHEMA)
    groups: list[tuple[int, int]] = []
    start, size = 0, 0
    for i, text in enumerate(cols["text"][lo:hi]):
        n = len(text.encode("utf-8"))
        if i > start and size + n > row_group_bytes:   # i > start: a group is never empty
            groups.append((start, i - start))
            start, size = i, 0
        size += n
    groups.append((start, hi - lo - start))           # the last (possibly empty-shard) group
    with pq.ParquetWriter(path, SCHEMA, compression="zstd") as w:
        for s, length in groups:
            # row_group_size = the group's length, so pyarrow does not re-split it
            w.write_table(table.slice(s, length), row_group_size=max(1, length))
```

The writer: columns, shards into a temporary directory, the manifest, then the swap:

```python
    cols, pii = _rows(docs, val_permille)                  # all validation happens here, before any file
    n_docs = len(cols["id"])
    n = max(1, -(-n_docs // shard_rows))                   # ceil, and at least one (empty) shard
    out = Path(out)
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)                 # a stale .tmp from a crashed run
    tmp.mkdir(parents=True)
    shards = []
    for i in range(n):
        lo, hi = i * shard_rows, min(n_docs, (i + 1) * shard_rows)
        name = shard_name(i, n)
        _write_one(tmp / name, cols, lo, hi, row_group_bytes)
        shards.append({"name": name, "sha256": _file_sha256(tmp / name), "rows": hi - lo})
    manifest = {
        "dataset": dataset, "version": version, "n_docs": n_docs, "n_shards": n,
        "shards": shards,
        "filters": dict(filters or {}),
        "dedup": {**DEDUP_DEFAULTS, **dict(dedup or {})},  # defaults, overridden by the caller
        "pii": pii,
        "ledger_ref": ledger_ref,
        "config_sha256": config_sha256,                    # no timestamp: same input, same bytes
    }
    (tmp / MANIFEST).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    if out.exists():
        shutil.rmtree(out)
    os.rename(tmp, out)                                    # the new output appears whole
    return manifest
```

The reader verifies before it yields:

```python
    for s in manifest["shards"]:                           # manifest order, then row order
        path = out / s["name"]
        if _file_sha256(path) != s["sha256"]:
            raise ValueError(f"{path}: sha256 differs from the manifest")
        table = pq.read_table(path, schema=SCHEMA)
        if table.num_rows != s["rows"]:
            raise ValueError(...)
        for r, row in enumerate(table.to_pylist()):
            if split is not None and row["split"] != split:
                continue
            # meta: url, license_spdx, lang, n_chars, minhash_cluster, pii_redactions, split,
            # plus sha256 (hex), shard, row: enough for data.08 to trace every row
            # ...
```

Full solution: [`shard.py`](../../../../course/solutions/python/corpus/shard.py).

### 5.3 Why this solution

- **Cost.** Sorting is $O(N \log N)$; hashing and writing are linear in the corpus bytes. Memory holds every column at once, $O(\text{corpus})$: the price of a total sort in one process. `read_shards` rehashes each file ($O$ file size) and then loads one shard at a time.
- **A hash split over a random split.** `random() < 0.005` changes with the seed and the order, and puts an exact copy on both sides about 1% of the time per pair. Splitting by id hash would be stable but lets copies under different ids straddle the split. Hashing the text fixes both.
- **Rows sorted by id over arrival order.** Arrival order depends on fetch concurrency and worker count; sorting makes the output a function of the documents only, which `data.09` relies on to compare retries by hash.
- **Holding everything in memory** is the accepted tradeoff at this scale. A production writer (datatrove) streams into rotating files and sorts externally; the contract (sorted, hash-split, manifest last) would stay the same.
- **The swap is "remove, then rename".** `shutil.rmtree(out)` followed by `os.rename(tmp, out)` is not a single atomic step: a crash between the two leaves no `out` but a complete `out.tmp`. It never leaves a half-written `out`, which is the property that matters; a rerun rebuilds from scratch. A fully atomic swap would rename `out` aside first, or point a symlink at a versioned directory.

### 5.4 Explain it back

1. Why is the split computed from the text's hash and not from the document id?
   <details><summary>Answer</summary>An exact copy of a validation story can arrive under another id (another source, another position). With an id-based split it could land in train while the original is in val, so validation loss measures recall. A text hash puts every copy on the same side.</details>
2. Why does `"a:10"` sort before `"a:2"`, and why not sort numerically?
   <details><summary>Answer</summary>Strings compare character by character and `"1" < "2"`. Numeric sorting would require parsing ids, which are opaque strings by contract; string order is the one order every implementation reproduces exactly.</details>
3. How does `first_row.setdefault(cluster, row)` produce the smallest row index of each cluster?
   <details><summary>Answer</summary>Rows are enumerated in output order, so the first time a cluster tag is seen is at its smallest row; `setdefault` stores only that first value and returns it for every later member.</details>
4. Why is there no timestamp in the manifest?
   <details><summary>Answer</summary>The manifest's bytes are the identity of the output; `data.09` compares hashes to decide whether a retried run produced the same corpus. A timestamp would make every run differ even when every shard is identical.</details>
5. Why does `_write_one` check `i > start` before closing a group?
   <details><summary>Answer</summary>So a group always holds at least one row. A document longer than the budget then becomes a group of its own instead of producing an empty group in front of it, including when it is the first row of a shard.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. keeping arrival order, or sorting ids numerically | shards differ between runs, or from every other implementation | `sorted(docs, key=lambda d: d.id)` on strings |
| 2. splitting by the id, or by a random draw | an exact copy under another id lands in val while the original trains | `split_of` hashes the text |
| 3. cutting a row group only after it overflows | groups exceed the budget; a reader's memory is unbounded | closes the group when `size + n > row_group_bytes`, before adding |
| 4. defaulting a missing license to "" | rows the ledger cannot trace reach training | `d.meta["url"]` and `d.meta["license_spdx"]` without defaults; `KeyError` becomes `ValueError` |
| 5. writing straight into `out` | a crash leaves half a corpus that looks complete | writes into `out.tmp`, renames at the end |
| 6. reading the first 8 bytes little-endian | a different, still uniform split: every other implementation disagrees | `int.from_bytes(sha256[:8], "big")` |
| 7. a timestamp in the manifest | the output hash changes on every run | the manifest holds only content-derived fields |
| 8. trusting the shard bytes in `read_shards` | a rewritten shard feeds the tokenizer silently | rehashes every file against the manifest first |

## 7. Where it's used next


| Direction | Module | How it uses this |
|---|---|---|
| Back | `data.02` | `Doc`, and `meta["lang"]` from the language filter |
| Back | `data.04` | `meta["minhash_cluster"]` becomes the row-index column |
| Back | `data.05` | `pii_redactions`, and the per-kind counts renamed by `manifest_counts` |
| Forward | `data.07` | `read_shards(out, split)` feeds the tokenizer, train then val, in manifest order |
| Forward | `data.08` | every row's `source_id` and `license_spdx` are checked against the ledger; the datasheet reads the manifest |
| Forward | `data.09` | `CorpusBuild`'s `shard` activity; its output hash is the retry check |

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `write_shards` | datatrove `ParquetWriter`, Hugging Face `datasets` | streaming writers that never hold the corpus, file rotation by size, remote filesystems | datatrove `src/datatrove/pipeline/writers/parquet.py` |
| the manifest | Delta Lake and Apache Iceberg | transaction logs with snapshots, schema evolution, time travel | the Iceberg table spec |
| `split_of` | Hugging Face `datasets` train/test splits, TFDS | deterministic splits by hash of a key; named splits in metadata | `tensorflow_datasets/core/splits.py` |
| atomic rename | object-store commit protocols | multi-file commits without rename (S3 has none): manifest files written last | the Iceberg commit protocol |
