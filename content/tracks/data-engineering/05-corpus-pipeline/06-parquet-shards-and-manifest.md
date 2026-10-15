<!-- ol:module data.06 -->
# Parquet shards, manifest, and the document-hash split

## Overview

| | |
|---|---|
| **Module** | `data.06` · build · Python · Pass 3 · 3 to 4 h |
| **You build** | `python/corpus/shard.py`: `split_of`, `shard_name`, `write_shards`, `read_shards`, and the tables `SCHEMA`, `DEDUP_DEFAULTS`, `DEFAULT_ROW_GROUP_BYTES` |
| **Contract** | [`course/contracts/py/corpus/shard.pyi`](../../../../course/contracts/py/corpus/shard.pyi) · files: [`formats/corpus-shard.md`](../../../../course/contracts/formats/corpus-shard.md), [`formats/corpus-manifest.schema.json`](../../../../course/contracts/formats/corpus-manifest.schema.json) · allowed library: `pyarrow` ([`allowed-deps.toml`](../../../../course/contracts/allowed-deps.toml)) |
| **Tests** | `course/tests/data.06/` (what they check: section 4) |
| **Needs** | `data.02` `Doc` · `data.04` the `minhash_cluster` tag · `data.05` the PII counts and `manifest_counts` (or `--ref-deps`) · reading: [Storage & Warehousing](../02-storage-warehousing/) (columnar files) |
| **Used by** | `data.07` tokenizes the shards through `read_shards` · `data.08` joins every row with the ledger |
| **Milestone** | `MS-corpus` |
| **Optional depth** | Apache Parquet [file format specification](https://parquet.apache.org/docs/file-format/) (free); Abadi et al., *The Design and Implementation of Modern Column-Oriented Database Systems* (2013), chapters 1 to 3 (free); the FineWeb datatrove writers (going further) |

## Key Takeaways

- The corpus becomes files with a contract: Parquet shards in a fixed column order and type, sorted by id, cut every `shard_rows` rows, and a manifest that records each file's hash and every count the pipeline dropped (`test_schema_compression_and_manifest`).
- A document's split is a function of its text alone, the first 8 bytes of its SHA-256 modulo 1000, so exact copies can never straddle train and val and the split never moves when documents are added or reordered (`test_split_follows_the_hash`).
- The same documents in any order give byte-identical shards and manifest (`test_output_is_deterministic`), and a failed write never damages the previous output (`test_atomic_replace`).

## How to work this chapter

```bash
ol start data.06              # stubs shard.py into python/corpus/
ol tests data.06              # the course test catalog
ol tdd red data.06            # rung R3: your tests first, failing against the stubs
ol check data.06              # exit code is the verdict
ol check data.06 --ref-deps   # only if data.02, data.04, or data.05 is not passing yet
ol mutate data.06             # how many planted bugs your tests catch
ol diff  data.06              # after passing: your code against the reference
```

Add `pyarrow` to your `python/pyproject.toml` (`dependencies = ["numpy>=2", "pyarrow>=18", "zstandard>=0.23"]`); the contract pre-check allows it for `corpus` units only.

---

## 1. Why now

After `data.05` the cleaned corpus exists only as a Python iterator. Nothing downstream can use it: the tokenizer (`data.07`) needs files it can read in a fixed order, the training loader needs a validation split that never overlaps training, the ledger check (`data.08`) needs every row's source and license, and the durable `CorpusBuild` (`data.09`) needs an output whose hash says whether a retried run produced the same thing. A split made by `random()` would move documents between train and val on every run, and an exact copy that sits in both makes your validation loss lie. This module writes the corpus as sorted, hash-split Parquet shards with a manifest, atomically.

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

**Columns from upstream.** `minhash_cluster`: rows sharing `meta["minhash_cluster"]` (the root id `data.04` wrote; a document without it is its own cluster) form a cluster, and its value is the **smallest row index** of the cluster in output order, a number you can group by without the ids. `pii_redactions` is `meta["pii_redactions"]`; the manifest's `pii` object sums `data.05`'s `manifest_counts(meta["pii"])` over the documents. `lang` defaults to `und` (undetermined), `n_chars` counts code points. A document without `url` or `license_spdx` is an error: the ledger could not trace it.

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

This is `test_hand_example`; `read_shards` gives the same three rows back in `test_read_shards`.

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

`write_shards` has to read every document before writing (it sorts), so it is the end of the streaming part of the pipeline. Write with `pq.ParquetWriter(path, SCHEMA, compression="zstd")` and one `write_table` per row group (`row_group_size` set to the group's length, so pyarrow does not split it again).

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example` | unit | the section 3 order, splits, clusters, character counts, and shards | you and the tests agree on the definitions |
| `test_schema_compression_and_manifest` | conformance | exact Arrow schema, zstd, manifest valid against the schema, file hashes and row counts true, no stray files | every reader relies on the contract |
| `test_sorted_by_id_in_byte_order` | unit | string order, not numeric; `n_chars` in code points; `und` default; untagged rows are their own cluster | rows land in the same shard on every machine |
| `test_split_follows_the_hash` | property | the big-endian rule on every row; exact copies on one side; split stable under reordering and subsets | no val text leaks into training |
| `test_val_share_matches_permille` | statistical | 4000 documents at 100 permille: val within 4 standard deviations of 400; bad arguments raise | the configured share is what you get |
| `test_shards_and_names` | boundary | 23 rows at 10 per shard: 10, 10, 3; names; an empty corpus has one empty shard | readers can always open shard 0 |
| `test_row_groups_are_cut_by_text_bytes` | boundary | groups fit the byte budget; a long document gets its own group, also first in a shard | bounded reader memory |
| `test_output_is_deterministic` | property | shuffled input gives byte-identical files | MS-corpus and `data.09` compare hashes |
| `test_cluster_column_from_near_dedup` | unit | `data.04`'s tags become the smallest row index | the column analysts group by |
| `test_pii_totals` | unit | `data.05`'s counts summed into the manifest; the per-row column | the datasheet's numbers |
| `test_counts_and_key_order` | unit | filters as given, dedup over the defaults, the format's key order and serialization | two writers, one manifest |
| `test_rejects_bad_input` | boundary | missing url or license, repeated id, `shard_rows` 0: errors, nothing written | the ledger can trace every row |
| `test_atomic_replace` | fault | a failed run leaves the old output; a stale `.tmp` is cleared | retried activities are safe |
| `test_read_shards` | unit | manifest and row order, split filter, meta columns, a rewritten shard raises | `data.07` and `data.08` read through it |

**Your tests (rung R3, red then green).** Under `python/tests/data-06-shard/`, failing first against the stubs (`ol tdd red data.06`): the hand example; the split as a function of the text (with the big-endian rule written out); the 248 boundary; shard counts and names, empty corpus included; string order of ids; a missing license raising; a stale `.tmp` cleared on success; PII totals and file hashes; same input, same bytes; column defaults; recorded counts; repeated ids and a tampered shard. `ol mutate data.06` grades them: 0.70 of the mutants, including the one behind Pitfall 2.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. keeping arrival order, or sorting ids numerically | shards differ between runs, or from every other implementation | `test_sorted_by_id_in_byte_order`, `test_output_is_deterministic` (mutants `s01`, `s08`) |
| 2. splitting by the id, or by a random draw | an exact copy under another id lands in val while the original trains | `test_split_follows_the_hash` (mutant `s02`) |
| 3. cutting a row group only after it overflows | groups exceed the budget; a reader's memory is unbounded | `test_row_groups_are_cut_by_text_bytes` (mutant `s10`) |
| 4. defaulting a missing license to "" | rows the ledger cannot trace reach training | `test_rejects_bad_input` (mutant `s12`) |
| 5. writing straight into `out` | a crash leaves half a corpus that looks complete | `test_atomic_replace` (mutant `s13`) |
| 6. reading the first 8 bytes little-endian | a different, still uniform split: every other implementation disagrees | `test_split_follows_the_hash` (mutant `s03`) |
| 7. a timestamp in the manifest | the output hash changes on every run | `test_output_is_deterministic` (mutant `s11`) |
| 8. trusting the shard bytes in `read_shards` | a rewritten shard feeds the tokenizer silently | `test_read_shards` (mutant `s14`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `data.02` | `Doc`, and `meta["lang"]` from the language filter |
| Back | `data.04` | `meta["minhash_cluster"]` becomes the row-index column |
| Back | `data.05` | `pii_redactions`, and the per-kind counts renamed by `manifest_counts` |
| Forward | `data.07` | `read_shards(out, split)` feeds the tokenizer, train then val, in manifest order |
| Forward | `data.08` | every row's `source_id` and `license_spdx` are checked against the ledger; the datasheet reads the manifest |
| Forward | `data.09` | `CorpusBuild`'s `shard` activity; its output hash is the retry check |

If you skip this module, `ol check data.07` stops with `data.07 needs data.06`: build it, or rerun with `--ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `write_shards` | datatrove `ParquetWriter`, Hugging Face `datasets` | streaming writers that never hold the corpus, file rotation by size, remote filesystems | datatrove `src/datatrove/pipeline/writers/parquet.py` |
| the manifest | Delta Lake and Apache Iceberg | transaction logs with snapshots, schema evolution, time travel | the Iceberg table spec |
| `split_of` | Hugging Face `datasets` train/test splits, TFDS | deterministic splits by hash of a key; named splits in metadata | `tensorflow_datasets/core/splits.py` |
| atomic rename | object-store commit protocols | multi-file commits without rename (S3 has none): manifest files written last | the Iceberg commit protocol |
