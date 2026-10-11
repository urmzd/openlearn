# Corpus Pipeline

## Overview

- **Primary references**: Penedo et al., [*The FineWeb Datasets*](https://arxiv.org/abs/2406.17557) (free); Soldaini et al., [*Dolma*](https://arxiv.org/abs/2402.00159) (free); Hugging Face, [datatrove](https://github.com/huggingface/datatrove) (free, Apache-2.0)
- **Supplementary**: Lee et al., [*Deduplicating Training Data Makes Language Models Better*](https://arxiv.org/abs/2107.06499) (free); Broder, *On the Resemblance and Containment of Documents* (1997, MinHash); Rae et al., [*Scaling Language Models: Methods, Analysis & Insights from Training Gopher*](https://arxiv.org/abs/2112.11446) (the Gopher quality rules, free); Leskovec, Rajaraman, and Ullman, [*Mining of Massive Datasets*](http://www.mmds.org/) chapter 3 (free)
- **Prerequisites**: [Data Engineering Foundations](../01-foundations/), [Storage & Warehousing](../02-storage-warehousing/) (columnar files), the asyncio primer `lang.08`, the [responsible-AI topics on licensing and PII](../../responsible-ai/)
- **Estimated time**: 3 to 4 weeks in course Pass 3, plus one module in Pass 8 when the pipeline becomes a durable workflow

## Key Takeaways

- **Data quality bounds model quality.** Filtering, deduplication, and decontamination change a model's loss and its benchmark numbers more than most architecture choices at small scale.
- **A pipeline is a composition of streaming stages** over one `Doc` type: fetch, normalize, filter, exact dedup, near dedup, PII scrub, shard, tokenize. Each stage is a generator, so memory stays flat on any corpus size.
- **Deduplication has two halves**: exact (hashes, confirmed by sort-merge) and near (MinHash signatures, LSH banding, union-find clusters). Decontamination is near-dedup against your eval sets.
- **Every output byte traces to a licensed source.** The ledger records license and provenance per source; a shard whose source is unknown or not allowlisted fails the build.
- **Determinism is a feature**: the same config gives the same output hash regardless of worker count, which is what makes a durable, resumable pipeline possible later.

## How to Study

Read the FineWeb paper end to end (it documents each filter's measured effect), then skim datatrove's pipeline stages. Work the MinHash and LSH chapter of *Mining of Massive Datasets* by hand for a 3-band example. In the course, build `data.01` to `data.08` in Pass 3 after the tokenizers, and `data.09` in Pass 8 once the durable engine exists.

---

# Concepts & Techniques

## Core Insight

A training corpus is the output of a program, and like any program it has bugs, tests, and provenance. Treating the corpus as code means stages with typed interfaces, tests on hand-labelled fixtures, a manifest with content hashes, and a ledger that answers "where did this text come from and may we use it" for every shard.

## 1. Fetch and stages

**Key ideas**:
- **Async fetch** with resume via HTTP `Range`, checksums, and quarantine on mismatch (`data.01`).
- **Stages** are `Callable[[Iterator[Doc]], Iterator[Doc]]`, composed with `compose` (`data.02`): Unicode normalization, language ID, Gopher rules, repetition filters, and a perplexity filter that plugs in the `L2.1` n-gram model in C1.

## 2. Deduplication and decontamination

**Key ideas**:
- **Exact**: paragraph hashes, a Bloom screen (`ds.08`), then a sort-merge confirm so no true document is dropped (`data.03`).
- **Near**: MinHash with 128 permutations, LSH with $b$ bands of $r$ rows (threshold about $(1/b)^{1/r}$), union-find clusters, process-parallel and worker-count invariant (`data.04`).
- **Decontamination**: drop any document that shares a 13-gram with a protected eval or validation set, and record the count.

## 3. Privacy, shards, and tokens

**Key ideas**:
- **PII scrub** with typed placeholders and audit spans (`data.05`; policy in `ethics.02`).
- **Parquet shards** and a manifest, with a document-hash train/val split (`data.06`).
- **Tokenize and pack** to the llm.c `.bin` format read by `L0.6` `TokenStream` (`data.07`).

## 4. Ledger, datasheet, and the durable pipeline

**Key ideas**:
- **Ledger verification** against `formats/ledger.schema.json` and a generated datasheet (`data.08`; policy in `ethics.01`).
- **`CorpusBuild`** runs the stages as subprocess activities on the learner's durable engine; a killed worker resumes and every shard is written once (`data.09`).

## Course modules

| Module | Topic | Kind | Pass |
|---|---|---|---|
| `data.01` | Async fetch with resume, checksums, license capture (ledger rows) | build | 3 |
| `data.02` | Extract, normalize, quality filters (generator stages) | build | 3 |
| `data.03` | Exact dedup: paragraph hashes, Bloom screen, sort-merge confirm | build | 3 |
| `data.04` | Near-dup MinHash + LSH + union-find, process-parallel; **decontamination** against protected eval and validation sets | build | 3 |
| `data.05` | PII scrub with typed placeholders and audit spans | build | 3 |
| `data.06` | Parquet shards, manifest, document-hash train/val split (uses `pyarrow`) | build | 3 |
| `data.07` | Tokenize and pack to llm.c `.bin` | build | 3 |
| `data.08` | Licensing ledger verification and datasheet | build | 3 |
| `data.09` | Pipeline as a durable workflow `CorpusBuild` | build | 8 |

Milestone `MS-corpus` closes the pipeline: shards, manifest, ledger, and `.bin` files pass the format conformance suite with a deterministic output hash across two runs and worker counts 1 and 4.

## Chapters

<!-- ol:chapters -->
| # | Module | Chapter | Kind | Pass |
|---|---|---|---|---|
| 1 | `data.01` | [Async fetch with resume, checksums, license capture](01-async-fetch-resume-checksums.md) | build | 3 |
| 2 | `data.02` | [Extract, normalize, quality filters (generator stages)](02-extract-normalize-and-quality-filters.md) | build | 3 |
| 3 | `data.03` | [Exact dedup: paragraph hashes, Bloom screen, sort-merge confirm](03-exact-dedup-bloom-and-sort-merge.md) | build | 3 |
| 4 | `data.04` | [Near-duplicate dedup: MinHash, LSH, union-find, and decontamination](04-near-dedup-minhash-lsh.md) | build | 3 |
| 5 | `data.05` | [PII scrub with typed placeholders and audit spans](05-pii-scrub.md) | build | 3 |
| 6 | `data.06` | [Parquet shards, manifest, and the document-hash split](06-parquet-shards-and-manifest.md) | build | 3 |
| 7 | `data.07` | [Tokenize and pack to llm.c .bin](07-tokenize-and-pack.md) | build | 3 |
| 8 | `data.08` | [Licensing ledger verification and the datasheet](08-ledger-and-datasheet.md) | build | 3 |
| 9 | `data.09` | [Pipeline as a durable workflow CorpusBuild](09-corpusbuild-as-a-durable-workflow.md) | build | 8 |
<!-- /ol:chapters -->

## Connections to Other Tracks

| Track | Connection |
|---|---|
| [Storage & Warehousing](../02-storage-warehousing/) | Parquet and columnar layout behind `data.06` |
| [Orchestration & Modeling](../04-orchestration-modeling/) | Airflow and dbt as the going-further for `data.09` |
| [Responsible AI](../../responsible-ai/) | data licensing (`ethics.01`) and privacy (`ethics.02`) |
| [Systems Data Structures](../../algorithms/16-systems-data-structures/) | the Bloom filter (`ds.08`) |
| [tinyllm Part 1](../../ml/08-tinyllm/p01-tokenizers/) | the tokenizer that `data.07` runs |
| [Durable Orchestration & Workers](../../ai-platform-engineering/05-durable-orchestration-and-workers/) | the engine `CorpusBuild` runs on |

## Company Relevance

| Company | Practice |
|---|---|
| Hugging Face | FineWeb and datatrove: documented, reproducible web-scale filtering |
| Allen Institute for AI | Dolma: an open corpus with its toolkit and datasheet |
| Every frontier lab | decontamination against eval sets before reporting benchmark numbers |
