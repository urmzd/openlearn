# Corpus Pipeline

A training corpus is the output of a program, and like any program it has bugs, tests, and provenance. This chapter treats the corpus as code: stages with typed interfaces, tests on hand-labelled fixtures, a manifest with content hashes, and a ledger that answers "where did this text come from, and may we use it?" for every shard. The payoff is concrete. Data quality bounds model quality: filtering, deduplication, and decontamination change a model's loss and its benchmark numbers more than most architecture choices at small scale, and the FineWeb paper documents each filter's measured effect.

The chapter builds one pipeline, end to end, over a single `Doc` type. It starts by fetching sources with resumable downloads, pinned checksums, and license capture. It then composes streaming stages, each a generator so memory stays flat on any corpus size: Unicode normalization, language identification, the Gopher quality rules, repetition and perplexity filters. Deduplication comes in two halves: exact (paragraph hashes screened by a Bloom filter and confirmed by sort-merge) and near (MinHash signatures, LSH banding, union-find clusters), with decontamination against eval sets as near dedup's sibling. A PII scrub replaces personal data with typed placeholders. The survivors are written as hash-split Parquet shards with a manifest, tokenized and packed into llm.c `.bin` files, and checked against the licensing ledger, which also generates the datasheet. The last section wraps the whole pipeline as `CorpusBuild`, a durable workflow that survives worker kills and cleans up after cancels.

Two properties run through every section. Every output byte traces to a licensed source: a shard whose source is unknown or not allowlisted fails the build. And determinism is a feature: the same config gives the same output hash regardless of worker count, which is what makes the durable, resumable pipeline of the final section possible. The pipeline is complete when, built from the small reference corpus config, its shards, manifest, ledger, and `.bin` files match the format contracts with a deterministic output hash across two runs and across worker counts 1 and 4.

> **Learning objectives**
> - Build a corpus pipeline as composed streaming stages over one document type, with flat memory on any corpus size.
> - Remove exact and near duplicates correctly and at scale, and decontaminate a corpus against protected eval sets.
> - Scrub PII with typed placeholders and an audit trail that never copies the PII.
> - Write sorted, hash-split Parquet shards and llm.c token files with manifests whose bytes are a function of the input.
> - Verify that every row of training text traces to a license that permits training, and generate a datasheet from the files.
> - Run the pipeline as a durable workflow whose output hash does not depend on kills, retries, or worker count.

## Before you begin

You should be comfortable with the material in [Foundations](../01-foundations/README.md) and with columnar files from [Storage and Warehousing](../02-storage-warehousing/README.md). The fetch stage uses [Python asyncio](../../programming-and-algorithms/01-language-primers/08-python-asyncio.md) from *Programming and Algorithms*, and the ledger and PII sections enforce policies argued in *AI Engineering*: [Data licensing and the ledger](../../ai-engineering/33-data-licensing/01-data-licensing-and-the-ledger.md) and [Privacy and PII policy](../../ai-engineering/34-privacy-and-pii/01-privacy-and-pii-policy.md). The dedup sections use the Bloom filter from [A Bloom filter](../../programming-and-algorithms/15-systems-data-structures/08-bloom-filter.md) in *Programming and Algorithms* and the hashing of [Modular arithmetic, hashing, and PCG32](../../mathematics/12-discrete-math-2/03-modular-arithmetic-hashing-and-pcg32.md) in *Mathematics*. The tokenize section runs the tokenizer of [Tokenizers](../../ai-engineering/07-tokenizers/README.md) in *AI Engineering*, and the final section runs on the engine of [Durable Orchestration and Workers](../../systems/07-durable-orchestration/README.md) in *Systems Engineering*, so read that chapter before the last section. A good warm-up is to work the MinHash and LSH chapter of *Mining of Massive Datasets* by hand for a 3-band example.

## In this chapter

1. [Async fetch with resume, checksums, and license capture](01-async-fetch-resume-checksums.md): download each configured source with HTTP Range resume, verify it against the pinned sha256, quarantine mismatches, and write one ledger row per source so reruns cost nothing.
2. [Extract, normalize, and quality filters](02-extract-normalize-and-quality-filters.md): define the `Doc` and `Stage` types every later stage shares, and build idempotent NFC normalization, stop-word language identification, the Gopher rules, and repetition and perplexity filters.
3. [Exact dedup: paragraph hashes, a Bloom screen, and a sort-merge confirm](03-exact-dedup-bloom-and-sort-merge.md): keep the first occurrence of every paragraph using a Bloom filter that only nominates candidates and a sort-merge that confirms them, so no unique paragraph is ever dropped.
4. [Near-duplicate dedup: MinHash, LSH, union-find, and decontamination](04-near-dedup-minhash-lsh.md): find almost-copies in roughly linear time, cluster them deterministically, and drop any document that shares a 13-gram with a protected eval set.
5. [PII scrub with typed placeholders and audit spans](05-pii-scrub.md): detect emails, phones, cards, IPs, and keys with regex-plus-validator detectors, redact them, and record kinds and offsets under a per-source policy.
6. [Parquet shards, the manifest, and the document-hash split](06-parquet-shards-and-manifest.md): write sorted, zstd-compressed shards split by the hash of each text, with a manifest that is the output's identity, swapped in atomically.
7. [Tokenize and pack to llm.c .bin](07-tokenize-and-pack.md): turn each split into memory-mappable token files with separators, whole-document rollover, exact conservation, and the bytes-per-token factor for bits per byte.
8. [Licensing ledger verification and the datasheet](08-ledger-and-datasheet.md): evaluate SPDX expressions against an allowlist, join the ledger with every shard row, fail with a non-retryable exit 65, and generate the datasheet.
9. [CorpusBuild: the pipeline as a durable workflow](09-corpusbuild-as-a-durable-workflow.md): run fetch, shard, and tokenize as activities keyed by the config's content hash, so kills resume and cancels clean up.

## Where this chapter connects

The Parquet layout behind the shard writer comes from [Storage and Warehousing](../02-storage-warehousing/README.md), and [Orchestration and Modeling](../04-orchestration-modeling/README.md) covers Airflow and dbt, the conventional alternatives to the durable workflow of the last section. The corpus this chapter builds is the one [TinyStories, owned end to end](../../ai-engineering/capstones/01-tinystories.md) in *AI Engineering* trains on.

The same practices appear across the field. Hugging Face's FineWeb and datatrove are documented, reproducible web-scale filtering; the Allen Institute for AI's Dolma is an open corpus with its toolkit and datasheet; and every frontier lab decontaminates against eval sets before reporting benchmark numbers.

## Further reading

- Penedo et al., [*The FineWeb Datasets*](https://arxiv.org/abs/2406.17557): read it end to end; it documents each filter's measured effect.
- Soldaini et al., [*Dolma*](https://arxiv.org/abs/2402.00159).
- Hugging Face, [datatrove](https://github.com/huggingface/datatrove) (Apache-2.0): skim its pipeline stages alongside this chapter.
- Lee et al., [*Deduplicating Training Data Makes Language Models Better*](https://arxiv.org/abs/2107.06499).
- Broder, *On the Resemblance and Containment of Documents* (1997): the origin of MinHash.
- Rae et al., [*Scaling Language Models: Methods, Analysis & Insights from Training Gopher*](https://arxiv.org/abs/2112.11446): the Gopher quality rules.
- Leskovec, Rajaraman, and Ullman, [*Mining of Massive Datasets*](http://www.mmds.org/), chapter 3.
