# Datasheet: <dataset> <version>

Following Gebru et al., "Datasheets for Datasets".

## Motivation

- <Why the dataset was built, by whom, for which model and task.>

## Composition

- <What a document is; how many (n_docs from _MANIFEST.json); languages;
  the splits and how they were made (document-hash split, val_permille).>
- <Does it contain personal data? What the PII scrub found (counts by type)
  and what remains possible.>

## Collection

- <Every source: source_id, URL, license (SPDX), retrieved_at, sha256, from
  corpus/LEDGER.jsonl. How it was fetched.>

## Preprocessing

- <Normalization, each quality filter and how many documents it dropped,
  exact and near deduplication (parameters and counts), decontamination
  against the protected sets (count), tokenization (tokenizer id).>

## Uses

- <Allowed uses per the ledger (train, eval), and uses to avoid.>

## Distribution and maintenance

- <Where the files live (/artifacts paths), the config sha256 that
  reproduces them, how a source revocation is handled (drill ops.08: purge
  derived shards, retrain decision).>

<!-- contracts/templates/DATASHEET.md (data.08, ethics.01): `{corpus}
     datasheet` fills the counts; the prose is written by hand. Copy to DATASHEET.md. -->
