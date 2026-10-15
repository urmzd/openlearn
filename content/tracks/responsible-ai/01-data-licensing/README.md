# Data Licensing

## Overview

- **Primary references**: Longpre et al., [*The Data Provenance Initiative*](https://arxiv.org/abs/2310.16787) (free); the [SPDX License List](https://spdx.org/licenses/) (free)
- **Supplementary**: [Creative Commons licenses](https://creativecommons.org/licenses/list.en) (free); [Choose a License](https://choosealicense.com/) (free); Gokaslan et al., [*CommonCanvas*](https://arxiv.org/abs/2310.16825) (training on openly licensed data, free); [Common Crawl terms of use](https://commoncrawl.org/terms-of-use) (free)
- **Prerequisites**: none for the reading; `data.01` (license capture at fetch time) for the course module
- **Estimated time**: 3 to 5 h in course Pass 3

## Key Takeaways

- **A dataset's license is a property of every source in it**, not of the dataset. Aggregators routinely relabel; the Data Provenance audit found widespread license omission and miscategorization.
- **Record the license when you fetch, not when you are asked.** A ledger row per source (URL, license id, retrieval date, checksum, terms) is cheap at fetch time and nearly impossible to reconstruct later.
- **An allowlist is a policy, written down.** Your pipeline fails the build when a shard traces to a source whose license is unknown or not on the list.
- **This is engineering, not legal advice**: the course teaches you to make provenance auditable, and a lawyer decides what the license permits.

## How to Study

Read the Data Provenance Initiative paper and look up the SPDX ids of five datasets you have used. In the course, `ethics.01` writes your licensing policy and allowlist; `data.08` then verifies every shard of your corpus against it.

---

# Concepts & Techniques

## Core Insight

You cannot honor terms you did not record. Licensing for training data is a provenance problem: each byte of the corpus must trace back to a source, and each source must carry the terms under which it was obtained. Once provenance is data, compliance becomes a check that runs in CI.

## 1. Licenses and terms

**Key ideas**:
- **Permissive** (MIT, Apache-2.0, CC BY), **share-alike** (CC BY-SA, GPL), **non-commercial** (CC BY-NC), **no license** (all rights reserved by default), and **terms of service** that bind independently of copyright.
- **SPDX identifiers** give every license a stable machine-readable name for the ledger.

## 2. The ledger and the allowlist

**Key ideas**:
- **Ledger rows** are written by `data.01` at fetch and validated against `formats/ledger.schema.json`.
- **Verification** (`data.08`): every shard row traces to a source id; an unknown or non-allowlisted license exits 65, which the subprocess activity contract treats as non-retryable.
- **Release gate**: `dur.12` refuses to release a model whose data ledger does not verify.

## Course modules

| Module | Topic | Kind | Pass |
|---|---|---|---|
| `ethics.01` | Data licensing and the ledger | practice | 3 |

## Chapters

<!-- ol:chapters -->
| # | Module | Chapter | Kind | Pass |
|---|---|---|---|---|
| 1 | `ethics.01` | [Data licensing and the ledger](01-data-licensing-and-the-ledger.md) | practice | 3 |
<!-- /ol:chapters -->

## Connections to Other Tracks

| Track | Connection |
|---|---|
| [Responsible AI](../) | the track overview and how the six topics connect |
| [Corpus Pipeline](../../data-engineering/05-corpus-pipeline/) | license capture (`data.01`) and ledger verification (`data.08`) |
| [Model and Data Cards](../03-model-and-data-cards/) | the datasheet reports what the ledger records |
