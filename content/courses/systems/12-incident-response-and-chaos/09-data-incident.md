# A data incident

Every drill so far has had a symptom a dashboard could see: a crashloop, a TTFT burn, a dead letter, a latency shift. A data incident can corrupt training or evaluation while every service stays green. Jobs complete, endpoints respond, and the model quietly learns from a source it was never allowed to use, or from a shard whose content changed after it was recorded. The only way to find what was affected is provenance: the ledger and manifests built in [Licensing ledger verification and the datasheet](../../data-engineering/05-corpus-pipeline/08-ledger-and-datasheet.md) in *Data Engineering*.

This drill follows [A performance regression bisect](08-performance-regression.md), and the shape of its response is different. Where the bisect searched history for a cause, a data incident starts from a known bad source and traces forward to everything built from it. And because the damage spreads every time a downstream job runs, containment comes before investigation.

The section states the principles and the drill scenario, lists what a correct runbook must cover, and derives the procedure, ending with the model runbook `docs/runbooks/DataIncident.md`. Try to write the procedure yourself before you read the solution.

> **Learning objectives**
> - Explain why a data incident is invisible to health metrics and what detects it instead.
> - Quarantine a corpus before scoping, and preserve the ledger and manifests as evidence.
> - Trace a bad source forward from its ledger row to shards, token streams, checkpoints, evaluations, and releases.
> - Recover by rebuilding from the last verified snapshot rather than patching shards in place.

## Principles

Quarantine suspect inputs, preserve the ledger and manifests, and stop downstream jobs from publishing derived artifacts until scope is known. Then trace forward from the bad source (its ledger row, by `source_id` and `sha256`) to shards, token streams, checkpoints, and evaluations, and rebuild from the last verified snapshot.

## The drill scenario

A fixture source violates its license or a shard hash changes. Trace its ledger row to token streams, checkpoints, and evaluations. Rebuild from the last verified source snapshot.

## What a correct runbook must cover

A runbook for a data incident is complete when it answers five questions:

| Case | What the runbook must contain | Why it matters |
|---|---|---|
| Owner and trigger | who owns the corpus; the trigger (a license violation, a changed hash) | a clear start and owner |
| Quarantine | pause releases and training that consume the corpus; preserve `LEDGER.jsonl` and manifests | stops the spread |
| Scope | the ledger row (`source_id`, `license_spdx`, `sha256`, `allowed_uses`) traced to shards, checkpoints, and evaluations | knows exactly what to rebuild |
| Recovery | remove or fix the source, rebuild from the last verified snapshot, re-run `ledger verify` | a clean, verified corpus |
| Verification | `ledger verify` passes; affected models retrained or withdrawn; release gate re-checked | done means verified |

## Deriving the procedure

### Reasoning from first principles

Why do services stay green? Bad data does not crash anything; it changes what a model learns or how it is scored. So detection comes from provenance checks (license allowlist, hash mismatch), not from health metrics.

Contain before investigating. Every job that reads the corpus can spread the problem into checkpoints and published models. Pausing releases and training is cheap and reversible; publishing is not.

The ledger is the map. Each row records a source's `url`, `license_spdx`, `sha256`, `n_docs`, `allowed_uses`, and what was kept. A changed hash or a license not on the allowlist (verification fails with exit 65) names the source; manifests then map it to shards and the runs that consumed them.

Rebuild, do not patch. The last verified snapshot is known good; rebuilding from it and re-verifying is deterministic, while editing shards in place cannot be verified.

Release gates enforce it. A model release refuses a model trained on a source whose `allowed_uses` lacks `train`, so the ledger fix also blocks the bad model from shipping.

### The model runbook

The procedure written out as `docs/runbooks/DataIncident.md`:

````markdown
# Runbook: data incident

Owner: the data pipeline on-call.

## Trigger

A source violates its license (verification exits 65 on an unknown or
disallowed SPDX id) or a shard or source hash changes from its ledger row.

## Contain

1. Pause model releases and training runs that read the affected corpus.
2. Copy `corpus/LEDGER.jsonl` and the corpus and token manifests aside as evidence.

## Scope

1. Find the source's ledger row: `source_id`, `url`, `license_spdx`, `sha256`, `allowed_uses`.
2. From the manifests, list the shards and token streams containing its documents.
3. List checkpoints and evaluations that consumed those shards, and any
   released model trained on them.

## Recover

Remove or fix the source, rebuild the corpus from the last verified snapshot,
and run `{corpus} ledger verify`. Retrain or withdraw affected models; re-run
affected evaluations.

## Verification

`ledger verify` passes; the release gate accepts the rebuilt model; the
incident record lists every affected artifact and its disposition.
````

### Why this solution

Containment first limits the blast radius to what already exists. Provenance beats inspection: tracing by hash and manifest is exact, while reading data to guess what is affected is not. And deleting the bad documents from the shards, the obvious shortcut, produces a corpus with no verified lineage; a rebuild from a verified snapshot has one.

## What can go wrong

Three mistakes let the incident spread or leave it half-fixed:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| Investigating before containing | a release ships a model trained on the bad source | releases and training pause first |
| Patching shards in place | a corpus with no verified lineage | rebuild from the last verified snapshot and run `ledger verify` |
| Scoping only the corpus, not its consumers | checkpoints and evaluations built on bad data stay in use | the scope step traces to checkpoints, evaluations, and releases |

> **Warning:** A model published on bad data cannot be recalled from every consumer. Pause releases before you start scoping.

## Where this leads

The ledger and verification this runbook relies on come from [Licensing ledger verification and the datasheet](../../data-engineering/05-corpus-pipeline/08-ledger-and-datasheet.md), and the drill follows [A performance regression bisect](08-performance-regression.md). The next drill, [A noisy neighbor](10-noisy-neighbor.md), returns to the serving path, where one tenant's load hurts everyone else. To compare this inject with the adjacent drills and reuse the same containment and verification fields, see the [chapter opening](README.md).

## Key terms

- **Data incident**: bad or disallowed input that corrupts training or evaluation without any service failing.
- **Ledger row**: the provenance record of one source: `url`, `license_spdx`, `sha256`, `n_docs`, `allowed_uses`, and what was kept.
- **Quarantine**: pausing releases and training that consume a suspect corpus until its scope is known.
- **Forward trace**: following a bad source through manifests to the shards, token streams, checkpoints, evaluations, and releases built from it.
- **Verified snapshot**: the last corpus state that passed `ledger verify`, used as the base for a rebuild.

## Summary

- A data incident can corrupt training or evaluation while every service stays green; provenance checks, not health metrics, detect it.
- Quarantine first: pause downstream releases and training, and preserve the ledger and manifests as evidence.
- Trace forward from the bad source's ledger row by `source_id` and `sha256` to everything built from it.
- Rebuild from the last verified snapshot and re-run `ledger verify`; release gates then block any model trained on a disallowed source.

## Review questions

1. Why can a data incident go unnoticed by every dashboard?
   <details><summary>Answer</summary>Nothing fails: jobs complete and services respond. Only provenance checks (hashes, license allowlist) see that the input changed.</details>
2. Why pause releases before scoping the incident?
   <details><summary>Answer</summary>A published model built on bad data cannot be unpublished from every consumer; pausing is reversible and cheap.</details>
3. What in the ledger row identifies a silently changed source?
   <details><summary>Answer</summary>Its `sha256`: a re-fetched source whose content hash differs from the recorded one has changed, whatever its URL says.</details>
