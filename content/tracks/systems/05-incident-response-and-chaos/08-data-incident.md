# Drill: data incident

## Overview

| | |
|---|---|
| **Module** | `ops.08` · drill · ops, docs · Pass 11 · 1 to 2 h |
| **Solution** | the model runbook `docs/runbooks/DataIncident.md`, written out with its reasoning in section 5 |
| **Needs** | none |
| **Used by** | no code call site: a drill, one of the Pass 11 operations drills |

## Key Takeaways

- A data incident can corrupt training or evaluation while every service stays green; the ledger and manifests are the only way to find what was affected.
- **Quarantine first**: stop downstream jobs from publishing derived artifacts until the scope is known; preserve the ledger and manifests as evidence.
- Trace forward from the bad source (ledger row, by `source_id` and `sha256`) to shards, token streams, checkpoints, and evaluations, then rebuild from the last verified snapshot.

## How to read this chapter

This chapter is a worked drill. Sections 1 to 3 state the risk, the principles, and the drill scenario. Section 4 lists what a correct runbook must cover. Section 5 derives the procedure from first principles, gives the model runbook, and explains why each step is the right one. Try to write the procedure yourself from sections 2 and 3 before reading section 5.

---

## 1. Why now

A data incident can corrupt training or evaluation while leaving services green. Provenance and hashes help determine what was affected.

## 2. Principles

Quarantine suspect inputs, preserve the ledger and manifests, and stop downstream jobs from publishing derived artifacts until scope is known.

## 3. Worked example

A fixture source violates its license or a shard hash changes. Trace its ledger row to token streams, checkpoints, and evaluations. Rebuild from the last verified source snapshot.

## 4. What a correct runbook must cover

| Case | What the runbook must contain | Why it matters |
|---|---|---|
| Owner and trigger | who owns the corpus; the trigger (a license violation, a changed hash) | a clear start and owner |
| Quarantine | pause releases and training that consume the corpus; preserve `LEDGER.jsonl` and manifests | stops the spread |
| Scope | the ledger row (`source_id`, `license_spdx`, `sha256`, `allowed_uses`) traced to shards, checkpoints, and evaluations | knows exactly what to rebuild |
| Recovery | remove or fix the source, rebuild from the last verified snapshot, re-run `ledger verify` | a clean, verified corpus |
| Verification | `ledger verify` passes; affected models retrained or withdrawn; release gate re-checked | done means verified |

## 5. The solution, step by step

### 5.1 From first principles

1. **Why do services stay green?** Bad data does not crash anything; it changes what a model learns or how it is scored. So detection comes from provenance checks (license allowlist, hash mismatch), not from health metrics.
2. **Contain before investigating.** Every job that reads the corpus can spread the problem into checkpoints and published models. Pausing releases and training is cheap and reversible; publishing is not.
3. **The ledger is the map.** Each row records a source's `url`, `license_spdx`, `sha256`, `n_docs`, `allowed_uses`, and what was kept. A changed hash or a license not on the allowlist (verification fails with exit 65) names the source; manifests then map it to shards and the runs that consumed them.
4. **Rebuild, do not patch.** The last verified snapshot is known good; rebuilding from it and re-verifying is deterministic, while editing shards in place cannot be verified.
5. **Release gates enforce it.** A model release refuses a model trained on a source whose `allowed_uses` lacks `train`, so the ledger fix also blocks the bad model from shipping.

### 5.2 The model runbook

`docs/runbooks/DataIncident.md`:

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

### 5.3 Why this solution

- **Containment first** limits the blast radius to what already exists.
- **Provenance over inspection**: tracing by hash and manifest is exact; reading data to guess what is affected is not.
- **Why not delete the bad documents from the shards?** The result has no verified lineage; a rebuild from a verified snapshot does.

### 5.4 Explain it back

1. Why can a data incident go unnoticed by every dashboard?
   <details><summary>Answer</summary>Nothing fails: jobs complete and services respond. Only provenance checks (hashes, license allowlist) see that the input changed.</details>
2. Why pause releases before scoping the incident?
   <details><summary>Answer</summary>A published model built on bad data cannot be unpublished from every consumer; pausing is reversible and cheap.</details>
3. What in the ledger row identifies a silently changed source?
   <details><summary>Answer</summary>Its `sha256`: a re-fetched source whose content hash differs from the recorded one has changed, whatever its URL says.</details>

## 6. Pitfalls

| # | Pitfall | Symptom | How the solution avoids it |
|---|---|---|---|
| 1 | Investigating before containing | a release ships a model trained on the bad source | releases and training pause first |
| 2 | Patching shards in place | a corpus with no verified lineage | rebuild from the last verified snapshot and run `ledger verify` |
| 3 | Scoping only the corpus, not its consumers | checkpoints and evaluations built on bad data stay in use | the scope step traces to checkpoints, evaluations, and releases |

## 7. Where it's used next

| Direction | Module | Connection |
|---|---|---|
| Back | `ops.07` | A prerequisite: the contract or evidence this runbook relies on. |
| Back | `data.08` | A prerequisite: the contract or evidence this runbook relies on. |

## Going further

| Resource | What to inspect |
|---|---|
| [Incident drill catalog](README.md) | Compare this inject with the adjacent failure drills and use the same containment and verification fields. |
