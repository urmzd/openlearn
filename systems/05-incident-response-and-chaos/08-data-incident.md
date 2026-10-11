<!-- ol:module ops.08 -->
# Drill: data incident

## Overview

| | |
|---|---|
| **Module** | `ops.08` · drill · ops, docs · Pass 11 · 1 to 2 h |
| **You build** | docs/runbooks/DataIncident.md |
| **Tests** | `course/tests/ops.08` (named below, with why each exists) |
| **Needs** | none |
| **Used by** | no code call site; this is an independent artifact |
| **Milestone** | `MS-P11` |

## Key Takeaways

- Tie each claim to the exact platform evidence that `Drill: data incident` produces.
- State the owner, success condition, and recovery action before declaring the work complete.
- Record unresolved assumptions so the next review can test them.

## How to work this chapter

```bash
ol start ops.08
ol tests ops.08
ol check ops.08
```

---

## 1. Why now

A data incident can corrupt training or evaluation while leaving services green. Provenance and hashes help determine what was affected.

## 2. Principles

Quarantine suspect inputs, preserve the ledger and manifests, and stop downstream jobs from publishing derived artifacts until scope is known.

## 3. Worked example

A fixture source violates its license or a shard hash changes. Trace its ledger row to token streams, checkpoints, and evaluations. Rebuild from the last verified source snapshot.

## 4. The artifact and its check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_artifact_paths_are_declared` | artifact | Confirms the submitted paths and required evidence exist, so a plausible narrative cannot pass without the reviewable deliverable. |

## 5. Pitfalls

| # | Pitfall | Symptom | Caught by |
|---|---|---|---|
| 1 | The canary advances before the rollback trigger is recorded | The exercise reports success without proving the old format or route can be restored. | `test_artifact_paths_are_declared` |
| 2 | The team watches request errors but omits the migration-specific signal | A partial rollout can corrupt or strand work while the generic health check stays green. | `test_artifact_paths_are_declared` |
| 3 | No owner records the decision and verification evidence | The next on-call cannot tell whether the incident is contained. | `test_artifact_paths_are_declared` |

## 6. Where it's used next

| Direction | Module | Connection |
|---|---|---|
| Back | `ops.07` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |
| Back | `data.08` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |

## Going further

| Resource | What to inspect |
|---|---|
| [Incident drill catalog](README.md) | Compare this inject with the adjacent failure drills and use the same containment and verification fields. |
