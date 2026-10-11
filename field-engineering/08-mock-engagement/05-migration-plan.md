<!-- ol:module field.05 -->
# Mock engagement: migration plan

## Overview

| | |
|---|---|
| **Module** | `field.05` · practice · docs · Pass 11 · 1 to 2 h |
| **You build** | docs/engagement/migration.md |
| **Tests** | `course/tests/field.05` (named below, with why each exists) |
| **Needs** | none |
| **Used by** | no code call site; this is an independent artifact |
| **Milestone** | `MS-P11` |

## Key Takeaways

- Tie each claim to the exact platform evidence that `Mock engagement: migration plan` produces.
- State the owner, success condition, and recovery action before declaring the work complete.
- Record unresolved assumptions so the next review can test them.

## How to work this chapter

```bash
ol start field.05
ol tests field.05
ol check field.05
```

---

## 1. Why now

A successful evaluation still leaves a production transition to plan. The migration must protect existing clients and data while giving the customer a clear exit path.

## 2. Principles

Stage changes, preserve compatibility during the agreed window, define rollback triggers, and assign owners on both sides. Include contract, artifact, and training-data dependencies.

## 3. Worked example

Plan a KV or API version rollout with a canary cohort, mixed-version checks, health thresholds, expansion schedule, and a backward route to the known-good version.

## 4. The artifact and its check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_artifact_paths_are_declared` | artifact | Confirms the submitted paths and required evidence exist, so a plausible narrative cannot pass without the reviewable deliverable. |

## 5. Pitfalls

| # | Pitfall | Symptom | Caught by |
|---|---|---|---|
| 1 | The deliverable records a conclusion without its customer evidence | Another engineer cannot reproduce the sizing or recommendation. | `test_artifact_paths_are_declared` |
| 2 | An assumption is written as a measured fact | The engagement plan can promise capacity or security the system has not demonstrated. | `test_artifact_paths_are_declared` |
| 3 | The next owner and decision gate are missing | Work stalls at handoff even when the technical work is complete. | `test_artifact_paths_are_declared` |

## 6. Where it's used next

| Direction | Module | Connection |
|---|---|---|
| Back | `field.04` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |
| Back | `craft.13` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |
| Back | `craft.14` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |

## Going further

| Resource | What to inspect |
|---|---|
| [Mock engagement sequence](README.md) | Carry the evidence and open decisions into the next customer-facing artifact in the sequence. |
