<!-- ol:module field.04 -->
# Mock engagement: POC and evaluation

## Overview

| | |
|---|---|
| **Module** | `field.04` · practice · docs · Pass 11 · 1 to 2 h |
| **You build** | docs/engagement/poc.md |
| **Tests** | `course/tests/field.04` (named below, with why each exists) |
| **Needs** | none |
| **Used by** | no code call site; this is an independent artifact |
| **Milestone** | `MS-P11` |

## Key Takeaways

- Tie each claim to the exact platform evidence that `Mock engagement: POC and evaluation` produces.
- State the owner, success condition, and recovery action before declaring the work complete.
- Record unresolved assumptions so the next review can test them.

## How to work this chapter

```bash
ol start field.04
ol tests field.04
ol check field.04
```

---

## 1. Why now

A proof of concept needs a decision at its end. Evaluation criteria chosen after seeing results invite cherry-picking.

## 2. Principles

Set acceptance thresholds, representative tasks, comparison baseline, data handling, and rollback before the first run. Include operational and safety behavior as well as quality.

## 3. Worked example

Choose a small evaluation set with held-out cases. Compare candidate and baseline under a fixed seed and report quality, latency, cost, refusal behavior, and failure examples. State what the sample cannot establish.

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
| Back | `field.03` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |
| Back | `L6.7` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |
| Back | `ag.09` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |

## Going further

| Resource | What to inspect |
|---|---|
| [Mock engagement sequence](README.md) | Carry the evidence and open decisions into the next customer-facing artifact in the sequence. |
