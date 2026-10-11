<!-- ol:module field.03 -->
# Mock engagement: performance engagement

## Overview

| | |
|---|---|
| **Module** | `field.03` · practice · docs · Pass 11 · 1 to 2 h |
| **You build** | docs/engagement/performance.md |
| **Tests** | `course/tests/field.03` (named below, with why each exists) |
| **Needs** | none |
| **Used by** | no code call site; this is an independent artifact |
| **Milestone** | `MS-P11` |

## Key Takeaways

- Tie each claim to the exact platform evidence that `Mock engagement: performance engagement` produces.
- State the owner, success condition, and recovery action before declaring the work complete.
- Record unresolved assumptions so the next review can test them.

## How to work this chapter

```bash
ol start field.03
ol tests field.03
ol check field.03
```

---

## 1. Why now

A performance engagement should answer a customer question with repeatable measurements. It must separate bottlenecks from noisy environmental effects.

## 2. Principles

Agree on workload, baseline, success threshold, system configuration, and stop conditions before testing. Preserve reproducibility and explain statistical spread.

## 3. Worked example

Run the same open-loop request mix against the baseline and candidate. Report throughput, TTFT, TPOT, errors, and saturation. Use the perf bisect when a change creates a meaningful regression.

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
| Back | `field.02` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |
| Back | `load.01` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |
| Back | `craft.16` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |

## Going further

| Resource | What to inspect |
|---|---|
| [Mock engagement sequence](README.md) | Carry the evidence and open decisions into the next customer-facing artifact in the sequence. |
