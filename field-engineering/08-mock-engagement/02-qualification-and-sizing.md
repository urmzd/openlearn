<!-- ol:module field.02 -->
# Mock engagement: qualification and sizing

## Overview

| | |
|---|---|
| **Module** | `field.02` · practice · docs · Pass 11 · 1 to 2 h |
| **You build** | docs/engagement/qualification.md |
| **Tests** | `course/tests/field.02` (named below, with why each exists) |
| **Needs** | none |
| **Used by** | no code call site; this is an independent artifact |
| **Milestone** | `MS-P11` |

## Key Takeaways

- Tie each claim to the exact platform evidence that `Mock engagement: qualification and sizing` produces.
- State the owner, success condition, and recovery action before declaring the work complete.
- Record unresolved assumptions so the next review can test them.

## How to work this chapter

```bash
ol start field.02
ol tests field.02
ol check field.02
```

---

## 1. Why now

Discovery becomes an engagement only when the problem, sponsor, resources, and decision are clear. Sizing translates workload into a testable capacity estimate.

## 2. Principles

Qualify urgency, authority, data access, budget, and decision date. Show arithmetic and uncertainty ranges for compute, memory, throughput, and integration effort.

## 3. Worked example

Use the customer’s arrival rate, prompt and output lengths, and model dimensions to estimate KV memory and concurrency. Compare the estimate with load reports; name the variables that could change it.

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
| Back | `field.01` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |
| Back | `M05.1` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |
| Back | `load.01` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |

## Going further

| Resource | What to inspect |
|---|---|
| [Mock engagement sequence](README.md) | Carry the evidence and open decisions into the next customer-facing artifact in the sequence. |
