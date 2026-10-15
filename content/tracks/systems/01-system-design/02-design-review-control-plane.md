<!-- ol:module review.02 -->
# Design review: control plane

## Overview

| | |
|---|---|
| **Module** | `review.02` · proof · docs · Pass 11 · 1 to 2 h |
| **You build** | a written review with supporting evidence |
| **Tests** | `course/solve/review.02` (`q1` is self-graded against `course/rubrics/design-review.md`) |
| **Needs** | none |
| **Used by** | no code call site; this is an independent artifact |
| **Milestone** | `MS-P11` |

## Key Takeaways

- Tie each claim to the exact platform evidence that `Design review: control plane` produces.
- State the owner, success condition, and recovery action before declaring the work complete.
- Record unresolved assumptions so the next review can test them.

## How to work this chapter

```bash
ol start review.02
ol tests review.02
ol check review.02
```

---

## 1. Why now

The control plane contains durable workflows, workers, queues, an agent runtime, and a usage gateway. Their failure semantics determine whether retries preserve user intent or repeat side effects.

## 2. Principles

Review contracts, data ownership, idempotency, tenant boundaries, load shedding, observability, and recovery as one system. Distinguish durable state from process memory and identify who owns every transition.

## 3. Worked example

Trace one CorpusBuild activity from workflow history to the Python subprocess and artifact files. Then trace an AgentRun tool call that needs approval. Identify what is persisted before each external effect and how replay behaves after a crash.

## 4. Claim and rubric

Write `docs/reviews/control-plane.md` as the worked reflection or design review. Answer each line of `course/rubrics/design-review.md` with evidence from the completed system, then self-grade q1; each `no` becomes a review note with an owner and next step. The check verifies the artifact and records the rubric self-grade.

## 5. Pitfalls

| # | Pitfall | Symptom | Caught by |
|---|---|---|---|
| 1 | Treating version selection or a document claim as implicit | A reader or client cannot determine which contract or evidence applies. | artifact and rubric self-grade |
| 2 | Skipping the failure path | The successful example works, but malformed input or rollback breaks the system. | artifact and rubric self-grade |
| 3 | Omitting a named owner and recovery condition | The artifact cannot guide an operator when conditions change. | artifact and rubric self-grade |

## 6. Where it's used next

| Direction | Module | Connection |
|---|---|---|
| Back | `review.01` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |
| Back | `dur.01` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |
| Back | `ag.05` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |

## Going further

| Resource | What to inspect |
|---|---|
| [Course system contract](../../../../course/DESIGN.md) | Use the current contracts and system boundaries as evidence when defending this review. |
