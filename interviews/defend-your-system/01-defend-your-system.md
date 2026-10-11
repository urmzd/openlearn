<!-- ol:module iv.01 -->
# Defend your system

## Overview

| | |
|---|---|
| **Module** | `iv.01` · proof · docs · Pass 11 · 1 to 2 h |
| **You build** | a written review with supporting evidence |
| **Tests** | `course/solve/iv.01` (`q1` is self-graded against `course/rubrics/design-review.md`) |
| **Needs** | none |
| **Used by** | no code call site; this is an independent artifact |
| **Milestone** | `MS-P11` |

## Key Takeaways

- Tie each claim to the exact platform evidence that `Defend your system` produces.
- State the owner, success condition, and recovery action before declaring the work complete.
- Record unresolved assumptions so the next review can test them.

## How to work this chapter

```bash
ol start iv.01
ol tests iv.01
ol check iv.01
```

---

## 1. Why now

A technical defense tests whether you understand the system you built and can explain its limits to another engineer.

## 2. Principles

Answer from evidence. State the constraints, rejected alternatives, failure behavior, and a number you can reproduce. Admit when data is unavailable and name how you would obtain it.

## 3. Worked example

Defend why the engine and gateway use separate HTTP and gRPC boundaries. Describe how a rolling API migration behaves during a worker restart, then calculate one capacity limit from the design.

## 4. Claim and rubric

Write `docs/defense/answers.md` as the worked reflection or design review. Answer each line of `course/rubrics/design-review.md` with evidence from the completed system, then self-grade q1; each `no` becomes a review note with an owner and next step. The check verifies the artifact and records the rubric self-grade.

## 5. Pitfalls

| # | Pitfall | Symptom | Caught by |
|---|---|---|---|
| 1 | Treating version selection or a document claim as implicit | A reader or client cannot determine which contract or evidence applies. | artifact and rubric self-grade |
| 2 | Skipping the failure path | The successful example works, but malformed input or rollback breaks the system. | artifact and rubric self-grade |
| 3 | Omitting a named owner and recovery condition | The artifact cannot guide an operator when conditions change. | artifact and rubric self-grade |

## 6. Where it's used next

| Direction | Module | Connection |
|---|---|---|
| Back | `review.03` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |
| Back | `field.07` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |

## Going further

| Resource | What to inspect |
|---|---|
| [Course system contract](../../course/DESIGN.md) | Use the current contracts and system boundaries as evidence when defending this review. |
