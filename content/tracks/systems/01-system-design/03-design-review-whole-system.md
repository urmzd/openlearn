<!-- ol:module review.03 -->
# Design review: the whole system

## Overview

| | |
|---|---|
| **Module** | `review.03` · proof · docs · Pass 11 · 1 to 2 h |
| **You build** | a written review with supporting evidence |
| **Tests** | `course/solve/review.03` (`q1` is self-graded against `course/rubrics/design-review.md`) |
| **Needs** | none |
| **Used by** | no code call site; this is an independent artifact |
| **Milestone** | `MS-P11` |

## Key Takeaways

- Tie each claim to the exact platform evidence that `Design review: the whole system` produces.
- State the owner, success condition, and recovery action before declaring the work complete.
- Record unresolved assumptions so the next review can test them.

## How to work this chapter

```bash
ol start review.03
ol tests review.03
ol check review.03
```

---

## 1. Why now

A complete platform review checks whether the training, serving, gateway, durable, agent, deployment, and operations layers fit together under real constraints.

## 2. Principles

Use the whole-system boundary map. Evaluate the same options against cost, latency, reliability, security, maintainability, and learner-owned implementation effort. Make tradeoffs explicit and support measurements with reproducible evidence.

## 3. Worked example

Follow a customer request from a cited agent answer through gateway policy, engine inference, retrieval, audit, and telemetry. Follow a model update from licensed data through training, evaluation, release approval, canary, rollback, and serving.

## 4. Claim and rubric

Write `docs/reviews/whole-system.md` as the worked reflection or design review. Answer each line of `course/rubrics/design-review.md` with evidence from the completed system, then self-grade q1; each `no` becomes a review note with an owner and next step. The check verifies the artifact and records the rubric self-grade.

## 5. Pitfalls

| # | Pitfall | Symptom | Caught by |
|---|---|---|---|
| 1 | Treating version selection or a document claim as implicit | A reader or client cannot determine which contract or evidence applies. | artifact and rubric self-grade |
| 2 | Skipping the failure path | The successful example works, but malformed input or rollback breaks the system. | artifact and rubric self-grade |
| 3 | Omitting a named owner and recovery condition | The artifact cannot guide an operator when conditions change. | artifact and rubric self-grade |

## 6. Where it's used next

| Direction | Module | Connection |
|---|---|---|
| Back | `review.02` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |
| Back | `craft.09` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |

## Going further

| Resource | What to inspect |
|---|---|
| [Course system contract](../../../../course/DESIGN.md) | Use the current contracts and system boundaries as evidence when defending this review. |
