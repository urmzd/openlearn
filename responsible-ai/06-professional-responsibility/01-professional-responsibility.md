<!-- ol:module ethics.06 -->
# Professional responsibility and disclosure

## Overview

| | |
|---|---|
| **Module** | `ethics.06` · proof · docs · Pass 11 · 1 to 2 h |
| **You build** | a written review with supporting evidence |
| **Tests** | `course/solve/ethics.06` (`q1` is self-graded against `course/rubrics/design-review.md`) |
| **Needs** | none |
| **Used by** | no code call site; this is an independent artifact |
| **Milestone** | `MS-P11` |

## Key Takeaways

- Tie each claim to the exact platform evidence that `Professional responsibility and disclosure` produces.
- State the owner, success condition, and recovery action before declaring the work complete.
- Record unresolved assumptions so the next review can test them.

## How to work this chapter

```bash
ol start ethics.06
ol tests ethics.06
ol check ethics.06
```

---

## 1. Why now

A production system affects people through data choices, model behavior, access controls, and incident decisions. Professional responsibility includes making those effects legible to users and decision makers.

## 2. Principles

Disclose capabilities and limits in language suited to the audience. Correct material errors promptly. Protect confidential information, distinguish measured results from assumptions, and escalate harm through a named channel.

## 3. Worked example

Choose one consequential claim from the model card or customer evaluation. Explain what evidence supports it, what is unknown, who could be harmed if it is overstated, and what wording or product control reduces that risk.

## 4. Claim and rubric

Write `docs/responsibility/reflection.md` as the worked reflection or design review. Answer each line of `course/rubrics/design-review.md` with evidence from the completed system, then self-grade q1; each `no` becomes a review note with an owner and next step. The check verifies the artifact and records the rubric self-grade.

## 5. Pitfalls

| # | Pitfall | Symptom | Caught by |
|---|---|---|---|
| 1 | Treating version selection or a document claim as implicit | A reader or client cannot determine which contract or evidence applies. | artifact and rubric self-grade |
| 2 | Skipping the failure path | The successful example works, but malformed input or rollback breaks the system. | artifact and rubric self-grade |
| 3 | Omitting a named owner and recovery condition | The artifact cannot guide an operator when conditions change. | artifact and rubric self-grade |

## 6. Where it's used next

| Direction | Module | Connection |
|---|---|---|
| Back | `ethics.05` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |
| Back | `craft.19` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |

## Going further

| Resource | What to inspect |
|---|---|
| [Course system contract](../../course/DESIGN.md) | Use the current contracts and system boundaries as evidence when defending this review. |
