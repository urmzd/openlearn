<!-- ol:module craft.10 -->
# Runbooks and Diataxis docs

## Overview

| | |
|---|---|
| **Module** | `craft.10` · practice · docs · Pass 11 · 1 to 2 h |
| **You build** | docs/runbooks, docs/README.md |
| **Tests** | `course/tests/craft.10` (named below, with why each exists) |
| **Needs** | none |
| **Used by** | no code call site; this is an independent artifact |
| **Milestone** | `MS-P11` |

## Key Takeaways

- Tie each claim to the exact platform evidence that `Runbooks and Diataxis docs` produces.
- State the owner, success condition, and recovery action before declaring the work complete.
- Record unresolved assumptions so the next review can test them.

## How to work this chapter

```bash
ol start craft.10
ol tests craft.10
ol check craft.10
```

---

## 1. Why now

Incidents recur when operating knowledge lives only in chat. Runbooks make detection and recovery executable; Diátaxis helps readers find tutorials, task procedures, explanations, and reference.

## 2. Principles

A runbook starts with impact and a trigger, names a safe first action, and separates diagnosis from recovery. Each command has expected output and a rollback condition.

## 3. Worked example

For an availability burn alert, identify the affected SLO, query the 5xx and request-rate series, compare the last deployment, then shift traffic or roll back. Verify the alert clears and record the incident timeline.

## 4. The artifact and its check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_artifact_paths_are_declared` | artifact | Confirms the submitted paths and required evidence exist, so a plausible narrative cannot pass without the reviewable deliverable. |

## 5. Pitfalls

| # | Pitfall | Symptom | Caught by |
|---|---|---|---|
| 1 | The artifact names a decision without the evidence that supports it | Reviewers cannot verify the claim against the repository or system. | `test_artifact_paths_are_declared` |
| 2 | The failure or rollback case is omitted | The happy path passes while an operator has no safe response under failure. | `test_artifact_paths_are_declared` |
| 3 | The owner, threshold, or scope is implicit | Two reviewers can reach different release or risk decisions from the same evidence. | `test_artifact_paths_are_declared` |

## 6. Where it's used next

| Direction | Module | Connection |
|---|---|---|
| Back | `craft.02` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |
| Back | `ops.00` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |

## Going further

| Resource | What to inspect |
|---|---|
| [C4 model and architecture rules](../../course/DESIGN.md), section 2.3 | Compare the submitted views with the course system boundary and its process/file interfaces. |
