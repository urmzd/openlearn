<!-- ol:module craft.12 -->
# Deprecation policy

## Overview

| | |
|---|---|
| **Module** | `craft.12` · practice · docs · Pass 11 · 1 to 2 h |
| **You build** | docs/DEPRECATION.md |
| **Tests** | `course/tests/craft.12` (named below, with why each exists) |
| **Needs** | none |
| **Used by** | no code call site; this is an independent artifact |
| **Milestone** | `MS-P11` |

## Key Takeaways

- Tie each claim to the exact platform evidence that `Deprecation policy` produces.
- State the owner, success condition, and recovery action before declaring the work complete.
- Record unresolved assumptions so the next review can test them.

## How to work this chapter

```bash
ol start craft.12
ol tests craft.12
ol check craft.12
```

---

## 1. Why now

A stable service accumulates interfaces that eventually need replacement. A deprecation policy gives consumers time and operators evidence before removal.

## 2. Principles

Deprecate before removal. Name the replacement, affected versions, sunset date or condition, compatibility window, owner, and support path. Measure real use before setting a removal date.

## 3. Worked example

For KV v1, announce the v2 envelope, accept both formats during migration, count v1 reads and writes, alert on remaining consumers, then remove v1 only after the stated threshold and window.

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
| Back | `craft.11` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |

## Going further

| Resource | What to inspect |
|---|---|
| [C4 model and architecture rules](../../../../course/DESIGN.md), section 2.3 | Compare the submitted views with the course system boundary and its process/file interfaces. |
