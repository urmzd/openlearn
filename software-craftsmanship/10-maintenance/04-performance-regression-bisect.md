<!-- ol:module craft.16 -->
# Perf regression bisect

## Overview

| | |
|---|---|
| **Module** | `craft.16` · practice · docs · Pass 11 · 1 to 2 h |
| **You build** | docs/maintenance/perf-bisect.md |
| **Tests** | `course/tests/craft.16` (named below, with why each exists) |
| **Needs** | none |
| **Used by** | no code call site; this is an independent artifact |
| **Milestone** | `MS-P11` |

## Key Takeaways

- Tie each claim to the exact platform evidence that `Perf regression bisect` produces.
- State the owner, success condition, and recovery action before declaring the work complete.
- Record unresolved assumptions so the next review can test them.

## How to work this chapter

```bash
ol start craft.16
ol tests craft.16
ol check craft.16
```

---

## 1. Why now

A latency regression costs users and capacity. A bisect turns “it got slower” into a reproducible change range.

## 2. Principles

Use a stable workload, fixed machine and configuration, warm-up, repeated samples, and a threshold chosen before inspecting commits. Validate both endpoints before trusting the midpoint result.

## 3. Worked example

Reproduce a 20% TTFT regression with the saved request fixture. Run the same benchmark at the base and head, then use `git bisect run` with a script that exits nonzero only when the median exceeds the agreed bound. Re-run the identified commit.

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
| Back | `craft.06` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |
| Back | `ops.07` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |

## Going further

| Resource | What to inspect |
|---|---|
| [C4 model and architecture rules](../../course/DESIGN.md), section 2.3 | Compare the submitted views with the course system boundary and its process/file interfaces. |
