<!-- ol:module craft.09 -->
# C4 diagrams in D2

## Overview

| | |
|---|---|
| **Module** | `craft.09` · practice · docs · Pass 11 · 1 to 2 h |
| **You build** | docs/c4/context.d2, docs/c4/containers.d2, docs/c4/components-engine.d2 |
| **Tests** | `course/tests/craft.09` (named below, with why each exists) |
| **Needs** | none |
| **Used by** | no code call site; this is an independent artifact |
| **Milestone** | `MS-P11` |

## Key Takeaways

- Tie each claim to the exact platform evidence that `C4 diagrams in D2` produces.
- State the owner, success condition, and recovery action before declaring the work complete.
- Record unresolved assumptions so the next review can test them.

## How to work this chapter

```bash
ol start craft.09
ol tests craft.09
ol check craft.09
```

---

## 1. Why now

The architecture is now too large to hold in one view. A C4 set lets a new maintainer move from users and external systems to containers and then to engine components.

## 2. Principles

Keep the context diagram about people and external dependencies. Show deployable processes at container level. Put internal services and stores in the component view. Every arrow names a protocol or file boundary from DESIGN 2.3.

## 3. Worked example

Trace one completion request from CLI through gateway, engine, artifact storage, and telemetry. Draw one view per abstraction level; check that the same system boundary is preserved and the engine subcomponents appear only in the component view.

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
| Back | `review.03` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |

## Going further

| Resource | What to inspect |
|---|---|
| [C4 model and architecture rules](../../../../course/DESIGN.md), section 2.3 | Compare the submitted views with the course system boundary and its process/file interfaces. |
