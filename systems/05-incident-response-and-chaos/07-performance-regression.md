<!-- ol:module ops.07 -->
# Drill: performance regression bisect

## Overview

| | |
|---|---|
| **Module** | `ops.07` · drill · ops, docs · Pass 11 · 1 to 2 h |
| **You build** | docs/runbooks/PerformanceRegression.md |
| **Tests** | `course/tests/ops.07` (named below, with why each exists) |
| **Needs** | none |
| **Used by** | no code call site; this is an independent artifact |
| **Milestone** | `MS-P11` |

## Key Takeaways

- Tie each claim to the exact platform evidence that `Drill: performance regression bisect` produces.
- State the owner, success condition, and recovery action before declaring the work complete.
- Record unresolved assumptions so the next review can test them.

## How to work this chapter

```bash
ol start ops.07
ol tests ops.07
ol check ops.07
```

---

## 1. Why now

A performance regression can consume headroom gradually and evade functional tests. A controlled bisect narrows the change while preserving a stable workload.

## 2. Principles

Use the same machine class, model, request mix, warmup, and sample count. Set a threshold and rerun the candidate commit before declaring it bad.

## 3. Worked example

Inject the seeded regression, run the saved load profile, identify the first bad revision with git bisect, and repeat base/head/candidate measurements.

## 4. The artifact and its check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_artifact_paths_are_declared` | artifact | Confirms the submitted paths and required evidence exist, so a plausible narrative cannot pass without the reviewable deliverable. |

## 5. Pitfalls

| # | Pitfall | Symptom | Caught by |
|---|---|---|---|
| 1 | The canary advances before the rollback trigger is recorded | The exercise reports success without proving the old format or route can be restored. | `test_artifact_paths_are_declared` |
| 2 | The team watches request errors but omits the migration-specific signal | A partial rollout can corrupt or strand work while the generic health check stays green. | `test_artifact_paths_are_declared` |
| 3 | No owner records the decision and verification evidence | The next on-call cannot tell whether the incident is contained. | `test_artifact_paths_are_declared` |

## 6. Where it's used next

| Direction | Module | Connection |
|---|---|---|
| Back | `craft.16` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |
| Back | `ops.06` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |

## Going further

| Resource | What to inspect |
|---|---|
| [Incident drill catalog](README.md) | Compare this inject with the adjacent failure drills and use the same containment and verification fields. |
