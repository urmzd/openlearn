<!-- ol:module ops.06 -->
# Drill: breaking dependency upgrade

## Overview

| | |
|---|---|
| **Module** | `ops.06` · drill · ops, docs · Pass 11 · 1 to 2 h |
| **You build** | docs/runbooks/DependencyUpgrade.md |
| **Tests** | `course/tests/ops.06` (named below, with why each exists) |
| **Needs** | none |
| **Used by** | no code call site; this is an independent artifact |
| **Milestone** | `MS-P11` |

## Key Takeaways

- Tie each claim to the exact platform evidence that `Drill: breaking dependency upgrade` produces.
- State the owner, success condition, and recovery action before declaring the work complete.
- Record unresolved assumptions so the next review can test them.

## How to work this chapter

```bash
ol start ops.06
ol tests ops.06
ol check ops.06
```

---

## 1. Why now

A breaking dependency release can change runtime behavior without changing your source-level contract. This drill practices containment and a reproducible rollback.

## 2. Principles

Inspect the resolved dependency graph and release notes. Change one dependency, keep lockfiles, run the focused contract suite, and compare the same load profile before deployment.

## 3. Worked example

Upgrade the designated major version in a scratch branch. Build, run native and contract tests, then compare latency and error distributions with the last known good image. Deploy only after the gates pass.

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
| Back | `craft.15` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |
| Back | `ops.05` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |

## Going further

| Resource | What to inspect |
|---|---|
| [Incident drill catalog](README.md) | Compare this inject with the adjacent failure drills and use the same containment and verification fields. |
