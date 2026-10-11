<!-- ol:module ops.10 -->
# Drill: runaway agent

## Overview

| | |
|---|---|
| **Module** | `ops.10` · drill · ops, docs · Pass 11 · 1 to 2 h |
| **You build** | docs/runbooks/RunawayAgent.md |
| **Tests** | `course/tests/ops.10` (named below, with why each exists) |
| **Needs** | none |
| **Used by** | no code call site; this is an independent artifact |
| **Milestone** | `MS-P11` |

## Key Takeaways

- Tie each claim to the exact platform evidence that `Drill: runaway agent` produces.
- State the owner, success condition, and recovery action before declaring the work complete.
- Record unresolved assumptions so the next review can test them.

## How to work this chapter

```bash
ol start ops.10
ol tests ops.10
ol check ops.10
```

---

## 1. Why now

An agent loop can repeatedly call tools or consume its entire budget without producing useful work. The gateway and SDK need bounded behavior.

## 2. Principles

Set iteration, token, wall-clock, and tool concurrency budgets. Require gates for consequential actions, and make cancellation propagate through active calls.

## 3. Worked example

Run a seeded agent that repeats a tool call and another that fans out parallel reads. Confirm the loop stops at the configured budget and reports a useful terminal reason.

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
| Back | `ops.09` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |
| Back | `ag.03` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |
| Back | `ag.04` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |

## Going further

| Resource | What to inspect |
|---|---|
| [Incident drill catalog](README.md) | Compare this inject with the adjacent failure drills and use the same containment and verification fields. |
