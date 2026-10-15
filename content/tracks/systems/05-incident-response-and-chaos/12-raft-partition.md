<!-- ol:module ops.12 -->
# Drill: Raft partition (optional)

## Overview

| | |
|---|---|
| **Module** | `ops.12` · drill · ops, docs · Pass 11 · 1 to 2 h |
| **You build** | docs/runbooks/RaftPartition.md |
| **Tests** | `course/tests/ops.12` (named below, with why each exists) |
| **Needs** | none |
| **Used by** | no code call site; this is an independent artifact |
| **Milestone** | `MS-durable-ha` |

## Key Takeaways

- Tie each claim to the exact platform evidence that `Drill: Raft partition (optional)` produces.
- State the owner, success condition, and recovery action before declaring the work complete.
- Record unresolved assumptions so the next review can test them.

## How to work this chapter

```bash
ol start ops.12
ol tests ops.12
ol check ops.12
```

---

## 1. Why now

A network partition can create competing Raft leadership and threaten durable ordering. The drill is optional because it requires the HA module.

## 2. Principles

Use a deterministic partition harness, preserve quorum rules, and verify linearizable reads and acknowledged writes. Heal the network before promoting a replacement.

## 3. Worked example

Partition one follower, then isolate the leader from quorum. Check that minority writes are rejected; heal links and confirm committed entries converge without gaps.

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
| Back | `ops.11` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |
| Back | `dur.10` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |

## Going further

| Resource | What to inspect |
|---|---|
| [Incident drill catalog](README.md) | Compare this inject with the adjacent failure drills and use the same containment and verification fields. |
