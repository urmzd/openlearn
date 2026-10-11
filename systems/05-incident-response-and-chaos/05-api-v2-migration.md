<!-- ol:module ops.05 -->
# Drill: API v2 migration

## Overview

| | |
|---|---|
| **Module** | `ops.05` · drill · ops, docs · Pass 11 · 1 to 2 h |
| **You build** | docs/runbooks/ApiV2Migration.md |
| **Tests** | `course/tests/ops.05` (named below, with why each exists) |
| **Needs** | craft.14 |
| **Used by** | no code call site; this is an independent artifact |
| **Milestone** | `MS-P11` |

## Key Takeaways

- Tie each claim to the exact platform evidence that `Drill: API v2 migration` produces.
- State the owner, success condition, and recovery action before declaring the work complete.
- Record unresolved assumptions so the next review can test them.

## How to work this chapter

```bash
ol start ops.05
ol tests ops.05
ol check ops.05
```

---

## 1. Why now

A simultaneous API rollout can break clients even when the new engine is healthy. Version-specific evidence is needed to distinguish protocol errors from service failures.

## 2. Principles

Keep v1 available while validating v2. Route by an explicit version, compare success and latency by version, and define a rollback threshold before increasing the canary.

## 3. Worked example

Send synthetic v1 and v2 streams through the gateway. Verify both OpenAPI suites and usage-ledger totals. Expand the v2 cohort in steps while watching 4xx, 5xx, disconnects, and token accounting.

## 4. The artifact and its check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_runbook_has_cohort_gates_and_rollback` | artifact | Checks that the runbook names an owner, preconditions, canary gate, rollback, verification, both API versions, sunset, API version routing, and token evidence. | Version-specific cohort and sunset checks let the operator roll back before client impact spreads. |

## 5. Pitfalls

| # | Pitfall | Symptom | Caught by |
|---|---|---|---|
| 1 | The canary advances before the rollback trigger is recorded | The exercise reports success without proving the old format or route can be restored. | `test_runbook_has_cohort_gates_and_rollback` |
| 2 | The team watches request errors but omits the migration-specific signal | A partial rollout can corrupt or strand work while the generic health check stays green. | `test_runbook_has_cohort_gates_and_rollback` |
| 3 | No owner records the decision and verification evidence | The next on-call cannot tell whether the incident is contained. | `test_runbook_has_cohort_gates_and_rollback` |

## 6. Where it's used next

| Direction | Module | Connection |
|---|---|---|
| Back | `craft.14` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |
| Back | `ops.04` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |

## Going further

| Resource | What to inspect |
|---|---|
| [Incident drill catalog](README.md) | Compare this inject with the adjacent failure drills and use the same containment and verification fields. |
