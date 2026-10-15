<!-- ol:module ops.04 -->
# Drill: KV v2 migration

## Overview

| | |
|---|---|
| **Module** | `ops.04` · drill · ops, docs · Pass 11 · 1 to 2 h |
| **You build** | docs/runbooks/KvV2Migration.md |
| **Tests** | `course/tests/ops.04` (named below, with why each exists) |
| **Needs** | craft.13 |
| **Used by** | no code call site; this is an independent artifact |
| **Milestone** | `MS-P11` |

## Key Takeaways

- Tie each claim to the exact platform evidence that `Drill: KV v2 migration` produces.
- State the owner, success condition, and recovery action before declaring the work complete.
- Record unresolved assumptions so the next review can test them.

## How to work this chapter

```bash
ol start ops.04
ol tests ops.04
ol check ops.04
```

---

## 1. Why now

A format change affects every prefill/decode boundary and stored block. A partial rollout can mix fp8 payloads with f16 readers.

## 2. Principles

Before changing writers, verify v1/v2 negotiation, v2 scale metadata, peer capabilities, and spare capacity. Protect the no-loss invariant and keep a tested downgrade.

## 3. Worked example

Canary one engine pair on a known fixture. Confirm v2 hashes, transfer and reconstruct blocks, then raise v2 traffic while watching negotiation failures and request errors. Roll back the writer flag if any peer lacks support.

## 4. The artifact and its check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_runbook_has_operational_gates_and_recovery` | artifact | Checks that the runbook names an owner, preconditions, canary gate, rollback, verification, both formats, scale, and hash evidence. | The example canary, format hashes, and recovery are concrete requirements the on-call can verify. |

## 5. Pitfalls

| # | Pitfall | Symptom | Caught by |
|---|---|---|---|
| 1 | The canary advances before the rollback trigger is recorded | The exercise reports success without proving the old format or route can be restored. | `test_runbook_has_operational_gates_and_recovery` |
| 2 | The team watches request errors but omits the migration-specific signal | A partial rollout can corrupt or strand work while the generic health check stays green. | `test_runbook_has_operational_gates_and_recovery` |
| 3 | No owner records the decision and verification evidence | The next on-call cannot tell whether the incident is contained. | `test_runbook_has_operational_gates_and_recovery` |

## 6. Where it's used next

| Direction | Module | Connection |
|---|---|---|
| Back | `craft.13` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |
| Back | `ops.03` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |

## Going further

| Resource | What to inspect |
|---|---|
| [Incident drill catalog](README.md) | Compare this inject with the adjacent failure drills and use the same containment and verification fields. |
