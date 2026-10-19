# Drill: event log disk full

## Overview

| | |
|---|---|
| **Module** | `ops.11` · drill · ops, docs · Pass 11 · 1 to 2 h |
| **Solution** | the model runbook `docs/runbooks/EventlogDiskFull.md`, written out with its reasoning in section 5 |
| **Needs** | none |
| **Used by** | no code call site: a drill, one of the Pass 11 operations drills |

## Key Takeaways

- An append is acknowledged only after its fsync; an append that would exceed `[durable].wal_max_bytes` fails with gRPC `RESOURCE_EXHAUSTED`, appends nothing, and leaves reads working.
- So a full log stops progress but never loses or invents acknowledged work: no workflow reports success for a rejected append.
- Alert on `tl_durable_wal_bytes` against the quota **before** the filesystem fills; recovery is raising the cap (or freeing space) and restarting.

## How to read this chapter

This chapter is a worked drill. Sections 1 to 3 state the risk, the principles, and the drill scenario. Section 4 lists what a correct runbook must cover. Section 5 derives the procedure from first principles, gives the model runbook, and explains why each step is the right one. Try to write the procedure yourself from sections 2 and 3 before reading section 5.

---

## 1. Why now

A full event-log disk can stop workflow progress and risk losing acknowledged work. The WAL quota provides a controlled boundary.

## 2. Principles

An append is acknowledged only after its durability point. Reject new work cleanly at quota, preserve readable history, and alert before the filesystem is exhausted.

## 3. Worked example

Fill the test WAL to its configured limit. Confirm the next append fails with RESOURCE_EXHAUSTED and no workflow reports success for it. Free or expand space and resume.

## 4. What a correct runbook must cover

| Case | What the runbook must contain | Why it matters |
|---|---|---|
| Owner and detection | `tl_durable_wal_bytes` near `wal_max_bytes`; clients seeing `RESOURCE_EXHAUSTED` | a symptom and its metric |
| Safety | rejected appends append nothing; reads keep working; acknowledged history intact | no data lost or invented |
| Containment | pause new submissions; do not delete segments by hand | protects the only copy of history |
| Recovery | raise `wal_max_bytes` (`walMaxBytes` in the chart) or expand the volume, restart, confirm writes resume | progress resumes |
| Verification | a canary workflow completes; affected clients retried with the same idempotency keys | done means measured |

## 5. The solution, step by step

### 5.1 From first principles

1. **What does the contract promise?** Every append rpc answers only after its records are written and fsynced, and an append that fails (I/O error or quota) leaves the log as it was. So at quota the safe outcome is a clean rejection, not a partial write.
2. **Why a quota below the disk size?** A filesystem that fills mid-write can fail in ways the server cannot control. A configured cap (`wal_max_bytes`, default 2 GiB) rejects with `RESOURCE_EXHAUSTED` while there is still room, and reads keep working.
3. **What do clients see?** The rpc fails, so no workflow is told its append succeeded; callers retry with the same idempotency key once space returns, and nothing is duplicated.
4. **Detect early.** `tl_durable_wal_bytes` compared with the quota (the WAL size panel of `obs.05`) shows the approach; alert at a fraction such as 80%.
5. **Recover without touching segments.** Raise the cap or expand the volume and restart: writes resume. Deleting segments by hand would destroy acknowledged history.

### 5.2 The model runbook

`docs/runbooks/EventlogDiskFull.md`:

````markdown
# Runbook: event log disk full

Owner: the durable control-plane on-call (`<system>-durable`).

## Symptoms

Workflow starts and completions fail with `RESOURCE_EXHAUSTED`; the WAL size
panel shows `sum(tl_durable_wal_bytes)` at `wal_max_bytes`; reads
(`wf describe`) still work; queues stop draining.

## Diagnosis

```bash
kubectl -n <system> logs statefulset/<system>-durable --tail=100 | grep -i exhausted
kubectl -n <system> exec <system>-durable-0 -- df -h /var/lib/durable/wal
```

Confirm the quota, not an I/O error, is the cause, and how much space the
volume has left.

## Containment

Pause new submissions. Do not delete WAL segments: they are the only copy of
every workflow's history.

## Recovery

Raise `walMaxBytes` in the durable chart (or expand the volume first if the
filesystem is the limit), `helm upgrade`, and wait for the rollout. Confirm a
canary workflow appends and completes.

## Verification

Retry the failed client requests with their original idempotency keys; no
workflow reports success for an append that was rejected; add or tune the
alert on WAL bytes at 80% of the quota.
````

### 5.3 Why this solution

- **A quota below the disk** turns an uncontrolled failure into a clean, documented one.
- **Reject, do not degrade**: acknowledging an append that was not durable would break the engine's only promise.
- **Why not delete old segments to make room?** They hold acknowledged history that recovery and replay need; compaction is a designed operation, not an incident action.

### 5.4 Explain it back

1. Why is no work lost when the WAL is full?
   <details><summary>Answer</summary>An append is acknowledged only after it is written and fsynced; at quota the append fails with `RESOURCE_EXHAUSTED` and appends nothing, so every acknowledged event is still there and nothing unacknowledged was claimed.</details>
2. Why do reads keep working at quota?
   <details><summary>Answer</summary>Reads do not append; the quota only gates new records, so history stays readable for diagnosis.</details>
3. Why retry failed requests with the same idempotency keys?
   <details><summary>Answer</summary>If any did succeed before the quota hit, the retry returns the same result instead of creating a duplicate.</details>

## 6. Pitfalls

| # | Pitfall | Symptom | How the solution avoids it |
|---|---|---|---|
| 1 | Deleting WAL segments to free space | acknowledged history lost; workflows cannot replay | the runbook forbids it; recovery raises the cap or expands the volume |
| 2 | No alert before the quota | the first signal is clients failing | alert on `tl_durable_wal_bytes` at 80% of `wal_max_bytes` |
| 3 | Retrying with new keys | duplicate workflows after recovery | retries reuse the original idempotency keys |

## 7. Where it's used next

| Direction | Module | Connection |
|---|---|---|
| Back | `ops.10` | A prerequisite: the contract or evidence this runbook relies on. |
| Back | `dur.01` | A prerequisite: the contract or evidence this runbook relies on. |

## Going further

| Resource | What to inspect |
|---|---|
| [Incident drill catalog](README.md) | Compare this inject with the adjacent failure drills and use the same containment and verification fields. |
