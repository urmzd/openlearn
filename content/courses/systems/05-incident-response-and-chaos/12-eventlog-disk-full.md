# The event log disk fills up

A full event-log disk can stop workflow progress and, handled badly, risk losing acknowledged work. [The durable server is SIGKILLed in a loop during a corpus build](03-durable-kill9.md) showed that the log of [The append-only event log](../../ai-platform-engineering/05-durable-orchestration-and-workers/01-event-log.md) in *AI Platform Engineering* survives a kill because it acknowledges only after fsync. This drill puts the same log under a different pressure: no more room. The WAL quota provides a controlled boundary, and the drill checks that the boundary behaves as the contract says.

The contract is short. An append is acknowledged only after its fsync; an append that would exceed `[durable].wal_max_bytes` fails with gRPC `RESOURCE_EXHAUSTED`, appends nothing, and leaves reads working. So a full log stops progress but never loses or invents acknowledged work: no workflow reports success for a rejected append. The drill follows [A runaway agent](11-runaway-agent.md) in the sequence of operations drills.

The section states the principles and the drill scenario, lists what a correct runbook must cover, and derives the procedure, ending with the model runbook `docs/runbooks/EventlogDiskFull.md`. Try to write the procedure yourself before you read the solution.

> **Learning objectives**
> - Explain why a WAL quota below the disk size turns an uncontrolled failure into a clean rejection.
> - Show that at quota no acknowledged work is lost and no rejected append is reported as success.
> - Alert on WAL bytes against the quota before the filesystem fills.
> - Recover by raising the cap or expanding the volume, never by deleting segments, and retry with the original idempotency keys.

## Principles

An append is acknowledged only after its durability point. Reject new work cleanly at quota, preserve readable history, and alert before the filesystem is exhausted. Alert on `tl_durable_wal_bytes` against the quota *before* the filesystem fills; recovery is raising the cap (or freeing space) and restarting.

## The drill scenario

Fill the test WAL to its configured limit. Confirm the next append fails with `RESOURCE_EXHAUSTED` and no workflow reports success for it. Free or expand space and resume.

## What a correct runbook must cover

A runbook for a full log is complete when it answers five questions:

| Case | What the runbook must contain | Why it matters |
|---|---|---|
| Owner and detection | `tl_durable_wal_bytes` near `wal_max_bytes`; clients seeing `RESOURCE_EXHAUSTED` | a symptom and its metric |
| Safety | rejected appends append nothing; reads keep working; acknowledged history intact | no data lost or invented |
| Containment | pause new submissions; do not delete segments by hand | protects the only copy of history |
| Recovery | raise `wal_max_bytes` (`walMaxBytes` in the chart) or expand the volume, restart, confirm writes resume | progress resumes |
| Verification | a canary workflow completes; affected clients retried with the same idempotency keys | done means measured |

## Deriving the procedure

### Reasoning from first principles

What does the contract promise? Every append rpc answers only after its records are written and fsynced, and an append that fails (I/O error or quota) leaves the log as it was. So at quota the safe outcome is a clean rejection, not a partial write.

Why a quota below the disk size? A filesystem that fills mid-write can fail in ways the server cannot control. A configured cap (`wal_max_bytes`, default 2 GiB) rejects with `RESOURCE_EXHAUSTED` while there is still room, and reads keep working.

What do clients see? The rpc fails, so no workflow is told its append succeeded; callers retry with the same idempotency key once space returns, and nothing is duplicated.

Detect early. `tl_durable_wal_bytes` compared with the quota (the WAL size panel of the control-plane dashboard from [Observability for the control plane](../04-observability/06-control-plane-observability.md)) shows the approach; alert at a fraction such as 80%.

Recover without touching segments. Raise the cap or expand the volume and restart: writes resume. Deleting segments by hand would destroy acknowledged history.

### The model runbook

The procedure written out as `docs/runbooks/EventlogDiskFull.md`:

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

### Why this solution

A quota below the disk turns an uncontrolled failure into a clean, documented one. Reject, do not degrade: acknowledging an append that was not durable would break the engine's only promise. And deleting old segments to make room, the instinctive fix, destroys acknowledged history that recovery and replay need; compaction is a designed operation, not an incident action.

## What can go wrong

Three mistakes turn a stall into data loss or duplication:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| Deleting WAL segments to free space | acknowledged history lost; workflows cannot replay | the runbook forbids it; recovery raises the cap or expands the volume |
| No alert before the quota | the first signal is clients failing | alert on `tl_durable_wal_bytes` at 80% of `wal_max_bytes` |
| Retrying with new keys | duplicate workflows after recovery | retries reuse the original idempotency keys |

> **Warning:** WAL segments are the only copy of every workflow's history. A full log costs progress; deleting segments costs the history itself.

## Where this leads

The contract tested here is that of [The append-only event log](../../ai-platform-engineering/05-durable-orchestration-and-workers/01-event-log.md), and the drill follows [A runaway agent](11-runaway-agent.md). The last, optional drill, [A Raft partition](13-raft-partition.md), replicates the same log across three servers and asks what a network split does to it. To compare this inject with the adjacent drills and reuse the same containment and verification fields, see the [chapter opening](README.md).

## Key terms

- **WAL quota**: `[durable].wal_max_bytes` (default 2 GiB, `walMaxBytes` in the chart), the cap above which appends are rejected.
- **`RESOURCE_EXHAUSTED`**: the gRPC code for an append rejected at quota; nothing is appended.
- **Durability point**: the fsync after which an append is acknowledged.
- **Compaction**: a designed operation that reclaims log space safely, as opposed to deleting segments by hand.

## Summary

- An append is acknowledged only after its fsync; an append past `wal_max_bytes` fails with `RESOURCE_EXHAUSTED`, appends nothing, and leaves reads working.
- A full log stops progress but never loses or invents acknowledged work.
- Alert on `tl_durable_wal_bytes` at a fraction of the quota, such as 80%, before the filesystem fills.
- Recover by raising the cap or expanding the volume and restarting; never delete segments; retry with the original idempotency keys.

## Review questions

1. Why is no work lost when the WAL is full?
   <details><summary>Answer</summary>An append is acknowledged only after it is written and fsynced; at quota the append fails with `RESOURCE_EXHAUSTED` and appends nothing, so every acknowledged event is still there and nothing unacknowledged was claimed.</details>
2. Why do reads keep working at quota?
   <details><summary>Answer</summary>Reads do not append; the quota only gates new records, so history stays readable for diagnosis.</details>
3. Why retry failed requests with the same idempotency keys?
   <details><summary>Answer</summary>If any did succeed before the quota hit, the retry returns the same result instead of creating a duplicate.</details>
