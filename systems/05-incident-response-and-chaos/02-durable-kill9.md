<!-- ss:module ops.02 -->
# The durable server is SIGKILLed in a loop during a corpus build

## Overview

| | |
|---|---|
| **Module** | `ops.02` · drill · ops, docs · Pass 8 · 2 to 3 h |
| **You build** | `deploy/observability/rules/durable.yaml` (the alerts `DurableRedeliveriesSpike` and `DurableDeadLetters`, shared with `ops.03`), `docs/runbooks/DurableRedeliveriesSpike.md` (Symptoms, Diagnosis, Mitigation), and `docs/postmortems/<date>-durable-kill9.md`; and you show that a corpus build survives five SIGKILLs of its server with the output of a clean run |
| **Contract** | the drill spec `course/drills/durable-kill9/drill.toml` (what is injected, detected, and checked: section 4); `tl_durable_redeliveries_total`, `tl_durable_task_queue_depth`, and `tl_durable_dlq_size` in [`otel/metrics.yaml`](../../course/contracts/otel/metrics.yaml); the `ctl` verbs fixed by MS-durable (`course/milestones/MS-durable.toml`) |
| **Tests** | graded by `ss drill end`; no course tests to read (section 4 lists every check) |
| **Needs** | `dep.06` (the durable StatefulSet and the worker Deployment it kills pods of) and `obs.05` (the scrapes and the control-plane dashboard you diagnose on); reading: `ops.01` (how a drill runs), `dur.09` (how a killed activity resumes), and the log, lease, and replay chapters of this pass |
| **Used by** | no call site: a drill. MS-durable requires it; MS-ops reruns the drills |
| **Milestone** | MS-durable (drill `ops.02` resolved) |
| **Optional depth** | *Site Reliability Engineering*, ch. 22 Addressing Cascading Failures and ch. 26 Data Integrity ([free](https://sre.google/sre-book/table-of-contents/)); [Temporal: activity timeouts and retries](https://docs.temporal.io/encyclopedia/detecting-activity-failures) (free); Kleppmann, *Designing Data-Intensive Applications*, ch. 8 (process pauses and leases) |

## Key Takeaways

- Killing the server of a durable engine must cost **time, never correctness**: every acknowledged event is in the fsynced log, and recovery truncates only the torn record that was never acknowledged.
- **Leases** turn a dead worker or a dead server into a timeout: work in flight is delivered again after the visibility timeout, so each kill shows up as a **redeliveries spike**, which is what pages.
- Redelivery is safe only because activities are **idempotent**: a rerun with the same key finds its work directory and resumes or re-emits `DONE.json`. The proof is the output, not the logs: the corpus equals a clean run's and the ledger has every row once.
- Mitigation for a self-healing fault is to **confirm, not to act**: the StatefulSet restarts the pod on the same claim. Deleting the WAL claim "to reset" is the one action that loses data.
- The postmortem's action items are about **exposure**: one replica, and long activities without checkpoints that redo minutes of work.

## How to work this chapter

```bash
kubectl apply -f deploy/observability/rules/durable.yaml   # your two alerts (section 4)
{ctl} data build --config course/fixtures/MS-corpus/corpus.toml --id drill-kill9 --durable 127.0.0.1:30733 --json
ss drill start durable-kill9      # five SIGKILLs of the durable pod, 30 s apart, then only the page
# ... dashboard, kubectl, the runbook; wait for the build ...
{ctl} data build --config course/fixtures/MS-corpus/corpus.toml --id drill-kill9 --durable 127.0.0.1:30733 --wait --json
TL_ARTIFACTS=artifacts {corpus} ledger verify --config course/fixtures/MS-corpus/corpus.toml
ss drill end                      # grades detection, resolution, evidence, runbook, postmortem
ss drill reset
```

`{ctl}` and `{corpus}` are your `[entry].ctl` and `[entry].corpus` (`spec/cli-roles.md`); the `ctl` verbs are fixed in MS-durable. `ss drill start` blocks while it kills (about two and a half minutes), so the build starts first. The two commands after the wait are the pass condition `ss drill end` also runs itself (section 4).

---

## 1. Why now

Pass 8 built a durable execution engine: a write-ahead log (`dur.01`), leases with visibility timeouts and a dead-letter queue (`dur.03`), workers (`dur.04`), idempotent activities (`dur.05`, `dur.09`), and replayed workflows (`dur.06`), deployed on kind as a StatefulSet with its log on a persistent volume (`dep.06`) and observed by `obs.05`. Every one of those modules was tested with kills in a test process. This drill does it to the deployed system, during a real corpus build, the way a node failure or an OOM kill would: SIGKILL, five times, with no chance to flush or shut down cleanly. The question is the one the whole pass exists to answer: after it, is the result exactly what an undisturbed run produces?

## 2. Principles

### 2.1 What SIGKILL leaves behind

| Symbol | Meaning | Type |
|---|---|---|
| $k$ | number of kills | 5 here |
| $\Delta$ | time between kills | 30 s |
| $r$ | time for the pod to restart and recover the log | seconds (about 8) |
| $V$ | visibility timeout of an activity lease | 30 s (`[durable].visibility_timeout_ms`) |
| $a_i$ | activity tasks whose lease was live at kill $i$ | count |

SIGKILL cannot be caught: the process ends between two instructions. What survives is what was on disk: the log up to the last fsync. An append is acknowledged only after its fsync (`dur.01`), so every acknowledged event survives; the record being written at the instant of the kill may be torn, and recovery truncates it, which is correct because nobody was told it happened.

### 2.2 Leases turn a death into a timeout

A worker holds each activity task under a lease that expires $V$ after its last heartbeat. While the server is down ($r$ seconds), workers can neither heartbeat nor complete. After the restart the server's queue state is rebuilt from the log with the leases' deadlines, and every lease whose deadline passes without a completion makes its task visible again: a **redelivery**. So each kill produces about $a_i$ redeliveries roughly $V$ after it, and the counter `tl_durable_redeliveries_total` climbs in steps.

### 2.3 Why the second delivery has no second effect

A redelivered activity runs again with the same idempotency key (`TL_IDEMPOTENCY_KEY`), in the same work directory (`spec/subprocess-activity.md`). Outputs are written as `<name>.tmp` and renamed, `DONE.json` last; a rerun that finds `DONE.json` re-emits its result and exits. So whether the first attempt finished, half-finished, or never started, the second leaves the same files. The ledger (`data.08`) records each source document and shard by content hash, so a duplicate would show up as a second row, and `{corpus} ledger verify` fails on it.

### 2.4 Detection: alert on the symptom you can see

The page comes from the redeliveries rate, not from pod restarts: a restart that causes no redelivery hurts nobody, and redeliveries also catch dying workers and activities that outlive their lease. Your rule (section 4) fires when one queue sees more than 5 redeliveries in 2 minutes for 20 s. Normal churn (a deploy, a preempted node) stays under it.

## 3. Worked example by hand

$k = 5$ kills at $t = 20, 50, 80, 110, 140$ s after `ss drill start`, $r = 8$ s, $V = 30$ s. The build is in its dedup and shard stages, with about 3 activity tasks leased at any moment.

| Kill | Server down | Leases that expire unanswered | Redeliveries (cumulative) |
|---|---|---|---|
| 1 at 20 s | 20 to 28 s | the 3 live at 20 s, at about 50 s | 3 |
| 2 at 50 s | 50 to 58 s | 3 more, at about 80 s | 6 |
| 3 at 80 s | 80 to 88 s | 3 more, at about 110 s | 9 |
| 4 at 110 s | 110 to 118 s | 2 (a quieter stage), at about 140 s | 11 |
| 5 at 140 s | 140 to 148 s | 3, at about 170 s | 14 |

The alert's expression is the increase over 2 minutes. At $t = 80$ s the window holds 6, more than 5; with `for: 20s` it fires at about $t = 100$ s, 80 s after the first kill, inside the drill's 300 s. Without the kills the build runs about 150 s; with them each kill costs about $r + V \approx 38$ s for the tasks it hit, and the stages overlap, so expect roughly 3 to 4 extra minutes, not $5 \times 38$.

Then check the output by hand: the clean numbers of the MS-corpus fixture are 169 documents, 3 shards, 75164 training tokens, and 6868 validation tokens. The same start run again (`--wait`) must report exactly those, with `started: false` (no second run was created).

## 4. Inject, detect, mitigate, verify

| Step | What `ss drill` does | What you do |
|---|---|---|
| inject | 20 s after `start`, `kubectl delete pod --grace-period=0 --force` on a Ready pod of `{deploy.services.durable}`; four more, each 30 s after the previous | start the build first; then nothing: the page is your first signal |
| detect | at `end`, reads the `ALERTS` series: `DurableRedeliveriesSpike` must have fired within 300 s of the first kill | open the control-plane dashboard (`obs.05`) and your runbook |
| mitigate | | confirm the server recovers each time (logs: torn tail truncated, no `ErrCorrupt`), the workers reconnect, and the build progresses; touch nothing that holds state |
| verify | at `end`: both rollouts complete; dead letters at 0 and every queue at depth 0, each held 60 s; a `workflow CorpusBuild` trace in the window | run the two pass-condition commands below; write the runbook and the postmortem |

Your `system.toml` needs `[deploy].services.durable` (`statefulset/<system>-durable`), `[deploy].services.worker` (`deploy/<system>-worker`), `[deploy].prometheus`, `[deploy].traces`, and `[entry].ctl` and `[entry].corpus`.

Your rules file, a `PrometheusRule` with the label `release: observability` (`contracts/helm/observability.md`), holds both durable alerts:

```yaml
- alert: DurableRedeliveriesSpike
  expr: sum by (namespace, queue) (increase(tl_durable_redeliveries_total{namespace="<system>"}[2m])) > 5
  for: 20s
  labels: {severity: page}
  annotations: {runbook: docs/runbooks/DurableRedeliveriesSpike.md}
- alert: DurableDeadLetters
  expr: sum by (namespace, queue) (tl_durable_dlq_size{namespace="<system>"}) > 0
  for: 15s
  labels: {severity: page}
  annotations: {runbook: docs/runbooks/DurableDeadLetters.md}
```

### What `ss drill end` checks

| Check | Passes when |
|---|---|
| detected | `DurableRedeliveriesSpike` fired within `within_s = 300` of the first kill (TTD is reported) |
| resolved | `rollout status` of the durable StatefulSet and the worker Deployment; `tl_durable_dlq_size` and `tl_durable_task_queue_depth` summed to 0, each held 60 s |
| trace evidence | a trace in the window has the span `workflow CorpusBuild` from `<system>-worker` |
| runbook | `docs/runbooks/DurableRedeliveriesSpike.md` has Symptoms, Diagnosis, Mitigation |
| postmortem | `docs/postmortems/<date>-durable-kill9.md` has the seven sections |
| time limit | 45 minutes from start to end (not graded for the CI responder) |

The drill's pass condition also holds two commands, `[[resolve.run]]` in the spec, that `ss drill end` runs and grades: the same start with `--wait` returns `COMPLETED` with 169 documents, 3 shards, 75164 and 6868 tokens; and `{corpus} ledger verify` over `./artifacts` exits 0. Run both yourself first, and put their output in your postmortem's Resolution.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. Alerting on restarts instead of redeliveries, or no durable rule loaded | no page, or a page for restarts that hurt nobody | ss drill end, check detected |
| 2. Deleting the WAL claim or scaling the StatefulSet to 0 to "reset" the server | every workflow's history is gone; the build never completes | the rollout and queue-depth checks; the build command fails |
| 3. An activity that writes outputs in place, without a temporary name and a rename | a redelivered attempt sees a half-written shard and keeps it; the token counts differ | the pass-condition commands (clean-run numbers, ledger verify) |
| 4. A rerun that ignores the idempotency key | a shard written twice, two ledger rows | ledger verify |
| 5. Declaring the incident over while the build still runs | the queue-depth check fails its hold | resolve check 4 |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `dep.06` | the StatefulSet, its WAL claim, and the worker Deployment the kills land on |
| Back | `obs.05` | the dashboard and scrapes the diagnosis reads |
| Forward | `ops.03` | the second alert of the same rules file, and a fault that does not heal itself |
| Forward | `ops.11` | the same log under a full disk instead of a kill |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| one durable replica | Temporal's history service on Cassandra or Postgres; Raft-replicated logs | a kill is not an outage | `dur.10` (optional), etcd |
| a kill loop in a drill | Jepsen, Chaos Mesh `PodChaos` on a schedule | randomized faults with a checker over the history | [jepsen.io](https://jepsen.io/) |
| a fixed visibility timeout | heartbeat timeouts per activity, sticky queues | a dead worker is noticed in seconds, not a full timeout | Temporal docs |
| a ledger check after the fact | end-to-end checksums and exactly-once sinks | duplicates caught at write time | Kafka transactions, Flink two-phase commit |
