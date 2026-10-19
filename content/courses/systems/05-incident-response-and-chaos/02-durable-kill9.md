# The durable server is SIGKILLed in a loop during a corpus build

*AI Platform Engineering* builds a durable execution engine piece by piece: a write-ahead log ([The append-only event log](../../ai-platform-engineering/05-durable-orchestration-and-workers/01-event-log.md)), leases with visibility timeouts and a dead-letter queue ([The task queue](../../ai-platform-engineering/05-durable-orchestration-and-workers/03-task-queue.md)), workers ([The task protocol and workers](../../ai-platform-engineering/05-durable-orchestration-and-workers/04-task-protocol-and-workers.md)), idempotent activities ([Activities](../../ai-platform-engineering/05-durable-orchestration-and-workers/05-activities.md) and [Subprocess activities](../../ai-platform-engineering/05-durable-orchestration-and-workers/09-subprocess-activities.md)), and replayed workflows ([Deterministic replay](../../ai-platform-engineering/05-durable-orchestration-and-workers/06-deterministic-replay.md)). [Durable and worker images and charts](../../infrastructure/01-containers-kubernetes/06-durable-and-worker-charts.md) in *Infrastructure* deploys it on kind as a StatefulSet with its log on a persistent volume, and [Observability for the control plane](../04-observability/05-control-plane-observability.md) supplies the scrapes and dashboard. Every one of those pieces was exercised with kills in a test process.

This drill does it to the deployed system, during a real corpus build, the way a node failure or an OOM kill would: SIGKILL, five times, with no chance to flush or shut down cleanly. The question is the one the whole durable engine exists to answer: after it, is the result exactly what an undisturbed run produces? Killing the server of a durable engine must cost time, never correctness.

The section assumes you know how a drill runs from [A decode worker dies mid-stream](01-kill-decode.md). It builds the reasoning (what SIGKILL leaves, how leases turn a death into a redelivery, why a redelivery has no second effect), works the kill schedule by hand, states what is injected and what "resolved" means, and walks through the response. The solution consists of the alert rules [`deploy/observability/rules/durable.yaml`](../../../../course/solutions/entry/deploy/observability/rules/durable.yaml) (`DurableRedeliveriesSpike` and `DurableDeadLetters`, shared with [A poison task lands in the dead-letter queue](03-poison-task.md)), the model runbook [`docs/runbooks/DurableRedeliveriesSpike.md`](../../../../course/solutions/docs/runbooks/DurableRedeliveriesSpike.md), and the model postmortem [`docs/postmortems/2026-10-09-durable-kill9.md`](../../../../course/solutions/docs/postmortems/2026-10-09-durable-kill9.md). The metrics it reads, `tl_durable_redeliveries_total`, `tl_durable_task_queue_depth`, and `tl_durable_dlq_size`, are defined in [`otel/metrics.yaml`](../../../../course/contracts/otel/metrics.yaml), and the `ctl` verbs in `spec/cli-roles.md`.

The drill runs against a corpus build started first, with the solution's alert rules loaded. `{ctl}` and `{corpus}` are the system's control CLI and corpus CLI:

```bash
kubectl apply -f deploy/observability/rules/durable.yaml   # the two durable alerts
{ctl} data build --config <small reference corpus config> --id drill-kill9 --durable 127.0.0.1:30733 --json
```

> **Learning objectives**
> - Explain what a SIGKILL can and cannot destroy in an fsync-before-acknowledge log.
> - Trace how leases and visibility timeouts turn each kill into redeliveries about $V$ seconds later, and compute when the alert fires.
> - Explain why idempotent activities make a second delivery harmless, and verify it by output rather than by health.
> - Mitigate a self-healing fault by confirming recovery, and name the one action that loses data.
> - Write a postmortem that separates the trigger from the exposure.

## What SIGKILL leaves behind

Name the quantities first:

| Symbol | Meaning | Type |
|---|---|---|
| $k$ | number of kills | 5 here |
| $\Delta$ | time between kills | 30 s |
| $r$ | time for the pod to restart and recover the log | seconds (about 8) |
| $V$ | visibility timeout of an activity lease | 30 s (`[durable].visibility_timeout_ms`) |
| $a_i$ | activity tasks whose lease was live at kill $i$ | count |

SIGKILL cannot be caught: the process ends between two instructions. What survives is what was on disk: the log up to the last fsync. An append is acknowledged only after its fsync, so every acknowledged event survives. The record being written at the instant of the kill may be torn, and recovery truncates it, which is correct because nobody was told it happened.

## Leases turn a death into a timeout

The log explains why nothing acknowledged is lost. Leases explain what happens to the work in flight. A worker holds each activity task under a lease that expires $V$ after its last heartbeat. While the server is down ($r$ seconds), workers can neither heartbeat nor complete. After the restart the server's queue state is rebuilt from the log with the leases' deadlines, and every lease whose deadline passes without a completion makes its task visible again: a **redelivery**. So each kill produces about $a_i$ redeliveries roughly $V$ after it, and the counter `tl_durable_redeliveries_total` climbs in steps.

## Why the second delivery has no second effect

A redelivered activity runs again with the same idempotency key (`TL_IDEMPOTENCY_KEY`), in the same work directory (`spec/subprocess-activity.md`). Outputs are written as `<name>.tmp` and renamed, `DONE.json` last; a rerun that finds `DONE.json` re-emits its result and exits. So whether the first attempt finished, half-finished, or never started, the second leaves the same files. The ledger of [Licensing ledger verification and the datasheet](../../data-engineering/05-corpus-pipeline/08-ledger-and-datasheet.md) in *Data Engineering* records each source document and shard by content hash, so a duplicate would show up as a second row, and `{corpus} ledger verify` fails on it. The proof of correctness is therefore the output, not the logs: the corpus must equal a clean run's, and the ledger must have every row once.

## Detection: alert on the symptom you can see

The page comes from the redeliveries rate, not from pod restarts. A restart that causes no redelivery hurts nobody, and redeliveries also catch dying workers and activities that outlive their lease. The solution's rule fires when one queue sees more than 5 redeliveries in 2 minutes for 20 s. Normal churn (a deploy, a preempted node) stays under it.

## Working the kill schedule by hand

Take $k = 5$ kills at $t = 20, 50, 80, 110, 140$ s after the drill starts, $r = 8$ s, $V = 30$ s. The build is in its dedup and shard stages, with about 3 activity tasks leased at any moment. Each kill's redeliveries arrive about $V$ later:

| Kill | Server down | Leases that expire unanswered | Redeliveries (cumulative) |
|---|---|---|---|
| 1 at 20 s | 20 to 28 s | the 3 live at 20 s, at about 50 s | 3 |
| 2 at 50 s | 50 to 58 s | 3 more, at about 80 s | 6 |
| 3 at 80 s | 80 to 88 s | 3 more, at about 110 s | 9 |
| 4 at 110 s | 110 to 118 s | 2 (a quieter stage), at about 140 s | 11 |
| 5 at 140 s | 140 to 148 s | 3, at about 170 s | 14 |

The alert's expression is the increase over 2 minutes. At $t = 80$ s the window holds 6, more than 5; with `for: 20s` it fires at about $t = 100$ s, 80 s after the first kill, inside the drill's 300 s. Without the kills the build runs about 150 s. With them, each kill costs about $r + V \approx 38$ s for the tasks it hit, and the stages overlap, so expect roughly 3 to 4 extra minutes, not $5 \times 38$.

Then check the output by hand. A clean build of the small reference corpus config gives 169 documents, 3 shards, 75164 training tokens, and 6868 validation tokens. The same start run again (`--wait`) must report exactly those, with `started: false` (no second run was created).

## The incident: what is injected, detected, and resolved

The drill runs in four steps:

| Step | What happens |
|---|---|
| workload | `{ctl} data build --config <small reference corpus config> --id drill-kill9 --durable 127.0.0.1:30733 --json` is running |
| inject | 20 s in, `kubectl delete pod --grace-period=0 --force` on the Ready pod of `statefulset/<system>-durable`; four more, each 30 s after the previous |
| detect | `DurableRedeliveriesSpike` must fire within 300 s of the first kill |
| page | *DurableRedeliveriesSpike: activity tasks are being delivered again and again; the corpus build drill-kill9 has not finished* |

The incident is resolved when all of these hold:

| Condition | Holds for |
|---|---|
| `rollout status` of the durable StatefulSet and the worker Deployment complete | within 120 s each |
| `(sum(tl_durable_dlq_size{namespace="<system>"}) or vector(0)) == 0` | 60 s |
| `(sum(tl_durable_task_queue_depth{namespace="<system>"}) or vector(0)) == 0` | 60 s |
| the same start with `--wait --json` returns `COMPLETED` with `n_docs` 169, `n_shards` 3, `train_tokens` 75164, `val_tokens` 6868 | the clean-run numbers |
| `TL_ARTIFACTS=artifacts {corpus} ledger verify --config <small reference corpus config>` exits 0 | every ledger row once |
| a trace in the window has `workflow CorpusBuild` from `<system>-worker` | evidence for the timeline |
| `docs/runbooks/DurableRedeliveriesSpike.md` (Symptoms, Diagnosis, Mitigation) and `docs/postmortems/<date>-durable-kill9.md` (the seven sections) exist | at the end |
| all within 45 minutes | |

The two commands are the real pass condition: they prove the output equals an undisturbed run's, which no metric can.

## Working the incident step by step

### Reasoning from first principles

Start with what a SIGKILL can break: only what was not yet on disk. Acknowledged events were fsynced first, so the only possible loss is the record being written, which nobody was told about. The responder's first job is to confirm recovery did exactly that: a torn tail truncated, nothing more.

Next, why redeliveries appear. While the server is down, workers cannot heartbeat or complete, so leases expire $V$ after the kill and the tasks become visible again. Redeliveries are the designed recovery, not a malfunction, and they lag each kill by about $V$. They are also the right thing to alert on, rather than restarts, because they measure the effect users feel (redone work, slower builds) and also catch dying workers and activities that outlive their lease. A threshold above normal churn (more than 5 per queue in 2 minutes, held 20 s) separates an incident from a deploy.

Is any effect happening twice? Not if activities are idempotent, and the proof is the output: the clean-run counts and a ledger with every row once.

What should the responder do? The StatefulSet restarts the pod on the same WAL claim, and the workers reconnect with backoff. So mitigation is to confirm, and to touch nothing that holds state. Deleting the claim or scaling to 0 is the one action that turns a slow build into data loss.

Finally, the root cause is not the kill. A single replica makes every kill a control-plane outage, and long activities without checkpoints redo minutes of work per redelivery. Those exposures are the action items.

### The alert rules

Both durable alerts live in one `PrometheusRule`, evaluated every 5 s so a 20 s `for` is several evaluations:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: forge-durable
  namespace: observability
  labels:
    release: observability          # the kube-prometheus-stack rule selector
    app.kubernetes.io/part-of: forge
spec:
  groups:
    - name: forge-durable
      interval: 5s
      rules:
        # Workers or the server keep dying: tasks whose lease expired are
        # handed out again. A handful per minute is normal churn (a deploy, a
        # preempted node); more than 5 in 2 minutes for 20 s is not.
        - alert: DurableRedeliveriesSpike
          expr: sum by (namespace, queue) (increase(tl_durable_redeliveries_total{namespace="forge"}[2m])) > 5
          for: 20s
          labels:
            severity: page
          annotations:
            summary: "{{ $labels.queue }}: {{ $value | humanize }} redeliveries in 2 minutes"
            runbook: docs/runbooks/DurableRedeliveriesSpike.md
        # A task exhausted its attempts. Nothing retries it any more: the
        # workflow waiting on it is stuck until someone redrives it.
        - alert: DurableDeadLetters
          expr: sum by (namespace, queue) (tl_durable_dlq_size{namespace="forge"}) > 0
          for: 15s
          # ... severity: page, runbook: docs/runbooks/DurableDeadLetters.md
```

`increase(...[2m])` is used rather than `rate` because the threshold is a count a human can reason about ("more than 5 tasks redone"), and `by (namespace, queue)` makes the page name the queue. The full rules are in [`rules/durable.yaml`](../../../../course/solutions/entry/deploy/observability/rules/durable.yaml). With the worked kill schedule, the 2-minute increase is 6 at about $t = 80$ s, and `for: 20s` fires the page at about $t = 100$ s, 80 s after the first kill.

### Diagnosis

The first two steps are commands:

```bash
# 1. Who is restarting: the server, the workers, or neither? RESTARTS growing
#    on <system>-durable-0 and Killing events: a server that keeps dying. A
#    log that ends mid-line with no shutdown message was killed, not stopped.
kubectl -n <system> get pods -o wide
kubectl -n <system> get events --sort-by=.lastTimestamp | tail -20
kubectl -n <system> logs statefulset/<system>-durable --previous --tail=50

# 2. Did the server recover its log each time? A truncated torn tail is
#    expected after a kill. ErrCorrupt in an older segment is not: stop and
#    page the owner, data loss is possible.
kubectl -n <system> logs statefulset/<system>-durable --tail=200 | grep -E 'recover|torn|truncat'
```

The third step applies if the server is steady: then the workers are dying instead. Check `kubectl describe pod` for `Reason: OOMKilled`, and `{ctl} wf describe <id> --json` for a `last_failure.type` of `Timeout` (an activity outliving its lease without heartbeating). The fourth step asks whether work is still done exactly once. A redelivered activity runs with the same idempotency key, finds its work directory, and resumes or re-emits `DONE.json`. Duplicated outputs would mean an activity ignores its key: a bug, not this incident.

On the control-plane dashboard the same story is one screen: the redeliveries rate in steps, queue depth that stops falling, schedule-to-start growing, and the WAL size still climbing (every redelivery is a record).

### Mitigation and verification

```bash
# Confirm the server comes back Ready on the same claim, and stays Ready.
kubectl -n <system> rollout status statefulset/<system>-durable
# Never: deleting the WAL PVC, or scaling the StatefulSet to 0 "to reset it".

# When redeliveries stop and queues drain: the workflows complete, nothing is
# dead-lettered.
{ctl} wf describe <workflow id> --durable 127.0.0.1:30733 --json
{ctl} wf dlq list data --durable 127.0.0.1:30733 --json

# The proof: the same start returns the finished run (started: false) with
# the clean-run numbers, and the ledger holds every row once.
{ctl} data build --config <small reference corpus config> --id drill-kill9 --durable 127.0.0.1:30733 --wait --json
TL_ARTIFACTS=artifacts {corpus} ledger verify --config <small reference corpus config>
```

`started: false` matters as much as the counts: it shows the start was idempotent too, returning the existing workflow instead of creating a second build.

### The model runbook

The runbook first explains what the alert means and the three ways it happens, so the reader can classify before acting:

```markdown
The alert means: more than 5 activity tasks on one queue were delivered
again in the last 2 minutes, held for 20 s. A redelivery happens when a
task's lease expired before the worker completed it: the worker died, the
server died and the lease ran out while it was down, or the activity runs
longer than its visibility timeout without heartbeating.
```

Its Mitigation states the one forbidden action plainly: "Do not delete the WAL claim and do not scale the StatefulSet to 0 'to reset it': the log is the only copy of every workflow's history." And it answers the third cause correctly: "an activity that runs past its visibility timeout needs heartbeats, not a longer timeout" (a longer timeout would make every real worker death take longer to recover). The full runbook is [`DurableRedeliveriesSpike.md`](../../../../course/solutions/docs/runbooks/DurableRedeliveriesSpike.md).

### The model postmortem

The model postmortem states its impact in the terms that matter, correctness first: "No acknowledged event was lost, no activity's effect happened twice, and the build completed with the same 169 documents, 3 shards, and token counts as a clean run, 3 minutes 40 seconds later than a clean run." Its root cause separates trigger from exposure:

```markdown
## Root cause

The durable server is a single replica, so every kill is a full outage of
the control plane for the time it takes to start and recover the log (about
8 s here). Activities whose worker could not reach the server to heartbeat
or complete kept their leases until the visibility timeout (30 s), and were
then delivered again. That is the design working: leases make a lost worker
or server safe, at the cost of redoing work. The drill's kills are the
trigger; the exposure is one replica and a 30 s visibility timeout on
activities that do not checkpoint.
```

Its action items each reduce that exposure:

| Action | Type | Owner |
|---|---|---|
| Alert on durable server restarts (`kube_pod_container_status_restarts_total`), not only on their effect | detect | control plane |
| Heartbeat from the long corpus stages so a redelivery resumes instead of redoing a stage | mitigate | data pipeline |
| Run the durable server with three replicas ([Raft high availability](../../ai-platform-engineering/05-durable-orchestration-and-workers/10-raft-ha.md), optional) so one kill is not an outage | prevent | control plane |
| Make the CLI's start retry visible (a log line per retry) so users see an outage, not a slow command | detect | CLI |

The full postmortem is [`2026-10-09-durable-kill9.md`](../../../../course/solutions/docs/postmortems/2026-10-09-durable-kill9.md).

### Why this solution

Confirm, do not act. Every component already does the right thing: the StatefulSet restarts the pod on its claim, recovery truncates the torn tail, leases redeliver, activities are idempotent. Any intervention can only add risk, and the one tempting intervention (resetting state) destroys the only copy of history.

Verify by output, not by health. Healthy pods and an empty queue say the system is up; only the clean-run counts and `ledger verify` say the result is right, which is what a durable engine promises.

Why not a longer visibility timeout? It would cut redeliveries in this drill but make every genuine worker death take longer to recover. Heartbeats give both: fast detection of dead workers and no redelivery for live ones.

## What can go wrong

Two of these pitfalls hide the incident, one destroys data, and two let a wrong result through:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| Alerting on restarts instead of redeliveries, or no durable rule loaded | no page, or a page for restarts that hurt nobody | `DurableRedeliveriesSpike` pages on the effect, from a rules file with `release: observability` |
| Deleting the WAL claim or scaling the StatefulSet to 0 to "reset" the server | every workflow's history is gone; the build never completes | the runbook forbids it; mitigation only confirms the restarts |
| An activity that writes outputs in place, without a temporary name and a rename | a redelivered attempt sees a half-written shard and keeps it; the token counts differ | outputs go through `<name>.tmp` and a rename, `DONE.json` last; verified by the clean-run counts |
| A rerun that ignores the idempotency key | a shard written twice, two ledger rows | activities key their work directory on `TL_IDEMPOTENCY_KEY`; verified by `ledger verify` |
| Declaring the incident over while the build still runs | queue depth is not 0; the result is not yet checked | verification waits for depth 0 held 60 s and runs the two pass-condition commands |

> **Warning:** The WAL is the only copy of every workflow's history. The kills cost time; deleting the claim costs everything.

## Where this leads

This drill landed on the StatefulSet, WAL claim, and worker Deployment of [Durable and worker images and charts](../../infrastructure/01-containers-kubernetes/06-durable-and-worker-charts.md), and its diagnosis read the dashboard of [Observability for the control plane](../04-observability/05-control-plane-observability.md). [A poison task lands in the dead-letter queue](03-poison-task.md) uses the second alert of the same rules file, for a fault that does not heal itself, and [The event log disk fills up](11-eventlog-disk-full.md) puts the same log under a full disk instead of a kill.

### Further reading

Each simple piece here has a production equivalent:

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| one durable replica | Temporal's history service on Cassandra or Postgres; Raft-replicated logs | a kill is not an outage | [Raft high availability for the durable log](../../ai-platform-engineering/05-durable-orchestration-and-workers/10-raft-ha.md) (optional), etcd |
| a kill loop in a drill | Jepsen, Chaos Mesh `PodChaos` on a schedule | randomized faults with a checker over the history | [jepsen.io](https://jepsen.io/) |
| a fixed visibility timeout | heartbeat timeouts per activity, sticky queues | a dead worker is noticed in seconds, not a full timeout | Temporal docs |
| a ledger check after the fact | end-to-end checksums and exactly-once sinks | duplicates caught at write time | Kafka transactions, Flink two-phase commit |

- *Site Reliability Engineering* (Google), ch. 22 Addressing Cascading Failures and ch. 26 Data Integrity ([free](https://sre.google/sre-book/table-of-contents/)).
- [Temporal: activity timeouts and retries](https://docs.temporal.io/encyclopedia/detecting-activity-failures) (free).
- Kleppmann, *Designing Data-Intensive Applications*, ch. 8 (process pauses and leases).

## Key terms

- **SIGKILL**: an uncatchable signal that ends a process between two instructions, leaving only what is on disk.
- **Torn record**: a partially written log record from the instant of a crash, never acknowledged and safely truncated on recovery.
- **Lease and visibility timeout**: a worker's time-limited claim on a task; if it is not renewed by a heartbeat within $V$, the task becomes visible again.
- **Redelivery**: handing a task out again after its lease expired.
- **Idempotency key**: the key an activity uses to find its work directory so a rerun produces the same files as the first run.
- **Exposure**: the system properties (one replica, no checkpoints) that turn an ordinary trigger into an incident.

## Summary

- Killing the server of a durable engine costs time, never correctness: every acknowledged event is in the fsynced log, and recovery truncates only the torn record that was never acknowledged.
- Leases turn a dead worker or server into a timeout: in-flight work is redelivered after the visibility timeout, so each kill shows up as a redeliveries spike, which is what pages.
- Redelivery is safe only because activities are idempotent; the proof is the output, the clean-run counts and a ledger with every row once.
- Mitigation for a self-healing fault is to confirm, not to act; deleting the WAL claim is the one action that loses data.
- The postmortem's action items are about exposure: one replica, and long activities without checkpoints.

## Review questions

1. After a SIGKILL, recovery truncates the last record of the log. Why is that not data loss?
   <details><summary>Answer</summary>An append is acknowledged only after its fsync. The record being written when the process died was never fsynced as a whole, so no caller was told it happened. Truncating it restores the log to exactly the set of acknowledged events.</details>
2. Why does each kill show up as redeliveries about 30 s later rather than at once?
   <details><summary>Answer</summary>A task is redelivered when its lease expires, $V = 30$ s after the last heartbeat. Workers could not heartbeat while the server was down, so the leases live at the kill run out about $V$ later, and only then do the tasks become visible again.</details>
3. A redelivered shard-writing activity runs twice. What guarantees the shard is not written twice?
   <details><summary>Answer</summary>The rerun uses the same idempotency key and work directory. Outputs are written to `<name>.tmp` and renamed, `DONE.json` last; a rerun that finds `DONE.json` re-emits its result and exits, and one that finds partial output resumes or rewrites it to the same content. The ledger keys rows by content hash, so a duplicate would fail `ledger verify`.</details>
4. Why is deleting the WAL claim worse than anything the kills did?
   <details><summary>Answer</summary>The kills cost time: every acknowledged event survived on the claim. The WAL is the only copy of every workflow's history; deleting it loses all of it, so the build can never complete and its partial effects can no longer be reconciled.</details>
5. Why does the runbook prefer heartbeats over a longer visibility timeout?
   <details><summary>Answer</summary>A longer timeout would reduce redeliveries in this drill but make every genuine worker death take longer to notice and recover. Heartbeats keep live workers' leases fresh while still letting a dead worker's lease expire quickly.</details>
