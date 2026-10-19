# Runbook: DurableRedeliveriesSpike

Owner: the durable control plane (`<system>-durable`, `<system>-worker`).
Linked from the alert's `runbook` annotation
(`deploy/observability/rules/durable.yaml`). Written after drill `ops.02`.
The alert means: more than 5 activity tasks on one queue were delivered
again in the last 2 minutes, held for 20 s. A redelivery happens when a
task's lease expired before the worker completed it: the worker died, the
server died and the lease ran out while it was down, or the activity runs
longer than its visibility timeout without heartbeating.

## Symptoms

- The page `DurableRedeliveriesSpike` (`severity: page`) naming a queue.
- The control-plane dashboard (obs.05): the redeliveries rate rises, queue
  depth stops falling or climbs, schedule-to-start grows; the WAL size keeps
  growing (every redelivery is a record).
- Users: `{ctl} data build` or `{ctl} train` runs take longer than usual; a
  workflow's `wf describe` shows activities at attempt 2 or more.

## Diagnosis

1. Who is restarting: the server, the workers, or neither?

   ```bash
   kubectl -n <system> get pods -o wide
   kubectl -n <system> get events --sort-by=.lastTimestamp | tail -20
   kubectl -n <system> logs statefulset/<system>-durable --previous --tail=50
   ```

   `RESTARTS` growing on `<system>-durable-0`, or `Killing` events, is a
   server that keeps dying. A server log that ends mid-line with no shutdown
   message was killed (SIGKILL, OOM), not stopped.

2. Did the server recover its log each time? The first lines after a start
   report recovery:

   ```bash
   kubectl -n <system> logs statefulset/<system>-durable --tail=200 | grep -E 'recover|torn|truncat'
   ```

   A truncated torn tail is expected after a kill (the record being written
   when it died was never acknowledged). `ErrCorrupt` in an older segment is
   not: stop here and page the owner (data loss is possible).

3. Are workers dying instead? If the server is steady, check the worker
   pods for restarts and OOM kills (`kubectl describe pod`, `Last State:
   Terminated, Reason: OOMKilled`), and the activity's heartbeats in
   `{ctl} wf describe <id> --json` (a heartbeat timeout shows as
   `last_failure.type` `Timeout`).

4. Is work still being done exactly once? A redelivered activity runs again
   with the same idempotency key and finds its work directory: it resumes or
   re-emits `DONE.json` (spec/subprocess-activity.md). Duplicated outputs mean
   an activity ignores its key: a bug, not this incident.

## Mitigation

- A server that is being killed by something (a node problem, a bad
  liveness probe, an OOM): fix the cause; `kubectl rollout status
  statefulset/<system>-durable` until it is Ready and stays Ready.
- Do not delete the WAL claim and do not scale the StatefulSet to 0 "to
  reset it": the log is the only copy of every workflow's history.
- Workers dying: raise their memory limit or fix the crashing activity; an
  activity that runs past its visibility timeout needs heartbeats, not a
  longer timeout.
- Verify: redeliveries fall to zero, queues drain to zero, nothing in the
  dead-letter queue, and the affected workflows complete:

  ```bash
  {ctl} wf describe <workflow id> --durable 127.0.0.1:30733 --json
  {ctl} wf dlq list data --durable 127.0.0.1:30733 --json
  ```
