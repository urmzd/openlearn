# Runbook: DurableDeadLetters

Owner: the durable control plane (`<system>-durable`, `<system>-worker`).
Linked from the alert's `runbook` annotation
(`deploy/observability/rules/durable.yaml`). Written after drill `ops.03`.
The alert means: at least one task on a queue failed on every one of its
attempts (`[durable].dlq_after_attempts`, 5 by default) and was moved to the
dead-letter queue. Nothing retries it any more, so the workflow waiting on
it is stuck until someone redrives it. Redriving before fixing the cause
only buys five more failures.

## Symptoms

- The page `DurableDeadLetters` (`severity: page`) naming a queue.
- The control-plane dashboard (obs.05): the dead-letter panel is red; the
  queue's depth may be flat (everything else finished) while a workflow
  stays `RUNNING`.
- Users: a build or training run never finishes; `{ctl} wf describe <id>
  --json` shows `dead_lettered: 1` or more.

## Diagnosis

1. What is dead-lettered, and how did its last attempt fail?

   ```bash
   {ctl} wf dlq list data --durable 127.0.0.1:30733 --json
   {ctl} wf describe <workflow id> --durable 127.0.0.1:30733 --json
   ```

   Read `last_failure`: its `type` (`Panic`, `ExitCode65`, `ExitCode1`,
   `Timeout`) and `message`. One task failing the same way five times is a
   **poison task**: something about this input or this code path, not a
   flaky dependency. Many different tasks failing is an outage, not poison.

2. What changed? A poison task that appears without a code or data change
   usually came with a config change. Compare the worker's current spec
   with the previous revision:

   ```bash
   kubectl -n <system> rollout history deploy/<system>-worker
   kubectl -n <system> get deploy/<system>-worker -o jsonpath='{.spec.template.spec.containers[0].env}'
   ```

   A debug or fault setting (for example `TL_FAILPOINTS`) in a production
   environment is a root cause by itself.

3. Does it fail for every input? The other tasks of the same activity type
   succeeded: the failure is specific to this task's input (one shard), so
   reproduce it with that input alone before you redrive.

## Mitigation

- Fix the cause first: roll back the bad configuration
  (`kubectl -n <system> rollout undo deploy/<system>-worker`, or remove the
  setting with `kubectl set env ... NAME-`), or ship the code fix, or
  correct the input. Wait for `kubectl rollout status`.
- Then redrive, and only the tasks you fixed:

  ```bash
  {ctl} wf dlq redrive data <task id> --durable 127.0.0.1:30733 --json
  ```

- Verify: the dead-letter queue is empty, the queue drains, and the
  workflow completes (`wf describe` shows `COMPLETED`, `dead_lettered: 0`).
- If the input itself is bad and cannot be fixed, the workflow needs a
  decision (skip the shard and record it in the ledger, or fail the build):
  never delete a dead letter without recording why.
