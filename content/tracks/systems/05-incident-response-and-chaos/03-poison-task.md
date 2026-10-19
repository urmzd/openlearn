# A poison task lands in the dead-letter queue

## Overview

| | |
|---|---|
| **Module** | `ops.03` · drill · ops, docs · Pass 8 · 2 to 3 h |
| **Solution** | a worked incident (section 5): finding the cause of a task that fails every attempt, removing it, redriving the task, and seeing the build complete; the model runbook [`docs/runbooks/DurableDeadLetters.md`](../../../../course/solutions/docs/runbooks/DurableDeadLetters.md), the model postmortem [`docs/postmortems/2026-10-09-poison-task.md`](../../../../course/solutions/docs/postmortems/2026-10-09-poison-task.md), and the alert `DurableDeadLetters` in [`deploy/observability/rules/durable.yaml`](../../../../course/solutions/entry/deploy/observability/rules/durable.yaml) (shared with `ops.02`) |
| **Contract** | the fault, detection, and resolve conditions in section 4; `tl_durable_dlq_size` and `tl_durable_task_queue_depth` in [`otel/metrics.yaml`](../../../../course/contracts/otel/metrics.yaml); the failpoint hook the worker evaluates (section 2.4); the `ctl` verbs `wf dlq list`, `wf dlq redrive`, `wf describe` of the durable system |
| **Needs** | `dep.06` (the worker Deployment the fault is shipped to) and `obs.05` (the scrapes and the dead-letter panel); reading: `ops.02` (the rules file and the drill routine), `dur.09` (exit codes and retryable failures), the queue chapter of this pass (retries, the dead-letter queue, redrive) |
| **Used by** | no call site: a drill. The durable system is expected to recover from it, and the operations pass reruns the drills |
| **Optional depth** | [AWS: Amazon SQS dead-letter queues](https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/sqs-dead-letter-queues.html) (free); *Site Reliability Engineering*, ch. 14 Managing Incidents ([free](https://sre.google/sre-book/table-of-contents/)); Hohpe and Woolf, *Enterprise Integration Patterns*, "Dead Letter Channel" and "Invalid Message Channel" |

## Key Takeaways

- A **poison task** fails the same way on every attempt. Retries cannot fix it; they only decide how long it takes to stop trying. After `dlq_after_attempts` the queue moves it aside so the rest of the work keeps flowing.
- A dead letter is a **paused workflow**, not a finished one: the build waits for that one shard forever, so the alert pages on the first dead letter.
- Diagnosis asks two questions in order: **what failed** (the dead letter's last failure) and **what changed** (rollout history, the environment). A deterministic failure that appeared without a code change came with a configuration change.
- **Fix, then redrive.** Redriving first buys five more failures and a second page. Redrive only the tasks that were fixed.
- The postmortem's best action item usually **prevents the configuration** from reaching production at all (a policy check on the chart), not just a better alert.

## How to read this chapter

This chapter is a worked incident. Sections 1 to 3 build the reasoning (retry budgets, retryable failures, why fix comes before redrive) and work the retry schedule by hand. Section 4 states what is injected, how it is detected, and what "resolved" means. Section 5 walks through the response with the model runbook and postmortem, and explains why each step is the right one. Before reading each diagnosis step, predict what its output will show.

The drill uses the `DurableDeadLetters` alert of `ops.02`'s rules file (`kubectl apply -f deploy/observability/rules/durable.yaml`). `{ctl}` is the system's control CLI (`spec/cli-roles.md`); its `wf` and `data build` verbs belong to the durable system. The build starts after the poisoned rollout, as a scheduled build would.

---

## 1. Why now

`ops.02` killed the server and the system healed itself: leases expired, tasks were delivered again, every effect happened once. This drill is the other kind of failure: one task that can never succeed, however often it is retried. The queue (`dur.03`) handles it by moving the task to a dead-letter queue after a bounded number of attempts, and the dashboard (`obs.05`) shows the dead-letter count. That keeps one bad task from spinning forever, but it also means a human has to act: the workflow waiting on the task is stuck until someone understands why it failed, fixes that, and redrives it.

## 2. Principles

### 2.1 Retry budgets

| Symbol | Meaning | Type |
|---|---|---|
| $A$ | attempts before the dead-letter queue (`[durable].dlq_after_attempts`) | 5 |
| $b_n$ | backoff before attempt $n + 1$: $\min(b_0 \beta^{n-1}, b_{max})$, then full jitter (`RetryPolicy`, `dur.05`) | seconds |
| $b_0$, $\beta$, $b_{max}$ | initial interval, multiplier, cap | 1 s, 2, 60 s |
| $T_{dlq}$ | time from the first attempt to the dead letter | seconds |

Without jitter the waits are $b_1 = 1$, $b_2 = 2$, $b_3 = 4$, $b_4 = 8$ s, so $T_{dlq}$ is at most $15$ s plus five attempts' run time. Full jitter draws each wait uniformly in $[0, b_n]$, so on average it is half that. The point of the budget: a transient failure (a flaky download) succeeds within a few attempts; a deterministic one is set aside in seconds instead of blocking a worker slot forever.

### 2.2 Retryable or not

The activity's failure type decides what the queue does (`spec/subprocess-activity.md`, `dur.09`): exit 65 (bad input) is non-retryable and fails at once; a crash, a panic, or exit 75 is retryable. A panic is classified retryable because most panics are not deterministic, which is exactly why a deterministic one walks the whole retry budget before anyone notices. Only the dead-letter queue stops it.

### 2.3 Fix, then redrive

Redrive (`RedriveDeadLetter`) puts a dead-lettered task back in its queue with a fresh attempt budget. It is safe because the task is idempotent (the same key, the same work directory), and it is useless until the cause is gone. The order is: find the cause, remove it, confirm the removal is live (a completed rollout), then redrive the specific task ids.

### 2.4 The fault in this drill

A bad rollout ships a test-only setting to production: the worker Deployment's environment gains `TL_FAILPOINTS=dur/activity/poison/shard-00001=panic`. The worker (`dur.04`) evaluates the failpoint `dur/activity/poison/<activity id>` before it runs each activity task (the `TL_FAILPOINTS` actions are `crash`, `panic`, `error(msg)`, `sleep(d)`), and recovers a panic in an activity as a retryable failure of type `Panic`. CorpusBuild (`data.09`) names its per-shard tasks `shard-00000`, `shard-00001`, and so on. So exactly one task panics, on every attempt, on every worker. That is how real poison looks from the outside: one input, one code path, always the same failure.

## 3. Worked example by hand

Backoff without jitter (the worst case), each attempt failing in about 0.1 s, the scrape every 15 s, and the rule `tl_durable_dlq_size > 0` for 15 s.

| Time after the build reaches shard 1 | Event | Dead letters |
|---|---|---|
| 0 s | attempt 1 panics | 0 |
| 1 s | attempt 2 (after $b_1 = 1$ s) panics | 0 |
| 3 s | attempt 3 (after $b_2 = 2$ s) panics | 0 |
| 7 s | attempt 4 (after $b_3 = 4$ s) panics | 0 |
| 15 s | attempt 5 (after $b_4 = 8$ s) panics: $A = 5$ reached, the task is dead-lettered | 1 |
| at most 30 s | the next scrape sees `tl_durable_dlq_size = 1` | 1 |
| at most 45 s | the rule has held for 15 s: **`DurableDeadLetters` fires** | 1 |

So the page arrives at most about 45 s after the first failure, well inside the drill's 600 s. Meanwhile shards 0 and 2 completed, and the build reports `RUNNING` with `dead_lettered: 1`. After the fix and the redrive, the one task runs once more and succeeds; the build completes with the same numbers as a clean run (169 documents, 3 shards).

## 4. The incident: what is injected, detected, and resolved

| Step | What happens |
|---|---|
| inject | a JSON patch adds `TL_FAILPOINTS=dur/activity/poison/shard-00001=panic` to the environment of `deploy/<system>-worker`; a rollout follows |
| workload | after the rollout completes, `{ctl} data build --config <small reference corpus config> --id drill-poison --durable 127.0.0.1:30733 --json` starts |
| detect | `DurableDeadLetters` must fire within 600 s of the injection (5 attempts with backoff 1, 2, 4, 8 s, plus the rollout and the scrape) |
| page | *DurableDeadLetters: queue data has dead-lettered tasks; the corpus build drill-poison has stopped making progress* |

**Resolved means** all of these hold:

| Condition | Holds for |
|---|---|
| `rollout status` of the worker Deployment complete | within 120 s |
| `(sum(tl_durable_dlq_size{namespace="<system>"}) or vector(0)) == 0` | 60 s |
| `(sum(tl_durable_task_queue_depth{namespace="<system>"}) or vector(0)) == 0` | 60 s |
| `{ctl} wf describe drill-poison --durable 127.0.0.1:30733 --json` reports `status: COMPLETED` and `dead_lettered: 0` | at the end |
| `docs/runbooks/DurableDeadLetters.md` (Symptoms, Diagnosis, Mitigation) and `docs/postmortems/<date>-poison-task.md` (the seven sections) exist | at the end |
| all within 45 minutes | |

The last command is the real pass condition: an empty dead-letter queue could also mean someone deleted the task, but only a completed workflow means the work was done.

## 5. The solution, step by step

### 5.1 From first principles

1. **A dead letter is a paused workflow.** The page fires on the first one because the build will wait for that shard forever (Key Takeaways); nothing else will retry it.
2. **First question: what failed?** The dead letter records its last failure. One task failing the same way on all five attempts is poison: something about this input or code path. Many different tasks failing would be an outage, a different runbook.
3. **Second question: what changed?** A deterministic failure that appears with no code or data change came with a configuration change. The rollout history and the worker's environment answer it, and here they show `TL_FAILPOINTS`, a test-only setting.
4. **Is it specific to this input?** The other shard tasks of the same activity succeeded, so the failure is tied to `shard-00001`, which matches the failpoint name.
5. **Fix, confirm, redrive** (section 2.3). Remove the setting; wait for the rollout so no worker still has it; only then redrive, and only the task that was fixed. Redriving first spends another five attempts and dead-letters it again.
6. **Verify by completion.** The workflow `COMPLETED` with `dead_lettered: 0` is the only proof the work was done; the shard is identical to a clean run's because the activity is idempotent.
7. **Root cause is the path, not the variable.** A test setting reached production because nothing checked for it. The best action item blocks that path.

### 5.2 Diagnosis, annotated

```bash
# 1. What is dead-lettered, and how did its last attempt fail? Read
#    last_failure: type (Panic, ExitCode65, ExitCode1, Timeout) and message.
#    Here: shard-00001, 5 attempts, Panic: failpoint dur/activity/poison/shard-00001.
{ctl} wf dlq list data --durable 127.0.0.1:30733 --json
{ctl} wf describe drill-poison --durable 127.0.0.1:30733 --json

# 2. What changed? Compare the worker's spec with the previous revision. A
#    debug or fault setting in a production environment is a root cause by itself.
kubectl -n <system> rollout history deploy/<system>-worker
kubectl -n <system> get deploy/<system>-worker -o jsonpath='{.spec.template.spec.containers[0].env}'
```

3. Does it fail for every input? No: the other shards completed, so the failure is specific to this task's input, and in general it should be reproduced with that input alone before redriving.

### 5.3 Mitigation and verification

```bash
# Fix the cause first: remove the setting (or roll back the revision that
# added it) and wait until every worker runs without it.
kubectl -n <system> set env deploy/<system>-worker TL_FAILPOINTS-
kubectl -n <system> rollout status deploy/<system>-worker

# Then redrive exactly the task that was fixed.
{ctl} wf dlq redrive data <task id> --durable 127.0.0.1:30733 --json

# Verify: COMPLETED, dead_lettered 0; the dead-letter queue empty, the queue drained.
{ctl} wf describe drill-poison --durable 127.0.0.1:30733 --json
```

`kubectl set env NAME-` and `rollout undo` both work; `set env` removes exactly the bad setting, while `undo` also reverts anything else in that revision. Restarting the workers without removing the setting does nothing: new pods read the same environment.

### 5.4 The model runbook

The runbook opens with the meaning of the alert and the one rule that matters most:

```markdown
The alert means: at least one task on a queue failed on every one of its
attempts (`[durable].dlq_after_attempts`, 5 by default) and was moved to the
dead-letter queue. Nothing retries it any more, so the workflow waiting on
it is stuck until someone redrives it. Redriving before fixing the cause
only buys five more failures.
```

Diagnosis is the three steps of 5.2; Mitigation is 5.3, plus the case where the input itself is bad and cannot be fixed: "the workflow needs a decision (skip the shard and record it in the ledger, or fail the build): never delete a dead letter without recording why." Full runbook: [`DurableDeadLetters.md`](../../../../course/solutions/docs/runbooks/DurableDeadLetters.md).

### 5.5 The model postmortem

```markdown
## Root cause

A configuration meant for tests (a failpoint that panics one activity)
reached the production worker Deployment. Nothing stopped it: the chart
accepts any environment variable, and the rollout had no check for fault or
debug settings. The retry policy did what it should with a deterministic
failure: five attempts, then the dead-letter queue, so the problem stayed
contained to one task instead of retrying forever.
```

The Detection section notes the gap: five `Panic` failures in 16 s were visible only in `wf describe`, so an alert on activity failure rate would have paged a minute earlier. The action items:

| Action | Type | Owner |
|---|---|---|
| Reject `TL_FAILPOINTS` in the worker chart's values schema outside test profiles (a policy test over `helm template`) | prevent | platform |
| Alert on the activity failure rate per type, not only on dead letters | detect | control plane |
| Add "what changed?" (`rollout history`, environment diff) as step 2 of every control-plane runbook | process | on-call |
| Show `last_failure` in the dead-letter panel so the page carries the cause | detect | observability |

Full postmortem: [`2026-10-09-poison-task.md`](../../../../course/solutions/docs/postmortems/2026-10-09-poison-task.md). The alert itself is in [`rules/durable.yaml`](../../../../course/solutions/entry/deploy/observability/rules/durable.yaml) (section 5.2 of `ops.02`).

### 5.6 Why this solution

- **The dead-letter queue is the containment; the human is the fix.** Bounded retries stop a deterministic failure from holding a worker slot forever, at the price of needing someone to decide. That is the right trade: the system cannot know whether to fix code, config, or input.
- **Redrive one task, not the queue.** Redriving everything would also resubmit tasks dead-lettered for other reasons, still unfixed.
- **Prevention beats detection.** A values-schema rule that rejects `TL_FAILPOINTS` makes this incident impossible; a better alert only makes it shorter.

### 5.7 Explain it back

1. Why does a panic walk the whole retry budget instead of failing at once?
   <details><summary>Answer</summary>A panic is classified retryable, because most panics are not deterministic (a race, a transient state). Only exit 65 (bad input) is non-retryable. So a deterministic panic is retried $A = 5$ times with backoff before the dead-letter queue stops it.</details>
2. What happens if the dead letter is redriven before the setting is removed?
   <details><summary>Answer</summary>The task gets a fresh budget, runs on a worker that still has `TL_FAILPOINTS`, panics five more times, and is dead-lettered again; the alert fires a second time and nothing has changed.</details>
3. Why is deleting the dead letter not a fix even though the alert clears?
   <details><summary>Answer</summary>The workflow is waiting on that task's result. Deleting it removes the signal, not the dependency: the build stays `RUNNING` forever, and the missing shard is not recorded anywhere.</details>
4. Why is the redriven shard identical to a clean run's?
   <details><summary>Answer</summary>The panicking attempts failed before running, so they wrote nothing; the redriven attempt runs with the same idempotency key and work directory and produces the same deterministic output a clean run does.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. Redriving before fixing | five more panics, a second dead letter, the alert fires again | mitigation removes the setting and waits for the rollout before redriving |
| 2. Deleting the dead letter to clear the alert | the alert clears, the build waits forever | verification requires the workflow `COMPLETED`, not only an empty queue |
| 3. Restarting the workers without removing the setting | the new pods read the same environment and panic the same way | the fix is `kubectl set env ... TL_FAILPOINTS-`, a change to the pod template |
| 4. No alert on dead letters, or on the raw count without a `for` | no page, or a page on a scrape blip | `DurableDeadLetters` is `> 0` with `for: 15s` |
| 5. A postmortem that stops at "removed the variable" | the next test setting reaches production the same way | the model action items add a chart policy that rejects the setting |

## 7. Where it's used next
## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `dep.06` | the worker Deployment the fault is shipped to and rolled back on |
| Back | `obs.05` | the dead-letter panel and the scrapes the alert reads |
| Forward | `ops.10` | a runaway agent stopped by a budget and resumed after a grant: the same pause, fix, resume shape |
| Forward | `ops.08` | a data incident traced through the ledger, the other kind of bad input |

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| one dead-letter queue per queue | SQS redrive policies, RabbitMQ dead-letter exchanges | per-queue budgets, automatic redrive to the source | AWS SQS docs |
| redrive by hand | Temporal's failed-activity handling in the workflow; DLQ consumers | the workflow decides: skip, compensate, or fail | `dur.08` (sagas) |
| a failpoint in the environment | feature flags with audit trails, admission policies | test settings cannot reach production | OPA Gatekeeper, Kyverno |
| an alert on the dead-letter count | alerts on failure rate per activity type | the page comes before the budget is spent | obs.05's dashboard |
