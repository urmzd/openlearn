<!-- ol:module ops.03 -->
# A poison task lands in the dead-letter queue

## Overview

| | |
|---|---|
| **Module** | `ops.03` · drill · ops, docs · Pass 8 · 2 to 3 h |
| **You build** | `docs/runbooks/DurableDeadLetters.md` (Symptoms, Diagnosis, Mitigation), `docs/postmortems/<date>-poison-task.md`, and the alert `DurableDeadLetters` in `deploy/observability/rules/durable.yaml` (written with `ops.02`); and you find the cause of a task that fails every attempt, remove it, redrive the task, and see the build complete |
| **Contract** | the drill spec `course/drills/poison-task/drill.toml` (section 4); `tl_durable_dlq_size` and `tl_durable_task_queue_depth` in [`otel/metrics.yaml`](../../../../course/contracts/otel/metrics.yaml); the failpoint hook your worker evaluates (section 2.4); the `ctl` verbs `wf dlq list`, `wf dlq redrive`, `wf describe` fixed by MS-durable |
| **Tests** | graded by `ol drill end`; no course tests to read (section 4 lists every check) |
| **Needs** | `dep.06` (the worker Deployment the fault is shipped to) and `obs.05` (the scrapes and the dead-letter panel); reading: `ops.02` (the rules file and the drill routine), `dur.09` (exit codes and retryable failures), the queue chapter of this pass (retries, the dead-letter queue, redrive) |
| **Used by** | no call site: a drill. MS-durable requires it; MS-ops reruns the drills |
| **Milestone** | MS-durable (drill `ops.03` resolved) |
| **Optional depth** | [AWS: Amazon SQS dead-letter queues](https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/sqs-dead-letter-queues.html) (free); *Site Reliability Engineering*, ch. 14 Managing Incidents ([free](https://sre.google/sre-book/table-of-contents/)); Hohpe and Woolf, *Enterprise Integration Patterns*, "Dead Letter Channel" and "Invalid Message Channel" |

## Key Takeaways

- A **poison task** fails the same way on every attempt. Retries cannot fix it; they only decide how long it takes to stop trying. After `dlq_after_attempts` the queue moves it aside so the rest of the work keeps flowing.
- A dead letter is a **paused workflow**, not a finished one: the build waits for that one shard forever, so the alert pages on the first dead letter.
- Diagnosis asks two questions in order: **what failed** (the dead letter's last failure) and **what changed** (rollout history, the environment). A deterministic failure that appeared without a code change came with a configuration change.
- **Fix, then redrive.** Redriving first buys five more failures and a second page. Redrive only the tasks you fixed.
- The postmortem's best action item usually **prevents the configuration** from reaching production at all (a policy check on the chart), not just a better alert.

## How to work this chapter

```bash
kubectl apply -f deploy/observability/rules/durable.yaml   # DurableDeadLetters (written with ops.02)
ol drill start poison-task             # ships the fault to the workers, prints the page
kubectl -n <system> rollout status deploy/<system>-worker
{ctl} data build --config course/fixtures/MS-corpus/corpus.toml --id drill-poison --durable 127.0.0.1:30733 --json
# ... wait for the page; dashboard, wf dlq list, the runbook; fix; redrive ...
{ctl} wf describe drill-poison --durable 127.0.0.1:30733 --json   # COMPLETED, dead_lettered 0
ol drill end
ol drill reset
```

`{ctl}` is your `[entry].ctl`; its `wf` and `data build` verbs are fixed in MS-durable. The build starts after the poisoned rollout, as a scheduled build would.

---

## 1. Why now

`ops.02` killed the server and the system healed itself: leases expired, tasks were delivered again, every effect happened once. This drill is the other kind of failure: one task that can never succeed, however often it is retried. Your queue (`dur.03`) handles it by moving the task to a dead-letter queue after a bounded number of attempts, and your dashboard (`obs.05`) shows the dead-letter count. That keeps one bad task from spinning forever, but it also means a human has to act: the workflow waiting on the task is stuck until someone understands why it failed, fixes that, and redrives it.

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

A bad rollout ships a test-only setting to production: the worker Deployment's environment gains `TL_FAILPOINTS=dur/activity/poison/shard-00001=panic`. Your worker (`dur.04`) evaluates the failpoint `dur/activity/poison/<activity id>` before it runs each activity task (the `TL_FAILPOINTS` syntax of the course testkit: `crash`, `panic`, `error(msg)`, `sleep(d)`), and recovers a panic in an activity as a retryable failure of type `Panic`. CorpusBuild (`data.09`) names its per-shard tasks `shard-00000`, `shard-00001`, and so on. So exactly one task panics, on every attempt, on every worker. That is how real poison looks from the outside: one input, one code path, always the same failure.

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

## 4. Inject, detect, mitigate, verify

| Step | What `ol drill` does | What you do |
|---|---|---|
| inject | adds `TL_FAILPOINTS=dur/activity/poison/shard-00001=panic` to the environment of `{deploy.services.worker}` (a JSON patch; the undo restores the old environment) | wait for the rollout, start the build `drill-poison` |
| detect | at `end`, reads the `ALERTS` series: `DurableDeadLetters` must have fired within 600 s of the injection | read the page, open the dead-letter panel and your runbook |
| mitigate | | `wf dlq list data` (what failed and how), `rollout history` and the environment (what changed), remove the setting, wait for the rollout, `wf dlq redrive data <task id>` |
| verify | at `end`: the worker rollout is complete; dead letters at 0 and every queue at depth 0, each held 60 s | `wf describe drill-poison` shows `COMPLETED` and `dead_lettered: 0`; write the runbook and the postmortem |

### What `ol drill end` checks

| Check | Passes when |
|---|---|
| detected | `DurableDeadLetters` fired within `within_s = 600` of the injection |
| resolved | `rollout status` of the worker Deployment; `tl_durable_dlq_size` and `tl_durable_task_queue_depth` summed to 0, each held 60 s |
| runbook | `docs/runbooks/DurableDeadLetters.md` has Symptoms, Diagnosis, Mitigation |
| postmortem | `docs/postmortems/<date>-poison-task.md` has the seven sections |
| time limit | 45 minutes from start to end (not graded for the CI responder) |

The design's last condition, that the workflow completes, is a command check `ol drill end` runs and grades (`[[resolve.run]]` in the spec): `{ctl} wf describe drill-poison --json` must report `COMPLETED` with `dead_lettered: 0`. Run it yourself first and quote it in your Resolution.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. Redriving before fixing | five more panics, a second dead letter, the alert fires again | the dead-letter check fails its hold |
| 2. Deleting the dead letter to clear the alert | the alert clears, the build waits forever | the queue-depth check; the workflow never completes |
| 3. Restarting the workers without removing the setting | the new pods read the same environment and panic the same way | the dead-letter check after the redrive |
| 4. No alert on dead letters, or on the raw count without a `for` | no page, or a page on a scrape blip | ol drill end, check detected |
| 5. A postmortem that stops at "removed the variable" | the next test setting reaches production the same way | ol drill end, check postmortem; the self review against course/rubrics/postmortem.md |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `dep.06` | the worker Deployment the fault is shipped to and rolled back on |
| Back | `obs.05` | the dead-letter panel and the scrapes the alert reads |
| Forward | `ops.10` | a runaway agent stopped by a budget and resumed after a grant: the same pause, fix, resume shape |
| Forward | `ops.08` | a data incident traced through the ledger, the other kind of bad input |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| one dead-letter queue per queue | SQS redrive policies, RabbitMQ dead-letter exchanges | per-queue budgets, automatic redrive to the source | AWS SQS docs |
| redrive by hand | Temporal's failed-activity handling in the workflow; DLQ consumers | the workflow decides: skip, compensate, or fail | `dur.08` (sagas) |
| a failpoint in the environment | feature flags with audit trails, admission policies | test settings cannot reach production | OPA Gatekeeper, Kyverno |
| an alert on the dead-letter count | alerts on failure rate per activity type | the page comes before the budget is spent | obs.05's dashboard |
