# Design review: control plane

## Overview

| | |
|---|---|
| **Module** | `review.02` · proof · docs · Pass 11 · 1 to 2 h |
| **Solution** | a model review of the control plane, `docs/reviews/control-plane.md`, written out in section 5: the design-review rubric answered line by line with evidence from the solution code, each "no" a note with an owner and a next step, and a decision |
| **Needs** | `review.01` (the rubric and how a review is written), `dur.01` to `dur.12` (the durable engine and its workflows), `ag.05` (durable agent runs) |
| **Used by** | no code call site; `review.03` reviews the whole system with the same rubric |

## Key Takeaways

- Tie each claim to the exact evidence that supports it: a contract file, a line of the solution, a postmortem's numbers.
- In a control plane the review question is always the same: what is persisted before each external effect, and what does replay do after a crash at any point.
- State the owner, success condition, and recovery action before declaring the work complete.
- Record unresolved assumptions so the next review can test them.

## How to read this chapter

Sections 1 to 3 say what a control-plane review looks for and trace two paths through the system by hand. Section 4 states the task and the rubric. Section 5 is the model review itself, with the reasoning behind each answer. Try answering each rubric line from the traces in section 3 before reading the model answer.

---

## 1. Why now

The control plane contains durable workflows, workers, queues, an agent runtime, and a usage gateway. Their failure semantics determine whether retries preserve user intent or repeat side effects.

## 2. Principles

Review contracts, data ownership, idempotency, tenant boundaries, load shedding, observability, and recovery as one system. Distinguish durable state from process memory and identify who owns every transition.

## 3. Worked example

Trace one CorpusBuild activity from workflow history to the Python subprocess and artifact files. Then trace an AgentRun tool call that needs approval. Identify what is persisted before each external effect and how replay behaves after a crash.

**CorpusBuild, stage `shard`.** Each step below names what is durable when it happens.

1. The workflow ([`corpus_build.go`](../../../../course/solutions/go/workflows/corpus_build.go)) registers its compensation (`corpus.cleanup`) and schedules activity `corpus.stage` with activity id `shard`. The command is appended to the workflow's history in the segmented log, which fsyncs before acknowledging ([`log.go`](../../../../course/solutions/go/durable/log/log.go), [`formats/wal.md`](../../../../course/contracts/formats/wal.md)).
2. The task is enqueued; every queue transition is logged before it takes effect ([`queue.go`](../../../../course/solutions/go/durable/queue/queue.go)). A worker long-polls and receives a **lease**: a visibility timeout and a fencing token carried inside the opaque task token ([`tasks.go`](../../../../course/solutions/go/durable/server/tasks.go)).
3. The worker runs `{corpus} run --stage shard` as a child process ([`subprocess.go`](../../../../course/solutions/go/activities/subprocess.go), [`spec/subprocess-activity.md`](../../../../course/contracts/spec/subprocess-activity.md)) with `TL_IDEMPOTENCY_KEY=<workflow id>/shard`, the same on every attempt. It tails the child's progress file and turns checkpoints into heartbeats (heartbeat timeout 2 minutes, start-to-close 2 hours, 5 attempts).
4. The child writes shards under `corpus/<dataset>/<version>/` and exits: 0 (done), 65 (bad input, never retried), 75 (temporary, retried), 130 (cancelled after SIGTERM). The exit code becomes the activity's result or failure, and the completion is logged.

**Replay after a crash.** If the worker dies at step 3 or 4, the lease expires and the task is delivered again; the rerun sees the same idempotency key and its work directory and resumes rather than duplicating. If the server is SIGKILLed, it truncates the torn last record on restart, rebuilds queues from the log, and redelivers the leases that were live. A late completion from the old worker carries an old fencing token and is refused. The effect is at least once; the idempotency key is what makes it look exactly once. The `ops.02` postmortem measured it: five kills, 14 tasks delivered twice and 3 three times, no acknowledged event lost, and the shards hold every row once ([`2026-10-09-durable-kill9.md`](../../../../course/solutions/docs/postmortems/2026-10-09-durable-kill9.md)).

**AgentRun, a write tool that needs approval** ([`run.go`](../../../../course/solutions/go/agent/durableagent/run.go), [`journal.go`](../../../../course/solutions/go/agent/durableagent/journal.go)).

1. `Run` appends `run.started` with the input messages. Each model call and tool call is a step: `step.started` is recorded before it runs and `step.completed` with its result before the result is used; every append is fsynced.
2. The gate classifies the tool as a write that needs approval; the loop stops with an approval error, and `Run` returns `awaiting_approval` with the pending markers. Nothing external has happened.
3. A human sends the `approve` signal with the marker; it is appended as a `signal` record.
4. The next `Run` replays every completed step from the journal (no model call, no tool call again), sees the approval, and runs the write.

**Replay after a crash.** A crash between `step.started` and `step.completed` of a *read* re-runs the read. For a *write* the outcome is unknown, so the run stops as `indeterminate` until a human sends `reconcile`: the journal never guesses whether a write happened. Running the same run id with different input is refused (`ErrInputMismatch`).

## 4. Claim and rubric

Write `docs/reviews/control-plane.md` as the design review of the control plane. Answer each line of the design-review rubric (the eight lines of [`review.01`](01-design-review-tracer-and-serving-platform.md) section 2.1) with evidence from the completed system. Each "no" becomes a review note with an owner and a next step, and the review ends with a decision.

## 5. The solution, step by step

### 5.1 From first principles

1. **Durability is decided per transition.** For each arrow in the traces of section 3, ask: is the fact on disk before anything outside the process acts on it? The log's fsync-before-ack, the queue's log-before-effect, and the journal's started-before-run are the three answers.
2. **At-least-once delivery is the given; idempotency is the design.** Leases expire and tasks are redelivered, so every activity's effect must be keyed: `<workflow id>/<activity id>` for activities, the run id plus step for agent runs.
3. **Late actors must be fenced.** A worker that was presumed dead can come back; the fencing token in the task token makes its completion harmless.
4. **Unknown outcomes are escalated, not guessed.** An agent write whose result was not recorded becomes `indeterminate`; a poison activity becomes a dead letter. Both page a human with a runbook.
5. **A review that finds nothing is a weak review.** The rubric lines that lack evidence become notes with owners.

### 5.2 The model review

> **Review: the control plane** (`docs/reviews/control-plane.md`). Scope: the durable server (log, queue, workflows, activities, timers, signals), the workers and subprocess activities, the platform workflows (CorpusBuild, TrainRun, EvalSuite, ModelRelease), durable agent runs, and the gateway's usage ledger and policy as they feed the control plane.

| Rubric line | Answer | Evidence or note |
|---|---|---|
| problem, users, constraints first | yes | workflows that outlive any process: a corpus build or a training run takes minutes to hours and must survive worker and server crashes. Users: the operator (`{ctl} wf describe`, `wf dlq list`), the release approver, agent runs. Constraint: a single kind node, the durable server as one StatefulSet pod on a WAL volume claim |
| two options, same criteria, numbers | yes | single-node log versus the optional three-replica Raft log ([`raft.go`](../../../../course/solutions/go/durable/raft/raft.go)): the single node loses availability (not data) while its pod restarts; Raft tolerates one of three replicas down (a majority of 2) at the cost of two more pods and a network round trip before each commit. Single node chosen for one kind node; Raft replicates exactly the WAL record bytes, so switching is a deployment change |
| what the choice gives up | yes | availability during a restart: the `ops.02` drill measured a build taking 6 min 10 s instead of 2 min 30 s under five kills, with nothing lost |
| every interface names its contract | yes | [`proto/tl/durable/v1`](../../../../course/contracts/proto/tl/durable/), [`formats/wal.md`](../../../../course/contracts/formats/wal.md), [`spec/subprocess-activity.md`](../../../../course/contracts/spec/subprocess-activity.md), [`proto/tl/raft`](../../../../course/contracts/proto/tl/raft/), [`formats/corpus-config.schema.json`](../../../../course/contracts/formats/corpus-config.schema.json), [`formats/usage.v1.sql`](../../../../course/contracts/formats/usage.v1.sql), [`formats/policy.v1.schema.json`](../../../../course/contracts/formats/policy.v1.schema.json) |
| failure modes, detection, degradation | yes | table below |
| capacity and latency budgets, arithmetic | **no** | note N1 |
| rollout and rollback, migrations | **no** | note N2 |
| open questions with an owner | yes | notes N1 to N3 |

**Failure modes.**

| Failure | Detection | Degradation and recovery |
|---|---|---|
| durable server SIGKILLed | `DurableRedeliveriesSpike` ([runbook](../../../../course/solutions/docs/runbooks/DurableRedeliveriesSpike.md)) | torn tail truncated on restart; live leases redelivered after their visibility timeout; reruns keyed by idempotency key |
| a worker dies mid-activity | heartbeat timeout (2 min for corpus stages) | the lease expires and the task is redelivered; a late completion is refused by its fencing token |
| a poison task fails every attempt | `DurableDeadLetters` after 5 attempts ([runbook](../../../../course/solutions/docs/runbooks/DurableDeadLetters.md)) | the waiting workflow is stuck until the cause is fixed and the task redriven ([`ops.03` postmortem](../../../../course/solutions/docs/postmortems/2026-10-09-poison-task.md)) |
| a cancelled or failed corpus build | the workflow's result | the saga runs `corpus.cleanup`: no partial shards or token files remain; fetched raw documents stay as a shared cache |
| a subprocess ignores SIGTERM | exit code after the 30 s grace | SIGKILL; the attempt fails and is retried |
| the WAL fills its `wal_max_bytes` quota | appends fail with `RESOURCE_EXHAUSTED` | new work is refused rather than the disk filled; existing history stays readable |
| an agent write with an unknown outcome | the run's status `indeterminate` | stops until a human `reconcile` signal; never re-executed blindly |

**Notes.**

- **N1, capacity (owner: durable control plane, before Pass 12).** The design states no budget for WAL growth or redelivery cost. Next step: from the `ops.02` numbers (17 extra deliveries over five kills) and the segment size (64 MiB per segment), write the WAL bytes per CorpusBuild and per TrainRun, and set `wal_max_bytes` from that with headroom, so the quota is a known number of builds rather than a guess.
- **N2, rollout (owner: durable control plane).** Workflow code is replayed against recorded history, and a change to a workflow's sequence of commands makes replay of a run started under the old code fail with a nondeterminism error ([`replay.go`](../../../../course/solutions/go/durable/workflow/replay.go)). The mechanism to avoid it exists, `workflow.GetVersion` ([`workflow.go`](../../../../course/solutions/go/durable/workflow/workflow.go)), which records a version marker for new runs and returns the old version when replaying old ones, but no document states the rule for using it. Next step: write that every change to a workflow's command sequence goes behind `GetVersion`, and say how a worker rollout treats runs in flight.
- **N3, assumption to test (owner: agent runtime).** An approval signal names a marker, but the review found no stated expiry for approvals. Next step: decide whether an approval is valid forever or only for the run attempt that asked, and test it in the next drill.

**Decision: approve with changes.** N1 and N2 must be closed (a WAL budget and a workflow-versioning rule) before the control plane is called production ready; N3 is an accepted risk until the agent runtime's next review.

### 5.3 Why this review

- **Evidence over assertion.** Every "yes" names a file or a measured number; a "yes" without one is a "no" waiting to happen.
- **Two traces cover most of the rubric.** Following CorpusBuild and an approved agent write exercises the log, the queue, leases, idempotency, sagas, signals, and the journal; the failure table falls out of asking "crash here?" at every step.
- **Honest "no"s.** Capacity and rollout are where control planes usually fail in production (unbounded logs, replay-breaking deploys), and they are the lines the solution's documents do not yet answer. Writing them as owned notes is the useful output of the review.

### 5.4 Explain it back

1. Why does an activity need an idempotency key if the durable server already records its completion?
   <details><summary>Answer</summary>
   The completion is recorded only after the effect. A worker can finish the effect and die before reporting it, or report it with an expired lease; the task is then delivered again and the effect runs a second time. The key, stable across attempts, lets the effect detect and skip or resume the duplicate.
   </details>
2. What does the fencing token protect against that the visibility timeout does not?
   <details><summary>Answer</summary>
   The timeout decides when a task is given to someone else; it cannot stop the first worker from coming back. The token grows with every delivery, and only the current lease's token may heartbeat or complete, so a late completion from the earlier lease is refused instead of overwriting the new attempt's result.
   </details>
3. Why does the durable agent stop as `indeterminate` for a write but simply re-run a read?
   <details><summary>Answer</summary>
   A read has no external effect, so running it again is safe. A write whose `step.started` was recorded but whose `step.completed` was not may or may not have happened; re-running it could double the effect, and skipping it could lose it. Only a human (or a system that can check the outside world) can decide, so the run waits for `reconcile`.
   </details>
4. Why is "a cancelled build leaves no partial shards" achieved with a saga and not a transaction?
   <details><summary>Answer</summary>
   The effects are files written by separate processes over minutes; there is no transaction spanning them. A saga records a compensating action (`corpus.cleanup`) before the first step whose outputs it must undo, and runs it on cancel or failure.
   </details>

## 6. Pitfalls

| # | Pitfall | Symptom | How the solution avoids it |
|---|---|---|---|
| 1 | Treating version selection or a document claim as implicit | A reader or client cannot determine which contract or evidence applies. | every rubric answer names its contract file or measured number |
| 2 | Skipping the failure path | The successful example works, but malformed input or rollback breaks the system. | each trace in section 3 is followed by "replay after a crash", and the failure table lists detection and recovery per row |
| 3 | Omitting a named owner and recovery condition | The artifact cannot guide an operator when conditions change. | every "no" is a note with an owner and a next step; the decision names what closes it |

## 7. Where it's used next

| Direction | Module | Connection |
|---|---|---|
| Back | `review.01` | the rubric and the shape of a review |
| Back | `dur.01` | the log whose fsync-before-ack every durability claim rests on |
| Back | `ag.05` | the durable agent journal traced in section 3 |
| Forward | `review.03` | the whole system, reviewed with the same rubric |
