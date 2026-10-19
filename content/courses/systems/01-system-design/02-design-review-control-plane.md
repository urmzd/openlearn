# Design review: the control plane

The control plane is the part of the system that keeps working after any one process dies. It contains durable workflows, workers, queues, an agent runtime, and the usage gateway that feeds them. Their failure semantics decide whether a retry preserves what the user intended or repeats a side effect, which is why the control plane deserves its own review rather than a paragraph in the whole-system one. In a control plane the review question is always the same: what is persisted before each external effect, and what does replay do after a crash at any point?

This section applies the eight-line rubric from [Design review: the tracer and the serving platform](01-design-review-tracer-and-serving-platform.md) to that question. It assumes you know the durable engine and its workflows from [Durable Orchestration and Workers](../../ai-platform-engineering/05-durable-orchestration-and-workers/README.md) in *AI Platform Engineering*, and the journal from [Durable agent runs](../../ai-platform-engineering/13-agent-sdk/05-durable-agent-runs.md) in the same book. You will trace two paths through the system by hand, a corpus build activity and an agent write that needs approval, and then read a model review, `docs/reviews/control-plane.md`, written out in full below: every rubric line answered with evidence from the solution code, every "no" turned into a note with an owner and a next step, and a decision at the end.

Try answering each rubric line from the two traces before reading the model's answer. The lines you cannot answer from the traces are usually the ones the review ends up marking "no".

> **Learning objectives**
> - Review contracts, data ownership, idempotency, tenant boundaries, load shedding, observability, and recovery as one system.
> - For each transition in a durable workflow, say what is on disk before anything outside the process acts on it.
> - Explain how leases, fencing tokens, and idempotency keys turn at-least-once delivery into an effect that looks exactly once.
> - Explain why an agent write with an unknown outcome is escalated to a human rather than retried or skipped.
> - Tie every rubric answer to evidence and turn every missing answer into an owned note.

## What a control-plane review looks for

A control-plane review looks at contracts, data ownership, idempotency, tenant boundaries, load shedding, observability, and recovery together, because in this part of the system they fail together. Two distinctions organize the work. The first is between durable state and process memory: anything that lives only in a process is gone after a crash, so every claim about recovery must point at something on disk. The second is ownership: for every transition, someone (a component, a person) owns it, and the review identifies who.

From there the principles follow in order. Durability is decided per transition: for each arrow in a trace, ask whether the fact is on disk before anything outside the process acts on it. At-least-once delivery is the given and idempotency is the design: leases expire and tasks are redelivered, so every activity's effect must be keyed, by `<workflow id>/<activity id>` for activities and by the run id plus step for agent runs. Late actors must be fenced: a worker that was presumed dead can come back, and its completion has to be made harmless. Unknown outcomes are escalated, not guessed: an agent write whose result was not recorded becomes `indeterminate`, and a poison activity becomes a dead letter, and both page a human with a runbook. Finally, a review that finds nothing is a weak review; the rubric lines that lack evidence become notes with owners.

## Tracing two paths by hand

The two traces below exercise most of the control plane between them: the log, the queue, leases, idempotency, sagas, signals, and the journal. Each step names what is durable when it happens.

### A CorpusBuild activity, stage `shard`

1. The workflow ([`corpus_build.go`](../../../../course/solutions/go/workflows/corpus_build.go)) registers its compensation (`corpus.cleanup`) and schedules activity `corpus.stage` with activity id `shard`. The command is appended to the workflow's history in the segmented log, which fsyncs before acknowledging ([`log.go`](../../../../course/solutions/go/durable/log/log.go), [`formats/wal.md`](../../../../course/contracts/formats/wal.md)).
2. The task is enqueued; every queue transition is logged before it takes effect ([`queue.go`](../../../../course/solutions/go/durable/queue/queue.go)). A worker long-polls and receives a **lease**: a visibility timeout and a fencing token carried inside the opaque task token ([`tasks.go`](../../../../course/solutions/go/durable/server/tasks.go)).
3. The worker runs `{corpus} run --stage shard` as a child process ([`subprocess.go`](../../../../course/solutions/go/activities/subprocess.go), [`spec/subprocess-activity.md`](../../../../course/contracts/spec/subprocess-activity.md)) with `TL_IDEMPOTENCY_KEY=<workflow id>/shard`, the same on every attempt. It tails the child's progress file and turns checkpoints into heartbeats (heartbeat timeout 2 minutes, start-to-close 2 hours, 5 attempts).
4. The child writes shards under `corpus/<dataset>/<version>/` and exits: 0 (done), 65 (bad input, never retried), 75 (temporary, retried), 130 (cancelled after SIGTERM). The exit code becomes the activity's result or failure, and the completion is logged.

Now ask "crash here?" at each step. If the worker dies at step 3 or 4, the lease expires and the task is delivered again; the rerun sees the same idempotency key and its work directory, and resumes rather than duplicating. If the server is SIGKILLed, it truncates the torn last record on restart, rebuilds queues from the log, and redelivers the leases that were live. A late completion from the old worker carries an old fencing token and is refused. The effect is at least once; the idempotency key is what makes it look exactly once.

This is not a hypothetical. The drill [The durable server is SIGKILLed in a loop during a corpus build](../05-incident-response-and-chaos/03-durable-kill9.md) measured it: five kills, 14 tasks delivered twice and 3 three times, no acknowledged event lost, and the shards hold every row once ([`2026-10-09-durable-kill9.md`](../../../../course/solutions/docs/postmortems/2026-10-09-durable-kill9.md)).

### An AgentRun write tool that needs approval

The second trace follows the durable agent ([`run.go`](../../../../course/solutions/go/agent/durableagent/run.go), [`journal.go`](../../../../course/solutions/go/agent/durableagent/journal.go)).

1. `Run` appends `run.started` with the input messages. Each model call and tool call is a step: `step.started` is recorded before it runs and `step.completed` with its result before the result is used; every append is fsynced.
2. The gate classifies the tool as a write that needs approval; the loop stops with an approval error, and `Run` returns `awaiting_approval` with the pending markers. Nothing external has happened.
3. A human sends the `approve` signal with the marker; it is appended as a `signal` record.
4. The next `Run` replays every completed step from the journal (no model call, no tool call again), sees the approval, and runs the write.

Again ask what a crash does. A crash between `step.started` and `step.completed` of a *read* re-runs the read. For a *write* the outcome is unknown, so the run stops as `indeterminate` until a human sends `reconcile`: the journal never guesses whether a write happened. Running the same run id with different input is refused (`ErrInputMismatch`).

## The task

The task is to write `docs/reviews/control-plane.md` as the design review of the control plane. Answer each of the eight rubric lines (defined in [Design documents and design reviews](01-design-review-tracer-and-serving-platform.md#design-documents-and-design-reviews)) with evidence from the completed system. Each "no" becomes a review note with an owner and a next step, and the review ends with a decision.

## The model review

The review opens with its scope:

> **Review: the control plane** (`docs/reviews/control-plane.md`). Scope: the durable server (log, queue, workflows, activities, timers, signals), the workers and subprocess activities, the platform workflows (CorpusBuild, TrainRun, EvalSuite, ModelRelease), durable agent runs, and the gateway's usage ledger and policy as they feed the control plane.

It then answers the rubric, one row per line, each "yes" backed by a file or a measured number:

| Rubric line | Answer | Evidence or note |
|---|---|---|
| problem, users, constraints first | yes | workflows that outlive any process: a corpus build or a training run takes minutes to hours and must survive worker and server crashes. Users: the operator (`{ctl} wf describe`, `wf dlq list`), the release approver, agent runs. Constraint: a single kind node, the durable server as one StatefulSet pod on a WAL volume claim |
| two options, same criteria, numbers | yes | single-node log versus the optional three-replica Raft log ([`raft.go`](../../../../course/solutions/go/durable/raft/raft.go)): the single node loses availability (not data) while its pod restarts; Raft tolerates one of three replicas down (a majority of 2) at the cost of two more pods and a network round trip before each commit. Single node chosen for one kind node; Raft replicates exactly the WAL record bytes, so switching is a deployment change |
| what the choice gives up | yes | availability during a restart: the SIGKILL drill measured a build taking 6 min 10 s instead of 2 min 30 s under five kills, with nothing lost |
| every interface names its contract | yes | [`proto/tl/durable/v1`](../../../../course/contracts/proto/tl/durable/), [`formats/wal.md`](../../../../course/contracts/formats/wal.md), [`spec/subprocess-activity.md`](../../../../course/contracts/spec/subprocess-activity.md), [`proto/tl/raft`](../../../../course/contracts/proto/tl/raft/), [`formats/corpus-config.schema.json`](../../../../course/contracts/formats/corpus-config.schema.json), [`formats/usage.v1.sql`](../../../../course/contracts/formats/usage.v1.sql), [`formats/policy.v1.schema.json`](../../../../course/contracts/formats/policy.v1.schema.json) |
| failure modes, detection, degradation | yes | table below |
| capacity and latency budgets, arithmetic | **no** | note N1 |
| rollout and rollback, migrations | **no** | note N2 |
| open questions with an owner | yes | notes N1 to N3 |

### Failure modes

The failure table falls out of the traces: each row is a "crash here?" question with its detection and its recovery.

| Failure | Detection | Degradation and recovery |
|---|---|---|
| durable server SIGKILLed | `DurableRedeliveriesSpike` ([runbook](../../../../course/solutions/docs/runbooks/DurableRedeliveriesSpike.md)) | torn tail truncated on restart; live leases redelivered after their visibility timeout; reruns keyed by idempotency key |
| a worker dies mid-activity | heartbeat timeout (2 min for corpus stages) | the lease expires and the task is redelivered; a late completion is refused by its fencing token |
| a poison task fails every attempt | `DurableDeadLetters` after 5 attempts ([runbook](../../../../course/solutions/docs/runbooks/DurableDeadLetters.md)) | the waiting workflow is stuck until the cause is fixed and the task redriven ([poison-task postmortem](../../../../course/solutions/docs/postmortems/2026-10-09-poison-task.md)) |
| a cancelled or failed corpus build | the workflow's result | the saga runs `corpus.cleanup`: no partial shards or token files remain; fetched raw documents stay as a shared cache |
| a subprocess ignores SIGTERM | exit code after the 30 s grace | SIGKILL; the attempt fails and is retried |
| the WAL fills its `wal_max_bytes` quota | appends fail with `RESOURCE_EXHAUSTED` | new work is refused rather than the disk filled; existing history stays readable |
| an agent write with an unknown outcome | the run's status `indeterminate` | stops until a human `reconcile` signal; never re-executed blindly |

### Notes

The two "no" lines, and one assumption the review could not settle, become notes with owners:

- **N1, capacity (owner: durable control plane, before the next stage of the build).** The design states no budget for WAL growth or redelivery cost. Next step: from the SIGKILL drill's numbers (17 extra deliveries over five kills) and the segment size (64 MiB per segment), write the WAL bytes per CorpusBuild and per TrainRun, and set `wal_max_bytes` from that with headroom, so the quota is a known number of builds rather than a guess.
- **N2, rollout (owner: durable control plane).** Workflow code is replayed against recorded history, and a change to a workflow's sequence of commands makes replay of a run started under the old code fail with a nondeterminism error ([`replay.go`](../../../../course/solutions/go/durable/workflow/replay.go)). The mechanism to avoid it exists, `workflow.GetVersion` ([`workflow.go`](../../../../course/solutions/go/durable/workflow/workflow.go)), which records a version marker for new runs and returns the old version when replaying old ones, but no document states the rule for using it. Next step: write that every change to a workflow's command sequence goes behind `GetVersion`, and say how a worker rollout treats runs in flight.
- **N3, assumption to test (owner: agent runtime).** An approval signal names a marker, but the review found no stated expiry for approvals. Next step: decide whether an approval is valid forever or only for the run attempt that asked, and test it in the next drill.

### The decision

**Approve with changes.** N1 and N2 must be closed (a WAL budget and a workflow-versioning rule) before the control plane is called production ready; N3 is an accepted risk until the agent runtime's next review.

### Why this review

Evidence beats assertion: every "yes" names a file or a measured number, and a "yes" without one is a "no" waiting to happen. Two traces cover most of the rubric, because following a CorpusBuild and an approved agent write exercises the log, the queue, leases, idempotency, sagas, signals, and the journal, and the failure table falls out of asking "crash here?" at every step. And the "no"s are honest. Capacity and rollout are where control planes usually fail in production, through unbounded logs and replay-breaking deploys, and they are exactly the lines the solution's documents do not yet answer. Writing them as owned notes is the useful output of the review.

## What can go wrong

The ways a control-plane review fails mirror the principles it is meant to check. If version selection or a document claim is left implicit, a reader or client cannot determine which contract or evidence applies; the model review avoids this because every rubric answer names its contract file or measured number. If the failure path is skipped, the successful example works but malformed input or a rollback breaks the system; here each trace is followed by its replay-after-a-crash analysis, and the failure table lists detection and recovery for every row.

> **Warning:** A review that omits a named owner and recovery condition cannot guide an operator when conditions change. Every "no" in the model is a note with an owner and a next step, and the decision names what closes it.

## Where this leads

This review rests on two earlier pieces: the rubric and the shape of a review from [Design review: the tracer and the serving platform](01-design-review-tracer-and-serving-platform.md), and the log whose fsync-before-ack every durability claim depends on, from [The append-only event log](../../ai-platform-engineering/05-durable-orchestration-and-workers/01-event-log.md). The agent trace comes from [Durable agent runs](../../ai-platform-engineering/13-agent-sdk/05-durable-agent-runs.md). Next, [Design review: the whole system](03-design-review-whole-system.md) applies the same rubric to everything at once.

## Key terms

- **Lease**: a time-limited claim on a task, made of a visibility timeout and a fencing token.
- **Fencing token**: a number that grows with every delivery of a task; only the current lease's token may heartbeat or complete.
- **Idempotency key**: an identifier stable across attempts (`<workflow id>/<activity id>`) that lets an effect detect and skip or resume a duplicate.
- **Saga**: a sequence of steps with recorded compensating actions that undo earlier steps on cancel or failure.
- **Indeterminate**: the status of an agent run whose write was started but whose outcome was not recorded; it waits for a human `reconcile`.

## Summary

- Tie each claim to the exact evidence that supports it: a contract file, a line of the solution, a postmortem's numbers.
- In a control plane the review question is always what is persisted before each external effect, and what replay does after a crash at any point.
- At-least-once delivery is the given; idempotency keys and fencing tokens make the effect look exactly once.
- Unknown outcomes are escalated to a human, never guessed.
- State the owner, success condition, and recovery action before declaring the work complete, and record unresolved assumptions so the next review can test them.

## Review questions

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
