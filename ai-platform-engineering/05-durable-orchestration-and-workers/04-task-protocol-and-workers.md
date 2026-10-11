<!-- ol:module dur.04 -->
# Task gRPC protocol, Go worker SDK, worker pool

## Overview

| | |
|---|---|
| **Module** | `dur.04` · build · Go · Pass 8 · 5 to 7 h |
| **You build** | `go/durable/server/tasks.go`: `PollWorkflowTask`, `CompleteWorkflowTask` (and the command table), `PollActivityTask`, `CompleteActivityTask`; `go/durable/worker/worker.go`: `New`, `RegisterActivity`, `Run`, `Heartbeat`, `InfoFrom`, `ReconnectDelay` |
| **Contract** | `tl.durable.v1.TaskService` in [`proto/tl/durable/v1/durable.proto`](../../course/contracts/proto/tl/durable/v1/durable.proto); the Go SDK surface of DESIGN 2.7; the package API is section 4 |
| **Tests** | `course/tests/go/dur_04/` (what they check: section 4) |
| **Needs** | `dur.02` the server core, `dur.03` the queue; reading: [`obs.01` tracing](../../systems/04-observability/01-tracing-for-the-serving-path.md), [`lang.10` gRPC](../../software-craftsmanship/12-language-and-tool-primers/10-protocol-buffers-and-grpc.md), [RPC and protocols](../02-rpc-and-protocols/) |
| **Used by** | `dur.05` activity failures, retries, and heartbeats and `dur.08` cancellation go through these task and activity handlers |
| **Milestone** | MS-durable |
| **Optional depth** | the Temporal worker docs (pollers, slots, graceful shutdown); Brooker, *Exponential Backoff and Jitter* (AWS Architecture Blog, 2015) |

## Key Takeaways

- A workflow task is a lease on the right to decide the run's next step: polling appends `WorkflowTaskStarted`, completing appends `WorkflowTaskCompleted` and the events of the commands in one batch, and only the current task with a live lease may complete (`TestTaskProtocolHandExample`, `TestCompleteWorkflowTaskFenced`).
- History is the truth and queue tasks are hints: a delivered task whose scheduled event is no longer pending is acknowledged and skipped (`TestStaleQueueTaskSkipped`).
- The worker takes a pool slot **before** it polls, so it never leases work it cannot run (`TestWorkerBoundedConcurrency`).
- Shutdown drains: polling stops at once, running activities finish and report (`TestWorkerDrainsOnCancel`); an activity that misses its heartbeat is canceled and reports nothing (`TestMissedHeartbeatCancelsActivity`).

## How to work this chapter

```bash
ol start dur.04
ol tests dur.04
ol check dur.04          # needs dur.02 and dur.03 (or --ref-deps)
ol diff  dur.04
```

Start with the server half (`tasks.go`) and the worked example, driving it by hand over gRPC as the tests do; then the worker.

---

## 1. Why now

The server (`dur.02`) can start runs and the queue (`dur.03`) can lease tasks, but nothing executes anything. This module is the protocol between the server and the processes that do the work, and the Go SDK of those processes: a worker that long-polls for workflow tasks (decide what to do next) and activity tasks (do it), runs activities in a bounded pool, survives server restarts, and shuts down without losing or duplicating work. After it, `{worker} --queue default` processes a queue; data.09, dur.11, dur.12, and ag.05 register their workflows and activities on it.

## 2. Principles

### 2.1 Workflow tasks

| Symbol | Meaning |
|---|---|
| $S$ | event id of the run's pending `WorkflowTaskScheduled` |
| $T$ | event id of its latest `WorkflowTaskStarted` |
| $k$ | the queue lease token of the delivery |

`PollWorkflowTask` leases task `<run>/wt/S` from queue `wf:<task_queue>` (long poll, 30 s; an empty `task_token` when nothing came). If the run's pending workflow task is no longer $S$ (it already completed, or the run closed), the delivery is stale: acknowledge it and keep polling. Otherwise append `WorkflowTaskStarted{scheduled_event_id: S}` (its id is $T$) and return the history through $T$, at most 1 MiB of it (the first page) with a `next_page_token` for the rest, and `continue_as_new_suggested` when the history has passed 10,000 events or 32 MiB (`dur.06`). The task token carries the queue, task id, $k$, run id, $S$, and $T$.

`CompleteWorkflowTask(token, commands)` checks that the run is open, its pending task is still $(S, T)$, and the lease $(k)$ is live; anything else is `FAILED_PRECONDITION` and appends nothing. Then one batch: `WorkflowTaskCompleted{S, T}` followed by each command's events, committed atomically, then the lease is acknowledged. Each command kind has a handler in a table (`registerCommand`): this module registers `schedule_activity` (appends `ActivityTaskScheduled`, whose effect enqueues the activity task), `complete`, and `fail`; `dur.06`, `dur.07`, and `dur.08` register theirs. A command after the one that ends the run is `INVALID_ARGUMENT`; an unknown kind is `UNIMPLEMENTED`.

A worker that dies after polling never completes; the lease expires (the workflow task timeout, 10 s) and the next poll appends a second `WorkflowTaskStarted` for the same $S$. The old token now names a $T$ that is not the latest and is refused.

**One task at a time.** A trigger (an activity result) that arrives while a workflow task is running is appended to history but schedules nothing; when the running task completes, its batch schedules the next one (`dur.02` section 2.1). The worker that runs the current task never sees events after its $T$.

### 2.2 Activity tasks and fencing

`PollActivityTask` leases `<run>/a/<scheduled id>` from `act:<task_queue>`; a delivery whose activity is no longer pending is acknowledged and skipped. The `ActivityTask` carries the input, the attempt (from the queue), the lease token, the deadline, the heartbeat timeout, the starter's trace context, and the **idempotency key** `"<workflow_id>/<activity_id>"`: the same on every attempt, so the activity's side effect can be made idempotent (`dur.05`). It uses the workflow id, not the run id, because ContinueAsNew changes the run id.

`CompleteActivityTask(token, lease_token, result)` checks the lease with `Queue.Check` **before** appending anything: a stale token (the task was redelivered) is `FAILED_PRECONDITION` and the result is dropped. Then one batch, `ActivityTaskStarted{attempt, lease_token}` and `ActivityTaskCompleted{result}` (and the `WorkflowTaskScheduled` the trigger needs), then the ack. Payloads over 2 MiB (activity input, activity result, workflow result) are `INVALID_ARGUMENT`; large data goes to `/artifacts` and travels by path.

### 2.3 The worker

```
Run(ctx):
    activity poller:  loop { take a slot; poll; run the task in a goroutine that frees the slot }
    workflow poller:  loop { poll; fetch every history page; commands = handler(task, history); complete }
    on ctx done: stop polling now; wait for running activities (up to DrainTimeout), then cancel them
```

**Bounded pool, slot first.** With `MaxActivities` = $n$, the poller acquires a slot before polling. Polling first and then waiting for a slot holds a lease on a task the worker cannot start; it expires and is redelivered, counting as a failed attempt.

**Activity contexts are not Run's context.** SIGTERM ends `Run`'s context; an activity in flight must be allowed to finish and report, so its context derives from a separate "work" context that only the drain timeout cancels.

**Heartbeat watchdog.** An activity with a heartbeat timeout $h$ must call `Heartbeat` at least every $h$; the server gives its task to someone else after $h$. The worker watches the same interval on its own clock and, when it passes, cancels the activity's context with cause `ErrHeartbeatTimeout` and reports **nothing**: the lease is gone, and a report could overwrite the live attempt. A heartbeat refused with `FAILED_PRECONDITION` cancels with `ErrLeaseLost` the same way.

**Reconnect with backoff.** A failed poll (the server restarted) waits $\min(b_0 \cdot 2^{n-1}, b_{\max})$ after the $n$-th consecutive failure (`ReconnectDelay`, 100 ms doubling to 5 s by default) and tries again; reports (complete, fail) are retried while the server answers `UNAVAILABLE`, because the lease survives a server restart.

**Errors to failures.** An activity's error becomes a `Failure{message, type}`; an error with a `NonRetryable() bool` method returning true is non-retryable and one with `FailureType() string` names the type (`dur.05` builds such errors). A panic becomes type `Panic` with its stack.

## 3. Worked example by hand

`Echo("hi")`: one workflow task schedules activity `Upper("hi")`, the next completes the run with its result. Events as the server appends them (fake clock, run `run-1`):

| Step | Call | Events appended | Notes |
|---|---|---|---|
| 1 | `StartWorkflow(demo-1, Echo, "hi")` | 1 `started`, 2 `wt_scheduled` | `dur.02` |
| 2 | `PollWorkflowTask` | 3 `wt_started{scheduled 2}` | history 1..3 returned; lease token 1 |
| 3 | `CompleteWorkflowTask([ScheduleActivity{1, Upper, "hi"}])` | 4 `wt_completed{2, 3}`, 5 `act_scheduled{id 1, wt_completed 4}` | task `run-1/a/5` enqueued |
| 4 | `PollActivityTask` | none | attempt 1, key `demo-1/1`, lease token 2, deadline +10 s |
| 5 | `CompleteActivityTask("HI")` | 6 `act_started{5, attempt 1, lease 2}`, 7 `act_completed{5, 6, "HI"}`, 8 `wt_scheduled` | the result is a trigger |
| 6 | `PollWorkflowTask` | 9 `wt_started{scheduled 8}` | history 1..9 |
| 7 | `CompleteWorkflowTask([CompleteWorkflow{"HI"}])` | 10 `wt_completed{8, 9}`, 11 `completed{"HI"}` | the run is COMPLETED |

This is `TestTaskProtocolHandExample`.

## 4. The interface

```go
// go/durable/server/tasks.go (methods on *Server)
func (s *Server) PollWorkflowTask(ctx context.Context, req *durablev1.PollRequest) (*durablev1.WorkflowTask, error)
func (s *Server) CompleteWorkflowTask(ctx context.Context, req *durablev1.CompleteWorkflowTaskRequest) (*durablev1.Empty, error)
func (s *Server) PollActivityTask(ctx context.Context, req *durablev1.PollRequest) (*durablev1.ActivityTask, error)
func (s *Server) CompleteActivityTask(ctx context.Context, req *durablev1.CompleteActivityRequest) (*durablev1.Empty, error)
func CommandKind(c *durablev1.Command) string
// unexported, used by the later files: registerCommand(kind, handler), commandHandler,
// taskToken (encodeToken, decodeToken), pendingActivity, firstPage

package worker // import "tinyllm/durable/worker"

type ActivityFunc func(ctx context.Context, input []byte) ([]byte, error)
type WorkflowHandler interface { // dur.06's workflow.Registry implements it
	HandleWorkflowTask(ctx context.Context, task *durablev1.WorkflowTask, history []*durablev1.HistoryEvent) ([]*durablev1.Command, error)
}
type Clock interface { Now() time.Time; After(d time.Duration) <-chan time.Time }
type Options struct {
	TaskQueue string; Identity string; MaxActivities int; Workflows WorkflowHandler; Clock Clock
	ReconnectInitial, ReconnectMax, DrainTimeout time.Duration; Logf func(format string, args ...any)
	Failpoint func(name string) error // "dur/activity/poison/<activity id>" before each activity (drill ops.03)
}
type Info struct {
	WorkflowID, RunID, ActivityID, ActivityType, IdempotencyKey string
	Attempt int; HeartbeatDetails []byte; TraceContext map[string]string; Deadline time.Time
}
var ErrHeartbeatTimeout, ErrLeaseLost, ErrCancelRequested, ErrNotActivity error

func New(conn grpc.ClientConnInterface, o Options) *Worker
func (w *Worker) RegisterActivity(name string, fn ActivityFunc)
func (w *Worker) Run(ctx context.Context) error // returns nil after the drain
func InfoFrom(ctx context.Context) (Info, bool)
func Heartbeat(ctx context.Context, details []byte) error
func ReconnectDelay(n int, initial, max time.Duration) time.Duration
```

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `TestTaskProtocolHandExample` | unit | section 3, event by event, over gRPC | the protocol every worker speaks |
| `TestCompleteWorkflowTaskFenced` | unit, fault | an expired task's token, and a second completion, are `FAILED_PRECONDITION` | two workers never both decide |
| `TestStaleActivityLeaseDropsResult` | fault | attempt 2 after an expiry; the late attempt 1 result is refused; attempt 2's is recorded | a presumed-dead worker cannot write |
| `TestStaleQueueTaskSkipped` | fault | a leftover task for an already-run workflow task is acknowledged and skipped | crash leftovers never rerun |
| `TestTriggerWhileWorkflowTaskInFlight` | unit | a result during a running task schedules nothing until that task completes | one task at a time |
| `TestFirstPageWithinOneMiB` | boundary | the first page fits 1 MiB; first page + `GetHistory` rest = the full history | 4 MiB message cap |
| `TestActivityPayloadLimits` | boundary | oversized input, result, and workflow result refused; 2 MiB accepted | payloads by path |
| `TestReconnectDelaySchedule` | unit | 100, 200, 400 ms ... capped at 5 s | restarts are not hammered |
| `TestWorkerRunsWorkflowEndToEnd` | unit | handler, activities, `Info` keys, identity, result | the SDK as data.09 and dur.11 use it |
| `TestWorkerFetchesEveryHistoryPage` | unit | a history over 1 MiB reaches the handler whole | replay needs every event |
| `TestWorkerBoundedConcurrency` | unit, property | 2 slots: exactly 2 running and 2 leased while 4 wait | the pool bound and slot-first polling |
| `TestWorkerDrainsOnCancel` | fault | polling stops, the running activity finishes uncanceled and reports, then `Run` returns | rolling deploys |
| `TestMissedHeartbeatCancelsActivity` | fault | canceled at exactly the heartbeat timeout with `ErrHeartbeatTimeout`; nothing reported | no two copies write |
| `TestWorkerReconnectsAfterServerRestart` | fault | the worker keeps working across a server restart on the same address | drill `ops.02` |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. returning a workflow task without committing its `WorkflowTaskStarted` | the token names an event that does not exist; every completion is refused | `TestTaskProtocolHandExample` (mutant `s01`) |
| 2. completing without fencing (workflow or activity) | a worker presumed dead decides or writes over the live one | `TestCompleteWorkflowTaskFenced`, `TestStaleActivityLeaseDropsResult` (mutants `s04`, `s07`) |
| 3. trusting a queue task over history | a stale duplicate starts a workflow task that already ran | `TestStaleQueueTaskSkipped` (mutant `s09`) |
| 4. an idempotency key that changes (run id, activity id alone) | effects repeat after ContinueAsNew, or collide across workflows | `TestTaskProtocolHandExample`, `TestWorkerRunsWorkflowEndToEnd` (mutants `s02`, `s16`) |
| 5. treating a completion with no commands as "nothing to do" | the workflow task never completes and news that arrived meanwhile is never delivered | `TestTriggerWhileWorkflowTaskInFlight` (mutant `s10`) |
| 6. wrong cross-references in events (the completed id, the started event, the attempt, the identity) | replay (`dur.06`) cannot match commands; describe and audits are wrong | `TestTaskProtocolHandExample`, `TestCompleteWorkflowTaskFenced`, `TestStaleActivityLeaseDropsResult`, `TestWorkerRunsWorkflowEndToEnd` (mutants `s03`, `s05`, `s06`, `s17`) |
| 7. sizes: a first page as big as a message, oversized payloads accepted, `next_page_token` ignored | messages over 4 MiB fail; replay runs on a truncated history | `TestFirstPageWithinOneMiB`, `TestActivityPayloadLimits`, `TestWorkerFetchesEveryHistoryPage` (mutants `s12`, `s13`, `s14`, `s18`) |
| 8. the pool: one slot too many, or polling before taking a slot | more activities than configured; leases expire while tasks wait for a slot | `TestWorkerBoundedConcurrency` (mutants `s19`, `s20`) |
| 9. backoff without a cap, or giving up after one failed poll | a worker waits minutes after a restart, or stops working for good | `TestReconnectDelaySchedule`, `TestWorkerReconnectsAfterServerRestart` (mutants `s15`, `s25`) |
| 10. activities under `Run`'s context, or `Run` returning before they finish | SIGTERM aborts work mid-flight and it is redone; results are lost | `TestWorkerDrainsOnCancel` (mutants `s21`, `s22`) |
| 11. no heartbeat watchdog, or reporting after the heartbeat timeout | a stuck attempt runs on after its task was redelivered, and both write | `TestMissedHeartbeatCancelsActivity` (mutants `s23`, `s24`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `dur.02` | completions build batches and `commit` them; effects enqueue the tasks polled here |
| Back | `dur.03` | every delivery is a lease; `Check` fences completions |
| Forward | `dur.05` | `FailActivityTask` and `RecordHeartbeat` complete the protocol; `activity.IdempotencyKey` reads `Info` |
| Forward | `dur.06` | `workflow.Registry` is the `WorkflowHandler`; `dur.07` and `dur.08` register more commands |
| Forward | `dur.08` | cancellation and signals extend these task handlers and the worker's activity context |
| Forward | `dur.09` | the subprocess runner is an `ActivityFunc` that heartbeats checkpoints |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| one worker build serves every run | worker versioning (build ids) | old runs stay on the build that started them while new runs use the new one | Temporal "Worker Versioning" |
| every activity through the server | local activities | short activities run in the workflow worker, recorded as markers, no round trip | Temporal "Local Activities" |
| one namespace, one service | Nexus | typed cross-namespace calls between teams' workflows | Temporal "Nexus" |
| one poller per kind | poller autoscaling and slot suppliers | pollers and slots sized from load and resource use | Temporal Go SDK `worker.Options` |
