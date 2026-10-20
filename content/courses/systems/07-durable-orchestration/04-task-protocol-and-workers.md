# The task protocol and workers

[The workflow service](02-workflow-service.md) can start runs and [the task queue](03-task-queue.md) can lease tasks, but nothing yet executes anything. This section builds the protocol between the server and the processes that do the work, and the Go SDK for those processes. A worker long-polls for two kinds of task: **workflow tasks**, which decide what to do next, and **activity tasks**, which do it. It runs activities in a bounded pool, survives server restarts, and shuts down without losing or duplicating work.

Once this section is done, `{worker} --queue default` processes a queue, and the rest of the platform builds on it: [CorpusBuild as a durable workflow](../../data-engineering/05-corpus-pipeline/09-corpusbuild-as-a-durable-workflow.md) in *Data Engineering*, [TrainRun and EvalSuite](11-trainrun-and-evalsuite.md), [ModelRelease](12-model-release.md), and [durable agent runs](../../ai-engineering/28-agents/05-durable-agent-runs.md) all register their workflows and activities on this worker.

You will need the server core and the queue from the previous two sections, gRPC from [Protocol Buffers and gRPC](../../programming-and-algorithms/01-language-primers/10-protocol-buffers-and-grpc.md) in *Programming and Algorithms*, the partial-failure thinking of [RPC and protocols](../03-rpc-and-protocols/), and, for the trace context that rides on every task, [Tracing for the serving path](../11-observability/02-tracing-for-the-serving-path.md) in *Systems Engineering*. The contract is `tl.durable.v1.TaskService` in [`proto/tl/durable/v1/durable.proto`](../../../../course/contracts/proto/tl/durable/v1/durable.proto); the Go SDK surface and package API are given below. The full solution is the server half in [`go/durable/server/tasks.go`](../../../../course/solutions/go/durable/server/tasks.go) (`PollWorkflowTask`, `CompleteWorkflowTask` and the command table, `PollActivityTask`, `CompleteActivityTask`) and the worker in [`go/durable/worker/worker.go`](../../../../course/solutions/go/durable/worker/worker.go) (`New`, `RegisterActivity`, `Run`, `Heartbeat`, `InfoFrom`, `ReconnectDelay`).

> **Learning objectives**
> - Trace a workflow from start to completion through the task protocol, event by event.
> - Explain how the $(S, T)$ check and the queue lease token together fence workflow-task completions.
> - Explain why history outranks queue tasks, and how stale deliveries are skipped.
> - Build a worker that takes a pool slot before polling, reconnects with capped backoff, and drains on shutdown.
> - Distinguish shutdown from lease loss, and explain why an activity that loses its lease must report nothing.

## Workflow tasks

A workflow task is a lease on the right to decide the run's next step. Three quantities identify one:

| Symbol | Meaning |
|---|---|
| $S$ | event id of the run's pending `WorkflowTaskScheduled` |
| $T$ | event id of its latest `WorkflowTaskStarted` |
| $k$ | the queue lease token of the delivery |

`PollWorkflowTask` leases task `<run>/wt/S` from queue `wf:<task_queue>` with a 30 s long poll, answering with an empty `task_token` when nothing came. If the run's pending workflow task is no longer $S$ (it already completed, or the run closed), the delivery is stale: the server acknowledges it and keeps polling. Otherwise it appends `WorkflowTaskStarted{scheduled_event_id: S}`, whose id is $T$, and returns the history through $T$. It sends at most 1 MiB of that history (the first page), with a `next_page_token` for the rest, and sets `continue_as_new_suggested` once the history has passed 10,000 events or 32 MiB (see [deterministic replay](06-deterministic-replay.md)). The task token carries the queue, task id, $k$, run id, $S$, and $T$.

`CompleteWorkflowTask(token, commands)` checks that the run is open, its pending task is still $(S, T)$, and the lease $k$ is live; anything else is `FAILED_PRECONDITION` and appends nothing. Then it builds one batch, `WorkflowTaskCompleted{S, T}` followed by each command's events, commits it atomically, and acknowledges the lease.

Each command kind has a handler in a table (`registerCommand`). This section registers `schedule_activity` (which appends `ActivityTaskScheduled`, whose effect enqueues the activity task), `complete`, and `fail`; [deterministic replay](06-deterministic-replay.md), [durable timers](07-durable-timers.md), and [signals, cancellation, and sagas](08-signals-cancellation-and-sagas.md) register theirs. A command after the one that ends the run is `INVALID_ARGUMENT`, and an unknown kind is `UNIMPLEMENTED`.

What happens when a worker dies after polling? It never completes, so the lease expires (the workflow task timeout, 10 s), and the next poll appends a second `WorkflowTaskStarted` for the same $S$. The old token now names a $T$ that is not the latest and is refused.

There is only ever one workflow task at a time. A trigger, such as an activity result, that arrives while a workflow task is running is appended to history but schedules nothing; when the running task completes, its batch schedules the next one (the `needsWT` rule from [the workflow service](02-workflow-service.md)). The worker that runs the current task never sees events after its $T$.

## Activity tasks and fencing

Activity tasks follow the same pattern with one difference: they have no $T$. `PollActivityTask` leases `<run>/a/<scheduled id>` from `act:<task_queue>`, and a delivery whose activity is no longer pending is acknowledged and skipped. The `ActivityTask` carries the input, the attempt (from the queue), the lease token, the deadline, the heartbeat timeout, the starter's trace context, and the **idempotency key** `"<workflow_id>/<activity_id>"`. The key is the same on every attempt, so the activity's side effect can be made idempotent (the subject of [activities](05-activities.md)). It uses the workflow id, not the run id, because Continue-As-New changes the run id.

`CompleteActivityTask(token, lease_token, result)` checks the lease with `Queue.Check` **before** appending anything: a stale token (the task was redelivered) is `FAILED_PRECONDITION`, and the result is dropped. Otherwise it commits one batch, `ActivityTaskStarted{attempt, lease_token}` and `ActivityTaskCompleted{result}` (plus the `WorkflowTaskScheduled` the trigger needs), then acknowledges. Payloads over 2 MiB (activity input, activity result, workflow result) are `INVALID_ARGUMENT`; large data goes to `/artifacts` and travels by path.

## The worker

That is the server's half. The worker's half fits in a few lines of pseudocode:

```
Run(ctx):
    activity poller:  loop { take a slot; poll; run the task in a goroutine that frees the slot }
    workflow poller:  loop { poll; fetch every history page; commands = handler(task, history); complete }
    on ctx done: stop polling now; wait for running activities (up to DrainTimeout), then cancel them
```

Five design decisions hide in that sketch.

**A bounded pool, slot first.** With `MaxActivities` set to $n$, the poller acquires a slot before polling. Polling first and then waiting for a slot holds a lease on a task the worker cannot start; the lease expires, the task is redelivered, and the expiry counts as a failed attempt.

**Activity contexts are not `Run`'s context.** SIGTERM ends `Run`'s context, but an activity in flight must be allowed to finish and report, so its context derives from a separate "work" context that only the drain timeout cancels.

**A heartbeat watchdog.** An activity with a heartbeat timeout $h$ must call `Heartbeat` at least every $h$, because the server gives its task to someone else after $h$. The worker watches the same interval on its own clock, and when it passes, cancels the activity's context with cause `ErrHeartbeatTimeout` and reports **nothing**: the lease is gone, and a report could overwrite the live attempt. A heartbeat refused with `FAILED_PRECONDITION` cancels with `ErrLeaseLost` in the same way.

**Reconnect with backoff.** After the $n$-th consecutive failed poll (the server restarted, say), the worker waits $\min(b_0 \cdot 2^{n-1}, b_{\max})$ (`ReconnectDelay`, 100 ms doubling to 5 s by default) and tries again. Reports (complete, fail) are retried while the server answers `UNAVAILABLE`, because the lease survives a server restart.

**Errors become failures.** An activity's error becomes a `Failure{message, type}`. An error with a `NonRetryable() bool` method returning true is non-retryable, and one with `FailureType() string` names the type ([activities](05-activities.md) builds such errors). A panic becomes type `Panic` with its stack.

## Working it by hand

Consider `Echo("hi")`: one workflow task schedules activity `Upper("hi")`, and the next completes the run with its result. Here are the events as the server appends them, on a fake clock with run `run-1`:

| Step | Call | Events appended | Notes |
|---|---|---|---|
| 1 | `StartWorkflow(demo-1, Echo, "hi")` | 1 `started`, 2 `wt_scheduled` | as in the workflow service |
| 2 | `PollWorkflowTask` | 3 `wt_started{scheduled 2}` | history 1..3 returned; lease token 1 |
| 3 | `CompleteWorkflowTask([ScheduleActivity{1, Upper, "hi"}])` | 4 `wt_completed{2, 3}`, 5 `act_scheduled{id 1, wt_completed 4}` | task `run-1/a/5` enqueued |
| 4 | `PollActivityTask` | none | attempt 1, key `demo-1/1`, lease token 2, deadline +10 s |
| 5 | `CompleteActivityTask("HI")` | 6 `act_started{5, attempt 1, lease 2}`, 7 `act_completed{5, 6, "HI"}`, 8 `wt_scheduled` | the result is a trigger |
| 6 | `PollWorkflowTask` | 9 `wt_started{scheduled 8}` | history 1..9 |
| 7 | `CompleteWorkflowTask([CompleteWorkflow{"HI"}])` | 10 `wt_completed{8, 9}`, 11 `completed{"HI"}` | the run is COMPLETED |

Notice step 4: polling an activity appends nothing. The start of an activity is recorded only when it completes, in step 5.

## Designing the API

The server methods and the worker package together look like this:

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

The `WorkflowHandler` is implemented by the workflow registry in [deterministic replay](06-deterministic-replay.md). The `Failpoint` hook is what the [poison-task drill](../12-incident-response-and-chaos/04-poison-task.md) uses to make an activity fail on purpose.

## Building it step by step

The reasoning behind the implementation runs as follows. Try to predict each step.

1. **Deciding the next step must be exclusive.** If two workers could each complete a workflow task for the same run, the history would contain two decisions from the same state. So a workflow task is a lease, and a completion must name the exact task it was given: the pending scheduled event $S$, the latest started event $T$, and the live lease token $k$.
2. **The token must carry everything the server needs to check that.** The server keeps no state about deliveries beyond the queue and history, so the opaque `task_token` encodes queue, task id, $k$, run id, $S$, and $T$. A redelivery appends a new $T$, which makes every older token wrong by construction.
3. **History outranks the queue.** A queue task can outlive its purpose (a crash between commit and ack, or recovery re-enqueuing). The server checks history before acting on a delivery, and acknowledges and skips any delivery that no longer matches.
4. **Completion is atomic.** `WorkflowTaskCompleted` and all events of the commands go in one `commit`, so a crash cannot record half a decision. The lease is acknowledged only after the commit.
5. **Activity results are fenced before anything is written.** A late result from attempt 1 after attempt 2 started must be dropped, so `Queue.Check` runs before the append. The idempotency key must be stable across attempts and across Continue-As-New, so it is built from the workflow id and activity id, not the run id.
6. **On the worker, capacity is claimed before work.** A lease starts its clock at poll time, so a worker that polls without a free slot burns the lease while waiting. Take the slot first.
7. **Shutdown and lease loss are different cancellations.** SIGTERM should let running activities finish and report; losing the lease should stop the activity and report nothing. Two contexts (`Run`'s, and a separate work context) and a cancel cause express that.

### Starting a workflow task

In [`tasks.go`](../../../../course/solutions/go/durable/server/tasks.go), `PollWorkflowTask` loops over `Queue.Poll` on `wf:<task_queue>` until the poll wait is used up, skipping stale deliveries. Each delivery goes through `startWorkflowTask`:

```go
func (s *Server) startWorkflowTask(ctx context.Context, l queue.Lease, identity string) (*durablev1.WorkflowTask, error) {
	runID, _, sched, err := parseTaskID(l.Task.ID)
	if err != nil {
		s.o.Queue.Complete(ctx, l) // garbage in the queue: drop it
		return nil, nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	r := s.runs[runID]
	// History is the truth: the delivery is valid only if S is still the pending task.
	if r == nil || r.status != durablev1.WorkflowStatus_WORKFLOW_STATUS_RUNNING || r.wtScheduled != sched {
		s.o.Queue.Complete(ctx, l) // stale: ack and let the poll loop continue
		return nil, nil
	}
	b := &batch{r: r}
	st := s.add(b, &durablev1.HistoryEvent{Attrs: &durablev1.HistoryEvent_WtStarted{
		WtStarted: &durablev1.WorkflowTaskStarted{ScheduledEventId: sched, Identity: identity},
	}})
	if err := s.commit(ctx, b); err != nil { // T must exist in history before the token names it
		return nil, err
	}
	page, next := firstPage(r.history)
	return &durablev1.WorkflowTask{
		TaskToken:  encodeToken(taskToken{Queue: l.Queue, TaskID: l.Task.ID, Lease: l.Token, RunID: r.id, Sched: sched, Started: st.GetEventId()}),
		WorkflowId: r.workflowID, RunId: r.id, WorkflowType: r.wfType,
		History: page, NextPageToken: next,
		ContinueAsNewSuggested: len(r.history) > ContinueAsNewEvents || r.bytes > ContinueAsNewBytes,
	}, nil
}

// The longest prefix within 1 MiB (always at least one event), so a task fits
// under gRPC's 4 MiB message cap; the rest comes from GetHistory.
func firstPage(h []*durablev1.HistoryEvent) ([]*durablev1.HistoryEvent, []byte) {
	size := 0
	for i, e := range h {
		size += proto.Size(e)
		if size > FirstPageBytes && i > 0 {
			return h[:i:i], PageToken(int64(i + 1)) // full slice expression: the caller cannot append into history
		}
	}
	return h[:len(h):len(h)], nil
}
```

### Completing it: fence, then one batch

Completion checks both fences, then builds and commits the batch:

```go
func (s *Server) CompleteWorkflowTask(ctx context.Context, req *durablev1.CompleteWorkflowTaskRequest) (*durablev1.Empty, error) {
	tok, err := decodeToken(req.GetTaskToken())
	// ...
	s.mu.Lock()
	defer s.mu.Unlock()
	r := s.runs[tok.RunID]
	// The run's current task must be exactly (S, T): a redelivered task has a newer T.
	if r == nil || r.status != durablev1.WorkflowStatus_WORKFLOW_STATUS_RUNNING || r.wtScheduled != tok.Sched || r.wtStarted != tok.Started {
		return nil, status.Error(codes.FailedPrecondition, "not the run's current workflow task")
	}
	if err := s.o.Queue.Check(tok.lease()); err != nil { // and its lease must be live
		return nil, status.Error(codes.FailedPrecondition, "workflow task lease lost")
	}
	b := &batch{r: r}
	// Always appended, even with zero commands: this is what ends the task and
	// lets commit schedule the next one if news arrived meanwhile (needsWT).
	done := s.add(b, &durablev1.HistoryEvent{Attrs: &durablev1.HistoryEvent_WtCompleted{
		WtCompleted: &durablev1.WorkflowTaskCompleted{ScheduledEventId: tok.Sched, StartedEventId: tok.Started, Identity: req.GetIdentity()},
	}})
	for _, c := range req.GetCommands() {
		if closing(b.events[len(b.events)-1]) {
			return nil, status.Error(codes.InvalidArgument, "a command after the one that ends the run")
		}
		h := commandHandlers[CommandKind(c)] // the table later sections extend
		if h == nil {
			return nil, status.Errorf(codes.Unimplemented, "command %q is not supported", CommandKind(c))
		}
		if err := h(s, b, done.GetEventId(), c); err != nil {
			return nil, err // nothing committed: the batch is discarded
		}
	}
	if err := s.commit(ctx, b); err != nil {
		return nil, err
	}
	s.o.Queue.Complete(ctx, tok.lease()) // ack only after the decision is durable
	return &durablev1.Empty{}, nil
}
```

The `schedule_activity` handler, `scheduleActivity`, validates the command (ids required, input at most 2 MiB, `start_to_close_ms` set, the activity id not already pending) and appends `ActivityTaskScheduled` with `WorkflowTaskCompletedEventId: completed`. That cross-reference is what replay uses to match commands to events. Its effect in `commit` enqueues `<run>/a/<scheduled id>`.

### Activity tasks: stable key, fence before append

Activity delivery builds the task with its stable idempotency key; `pendingActivity` is the fence every activity call goes through:

```go
func (s *Server) startActivityTask(ctx context.Context, l queue.Lease) *durablev1.ActivityTask {
	// ... resolve run and r.activities[sched]; ack and skip when not pending
	a.attempt = int32(l.Attempt) // the attempt number comes from the queue's delivery count
	details := l.Details
	if details == nil {
		details = a.lastHeartbeat
	}
	return &durablev1.ActivityTask{
		// ... token (no Started: activities have no T), ids, type, input, attempt
		IdempotencyKey:       r.workflowID + "/" + a.attrs.GetActivityId(), // same on every attempt and across runs
		LastHeartbeatDetails: details, TraceContext: r.traceContext, LeaseToken: l.Token,
		DeadlineUnixMs: l.Deadline.UnixMilli(), HeartbeatTimeoutMs: a.attrs.GetOptions().GetHeartbeatTimeoutMs(),
	}
}

func (s *Server) pendingActivity(tok taskToken) (*run, *activity, error) {
	r := s.runs[tok.RunID]
	if r == nil || r.status != durablev1.WorkflowStatus_WORKFLOW_STATUS_RUNNING || r.activities[tok.Sched] == nil {
		return nil, nil, status.Error(codes.FailedPrecondition, "the activity is no longer pending")
	}
	if err := s.o.Queue.Check(tok.lease()); err != nil {
		return nil, nil, status.Error(codes.FailedPrecondition, "stale lease token: the activity was redelivered or timed out")
	}
	return r, r.activities[tok.Sched], nil
}
```

`CompleteActivityTask` checks the result size, calls `pendingActivity` under `s.mu`, then commits `ActivityTaskStarted{attempt, lease_token}` and `ActivityTaskCompleted{started id, result}` together. Recording the start only at completion keeps history free of attempts that never finished, and the attempt and lease token in it say which delivery produced the result. `act_completed` is a trigger, so `commit` adds the next `wt_scheduled`, as in step 5 of the worked example.

### The worker's run loop

The worker lives in [`worker.go`](../../../../course/solutions/go/durable/worker/worker.go). `Run` sets up the two contexts and the slot semaphore, then waits for shutdown and drains:

```go
func (w *Worker) Run(ctx context.Context) error {
	// ...
	// work survives ctx's cancellation; only the drain timeout cancels it.
	work, cancelWork := context.WithCancel(context.WithoutCancel(ctx))
	defer cancelWork()
	var pollers, inflight sync.WaitGroup
	slots := make(chan struct{}, w.o.MaxActivities) // a counting semaphore
	// ... start pollActivities(ctx, work, slots, &inflight) and, if set, pollWorkflows(ctx, work)
	<-ctx.Done()
	pollers.Wait() // polling stops at once (polls use ctx)
	done := make(chan struct{})
	go func() { inflight.Wait(); close(done) }()
	select {
	case <-done: // every running activity finished and reported
	case <-w.o.Clock.After(w.o.DrainTimeout):
		cancelWork() // out of time: cancel what is left
		<-done
	}
	return nil
}

func (w *Worker) pollActivities(ctx, work context.Context, slots chan struct{}, inflight *sync.WaitGroup) {
	failures := 0
	for {
		select {
		case slots <- struct{}{}: // slot first: never lease a task we cannot start
		case <-ctx.Done():
			return
		}
		t, err := w.tasks.PollActivityTask(ctx, &durablev1.PollRequest{TaskQueue: w.o.TaskQueue, Identity: w.o.Identity})
		if err != nil {
			<-slots
			// ... return on ctx end; otherwise failures++ and back off ReconnectDelay(failures)
			continue
		}
		failures = 0
		if len(t.GetTaskToken()) == 0 { // an empty long poll
			<-slots
			continue
		}
		inflight.Add(1)
		go func() {
			defer inflight.Done()
			defer func() { <-slots }()
			w.runActivity(work, t) // under work, not ctx
		}()
	}
}
```

### One activity attempt: watchdog, cause, report

Inside `runActivity`, the cancel cause is what distinguishes "the lease is gone" from every other ending:

```go
ctx, cancel := context.WithCancelCause(work)
defer cancel(nil)
beat := make(chan struct{}, 1)
// ... inf := &Info{... IdempotencyKey, Attempt, HeartbeatDetails, Deadline ...}
inf.heartbeat = func(details []byte) error {
	resp, err := w.tasks.RecordHeartbeat(ctx, &durablev1.HeartbeatRequest{TaskToken: t.GetTaskToken(), LeaseToken: t.GetLeaseToken(), Details: details})
	if status.Code(err) == codes.FailedPrecondition {
		cancel(ErrLeaseLost) // the server gave the task away
		return ErrLeaseLost
	}
	// ... cancel(ErrCancelRequested) when the server says so (dur.08)
	select {
	case beat <- struct{}{}: // reset the watchdog; non-blocking, one pending beat is enough
	default:
	}
	return nil
}
if hb := time.Duration(t.GetHeartbeatTimeoutMs()) * time.Millisecond; hb > 0 {
	go w.watchdog(ctx, cancel, beat, hb) // cancels with ErrHeartbeatTimeout after hb of silence
}
res, err := call(context.WithValue(ctx, infoKey{}, inf), w.poisoned(t.GetActivityId(), fn), t.GetInput())
if cause := context.Cause(ctx); errors.Is(cause, ErrHeartbeatTimeout) || errors.Is(cause, ErrLeaseLost) {
	return // the lease is gone: any report could overwrite the live attempt
}
if err != nil {
	w.fail(work, t, failureOf(err))
	return
}
w.retry(work, func(ctx context.Context) error { /* CompleteActivityTask with the lease token */ })
```

`call` recovers a panic into an error of type `Panic` with the stack. `failureOf` uses `errors.As` to find `FailureType()` and `NonRetryable()` anywhere in the wrap chain. `retry` repeats a report only while the server answers `UNAVAILABLE` (the server keeps the lease across a restart), up to 20 times with backoff; any other answer, including a refusal, is final.

The reconnect delay is a capped doubling:

```go
func ReconnectDelay(n int, initial, max time.Duration) time.Duration {
	if n < 1 {
		return 0
	}
	d := initial
	for i := 1; i < n && d < max; i++ { // stop doubling at the cap: no overflow for large n
		d *= 2
	}
	if d > max {
		d = max
	}
	return d // 100, 200, 400 ms ... 5 s
}
```

The workflow poller handles one task at a time: poll, `fullHistory` (the first page plus a `GetHistory` stream from `next_page_token`), the handler, then a retried `CompleteWorkflowTask`. A handler error leaves the task to time out rather than completing with a guess, so a broken workflow is visibly stuck instead of silently wrong.

The full solution is in [`tasks.go`](../../../../course/solutions/go/durable/server/tasks.go) and [`worker.go`](../../../../course/solutions/go/durable/worker/worker.go).

### Why this design

- **Cost.** Each workflow task costs two commits (started, completed) and a history transfer. Sending the whole history every time is O(history) per task, which is why the first page is capped and why production adds sticky queues. Each activity costs one commit, at completion.
- **Fencing in two layers.** The $(S, T)$ check protects against decisions on stale history; the queue token protects against expired leases. Either alone leaves a gap: $(S, T)$ alone would accept a completion whose lease already expired but has not yet been redelivered, and the token alone would not catch a task whose run state moved on.
- **Command table over a switch.** Later sections add commands without editing this file, which keeps each change local.
- **Worker-side watchdog.** The server already reassigns the task after the heartbeat timeout; the worker's own watchdog is what stops the stale copy from continuing to do work (and side effects) in the background.

## What a correct implementation must handle

The tests check the protocol and the worker against these cases:

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| hand example protocol | unit | the worked example, event by event, over gRPC | the protocol every worker speaks |
| workflow task completion fenced | unit, fault | an expired task's token, and a second completion, are `FAILED_PRECONDITION` | two workers never both decide |
| stale activity lease drops its result | fault | attempt 2 after an expiry; the late attempt 1 result is refused; attempt 2's is recorded | a presumed-dead worker cannot write |
| stale queue task skipped | fault | a leftover task for an already-run workflow task is acknowledged and skipped | crash leftovers never rerun |
| trigger while a workflow task is in flight | unit | a result during a running task schedules nothing until that task completes | one task at a time |
| first page within 1 MiB | boundary | the first page fits 1 MiB; first page + `GetHistory` rest = the full history | 4 MiB message cap |
| activity payload limits | boundary | oversized input, result, and workflow result refused; 2 MiB accepted | payloads by path |
| reconnect delay schedule | unit | 100, 200, 400 ms ... capped at 5 s | restarts are not hammered |
| worker runs a workflow end to end | unit | handler, activities, `Info` keys, identity, result | the SDK as the platform workflows use it |
| worker fetches every history page | unit | a history over 1 MiB reaches the handler whole | replay needs every event |
| worker bounded concurrency | unit, property | 2 slots: exactly 2 running and 2 leased while 4 wait | the pool bound and slot-first polling |
| worker drains on cancel | fault | polling stops, the running activity finishes uncanceled and reports, then `Run` returns | rolling deploys |
| missed heartbeat cancels the activity | fault | canceled at exactly the heartbeat timeout with `ErrHeartbeatTimeout`; nothing reported | no two copies write |
| worker reconnects after a server restart | fault | the worker keeps working across a server restart on the same address | the [SIGKILL-loop drill](../12-incident-response-and-chaos/03-durable-kill9.md) |

## What can go wrong

Most mistakes here break exclusivity: two workers act on the same decision or the same activity.

Returning a workflow task without committing its `WorkflowTaskStarted` means the token names an event that does not exist, and every completion is refused. `startWorkflowTask` commits `WorkflowTaskStarted` before building the token.

> **Warning:** Completing without fencing, for workflows or activities, lets a worker presumed dead decide or write over the live one. The solution runs the $(S, T)$ check plus `Queue.Check` before any append.

Trusting a queue task over history lets a stale duplicate start a workflow task that already ran; a delivery whose $S$ is not `r.wtScheduled` is acknowledged and skipped. An idempotency key that changes (built from the run id, or the activity id alone) makes effects repeat after Continue-As-New, or collide across workflows; the key is `r.workflowID + "/" + activity id`.

Treating a completion with no commands as "nothing to do" means the workflow task never completes and news that arrived meanwhile is never delivered. `WorkflowTaskCompleted` is appended for every completion, so `commit` can honour `needsWT`. Related to that, wrong cross-references in events (the completed id, the started event, the attempt, the identity) leave replay unable to match commands and make describe and audits wrong; each event carries the ids from the token (`Sched`, `Started`, completed id, attempt, lease token, identity).

Size limits fail in three ways: a first page as big as a message, oversized payloads accepted, or `next_page_token` ignored. Messages over 4 MiB fail, or replay runs on a truncated history. `firstPage` caps at 1 MiB, payloads are checked against `MaxPayloadBytes`, and `fullHistory` follows `next_page_token`.

On the worker, a pool with one slot too many, or polling before taking a slot, runs more activities than configured and lets leases expire while tasks wait for a slot; the fix is a `MaxActivities`-sized semaphore taken before `PollActivityTask`. Backoff without a cap makes a worker wait minutes after a restart, and giving up after one failed poll stops it working for good; `ReconnectDelay` caps at `ReconnectMax`, and the pollers back off and continue. Running activities under `Run`'s context, or returning from `Run` before they finish, means SIGTERM aborts work mid-flight to be redone and loses results; activities run under `work`, and `Run` waits on `inflight` before returning.

> **Warning:** Without a heartbeat watchdog, or if the worker reports after the heartbeat timeout, a stuck attempt runs on after its task was redelivered, and both copies write. The watchdog cancels with `ErrHeartbeatTimeout`, and `runActivity` returns without reporting.

## Where this leads

This section built on [the workflow service](02-workflow-service.md), whose `commit` every completion goes through and whose effects enqueue the tasks polled here, and on [the task queue](03-task-queue.md), where every delivery is a lease and `Check` fences completions.

Looking ahead, [activities](05-activities.md) completes the protocol with `FailActivityTask` and `RecordHeartbeat`, and its `activity.IdempotencyKey` reads `Info`. In [deterministic replay](06-deterministic-replay.md), `workflow.Registry` becomes the `WorkflowHandler`, and [durable timers](07-durable-timers.md) and [signals, cancellation, and sagas](08-signals-cancellation-and-sagas.md) register more commands; the latter also extends these task handlers and the worker's activity context for cancellation and signals. [Subprocess activities](09-subprocess-activities.md) is an `ActivityFunc` that heartbeats checkpoints.

Production workers go further:

- **Worker versioning.** Instead of one worker build serving every run, build ids keep old runs on the build that started them while new runs use the new one. See Temporal "Worker Versioning".
- **Local activities.** Instead of every activity going through the server, short activities run in the workflow worker, recorded as markers, with no round trip. See Temporal "Local Activities".
- **Nexus.** Beyond one namespace and one service, typed cross-namespace calls between teams' workflows. See Temporal "Nexus".
- **Poller autoscaling and slot suppliers.** Instead of one poller per kind, pollers and slots are sized from load and resource use. See the Temporal Go SDK `worker.Options`.

### Further reading

- The Temporal worker docs (pollers, slots, graceful shutdown).
- Marc Brooker, *Exponential Backoff and Jitter* (AWS Architecture Blog, 2015).

## Key terms

- **Workflow task**: a lease on the right to decide a run's next step, identified by $(S, T)$ and a lease token.
- **Activity task**: a lease on running one activity attempt.
- **Task token**: the opaque token encoding queue, task id, lease token, run id, $S$, and $T$, which a completion must present.
- **Command**: a decision returned by a workflow task (schedule an activity, complete, fail, ...), turned into events by a handler in the command table.
- **Idempotency key**: `"<workflow_id>/<activity_id>"`, stable across attempts and runs, so an activity's effect can be applied once.
- **Heartbeat watchdog**: the worker-side timer that cancels an activity, without reporting, once its heartbeat timeout passes.
- **Drain**: on shutdown, stopping polls at once while letting running activities finish and report, up to a timeout.

## Summary

- A workflow task is a lease on the right to decide the run's next step: polling appends `WorkflowTaskStarted`, completing appends `WorkflowTaskCompleted` and the events of the commands in one batch, and only the current task with a live lease may complete.
- History is the truth and queue tasks are hints: a delivered task whose scheduled event is no longer pending is acknowledged and skipped.
- The worker takes a pool slot **before** it polls, so it never leases work it cannot run.
- Shutdown drains: polling stops at once, and running activities finish and report; an activity that misses its heartbeat is canceled and reports nothing.

## Review questions

1. A worker polls a workflow task, pauses for 15 s, then completes it. What does the server answer, and which check fires?

   <details><summary>Answer</summary>

   The 10 s lease expired; the next poll appended a second `WorkflowTaskStarted`, so `r.wtStarted` is newer than the token's `Started`. The $(S, T)$ check fails first with `FAILED_PRECONDITION`. If no one had polled yet, `Queue.Check` would fail on the expired lease instead.

   </details>

2. Why is the idempotency key `"<workflow_id>/<activity_id>"` and not `"<run_id>/<activity_id>"`?

   <details><summary>Answer</summary>

   Continue-As-New starts a new run with a new run id but the same workflow id. A key with the run id would let a continued workflow repeat an effect that the previous run already performed; the workflow id plus activity id names the piece of work regardless of run.

   </details>

3. Why does `CompleteWorkflowTask` append `WorkflowTaskCompleted` even when the command list is empty?

   <details><summary>Answer</summary>

   It ends the current task so another can be scheduled. If an activity result arrived while the task ran, `needsWT` is set and only the batch with `wt_completed` schedules the next task; skipping it would leave the run with a task that never completes and news that is never delivered.

   </details>

4. Why does the activity poller take a slot before polling instead of after?

   <details><summary>Answer</summary>

   The lease's deadline starts at poll time. Polling without capacity holds a lease the worker cannot use; it expires while the task waits for a slot, the task is redelivered, and the expiry counts as a failed attempt toward the DLQ.

   </details>

5. On SIGTERM, why do running activities not see their context canceled?

   <details><summary>Answer</summary>

   Their context derives from `work`, built with `context.WithoutCancel(ctx)`, so canceling `Run`'s context only stops polling. Activities finish and report, which avoids redoing their work on another worker. Only the drain timeout cancels `work`.

   </details>
