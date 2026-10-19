# Task gRPC protocol, Go worker SDK, worker pool

## Overview

| | |
|---|---|
| **Module** | `dur.04` · build · Go · Pass 8 · 5 to 7 h |
| **Solution** | [`go/durable/server/tasks.go`](../../../../course/solutions/go/durable/server/tasks.go): `PollWorkflowTask`, `CompleteWorkflowTask` (and the command table), `PollActivityTask`, `CompleteActivityTask`; [`go/durable/worker/worker.go`](../../../../course/solutions/go/durable/worker/worker.go): `New`, `RegisterActivity`, `Run`, `Heartbeat`, `InfoFrom`, `ReconnectDelay` |
| **Contract** | `tl.durable.v1.TaskService` in [`proto/tl/durable/v1/durable.proto`](../../../../course/contracts/proto/tl/durable/v1/durable.proto); the Go SDK surface and package API are section 4 |
| **Needs** | `dur.02` the server core, `dur.03` the queue; reading: [`obs.01` tracing](../../systems/04-observability/01-tracing-for-the-serving-path.md), [`lang.10` gRPC](../../programming-languages/02-language-and-tool-primers/10-protocol-buffers-and-grpc.md), [RPC and protocols](../02-rpc-and-protocols/) |
| **Used by** | `dur.05` activity failures, retries, and heartbeats and `dur.08` cancellation go through these task and activity handlers |
| **Optional depth** | the Temporal worker docs (pollers, slots, graceful shutdown); Brooker, *Exponential Backoff and Jitter* (AWS Architecture Blog, 2015) |

## Key Takeaways

- A workflow task is a lease on the right to decide the run's next step: polling appends `WorkflowTaskStarted`, completing appends `WorkflowTaskCompleted` and the events of the commands in one batch, and only the current task with a live lease may complete.
- History is the truth and queue tasks are hints: a delivered task whose scheduled event is no longer pending is acknowledged and skipped.
- The worker takes a pool slot **before** it polls, so it never leases work it cannot run.
- Shutdown drains: polling stops at once, running activities finish and report; an activity that misses its heartbeat is canceled and reports nothing.

## How to read this chapter

Sections 1 to 3 build the task protocol from first principles and trace one workflow event by event. Section 4 fixes the interface. Section 5 walks through the server half (`tasks.go`) and then the worker SDK (`worker.go`), explaining why each choice is the right one. Try to predict each step of the solution before reading it.

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

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| hand example protocol | unit | section 3, event by event, over gRPC | the protocol every worker speaks |
| workflow task completion fenced | unit, fault | an expired task's token, and a second completion, are `FAILED_PRECONDITION` | two workers never both decide |
| stale activity lease drops its result | fault | attempt 2 after an expiry; the late attempt 1 result is refused; attempt 2's is recorded | a presumed-dead worker cannot write |
| stale queue task skipped | fault | a leftover task for an already-run workflow task is acknowledged and skipped | crash leftovers never rerun |
| trigger while a workflow task is in flight | unit | a result during a running task schedules nothing until that task completes | one task at a time |
| first page within 1 MiB | boundary | the first page fits 1 MiB; first page + `GetHistory` rest = the full history | 4 MiB message cap |
| activity payload limits | boundary | oversized input, result, and workflow result refused; 2 MiB accepted | payloads by path |
| reconnect delay schedule | unit | 100, 200, 400 ms ... capped at 5 s | restarts are not hammered |
| worker runs a workflow end to end | unit | handler, activities, `Info` keys, identity, result | the SDK as data.09 and dur.11 use it |
| worker fetches every history page | unit | a history over 1 MiB reaches the handler whole | replay needs every event |
| worker bounded concurrency | unit, property | 2 slots: exactly 2 running and 2 leased while 4 wait | the pool bound and slot-first polling |
| worker drains on cancel | fault | polling stops, the running activity finishes uncanceled and reports, then `Run` returns | rolling deploys |
| missed heartbeat cancels the activity | fault | canceled at exactly the heartbeat timeout with `ErrHeartbeatTimeout`; nothing reported | no two copies write |
| worker reconnects after a server restart | fault | the worker keeps working across a server restart on the same address | drill `ops.02` |

## 5. The solution, step by step

### From first principles

1. **Deciding the next step must be exclusive.** If two workers could each complete a workflow task for the same run, the history would contain two decisions from the same state. So a workflow task is a lease (from `dur.03`) and a completion must name the exact task it was given: the pending scheduled event $S$, the latest started event $T$, and the live lease token $k$.
2. **The token must carry everything the server needs to check that.** The server is stateless about deliveries beyond the queue and history, so the opaque `task_token` encodes queue, task id, $k$, run id, $S$, and $T$. A redelivery appends a new $T$, which makes every older token wrong by construction.
3. **History outranks the queue.** A queue task can outlive its purpose (a crash between commit and ack, or recovery re-enqueuing). The server checks history before acting on a delivery, and acknowledges and skips any that no longer match.
4. **Completion is atomic.** `WorkflowTaskCompleted` and all events of the commands go in one `commit`, so a crash cannot record half a decision. The lease is acknowledged only after the commit.
5. **Activity results are fenced before anything is written.** A late result from attempt 1 after attempt 2 started must be dropped, so `Queue.Check` runs before the append. The idempotency key must be stable across attempts and across Continue-As-New, so it is built from the workflow id and activity id, not the run id.
6. **On the worker, capacity is claimed before work.** A lease starts its clock at poll time, so a worker that polls without a free slot burns the lease while waiting. Take the slot first.
7. **Shutdown and lease loss are different cancellations.** SIGTERM should let running activities finish and report; losing the lease should stop the activity and report nothing. Two contexts (Run's, and a separate work context) and a cancel cause express that.

### The code, annotated

**Starting a workflow task** ([`tasks.go`](../../../../course/solutions/go/durable/server/tasks.go)). `PollWorkflowTask` loops over `Queue.Poll` on `wf:<task_queue>` until the poll wait is used up, skipping stale deliveries.

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

**Completing it: fence, then one batch.**

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
		h := commandHandlers[CommandKind(c)] // the table later modules extend
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

`scheduleActivity` validates the command (ids required, input at most 2 MiB, `start_to_close_ms` set, the activity id not already pending) and appends `ActivityTaskScheduled` with `WorkflowTaskCompletedEventId: completed`, the cross-reference replay uses to match commands to events (`dur.06`). Its effect in `commit` enqueues `<run>/a/<scheduled id>`.

**Activity tasks: stable key, fence before append.**

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

`CompleteActivityTask` checks the result size, calls `pendingActivity` under `s.mu`, then commits `ActivityTaskStarted{attempt, lease_token}` and `ActivityTaskCompleted{started id, result}` together. Recording the start only at completion keeps history free of attempts that never finished; the attempt and lease token in it say which delivery produced the result. `act_completed` is a trigger, so `commit` adds the next `wt_scheduled` (step 5 of the worked example).

**The worker's run loop** ([`worker.go`](../../../../course/solutions/go/durable/worker/worker.go)).

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

**One activity attempt: watchdog, cause, report.**

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

`call` recovers a panic into an error of type `Panic` with the stack. `failureOf` uses `errors.As` to find `FailureType()` and `NonRetryable()` anywhere in the wrap chain. `retry` repeats a report only while the server answers `UNAVAILABLE` (it keeps the lease across a restart), up to 20 times with backoff; any other answer, including a refusal, is final.

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

Full solution: [`tasks.go`](../../../../course/solutions/go/durable/server/tasks.go), [`worker.go`](../../../../course/solutions/go/durable/worker/worker.go).

### Why this solution

- **Cost.** Each workflow task costs two commits (started, completed) and a history transfer. Sending the whole history every time is O(history) per task, which is why the first page is capped and why production adds sticky queues (Going further). Each activity costs one commit at completion.
- **Fencing in two layers.** The $(S, T)$ check protects against decisions on stale history; the queue token protects against expired leases. Either alone leaves a gap: $(S, T)$ alone would accept a completion whose lease already expired but has not yet been redelivered; the token alone would not catch a task whose run state moved on.
- **Command table over a switch.** Later modules add commands without editing this file, which keeps each module's change local.
- **Worker-side watchdog.** The server already reassigns the task after the heartbeat timeout; the worker's own watchdog is what stops the stale copy from continuing to do work (and side effects) in the background.

### Explain it back

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

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. returning a workflow task without committing its `WorkflowTaskStarted` | the token names an event that does not exist; every completion is refused | `startWorkflowTask` commits `WorkflowTaskStarted` before building the token |
| 2. completing without fencing (workflow or activity) | a worker presumed dead decides or writes over the live one | the $(S, T)$ check plus `Queue.Check` before any append |
| 3. trusting a queue task over history | a stale duplicate starts a workflow task that already ran | a delivery whose $S$ is not `r.wtScheduled` is acknowledged and skipped |
| 4. an idempotency key that changes (run id, activity id alone) | effects repeat after ContinueAsNew, or collide across workflows | the key is `r.workflowID + "/" + activity id` |
| 5. treating a completion with no commands as "nothing to do" | the workflow task never completes and news that arrived meanwhile is never delivered | `WorkflowTaskCompleted` is appended for every completion, so `commit` can honour `needsWT` |
| 6. wrong cross-references in events (the completed id, the started event, the attempt, the identity) | replay (`dur.06`) cannot match commands; describe and audits are wrong | each event carries the ids from the token (`Sched`, `Started`, completed id, attempt, lease token, identity) |
| 7. sizes: a first page as big as a message, oversized payloads accepted, `next_page_token` ignored | messages over 4 MiB fail; replay runs on a truncated history | `firstPage` caps at 1 MiB; payload checks against `MaxPayloadBytes`; `fullHistory` follows `next_page_token` |
| 8. the pool: one slot too many, or polling before taking a slot | more activities than configured; leases expire while tasks wait for a slot | a `MaxActivities`-sized semaphore taken before `PollActivityTask` |
| 9. backoff without a cap, or giving up after one failed poll | a worker waits minutes after a restart, or stops working for good | `ReconnectDelay` caps at `ReconnectMax`; the pollers back off and continue |
| 10. activities under `Run`'s context, or `Run` returning before they finish | SIGTERM aborts work mid-flight and it is redone; results are lost | activities run under `work`; `Run` waits on `inflight` before returning |
| 11. no heartbeat watchdog, or reporting after the heartbeat timeout | a stuck attempt runs on after its task was redelivered, and both write | the watchdog cancels with `ErrHeartbeatTimeout`, and `runActivity` returns without reporting |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `dur.02` | completions build batches and `commit` them; effects enqueue the tasks polled here |
| Back | `dur.03` | every delivery is a lease; `Check` fences completions |
| Forward | `dur.05` | `FailActivityTask` and `RecordHeartbeat` complete the protocol; `activity.IdempotencyKey` reads `Info` |
| Forward | `dur.06` | `workflow.Registry` is the `WorkflowHandler`; `dur.07` and `dur.08` register more commands |
| Forward | `dur.08` | cancellation and signals extend these task handlers and the worker's activity context |
| Forward | `dur.09` | the subprocess runner is an `ActivityFunc` that heartbeats checkpoints |

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| one worker build serves every run | worker versioning (build ids) | old runs stay on the build that started them while new runs use the new one | Temporal "Worker Versioning" |
| every activity through the server | local activities | short activities run in the workflow worker, recorded as markers, no round trip | Temporal "Local Activities" |
| one namespace, one service | Nexus | typed cross-namespace calls between teams' workflows | Temporal "Nexus" |
| one poller per kind | poller autoscaling and slot suppliers | pollers and slots sized from load and resource use | Temporal Go SDK `worker.Options` |
