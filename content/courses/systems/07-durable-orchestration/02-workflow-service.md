# The workflow service

You now have a durable log, from [the append-only event log](01-event-log.md), and the next section builds a durable queue on it in [the task queue](03-task-queue.md). But nothing yet knows what a workflow is. The CLI command `<system> wf start CorpusBuild --id corpus/tinystories/v1` must create exactly one run however many times it is retried; `wf describe` and `wf list` must answer from the same truth after a crash; and workers, which arrive in [the task protocol and workers](04-task-protocol-and-workers.md), need a server that hands out a run's history.

This section builds that server's core: the state of every run, rebuilt from its history, and the four read and start RPCs the CLI calls. The central idea is event sourcing. A run's history is the only truth, and everything else the server knows about a run is computed from it. Once you see that, crash recovery stops being a special case and becomes the same code that runs on every commit.

The section uses the log and the queue (read the queue section next if you want the details of idempotent enqueue first), gRPC from [Protocol Buffers and gRPC](../../programming-and-algorithms/01-language-primers/10-protocol-buffers-and-grpc.md) in *Programming and Algorithms*, and the check-then-act discussion of the [exactly-once event API](../13-case-study-event-api/) case study in *Case Studies*. The contract is `tl.durable.v1.WorkflowService` in [`proto/tl/durable/v1/durable.proto`](../../../../course/contracts/proto/tl/durable/v1/durable.proto), with generated Go in `contracts/go/gen/tl/durable/v1`; storage follows [`formats/wal.md`](../../../../course/contracts/formats/wal.md). The full solution is the event-sourced core in [`go/durable/server/server.go`](../../../../course/solutions/go/durable/server/server.go) (`Open`, `apply`, `effect`, `commit`, recovery) and the RPCs in [`go/durable/server/workflows.go`](../../../../course/solutions/go/durable/server/workflows.go) (`StartWorkflow`, `DescribeWorkflow`, `GetHistory`, `ListWorkflows`).

> **Learning objectives**
> - Describe a run's state as a fold of `apply` over its history, and explain why one function must serve both commit and recovery.
> - Trace a commit through its four steps and say which one is the commit point.
> - Explain how recovery repairs a crash between a history append and its effect without doubling tasks.
> - Implement idempotent `StartWorkflow` on `workflow_id` without a check-then-act race.
> - Page through histories and run lists with stable, opaque tokens.

## Event sourcing: history first, state derived

Start with notation. A run's history and its state are related by a fold:

| Symbol | Meaning |
|---|---|
| $h = (e_1, \dots, e_n)$ | a run's history: events with ids $1..n$, dense |
| $\sigma_0$ | the empty run state |
| $\mathrm{apply}(\sigma, e)$ | the state after one more event |
| $\sigma_n = \mathrm{apply}(\dots \mathrm{apply}(\sigma_0, e_1) \dots, e_n)$ | a run's state: a fold over its history |

The server never stores state separately from history. `apply` is the only code that changes a run, so the state after a restart (replaying the stored events) is exactly the state before it. Here is what `apply` tracks for each kind of event:

| Event | State change |
|---|---|
| `started` | type, task queue, input; status RUNNING |
| `wt_scheduled` / `wt_started` / `wt_completed` | the pending workflow task and its latest start |
| `act_scheduled` / `act_completed`, `act_failed`, `act_timed_out`, `act_canceled` | the pending activities, by scheduled event id |
| `timer_started` / `timer_fired`, `timer_canceled` | pending timers |
| `completed`, `failed`, `canceled`, `continued` | the run is closed |

### Triggers and workflow tasks

Some events are news the workflow must react to: `started`, an activity's result, a timer firing, a signal, a cancel request. Call these **triggers**. When a batch contains a trigger, `commit` adds a `WorkflowTaskScheduled` to the same batch, unless a workflow task is already pending and will see the news anyway. If that pending task is already running on a worker (started, not completed), it cannot see the news, so the run remembers it (`needsWT`) and the next `WorkflowTaskCompleted` batch schedules another one. Allowing one workflow task at a time per run is what makes the workflow's view of history consistent, a property that [the task protocol and workers](04-task-protocol-and-workers.md) and [deterministic replay](06-deterministic-replay.md) both depend on.

## Commit: one batch, one append, then effects

State is a fold over history, so changing a run means appending to its history. An RPC builds a **batch** of events for one run and calls `commit`, which proceeds in four steps:

1. decide whether the batch needs a `WorkflowTaskScheduled` (as described above);
2. append all events to stream `run/<run_id>` as one log record, with `expected` equal to the current history length; each event's data is a `tl.durable.v1.WalRecord{workflow_id, run_id, event}` (per `formats/wal.md`);
3. `apply` each event;
4. run each event's **effect**: `wt_scheduled` enqueues a workflow task, `act_scheduled` an activity task, `timer_started` arms a timer, and `timer_canceled` disarms one.

Step 2 is the commit point. If it fails, nothing changed and the RPC fails. If step 4 fails (say the queue's own append hit the quota), the RPC still succeeds: the events are durable, and recovery redoes the effect.

## Recovery: replay, then redo the effects of what is pending

That promise, "recovery redoes the effect", is what the next piece delivers. `Open` reads every `run/` stream and folds each through `apply`. It rebuilds two maps: `origin`, from workflow id to its first run, for idempotent start; and `current`, from workflow id to its newest run, for describe. Then, for every open run, it re-runs the effects of whatever is still pending: it enqueues the run's pending workflow task and its pending (not dead-lettered) activities, and arms its timers.

Enqueueing is idempotent in [the task queue](03-task-queue.md): a task id already in the queue is a no-op. So a task that survived in the queue is not doubled, and a task that was lost between the history append and the enqueue is restored.

There is one more repair. A run that ended with `WorkflowExecutionContinuedAsNew` names a successor run. If a crash came between that append and the successor's first append, recovery starts the successor with the recorded id, type, and input.

For all of this to work, queue task ids must be derivable from history rather than generated. They carry where they come from: `"<run_id>/wt/<scheduled event id>"` on queue `wf:<task_queue>`, and `"<run_id>/a/<scheduled event id>"` on `act:<task_queue>`.

## Idempotent start

With commit and recovery settled, the first RPC is starting a run. `StartWorkflow(workflow_id, type, input)` behaves as follows: while any run with that id exists (in any status), the same type and input return its newest run id with `started = false`, and a different type or input is `FAILED_PRECONDITION`. The comparison is against the **first** run's start, because a workflow that continued as new has a different input in its newest run.

The check and the commit must happen under one lock. Check, unlock, commit lets two concurrent starts both pass the check: the classic check-then-act race, which is exactly what the exactly-once case study solves with a unique constraint. A new run's id comes from `Options.NewRunID` (32 random hex digits by default; tests inject `run-1`, `run-2`, and so on).

## Reads: describe, list, history pages

The remaining three RPCs only read. `DescribeWorkflow` reports a run (the newest when `run_id` is empty): status, start and close times, history length and bytes, pending activities, the continue-as-new links, and the result or failure. `ListWorkflows` returns runs newest start first (ties broken by workflow id, then run id), filtered by status and type, `page_size` at a time (0 means 100), with an opaque `next_page_token`. `GetHistory` streams a run's events from the event its page token names: `PageToken(n)` is the 8 big-endian bytes of $n$, and the empty token means event 1. Workers fetch the rest of a long history this way.

## Errors

Each failure maps to one gRPC code:

| Situation | gRPC code |
|---|---|
| missing id, type, or queue; input over 2 MiB; a malformed page token | `INVALID_ARGUMENT` |
| same workflow id, different type or input | `FAILED_PRECONDITION` |
| unknown workflow or run | `NOT_FOUND` |
| the log's `ErrQuota` | `RESOURCE_EXHAUSTED` |

A refused start must leave nothing behind: the run is registered in memory only after its first batch is durable.

## Working it by hand

Run `wf start Echo --id demo-1 --input '"hi"'` at 2026-01-01T00:00:00Z on an empty server whose run ids are `run-1`, `run-2`, and so on:

1. `origin["demo-1"]` is absent, so this is a new run, `run-1`.
2. The batch is `[e1 = started{Echo, default, "hi"}]`. `started` is a trigger and no workflow task is pending, so `commit` adds `e2 = wt_scheduled{default, attempt 1}`.
3. One log record goes on stream `run/run-1` with two events (versions 1 and 2), each a `WalRecord`. This is log seq 1.
4. `apply` sets status RUNNING, start time `1767225600000`, and pending workflow task = event 2.
5. The effect of `e2` enqueues task `run-1/wt/2` on queue `wf:default`; the queue appends its own record (log seq 2).
6. The answer is `{run_id: "run-1", started: true}`.

Run the same command again. It finds `origin["demo-1"] = run-1`, compares type `Echo` and input `"hi"` with that run's start (equal), and answers `{run_id: "run-1", started: false}` without appending anything. `wf describe demo-1` now reports RUNNING, 2 events.

## Designing the API

The package surface is small, because most of the work is in unexported helpers:

```go
package server // import "tinyllm/durable/server"

const MaxPayloadBytes = 2 << 20
const ContinueAsNewEvents, ContinueAsNewBytes, MaxHistoryEvents, FirstPageBytes = 10000, 32 << 20, 20000, 1 << 20

type Clock = queue.Clock
type TimerKey struct{ RunID, TimerID string }
type Timers interface { Schedule(key TimerKey, at time.Time); Cancel(key TimerKey) } // dur.07

type Options struct {
	Log *log.Log; Queue *queue.Queue // required; the queue opened over the same log
	Clock Clock; Timers Timers; NewRunID func() string
	PollWait, WorkflowTaskTimeout time.Duration; DLQAfterAttempts int; Seed uint64
	Jitter func(d time.Duration) time.Duration // dur.05
}

func Open(ctx context.Context, o Options) (*Server, error)
func Kind(e *durablev1.HistoryEvent) string     // the oneof name: "started", "wt_scheduled", ...
func PageToken(next int64) []byte
func ParsePageToken(tok []byte) (int64, error)

// Server implements durablev1.WorkflowServiceServer (this section: StartWorkflow,
// DescribeWorkflow, GetHistory, ListWorkflows; dur.05 and dur.08 add the rest)
// and durablev1.TaskServiceServer (dur.04, dur.05).
```

`Timers` is implemented in [durable timers](07-durable-timers.md), and `Jitter` comes from [activities](05-activities.md). Register the server on a gRPC server with `durablev1.RegisterWorkflowServiceServer(g, srv)` and `durablev1.RegisterTaskServiceServer(g, srv)`. The unexported helpers that the later files of the package use are built here: `run`, `batch`, `(*Server).add`, `commit`, `startRun`, `lookup`, `enqueueWorkflowTask`, `enqueueActivity`, `parseTaskID`, and `logErr`.

## Building it step by step

As before, the design follows from a short chain of reasoning. Try to predict each step.

1. **The server must answer the same after a crash as before.** The only thing that survives a crash is the log. So the log must hold everything needed to rebuild state, and state must be a pure function of it: event sourcing.
2. **A pure function of history needs one code path.** If commit-time updates and recovery-time updates were separate functions, they would drift. So one `apply` is used by both, and nothing else assigns to run state.
3. **Effects cannot be in the same atomic step as the append.** The queue is a separate stream, and there is no transaction across both. So the order is: append (the commit point), then apply, then effects, and recovery re-runs effects for whatever is still pending. That is safe only because enqueue is idempotent by task id, which is why task ids are derived from the run id and the scheduled event id rather than generated.
4. **Two RPCs on one run must not interleave.** Appending with `expected` equal to the history length makes the log refuse a second writer that read the same length. Event ids are then exactly stream versions.
5. **Idempotent start is check-then-act, so the check and the act must share a lock.** The comparison is against the first run of the id (`origin`), because later runs after Continue-As-New carry different inputs; the answer names the newest run (`current`).
6. **A run is registered only after its first batch is durable.** Otherwise a refused append (quota) would leave a run in memory that recovery would not find: a ghost.

### The state and its fold

In [`server.go`](../../../../course/solutions/go/durable/server/server.go), a `run` holds the history and the fields `apply` derives from it:

```go
type run struct {
	id, workflowID string
	// ... type, task queue, input, trace context, identity
	history          []*durablev1.HistoryEvent
	bytes            int64
	status           durablev1.WorkflowStatus
	startMs, closeMs int64
	// ... result, failure, continuedFrom, continuedAs

	wtScheduled int64 // event id of the pending WorkflowTaskScheduled; 0 = none
	wtStarted   int64 // event id of its latest WorkflowTaskStarted; 0 = not started
	needsWT     bool  // a trigger event arrived while a workflow task was in flight

	activities      map[int64]*activity // pending, by ActivityTaskScheduled event id
	timers          map[string]*timer   // pending, by timer id
	// ... cancelRequested, signals (dur.08)
}
```

The fold itself is one switch over the event's attributes:

```go
func apply(r *run, e *durablev1.HistoryEvent) {
	r.history = append(r.history, e)
	r.bytes += int64(proto.Size(e)) // describe's history_bytes is derived, not counted elsewhere
	// News arriving while a workflow task is running on a worker: that task
	// cannot see it, so remember to schedule another when it completes.
	if trigger(e) && r.wtScheduled != 0 && r.wtStarted != 0 {
		r.needsWT = true
	}
	switch a := e.GetAttrs().(type) {
	case *durablev1.HistoryEvent_Started:
		r.wfType, r.taskQueue, r.input = a.Started.GetWorkflowType(), a.Started.GetTaskQueue(), a.Started.GetInput()
		// ...
		r.status, r.startMs = durablev1.WorkflowStatus_WORKFLOW_STATUS_RUNNING, e.GetTsUnixMs() // time from the event
		r.activities, r.timers, r.signals = map[int64]*activity{}, map[string]*timer{}, map[string]bool{}
	case *durablev1.HistoryEvent_WtScheduled:
		r.wtScheduled, r.wtStarted, r.needsWT = e.GetEventId(), 0, false
	case *durablev1.HistoryEvent_WtStarted:
		r.wtStarted = e.GetEventId()
	case *durablev1.HistoryEvent_WtCompleted:
		r.wtScheduled, r.wtStarted = 0, 0
	case *durablev1.HistoryEvent_ActScheduled:
		r.activities[e.GetEventId()] = &activity{scheduledID: e.GetEventId(), attrs: a.ActScheduled}
	case *durablev1.HistoryEvent_ActCompleted:
		delete(r.activities, a.ActCompleted.GetScheduledEventId())
	// ... act_failed, act_timed_out, act_canceled delete; dead_lettered marks
	case *durablev1.HistoryEvent_TimerStarted:
		r.timers[a.TimerStarted.GetTimerId()] = &timer{id: a.TimerStarted.GetTimerId(), startedID: e.GetEventId(), fireAtMs: a.TimerStarted.GetFireAtUnixMs()}
	case *durablev1.HistoryEvent_TimerFired:
		delete(r.timers, a.TimerFired.GetTimerId())
	// ... timer_canceled, signal, cancel_requested
	case *durablev1.HistoryEvent_Completed:
		r.close(e, durablev1.WorkflowStatus_WORKFLOW_STATUS_COMPLETED)
		r.result = a.Completed.GetResult()
	// ... failed, canceled, continued (records continuedAs)
	}
}
```

Every value comes from the event itself, including times (`TsUnixMs`, stamped once by `add`), never from the clock at apply time. That is what makes a replay after restart produce the same state.

### Effects

Effects are the only bridge to the outside world, and each is keyed so that redoing it is harmless:

```go
func (s *Server) effect(ctx context.Context, r *run, e *durablev1.HistoryEvent) error {
	switch a := e.GetAttrs().(type) {
	case *durablev1.HistoryEvent_WtScheduled:
		return s.enqueueWorkflowTask(ctx, r, e.GetEventId())
	case *durablev1.HistoryEvent_ActScheduled:
		if act := r.activities[e.GetEventId()]; act != nil {
			return s.enqueueActivity(ctx, r, act)
		}
	case *durablev1.HistoryEvent_TimerStarted:
		// ... s.o.Timers.Schedule(TimerKey{r.id, timer id}, fire time)
	case *durablev1.HistoryEvent_TimerCanceled:
		// ... s.o.Timers.Cancel(...)
	}
	return nil
}

func workflowQueue(taskQueue string) string { return "wf:" + taskQueue } // workers poll these names
func activityQueue(taskQueue string) string { return "act:" + taskQueue }

// "<run_id>/wt/<scheduled event id>": deterministic, so a re-enqueue is a no-op.
func taskID(runID, kind string, scheduled int64) string {
	return fmt.Sprintf("%s/%s/%d", runID, kind, scheduled)
}
```

### Building and committing a batch

`add` stamps the next event id from the history length plus the batch length, so ids are dense without a separate counter. `commit` then implements the four steps:

```go
func (s *Server) add(b *batch, e *durablev1.HistoryEvent) *durablev1.HistoryEvent {
	e.EventId = int64(len(b.r.history) + len(b.events) + 1)
	e.TsUnixMs = s.o.Clock.Now().UnixMilli()
	b.events = append(b.events, e)
	return e
}

func (s *Server) commit(ctx context.Context, b *batch) error {
	r := b.r
	var trig, completes, closes, schedules bool
	for _, e := range b.events {
		trig = trig || trigger(e)
		closes = closes || closing(e)
		// ... completes = a wt_completed is in the batch, schedules = a wt_scheduled is
	}
	// Will a workflow task be pending after this batch? Then it will see the news.
	pendingAfter := (r.wtScheduled != 0 && !completes) || schedules
	open := !closes && (r.status == durablev1.WorkflowStatus_WORKFLOW_STATUS_RUNNING || Kind(b.events[0]) == "started")
	if open && !pendingAfter && (trig || (completes && r.needsWT)) {
		s.add(b, &durablev1.HistoryEvent{Attrs: &durablev1.HistoryEvent_WtScheduled{
			WtScheduled: &durablev1.WorkflowTaskScheduled{TaskQueue: r.taskQueueOr(b), Attempt: 1},
		}})
	}
	// ... refuse past MaxHistoryEvents (FAILED_PRECONDITION: continue as new)
	evs := make([]dlog.Event, len(b.events))
	for i, e := range b.events {
		// formats/wal.md: each event's data is a WalRecord naming both ids, so
		// the log alone says which workflow a run belongs to.
		data, err := proto.Marshal(&durablev1.WalRecord{WorkflowId: r.workflowID, RunId: r.id, Record: &durablev1.WalRecord_Event{Event: e}})
		// ...
		evs[i] = dlog.Event{Type: Kind(e), Data: data}
	}
	// The commit point. expected = history length: a concurrent commit fails here.
	if _, err := s.o.Log.Append(ctx, "run/"+r.id, int64(len(r.history)), evs...); err != nil {
		return logErr(err) // ErrQuota -> RESOURCE_EXHAUSTED, otherwise UNAVAILABLE
	}
	for _, e := range b.events {
		apply(r, e)
	}
	for _, e := range b.events {
		_ = s.effect(ctx, r, e) // durable already: a failed effect is redone by recovery
	}
	for _, f := range b.after {
		_ = f(ctx)
	}
	return nil
}
```

Check this against the hand example. The batch `[started]` has `trig` true, no pending task, and `open` true because its first event is `started`, so `commit` adds `wt_scheduled` as event 2, and the whole batch is one log record.

### Recovery in `Open`

Recovery replays, rebuilds the indexes, redoes pending effects, and repairs a lost Continue-As-New successor:

```go
for _, stream := range o.Log.Streams("run/") {
	evs, err := o.Log.Read(ctx, stream, 1, 0)
	// ...
	r := &run{id: strings.TrimPrefix(stream, "run/")}
	for _, e := range evs {
		var rec durablev1.WalRecord
		if err := proto.Unmarshal(e.Data, &rec); err != nil { /* refuse to start */ }
		r.workflowID = rec.GetWorkflowId()
		apply(r, rec.GetEvent()) // the same fold as at commit time
	}
	s.runs[r.id] = r
}
for _, r := range s.runs {
	if r.continuedFrom == "" {
		s.origin[r.workflowID] = r.id // the first run of the chain: idempotent start compares here
	}
	if r.continuedAs == "" { // the newest run of the chain: describe defaults here
		if cur, ok := s.current[r.workflowID]; !ok || s.runs[cur].startMs <= r.startMs {
			s.current[r.workflowID] = r.id
		}
	}
}
for _, r := range s.runs {
	if err := s.resume(ctx, r); err != nil { // re-enqueue pending tasks, re-arm timers
		return nil, err
	}
}
for _, r := range s.runs {
	if c := r.history[len(r.history)-1].GetContinued(); c != nil && s.runs[c.GetNewRunId()] == nil {
		// A crash after "continued" but before the successor's first append.
		if _, err := s.startRun(ctx, r.workflowID, c.GetNewRunId(), continuedStart(r, c)); err != nil {
			return nil, err
		}
	}
}
```

`resume` acts only on RUNNING runs. It re-enqueues the pending workflow task, every pending activity that is not dead-lettered (in scheduled-id order, so recovery is deterministic), and every pending timer at its recorded `fire_at`.

### Idempotent start

The start RPC lives in [`workflows.go`](../../../../course/solutions/go/durable/server/workflows.go). Notice where the lock is taken and how long it is held:

```go
func (s *Server) StartWorkflow(ctx context.Context, req *durablev1.StartWorkflowRequest) (*durablev1.StartWorkflowResponse, error) {
	switch {
	// ... workflow_id, workflow_type, task_queue required: INVALID_ARGUMENT
	case len(req.GetInput()) > MaxPayloadBytes: // ">": exactly 2 MiB is accepted
		return nil, status.Errorf(codes.InvalidArgument, "input is %d bytes; the limit is %d (pass large data by path)", len(req.GetInput()), MaxPayloadBytes)
	}
	s.mu.Lock()
	defer s.mu.Unlock() // held through the commit: check and act are one critical section
	if first, ok := s.origin[req.GetWorkflowId()]; ok {
		o := s.runs[first]
		if o.wfType != req.GetWorkflowType() || !bytes.Equal(o.input, req.GetInput()) { // full byte comparison
			return nil, status.Errorf(codes.FailedPrecondition,
				"workflow %q already exists with a different type or input", req.GetWorkflowId())
		}
		return &durablev1.StartWorkflowResponse{RunId: s.current[req.GetWorkflowId()], Started: false}, nil
	}
	r, err := s.startRun(ctx, req.GetWorkflowId(), s.o.NewRunID(), &durablev1.WorkflowExecutionStarted{ /* ... */ })
	if err != nil {
		return nil, err // nothing registered: startRun registers only after commit
	}
	s.origin[r.workflowID] = r.id
	return &durablev1.StartWorkflowResponse{RunId: r.id, Started: true}, nil
}

func (s *Server) startRun(ctx context.Context, workflowID, runID string, started *durablev1.WorkflowExecutionStarted) (*run, error) {
	r := &run{id: runID, workflowID: workflowID}
	b := &batch{r: r}
	s.add(b, &durablev1.HistoryEvent{Attrs: &durablev1.HistoryEvent_Started{Started: started}})
	if err := s.commit(ctx, b); err != nil {
		return nil, err
	}
	s.runs[r.id] = r // registered only once durable
	s.current[workflowID] = r.id
	return r, nil
}
```

### Reads

`GetHistory` copies the slice under the lock and streams after releasing it, so a slow client never blocks commits. Its token is 8 big-endian bytes of the next event id; `ParsePageToken` maps empty to 1 and refuses any length other than 8 or a value below 1. `ListWorkflows` filters, sorts newest start first with workflow id then run id as tie breakers, and uses the decimal offset as its opaque token. The tie breakers matter: they make the order total, so pages never repeat or skip a run.

```go
sort.Slice(rs, func(i, j int) bool {
	if rs[i].startMs != rs[j].startMs {
		return rs[i].startMs > rs[j].startMs // newest first
	}
	if rs[i].workflowID != rs[j].workflowID {
		return rs[i].workflowID < rs[j].workflowID
	}
	return rs[i].id < rs[j].id
})
// ... off >= len(rs): empty page; otherwise rs[off:end], NextPageToken only when more remain
```

The full solution is in [`server.go`](../../../../course/solutions/go/durable/server/server.go) and [`workflows.go`](../../../../course/solutions/go/durable/server/workflows.go).

### Why this design

- **Cost.** A commit is one log append (one fsync) plus O(batch) apply and effects. Recovery is O(total events). `DescribeWorkflow` is O(pending activities); `ListWorkflows` is O(runs log runs) for the sort, which is acceptable for a server of this size and is the reason production uses a visibility store.
- **One server mutex.** Every RPC that changes a run holds `s.mu` across its commit. That serialises writes across runs, which caps throughput at the fsync rate, but it makes check-then-act races impossible by construction. Sharding by workflow id is the production answer.
- **State derived, never stored.** The alternative, a separate state table updated alongside history, needs a transaction across both or it can disagree after a crash. A fold cannot disagree with its input.
- **Effects after commit, redone on recovery.** The alternative order (enqueue first) can leave a task for an event that was never written. Committing first means the worst case is a missing effect, which recovery repairs because effects are idempotent.

## What a correct server must handle

The solution's tests cover the following cases, each tied to something downstream that relies on it:

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| hand example start | unit | the worked example: events 1 and 2, describe, the second start, one log record for the batch | the CLI's `wf start` and `wf describe` |
| 64 concurrent identical starts | fault, property | 64 parallel starts: one `started`, one run id, one queued workflow task | retried CLIs, two operators |
| start conflict | unit, boundary | different input or type is `FAILED_PRECONDITION`; nothing appended | an id names one piece of work |
| start validation | boundary | required fields; 2 MiB input accepted, one byte more refused | payloads travel by path |
| start enqueues one workflow task | unit | task `run-1/wt/2` on `wf:<task_queue>` | where workers poll |
| history is WAL records | conformance | each event is a `WalRecord` with both ids; stream version = event id | `formats/wal.md`, Raft replication |
| recorded histories recover | conformance, golden | recorded runs written into an empty log: identical `GetHistory` (also from a middle page), describe and list derived from events alone | recovery is a fold |
| recovery rebuilds everything | fault | list and history identical after a restart; start still idempotent; no task doubled | every restart |
| recovery enqueues a lost task | fault | a history whose enqueue was lost gets its workflow task back | crash between append and effect |
| recovery re-arms timers | fault | recovery arms exactly the pending timers at their `fire_at` | durable timers |
| recovery starts a missing continued run | fault | the successor of a continued run is started if the crash lost it | Continue-As-New |
| list newest first, paged | unit | newest first, pages of 2, type and status filters | `wf list` |
| not found and bad tokens | boundary | unknown workflow or run is `NOT_FOUND`; bad tokens `INVALID_ARGUMENT` | CLI error messages |
| page token round trip | unit | `ParsePageToken(PageToken(n)) = n`; empty is 1; 0 is refused | workers pass tokens back unchanged |
| quota is RESOURCE_EXHAUSTED | fault | `RESOURCE_EXHAUSTED` at the WAL quota, no half-registered run, raising the quota resumes | the [disk-full drill](../12-incident-response-and-chaos/12-eventlog-disk-full.md) |

## What can go wrong

The bugs that matter here all break the promise that a workflow id names one piece of work, or that the server answers the same after a crash.

Idempotency is easy to key on the wrong thing, or to compare only part of the input. Then a retried start creates a second run, or a different request silently gets the old run. The solution keys on `workflow_id` via `origin` and compares type and the full input with `bytes.Equal`.

> **Warning:** Check, unlock, then commit is a race. Two concurrent starts both create runs, and race on the maps. `StartWorkflow` holds `s.mu` from the check through `startRun`.

Recovery has two ways to fall short. If it does not rebuild `origin` and `current`, a repeated start after a restart creates a duplicate run, and describe finds nothing; `Open` rebuilds `origin` from runs with no `continuedFrom`, and `current` from the newest open run. If it replays state but not effects, a run whose task was lost, a pending timer, or a continued run's successor waits forever; `resume` re-runs pending effects, and `Open` starts a missing successor with the recorded id.

Queue naming is a quieter failure: put workflow tasks on the bare task queue name and workers polling `wf:<queue>` never see them. `workflowQueue` prefixes `wf:`.

State that is not derived from the events (start time, attempts, result, bytes), or a `WalRecord` without the workflow id, makes describe disagree with history and leaves recovery unable to tell which workflow a run belongs to. `apply` reads every field from the event, and `commit` writes a `WalRecord` with both ids.

Boundaries need care too. An off-by-one payload limit refuses an input of exactly 2 MiB; the solution refuses only `len(input) > MaxPayloadBytes`. Sloppy list order, pages, and tokens make `wf list` show the oldest first, repeat a run across pages, or panic on a token for event 0, and an unknown workflow can panic too. The fixes are a total sort order, an offset token, `ParsePageToken` refusing values below 1, and `lookup` returning `NOT_FOUND`.

Finally, mapping the quota to `UNAVAILABLE` makes clients retry a full disk forever, and registering the run before its batch is durable makes a refused start leave a ghost run. `logErr` maps `ErrQuota` to `RESOURCE_EXHAUSTED`, and `startRun` registers after `commit`.

## Where this leads

This server stands on two earlier pieces. Each run is stream `run/<run_id>` in [the event log](01-event-log.md), and `commit` appends with `expected` equal to the history length. Effects enqueue tasks on [the task queue](03-task-queue.md), and recovery relies on its idempotent enqueue.

Going forward, [the task protocol and workers](04-task-protocol-and-workers.md) builds `PollWorkflowTask` and `CompleteWorkflowTask` as batches that `commit`, with command handlers registered in `tasks.go`. [Activities](05-activities.md), [deterministic replay](06-deterministic-replay.md), [durable timers](07-durable-timers.md), and [signals, cancellation, and sagas](08-signals-cancellation-and-sagas.md) each add RPCs and command handlers to the same `Server`, and none of them changes `apply`.

Production systems extend each piece:

- **Workflow id reuse policies.** Where this server has idempotent start on workflow id, Temporal adds id reuse and conflict policies: allow a new run after a closed one, terminate the old one, or fail; and Update-with-Start. See the Temporal docs, "Workflow Id Reuse Policy".
- **History shards.** Instead of one server with one lock, runs are partitioned over many history service hosts by workflow id. See `temporalio/temporal`: `service/history`.
- **A visibility store.** Instead of `ListWorkflows` over memory, Elasticsearch or SQL supports search attributes and queries over millions of runs. See the Temporal "Visibility" docs.
- **Namespaces and mTLS.** Instead of plaintext gRPC, multi-tenant isolation and authenticated workers. See Temporal "Namespaces" and "Security".

### Further reading

- Martin Fowler, *Event Sourcing*.
- Martin Kleppmann, *Designing Data-Intensive Applications*, chapter 11 (derived data).
- The Temporal docs on workflow id reuse and conflict policies.

## Key terms

- **Event sourcing**: storing only the history of changes and deriving all state from it.
- **Fold**: computing a value by applying a function to each element of a sequence in turn; here, run state is a fold of `apply` over the history.
- **Trigger**: an event the workflow must react to (`started`, an activity result, a timer firing, a signal, a cancel request), which causes a workflow task to be scheduled.
- **Effect**: an action outside the history derived from an event, such as enqueueing a task or arming a timer; keyed so that redoing it is harmless.
- **Commit point**: the log append after which a batch is durable and the RPC is considered done.
- **Idempotent start**: `StartWorkflow` returning the existing run for a repeated identical request instead of creating another.
- **Check-then-act race**: two callers both passing a check before either acts on it, which a shared lock across check and act prevents.

## Summary

- A run's history is the only truth. Its state (status, the pending workflow task, pending activities, timers) is a fold of `apply` over the events, the same function at commit time and at recovery.
- Side effects (queue tasks, timers) are derived from events by `effect`; a crash between an append and its effect is repaired by recovery, because the queue ignores a task id it already holds.
- `StartWorkflow` is idempotent on `workflow_id`: 64 concurrent identical starts give one run, and a different type or input is `FAILED_PRECONDITION`.
- Every RPC appends one batch with `expected` equal to the history's length, so two RPCs can never interleave events in one run.

## Review questions

1. A crash happens after the history append of `[started, wt_scheduled]` but before the enqueue. What does the next `Open` do, and why is the task not doubled if the enqueue had actually succeeded?

   <details><summary>Answer</summary>

   Replay gives `wtScheduled = 2`, so `resume` enqueues `run-1/wt/2` on `wf:default`. If that id is already in the queue, the queue's enqueue is a no-op for a known task id, so either way exactly one task exists.

   </details>

2. Why does idempotent start compare against `origin` (the first run) but answer with `current` (the newest)?

   <details><summary>Answer</summary>

   After Continue-As-New the newest run has a new input chosen by the workflow, so comparing a retried client start against it would wrongly report a conflict. The client's request matches the first run. But the caller wants to observe the live run, which is the newest.

   </details>

3. What goes wrong if `StartWorkflow` releases `s.mu` between the `origin` check and `startRun`?

   <details><summary>Answer</summary>

   Two concurrent starts can both see no `origin`, both create runs with different run ids, and both report `started: true`. The workflow id no longer names one piece of work, and the maps race.

   </details>

4. Why does `commit` use `expected = len(r.history)` on the append, given that it already holds `s.mu`?

   <details><summary>Answer</summary>

   It turns the invariant "event id equals stream version" into a check the log enforces, rather than a property of the locking. Any bug or future path that commits without the lock, or with a stale run, fails loudly with a version conflict instead of interleaving events.

   </details>

5. Why does a batch containing a trigger not always get a new `wt_scheduled`?

   <details><summary>Answer</summary>

   Only one workflow task per run may be pending. If one is already scheduled and not completed in this batch, it will read the full history when it runs and see the news. If it is already started on a worker, it cannot see the news, so `apply` sets `needsWT` and the batch that completes it schedules the next one.

   </details>
