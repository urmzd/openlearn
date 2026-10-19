# Workflow service + idempotent start, describe, list, paged history

## Overview

| | |
|---|---|
| **Module** | `dur.02` · build · Go · Pass 8 · 5 to 7 h |
| **Solution** | [`go/durable/server/server.go`](../../../../course/solutions/go/durable/server/server.go): the event-sourced core (`Open`, `apply`, `effect`, `commit`, recovery) and [`go/durable/server/workflows.go`](../../../../course/solutions/go/durable/server/workflows.go): `StartWorkflow`, `DescribeWorkflow`, `GetHistory`, `ListWorkflows` |
| **Contract** | `tl.durable.v1.WorkflowService` in [`proto/tl/durable/v1/durable.proto`](../../../../course/contracts/proto/tl/durable/v1/durable.proto) (generated Go in `contracts/go/gen/tl/durable/v1`); storage is [`formats/wal.md`](../../../../course/contracts/formats/wal.md); the Go API of the package is section 4 |
| **Needs** | `dur.01` the log, `dur.03` the task queue; reading: [`lang.10` gRPC](../../software-craftsmanship/12-language-and-tool-primers/10-protocol-buffers-and-grpc.md), [case study 03, exactly-once event API](../../../case-studies/03-exactly-once-event-api/) |
| **Used by** | `dur.04` task RPCs and `dur.08` signals commit through this core |
| **Optional depth** | Fowler, *Event Sourcing*; Kleppmann, *DDIA* ch. 11 (derived data); the Temporal docs on workflow id reuse and conflict policies |

## Key Takeaways

- A run's history is the only truth. Its state (status, the pending workflow task, pending activities, timers) is a fold of `apply` over the events, the same function at commit time and at recovery.
- Side effects (queue tasks, timers) are derived from events by `effect`; a crash between an append and its effect is repaired by recovery, because the queue ignores a task id it already holds.
- `StartWorkflow` is idempotent on `workflow_id`: 64 concurrent identical starts give one run, and a different type or input is `FAILED_PRECONDITION`.
- Every rpc appends one batch with `expected` = the history's length, so two rpcs can never interleave events in one run.

## How to read this chapter

Sections 1 to 3 build the event-sourced server from first principles and work one start by hand. Section 4 fixes the interface. Section 5 walks through the solution in reading order (`apply`, `effect`, `commit`, recovery, then the start and read RPCs) and explains why each choice is the right one. Try to predict each step of the solution before reading it.

---

## 1. Why now

You have a durable log (`dur.01`) and a durable queue (`dur.03`), but nothing that knows what a workflow is. The CLI command `<system> wf start CorpusBuild --id corpus/tinystories/v1` must create exactly one run however many times it is retried, `wf describe` and `wf list` must answer from the same truth after a crash, and workers (`dur.04`) need a server that hands out the run's history. This module is that server's core: the state of every run, rebuilt from its history, and the four read and start rpcs the CLI calls.

## 2. Principles

### 2.1 Event sourcing: history first, state derived

| Symbol | Meaning |
|---|---|
| $h = (e_1, \dots, e_n)$ | a run's history: events with ids $1..n$, dense |
| $\sigma_0$ | the empty run state |
| $\mathrm{apply}(\sigma, e)$ | the state after one more event |
| $\sigma_n = \mathrm{apply}(\dots \mathrm{apply}(\sigma_0, e_1) \dots, e_n)$ | a run's state: a fold over its history |

The server never stores state separately from history. `apply` is the only code that changes a run, so the state after a restart (replaying the stored events) is exactly the state before it. What `apply` tracks:

| Event | State change |
|---|---|
| `started` | type, task queue, input; status RUNNING |
| `wt_scheduled` / `wt_started` / `wt_completed` | the pending workflow task and its latest start |
| `act_scheduled` / `act_completed`, `act_failed`, `act_timed_out`, `act_canceled` | the pending activities, by scheduled event id |
| `timer_started` / `timer_fired`, `timer_canceled` | pending timers |
| `completed`, `failed`, `canceled`, `continued` | the run is closed |

**Triggers and workflow tasks.** Some events are news the workflow must react to: `started`, an activity's result, a timer firing, a signal, a cancel request. When a batch contains a trigger, `commit` adds a `WorkflowTaskScheduled` to the same batch, unless a workflow task is already pending and will see the news anyway. If that pending task is already running on a worker (started, not completed), the run remembers it (`needsWT`) and the next `WorkflowTaskCompleted` batch schedules another one. One workflow task at a time per run is what makes the workflow's view of history consistent (`dur.04`, `dur.06`).

### 2.2 Commit: one batch, one append, then effects

An rpc builds a **batch** of events for one run and calls `commit`:

1. decide whether the batch needs a `WorkflowTaskScheduled` (section 2.1);
2. append all events to stream `run/<run_id>` as one log record, with `expected` = the current history length; each event's data is a `tl.durable.v1.WalRecord{workflow_id, run_id, event}` (formats/wal.md);
3. `apply` each event;
4. run each event's **effect**: `wt_scheduled` enqueues a workflow task, `act_scheduled` an activity task, `timer_started` arms a timer, `timer_canceled` disarms one.

Step 2 is the commit point. If it fails, nothing changed and the rpc fails. If step 4 fails (the queue's own append hit the quota), the rpc still succeeds: the events are durable, and recovery redoes the effect.

### 2.3 Recovery: replay, then redo the effects of what is pending

`Open` reads every `run/` stream, folds each through `apply`, rebuilds two maps (`origin`: workflow id to its first run, for idempotent start; `current`: workflow id to its newest run, for describe), then for every open run re-runs the effects of whatever is still pending: enqueue its pending workflow task, its pending (not dead-lettered) activities, arm its timers. Enqueueing is idempotent in `dur.03` (a task id already in the queue is a no-op), so a task that survived in the queue is not doubled and a task that was lost between the history append and the enqueue is restored. One more repair: a run that ended with `WorkflowExecutionContinuedAsNew` names a successor run; if a crash came between that append and the successor's first append, recovery starts the successor with the recorded id, type, and input.

Queue task ids carry where they come from: `"<run_id>/wt/<scheduled event id>"` on queue `wf:<task_queue>` and `"<run_id>/a/<scheduled event id>"` on `act:<task_queue>`.

### 2.4 Idempotent start

`StartWorkflow(workflow_id, type, input)`: while any run with that id exists (any status), the same type and input return its newest run id with `started = false`; a different type or input is `FAILED_PRECONDITION`. The comparison is against the **first** run's start (a workflow that continued as new has a different input in its newest run). The check and the commit must happen under one lock: check, unlock, commit lets two concurrent starts both pass the check (the classic check-then-act race, exactly what case study 03 solves with a unique constraint). A new run's id comes from `Options.NewRunID` (32 random hex digits by default; tests inject `run-1`, `run-2`, ...).

### 2.5 Reads: describe, list, history pages

`DescribeWorkflow` reports a run (the newest when `run_id` is empty): status, start and close times, history length and bytes, pending activities, the continue-as-new links, the result or failure. `ListWorkflows` returns runs newest start first (ties by workflow id, then run id), filtered by status and type, `page_size` at a time (0 = 100) with an opaque `next_page_token`. `GetHistory` streams a run's events from the event its page token names: `PageToken(n)` is the 8 big-endian bytes of $n$, the empty token is event 1. Workers fetch the rest of a long history this way (`dur.04`).

### 2.6 Errors

| Situation | gRPC code |
|---|---|
| missing id, type, or queue; input over 2 MiB; a malformed page token | `INVALID_ARGUMENT` |
| same workflow id, different type or input | `FAILED_PRECONDITION` |
| unknown workflow or run | `NOT_FOUND` |
| the log's `ErrQuota` | `RESOURCE_EXHAUSTED` |

A refused start must leave nothing behind: the run is registered in memory only after its first batch is durable.

## 3. Worked example by hand

`wf start Echo --id demo-1 --input '"hi"'` at 2026-01-01T00:00:00Z on an empty server whose run ids are `run-1`, `run-2`, ...:

1. `origin["demo-1"]` is absent: a new run `run-1`.
2. The batch is `[e1 = started{Echo, default, "hi"}]`. `started` is a trigger and no workflow task is pending, so `commit` adds `e2 = wt_scheduled{default, attempt 1}`.
3. One log record on stream `run/run-1` with two events (versions 1 and 2), each a `WalRecord`. Log seq 1.
4. `apply`: status RUNNING, start time `1767225600000`, pending workflow task = event 2.
5. Effect of `e2`: enqueue task `run-1/wt/2` on queue `wf:default`; the queue appends its own record (log seq 2).
6. Answer: `{run_id: "run-1", started: true}`.

The same command again finds `origin["demo-1"] = run-1`, compares type `Echo` and input `"hi"` with that run's start (equal), and answers `{run_id: "run-1", started: false}` without appending anything. `wf describe demo-1` now reports RUNNING, 2 events.

## 4. The interface

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

// Server implements durablev1.WorkflowServiceServer (this module: StartWorkflow,
// DescribeWorkflow, GetHistory, ListWorkflows; dur.05 and dur.08 add the rest)
// and durablev1.TaskServiceServer (dur.04, dur.05).
```

Register it on a gRPC server with `durablev1.RegisterWorkflowServiceServer(g, srv)` and `durablev1.RegisterTaskServiceServer(g, srv)`. The unexported helpers the later files of the package use belong to this module: `run`, `batch`, `(*Server).add`, `commit`, `startRun`, `lookup`, `enqueueWorkflowTask`, `enqueueActivity`, `parseTaskID`, `logErr`.

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| hand example start | unit | section 3: events 1 and 2, describe, the second start, one log record for the batch | the CLI's `wf start` and `wf describe` |
| 64 concurrent identical starts | fault, property | 64 parallel starts: one `started`, one run id, one queued workflow task | retried CLIs, two operators |
| start conflict | unit, boundary | different input or type is `FAILED_PRECONDITION`; nothing appended | an id names one piece of work |
| start validation | boundary | required fields; 2 MiB input accepted, one byte more refused | payloads travel by path |
| start enqueues one workflow task | unit | task `run-1/wt/2` on `wf:<task_queue>` | where `dur.04` polls |
| history is WAL records | conformance | each event is a `WalRecord` with both ids; stream version = event id | formats/wal.md, `dur.10` |
| recorded histories recover | conformance, golden | recorded runs written into an empty log: identical `GetHistory` (also from a middle page), describe and list derived from events alone | recovery is a fold |
| recovery rebuilds everything | fault | list and history identical after a restart; start still idempotent; no task doubled | every restart |
| recovery enqueues a lost task | fault | a history whose enqueue was lost gets its workflow task back | crash between append and effect |
| recovery re-arms timers | fault | recovery arms exactly the pending timers at their `fire_at` | durable timers (`dur.07`) |
| recovery starts a missing continued run | fault | the successor of a continued run is started if the crash lost it | ContinueAsNew (`dur.06`) |
| list newest first, paged | unit | newest first, pages of 2, type and status filters | `wf list` |
| not found and bad tokens | boundary | unknown workflow or run is `NOT_FOUND`; bad tokens `INVALID_ARGUMENT` | CLI error messages |
| page token round trip | unit | `ParsePageToken(PageToken(n)) = n`; empty is 1; 0 is refused | workers pass tokens back unchanged |
| quota is RESOURCE_EXHAUSTED | fault | `RESOURCE_EXHAUSTED` at the WAL quota, no half-registered run, raising the quota resumes | drill `ops.11` |

## 5. The solution, step by step

### From first principles

1. **The server must answer the same after a crash as before.** The only thing that survives a crash is the log. So the log must hold everything needed to rebuild state, and state must be a pure function of it: event sourcing.
2. **A pure function of history needs one code path.** If commit-time updates and recovery-time updates were separate functions, they would drift. So one `apply` is used by both, and nothing else assigns to run state.
3. **Effects cannot be in the same atomic step as the append.** The queue is a separate stream; there is no transaction across both. So the order is: append (the commit point), then apply, then effects, and recovery re-runs effects for whatever is still pending. That is safe only because enqueue is idempotent by task id (`dur.03`), which is why task ids are derived from the run id and the scheduled event id rather than generated.
4. **Two RPCs on one run must not interleave.** Appending with `expected` = the history length makes the log refuse a second writer that read the same length. Event ids are then exactly stream versions.
5. **Idempotent start is check-then-act, so the check and the act must share a lock.** The comparison is against the first run of the id (`origin`), because later runs after Continue-As-New carry different inputs; the answer names the newest run (`current`).
6. **A run is registered only after its first batch is durable.** Otherwise a refused append (quota) would leave a run in memory that recovery would not find: a ghost.

### The code, annotated

**The state and its fold** ([`server.go`](../../../../course/solutions/go/durable/server/server.go)). A `run` holds the history and the fields `apply` derives from it.

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

**Effects** are the only bridge to the outside, and each is keyed so that redoing it is harmless.

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

**Building and committing a batch.** `add` stamps the next event id from history plus batch length, so ids are dense without a counter.

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

In the worked example, the batch `[started]` has `trig` true, no pending task, and `open` true because its first event is `started`, so `commit` adds `wt_scheduled` as event 2 and the whole batch is one log record.

**Recovery in `Open`.** Replay, rebuild the indexes, redo pending effects, repair a lost Continue-As-New successor.

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

`resume` only acts on RUNNING runs: it re-enqueues the pending workflow task, every pending activity that is not dead-lettered (in scheduled-id order, so recovery is deterministic), and every pending timer at its recorded `fire_at`.

**Idempotent start** ([`workflows.go`](../../../../course/solutions/go/durable/server/workflows.go)).

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

**Reads.** `GetHistory` copies the slice under the lock and streams after releasing it, so a slow client never blocks commits. Its token is 8 big-endian bytes of the next event id; `ParsePageToken` maps empty to 1 and refuses any length other than 8 or a value below 1. `ListWorkflows` filters, sorts newest start first with workflow id then run id as tie breakers (a total order, so pages never repeat or skip a run), and uses the decimal offset as its opaque token.

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

Full solution: [`server.go`](../../../../course/solutions/go/durable/server/server.go), [`workflows.go`](../../../../course/solutions/go/durable/server/workflows.go).

### Why this solution

- **Cost.** A commit is one log append (one fsync) plus O(batch) apply and effects. Recovery is O(total events). `DescribeWorkflow` is O(pending activities); `ListWorkflows` is O(runs log runs) for the sort, acceptable for a course-sized server and the reason production uses a visibility store.
- **One server mutex.** Every RPC that changes a run holds `s.mu` across its commit. That serialises writes across runs, which caps throughput at the fsync rate but makes check-then-act races impossible by construction. Sharding by workflow id is the production answer.
- **State derived, never stored.** The alternative, a separate state table updated alongside history, needs a transaction across both or it can disagree after a crash. A fold cannot disagree with its input.
- **Effects after commit, redone on recovery.** The alternative order (enqueue first) can leave a task for an event that was never written. Committing first means the worst case is a missing effect, which recovery repairs because effects are idempotent.

### Explain it back

1. A crash happens after the history append of `[started, wt_scheduled]` but before the enqueue. What does the next `Open` do, and why is the task not doubled if the enqueue had actually succeeded?

   <details><summary>Answer</summary>

   Replay gives `wtScheduled = 2`, so `resume` enqueues `run-1/wt/2` on `wf:default`. If that id is already in the queue, `dur.03`'s enqueue is a no-op for a known task id, so either way exactly one task exists.

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

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. idempotency keyed by the wrong thing, or a partial comparison of the input | a retried start creates a second run, or a different request silently gets the old run | keys on `workflow_id` via `origin` and compares type and the full input with `bytes.Equal` |
| 2. check, unlock, then commit | two concurrent starts both create runs (and race on the maps) | `StartWorkflow` holds `s.mu` from the check through `startRun` |
| 3. recovery that does not rebuild `origin` and `current` | after a restart, a repeated start creates a duplicate run; describe finds nothing | `Open` rebuilds `origin` from runs with no `continuedFrom` and `current` from the newest open run |
| 4. recovery that replays state but not effects | a run whose task was lost, a pending timer, or a continued run's successor waits forever | `resume` re-runs pending effects; `Open` starts a missing successor with the recorded id |
| 5. putting workflow tasks on the bare task queue name | workers polling `wf:<queue>` never see them | `workflowQueue` prefixes `wf:` |
| 6. state not derived from the events (start time, attempts, result, bytes) or a `WalRecord` without the workflow id | describe disagrees with history; recovery cannot tell which workflow a run belongs to | `apply` reads every field from the event; `commit` writes a `WalRecord` with both ids |
| 7. an off-by-one payload limit | an input of exactly 2 MiB is refused | refuses only `len(input) > MaxPayloadBytes` |
| 8. list order, pages, and tokens | `wf list` shows the oldest first, repeats a run across pages, or panics on a token for event 0; an unknown workflow panics | a total sort order, an offset token, `ParsePageToken` refusing values below 1, `lookup` returning `NOT_FOUND` |
| 9. the quota as `UNAVAILABLE`, or registering the run before its batch is durable | clients retry a full disk forever; a refused start leaves a ghost run | `logErr` maps `ErrQuota` to `RESOURCE_EXHAUSTED`; `startRun` registers after `commit` |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `dur.01` | each run is stream `run/<run_id>`; `commit` appends with `expected` = history length |
| Back | `dur.03` | effects enqueue tasks; recovery relies on idempotent enqueue |
| Forward | `dur.04` | `PollWorkflowTask` and `CompleteWorkflowTask` build batches and `commit` them; command handlers register in `tasks.go` |
| Forward | `dur.05`, `dur.06`, `dur.07`, `dur.08` | their server files add rpcs and command handlers to the same `Server` and never change `apply` |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| idempotent start on workflow id | Temporal workflow id reuse and conflict policies | allow a new run after a closed one, terminate the old one, or fail; Update-with-Start | Temporal docs, "Workflow Id Reuse Policy" |
| one server, one lock | history shards | runs are partitioned over many history service hosts by workflow id | `temporalio/temporal`: `service/history` |
| `ListWorkflows` over memory | a visibility store (Elasticsearch, SQL) | search attributes and queries over millions of runs | Temporal "Visibility" docs |
| plaintext gRPC | namespaces and mTLS | multi-tenant isolation and authenticated workers | Temporal "Namespaces", "Security" |
