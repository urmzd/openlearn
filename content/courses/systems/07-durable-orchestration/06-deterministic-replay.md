# Deterministic replay

Workers now run activities, but the logic that decides what to run next (train, then evaluate, then release if the loss is good, else retry with a smaller learning rate) still lives nowhere durable. If you write it as ordinary Go in a worker, a crash in the middle loses its local variables and its place in the code. Durable execution lets you keep writing ordinary Go and get crash-proof state anyway, by recording the outcome of every step in history and **replaying** the code against it.

This section builds the SDK that does it. The surprising idea at its heart is that a workflow never runs once from start to finish. Every workflow task re-runs it from the start against the history: past results come from events, and each command the code issues must match the event recorded for it. Once you accept that, the rest follows: the determinism rules, the escape hatches for time and randomness, versioning for code that changes under running workflows, and Continue-As-New for histories that would otherwise grow forever. After this section, `CorpusBuild`, `TrainRun`, `ModelRelease`, and `AgentRun` are plain functions that survive any crash.

The section relies on everything before it in the chapter, through [activities](05-activities.md) and its failure handling. The contract is the Go SDK surface given below and the `Command` and history messages of [`durable.proto`](../../../../course/contracts/proto/tl/durable/v1/durable.proto). The full solution is the SDK calls in [`go/durable/workflow/workflow.go`](../../../../course/solutions/go/durable/workflow/workflow.go) (`ExecuteActivity`, `Sleep`, `Now`, `SideEffect`, `GetVersion`, `ContinueAsNew`, `ContinueAsNewSuggested`), the replay engine in [`go/durable/workflow/replay.go`](../../../../course/solutions/go/durable/workflow/replay.go) (`Registry`, `ReplayHistory`, `LoadHistory`, `ReplayHistoryFromFile`), and the server's `record_marker` and `continue_as_new` commands in [`go/durable/server/continue.go`](../../../../course/solutions/go/durable/server/continue.go).

> **Learning objectives**
> - Replay a workflow history by hand, task by task, and say which events each past task sees.
> - Explain how command matching detects nondeterminism, and why a mismatch must stop the task.
> - Replace wall-clock time, randomness, and concurrency in workflow code with their deterministic equivalents.
> - Use `GetVersion` to change workflow code safely under runs already in flight.
> - Bound a long workflow's history with Continue-As-New.

## Replay

Here is the notation:

| Symbol | Meaning |
|---|---|
| $W$ | the workflow: deterministic code from its input and the results of its commands to commands |
| $h$ | the run's history, as kept by [the workflow service](02-workflow-service.md) |
| $T_1, \dots, T_m$ | its completed workflow tasks: each `WorkflowTaskCompleted` names the `WorkflowTaskStarted` it answers |
| $C_j$ | the command events recorded right after $T_j$'s `WorkflowTaskCompleted`, in one batch (see [the task protocol and workers](04-task-protocol-and-workers.md)) |
| $s_j$ | the event id of $T_j$'s `WorkflowTaskStarted` |
| $s_*$ | the id of the **current** task's `WorkflowTaskStarted`, the last one in $h$ |

To handle the current task, the engine runs this loop:

```
start W as a coroutine
for j in 1..m:                         # past tasks, in order
    deliver every event with id < s_j  # results resolve futures
    run W until it blocks; each command it issues must match the next event of C_j
    if a command differs, or C_j has events left over: ErrNondeterminism
deliver every event with id < s_*
run W until it blocks; the commands it issues now are new: send them
```

The workflow cannot tell a replay from a first run. `Get` on an activity future returns the recorded result if its `ActivityTaskCompleted` was delivered, and otherwise blocks. Because commands are matched in order, the engine detects any change in what the code does at any past point.

### The coroutine

The workflow runs in its own goroutine, but only one of the two (engine, workflow) runs at a time. The engine resumes the workflow through a channel and waits; the workflow, when it must wait on a future, signals back and waits. So a replay is a deterministic function of the code and the history. When the task is done, the engine kills the coroutine, which exits with `runtime.Goexit` from inside its wait.

### Matching

A command matches an event when the following fields agree:

| Command | Event | Compared |
|---|---|---|
| `ScheduleActivity` | `ActivityTaskScheduled` | activity id and activity type |
| `StartTimer` | `TimerStarted` | timer id |
| `RecordMarker` | `MarkerRecorded` | marker name |
| `CompleteWorkflow`, `FailWorkflow`, `ContinueAsNew` | `completed`, `failed`, `continued` | kind |

Activity and timer ids are the command's sequence number in the run ("1", "2", ...), so the same code issues the same ids in the same order on every replay.

## Determinism and its tools

Replay only works if running the same code twice takes the same path. Anything that can differ between two runs of the same code breaks it, and each source of difference has a deterministic replacement:

| Instead of | Use | Why it is safe |
|---|---|---|
| `time.Now()` | `workflow.Now(ctx)` | the time of the current task's `WorkflowTaskStarted`, recorded in history |
| `rand`, `uuid` | `workflow.SideEffect(ctx, f)` | `f` runs once; its JSON result is recorded as a `SideEffect` marker; replays return the recorded value and never call `f` |
| `time.Sleep` | `workflow.Sleep(ctx, d)` | a durable timer (see [durable timers](07-durable-timers.md)) |
| goroutines, channels, I/O | activities | activities run in workers and their results are recorded |
| iterating a map | a sorted slice | Go randomizes map order |

## Versioning

Determinism has a consequence for deploys. A deploy changes workflow code while runs are in flight, and their histories were made by the old code; replaying them with new code that issues different commands is nondeterminism. `GetVersion(ctx, changeID, min, max)` makes a change safe:

```go
if workflow.GetVersion(ctx, "use-b", 0, 1) == 0 {
	// old path, as the code was before the change
} else {
	// new path
}
```

On a first execution it returns `max` and records a `Version` marker `{change_id, version}`. On replay, if the history has that marker at this point, it returns the recorded version (and the marker is consumed); if not, the history predates the change, and it returns `min` without consuming anything. Later calls with the same change id return the same value.

## Continue-As-New and history size

Replay also has a cost: every workflow task replays the whole history. A workflow that loops (train for 10 epochs, evaluate every 500 steps for a week) grows its history without bound. The server suggests continuing once the history passes 10,000 events or 32 MiB (`continue_as_new_suggested`), and `workflow.ContinueAsNewSuggested(ctx)` recomputes the same rule from the history as of the current task, so it gives the same answer on every replay.

The workflow then returns `workflow.ContinueAsNew(ctx, state)`. The SDK issues a `ContinueAsNew` command; the server closes the run with `WorkflowExecutionContinuedAsNew{new_run_id, input}` and, once that is durable, starts the new run under the same workflow id with `continued_from_run_id`. The server refuses to grow a history past 20,000 events at all.

Two related limits apply. A history over 1 MiB reaches the worker in pages, and the engine needs every event, so the first page alone is not enough. And the server refuses payloads over 2 MiB: activity inputs and results and workflow results, as before, and now also marker details and continue-as-new inputs. Large data goes to `/artifacts` and travels by path.

## Testing workflows by replay

Replay also gives you a testing method. A set of small test workflows (`Sequence`, `Parallel`, `SideEffects`, `Versioned`, `Reorder`, `Clock`, `Counter`) is run once through the server and worker, and each history is saved as a JSON file (`HistoryFile`). A correct engine replays every one with the same result. The recipe for any workflow is the same two steps: run it once and fetch the history, then replay it with `ReplayHistory` (or `ReplayHistoryFromFile`). Any code change that breaks determinism then fails a test instead of a production run.

## Working it by hand

`Sequence(2)` calls `Upper("s1")`, then `Upper("s2")`, and returns the list. Here is its history after both activities, with the current task started at event 15:

| id | event |
|---|---|
| 1 | `started{Sequence, input 2}` |
| 2, 3 | `wt_scheduled`, `wt_started` |
| 4 | `wt_completed{started 3}` |
| 5 | `act_scheduled{id "1", Upper, "s1"}` |
| 6, 7 | `act_started`, `act_completed{scheduled 5, "S1"}` |
| 8, 9 | `wt_scheduled`, `wt_started` |
| 10 | `wt_completed{started 9}` |
| 11 | `act_scheduled{id "2", Upper, "s2"}` |
| 12, 13 | `act_started`, `act_completed{scheduled 11, "S2"}` |
| 14, 15 | `wt_scheduled`, `wt_started` (current) |

Handling the task whose history ends at 15 takes three steps:

1. **Past task 1** ($s_1 = 3$): deliver events 1 and 2 (nothing to resolve). Run: the code calls `ExecuteActivity("Upper", "s1")`, sequence 1, and blocks on `Get`. Command `ScheduleActivity{"1", Upper}` matches event 5. $C_1$ is consumed.
2. **Past task 2** ($s_2 = 9$): deliver events 3 to 8; event 7 resolves activity "1" with `"S1"`. Run: `Get` returns `"S1"`; the code calls `ExecuteActivity("Upper", "s2")`, sequence 2, and blocks. Command `"2"` matches event 11.
3. **Current task** ($s_* = 15$): deliver events 9 to 14; event 13 resolves "2" with `"S2"`. Run: the code returns `["S1","S2"]`. The new command is `CompleteWorkflow{["S1","S2"]}`.

Handling the earlier tasks (history up to 3, then up to 9) gives `ScheduleActivity{"1"}` and then `ScheduleActivity{"2"}`: the same function, run against more history.

## Designing the API

The SDK surface is small and deliberately looks like ordinary Go:

```go
package workflow // import "tinyllm/durable/workflow"

type Workflow func(ctx Context, input []byte) ([]byte, error)
type Context struct{ /* unexported */ }
var ErrNondeterminism = errors.New("durable: command does not match history")

type ActivityOptions struct {
	TaskQueue string; StartToClose, ScheduleToClose, HeartbeatTimeout time.Duration; Retry *durablev1.RetryPolicy
}
type ActivityError struct { ActivityID, ActivityType string; Failure *durablev1.Failure }
type Future[O any] interface { Get(ctx Context) (O, error); Ready() bool }

func ExecuteActivity[O any](ctx Context, name string, in any, o ActivityOptions) Future[O] // in: []byte as is, else JSON
func Sleep(ctx Context, d time.Duration) error
func Now(ctx Context) time.Time
func SideEffect[T any](ctx Context, f func() T) T
func GetVersion(ctx Context, changeID string, min, max int) int
func ContinueAsNew(ctx Context, input []byte) error // return it from the workflow
func ContinueAsNewSuggested(ctx Context) bool
func WorkflowID(ctx Context) string
func RunID(ctx Context) string

type Registry struct{ /* unexported */ }
func NewRegistry() *Registry
func (r *Registry) Register(name string, wf Workflow)
func (r *Registry) HandleWorkflowTask(ctx context.Context, task *durablev1.WorkflowTask, history []*durablev1.HistoryEvent) ([]*durablev1.Command, error)

type HistoryFile struct { WorkflowID, RunID, WorkflowType string; Events []json.RawMessage }
func LoadHistory(path string) (*HistoryFile, []*durablev1.HistoryEvent, error)
func ReplayHistory(r *Registry, workflowType string, history []*durablev1.HistoryEvent) ([]byte, error)
func ReplayHistoryFromFile(r *Registry, path string) ([]byte, error)

// go/durable/server/continue.go: registers "record_marker" and "continue_as_new" with the dur.04 command table
```

`Registry` is the `worker.WorkflowHandler` of the worker: `worker.New(conn, worker.Options{Workflows: reg, ...})`. Workflows are Go only; Python implements activities.

## Building it step by step

The design is forced by seven observations. Try to predict each.

1. **The workflow's state is its local variables and its position in the code, and neither survives a crash.** The only durable thing is history. So the SDK must reconstruct both from history, and the only general way to reconstruct the state of arbitrary code is to run it again.
2. **Running it again must take the same path.** That requires the code to be a deterministic function of its input and of the results it receives. Every source of difference (clock, randomness, map order, concurrency, I/O) must either be banned or routed through history.
3. **Results must arrive at the same points they did originally.** A workflow task saw the history up to its `WorkflowTaskStarted`, not more. So replaying past task $j$ delivers exactly the events before $s_j$, then runs the code until it blocks; it then stands exactly where it stood when task $j$ ran.
4. **What the code did at each past task is recorded: the command events after that task's `WorkflowTaskCompleted`.** Comparing the commands the replay issues against them, in order, both proves determinism and detects any code change that would alter the past. A mismatch must stop the task, not be patched over.
5. **Determinism needs a scheduler, not threads.** If the workflow ran freely alongside the engine, the interleaving of deliveries and code would depend on timing. A coroutine (one side runs while the other waits on a channel) makes the order a function of the code and the history only.
6. **Ids must be a function of the code path.** If activity ids came from a random generator or the clock, replay would issue different ids. A sequence counter incremented per command gives "1", "2", ... in the same order every time.
7. **The escape hatches follow from observation 2.** `Now` reads the current task's started time from history; `SideEffect` runs once and records its value as a marker; `GetVersion` records which code path a run took so later code can respect it. Continue-As-New bounds history by starting over with carried state.

### The coroutine

In [`replay.go`](../../../../course/solutions/go/durable/workflow/replay.go), the engine and the workflow goroutine hand control back and forth over two unbuffered channels:

```go
func (e *env) start(wf Workflow) {
	e.resume, e.yield, e.killed = make(chan struct{}), make(chan struct{}), make(chan struct{})
	go func() {
		defer func() {
			r := recover()
			if e.dead {
				return // killed: nobody waits for this goroutine
			}
			if r != nil {
				// ErrNondeterminism from consume arrives here as a panic.
				if err, ok := r.(error); ok {
					e.failure = err
				} else {
					e.failure = fmt.Errorf("workflow panic: %v", r)
				}
			}
			e.done = true
			e.yield <- struct{}{} // final hand-back
		}()
		select { // do not run a single line until the engine says so
		case <-e.resume:
		case <-e.killed:
			e.dead = true
			return
		}
		e.result, e.err = wf(Context{e}, e.input)
	}()
}

// Called by the workflow when it must wait (an unresolved future).
func (e *env) block() {
	e.yield <- struct{}{} // give control to the engine
	select {
	case <-e.resume:
	case <-e.killed:
		e.dead = true
		runtime.Goexit() // unwind the workflow's stack; deferred recover sees e.dead
	}
}
```

`step` resumes the workflow and waits for it to yield. When the workflow has returned, `step` issues its final command once: `ContinueAsNew` if the error is a `continueAsNewError`, `FailWorkflow` for any other error, and otherwise `CompleteWorkflow` with the result.

### Commands: new or matched

Every SDK call that has an effect goes through `emit`, which either records a new command or checks it against history:

```go
func (e *env) emit(c *durablev1.Command) {
	if e.replaying {
		e.consume(c) // a past task: must match what history recorded
		return
	}
	e.commands = append(e.commands, c) // the current task: new, sent to the server
}

func (e *env) consume(c *durablev1.Command) *durablev1.HistoryEvent {
	ev := e.peek()
	if ev == nil {
		panic(nondeterminism("the workflow issued %s, but the history has no further command for this task", commandKind(c)))
	}
	if !matches(c, ev) {
		panic(nondeterminism("the workflow issued %s, but the history has %s at event %d", commandKind(c), eventKind(ev), ev.GetEventId()))
	}
	e.expIdx++
	return ev
}

func matches(c *durablev1.Command, ev *durablev1.HistoryEvent) bool {
	switch x := c.GetCmd().(type) {
	case *durablev1.Command_ScheduleActivity:
		a := ev.GetActScheduled()
		// Both id and type: same position but a different activity is a code change.
		return a != nil && a.GetActivityId() == x.ScheduleActivity.GetActivityId() && a.GetActivityType() == x.ScheduleActivity.GetActivityType()
	case *durablev1.Command_RecordMarker:
		return ev.GetMarker() != nil && ev.GetMarker().GetMarkerName() == x.RecordMarker.GetMarkerName()
	case *durablev1.Command_Complete:
		return ev.GetCompleted() != nil
	// ... start_timer by timer id, fail, continue_as_new, and the dur.07/dur.08 cancel commands
	}
	return false
}
```

Panicking from `consume` unwinds the workflow out of whatever SDK call it was in; the coroutine's `recover` turns the panic into `e.failure`, and `step` returns it. The workflow cannot catch and ignore it.

### Delivering results

`deliver` resolves futures by the ids in events. `act_scheduled` maps the scheduled event id to the activity id, so a later `act_completed{scheduled 5}` resolves future `"a:1"`. Failures, timeouts, and cancellations resolve the future with an `*ActivityError`; `timer_fired` resolves `"t:<id>"`; other kinds go to `eventHooks`, which [signals, cancellation, and sagas](08-signals-cancellation-and-sagas.md) fills.

### The replay loop

This is the pseudocode from the start of the section, made real:

```go
func (e *env) replay(wf Workflow, h []*durablev1.HistoryEvent, current int64) ([]*durablev1.Command, error) {
	// ... h[0] must be started; e.input is its input
	sizes := make([]int64, len(h)+1) // prefix sums: bytes of h[:i]
	// ...
	at := func(started int64) {
		ev := h[started-1]
		e.now = time.UnixMilli(ev.GetTsUnixMs()).UTC()                        // Now() for this task
		e.suggested = started > suggestEvents || sizes[started] > suggestBytes // the server's rule, as of this task
	}
	e.start(wf)
	defer e.kill()
	next := 0
	deliverBefore := func(id int64) {
		for next < len(h) && h[next].GetEventId() < id { // strictly before s_j: what that task saw
			e.deliver(h[next])
			next++
		}
	}
	for i, ev := range h {
		c := ev.GetWtCompleted()
		if c == nil {
			continue
		}
		// Past task j: rewind the world to its WorkflowTaskStarted...
		deliverBefore(c.GetStartedEventId())
		at(c.GetStartedEventId())
		// ...and expect exactly the command events written in the same batch.
		e.replaying, e.expIdx, e.expected = true, 0, nil
		for j := i + 1; j < len(h) && isCommandEvent(h[j]); j++ {
			e.expected = append(e.expected, h[j])
		}
		if err := e.step(); err != nil {
			return nil, err
		}
		if e.expIdx != len(e.expected) { // the code issued fewer commands than history has
			ev := e.expected[e.expIdx]
			return nil, nondeterminism("the history has %s at event %d, which the workflow no longer issues", eventKind(ev), ev.GetEventId())
		}
	}
	e.replaying, e.expected = false, nil
	if current == 0 { // ReplayHistory of a closed run: the code must have finished
		// ...
	}
	deliverBefore(current) // the current task
	at(current)
	if err := e.step(); err != nil {
		return nil, err
	}
	return e.commands, nil
}
```

Both directions of mismatch are covered: an extra or different command panics in `consume`, and a missing one leaves `expIdx` short. `HandleWorkflowTask` finds the *last* `WorkflowTaskStarted` in the history as `current` (a redelivered task has several), and builds a fresh `env` per task, so nothing leaks between tasks.

### The SDK calls

In [`workflow.go`](../../../../course/solutions/go/durable/workflow/workflow.go), `ExecuteActivity` allocates a deterministic id, registers a future, and emits the command; `get` blocks until a delivered event resolves the future:

```go
func ExecuteActivity[O any](ctx Context, name string, in any, o ActivityOptions) Future[O] {
	e := ctx.e
	f := &future{}
	input, err := encode(in) // []byte as is, anything else JSON
	// ... an encoding error resolves the future with it, issuing no command
	e.seq++
	id := strconv.Itoa(e.seq) // deterministic id: the command's position in the run
	e.futures["a:"+id] = f
	e.activityTypes[id] = name
	e.emit(&durablev1.Command{Cmd: &durablev1.Command_ScheduleActivity{ScheduleActivity: &durablev1.ScheduleActivity{
		ActivityId: id, ActivityType: name, Input: input, Options: o.proto(),
	}}})
	return typedFuture[O]{f}
}

func (f *future) get(ctx Context) ([]byte, error) {
	for !f.ready {
		ctx.e.block() // yield until a delivered event resolves this future
	}
	return f.value, f.err
}
```

`SideEffect` and `GetVersion` are the two escape hatches that write markers:

```go
func SideEffect[T any](ctx Context, f func() T) T {
	e := ctx.e
	var out T
	if e.replaying {
		// Never call f on replay: take the recorded value.
		ev := e.consume(&durablev1.Command{Cmd: &durablev1.Command_RecordMarker{RecordMarker: &durablev1.RecordMarker{MarkerName: "SideEffect"}}})
		if err := json.Unmarshal(ev.GetMarker().GetDetails(), &out); err != nil {
			panic(nondeterminism("SideEffect marker at event %d does not decode: %v", ev.GetEventId(), err))
		}
		return out
	}
	out = f() // the only call, ever
	details, err := json.Marshal(out)
	// ...
	e.emit(&durablev1.Command{Cmd: &durablev1.Command_RecordMarker{RecordMarker: &durablev1.RecordMarker{MarkerName: "SideEffect", Details: details}}})
	return out
}

func GetVersion(ctx Context, changeID string, min, max int) int {
	e := ctx.e
	if v, ok := e.versions[changeID]; ok {
		return v // stable for the rest of the run
	}
	v := max // a first execution takes the newest path...
	if e.replaying {
		v = min // ...a history without the marker here predates the change
		if ev := e.peek(); ev != nil && ev.GetMarker().GetMarkerName() == "Version" {
			var vm versionMarker
			if json.Unmarshal(ev.GetMarker().GetDetails(), &vm) == nil && vm.ChangeID == changeID {
				e.expIdx++ // consume the marker only when it is ours
				v = vm.Version
			}
		}
	} else {
		details, _ := json.Marshal(versionMarker{ChangeID: changeID, Version: max})
		e.emit(/* RecordMarker{"Version", details} */)
	}
	// ... panic with ErrNondeterminism if v is outside [min, max]
	e.versions[changeID] = v
	return v
}
```

`GetVersion` peeks rather than calling `consume`, because a missing marker is a legitimate answer (old history), not a mismatch. `Now` returns `e.now` and `ContinueAsNewSuggested` returns `e.suggested`, both set by `at` from history. `ContinueAsNew` just returns a sentinel error that `step` turns into the command.

### The server side

[`continue.go`](../../../../course/solutions/go/durable/server/continue.go) registers two commands in the command table from [the task protocol and workers](04-task-protocol-and-workers.md). The interesting one is `continue_as_new`:

```go
func continueAsNew(s *Server, b *batch, completed int64, c *durablev1.Command) error {
	can := c.GetContinueAsNew()
	if len(can.GetInput()) > MaxPayloadBytes {
		return status.Errorf(codes.InvalidArgument, /* ... */)
	}
	r := b.r
	ev := &durablev1.WorkflowExecutionContinuedAsNew{
		NewRunId: s.o.NewRunID(), Input: can.GetInput(),
		WorkflowType: can.GetWorkflowType(), TaskQueue: can.GetTaskQueue(),
	}
	// Default to the current run's type and queue, or the next run is one no worker polls.
	if ev.WorkflowType == "" {
		ev.WorkflowType = r.wfType
	}
	if ev.TaskQueue == "" {
		ev.TaskQueue = r.taskQueue
	}
	s.add(b, &durablev1.HistoryEvent{Attrs: &durablev1.HistoryEvent_Continued{Continued: ev}})
	// Start the next run only after this batch is durable; recovery (dur.02)
	// starts it if the server dies in between, using the recorded new run id.
	b.after = append(b.after, func(ctx context.Context) error {
		_, err := s.startRun(ctx, r.workflowID, ev.GetNewRunId(), continuedStart(r, ev))
		return err
	})
	return nil
}
```

`recordMarker` requires a name, caps details at 2 MiB, and appends `MarkerRecorded` with the completed event id.

The full solution is in [`workflow.go`](../../../../course/solutions/go/durable/workflow/workflow.go), [`replay.go`](../../../../course/solutions/go/durable/workflow/replay.go), and [`continue.go`](../../../../course/solutions/go/durable/server/continue.go).

### Why this design

- **Cost.** Each workflow task replays the whole history: O(history) time per task, O(history squared) over a run's life. Continue-As-New bounds the history, and sticky caches remove the replay in production. The coroutine costs one goroutine and two channel handoffs per block.
- **Replay over snapshotting state.** Serialising a Go function's stack and locals is not possible in general; replaying deterministic code is. The price is the determinism rules, which the matcher enforces rather than trusts.
- **Strict matching over best effort.** A replay that continued after a mismatch would run new code on a history made by old code, producing decisions nobody wrote. Failing the task leaves the run stuck but correct, and fixable with `GetVersion`.
- **Sequence ids over named ids.** Users never have to invent ids, and a reorder of two activities is detected as a type mismatch at the first position.

## What a correct SDK must handle

The tests check the engine against these cases:

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| hand example replay | unit | the worked example: the three tasks issue `"1"`, `"2"`, then complete with `["S1","S2"]` | the definition of replay |
| recorded histories replay | conformance, golden | every recorded test run replays with its recorded result (continued runs end in ContinueAsNew) | the SDK conformance suite |
| nondeterminism detected | unit, fault | another activity type, fewer commands, an extra command: `ErrNondeterminism`, replayed and live | code changes cannot corrupt runs |
| reorder failpoint is nondeterministic | fault | `dur/workflow/reorder` flips the order: the same `ErrNondeterminism` three times; passes without it | the failpoint of the catalog |
| side effect not re-executed | unit | replay returns recorded values and never calls the function; a first run calls it once | random ids stay stable |
| `GetVersion` migration | unit, regression | old history on the old path, new history on the new path, a new run records `{use-b, 1}` | safe deploys of TrainRun and ModelRelease |
| `Now` comes from history | unit | `Now` equals the `WorkflowTaskStarted` times, on replay too | no wall clock in workflows |
| record, then replay a live run | unit | a run through the server and worker replays to the same result | how any workflow is tested |
| kill the server mid-workflow: same result | fault | the server dies mid-run; the result equals an uninterrupted run and the history replays | the durable engine's replay determinism |
| ContinueAsNew keeps history bounded | fault, boundary | 12,000 steps continue as new past 10,000 events; each run under 20,000; pages over 1 MiB fetched | `TrainRun`, `AgentRun` |
| server command limits | boundary | marker and continue-as-new payloads over 2 MiB refused; the next run keeps type and queue | payload limits |

## What can go wrong

Replay bugs are unforgiving: they either make every past task look nondeterministic, or, worse, let a changed workflow continue silently on a history it did not make.

The replay plumbing has several failure points: dropping a delivered result, collecting a task's commands from the wrong place, completing with the wrong value, or taking the first `WorkflowTaskStarted` as current. The workflow then sees empty results, every past task looks nondeterministic, or old commands are issued again. `deliver` resolves by scheduled id, `expected` is the command events after each `wt_completed`, and `HandleWorkflowTask` takes the last `wt_started`.

> **Warning:** Matching too loosely (id without type, or any completion) lets a changed workflow continue silently on a history it did not make. `matches` compares id and type, and checks each final command only against its own event kind.

Ids that do not advance with each command let two activities share an id, so results go to the wrong future; `e.seq++` runs before every activity and timer id. Calling a side effect again on replay produces a new random id each time and forks the workflow's state; `SideEffect` reads the marker when `e.replaying` and never calls `f`.

Versioning breaks every in-flight run at the next deploy if it takes the new path for old histories, records the wrong version, or fails to consume the marker. The solution returns `min` without a marker, records `max` on a first run, and does `expIdx++` when the marker is consumed.

Continue-As-New has its own mistakes: treating it as a failure, suggesting it too late, or starting the next run without its type and queue. Long workflows then fail, hit the history cap, or start a run no worker polls. `continueAsNewError` becomes a `ContinueAsNew` command, `at` recomputes the suggestion, and the server defaults type and queue.

Reading the wall clock makes each replay see a different time and take a different branch; `Now` returns the current task's `WorkflowTaskStarted` time. And without payload limits on markers or continue-as-new inputs, you get messages over 4 MiB and histories that cannot be served; `recordMarker` and `continueAsNew` refuse payloads over `MaxPayloadBytes`.

## Where this leads

Activity futures here resolve with the results, `ActivityError`s, and redriven attempts produced by [activities](05-activities.md). Next, [durable timers](07-durable-timers.md) makes `Sleep` issue `StartTimer`, with the server's timer commands and the timer service firing it, and [signals, cancellation, and sagas](08-signals-cancellation-and-sagas.md) adds event hooks and commands to the same engine. Beyond the engine itself, [TrainRun and EvalSuite](11-trainrun-and-evalsuite.md), [ModelRelease](12-model-release.md), [CorpusBuild as a durable workflow](../../data-engineering/05-corpus-pipeline/09-corpusbuild-as-a-durable-workflow.md) in *Data Engineering*, and [durable agent runs](../../ai-engineering/28-agents/05-durable-agent-runs.md) are all workflows on this SDK.

Production SDKs go further:

- **Sticky caches.** Instead of replaying from event 1 on every task, the worker keeps a run's coroutine between tasks and replays nothing, and the server sends only new events. See Temporal "Sticky Execution".
- **Full command state machines.** Instead of one coroutine per workflow, `workflow.Go`, selectors, child workflows, and cancellation scopes each get their own state machine. See `temporalio/sdk-go`: `internal/internal_decision_state_machine.go`.
- **Worker deployment versioning.** Instead of `GetVersion` by hand, old runs stay pinned to the build that started them. See Temporal "Worker Versioning".
- **Replay testing on upgrade.** Beyond replay on fixtures, replay production histories against new code in CI before deploying. See Temporal "Replay Testing".
- **Reconciling indeterminate effects.** Where a crash mid-activity here simply retries it, saige's `agent/durable/local` (`ErrIndeterminate` and `Reconcile`) reconciles an effect whose outcome is unknown instead of blindly retrying it.

### Further reading

- The Temporal docs on deterministic constraints and versioning.
- Maxim Fateev and Samar Abbas, talks on Temporal and durable execution.
- Hellerstein et al., on deterministic replay.

## Key terms

- **Replay**: re-running workflow code from the start against its history, so past results come from recorded events.
- **Command matching**: comparing each command the replaying code issues against the event recorded for it, in order.
- **Nondeterminism**: a mismatch between replayed commands and history, reported as `ErrNondeterminism`.
- **Coroutine**: the workflow goroutine that runs only while the engine waits, making replay order a function of code and history.
- **Side effect marker**: a `MarkerRecorded` event holding the value of a non-deterministic function, returned on replay instead of calling it.
- **Version marker**: a `MarkerRecorded` event that records which code path a run took at a change, read by `GetVersion`.
- **Continue-As-New**: closing a run and starting a fresh one under the same workflow id with carried state, to bound history.

## Summary

- A workflow never runs once from start to finish. Every workflow task re-runs it from the start against the history: past results come from events, and each command it issues must match the event recorded for it.
- Determinism is the price: no wall clock (`workflow.Now`), no randomness (`SideEffect`), no goroutines or I/O. A code change that reorders commands is caught as `ErrNondeterminism`, never silently continued.
- `GetVersion` lets a deploy change code under runs already in flight: old histories replay the old path, and new runs take the new one.
- A long workflow bounds its history with `ContinueAsNew` when the server suggests it, at 10,000 events or 32 MiB.
- Kill the server in the middle of a run: the worker replays, and the result equals an uninterrupted run's.

## Review questions

1. In the worked example, why does replay deliver only events 1 and 2 before running past task 1, not everything up to event 4?

   <details><summary>Answer</summary>

   Past task 1 started at event 3, so it saw the history before 3. Delivering later events (for example a result) would let the code take a path the original task could not have taken, and its commands would not match $C_1$.

   </details>

2. A developer swaps the order of `Upper("s1")` and `Lower("s2")` in a deployed workflow. What happens to a run that already completed task 1?

   <details><summary>Answer</summary>

   On its next task, replaying past task 1 issues `ScheduleActivity{"1", Lower}`, but the history has `act_scheduled{"1", Upper}`. `matches` compares types, so `consume` panics with `ErrNondeterminism`, the worker does not complete the task, and the run stays stuck until the change is guarded with `GetVersion`.

   </details>

3. Why does `SideEffect` call `f` only when not replaying, while `ExecuteActivity` issues its command either way?

   <details><summary>Answer</summary>

   Both issue (or match) a command either way. The difference is the value: an activity's result arrives later from history, so the call itself has no value to compute. A side effect's value is produced locally, so on replay it must come from the recorded marker; calling `f` again would produce a different random id and fork the workflow's state.

   </details>

4. Why does `GetVersion` return `min` when replaying a history that has no `Version` marker at that point?

   <details><summary>Answer</summary>

   The run started before the change was deployed, so the old code ran there and issued the old commands. Taking the old path keeps replay matching that history. New runs, which are not replaying, record `max` and take the new path.

   </details>

5. Why does `ContinueAsNewSuggested` recompute the threshold from history instead of reading the server's `continue_as_new_suggested` flag?

   <details><summary>Answer</summary>

   The flag describes the current task only. A past task replayed later must see the value it saw at the time, or the code would branch differently on replay. Recomputing from the prefix of history up to each task's `WorkflowTaskStarted` gives the same answer every time.

   </details>
