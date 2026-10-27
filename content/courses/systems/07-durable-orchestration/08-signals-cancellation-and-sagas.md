# Signals, cancellation, and sagas

By this point the engine can run a workflow to the end and survive any crash along the way, but nothing outside the workflow can talk to it while it runs. That gap matters as soon as real workflows show up. A model release needs a human to say "approve" partway through ([ModelRelease: gates, canary, burn check, and rollback](12-model-release.md)). A corpus build started with the wrong config needs to be stopped, and stopping it halfway leaves shards that look like a finished corpus ([CorpusBuild: the pipeline as a durable workflow](../../data-engineering/05-corpus-pipeline/09-corpusbuild-as-a-durable-workflow.md) in *Data Engineering*). A training run needs to be cancelled without throwing away its checkpoints ([Platform workflows: TrainRun and EvalSuite](11-trainrun-and-evalsuite.md)). Killing the process does none of this safely: the run would simply be redelivered to another worker.

This section adds the three ways the outside world steers a running workflow: **signals**, **cancellation**, and the **compensation** (saga) that undoes work cleanly. All three follow the rule that runs through the whole engine: history first, effects second. When you finish, an operator can approve or cancel a running workflow, and a workflow can unwind whatever it had already done.

The server half extends the package built in [the workflow service](02-workflow-service.md), [the task protocol and worker](04-task-protocol-and-workers.md), and [activities](05-activities.md); the SDK half plugs into the replay engine of [Deterministic replay](06-deterministic-replay.md) through its `eventHooks`; and signal timeouts use the [durable timers](07-durable-timers.md) from the previous section. The code is Go; [the Go primer](../../programming-and-algorithms/01-language-primers/06-go.md) in *Programming and Algorithms* covers what you need. The reference implementation lives in four files: [`go/durable/server/signals.go`](../../../../course/solutions/go/durable/server/signals.go) (the `SignalWorkflow` and `CancelWorkflow` rpcs and the `RequestCancelActivity` and `CancelWorkflowExecution` commands), [`go/durable/workflow/signal.go`](../../../../course/solutions/go/durable/workflow/signal.go) (the SDK half: signal channels, `Await`, `CancelRequested`, cancellable timers, `ExecuteActivityID`, `CancelRun`), [`go/workflows/runtime.go`](../../../../course/solutions/go/workflows/runtime.go) (the `Runtime` seam every platform workflow is written against, its adapter `FromContext`, the `Workflow` wrapper, `ErrCanceled`), and [`go/workflows/saga.go`](../../../../course/solutions/go/workflows/saga.go) (`Saga` with `Add`, `Compensate`, `Fail`). The wire contract is [`durable.proto`](../../../../course/contracts/proto/tl/durable/v1/durable.proto): the rpcs and commands marked `[dur.08]` and the events `SignalReceived`, `WorkflowExecutionCancelRequested`, `ActivityTaskCancelRequested`, `ActivityTaskCanceled`, and `WorkflowExecutionCanceled`. An operator reaches the two RPCs through a client such as the system's control CLI.

> **Learning objectives**
> - Explain why a signal is a history event rather than a message pushed to a worker, and why that makes it impossible to lose.
> - Make signals idempotent with request ids that survive a server restart.
> - Implement cancellation as a request: record it, tell exactly the activities pending at that moment, and let the workflow decide how to end.
> - Write a saga that compensates completed steps in reverse order, once each, with stable keys, on a runtime the cancel cannot interrupt.
> - Trace a signal and a saga through the history by hand.

## Signals are history

The first principle is the one replay forces on you: anything the workflow reacts to must be in history, or replay cannot reproduce the reaction. So a signal is not a message delivered to whichever worker holds the run. It is an event appended to the run's history.

`SignalWorkflow(workflow_id, signal_name, input, request_id)` appends `SignalReceived` to the run and returns. Appending it is news the workflow must react to, so the server's commit schedules a workflow task, or, when one is already in flight, marks the run to get another one right after (`needsWT`). This is the existing `commit` rule from the workflow service; no new delivery path is needed. The workflow then receives the signal by replaying its history. A signal that arrived before the workflow reached `AwaitSignal`, or while a workflow task was running, is simply an earlier event in the same history, so nothing can be lost: whichever workflow task runs next sees it. A signal to an unknown or closed run is refused with `NOT_FOUND`, and its input obeys the 2 MiB payload limit.

External callers retry, and a client retrying "approve" must not approve twice. A **request id** makes a signal idempotent: `apply` records each id in the run's state as it folds `SignalReceived` events, so a second signal with the same id is acknowledged and appends nothing. Because the state is rebuilt from the log, this holds after a restart too, with no extra storage.

## Cancellation is a request

Signals let the outside world feed a workflow. Cancellation lets it stop one, and here the tempting design, a kill, is wrong. Killing would lose the chance to undo, and the run would just be redelivered. So cancellation is a request that the workflow honours: activities pending at that moment are told by name, every blocking call starts returning `ErrCanceled`, and only the workflow may end the run as `CANCELED`, after cleaning up.

The table below follows one cancellation from the operator's call to the closed run.

| Step | Who | What is appended |
|---|---|---|
| `CancelWorkflow` | operator, CLI | `WorkflowExecutionCancelRequested`, and `ActivityTaskCancelRequested` for every activity pending right now (plus `ActivityTaskCanceled` for a dead-lettered one, which no attempt can acknowledge) |
| next heartbeat of a pending activity | worker | nothing: the answer says `cancel_requested`, the worker cancels the attempt's context, the subprocess runner sends SIGTERM ([Subprocess activities](09-subprocess-activities/README.md)) |
| next workflow task | workflow | every blocking `Runtime` call returns an error wrapping `ErrCanceled`; the workflow compensates |
| `CancelWorkflowExecution` command | workflow | `WorkflowExecutionCanceled`: the run is `CANCELED` |

Cancel is idempotent: a second request, or one for a run that already ended canceled, appends nothing. A run that completed or failed cannot be cancelled (`FAILED_PRECONDITION`). `CancelWorkflowExecution` without a request is refused, because a workflow ends itself by completing or failing; only an operator's request can make a run end canceled. Finally, `RequestCancelActivity` cancels a single activity by id, once, for cases the workflow itself decides: a race between branches, or a deadline the workflow enforces.

One subtlety drives much of the design. Cleanup must not itself be cancelled. The compensations a workflow schedules after the cancel are new activities; they must not receive the stop request, and the runtime they run on must not fail with `ErrCanceled`. That is why the server tells only the activities pending at cancel time, and why the SDK offers `Detached()`. The next idea, sagas, is where that matters.

## Sagas

A workflow that changes the world in several steps cannot roll them back in one transaction. A **saga** (Garcia-Molina and Salem, 1987) pairs each forward step $T_i$ with a **compensation** $C_i$ that semantically undoes it, and on failure after $T_k$ runs $C_k, C_{k-1}, \dots, C_1$. Here $T_i$ is the $i$-th forward step (an activity), $C_i$ is its compensation (an activity that undoes $T_i$, and must be idempotent), and $k$ is the last step that started.

Four rules make a saga safe in a durable engine:

1. **Register before the step** whose partial effects need undoing. A step interrupted halfway may have left something behind, and its compensation must run too.
2. **Reverse order.** $C_k$ runs first, because later steps build on earlier ones: stop the job before deleting the data it reads.
3. **Once, with stable keys.** A compensation is an activity whose id is `compensate-<n>` (or one you name), so on replay after a crash it is answered from history or redelivered with the same idempotency key. `Compensate` called again in the same run does nothing.
4. **Detached and complete.** Compensations run on `rt.Detached()`, which a cancel does not interrupt, and a failed compensation does not stop the others; the errors are joined to the original cause.

## The SDK half: hooks into the replay engine

Those are the server-side and conceptual pieces. On the worker, the workflow code needs to see signals and cancel requests as it replays. The replay engine from [Deterministic replay](06-deterministic-replay.md) resolves activity and timer events itself and hands every other event to `eventHooks`. `signal.go` registers two hooks: `SignalReceived` stores the payload as the run's next signal of that name, and `WorkflowExecutionCancelRequested` marks the run cancelled. Both keep their state where the engine keeps all per-run state, in the run's futures, so a replay rebuilds it from history. The keys are:

| Future key | Holds |
|---|---|
| `s:<name>:<n>` | the $n$-th signal named `<name>` |
| `sc:<name>` | how many of them the workflow consumed |
| `c:` | resolved once the run has a cancel request |

On top of these, `Await(ctx, cond)` blocks the workflow coroutine until `cond` holds, re-checking after every batch of events. That is how `Receive` waits for a signal and how the adapter waits for "the activity finished, or the run was cancelled". A durable timer that can be cancelled (`NewTimer`, `Cancel`) gives `AwaitSignal` its timeout. `ExecuteActivityID` schedules an activity under a chosen id, so the idempotency key is `<workflow_id>/<id>`. `CancelRun` issues `CancelWorkflowExecution` and parks the coroutine for good: the run is over when the task completes.

## The Runtime seam

The platform workflows (`CorpusBuild`, `TrainRun`, `EvalSuite`, `ModelRelease`) are not written against the SDK directly. They are plain Go functions of a `Runtime`, the slice of the SDK they need: `ExecuteActivity`, `Sleep`, `Now`, `AwaitSignal`, `Detached`. The worker's composition root adapts a `workflow.Context` to it when it registers them. The seam is what lets workflow logic be tested with a replaying simulator, without a server, and it states the cancel rule exactly once: after a cancel request, every blocking call of the Runtime fails with `ErrCanceled`, and no call of `Detached()` does.

## Working it by hand

With the pieces named, trace one signal and one saga event by event.

**A signal while the workflow is busy.** Workflow `release-7` starts (events 1 `started`, 2 `wt_scheduled`), and a worker polls (3 `wt_started`). While it works, the approver calls `SignalWorkflow(release-7, approve, {"by":"ana"}, request r1)`. The server appends 4 `signal` and, since a workflow task is in flight, only marks the run. The approver retries the same call; `r1` is already in the run's signal ids, so nothing is appended. The worker completes its task with `ScheduleActivity(1, prepare)`: 5 `wt_completed`, 6 `act_scheduled`, and because the run was marked, 7 `wt_scheduled`. The next task's history ends `... signal wt_completed act_scheduled wt_scheduled wt_started`, so the workflow replays the signal and proceeds.

**A saga that unwinds.** A booking workflow registers $C_1$ = `gpu.release` and runs $T_1$ = `gpu.reserve`; registers $C_2$ = `data.delete` and runs $T_2$ = `data.copy`; registers $C_3$ = `job.stop` and runs $T_3$ = `job.start`, which fails for good (`QuotaExceeded`). `Fail` runs `job.stop`, `data.delete`, `gpu.release`, in that order, with keys `book-1/compensate-3`, `-2`, `-1`, and returns the `QuotaExceeded` failure. The execution order is `gpu.reserve data.copy job.start job.stop data.delete gpu.release`. If the worker dies while `data.delete` runs, the replay answers `job.stop` from history and runs `data.delete` again with the same key, and the storage service deduplicates it.

## Designing the API

The interface spans the server, the SDK, the Runtime seam, and the saga. Listing 1 collects every signature.

```go
// go/durable/server/signals.go (package server)
func (s *Server) SignalWorkflow(ctx context.Context, req *durablev1.SignalWorkflowRequest) (*durablev1.SignalWorkflowResponse, error)
func (s *Server) CancelWorkflow(ctx context.Context, req *durablev1.CancelWorkflowRequest) (*durablev1.CancelWorkflowResponse, error)
// commands registered with registerCommand: "cancel_activity", "cancel_workflow"

// go/durable/workflow/signal.go (package workflow)
func CancelRequested(ctx Context) bool
func Await(ctx Context, cond func() bool)
func GetSignalChannel(ctx Context, name string) ReceiveChannel   // Len, ReceiveAsync, Receive
func NewTimer(ctx Context, d time.Duration) *Timer              // Fired, Cancel
func ExecuteActivityID[O any](ctx Context, id, name string, in any, o ActivityOptions) Future[O]
func CancelRun(ctx Context, details []byte)

// go/workflows/runtime.go
var ErrCanceled error
type StepOptions struct { ID string; StartToClose, HeartbeatTimeout time.Duration; MaxAttempts int }
type Runtime interface {
	ExecuteActivity(name string, in any, opts StepOptions, out any) error
	Sleep(d time.Duration) error
	Now() time.Time
	AwaitSignal(names []string, timeout time.Duration) (name string, payload []byte, err error)
	Detached() Runtime
}
type StepFailure struct { Activity, Type, Message string; NonRetryable bool }
func IsCanceled(err error) bool
func FromContext(ctx workflow.Context) Runtime
func Workflow[I, O any](f func(Runtime, I) (O, error)) workflow.Workflow

// go/workflows/saga.go
type Saga struct{ /* ... */ }
func (s *Saga) Add(name, activity string, in any, opts StepOptions)
func (s *Saga) Len() int
func (s *Saga) Compensate(rt Runtime) error
func (s *Saga) Fail(rt Runtime, cause error) error
```

## Building it step by step

The reasoning chain is short. Signals and cancel requests must be history events, because replay must reproduce the workflow's reaction. Appending them must wake the workflow, which the existing commit rule already does. External retries must not double an effect, so request ids go into run state. Cancellation cannot be a kill, so it is a request the workflow honours. Cleanup must not be cancelled, so the server tells only activities pending at cancel time and the SDK provides `Detached()`. And undo must happen in reverse, once, with stable identities, so compensations run newest first with deterministic ids and a flag makes `Compensate` idempotent within a run. The code follows from each of those in turn.

### SignalWorkflow

Listing 2 is the signal rpc from [`signals.go`](../../../../course/solutions/go/durable/server/signals.go).

```go
func (s *Server) SignalWorkflow(ctx context.Context, req *durablev1.SignalWorkflowRequest) (*durablev1.SignalWorkflowResponse, error) {
	// ... empty name, input over MaxPayloadBytes: INVALID_ARGUMENT
	s.mu.Lock()
	defer s.mu.Unlock()
	r, err := s.lookup(req.GetWorkflowId(), req.GetRunId()) // empty run id: the current run
	if err != nil {
		return nil, err
	}
	if r.status != durablev1.WorkflowStatus_WORKFLOW_STATUS_RUNNING {
		// Refuse loudly: an accepted signal to a closed run would vanish.
		return nil, status.Errorf(codes.NotFound, "workflow %q run %q is closed", r.workflowID, r.id)
	}
	if id := req.GetRequestId(); id != "" && r.signals[id] {
		return &durablev1.SignalWorkflowResponse{}, nil // a retry: acknowledged, nothing appended
	}
	b := &batch{r: r}
	s.add(b, &durablev1.HistoryEvent{Attrs: &durablev1.HistoryEvent_Signal{Signal: &durablev1.SignalReceived{
		SignalName: req.GetSignalName(), Input: req.GetInput(), RequestId: req.GetRequestId(),
	}}})
	if err := s.commit(ctx, b); err != nil { // a trigger: commit adds wt_scheduled or sets needsWT
		return nil, err
	}
	return &durablev1.SignalWorkflowResponse{}, nil
}
```

`r.signals` is filled by `apply` from every `SignalReceived` with a request id, so dedup survives a restart with no extra storage.

### CancelWorkflow: record once, tell what is pending now

Listing 3 is the body of the cancel rpc after the run lookup.

```go
switch r.status {
case durablev1.WorkflowStatus_WORKFLOW_STATUS_CANCELED:
	return &durablev1.CancelWorkflowResponse{}, nil // already done: idempotent
case durablev1.WorkflowStatus_WORKFLOW_STATUS_RUNNING:
default:
	return nil, status.Errorf(codes.FailedPrecondition, /* completed, failed, or continued */)
}
if r.cancelRequested {
	return &durablev1.CancelWorkflowResponse{}, nil // one request event however many calls
}
b := &batch{r: r}
s.add(b, /* WorkflowExecutionCancelRequested{reason} */)
// Every activity pending *now*, in scheduled order, in the same batch.
// Activities scheduled later (the compensations) are never in this set.
ids := make([]int64, 0, len(r.activities))
for id := range r.activities {
	ids = append(ids, id)
}
sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] }) // map order is random: sort for a deterministic history
for _, id := range ids {
	requestCancel(s, b, r.activities[id])
}
if err := s.commit(ctx, b); err != nil {
	return nil, err
}
return &durablev1.CancelWorkflowResponse{}, nil
```

Each pending activity goes through `requestCancel` (Listing 4), which is also the single place that handles the dead-letter case.

```go
func requestCancel(s *Server, b *batch, a *activity) {
	if a.cancelRequested || pendingIn(b, a.scheduledID) {
		return // once per activity, even if a workflow repeats the command in one task
	}
	s.add(b, /* ActivityTaskCancelRequested{scheduled id} */) // apply sets a.cancelRequested
	if a.deadLettered {
		// No attempt is running to hear it on a heartbeat: cancel it now, or
		// the workflow waits for an answer that never comes.
		s.add(b, /* ActivityTaskCanceled{scheduled id} */)
	}
}
```

The running attempt learns of the request from `RecordHeartbeat`, which answers `cancel_requested: a.cancelRequested` (see [Activities](05-activities.md)). The flag is per activity, not per run, which is exactly what lets compensations run to the end. `cancelActivity` (the `RequestCancelActivity` command) finds the pending activity by id and calls the same `requestCancel`; an unknown or finished id is ignored, because cancellation races completion. `cancelWorkflow` (the `CancelWorkflowExecution` command) is refused with `FAILED_PRECONDITION` unless `r.cancelRequested`, and then appends `WorkflowExecutionCanceled{details}`.

### The SDK hooks

On the worker side, [`signal.go`](../../../../course/solutions/go/durable/workflow/signal.go) keeps its state in the engine's futures map so a replay rebuilds it (Listing 5).

```go
func init() {
	eventHooks["signal"] = deliverSignal
	eventHooks["cancel_requested"] = deliverCancel
}

func deliverSignal(e *env, ev *durablev1.HistoryEvent) {
	s := ev.GetSignal()
	f := &future{}
	f.resolve(append([]byte(nil), s.GetInput()...), nil) // keep the payload: who approved
	// The n-th signal of this name, in arrival order.
	e.futures[sigKey(s.GetSignalName(), signalCount(e, s.GetSignalName())+1)] = f
}

func (c ReceiveChannel) ReceiveAsync() ([]byte, bool) {
	n := c.consumed()
	f := c.e.futures[sigKey(c.name, n+1)] // the oldest not yet consumed
	if f == nil {
		return nil, false
	}
	done := &future{}
	done.resolve([]byte(strconv.Itoa(n+1)), nil)
	c.e.futures["sc:"+c.name] = done // advance the consumed count, or every wait returns the same signal
	return f.value, true
}

func Await(ctx Context, cond func() bool) {
	for !cond() {
		ctx.e.block() // re-checked after every batch of delivered events
	}
}
```

The consumed count is workflow state, not history. It is rebuilt on replay because the same code consumes the same signals in the same order. `NewTimer` and `Timer.Cancel` issue `StartTimer` and `CancelTimer` with sequence ids, giving `AwaitSignal` its timeout. `ExecuteActivityID` refuses an empty id, a numeric one (those belong to `ExecuteActivity`'s numbering), or a repeated one, with a non-retryable `BadActivityID` failure. `CancelRun` emits `CancelWorkflowExecution` and then blocks forever; the run ends when the task completes.

### The Runtime adapter

The adapter in [`runtime.go`](../../../../course/solutions/go/workflows/runtime.go) makes every blocking call wait for "done or cancelled", unless the runtime is detached (Listing 6).

```go
func (s *sdk) canceled() bool {
	return !s.detached && workflow.CancelRequested(s.ctx) // Detached() ignores the cancel
}

func (s *sdk) ExecuteActivity(name string, in any, o StepOptions, out any) error {
	if s.canceled() {
		return fmt.Errorf("activity %s: %w", name, ErrCanceled) // do not start new work after a cancel
	}
	// ... options; ExecuteActivityID when o.ID is set, so the key is "<workflow_id>/<ID>"
	// The workflow stops waiting at the cancel; the activity hears it on its heartbeat.
	workflow.Await(s.ctx, func() bool { return f.Ready() || s.canceled() })
	if !f.Ready() {
		return fmt.Errorf("activity %s: %w", name, ErrCanceled)
	}
	raw, err := f.Get(s.ctx)
	if err != nil {
		var ae *workflow.ActivityError
		if errors.As(err, &ae) {
			sf := &StepFailure{Activity: name, Type: ae.Failure.GetType(), Message: ae.Failure.GetMessage(), NonRetryable: ae.Failure.GetNonRetryable()}
			if sf.Type == "Canceled" && !s.detached {
				return fmt.Errorf("%w: %w", sf, ErrCanceled) // both matchable with errors.Is/As
			}
			return sf
		}
		return err
	}
	// ... decode JSON into out
}
```

The other calls follow the same shape. `Sleep` creates a timer and waits for "fired or cancelled", cancelling the timer if it did not fire. `AwaitSignal` checks the named channels first (a buffered signal returns at once, with no timer); otherwise it starts a timer when a timeout is given, waits for any channel, the timer, or a cancel, and cancels the timer when a signal wins. `Workflow[I, O]` decodes JSON input, runs the function on `FromContext(ctx)`, and if the error matches `ErrCanceled` calls `workflow.CancelRun`, so the run ends `CANCELED` rather than `FAILED`.

### The saga

Finally, [`saga.go`](../../../../course/solutions/go/workflows/saga.go) implements the four saga rules in about thirty lines (Listing 7).

```go
func (s *Saga) Add(name, activity string, in any, opts StepOptions) {
	if opts.ID == "" {
		opts.ID = fmt.Sprintf("compensate-%d", len(s.steps)+1) // stable: same registration order on replay
	}
	s.steps = append(s.steps, Compensation{Name: name, Activity: activity, Input: in, Options: opts})
}

func (s *Saga) Compensate(rt Runtime) error {
	if s.done {
		return nil // failure path and cancel path may both call it
	}
	s.done = true
	d := rt.Detached() // a cancel must not interrupt the undo
	var errs []error
	for i := len(s.steps) - 1; i >= 0; i-- { // newest first
		c := s.steps[i]
		if err := d.ExecuteActivity(c.Activity, c.Input, c.Options, nil); err != nil {
			errs = append(errs, fmt.Errorf("compensation %q: %w", c.Name, err)) // keep going
		}
	}
	return errors.Join(errs...)
}

func (s *Saga) Fail(rt Runtime, cause error) error {
	if err := s.Compensate(rt); err != nil {
		return errors.Join(cause, err) // cause stays matchable, ErrCanceled included
	}
	return cause
}
```

In the worked example, `Fail` after `job.start` fails runs `compensate-3` (`job.stop`), `compensate-2` (`data.delete`), `compensate-1` (`gpu.release`) with keys `book-1/compensate-3`, `-2`, `-1`, and returns the `QuotaExceeded` failure.

The full solution is in [`signals.go`](../../../../course/solutions/go/durable/server/signals.go), [`signal.go`](../../../../course/solutions/go/durable/workflow/signal.go), [`runtime.go`](../../../../course/solutions/go/workflows/runtime.go), and [`saga.go`](../../../../course/solutions/go/workflows/saga.go).

### Why this design

The costs are modest. A signal or a cancel is one commit, and a cancel adds one event per pending activity. `signalCount` walks the futures for a name, which is O(signals of that name) and small in practice. A saga of $k$ steps costs $k$ compensation activities, run sequentially.

History beats push because pushing a signal to the worker that holds the run would need that worker to be alive, reachable, and to persist the signal; history already has all three properties. A request beats a kill because a cancelled workflow can clean up, record why, and end in a state that says what happened. The cost is that a workflow which ignores cancellation keeps running, but the Runtime's rule that every blocking call fails makes ignoring it hard. Compensations run sequentially: running them in parallel would be faster but would lose the reverse-dependency order, so the reference keeps the order and accepts the latency. One tradeoff is accepted outright: there are no queries. Reading workflow state without a history event is left out, so `describe` shows server-side state only.

## What can go wrong

A correct solution has to handle a long list of cases, each of which some later workflow relies on. The table groups them with the check that exercises them and the downstream reason.

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| hand example: signal while busy | unit | the worked example's events 1 to 7: the signal is event 4, the duplicate appends nothing, a new task carries it | ModelRelease waits for `approve` without losing an early click |
| signal before the first task | unit | a signal before the first poll is in the first task's history, one task scheduled | signals never race the worker |
| signal dedup survives a restart | fault | a request id dedups across a restart; signals without ids never dedup | client retries are harmless |
| signal refusals | boundary | unknown and closed runs `NOT_FOUND`, empty name and over-2-MiB input `INVALID_ARGUMENT` | a late approval cannot vanish silently |
| cancel is idempotent | unit | one request event for three calls; canceled run acknowledges; completed run refuses | scripts can retry a cancel |
| cancel reaches running activities | fault | pending activities get `ActivityTaskCancelRequested`; their heartbeat says `cancel_requested` | the subprocess runner SIGTERMs the training child |
| activity scheduled after the cancel runs | regression | an activity scheduled after the cancel is not told to stop; the run stays open | compensations finish |
| request cancel of one activity | unit | one event per activity even when the command repeats; unknown ids ignored | a workflow can cancel one branch |
| cancel a dead-lettered activity | boundary | a dead-lettered activity is canceled at once | a cancel never waits for ever |
| `CancelWorkflowExecution` needs a request | boundary | refused without a request; with one, `CANCELED` with details, late completions refused | only the operator ends a run as canceled |
| hand example: saga unwinds | unit | the worked saga: order, keys, the original failure | CorpusBuild deletes partial shards in the right order |
| compensate runs once | unit | a second `Compensate` does nothing | failure and cancel paths can both call it |
| compensations run after cancel | fault | after a cancel the compensations still run, on `Detached()` | cancel leaves nothing half done |
| failed compensation does not stop the others | unit | later compensations run; the error carries cause and failure | one stuck undo does not leak the rest |
| compensation survives a crash | fault | a crash mid-compensation reruns only the compensation in flight | exactly-once effects through idempotency keys |
| explicit ids are kept | boundary | a named compensation keeps its id; only empty ids are numbered | CorpusBuild keys its cleanup by dataset |
| `IsCanceled` matches wrapped errors | unit | wrapped and joined `ErrCanceled` match; look-alike messages do not | every workflow's cancel check |
| end to end: signal before wait | fault | with a real worker: a signal sent while the workflow is busy is received when it waits | ModelRelease's approval |
| end to end: signal timeout | unit | no signal: the 30 s timer fires on the fake clock and the workflow goes on | approvals expire |
| end to end: cancel mid-activity compensates once | fault | cancel during an activity that never heartbeats: the workflow stops waiting, compensates once, ends `CANCELED` | CorpusBuild cleans up after a cancel |
| end to end: signals in order | unit | two signals sent early are received one per wait, oldest first | queues of approvals |
| `ExecuteActivityID` keys | unit | named ids give the keys `<workflow_id>/<id>`; a repeated id fails at once | CorpusBuild keys stages by name |

The mistakes people actually make cluster around the same rules. On the signal side, the payload is easy to drop, or the signal never reaches the workflow at all, so the approval arrives without saying who approved, or not at all; `SignalReceived` carries the input and `deliverSignal` stores it in a future the workflow reads. A received signal that is never consumed makes every wait return the first approval again, which is why `ReceiveAsync` advances the `sc:<name>` count. And a wait without its timer lets an approval nobody gives block the release forever; `AwaitSignal` starts a durable timer whenever a timeout is given.

> **Warning:** Request-id dedup kept only in memory works until the first restart. A retried "approve" then approves twice. Keep the ids in run state rebuilt by `apply`.

Two more signal pitfalls are about refusing loudly. Accepting a signal to a closed run makes an approval for a finished release disappear, so `SignalWorkflow` returns `NOT_FOUND`. Accepting a signal with no payload limit lets one signal make the history unreadable, so input over `MaxPayloadBytes` is `INVALID_ARGUMENT`.

On the cancel side, a non-idempotent cancel turns three calls into three events and three compensations; the rpc returns early when the run is already `CANCELED` or `cancelRequested`. If pending activities are never told, the training run keeps going after the cancel, and if the workflow is never told, it waits for an activity that will never answer; the server calls `requestCancel` for every pending activity, and the Runtime's `Await` also returns on `canceled()`. A broken or repeated cancel by id stops the wrong branch or fills the history with duplicates, which matching by activity id plus the `cancelRequested` and `pendingIn` checks prevent. A dead-lettered activity that is only "asked" to cancel leaves the workflow waiting for an answer that never comes, so the server appends `ActivityTaskCanceled` right away. And a loose or missing `CancelWorkflowExecution` lets a workflow cancel itself, loses the reason, or ends a cancelled run as FAILED; `cancelWorkflow` requires `cancelRequested` and keeps the details, and `Workflow[I, O]` calls `CancelRun`.

The saga pitfalls follow its four rules. Compensating oldest first deletes the data while the job still reads it; `Compensate` iterates from the last step down. Unstable activity keys make a replay run a compensation or a stage under a new key, which means twice; `Add` assigns `compensate-<n>` only to empty ids and `ExecuteActivityID` keys by the chosen id. A non-idempotent `Compensate` undoes twice when both the failure and the cancel path call it; the `done` flag stops that.

> **Warning:** Compensating on the cancelled Runtime fails every compensation with `ErrCanceled`, and nothing is undone. `Compensate` must run on `rt.Detached()`.

Stopping at the first failed compensation leaves the GPU reserved because the copy could not be deleted; the loop collects errors and continues. Swallowing the cause makes a failed build report success, or only the cleanup error; `Fail` returns `errors.Join(cause, err)`. And comparing errors with `==` makes a wrapped `ErrCanceled` look like a failure, which is why `IsCanceled` uses `errors.Is`.

## Where this leads

This section reuses a good deal of what came before: `signals.go` extends the server's `batch`, `commit`, `registerCommand`, and `RecordHeartbeat` from [the workflow service](02-workflow-service.md), [the task protocol](04-task-protocol-and-workers.md), and [activities](05-activities.md); `signal.go` plugs into the replay engine's `eventHooks` and `FromContext` adapts its `Context` to `Runtime` ([Deterministic replay](06-deterministic-replay.md)); and `AwaitSignal`'s timeout is a [durable timer](07-durable-timers.md).

Going forward, CorpusBuild registers its cleanup with `Saga` before the shard stage ([CorpusBuild: the pipeline as a durable workflow](../../data-engineering/05-corpus-pipeline/09-corpusbuild-as-a-durable-workflow.md)). TrainRun ends on `ErrCanceled` after the train activity checkpointed ([Platform workflows: TrainRun and EvalSuite](11-trainrun-and-evalsuite.md)). ModelRelease waits for the `approve` signal through the Runtime ([ModelRelease](12-model-release.md)). Later, agent approvals are signals with request ids ([Durable agent runs](../../ai-engineering/28-agents/05-durable-agent-runs.md)).

Production engines go further in each direction. Temporal's signals have a sibling, Updates: a validated, synchronous mutation that returns a result to the caller (Temporal docs, "Workflow message passing"). Temporal cancellation scopes and parent-close policies let nested scopes cancel parts of a workflow and children follow their parent (Go SDK `workflow.WithCancel`, `NewDisconnectedContext`). Temporal's saga samples and AWS Step Functions' catch-and-compensate run compensations in parallel and retry each with its own policy (`temporalio/samples-go/saga`). And the queries left out here exist as `QueryWorkflow`, which reads a workflow's state without a history event (Temporal docs, "Queries").

### Further reading

- Hector Garcia-Molina and Kenneth Salem, ["Sagas"](https://www.cs.cornell.edu/andru/cs711/2002fa/reading/sagas.pdf), SIGMOD 1987.
- [Temporal: signals and cancellation](https://docs.temporal.io/develop/go/cancellation).
- [microservices.io: saga pattern](https://microservices.io/patterns/data/saga.html).

## Key terms

- **Signal**: an external input to a running workflow, recorded as a `SignalReceived` event in its history and delivered by replay.
- **Request id**: a caller-chosen id that makes a signal idempotent; stored in run state rebuilt from history.
- **Cancel request**: a `WorkflowExecutionCancelRequested` event that asks a workflow to stop; the workflow, not the server, ends the run.
- **Detached runtime**: a view of the Runtime, from `Detached()`, whose calls ignore the cancel request so cleanup can run.
- **Saga**: a sequence of forward steps each paired with a compensation, undone in reverse order on failure or cancel.
- **Compensation**: an idempotent activity that semantically undoes a forward step.

## Summary

- A signal is an event appended to the run's history, never a message pushed to a worker, so a signal sent before the workflow waits, or while it is busy, cannot be lost: the next workflow task delivers it.
- Signals carry an optional request id; a retried signal is acknowledged and appended once, also across a server restart, because the ids are rebuilt from history.
- A cancel is a request, not a kill: the server records it, tells every activity pending at that moment by name (their heartbeats answer `cancel_requested`), and the workflow decides how to end, usually by compensating and then `CancelWorkflowExecution`.
- A saga undoes completed steps in reverse order, each compensation once, each an activity with a stable key, on a detached runtime that the cancel does not interrupt; one failed compensation does not stop the others.
- Compensations scheduled after the cancel run to the end: only the activities pending when the cancel arrived are told to stop.

## Review questions

1. A signal arrives while the workflow's current task is running on a worker. Why is it not lost, and when does the workflow see it?

   <details><summary>Answer</summary>

   It is appended to history as `SignalReceived`. Because a task is in flight, `apply` sets `needsWT`; when that task completes, its batch schedules another workflow task, whose history contains the signal. In the worked example that is event 7, the `wt_scheduled` after the completion.

   </details>

2. Why does `CancelWorkflow` tell only activities pending at the moment of the request?

   <details><summary>Answer</summary>

   The workflow's compensations are activities scheduled after the cancel. If they inherited the cancel request, their heartbeats would answer `cancel_requested` and the undo would be interrupted. Only the work that was running when the operator asked to stop should stop.

   </details>

3. Why do compensations run on `rt.Detached()`?

   <details><summary>Answer</summary>

   After a cancel request, every blocking call of the normal Runtime returns `ErrCanceled` immediately. Compensating on it would fail every compensation and undo nothing. `Detached()` ignores the cancel request.

   </details>

4. A worker crashes while `data.delete` (the second compensation) runs. What happens on the next task?

   <details><summary>Answer</summary>

   Replay re-runs the workflow; `job.stop` (`compensate-3`) is answered from its `ActivityTaskCompleted` in history, and `data.delete` (`compensate-2`) is still pending, so its activity task is redelivered with the same idempotency key `book-1/compensate-2`. The storage service deduplicates it, and the saga continues with `gpu.release`.

   </details>

5. Why must a workflow return an error wrapping `ErrCanceled` rather than any error after a cancel?

   <details><summary>Answer</summary>

   `Workflow[I, O]` checks `IsCanceled(err)` with `errors.Is`; only then does it issue `CancelWorkflowExecution`, ending the run `CANCELED` with the reason. Any other error ends the run `FAILED`, which misreports an operator's decision as a fault.

   </details>

6. Why does a dead-lettered activity get `ActivityTaskCanceled` immediately instead of only `ActivityTaskCancelRequested`?

   <details><summary>Answer</summary>

   A cancel request is heard on a heartbeat, and a dead-lettered activity has no running attempt to heartbeat. Without the immediate `ActivityTaskCanceled`, the workflow would wait for an answer that never comes.

   </details>
