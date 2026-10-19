# Signals, cancellation, sagas (compensation)

## Overview

| | |
|---|---|
| **Module** | `dur.08` · build · Go · Pass 8 · 5 to 7 h |
| **Solution** | [`go/durable/server/signals.go`](../../../../course/solutions/go/durable/server/signals.go): the `SignalWorkflow` and `CancelWorkflow` rpcs and the `RequestCancelActivity` and `CancelWorkflowExecution` commands; [`go/durable/workflow/signal.go`](../../../../course/solutions/go/durable/workflow/signal.go): the SDK half (signal channels, `Await`, `CancelRequested`, cancellable timers, `ExecuteActivityID`, `CancelRun`); [`go/workflows/runtime.go`](../../../../course/solutions/go/workflows/runtime.go): the `Runtime` seam every platform workflow is written against, its adapter `FromContext`, the `Workflow` wrapper, `ErrCanceled`; [`go/workflows/saga.go`](../../../../course/solutions/go/workflows/saga.go): `Saga` with `Add`, `Compensate`, `Fail` |
| **Contract** | [`course/contracts/proto/tl/durable/v1/durable.proto`](../../../../course/contracts/proto/tl/durable/v1/durable.proto) (the rpcs and commands marked `[dur.08]`, the events `SignalReceived`, `WorkflowExecutionCancelRequested`, `ActivityTaskCancelRequested`, `ActivityTaskCanceled`, `WorkflowExecutionCanceled`) |
| **Needs** | `dur.02` server, `dur.04` task protocol and worker, `dur.05` activity failures (`signals.go` extends their package), `dur.06` the replay engine (`signal.go` plugs into its `eventHooks`), `dur.07` durable timers (signal timeouts); reading: `lang.06` Go |
| **Used by** | `data.09` (CorpusBuild compensates with a `Saga`) · `dur.11` (TrainRun ends on `ErrCanceled`) · `dur.12` (ModelRelease waits for the `approve` signal through the Runtime); later `ag.05` (approvals); with this module an operator can approve or cancel a running workflow |
| **Optional depth** | Hector Garcia-Molina and Kenneth Salem, ["Sagas"](https://www.cs.cornell.edu/andru/cs711/2002fa/reading/sagas.pdf) (SIGMOD 1987, free); [Temporal: signals and cancellation](https://docs.temporal.io/develop/go/cancellation) (free); [microservices.io: saga pattern](https://microservices.io/patterns/data/saga.html) (free) |

## Key Takeaways

- A **signal** is an event appended to the run's history, never a message pushed to a worker, so a signal sent before the workflow waits, or while it is busy, cannot be lost: the next workflow task delivers it.
- Signals carry an optional **request id**: a retried signal is acknowledged and appended once, also across a server restart, because the ids are rebuilt from history.
- A **cancel** is a request, not a kill: the server records it, tells every activity pending at that moment by name (their heartbeats answer `cancel_requested`), and the workflow decides how to end, usually by compensating and then `CancelWorkflowExecution`.
- A **saga** undoes completed steps in **reverse order**, each compensation once, each an activity with a **stable key**, on a **detached** runtime that the cancel does not interrupt; one failed compensation does not stop the others.
- Compensations scheduled after the cancel must **run to the end**: only the activities pending when the cancel arrived are told to stop.

## How to read this chapter

Sections 1 to 3 derive signals, cancellation, and sagas from first principles and trace one of each by hand. Section 4 fixes the interface. Section 5 walks through the server half (`signals.go`), the SDK hooks (`signal.go`), the `Runtime` adapter, and the `Saga`, explaining why each choice is the right one. An operator reaches the two RPCs (`SignalWorkflow`, `CancelWorkflow`) through a client such as the system's control CLI. Try to predict each step before reading it.

---

## 1. Why now

The engine can run a workflow to the end and survive any crash on the way, but nobody outside can talk to a workflow while it runs. A model release needs a human to say "approve" in the middle of it (`dur.12`); a corpus build started with the wrong config needs to be stopped, and stopping it halfway leaves shards that look like a finished corpus (`data.09`); a training run needs to be cancelled without throwing away its checkpoints (`dur.11`). A process-level kill does none of that safely: the run would just be redelivered. This module adds the three ways the outside world steers a running workflow, signals, cancellation, and the compensation that undoes work cleanly, all with the same rule as everything else in the engine: it is history first, effects second.

## 2. Principles

### 2.1 Signals are history

`SignalWorkflow(workflow_id, signal_name, input, request_id)` appends `SignalReceived` to the run and returns. Appending it is news the workflow must react to, so the server's commit schedules a workflow task, or, when one is already in flight, marks the run to get another one right after (`needsWT`). The workflow receives signals by replaying its history: a signal that arrived before the workflow reached `AwaitSignal`, or while a workflow task was running, is simply an earlier event in the same history. A signal to an unknown or closed run is `NOT_FOUND`; its input obeys the 2 MiB payload limit.

A **request id** makes a signal idempotent: `apply` records each id in the run's state as it folds `SignalReceived` events, so a second signal with the same id is acknowledged and appends nothing, including after a restart, when the state is rebuilt from the log.

### 2.2 Cancellation is a request

| Step | Who | What is appended |
|---|---|---|
| `CancelWorkflow` | operator, CLI | `WorkflowExecutionCancelRequested`, and `ActivityTaskCancelRequested` for every activity pending right now (plus `ActivityTaskCanceled` for a dead-lettered one, which no attempt can acknowledge) |
| next heartbeat of a pending activity | worker | nothing: the answer says `cancel_requested`, the worker cancels the attempt's context, the subprocess runner sends SIGTERM (`dur.09`) |
| next workflow task | workflow | every blocking `Runtime` call returns an error wrapping `ErrCanceled`; the workflow compensates |
| `CancelWorkflowExecution` command | workflow | `WorkflowExecutionCanceled`: the run is `CANCELED` |

Cancel is idempotent: a second request, or one for a run that already ended canceled, appends nothing; a run that completed or failed cannot be cancelled (`FAILED_PRECONDITION`). `CancelWorkflowExecution` without a request is refused: a workflow ends itself by completing or failing. `RequestCancelActivity` cancels one activity by id (a race, a deadline the workflow enforces), once.

### 2.3 Sagas

A workflow that changes the world in several steps cannot roll them back in one transaction. A **saga** (Garcia-Molina and Salem, 1987) pairs each forward step $T_i$ with a **compensation** $C_i$ that semantically undoes it, and on failure after $T_k$ runs $C_k, C_{k-1}, \dots, C_1$.

| Symbol | Meaning |
|---|---|
| $T_i$ | the $i$-th forward step (an activity) |
| $C_i$ | its compensation (an activity that undoes $T_i$, idempotent) |
| $k$ | the last step that started |

Four rules make it safe in a durable engine:

1. **Register before the step** whose partial effects need undoing: a step interrupted halfway may have left something, and its compensation must run too.
2. **Reverse order**: $C_k$ first, because later steps build on earlier ones (stop the job before deleting the data it reads).
3. **Once, with stable keys**: a compensation is an activity whose id is `compensate-<n>` (or one you name), so on replay after a crash it is answered from history or redelivered with the same idempotency key; `Compensate` called again in the same run does nothing.
4. **Detached and complete**: compensations run on `rt.Detached()`, which a cancel does not interrupt, and a failed compensation does not stop the others; the errors are joined to the original cause.

### 2.4 The SDK half: hooks into the replay engine

The replay engine of `dur.06` resolves activity and timer events itself and hands every other event to `eventHooks`. `signal.go` registers two: `SignalReceived` stores the payload as the run's next signal of that name, and `WorkflowExecutionCancelRequested` marks the run cancelled. Both keep their state where the engine keeps all per-run state, in the run's futures, so a replay rebuilds it from history:

| Future key | Holds |
|---|---|
| `s:<name>:<n>` | the $n$-th signal named `<name>` |
| `sc:<name>` | how many of them the workflow consumed |
| `c:` | resolved once the run has a cancel request |

`Await(ctx, cond)` blocks the workflow coroutine until `cond` holds, re-checking after every batch of events, which is how `Receive` waits for a signal and the adapter waits for "the activity finished, or the run was cancelled". A durable timer that can be cancelled (`NewTimer`, `Cancel`) gives `AwaitSignal` its timeout. `ExecuteActivityID` schedules an activity under a chosen id, so the idempotency key is `<workflow_id>/<id>`. `CancelRun` issues `CancelWorkflowExecution` and parks the coroutine for good: the run is over when the task completes.

### 2.5 The Runtime seam

The platform workflows (`CorpusBuild`, `TrainRun`, `EvalSuite`, `ModelRelease`) are plain Go functions of a `Runtime`, the slice of the SDK they need: `ExecuteActivity`, `Sleep`, `Now`, `AwaitSignal`, `Detached`. The worker's composition root adapts a `workflow.Context` (`dur.06`) to it when it registers them. The seam is what lets workflow logic be tested with a replaying simulator, without a server, and it states the cancel rule once: after a cancel request, every blocking call of the Runtime fails with `ErrCanceled`, and no call of `Detached()` does.

## 3. Worked example by hand

**A signal while the workflow is busy.** Workflow `release-7` starts (events 1 `started`, 2 `wt_scheduled`); a worker polls (3 `wt_started`). While it works, the approver calls `SignalWorkflow(release-7, approve, {"by":"ana"}, request r1)`: the server appends 4 `signal` and, since a workflow task is in flight, only marks the run. The approver retries the same call: `r1` is already in the run's signal ids, so nothing is appended. The worker completes its task with `ScheduleActivity(1, prepare)`: 5 `wt_completed`, 6 `act_scheduled`, and because the run was marked, 7 `wt_scheduled`. The next task's history ends `... signal wt_completed act_scheduled wt_scheduled wt_started`: the workflow replays the signal and proceeds.

**A saga that unwinds.** A booking workflow registers $C_1$ = `gpu.release`, runs $T_1$ = `gpu.reserve`; registers $C_2$ = `data.delete`, runs $T_2$ = `data.copy`; registers $C_3$ = `job.stop`, runs $T_3$ = `job.start`, which fails for good (`QuotaExceeded`). `Fail` runs `job.stop`, `data.delete`, `gpu.release`, in that order, with keys `book-1/compensate-3`, `-2`, `-1`, and returns the `QuotaExceeded` failure. The execution order is `gpu.reserve data.copy job.start job.stop data.delete gpu.release`. If the worker dies while `data.delete` runs, the replay answers `job.stop` from history and runs `data.delete` again with the same key: the storage service deduplicates it.

## 4. The interface

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

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| hand example: signal while busy | unit | section 3's events 1 to 7: the signal is event 4, the duplicate appends nothing, a new task carries it | `dur.12` waits for `approve` without losing an early click |
| signal before the first task | unit | a signal before the first poll is in the first task's history, one task scheduled | signals never race the worker |
| signal dedup survives a restart | fault | a request id dedups across a restart; signals without ids never dedup | client retries are harmless |
| signal refusals | boundary | unknown and closed runs `NOT_FOUND`, empty name and over-2-MiB input `INVALID_ARGUMENT` | a late approval cannot vanish silently |
| cancel is idempotent | unit | one request event for three calls; canceled run acknowledges; completed run refuses | scripts can retry a cancel |
| cancel reaches running activities | fault | pending activities get `ActivityTaskCancelRequested`; their heartbeat says `cancel_requested` | `dur.09` SIGTERMs the training child |
| activity scheduled after the cancel runs | regression | an activity scheduled after the cancel is not told to stop; the run stays open | compensations finish |
| request cancel of one activity | unit | one event per activity even when the command repeats; unknown ids ignored | a workflow can cancel one branch |
| cancel a dead-lettered activity | boundary | a dead-lettered activity is canceled at once | a cancel never waits for ever |
| `CancelWorkflowExecution` needs a request | boundary | refused without a request; with one, `CANCELED` with details, late completions refused | only the operator ends a run as canceled |
| hand example: saga unwinds | unit | section 3's saga: order, keys, the original failure | `data.09` deletes partial shards in the right order |
| compensate runs once | unit | a second `Compensate` does nothing | failure and cancel paths can both call it |
| compensations run after cancel | fault | after a cancel the compensations still run, on `Detached()` | cancel leaves nothing half done |
| failed compensation does not stop the others | unit | later compensations run; the error carries cause and failure | one stuck undo does not leak the rest |
| compensation survives a crash | fault | a crash mid-compensation reruns only the compensation in flight | exactly-once effects through idempotency keys |
| explicit ids are kept | boundary | a named compensation keeps its id; only empty ids are numbered | `data.09` keys its cleanup by dataset |
| `IsCanceled` matches wrapped errors | unit | wrapped and joined `ErrCanceled` match; look-alike messages do not | every workflow's cancel check |
| end to end: signal before wait | fault | with a real worker: a signal sent while the workflow is busy is received when it waits | `dur.12`'s approval |
| end to end: signal timeout | unit | no signal: the 30 s timer fires on the fake clock and the workflow goes on | approvals expire |
| end to end: cancel mid-activity compensates once | fault | cancel during an activity that never heartbeats: the workflow stops waiting, compensates once, ends `CANCELED` | `data.09` cleans up after a cancel |
| end to end: signals in order | unit | two signals sent early are received one per wait, oldest first | queues of approvals |
| `ExecuteActivityID` keys | unit | named ids give the keys `<workflow_id>/<id>`; a repeated id fails at once | `data.09` keys stages by name |

## 5. The solution, step by step

### From first principles

1. **Anything the workflow reacts to must be in history, or replay cannot reproduce the reaction.** So a signal and a cancel request are both events appended to the run, not messages to a worker. That alone makes them impossible to lose: whichever workflow task runs next sees them.
2. **Appending must wake the workflow.** Both events are triggers, so the existing `commit` rule (`dur.02`) schedules a workflow task or sets `needsWT` when one is in flight. No new delivery path is needed.
3. **Retries of an external call must not double its effect.** A client retrying "approve" must not approve twice. A request id stored in run state (rebuilt by `apply`) turns the signal into an idempotent append.
4. **Cancellation cannot be a kill.** Killing would lose the chance to undo; the run would also just be redelivered. So it is a request that the workflow honours: activities pending at that moment are told by name, every blocking call starts returning `ErrCanceled`, and only the workflow may end the run as `CANCELED`, after cleaning up.
5. **Cleanup must not itself be cancelled.** Compensations scheduled after the cancel are new activities; they must not receive the stop request, and the Runtime they run on must not fail with `ErrCanceled`. Hence "only activities pending at cancel time" on the server and `Detached()` in the SDK.
6. **Undo in reverse, once, with stable identities.** Later steps depend on earlier ones, so compensations run newest first. Each is an activity with a deterministic id so replay answers finished ones from history and reruns the one in flight with the same idempotency key. A flag makes `Compensate` idempotent within a run.

### The code, annotated

**`SignalWorkflow`** ([`signals.go`](../../../../course/solutions/go/durable/server/signals.go)).

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

`r.signals` is filled by `apply` from every `SignalReceived` with a request id (`dur.02`), so dedup survives a restart with no extra storage.

**`CancelWorkflow`: record once, tell what is pending now.**

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

The running attempt learns of the request from `RecordHeartbeat`, which answers `cancel_requested: a.cancelRequested` (`dur.05`): per activity, not per run, which is exactly what lets compensations run to the end. `cancelActivity` (the `RequestCancelActivity` command) finds the pending activity by id and calls the same `requestCancel`; an unknown or finished id is ignored, because cancellation races completion. `cancelWorkflow` (the `CancelWorkflowExecution` command) is refused with `FAILED_PRECONDITION` unless `r.cancelRequested`, then appends `WorkflowExecutionCanceled{details}`.

**The SDK hooks** ([`signal.go`](../../../../course/solutions/go/durable/workflow/signal.go)). State lives in the engine's futures map so a replay rebuilds it.

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

The consumed count is workflow state, not history: it is rebuilt on replay because the same code consumes the same signals in the same order. `NewTimer` and `Timer.Cancel` issue `StartTimer` and `CancelTimer` with sequence ids, giving `AwaitSignal` its timeout. `ExecuteActivityID` refuses an empty, numeric (those belong to `ExecuteActivity`'s numbering), or repeated id with a non-retryable `BadActivityID` failure. `CancelRun` emits `CancelWorkflowExecution` and then blocks forever: the run ends when the task completes.

**The Runtime adapter** ([`runtime.go`](../../../../course/solutions/go/workflows/runtime.go)). Every blocking call waits for "done or cancelled", unless detached.

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

`Sleep` creates a timer and waits for "fired or cancelled", cancelling the timer if it did not fire. `AwaitSignal` checks the named channels first (a buffered signal returns at once, no timer), else starts a timer when a timeout is given, waits for any channel, the timer, or a cancel, and cancels the timer when a signal wins. `Workflow[I, O]` decodes JSON input, runs the function on `FromContext(ctx)`, and if the error matches `ErrCanceled` calls `workflow.CancelRun`, so the run ends `CANCELED` rather than `FAILED`.

**The saga** ([`saga.go`](../../../../course/solutions/go/workflows/saga.go)).

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

Full solution: [`signals.go`](../../../../course/solutions/go/durable/server/signals.go), [`signal.go`](../../../../course/solutions/go/durable/workflow/signal.go), [`runtime.go`](../../../../course/solutions/go/workflows/runtime.go), [`saga.go`](../../../../course/solutions/go/workflows/saga.go).

### Why this solution

- **Cost.** A signal or cancel is one commit. Cancel adds one event per pending activity. `signalCount` walks the futures for a name, O(signals of that name), which is small in practice. A saga of $k$ steps costs $k$ compensation activities, run sequentially.
- **History over push.** Pushing a signal to the worker that holds the run would need that worker to be alive, reachable, and to persist it; history already has all three properties.
- **Request over kill.** A cancelled workflow can clean up, record why, and end in a state that says what happened. The cost is that a workflow which ignores cancellation keeps running; the Runtime's rule (every blocking call fails) makes ignoring it hard.
- **Sequential compensations.** Running them in parallel is faster but loses the reverse-dependency order; the reference keeps order and accepts the latency.
- **Accepted tradeoff.** No queries: reading workflow state without a history event is left out, so `describe` shows server-side state only.

### Explain it back

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

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. the signal's payload is dropped, or the signal never reaches the workflow | the approval arrives without who approved, or not at all | `SignalReceived` carries the input; `deliverSignal` stores it in a future the workflow reads |
| 2. no request-id dedup, or dedup only in memory | a retried "approve" approves twice; after a restart it does | request ids live in run state rebuilt by `apply` |
| 3. signals to closed runs accepted | an approval for a finished release disappears | `SignalWorkflow` returns `NOT_FOUND` for a closed run |
| 4. no payload limit | one signal makes the history unreadable | input over `MaxPayloadBytes` is `INVALID_ARGUMENT` |
| 5. cancel not idempotent | three cancels, three events, three compensations | returns early when already `CANCELED` or `cancelRequested` |
| 6. pending activities never told, or the workflow never told | the training run keeps going after the cancel; the workflow waits for an activity that will never answer | `requestCancel` for every pending activity; the Runtime's `Await` also returns on `canceled()` |
| 7. activity cancel by id broken or repeated | the wrong branch stops, or the history fills with duplicates | matches by activity id; `cancelRequested` and `pendingIn` make it once |
| 8. a dead-lettered activity only "asked" to cancel | the workflow waits for an answer that never comes | appends `ActivityTaskCanceled` right away for a dead-lettered activity |
| 9. `CancelWorkflowExecution` loose, or never sent | a workflow cancels itself; the reason is lost; a cancelled run ends FAILED | `cancelWorkflow` requires `cancelRequested` and keeps the details; `Workflow[I, O]` calls `CancelRun` |
| 10. compensations oldest first | the data is deleted while the job still reads it | `Compensate` iterates from the last step down |
| 11. unstable activity keys | a replay runs a compensation or a stage under a new key: twice | `Add` assigns `compensate-<n>` only to empty ids; `ExecuteActivityID` keys by the chosen id |
| 12. `Compensate` not idempotent | failure and cancel paths both undo | the `done` flag |
| 13. compensating on the cancelled Runtime | every compensation fails with `ErrCanceled`; nothing is undone | `Compensate` runs on `rt.Detached()` |
| 14. the cause swallowed | a failed build reports success, or only the cleanup error | `Fail` returns `errors.Join(cause, err)` |
| 15. stopping at the first failed compensation | the GPU stays reserved because the copy could not be deleted | the loop collects errors and continues |
| 16. comparing errors with `==` | a wrapped `ErrCanceled` looks like a failure | `IsCanceled` uses `errors.Is` |
| 17. a received signal not consumed | every wait returns the first approval again | `ReceiveAsync` advances the `sc:<name>` count |
| 18. a wait without its timer | an approval nobody gives blocks the release for ever | `AwaitSignal` starts a durable timer for a timeout |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `dur.02`, `dur.04`, `dur.05` | `signals.go` extends their server: `batch`, `commit`, `registerCommand`, `RecordHeartbeat` |
| Back | `dur.06` | `signal.go` plugs into the replay engine's `eventHooks`; `FromContext` adapts its `Context` to `Runtime` |
| Back | `dur.07` | `AwaitSignal`'s timeout is a durable timer |
| Forward | `data.09` | CorpusBuild registers its cleanup with `Saga` before the shard stage |
| Forward | `dur.11` | TrainRun ends on `ErrCanceled` after the train activity checkpointed |
| Forward | `dur.12` | ModelRelease waits for the `approve` signal |
| Forward | `ag.05` | agent approvals are signals with request ids |

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `SignalWorkflow` | Temporal signals and Updates | Updates: a validated, synchronous mutation that returns a result to the caller | Temporal docs, "Workflow message passing" |
| `CancelWorkflow` | Temporal cancellation scopes and parent-close policies | nested scopes cancel parts of a workflow; children follow their parent | Temporal Go SDK `workflow.WithCancel`, `NewDisconnectedContext` |
| `Saga` | Temporal saga samples, AWS Step Functions catch and compensate | compensations in parallel; retries per compensation with their own policies | `temporalio/samples-go/saga` |
| queries (left out of v1) | `QueryWorkflow` | read a workflow's state without a history event | Temporal docs, "Queries" |
