<!-- ol:module dur.08 -->
# Signals, cancellation, sagas (compensation)

## Overview

| | |
|---|---|
| **Module** | `dur.08` · build · Go · Pass 8 · 5 to 7 h |
| **You build** | `go/durable/server/signals.go`: the `SignalWorkflow` and `CancelWorkflow` rpcs and the `RequestCancelActivity` and `CancelWorkflowExecution` commands; `go/durable/workflow/signal.go`: the SDK half (signal channels, `Await`, `CancelRequested`, cancellable timers, `ExecuteActivityID`, `CancelRun`); `go/workflows/runtime.go`: the `Runtime` seam every platform workflow is written against, its adapter `FromContext`, the `Workflow` wrapper, `ErrCanceled`; `go/workflows/saga.go`: `Saga` with `Add`, `Compensate`, `Fail` |
| **Contract** | [`course/contracts/proto/tl/durable/v1/durable.proto`](../../course/contracts/proto/tl/durable/v1/durable.proto) (the rpcs and commands marked `[dur.08]`, the events `SignalReceived`, `WorkflowExecutionCancelRequested`, `ActivityTaskCancelRequested`, `ActivityTaskCanceled`, `WorkflowExecutionCanceled`) |
| **Tests** | `course/tests/go/dur_08/` (the server half over gRPC, sagas through a replaying simulator, and end to end with a real worker; section 4) |
| **Needs** | `dur.02` server, `dur.04` task protocol and worker, `dur.05` activity failures (`signals.go` extends their package), `dur.06` the replay engine (`signal.go` plugs into its `eventHooks`), `dur.07` durable timers (signal timeouts); reading: `lang.06` Go |
| **Used by** | `data.09` (CorpusBuild compensates with a `Saga`) · `dur.11` (TrainRun ends on `ErrCanceled`) · `dur.12` (ModelRelease waits for the `approve` signal through the Runtime); later `ag.05` (approvals) |
| **Milestone** | MS-durable (`<system> wf signal <id> approve`, `<system> wf cancel <id>`) |
| **Optional depth** | Hector Garcia-Molina and Kenneth Salem, ["Sagas"](https://www.cs.cornell.edu/andru/cs711/2002fa/reading/sagas.pdf) (SIGMOD 1987, free); [Temporal: signals and cancellation](https://docs.temporal.io/develop/go/cancellation) (free); [microservices.io: saga pattern](https://microservices.io/patterns/data/saga.html) (free) |

## Key Takeaways

- A **signal** is an event appended to the run's history, never a message pushed to a worker, so a signal sent before the workflow waits, or while it is busy, cannot be lost: the next workflow task delivers it.
- Signals carry an optional **request id**: a retried signal is acknowledged and appended once, also across a server restart, because the ids are rebuilt from history.
- A **cancel** is a request, not a kill: the server records it, tells every activity pending at that moment by name (their heartbeats answer `cancel_requested`), and the workflow decides how to end, usually by compensating and then `CancelWorkflowExecution`.
- A **saga** undoes completed steps in **reverse order**, each compensation once, each an activity with a **stable key**, on a **detached** runtime that the cancel does not interrupt; one failed compensation does not stop the others.
- Compensations scheduled after the cancel must **run to the end**: only the activities pending when the cancel arrived are told to stop.

## How to work this chapter

```bash
ol start dur.08          # stubs signals.go, runtime.go, saga.go
ol tests dur.08          # read the test catalog first
ol check dur.08          # the exit code is the verdict
ol diff  dur.08          # after passing: your code against the reference
```

Then add the verbs to your `{ctl}`: `wf signal <id> <name> [--input JSON] [--request-id ID]` and `wf cancel <id> [--reason TEXT]` call the two rpcs.

---

## 1. Why now

Your engine can run a workflow to the end and survive any crash on the way, but nobody outside can talk to a workflow while it runs. A model release needs a human to say "approve" in the middle of it (`dur.12`); a corpus build started with the wrong config needs to be stopped, and stopping it halfway leaves shards that look like a finished corpus (`data.09`); a training run needs to be cancelled without throwing away its checkpoints (`dur.11`). A process-level kill does none of that safely: the run would just be redelivered. This module adds the three ways the outside world steers a running workflow, signals, cancellation, and the compensation that undoes work cleanly, all with the same rule as everything else in the engine: it is history first, effects second.

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

The platform workflows (`CorpusBuild`, `TrainRun`, `EvalSuite`, `ModelRelease`) are plain Go functions of a `Runtime`, the slice of the SDK they need: `ExecuteActivity`, `Sleep`, `Now`, `AwaitSignal`, `Detached`. Your worker's composition root adapts a `workflow.Context` (`dur.06`) to it when it registers them. The seam is what lets the course test workflow logic with a replaying simulator, and it states the cancel rule once: after a cancel request, every blocking call of the Runtime fails with `ErrCanceled`, and no call of `Detached()` does.

## 3. Worked example by hand

**A signal while the workflow is busy.** Workflow `release-7` starts (events 1 `started`, 2 `wt_scheduled`); a worker polls (3 `wt_started`). While it works, the approver calls `SignalWorkflow(release-7, approve, {"by":"ana"}, request r1)`: the server appends 4 `signal` and, since a workflow task is in flight, only marks the run. The approver retries the same call: `r1` is already in the run's signal ids, so nothing is appended. The worker completes its task with `ScheduleActivity(1, prepare)`: 5 `wt_completed`, 6 `act_scheduled`, and because the run was marked, 7 `wt_scheduled`. The next task's history ends `... signal wt_completed act_scheduled wt_scheduled wt_started`: the workflow replays the signal and proceeds.

**A saga that unwinds.** A booking workflow registers $C_1$ = `gpu.release`, runs $T_1$ = `gpu.reserve`; registers $C_2$ = `data.delete`, runs $T_2$ = `data.copy`; registers $C_3$ = `job.stop`, runs $T_3$ = `job.start`, which fails for good (`QuotaExceeded`). `Fail` runs `job.stop`, `data.delete`, `gpu.release`, in that order, with keys `book-1/compensate-3`, `-2`, `-1`, and returns the `QuotaExceeded` failure. The execution order is `gpu.reserve data.copy job.start job.stop data.delete gpu.release`. If the worker dies while `data.delete` runs, the replay answers `job.stop` from history and runs `data.delete` again with the same key: the storage service deduplicates it.

These are `TestHandExampleSignalWhileBusy` and `TestHandExampleSagaUnwinds`.

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

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `TestHandExampleSignalWhileBusy` | unit | section 3's events 1 to 7: the signal is event 4, the duplicate appends nothing, a new task carries it | `dur.12` waits for `approve` without losing an early click |
| `TestSignalBeforeFirstTaskIsInItsHistory` | unit | a signal before the first poll is in the first task's history, one task scheduled | signals never race the worker |
| `TestSignalDedupSurvivesRestart` | fault | a request id dedups across a restart; signals without ids never dedup | client retries are harmless |
| `TestSignalRefusals` | boundary | unknown and closed runs `NOT_FOUND`, empty name and over-2-MiB input `INVALID_ARGUMENT` | a late approval cannot vanish silently |
| `TestCancelIsIdempotent` | unit | one request event for three calls; canceled run acknowledges; completed run refuses | scripts can retry a cancel |
| `TestCancelReachesRunningActivities` | fault | pending activities get `ActivityTaskCancelRequested`; their heartbeat says `cancel_requested` | `dur.09` SIGTERMs the training child |
| `TestCompensationRunsAfterCancel` | regression | an activity scheduled after the cancel is not told to stop; the run stays open | compensations finish |
| `TestRequestCancelActivity` | unit | one event per activity even when the command repeats; unknown ids ignored | a workflow can cancel one branch |
| `TestCancelDeadLetteredActivity` | boundary | a dead-lettered activity is canceled at once | a cancel never waits for ever |
| `TestCancelWorkflowExecutionNeedsARequest` | boundary | refused without a request; with one, `CANCELED` with details, late completions refused | only the operator ends a run as canceled |
| `TestHandExampleSagaUnwinds` | unit | section 3's saga: order, keys, the original failure | `data.09` deletes partial shards in the right order |
| `TestCompensateRunsOnce` | unit | a second `Compensate` does nothing | failure and cancel paths can both call it |
| `TestCompensationsRunAfterCancel` | fault | after a cancel the compensations still run, on `Detached()` | cancel leaves nothing half done |
| `TestFailedCompensationDoesNotStopTheOthers` | unit | later compensations run; the error carries cause and failure | one stuck undo does not leak the rest |
| `TestCompensationSurvivesACrash` | fault | a crash mid-compensation reruns only the compensation in flight | exactly-once effects through idempotency keys |
| `TestExplicitIDsAreKept` | boundary | a named compensation keeps its id; only empty ids are numbered | `data.09` keys its cleanup by dataset |
| `TestIsCanceledMatchesWrapped` | unit | wrapped and joined `ErrCanceled` match; look-alike messages do not | every workflow's cancel check |
| `TestEndToEndSignalBeforeWait` | fault | with a real worker: a signal sent while the workflow is busy is received when it waits | `dur.12`'s approval |
| `TestEndToEndSignalTimeout` | unit | no signal: the 30 s timer fires on the fake clock and the workflow goes on | approvals expire |
| `TestEndToEndCancelMidActivityCompensatesOnce` | fault | cancel during an activity that never heartbeats: the workflow stops waiting, compensates once, ends `CANCELED` | `data.09` cleans up after a cancel |
| `TestEndToEndSignalsInOrder` | unit | two signals sent early are received one per wait, oldest first | queues of approvals |
| `TestExecuteActivityIDKeys` | unit | named ids give the keys `<workflow_id>/<id>`; a repeated id fails at once | `data.09` keys stages by name |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. the signal's payload is dropped, or the signal never reaches the workflow | the approval arrives without who approved, or not at all | `TestHandExampleSignalWhileBusy`, `TestEndToEndSignalBeforeWait` (mutants `s01`, `s23`) |
| 2. no request-id dedup, or dedup only in memory | a retried "approve" approves twice; after a restart it does | `TestSignalDedupSurvivesRestart` (mutants `s02`, `s03`) |
| 3. signals to closed runs accepted | an approval for a finished release disappears | `TestSignalRefusals` (mutant `s04`) |
| 4. no payload limit | one signal makes the history unreadable | `TestSignalRefusals` (mutant `s05`) |
| 5. cancel not idempotent | three cancels, three events, three compensations | `TestCancelIsIdempotent` (mutants `s06`, `s07`) |
| 6. pending activities never told, or the workflow never told | the training run keeps going after the cancel; the workflow waits for an activity that will never answer | `TestCancelReachesRunningActivities`, `TestEndToEndCancelMidActivityCompensatesOnce` (mutants `s08`, `s26`) |
| 7. activity cancel by id broken or repeated | the wrong branch stops, or the history fills with duplicates | `TestRequestCancelActivity` (mutants `s09`, `s10`) |
| 8. a dead-lettered activity only "asked" to cancel | the workflow waits for an answer that never comes | `TestCancelDeadLetteredActivity` (mutant `s11`) |
| 9. `CancelWorkflowExecution` loose, or never sent | a workflow cancels itself; the reason is lost; a cancelled run ends FAILED | `TestCancelWorkflowExecutionNeedsARequest`, `TestEndToEndCancelMidActivityCompensatesOnce` (mutants `s12`, `s13`, `s28`) |
| 10. compensations oldest first | the data is deleted while the job still reads it | `TestHandExampleSagaUnwinds` (mutant `s14`) |
| 11. unstable activity keys | a replay runs a compensation or a stage under a new key: twice | `TestHandExampleSagaUnwinds`, `TestCompensationSurvivesACrash`, `TestExplicitIDsAreKept`, `TestExecuteActivityIDKeys` (mutants `s15`, `s21`, `s29`) |
| 12. `Compensate` not idempotent | failure and cancel paths both undo | `TestCompensateRunsOnce` (mutant `s16`) |
| 13. compensating on the cancelled Runtime | every compensation fails with `ErrCanceled`; nothing is undone | `TestCompensationsRunAfterCancel`, `TestEndToEndCancelMidActivityCompensatesOnce` (mutants `s17`, `s27`) |
| 14. the cause swallowed | a failed build reports success, or only the cleanup error | `TestCompensationsRunAfterCancel`, `TestFailedCompensationDoesNotStopTheOthers` (mutants `s18`, `s20`) |
| 15. stopping at the first failed compensation | the GPU stays reserved because the copy could not be deleted | `TestFailedCompensationDoesNotStopTheOthers` (mutant `s19`) |
| 16. comparing errors with `==` | a wrapped `ErrCanceled` looks like a failure | `TestIsCanceledMatchesWrapped` (mutant `s22`) |
| 17. a received signal not consumed | every wait returns the first approval again | `TestEndToEndSignalsInOrder` (mutant `s24`) |
| 18. a wait without its timer | an approval nobody gives blocks the release for ever | `TestEndToEndSignalTimeout` (mutant `s25`) |

## 6. Where it's used next

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

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `SignalWorkflow` | Temporal signals and Updates | Updates: a validated, synchronous mutation that returns a result to the caller | Temporal docs, "Workflow message passing" |
| `CancelWorkflow` | Temporal cancellation scopes and parent-close policies | nested scopes cancel parts of a workflow; children follow their parent | Temporal Go SDK `workflow.WithCancel`, `NewDisconnectedContext` |
| `Saga` | Temporal saga samples, AWS Step Functions catch and compensate | compensations in parallel; retries per compensation with their own policies | `temporalio/samples-go/saga` |
| queries (left out of v1) | `QueryWorkflow` | read a workflow's state without a history event | Temporal docs, "Queries" |
