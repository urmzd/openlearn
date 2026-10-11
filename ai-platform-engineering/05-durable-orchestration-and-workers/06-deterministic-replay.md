<!-- ol:module dur.06 -->
# Deterministic replay workflows, ContinueAsNew, history paging, payload limits

## Overview

| | |
|---|---|
| **Module** | `dur.06` · build · Go · Pass 8 · 6 to 8 h |
| **You build** | `go/durable/workflow/workflow.go`: `ExecuteActivity`, `Sleep`, `Now`, `SideEffect`, `GetVersion`, `ContinueAsNew`, `ContinueAsNewSuggested`; `go/durable/workflow/replay.go`: the replay engine, `Registry`, `ReplayHistory`, `LoadHistory`, `ReplayHistoryFromFile`; `go/durable/server/continue.go`: the `record_marker` and `continue_as_new` commands |
| **Contract** | the Go SDK surface of DESIGN 2.7, the `Command` and history messages of [`durable.proto`](../../course/contracts/proto/tl/durable/v1/durable.proto); the package API is section 4 |
| **Tests** | `course/tests/go/dur_06/` (what they check: section 4); recorded runs of the course test workflows in `course/fixtures/dur/histories/` |
| **Needs** | `dur.05` activities and their failures (and through it the whole server) |
| **Used by** | `dur.07` timers and `dur.08` signal/cancel handling extend this workflow SDK |
| **Milestone** | MS-durable |
| **Optional depth** | the Temporal docs on deterministic constraints and versioning; Fateev and Abbas, *Temporal: durable execution* talks; Hellerstein et al. on deterministic replay |

## Key Takeaways

- A workflow never runs once from start to finish. Every workflow task re-runs it from the start against the history: past results come from events, and each command it issues must match the event recorded for it (`TestReplayHandExample`).
- Determinism is the price: no wall clock (`workflow.Now`), no randomness (`SideEffect`), no goroutines or I/O. A code change that reorders commands is caught as `ErrNondeterminism`, never silently continued (`TestNondeterminismDetected`, `TestReorderFailpointNondeterministic`).
- `GetVersion` lets a deploy change code under runs already in flight: old histories replay the old path, new runs take the new one (`TestGetVersionMigration`).
- A long workflow bounds its history with `ContinueAsNew` when the server suggests it, at 10,000 events or 32 MiB (`TestContinueAsNewKeepsHistoryBounded`).
- Kill the server in the middle of a run: the worker replays and the result equals an uninterrupted run's (`TestKillServerMidWorkflowSameResult`).

## How to work this chapter

```bash
ol start dur.06
ol tests dur.06
ol check dur.06          # needs dur.05 (or --ref-deps)
ol diff  dur.06
```

Build the coroutine first (start, block, step) with a workflow that only calls `ExecuteActivity`; pass `TestReplayHandExample`; then the matcher and `ErrNondeterminism`; then markers; then ContinueAsNew.

---

## 1. Why now

Workers run activities, but the logic that decides what to run next (train, then evaluate, then if the loss is good release, else retry with a smaller learning rate) still lives nowhere durable. If you write it as ordinary Go in a worker, a crash in the middle loses its local variables and its place in the code. Durable execution keeps writing ordinary Go and gets crash-proof state anyway, by recording the outcome of every step in history and **replaying** the code against it. This module is the SDK that does it. After it, `CorpusBuild`, `TrainRun`, `ModelRelease`, and `AgentRun` are plain functions that survive any crash.

## 2. Principles

### 2.1 Replay

| Symbol | Meaning |
|---|---|
| $W$ | the workflow: deterministic code from its input and the results of its commands to commands |
| $h$ | the run's history (dur.02) |
| $T_1, \dots, T_m$ | its completed workflow tasks: each `WorkflowTaskCompleted` names the `WorkflowTaskStarted` it answers |
| $C_j$ | the command events recorded right after $T_j$'s `WorkflowTaskCompleted`, in one batch (dur.04) |
| $s_j$ | the event id of $T_j$'s `WorkflowTaskStarted` |
| $s_*$ | the id of the **current** task's `WorkflowTaskStarted`, the last one in $h$ |

To handle the current task:

```
start W as a coroutine
for j in 1..m:                         # past tasks, in order
    deliver every event with id < s_j  # results resolve futures
    run W until it blocks; each command it issues must match the next event of C_j
    if a command differs, or C_j has events left over: ErrNondeterminism
deliver every event with id < s_*
run W until it blocks; the commands it issues now are new: send them
```

The workflow cannot tell a replay from a first run: `Get` on an activity future returns the recorded result if its `ActivityTaskCompleted` was delivered, and otherwise blocks. Because commands are matched in order, the engine detects any change in what the code does at any past point.

**The coroutine.** The workflow runs in its own goroutine, but only one of the two (engine, workflow) runs at a time: the engine resumes it through a channel and waits; the workflow, when it must wait on a future, signals back and waits. So a replay is a deterministic function of the code and the history. When the task is done the engine kills the coroutine (it exits with `runtime.Goexit` from inside its wait).

**Matching.** A command matches an event when:

| Command | Event | Compared |
|---|---|---|
| `ScheduleActivity` | `ActivityTaskScheduled` | activity id and activity type |
| `StartTimer` | `TimerStarted` | timer id |
| `RecordMarker` | `MarkerRecorded` | marker name |
| `CompleteWorkflow`, `FailWorkflow`, `ContinueAsNew` | `completed`, `failed`, `continued` | kind |

Activity and timer ids are the command's sequence number in the run ("1", "2", ...), so the same code issues the same ids in the same order on every replay.

### 2.2 Determinism and its tools

Anything that can differ between two runs of the same code breaks replay:

| Instead of | Use | Why it is safe |
|---|---|---|
| `time.Now()` | `workflow.Now(ctx)` | the time of the current task's `WorkflowTaskStarted`, recorded in history |
| `rand`, `uuid` | `workflow.SideEffect(ctx, f)` | `f` runs once; its JSON result is recorded as a `SideEffect` marker; replays return the recorded value and never call `f` |
| `time.Sleep` | `workflow.Sleep(ctx, d)` | a durable timer (`dur.07`) |
| goroutines, channels, I/O | activities | activities run in workers and their results are recorded |
| iterating a map | a sorted slice | Go randomizes map order |

### 2.3 Versioning

A deploy changes workflow code while runs are in flight. Their histories were made by the old code; replaying them with new code that issues different commands is nondeterminism. `GetVersion(ctx, changeID, min, max)` makes a change safe:

```go
if workflow.GetVersion(ctx, "use-b", 0, 1) == 0 {
	// old path, as the code was before the change
} else {
	// new path
}
```

On a first execution it returns `max` and records a `Version` marker `{change_id, version}`. On replay, if the history has that marker at this point, it returns the recorded version (and the marker is consumed); if not, the history predates the change and it returns `min` without consuming anything. Later calls with the same change id return the same value.

### 2.4 ContinueAsNew and history size

A workflow that loops (train for 10 epochs, evaluate every 500 steps for a week) grows its history without bound, and every workflow task replays all of it. The server suggests continuing once the history passes 10,000 events or 32 MiB (`continue_as_new_suggested`); `workflow.ContinueAsNewSuggested(ctx)` recomputes the same rule from the history as of the current task, so it is the same on every replay. The workflow returns `workflow.ContinueAsNew(ctx, state)`: the SDK issues a `ContinueAsNew` command; the server closes the run with `WorkflowExecutionContinuedAsNew{new_run_id, input}` and, once that is durable, starts the new run under the same workflow id with `continued_from_run_id`. The server refuses to grow a history past 20,000 events at all.

**History paging.** A history over 1 MiB reaches the worker in pages (`dur.04`); the engine needs every event, and the first page alone is not enough.

**Payload limits.** Payloads over 2 MiB are refused by the server: activity inputs and results, workflow results (`dur.02`, `dur.04`), and here marker details and continue-as-new inputs. Large data goes to `/artifacts` and travels by path.

### 2.5 Testing workflows by replay

The course test workflows (`Sequence`, `Parallel`, `SideEffects`, `Versioned`, `Reorder`, `Clock`, `Counter`) were run once against the reference and their histories recorded in `course/fixtures/dur/histories/`. Your engine must replay them all with the same results. For your own workflows the recipe is the same in two steps: run once through your server and worker, fetch the history, and replay it with `ReplayHistory`; any change that breaks determinism then fails a test instead of a production run.

## 3. Worked example by hand

`Sequence(2)` calls `Upper("s1")`, then `Upper("s2")`, and returns the list. Its history after both activities, with the current task started at event 15:

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

Handling the task whose history ends at 15:

1. **Past task 1** ($s_1 = 3$): deliver events 1 and 2 (nothing to resolve). Run: the code calls `ExecuteActivity("Upper", "s1")`, sequence 1, and blocks on `Get`. Command `ScheduleActivity{"1", Upper}` matches event 5. $C_1$ is consumed.
2. **Past task 2** ($s_2 = 9$): deliver events 3 to 8; event 7 resolves activity "1" with `"S1"`. Run: `Get` returns `"S1"`; the code calls `ExecuteActivity("Upper", "s2")`, sequence 2, and blocks. Command `"2"` matches event 11.
3. **Current task** ($s_* = 15$): deliver events 9 to 14; event 13 resolves "2" with `"S2"`. Run: the code returns `["S1","S2"]`. The new command is `CompleteWorkflow{["S1","S2"]}`.

Handling the earlier tasks (history up to 3, then up to 9) gives `ScheduleActivity{"1"}` and then `ScheduleActivity{"2"}`: the same function, more history. This is `TestReplayHandExample`.

## 4. The interface

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

`Registry` is the `worker.WorkflowHandler` of `dur.04`: `worker.New(conn, worker.Options{Workflows: reg, ...})`. Workflows are Go only; Python implements activities (DESIGN 2.8).

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `TestReplayHandExample` | unit | section 3: the three tasks issue `"1"`, `"2"`, then complete with `["S1","S2"]` | the definition of replay |
| `TestRecordedHistoriesReplay` | conformance, golden | every recorded course test run replays with its recorded result (continued runs end in ContinueAsNew) | the SDK conformance suite |
| `TestNondeterminismDetected` | unit, fault | another activity type, fewer commands, an extra command: `ErrNondeterminism`, replayed and live | code changes cannot corrupt runs |
| `TestReorderFailpointNondeterministic` | fault | `dur/workflow/reorder` flips the order: the same `ErrNondeterminism` three times; passes without it | the failpoint of the catalog |
| `TestSideEffectNotReexecuted` | unit | replay returns recorded values and never calls the function; a first run calls it once | random ids stay stable |
| `TestGetVersionMigration` | unit, regression | old history on the old path, new history on the new path, a new run records `{use-b, 1}` | safe deploys of dur.11 and dur.12 |
| `TestNowComesFromHistory` | unit | `Now` equals the `WorkflowTaskStarted` times, on replay too | no wall clock in workflows |
| `TestRecordThenReplayOwnRun` | unit | a run through your server and worker replays to the same result | how you test your own workflows |
| `TestKillServerMidWorkflowSameResult` | fault | the server dies mid-run; the result equals an uninterrupted run and the history replays | MS-durable's replay determinism |
| `TestContinueAsNewKeepsHistoryBounded` | fault, boundary | 12,000 steps continue as new past 10,000 events; each run under 20,000; pages over 1 MiB fetched | `TrainRun`, `AgentRun` |
| `TestServerCommandLimits` | boundary | marker and continue-as-new payloads over 2 MiB refused; the next run keeps type and queue | payload limits (2.7) |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. replay plumbing: dropping a delivered result, collecting a task's commands from the wrong place, completing with the wrong value, taking the first `WorkflowTaskStarted` as current | the workflow sees empty results, every past task looks nondeterministic, or old commands are issued again | `TestReplayHandExample`, `TestRecordThenReplayOwnRun`, `TestKillServerMidWorkflowSameResult` (mutants `s02`, `s04`, `s07`, `s14`) |
| 2. matching too loosely (id without type, any completion) | a changed workflow continues silently on a history it did not make | `TestReorderFailpointNondeterministic`, `TestNondeterminismDetected` (mutants `s03`, `s09`) |
| 3. ids that do not advance with each command | two activities share an id; results go to the wrong future | `TestReplayHandExample` (mutant `s01`) |
| 4. calling a side effect again on replay | a new random id on every replay forks the workflow's state | `TestSideEffectNotReexecuted` (mutant `s05`) |
| 5. versioning: the new path for old histories, the wrong version recorded, the marker not consumed | every in-flight run breaks at the next deploy | `TestGetVersionMigration` (mutants `s06`, `s11`, `s12`) |
| 6. ContinueAsNew treated as a failure, suggested too late, or the next run without its type and queue | long workflows fail, hit the history cap, or start a run no worker polls | `TestContinueAsNewKeepsHistoryBounded`, `TestServerCommandLimits` (mutants `s10`, `s15`, `s16`) |
| 7. reading the wall clock | each replay sees a different time and takes a different branch | `TestNowComesFromHistory` (mutant `s13`) |
| 8. no payload limits on markers or continue-as-new inputs | messages over 4 MiB, histories that cannot be served | `TestServerCommandLimits` (mutants `s17`, `s18`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `dur.05` | activity futures resolve with results, `ActivityError`s, and redriven attempts |
| Forward | `dur.07` | `Sleep` issues `StartTimer`; the server's timer commands and the timer service fire it |
| Forward | `dur.08` | signals and cancellation add event hooks and commands to the same engine |
| Forward | `dur.11`, `dur.12`, `data.09`, `ag.05` | `TrainRun`, `EvalSuite`, `ModelRelease`, `CorpusBuild`, and `AgentRun` are workflows on this SDK |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| replay from event 1 on every task | sticky caches | the worker keeps a run's coroutine between tasks and replays nothing; the server sends only new events | Temporal "Sticky Execution" |
| one coroutine per workflow | full SDK command state machines | `workflow.Go`, selectors, child workflows, cancellation scopes, each with its own state machine | `temporalio/sdk-go`: `internal/internal_decision_state_machine.go` |
| `GetVersion` by hand | worker deployment versioning | old runs stay pinned to the build that started them | Temporal "Worker Versioning" |
| replay on fixtures | determinism checks on upgrade | replay production histories against new code in CI before deploying | Temporal "Replay Testing" |
| a crash mid-activity retries it | saige `agent/durable/local`: `ErrIndeterminate` and `Reconcile` | an effect whose outcome is unknown is reconciled, not blindly retried | saige `agent/durable/local` |
