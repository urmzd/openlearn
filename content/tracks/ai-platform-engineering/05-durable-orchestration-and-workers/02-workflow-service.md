<!-- ol:module dur.02 -->
# Workflow service + idempotent start, describe, list, paged history

## Overview

| | |
|---|---|
| **Module** | `dur.02` · build · Go · Pass 8 · 5 to 7 h |
| **You build** | `go/durable/server/server.go`: the event-sourced core (`Open`, `apply`, `effect`, `commit`, recovery) and `go/durable/server/workflows.go`: `StartWorkflow`, `DescribeWorkflow`, `GetHistory`, `ListWorkflows` |
| **Contract** | `tl.durable.v1.WorkflowService` in [`proto/tl/durable/v1/durable.proto`](../../../../course/contracts/proto/tl/durable/v1/durable.proto) (generated Go in `contracts/go/gen/tl/durable/v1`); storage is [`formats/wal.md`](../../../../course/contracts/formats/wal.md); the Go API of the package is section 4 |
| **Tests** | `course/tests/go/dur_02/` (what they check: section 4); recorded runs in `course/fixtures/dur/histories/` |
| **Needs** | `dur.01` the log, `dur.03` the task queue; reading: [`lang.10` gRPC](../../software-craftsmanship/12-language-and-tool-primers/10-protocol-buffers-and-grpc.md), [case study 03, exactly-once event API](../../../case-studies/03-exactly-once-event-api/) |
| **Used by** | `dur.04` task RPCs and `dur.08` signals commit through this core |
| **Milestone** | MS-durable |
| **Optional depth** | Fowler, *Event Sourcing*; Kleppmann, *DDIA* ch. 11 (derived data); the Temporal docs on workflow id reuse and conflict policies |

## Key Takeaways

- A run's history is the only truth. Its state (status, the pending workflow task, pending activities, timers) is a fold of `apply` over the events, the same function at commit time and at recovery (`TestRecordedHistoriesRecover`, `TestRecoveryRebuildsEverything`).
- Side effects (queue tasks, timers) are derived from events by `effect`; a crash between an append and its effect is repaired by recovery, because the queue ignores a task id it already holds (`TestRecoveryEnqueuesLostTask`).
- `StartWorkflow` is idempotent on `workflow_id`: 64 concurrent identical starts give one run, and a different type or input is `FAILED_PRECONDITION` (`TestStartIdempotentUnder64Concurrent`, `TestStartConflict`).
- Every rpc appends one batch with `expected` = the history's length, so two rpcs can never interleave events in one run (`TestHistoryIsWalRecords`).

## How to work this chapter

```bash
ol start dur.02          # writes go/durable/server/{server,workflows}.go (and stubs of the later files of the package)
ol tests dur.02
ol check dur.02          # dur.01 and dur.03 must pass first, or add --ref-deps
ol diff  dur.02
```

Write `apply` first (it is a big switch with one case per event), then `commit` and `StartWorkflow`, then the read rpcs, then `Open`.

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

`StartWorkflow(workflow_id, type, input)`: while any run with that id exists (any status), the same type and input return its newest run id with `started = false`; a different type or input is `FAILED_PRECONDITION`. The comparison is against the **first** run's start (a workflow that continued as new has a different input in its newest run). The check and the commit must happen under one lock: check, unlock, commit lets two concurrent starts both pass the check (the classic check-then-act race, exactly what case study 03 solves with a unique constraint). A new run's id comes from `Options.NewRunID` (32 random hex digits by default; the tests inject `run-1`, `run-2`, ...).

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

The same command again finds `origin["demo-1"] = run-1`, compares type `Echo` and input `"hi"` with that run's start (equal), and answers `{run_id: "run-1", started: false}` without appending anything. `wf describe demo-1` now reports RUNNING, 2 events. These are the numbers of `TestStartHandExample`.

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

Register it on a gRPC server with `durablev1.RegisterWorkflowServiceServer(g, srv)` and `durablev1.RegisterTaskServiceServer(g, srv)`. The unexported helpers the later files of the package use are part of this module's work: `run`, `batch`, `(*Server).add`, `commit`, `startRun`, `lookup`, `enqueueWorkflowTask`, `enqueueActivity`, `parseTaskID`, `logErr`. Your stub has their signatures.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `TestStartHandExample` | unit | section 3: events 1 and 2, describe, the second start, one log record for the batch | the CLI's `wf start` and `wf describe` |
| `TestStartIdempotentUnder64Concurrent` | fault, property | 64 parallel starts: one `started`, one run id, one queued workflow task | retried CLIs, two operators |
| `TestStartConflict` | unit, boundary | different input or type is `FAILED_PRECONDITION`; nothing appended | an id names one piece of work |
| `TestStartValidation` | boundary | required fields; 2 MiB input accepted, one byte more refused | payloads travel by path |
| `TestStartEnqueuesOneWorkflowTask` | unit | task `run-1/wt/2` on `wf:<task_queue>` | where `dur.04` polls |
| `TestHistoryIsWalRecords` | conformance | each event is a `WalRecord` with both ids; stream version = event id | formats/wal.md, `dur.10` |
| `TestRecordedHistoriesRecover` | conformance, golden | recorded runs written into an empty log: identical `GetHistory` (also from a middle page), describe and list derived from events alone | recovery is a fold |
| `TestRecoveryRebuildsEverything` | fault | list and history identical after a restart; start still idempotent; no task doubled | every restart |
| `TestRecoveryEnqueuesLostTask` | fault | a history whose enqueue was lost gets its workflow task back | crash between append and effect |
| `TestRecoveryRearmsTimers` | fault | recovery arms exactly the pending timers at their `fire_at` | durable timers (`dur.07`) |
| `TestRecoveryStartsMissingContinuedRun` | fault | the successor of a continued run is started if the crash lost it | ContinueAsNew (`dur.06`) |
| `TestListWorkflowsNewestFirstPaged` | unit | newest first, pages of 2, type and status filters | `wf list` |
| `TestNotFoundAndBadTokens` | boundary | unknown workflow or run is `NOT_FOUND`; bad tokens `INVALID_ARGUMENT` | CLI error messages |
| `TestPageTokenRoundTrip` | unit | `ParsePageToken(PageToken(n)) = n`; empty is 1; 0 is refused | workers pass tokens back unchanged |
| `TestQuotaResourceExhausted` | fault | `RESOURCE_EXHAUSTED` at the WAL quota, no half-registered run, raising the quota resumes | drill `ops.11` |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. idempotency keyed by the wrong thing, or a partial comparison of the input | a retried start creates a second run, or a different request silently gets the old run | `TestStartHandExample`, `TestStartConflict` (mutants `s01`, `s05`) |
| 2. check, unlock, then commit | two concurrent starts both create runs (and race on the maps) | `TestStartIdempotentUnder64Concurrent` (mutant `s04`) |
| 3. recovery that does not rebuild `origin` and `current` | after a restart, a repeated start creates a duplicate run; describe finds nothing | `TestRecoveryRebuildsEverything` (mutants `s09`, `s22`) |
| 4. recovery that replays state but not effects | a run whose task was lost, a pending timer, or a continued run's successor waits forever | `TestRecoveryEnqueuesLostTask`, `TestRecoveryRearmsTimers`, `TestRecoveryStartsMissingContinuedRun` (mutants `s12`, `s24`, `s25`) |
| 5. putting workflow tasks on the bare task queue name | workers polling `wf:<queue>` never see them | `TestStartEnqueuesOneWorkflowTask` (mutant `s07`) |
| 6. state not derived from the events (start time, attempts, result, bytes) or a `WalRecord` without the workflow id | describe disagrees with history; recovery cannot tell which workflow a run belongs to | `TestStartHandExample`, `TestHistoryIsWalRecords`, `TestRecordedHistoriesRecover` (mutants `s02`, `s03`, `s08`, `s10`, `s11`) |
| 7. an off-by-one payload limit | an input of exactly 2 MiB is refused | `TestStartValidation` (mutant `s06`) |
| 8. list order, pages, and tokens | `wf list` shows the oldest first, repeats a run across pages, or panics on a token for event 0; an unknown workflow panics | `TestListWorkflowsNewestFirstPaged`, `TestPageTokenRoundTrip`, `TestNotFoundAndBadTokens` (mutants `s13`, `s14`, `s15`, `s23`) |
| 9. the quota as `UNAVAILABLE`, or registering the run before its batch is durable | clients retry a full disk forever; a refused start leaves a ghost run | `TestQuotaResourceExhausted` (mutants `s16`, `s17`) |

## 6. Where it's used next

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
