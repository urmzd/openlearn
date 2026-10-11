<!-- ol:module ag.05 -->
# Durable agent runs (AgentRun)

## Overview

| | |
|---|---|
| **Module** | `ag.05` · build · Go · Pass 10 · 4 h |
| **You build** | `go/agent/durableagent/journal.go` (`Record`, `Store`, `FileStore`, `Journal`, `ErrIndeterminate`, `Reconcile`), `go/agent/durableagent/run.go` (`Config`, `Run`, `Signal`, `Outcome`) |
| **Contract** | the step names and the `StepRunner` seam of `ag.01`; the Go API is section 4 |
| **Tests** | `course/tests/go/ag_05/` (what they check: section 4) |
| **Needs** | [`ag.01`](01-types-and-provider.md) (`StepRunner`, `StepToolName`), [`ag.02`](02-tools-and-schemas.md) (`tool.Result`), [`ag.03`](03-agent-loop.md) (the loop), [`ag.04`](04-tool-gate-and-injection.md) (the gate and `IsWrite`) · reading: [Durable Orchestration & Workers](../05-durable-orchestration-and-workers/) |
| **Used by** | `ag.09` runs suite subjects as durable runs (`DurableSubject`) · `dep.07` deploys the agent worker when it lands |
| **Milestone** | MS-agent |
| **Optional depth** | Temporal, [*Workflow execution: event history and replay*](https://docs.temporal.io/workflow-execution) (free); Helland, *Life beyond Distributed Transactions* (ACM Queue, 2016, free) |

## Key Takeaways

- A durable run is a **journal of steps**: each model call and tool call records its start, then its result, before the result is used (`TestHandExample`).
- On restart, **completed steps are replayed** from the journal: the model is not called again and no tool runs twice (`TestKillResumesWithoutRecall`).
- A step that started and never completed is **re-run when it only reads** (a model call, a lookup) and **never guessed at when it writes**: the run stops with `ErrIndeterminate` until a human reconciles it (`TestIndeterminateWrite`, `TestReadStepRerunsAfterCrash`).
- Approvals and reconciliations are **signals**: records appended to the journal, applied at the next attempt (`TestHandExample`, `TestIndeterminateWrite`).
- A crash can tear the last record: the journal **ignores a torn tail** and cuts it off before the next append (`TestFileStoreTornTail`).

## How to work this chapter

```bash
ol start ag.05
ol tests ag.05
ol check ag.05          # ag.01 to ag.04 smoke tests run first
ol diff  ag.05
```

---

## 1. Why now

An agent run that files a ticket, waits an hour for a human to approve it, and then reports back outlives any single process: the worker is redeployed, evicted, or OOM-killed in the middle. Started again from scratch, the run pays for every model call a second time and, worse, repeats side effects: two tickets, two emails, two refunds. You built a durable engine in Pass 8 for exactly this problem; this module applies its core idea (record every step, replay instead of re-executing) to the agent loop, through the `StepRunner` seam the loop already has.

## 2. Principles

### 2.1 The journal

A run is identified by a **run id** and owns an append-only list of records:

| Record | Written | Holds |
|---|---|---|
| `run.started` | once, first | the input messages |
| `step.started` | before a step runs | the step name and kind |
| `step.completed` | after it returns, before its result is used | the result bytes |
| `signal` | by `Signal`, any time | `approve` with a marker, or `reconcile` with a step and an outcome |
| `run.completed` | when the loop answers | the final message |

The run's state is a fold over its records. That is event sourcing, the same idea as the durable engine's history (Pass 8): nothing else is stored, so nothing else can disagree.

### 2.2 Durable means fsynced, and crashes tear

`Append` returns only after the record is on disk (`f.Sync()`). A crash in the middle of a write can leave half a line at the end of the file. That record was never acknowledged, so `Load` ignores a final line without its newline, and the next `Append` truncates the file to the last newline before writing. A corrupt line in the middle (not a torn tail) is an error: something other than a crash damaged the file.

The run id becomes a file name (`<dir>/<run id>.jsonl`), and it arrives from a CLI flag or an API request, so it is validated: letters, digits, `_`, `.`, `-`, starting with a letter or digit, at most 128 characters. `../escape` never reaches the file system.

### 2.3 Replay, and the write you cannot see

| Symbol | Meaning |
|---|---|
| $S(s)$ | a `step.started` record exists for step $s$ |
| $C(s)$ | a `step.completed` record exists for $s$ |
| $W(s)$ | $s$ is a tool step whose tool the gate classifies as a write (`IsWrite`) |

`RunStep(s)` decides:

| Case | Action |
|---|---|
| $C(s)$ | return the recorded result; do not call `fn` |
| $S(s) \wedge \neg C(s) \wedge W(s)$, reconciled `done` | record the human's result as completed; do not call `fn` |
| $S(s) \wedge \neg C(s) \wedge W(s)$, reconciled `retry` | run it again |
| $S(s) \wedge \neg C(s) \wedge W(s)$, not reconciled | `ErrIndeterminate` |
| otherwise | record start, call `fn`, record result, return it |

The crash between "the tool did its thing" and "the result was recorded" is the hard case. For a read (a model call, a lookup) running it again is harmless. For a write nobody knows whether it happened: re-running could file the ticket twice, skipping could lose it. The journal refuses to guess. The run reports `Indeterminate` with the step's name, and a human checks (is there a ticket?) and sends `reconcile` with `done` and the result, or `retry`. The `step.started` record must therefore be written **before** `fn` runs; written after, a crash inside the write leaves no trace and the write is repeated.

An `fn` error is not recorded: the next attempt runs the step again (a transient model error should not become permanent). The loop (`ag.03`) ends the run on any `StepRunner` error, so `ErrIndeterminate` reaches `Run` instead of being shown to the model.

### 2.4 A run, start to finish

`Run(ctx, cfg, id, input)` loads the records. A completed run returns its recorded answer. A new run records `run.started`; an existing run with different input is `ErrInputMismatch` (a retried start with the same input is harmless). Then it invokes the loop **from the recorded input** with the journal as `StepRunner`, the gate, and every approved marker from `approve` signals. Completed steps replay; the first step without a result runs. The loop's outcome becomes the run's: `Completed` (recorded), `AwaitingApproval` with the pending markers, or `Indeterminate` with the step.

## 3. Worked example by hand

The model looks up the access policy, then files a ticket (a write), then answers. Run `a1`:

| Attempt | Records appended | Model calls (total) | Outcome |
|---|---|---|---|
| Run 1 | `run.started`; start and result of `llm/1` (call `lookup`), `tool/1/0/lookup`, `llm/2` (call `create_ticket`) | 2 | `AwaitingApproval`: the gate wants a human for `create_ticket` |
| `Signal(approve, marker)` | `signal approve` | 2 | |
| Run 2 | start and result of `tool/2/0/create_ticket`, `llm/3`; `run.completed` | 3 | `Completed`: "Ticket filed." |
| Run 3 | none | 3 | the recorded answer |

After Run 1 the journal holds exactly seven records (the input, then two per step). In Run 2, `llm/1`, `tool/1/0/lookup`, and `llm/2` come back from the journal (no model call, no second lookup), the approved marker lets the ticket through, and only `llm/3` calls the model. This is `TestHandExample`.

Now kill the worker inside `create_ticket` after the ticket exists. The journal ends with `step.started tool/2/0/create_ticket` and no result. The next worker replays to that step and stops: `Indeterminate`, step `tool/2/0/create_ticket`, still one ticket. `reconcile` with `done` and `TICKET-7` records the result, the next worker replays it as the tool's answer, and `llm/3` sees "TICKET-7". This is `TestKillResumesWithoutRecall`, with real SIGKILLs of a child process.

## 4. The interface

```go
package durableagent // import "tinyllm/agent/durableagent"

type Record struct { Seq int64; Type, Step string; Kind types.StepKind; Data json.RawMessage }
type Store interface {
	Load(ctx context.Context, runID string) ([]Record, error)
	Append(ctx context.Context, runID string, r Record) error // durable when it returns
}
func ValidRunID(id string) bool
func OpenFileStore(dir string) (*FileStore, error)
func NewJournal(store Store, runID string, records []Record, isWrite func(string) bool) *Journal
func (j *Journal) RunStep(ctx context.Context, name string, kind types.StepKind, fn func(context.Context) ([]byte, error)) ([]byte, error)
var ErrIndeterminate error
type IndeterminateError struct{ Step string } // Unwrap() is ErrIndeterminate
type Reconcile struct { Step, Outcome, Result string } // Outcome "done" or "retry"

type Config struct { Agent loop.Config; Gate *gate.PolicyGate; Store Store; Options []loop.Option }
type Outcome struct { Status Status; Message types.Message; Pending []loop.Pending; Step string }
var ErrInputMismatch error
func Run(ctx context.Context, cfg Config, runID string, input []types.Message) (Outcome, error)
func Signal(ctx context.Context, store Store, runID, name string, payload any) error // "approve" {"Marker": m}, "reconcile" Reconcile
```

Use only the standard library and the `ag.01` to `ag.04` packages.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `TestHandExample` | unit | section 3: the seven records, approval, replay with no new model calls, a completed run calls nothing | you and the tests agree on the journal |
| `TestIndeterminateWrite` | fault | a crash between the write and its record: `Indeterminate` until reconciled; `done` never re-runs the write, `retry` runs it once | no double tickets, no lost ones |
| `TestReadStepRerunsAfterCrash` | fault | an unrecorded read is re-run; the model call before it is not | reads do not block the run |
| `TestInputMismatch` | boundary | same id and input resumes, other input is refused, an unknown run cannot be resumed | a run id names one run |
| `TestFileStoreTornTail` | fault | a half-written last line is ignored and cut by the next append | a crash mid-write does not brick the run |
| `TestRunIDValidation` | boundary | path-like ids are refused; nothing is written outside the store; signals need an existing run | run ids come from users |
| `TestHelperAgentWorker` | fault | the child process of the kill test (skipped when run alone) | |
| `TestKillResumesWithoutRecall` | fault | SIGKILL inside a read and inside a write across five workers: 3 model calls, 1 ticket | the module's promise under a real kill |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. completed steps run again | every restart pays for every model call; tools repeat | `TestHandExample`, `TestKillResumesWithoutRecall` (mutant `s01`) |
| 2. an unrecorded write re-run, an unrecorded read treated as a write, or the start recorded after the step | two tickets after a crash; runs stuck on harmless reads | `TestIndeterminateWrite`, `TestReadStepRerunsAfterCrash`, `TestKillResumesWithoutRecall` (mutants `s02`, `s03`, `s14`) |
| 3. a torn last record kept, or treated as corruption | the next record is glued onto garbage; the run cannot load | `TestFileStoreTornTail` (mutants `s05`, `s06`) |
| 4. a `done` reconciliation that runs the write anyway | the human said it happened; it happens again | `TestIndeterminateWrite` (mutant `s07`) |
| 5. approve signals recorded but not applied | the run waits for approval forever | `TestHandExample` (mutant `s08`) |
| 6. a reused run id answering the old question | a client retrying with a new prompt gets the previous run's answer | `TestInputMismatch` (mutant `s10`) |
| 7. run ids used as paths unchecked | `../../somewhere` writes outside the store | `TestRunIDValidation` (mutant `s11`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `ag.01` | `StepRunner`, `StepKind`, `StepToolName` |
| Back | `ag.02` | a reconciled write is recorded as a `tool.Result` |
| Back | `ag.03` | `Run` invokes the loop with the journal as its step runner and resumes from the recorded input |
| Back | `ag.04` | `IsWrite` decides which unrecorded steps are writes |
| Forward | `ag.09` | `DurableSubject` runs each case as a durable run, so a killed eval worker resumes without re-calling the model |
| Forward | `dep.07` | the deployed agent worker runs durable agent workflows |

The catalog also places `AgentRun` on the durable engine (`dur.06` replay, `dur.08` signals); until those modules are registered the run keeps its own journal, and adapting `Store` to the engine's event log is the step that joins them (DEVIATIONS B121-04).

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `Journal` | saige durable agent | local and DBOS backends, WAL stores, budget reservations restored on replay | saige `agent/durable/{local,dbos}` |
| `ErrIndeterminate` and `reconcile` | saige `ErrIndeterminate` + `Reconcile` | the same refusal to guess, with fenced leases for remote workers | saige `agent/durable/local` |
| approvals as signals | Temporal signals and updates | validated synchronous updates; the worker is released while waiting (`ErrSuspended`) | [Temporal message passing](https://docs.temporal.io/encyclopedia/workflow-message-passing) (free) |
| replay | Temporal workflow replay | determinism checks on code changes, `GetVersion`, history paging | [Temporal docs](https://docs.temporal.io/) (free) |
