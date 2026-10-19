# Durable agent runs (AgentRun)

## Overview

| | |
|---|---|
| **Module** | `ag.05` · build · Go · Pass 10 · 4 h |
| **Solution** | [`journal.go`](../../../../course/solutions/go/agent/durableagent/journal.go) (`Record`, `Store`, `FileStore`, `Journal`, `ErrIndeterminate`, `Reconcile`), [`run.go`](../../../../course/solutions/go/agent/durableagent/run.go) (`Config`, `Run`, `Signal`, `Outcome`) |
| **Contract** | the step names and the `StepRunner` seam of `ag.01`; the Go API is section 4 |
| **Needs** | [`ag.01`](01-types-and-provider.md) (`StepRunner`, `StepToolName`), [`ag.02`](02-tools-and-schemas.md) (`tool.Result`), [`ag.03`](03-agent-loop.md) (the loop), [`ag.04`](04-tool-gate-and-injection.md) (the gate and `IsWrite`) · reading: [Durable Orchestration & Workers](../05-durable-orchestration-and-workers/) |
| **Used by** | `ag.09` runs suite subjects as durable runs (`DurableSubject`) · `dep.07` deploys the agent worker when it lands · the agent track's end-to-end run survives a killed worker because of it |
| **Optional depth** | Temporal, [*Workflow execution: event history and replay*](https://docs.temporal.io/workflow-execution) (free); Helland, *Life beyond Distributed Transactions* (ACM Queue, 2016, free) |

## Key Takeaways

- A durable run is a **journal of steps**: each model call and tool call records its start, then its result, before the result is used.
- On restart, **completed steps are replayed** from the journal: the model is not called again and no tool runs twice.
- A step that started and never completed is **re-run when it only reads** (a model call, a lookup) and **never guessed at when it writes**: the run stops with `ErrIndeterminate` until a human reconciles it.
- Approvals and reconciliations are **signals**: records appended to the journal, applied at the next attempt.
- A crash can tear the last record: the journal **ignores a torn tail** and cuts it off before the next append.

## How to read this chapter

Sections 1 to 3 build the journal from first principles and trace one run, with and without a crash, by hand. Section 4 fixes the interface. Section 5 walks through the solution and explains why each choice is the right one; try to predict each step before reading it.

---

## 1. Why now

An agent run that files a ticket, waits an hour for a human to approve it, and then reports back outlives any single process: the worker is redeployed, evicted, or OOM-killed in the middle. Started again from scratch, the run pays for every model call a second time and, worse, repeats side effects: two tickets, two emails, two refunds. Pass 8 built a durable engine for exactly this problem; this module applies its core idea (record every step, replay instead of re-executing) to the agent loop, through the `StepRunner` seam the loop already has.

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

After Run 1 the journal holds exactly seven records (the input, then two per step). In Run 2, `llm/1`, `tool/1/0/lookup`, and `llm/2` come back from the journal (no model call, no second lookup), the approved marker lets the ticket through, and only `llm/3` calls the model.

Now kill the worker inside `create_ticket` after the ticket exists. The journal ends with `step.started tool/2/0/create_ticket` and no result. The next worker replays to that step and stops: `Indeterminate`, step `tool/2/0/create_ticket`, still one ticket. `reconcile` with `done` and `TICKET-7` records the result, the next worker replays it as the tool's answer, and `llm/3` sees "TICKET-7". A correct solution keeps this promise under real `SIGKILL`s of a worker process, not only simulated crashes.

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

### What a correct solution must handle

| Case | Kind of check | What must hold | Why it matters downstream |
|---|---|---|---|
| the hand example (section 3) | unit | the seven records, approval, replay with no new model calls, a completed run calls nothing | the journal is pinned down |
| a crash between a write and its record | fault | `Indeterminate` until reconciled; `done` never re-runs the write, `retry` runs it once | no double tickets, no lost ones |
| a crash inside a read | fault | the unrecorded read is re-run; the model call before it is not | reads do not block the run |
| run ids and inputs | boundary | the same id and input resumes, other input is refused, an unknown run cannot be resumed | a run id names one run |
| a torn last line | fault | ignored by `Load` and cut by the next append | a crash mid-write does not brick the run |
| path-like run ids | boundary | refused; nothing is written outside the store; signals need an existing run | run ids come from users |
| real `SIGKILL`s across five worker processes | fault | killed inside a read and inside a write: 3 model calls in total, 1 ticket | the module's promise under a real kill |

## 5. The solution, step by step

### 5.1 From first principles

1. **What must survive a crash is the knowledge of what already happened.** If every step's result is written down before it is used, a new process can rebuild the run by reading, without redoing. So the state is an append-only journal, and everything else (completed steps, signals, the answer) is a fold over it.
2. **"Written down" means on disk.** A record counts only once `Append` has fsynced it; anything after the last newline was never acknowledged and is ignored.
3. **The loop is already deterministic given its step results.** With the same input and the same step results, `ag.03` makes the same decisions and names the same steps. So replay is just "run the loop again from the recorded input, and answer each `RunStep` from the journal when possible": no special replay mode in the loop.
4. **Record the start before the step, the result after.** Then the journal distinguishes three states per step: never started, started but unfinished, finished. Only the middle one is ambiguous.
5. **Ambiguity is harmless for reads and dangerous for writes.** A read can be re-run; a write may or may not have happened. The program cannot know, so it asks a human (`ErrIndeterminate`) rather than guessing in either direction.
6. **Human input is just more records.** Approvals and reconciliations are appended as signals and take effect at the next `Run`, which keeps the journal the single source of truth.
7. **A run id is user input that becomes a file name.** It is validated against a strict pattern before it touches the file system.

### 5.2 The code, annotated

**Appending durably** (`journal.go`), step 2. Each append reads the file, cuts any torn tail at the last newline, writes the new line there, and fsyncs:

```go
func (s *FileStore) Append(_ context.Context, runID string, r Record) error {
	p, err := s.path(runID) // validates the run id (step 7)
	// ... marshal r as one JSON line
	s.mu.Lock()
	defer s.mu.Unlock()
	f, err := os.OpenFile(p, os.O_RDWR|os.O_CREATE, 0o644)
	// ...
	b, err := io.ReadAll(f)
	// ...
	end := int64(bytes.LastIndexByte(b, '\n') + 1) // everything after the last newline is a torn write
	if err := f.Truncate(end); err != nil {
		return err
	}
	if _, err := f.WriteAt(append(line, '\n'), end); err != nil {
		return err
	}
	return f.Sync() // only now is the record acknowledged
}
```

**Loading** ignores a final line without its newline and refuses a corrupt line anywhere else:

```go
for {
	line, err := br.ReadBytes('\n')
	if err == io.EOF {
		break // a final line without its newline is a torn write: ignore it
	}
	var r Record
	if jerr := json.Unmarshal(line, &r); jerr != nil {
		return nil, fmt.Errorf("durableagent: %s: corrupt record %d: %w", p, len(out)+1, jerr) // not a crash: stop
	}
	out = append(out, r)
}
```

**Folding**, step 1. `NewJournal` turns the records into three maps (completed results, started steps, reconciliations) and the last sequence number.

**`RunStep`**, steps 3 to 5, is the decision table of section 2.3 in code:

```go
func (j *Journal) RunStep(ctx context.Context, name string, kind types.StepKind, fn func(context.Context) ([]byte, error)) ([]byte, error) {
	// ... read done[name], started[name], recon[name] under the lock
	if done {
		return out, nil // C(s): replay, fn is not called
	}
	toolName, isTool := types.StepToolName(name)
	if started && kind == types.StepTool && isTool && j.isWrite(toolName) { // S, not C, W
		switch {
		case reconciled && rc.Outcome == "done":
			res, _ := json.Marshal(tool.Result{Content: rc.Result}) // the human's answer; the loop sets the call id
			if err := j.append(ctx, Record{Type: StepCompleted, Step: name, Kind: kind, Data: res}); err != nil {
				return nil, err
			}
			// ... done[name] = res
			return res, nil // the write is not run again
		case reconciled && rc.Outcome == "retry":
			// a human checked: the write did not happen; fall through and run it
		default:
			return nil, &IndeterminateError{Step: name} // refuse to guess
		}
	}
	if err := j.append(ctx, Record{Type: StepStarted, Step: name, Kind: kind}); err != nil {
		return nil, err // before fn: a crash inside fn leaves a trace (step 4)
	}
	out, err := fn(ctx)
	if err != nil {
		return nil, err // not recorded: the next attempt runs it again
	}
	if err := j.append(ctx, Record{Type: StepCompleted, Step: name, Kind: kind, Data: out}); err != nil {
		return nil, err
	}
	// ... done[name] = out
	return out, nil
}
```

A started-but-unfinished **read** skips the `if` and simply runs again. Because the loop (`ag.03`) sets `res.CallID, res.Name` on every tool result it decodes, the reconciled result answers the right call even though the journal stored only its content.

**`Run`** (`run.go`), steps 3 and 6. It folds the run-level records, refuses a mismatched input, and invokes the ordinary loop with the journal plugged in:

```go
for _, r := range recs {
	switch r.Type {
	case RunStarted:
		// ... recorded = the input messages
	case RunCompleted:
		// ... return Outcome{Status: Completed, Message: m}: a finished run calls nothing
	case SignalRecord:
		if r.Step == "approve" {
			// ... approved = append(approved, marker)
		}
	}
}
j := NewJournal(cfg.Store, runID, recs, nil)
if recorded == nil {
	// ... new run (input required): append run.started with the input
	recorded = input
} else if input != nil && !sameMessages(input, recorded) {
	return Outcome{}, ErrInputMismatch // one run id, one question
}
opts := append([]loop.Option{loop.WithStepRunner(j), loop.WithApprovals(approved...)}, cfg.Options...)
if cfg.Gate != nil {
	j.isWrite = cfg.Gate.IsWrite // the gate's classes decide which steps are writes
	opts = append(opts, loop.WithGate(cfg.Gate))
}
msg, err := loop.New(cfg.Agent, opts...).Invoke(ctx, recorded).Result() // always from the recorded input
```

The loop's error is then mapped to an outcome: `*loop.ApprovalError` becomes `AwaitingApproval` with the pending markers, `*IndeterminateError` (which the loop passes through because it ends the run on any step error) becomes `Indeterminate` with the step, any other error is returned unrecorded (the next `Run` retries from the journal), and a clean answer is recorded as `run.completed`.

`Signal` checks the signal name and that the run exists, then appends one `signal` record with the payload.

Full solution: [`journal.go`](../../../../course/solutions/go/agent/durableagent/journal.go), [`run.go`](../../../../course/solutions/go/agent/durableagent/run.go).

### 5.3 Why this solution

- **Cost.** Two fsynced appends per step (a few milliseconds on an SSD), negligible next to a model call. `Append` re-reads the file to find the last newline, which is $O(\text{file size})$ per append; agent runs have tens of steps, so the file stays small. A long-lived journal would keep the end offset in memory instead.
- **Replay by re-running the loop** rather than serializing the loop's internal state: the loop stays one piece of code, and anything it computes from step results is recomputed identically. The price is that the loop must stay deterministic given those results (no clock reads or random choices outside steps).
- **Refusing to guess on writes** rather than "at least once" or "at most once": at-least-once duplicates tickets, at-most-once loses them. With no idempotency key at the tool, only a human (or a tool that can be queried) knows. Tools that accept an idempotency key could make retry safe and remove the human step.
- **A file per run** rather than a database: no dependency, trivially inspected with `cat`, and one writer per run. The durable engine's event log (Pass 8) is the scalable version; `Store` is the seam to move to it.
- **Tradeoff accepted:** an fsync per record, so a step costs two syncs. Batching would be faster and would lose acknowledged steps on a crash, which is the one thing this module exists to prevent.

### 5.4 Explain it back

1. Why must `step.started` be appended before `fn` runs?
   <details><summary>Answer</summary>If it were appended after, a crash inside the write would leave no record at all, and the next attempt would see a step that never started and run the write again: two tickets. Recording the start first turns that crash into the visible "started, not completed" state.</details>
2. In section 3, Run 2 makes only one model call. Which steps replay, and how does the ticket get through the gate?
   <details><summary>Answer</summary>`llm/1`, `tool/1/0/lookup`, and `llm/2` return their recorded results. `llm/2`'s recorded reply proposes `create_ticket`; the gate returns `NeedApproval` with the same marker as in Run 1 (same name and arguments), and `Run` passed that marker to `WithApprovals`, so the loop allows it. Only `llm/3` is new.</details>
3. A worker is killed inside `lookup` (a read). What does the next worker do?
   <details><summary>Answer</summary>`RunStep("tool/1/0/lookup")` finds the step started but not completed; `lookup` is not a write, so it records a new start and runs it again. The model call before it (`llm/1`) is completed and replays without a call.</details>
4. Why does a corrupt line in the middle of the file fail `Load`, while a broken last line is ignored?
   <details><summary>Answer</summary>A crash can only interrupt the last write, so a partial final line is expected and was never acknowledged. Damage earlier in the file means something else corrupted it; silently skipping a record would replay the run with a step missing, so the safe answer is to stop.</details>
5. What does a `reconcile` with `done` change, and what does the model see?
   <details><summary>Answer</summary>`RunStep` appends a `step.completed` record with the human's result and returns it without running the tool. The loop decodes it as a `tool.Result`, sets the call id and name, and appends the tool message, so the next model call sees, for example, "TICKET-7" as the tool's answer.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. completed steps run again | every restart pays for every model call; tools repeat | `RunStep` returns `done[name]` without calling `fn` |
| 2. an unrecorded write re-run, an unrecorded read treated as a write, or the start recorded after the step | two tickets after a crash; runs stuck on harmless reads | `ErrIndeterminate` only for started tool steps whose tool `IsWrite`; `step.started` appended before `fn` |
| 3. a torn last record kept, or treated as corruption | the next record is glued onto garbage; the run cannot load | `Load` ignores a final line without a newline; `Append` truncates to the last newline |
| 4. a `done` reconciliation that runs the write anyway | the human said it happened; it happens again | the `done` branch records the result and returns without calling `fn` |
| 5. approve signals recorded but not applied | the run waits for approval forever | `Run` collects every `approve` marker and passes them to `WithApprovals` |
| 6. a reused run id answering the old question | a client retrying with a new prompt gets the previous run's answer | `ErrInputMismatch` when the recorded input differs |
| 7. run ids used as paths unchecked | `../../somewhere` writes outside the store | `ValidRunID` in `Run` and in every `FileStore` path |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `ag.01` | `StepRunner`, `StepKind`, `StepToolName` |
| Back | `ag.02` | a reconciled write is recorded as a `tool.Result` |
| Back | `ag.03` | `Run` invokes the loop with the journal as its step runner and resumes from the recorded input |
| Back | `ag.04` | `IsWrite` decides which unrecorded steps are writes |
| Forward | `ag.09` | `DurableSubject` runs each case as a durable run, so a killed eval worker resumes without re-calling the model |
| Forward | `dep.07` | the deployed agent worker runs durable agent workflows |

The catalog also places `AgentRun` on the durable engine (`dur.06` replay, `dur.08` signals); for now the run keeps its own journal, and adapting `Store` to the engine's event log is the step that joins them.

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `Journal` | saige durable agent | local and DBOS backends, WAL stores, budget reservations restored on replay | saige `agent/durable/{local,dbos}` |
| `ErrIndeterminate` and `reconcile` | saige `ErrIndeterminate` + `Reconcile` | the same refusal to guess, with fenced leases for remote workers | saige `agent/durable/local` |
| approvals as signals | Temporal signals and updates | validated synchronous updates; the worker is released while waiting (`ErrSuspended`) | [Temporal message passing](https://docs.temporal.io/encyclopedia/workflow-message-passing) (free) |
| replay | Temporal workflow replay | determinism checks on code changes, `GetVersion`, history paging | [Temporal docs](https://docs.temporal.io/) (free) |
