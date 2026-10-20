# Durable agent runs

An agent run that files a ticket, waits an hour for a human to approve it, and then reports back outlives any single process. The worker is redeployed, evicted, or OOM-killed in the middle. Started again from scratch, the run pays for every model call a second time and, worse, repeats side effects: two tickets, two emails, two refunds.

The chapter [Durable Orchestration and Workers](../../systems/07-durable-orchestration/README.md) built a durable engine for exactly this problem. This section applies its core idea, record every step and replay instead of re-executing, to the agent loop, through the `StepRunner` seam the loop already has. A durable run is a **journal of steps**: each model call and tool call records its start, then its result, before the result is used. On restart, completed steps are replayed from the journal, so the model is not called again and no tool runs twice. The interesting part is what happens to a step that started and never finished, and the answer depends on whether it reads or writes.

This section builds on all four earlier sections of the chapter: `StepRunner` and `StepToolName` from [Types, an OpenAI-compatible provider, and a retry wrapper](01-types-and-provider.md), `tool.Result` from [Tools, a registry, and JSON Schema argument validation](02-tools-and-schemas.md), [The agent loop](03-agent-loop.md) itself, and the gate and its `IsWrite` from [The tool gate, the SQL safety gate, and the prompt-injection suite](04-tool-gate-and-injection.md). The contract is the step names and the `StepRunner` seam defined in the first section. The full solution is [`journal.go`](../../../../course/solutions/go/agent/durableagent/journal.go) (`Record`, `Store`, `FileStore`, `Journal`, `ErrIndeterminate`, `Reconcile`) and [`run.go`](../../../../course/solutions/go/agent/durableagent/run.go) (`Config`, `Run`, `Signal`, `Outcome`).

> **Learning objectives**
> - Describe a run as an append-only journal and its state as a fold over the records.
> - Explain how the journal survives a torn final record.
> - Apply the replay decision table, including the case of a write that started and never completed.
> - Trace a run through approval, replay, a crash inside a write, and reconciliation by hand.
> - Explain why replay works by re-running the ordinary loop from the recorded input.

## The journal

A run is identified by a **run id** and owns an append-only list of records. There are five kinds:

| Record | Written | Holds |
|---|---|---|
| `run.started` | once, first | the input messages |
| `step.started` | before a step runs | the step name and kind |
| `step.completed` | after it returns, before its result is used | the result bytes |
| `signal` | by `Signal`, any time | `approve` with a marker, or `reconcile` with a step and an outcome |
| `run.completed` | when the loop answers | the final message |

The run's state is a fold over its records. That is event sourcing, the same idea as the durable engine's history: nothing else is stored, so nothing else can disagree. Approvals and reconciliations are **signals**, which means they are simply more records, appended to the journal and applied at the next attempt.

## Durable means fsynced, and crashes tear

A record is only worth something if it survives the crash it is meant to protect against. `Append` returns only after the record is on disk (`f.Sync()`). A crash in the middle of a write can leave half a line at the end of the file. That record was never acknowledged, so `Load` ignores a final line without its newline, and the next `Append` truncates the file to the last newline before writing. The journal **ignores a torn tail** and cuts it off. A corrupt line in the middle of the file is a different matter: a crash cannot cause it, so something else damaged the file, and `Load` reports an error.

The run id becomes a file name (`<dir>/<run id>.jsonl`), and it arrives from a CLI flag or an API request, so it is validated: letters, digits, `_`, `.`, and `-`, starting with a letter or digit, at most 128 characters. `../escape` never reaches the file system.

## Replay, and the write you cannot see

With a durable journal in hand, the step runner can decide what to do with each step. Use these predicates:

| Symbol | Meaning |
|---|---|
| $S(s)$ | a `step.started` record exists for step $s$ |
| $C(s)$ | a `step.completed` record exists for $s$ |
| $W(s)$ | $s$ is a tool step whose tool the gate classifies as a write (`IsWrite`) |

`RunStep(s)` then follows this table:

| Case | Action |
|---|---|
| $C(s)$ | return the recorded result; do not call `fn` |
| $S(s) \wedge \neg C(s) \wedge W(s)$, reconciled `done` | record the human's result as completed; do not call `fn` |
| $S(s) \wedge \neg C(s) \wedge W(s)$, reconciled `retry` | run it again |
| $S(s) \wedge \neg C(s) \wedge W(s)$, not reconciled | `ErrIndeterminate` |
| otherwise | record start, call `fn`, record result, return it |

The crash between "the tool did its thing" and "the result was recorded" is the hard case. For a read (a model call, a lookup), running it again is harmless. For a write, nobody knows whether it happened: re-running could file the ticket twice, and skipping could lose it. The journal refuses to guess. The run reports `Indeterminate` with the step's name, and a human checks (is there a ticket?) and sends `reconcile` with `done` and the result, or with `retry`. So an unfinished step is **re-run when it only reads** and **never guessed at when it writes**. For this to work, the `step.started` record must be written **before** `fn` runs; written after, a crash inside the write would leave no trace and the write would be repeated.

An error returned by `fn` is not recorded, so the next attempt runs the step again; a transient model error should not become permanent. The loop ends the run on any `StepRunner` error, so `ErrIndeterminate` reaches `Run` instead of being shown to the model.

## A run, start to finish

`Run(ctx, cfg, id, input)` puts the pieces together. It loads the records. A completed run returns its recorded answer. A new run records `run.started`; an existing run with different input is `ErrInputMismatch` (a retried start with the same input is harmless). Then it invokes the loop **from the recorded input**, with the journal as `StepRunner`, the gate, and every approved marker from `approve` signals. Completed steps replay; the first step without a result runs. The loop's outcome becomes the run's: `Completed` (recorded), `AwaitingApproval` with the pending markers, or `Indeterminate` with the step.

## Tracing a run by hand

The model looks up the access policy, then files a ticket (a write), then answers. Here is run `a1` across its attempts:

| Attempt | Records appended | Model calls (total) | Outcome |
|---|---|---|---|
| Run 1 | `run.started`; start and result of `llm/1` (call `lookup`), `tool/1/0/lookup`, `llm/2` (call `create_ticket`) | 2 | `AwaitingApproval`: the gate wants a human for `create_ticket` |
| `Signal(approve, marker)` | `signal approve` | 2 | |
| Run 2 | start and result of `tool/2/0/create_ticket`, `llm/3`; `run.completed` | 3 | `Completed`: "Ticket filed." |
| Run 3 | none | 3 | the recorded answer |

After Run 1 the journal holds exactly seven records: the input, then two per step. In Run 2, `llm/1`, `tool/1/0/lookup`, and `llm/2` come back from the journal (no model call, no second lookup), the approved marker lets the ticket through, and only `llm/3` calls the model.

Now kill the worker inside `create_ticket` after the ticket exists. The journal ends with `step.started tool/2/0/create_ticket` and no result. The next worker replays to that step and stops: `Indeterminate`, step `tool/2/0/create_ticket`, and still one ticket. A `reconcile` with `done` and `TICKET-7` records the result, the next worker replays it as the tool's answer, and `llm/3` sees "TICKET-7". A correct implementation keeps this promise under real `SIGKILL`s of a worker process, not only simulated crashes.

## Designing the API

The package surface:

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

The implementation uses only the standard library and the four packages from earlier in this chapter.

## Building it step by step

The reasoning chain first; try to predict each step before reading it.

1. **What must survive a crash is the knowledge of what already happened.** If every step's result is written down before it is used, a new process can rebuild the run by reading, without redoing. So the state is an append-only journal, and everything else (completed steps, signals, the answer) is a fold over it.
2. **"Written down" means on disk.** A record counts only once `Append` has fsynced it; anything after the last newline was never acknowledged and is ignored.
3. **The loop is already deterministic given its step results.** With the same input and the same step results, the loop makes the same decisions and names the same steps. So replay is just "run the loop again from the recorded input, and answer each `RunStep` from the journal when possible," with no special replay mode in the loop.
4. **Record the start before the step, the result after.** Then the journal distinguishes three states per step: never started, started but unfinished, finished. Only the middle one is ambiguous.
5. **Ambiguity is harmless for reads and dangerous for writes.** A read can be re-run; a write may or may not have happened. The program cannot know, so it asks a human (`ErrIndeterminate`) rather than guessing in either direction.
6. **Human input is just more records.** Approvals and reconciliations are appended as signals and take effect at the next `Run`, which keeps the journal the single source of truth.
7. **A run id is user input that becomes a file name.** It is validated against a strict pattern before it touches the file system.

### Appending durably

Step 2 is `Append` in `journal.go`. Each append reads the file, cuts any torn tail at the last newline, writes the new line there, and fsyncs:

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

### Loading

`Load` ignores a final line without its newline and refuses a corrupt line anywhere else:

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

For step 1, `NewJournal` folds the records into three maps (completed results, started steps, reconciliations) and the last sequence number.

### Running a step

`RunStep` covers steps 3 to 5; it is the decision table above, in code:

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

A started-but-unfinished **read** skips the `if` and simply runs again. Because the loop sets `res.CallID, res.Name` on every tool result it decodes, the reconciled result answers the right call even though the journal stored only its content.

### Running the whole run

`Run` in `run.go` covers steps 3 and 6. It folds the run-level records, refuses a mismatched input, and invokes the ordinary loop with the journal plugged in:

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

The loop's error is then mapped to an outcome. `*loop.ApprovalError` becomes `AwaitingApproval` with the pending markers. `*IndeterminateError`, which the loop passes through because it ends the run on any step error, becomes `Indeterminate` with the step. Any other error is returned unrecorded, so the next `Run` retries from the journal. A clean answer is recorded as `run.completed`. `Signal` checks the signal name and that the run exists, then appends one `signal` record with the payload.

The full solution is in [`journal.go`](../../../../course/solutions/go/agent/durableagent/journal.go) and [`run.go`](../../../../course/solutions/go/agent/durableagent/run.go).

### Why this design

Each step costs two fsynced appends (a few milliseconds on an SSD), negligible next to a model call. `Append` re-reads the file to find the last newline, which is $O(\text{file size})$ per append; agent runs have tens of steps, so the file stays small. A long-lived journal would keep the end offset in memory instead.

Replay by re-running the loop, rather than serializing the loop's internal state, keeps the loop one piece of code: anything it computes from step results is recomputed identically. The price is that the loop must stay deterministic given those results, with no clock reads or random choices outside steps. Refusing to guess on writes beats both "at least once" and "at most once": at-least-once duplicates tickets, and at-most-once loses them. With no idempotency key at the tool, only a human (or a tool that can be queried) knows; tools that accept an idempotency key could make retry safe and remove the human step. A file per run, rather than a database, means no dependency, a journal you can inspect with `cat`, and one writer per run. The durable engine's event log is the scalable version, and `Store` is the seam for moving to it.

One tradeoff is accepted deliberately: an fsync per record, so a step costs two syncs. Batching would be faster and would lose acknowledged steps on a crash, which is the one thing this design exists to prevent.

## What can go wrong

A correct implementation passes these checks. The hand example yields exactly seven records after Run 1, approval works, replay makes no new model calls, and a completed run calls nothing. A crash between a write and its record leaves the run `Indeterminate` until reconciled; `done` never re-runs the write and `retry` runs it once, so there are no double tickets and no lost ones. A crash inside a read re-runs the unrecorded read but not the model call before it, so reads do not block the run. The same id and input resumes, other input is refused, and an unknown run cannot be resumed, so a run id names one run. A torn last line is ignored by `Load` and cut by the next append, so a crash mid-write does not brick the run. Path-like run ids are refused, nothing is written outside the store, and signals need an existing run, since run ids come from users. Finally, under real `SIGKILL`s across five worker processes, killed inside a read and inside a write, the run makes three model calls in total and files one ticket.

The mistakes that break these guarantees:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| completed steps run again | every restart pays for every model call; tools repeat | `RunStep` returns `done[name]` without calling `fn` |
| an unrecorded write re-run, an unrecorded read treated as a write, or the start recorded after the step | two tickets after a crash; runs stuck on harmless reads | `ErrIndeterminate` only for started tool steps whose tool `IsWrite`; `step.started` appended before `fn` |
| a torn last record kept, or treated as corruption | the next record is glued onto garbage; the run cannot load | `Load` ignores a final line without a newline; `Append` truncates to the last newline |
| a `done` reconciliation that runs the write anyway | the human said it happened; it happens again | the `done` branch records the result and returns without calling `fn` |
| approve signals recorded but not applied | the run waits for approval forever | `Run` collects every `approve` marker and passes them to `WithApprovals` |
| a reused run id answering the old question | a client retrying with a new prompt gets the previous run's answer | `ErrInputMismatch` when the recorded input differs |
| run ids used as paths unchecked | `../../somewhere` writes outside the store | `ValidRunID` in `Run` and in every `FileStore` path |

> **Warning:** Appending `step.started` after the step instead of before is an easy mistake that passes every test without crashes. It only shows up when a worker dies inside a write, as a duplicate side effect.

## Where this leads

Durability is what lets the agent track's end-to-end run survive a killed worker. The [eval runner](../29-evaluation/01-eval-runner.md)'s `DurableSubject` runs each case as a durable run, so a killed eval worker resumes without re-calling the model, and [The agent image and chart](../../systems/10-containers-kubernetes/08-agent-image-and-chart.md) in *Systems Engineering* deploys the agent worker that runs durable agent workflows. The longer-term plan places the run on the durable engine itself, using [deterministic replay](../../systems/07-durable-orchestration/06-deterministic-replay.md) and [signals, cancellation, and sagas](../../systems/07-durable-orchestration/08-signals-cancellation-and-sagas.md). For now the run keeps its own journal, and adapting `Store` to the engine's event log is the step that joins them.

Production systems carry the same ideas further. Saige's durable agent (`agent/durable/{local,dbos}`) has local and DBOS backends, WAL stores, and budget reservations restored on replay. Saige's `ErrIndeterminate` and `Reconcile` (`agent/durable/local`) make the same refusal to guess, with fenced leases for remote workers. Temporal signals and updates provide validated synchronous updates and release the worker while it waits (`ErrSuspended`), and Temporal workflow replay adds determinism checks on code changes, `GetVersion`, and history paging.

### Further reading

- Temporal, [*Workflow execution: event history and replay*](https://docs.temporal.io/workflow-execution) (free).
- Temporal, [message passing](https://docs.temporal.io/encyclopedia/workflow-message-passing) (free), and the [Temporal docs](https://docs.temporal.io/) (free).
- Helland, *Life beyond Distributed Transactions* (ACM Queue, 2016, free).

## Key terms

- **journal**: the append-only, fsynced list of records that is a run's only state.
- **event sourcing**: storing state as a sequence of events and computing it as a fold over them.
- **torn tail**: a partial final record left by a crash mid-write, never acknowledged and therefore ignored.
- **replay**: answering a step from its recorded result instead of executing it.
- **indeterminate step**: a write that started and has no recorded result, whose outcome the program cannot know.
- **reconcile**: a signal in which a human records a write's outcome (`done` with a result) or says to run it again (`retry`).

## Summary

- A durable run is a journal of steps; each step records its start before it runs and its result before the result is used.
- On restart, completed steps replay from the journal, so the model is not called again and no tool runs twice.
- An unfinished read is re-run; an unfinished write stops the run with `ErrIndeterminate` until a human reconciles it.
- Approvals and reconciliations are signals, appended to the journal and applied at the next attempt.
- The journal ignores a torn final record and cuts it off before the next append; corruption elsewhere is an error.

## Review questions

1. Why must `step.started` be appended before `fn` runs?
   <details><summary>Answer</summary>If it were appended after, a crash inside the write would leave no record at all, and the next attempt would see a step that never started and run the write again: two tickets. Recording the start first turns that crash into the visible "started, not completed" state.</details>
2. In the hand trace, Run 2 makes only one model call. Which steps replay, and how does the ticket get through the gate?
   <details><summary>Answer</summary>`llm/1`, `tool/1/0/lookup`, and `llm/2` return their recorded results. `llm/2`'s recorded reply proposes `create_ticket`; the gate returns `NeedApproval` with the same marker as in Run 1 (same name and arguments), and `Run` passed that marker to `WithApprovals`, so the loop allows it. Only `llm/3` is new.</details>
3. A worker is killed inside `lookup` (a read). What does the next worker do?
   <details><summary>Answer</summary>`RunStep("tool/1/0/lookup")` finds the step started but not completed; `lookup` is not a write, so it records a new start and runs it again. The model call before it (`llm/1`) is completed and replays without a call.</details>
4. Why does a corrupt line in the middle of the file fail `Load`, while a broken last line is ignored?
   <details><summary>Answer</summary>A crash can only interrupt the last write, so a partial final line is expected and was never acknowledged. Damage earlier in the file means something else corrupted it; silently skipping a record would replay the run with a step missing, so the safe answer is to stop.</details>
5. What does a `reconcile` with `done` change, and what does the model see?
   <details><summary>Answer</summary>`RunStep` appends a `step.completed` record with the human's result and returns it without running the tool. The loop decodes it as a `tool.Result`, sets the call id and name, and appends the tool message, so the next model call sees, for example, "TICKET-7" as the tool's answer.</details>
