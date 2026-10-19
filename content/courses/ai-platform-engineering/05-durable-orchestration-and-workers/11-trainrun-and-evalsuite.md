# Platform workflows TrainRun and EvalSuite

## Overview

| | |
|---|---|
| **Module** | `dur.11` · build · Go · Pass 8 · reading time 1.5 to 2 h |
| **Solution** | [`go/workflows/train_run.go`](../../../../course/solutions/go/workflows/train_run.go): `TrainRun`, `Segments`, `TrainOptions`; [`go/workflows/eval_suite.go`](../../../../course/solutions/go/workflows/eval_suite.go): `EvalSuite`, `EvalOptions`, `MarshalSpec`; the entry points that drive them: the activity form of the `{tinyllm}` CLI ([`entry/python/tinyllm/cli_activity.py`](../../../../course/solutions/entry/python/tinyllm/cli_activity.py), `train --spec` and `eval --spec`) and `{ctl} train` ([`entry/go/cmd/ctl/train.go`](../../../../course/solutions/entry/go/cmd/ctl/train.go)) |
| **Contract** | [`course/contracts/formats/train-spec.schema.json`](../../../../course/contracts/formats/train-spec.schema.json), [`course/contracts/formats/eval-spec.schema.json`](../../../../course/contracts/formats/eval-spec.schema.json), [`course/contracts/formats/eval-result.schema.json`](../../../../course/contracts/formats/eval-result.schema.json), and [`course/contracts/spec/subprocess-activity.md`](../../../../course/contracts/spec/subprocess-activity.md) |
| **Needs** | `dur.08` (the `Runtime` seam, `ErrCanceled`), `data.09` (TrainRun builds its corpus with CorpusBuild); reading: `dur.09` (the subprocess runner and `--resume`), `L0.6` (atomic checkpoints, the token cursor), `L6.7` (the eval suites) |
| **Used by** | `dur.12` (ModelRelease evaluates the candidate with EvalSuite); `ag.12` upgrades EvalSuite with paired experiments; `C1` runs the capstone as a TrainRun through `{ctl}`. End to end, `{ctl} train --spec specs/tiny.json` survives a worker kill with a bitwise-equal final loss. |
| **Optional depth** | [Temporal: long-running activities and heartbeating](https://docs.temporal.io/encyclopedia/detecting-activity-failures) (free); [PyTorch: saving and loading a general checkpoint](https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint_for_inference_and_training.html) (free) |

## Key Takeaways

- A training run is a workflow of **segments**: train to step $e$, evaluate that checkpoint, train to $2e$, and so on. Evaluation happens at the **same steps on every run**, and each evaluation runs **once**.
- Every segment's activity id names its end step (`train-000500`), so a kill **replays** finished segments and **redelivers** the one in flight, which resumes from its last heartbeated checkpoint.
- Resuming from a checkpoint that holds the model, the optimizer, the RNG state, and the data cursor gives **exactly the loss** of an uninterrupted run: the property the end-to-end kill loop checks bit for bit.
- EvalSuite runs **one activity per suite**: a crash in the middle of the zoo suite does not rerun the finished ppl suite.
- A cancel stops the run after the activity in flight **checkpointed**; nothing is deleted.

## How to read this chapter

Sections 1 to 3 derive the segment plan and the resume argument and walk one run by hand. Section 4 fixes the workflow API. Section 5 walks through `TrainRun` and `EvalSuite` and explains why each choice is the right one. Try to predict each step of the solution before reading it.

The workflows are driven by two entry points that are part of the solution. `{tinyllm} train --spec <dir>/spec.json --progress <file> [--resume <ckpt>]` runs under `tinyllm.io.activity.run` and resumes from `--resume`, or else from `<run>/ckpt/LATEST`; `{tinyllm} eval --spec` runs one eval spec. The worker registers them as the activities `train` and `eval` over the `dur.09` runner. `{ctl} train --spec <file> --id <id> [--wait]` starts `TrainRun` with that workflow id; a spec without `out` trains in `runs/<id>`, so every segment of one run resumes its own checkpoints and another run of the same spec starts fresh.

---

## 1. Why now

The system can train (`L0.5`, `L0.6`) and evaluate (`L6.7`) from the command line, and the durable engine can run Python as activities (`dur.09`). But a training run started by hand still dies with the terminal, evaluation is something you remember to run, and nothing records which checkpoint was evaluated with which result. The capstone (`C1`) trains for hours and is released by a workflow (`dur.12`) that must find evaluated checkpoints. `TrainRun` and `EvalSuite` make training and evaluation platform workflows: started by `{ctl}`, resumed after any kill, cancellable, and with evaluation at fixed points of the run.

## 2. Principles

### 2.1 Segments

| Symbol | Meaning |
|---|---|
| $N$ | the spec's `steps` |
| $e$ | the spec's `eval_every` (0: no periodic evaluation) |
| $b_i = \min(i \cdot e, N)$ | the end step of segment $i$, for $i = 1, 2, \dots$ until $b_i = N$ |

`Segments(N, e)` is $[e, 2e, \dots, N]$, ending exactly at $N$ even when $e$ does not divide it; $e = 0$ or $e \ge N$ is one segment. Segment $i$ is one `train` activity whose spec is the run's spec with `steps` set to $b_i$ and nothing else changed; it resumes from the run directory's `LATEST` checkpoint (the previous segment's) and stops at $b_i$. Then, when the input names suites, EvalSuite evaluates the checkpoint at $b_i$. The optional `eval_interval_seconds` field durably waits between that evaluation and the next segment, through `Runtime.Sleep`; it has no effect after the last segment. The run is optionally preceded by `CorpusBuild` (`data.09`) in the same workflow, so training never starts on a missing corpus.

### 2.2 Kills and resume

| What dies | What happens |
|---|---|
| the worker between activities | the workflow replays: finished segments and evaluations come from history |
| the worker during `train` | the activity times out (heartbeat) and is redelivered with attempt 2 and `--resume` the last heartbeated checkpoint (`dur.09`) |
| the server | it recovers every run from the WAL (`dur.01`, `dur.02`) and re-enqueues what was pending |

A checkpoint holds everything the next step depends on: weights, optimizer moments, the shuffle RNG state, and the token stream's cursor (`L0.6`). So the resumed run takes the same steps on the same batches, and its final loss equals the uninterrupted run's in every bit. The only waste of a kill is the steps since the last checkpoint.

### 2.3 Cancel

`CancelWorkflow` (dur.08) tells the train activity in flight to stop; the runner SIGTERMs it, it checkpoints and exits 130, and TrainRun returns an error wrapping `ErrCanceled`. TrainRun has nothing to compensate: the checkpoints are work done, and a later run with the same spec continues from them.

### 2.4 EvalSuite

One activity per suite, in the order given, each with the id `eval-<tag>-<suite>` (TrainRun's tag is the segment's end step) and a spec of that one suite. The first suite that fails for good fails the step. Each activity writes `evals/<suite>/<run>/results.jsonl` and `summary.json` (formats/eval-result.schema.json) and returns them as its outputs.

## 3. Worked example by hand

Spec `{"name": "tiny", "steps": 1200, "eval_every": 500, "eval_interval_seconds": 30, "seed": 7, ...}` with suites `["ppl"]`. `Segments(1200, 500)`: $b_1 = 500$, $b_2 = 1000$, $b_3 = \min(1500, 1200) = 1200$. The workflow records a 30 second durable wait after the first two evaluations. The commands, in order:

```
train-000500     spec steps 500          -> runs/tiny/ckpt/step-000500
eval-000500-ppl  subject tiny@500
sleep 30s        durable timer before the next segment
train-001000     spec steps 1000         -> resumes from step-000500
eval-001000-ppl
sleep 30s        durable timer before the next segment
train-001200     spec steps 1200
eval-001200-ppl
```

1200 optimizer steps in all, each checkpoint evaluated once. If the worker dies during a timer wait, replay reads the recorded sleep command and resumes with the same remaining timer, without repeating the evaluation. If it dies at step 740 of `train-001000` (last checkpoint 700), the activity is redelivered with `--resume runs/tiny/ckpt/step-000700`; steps 701 to 740 are redone, the run ends at the same final loss, and `eval-000500-ppl` is not rerun.

Section 5 shows where each of these steps comes from in the workflow code.

## 4. The interface

```go
const (
	ActivityTrain = "train"
	ActivityEval  = "eval"
)
type TrainRunInput struct { Spec, Corpus json.RawMessage; EvalSuites []string; EvalSeed int64 }
type TrainRunResult struct { Name string; Steps int; Checkpoint string; Segments []Segment; Corpus *CorpusBuildResult }
type Segment struct { Until int; Checkpoint string; Evals *EvalSuiteResult }
func Segments(steps, evalEvery int) []int
func TrainOptions(until int) StepOptions
func TrainRun(rt Runtime, in TrainRunInput) (TrainRunResult, error)

type EvalSubject struct { ID, Model string }
type EvalSuiteInput struct { Name, Tag string; Suites []string; Subjects []EvalSubject; Seed int64 }
type EvalSuiteResult struct { Suites []SuiteOutputs }
func EvalOptions(tag, suite string) StepOptions
func EvalSuite(rt Runtime, in EvalSuiteInput) (EvalSuiteResult, error)
func MarshalSpec(in EvalSuiteInput, suite string) ([]byte, error)
```

### What a correct solution must handle

| Case | Kind of check | What must hold | Why it matters downstream |
|---|---|---|---|
| the hand example's segments and evaluations | unit | section 3's ids, checkpoints, 1200 steps, one evaluation per checkpoint | `dur.12` finds every checkpoint's evaluation |
| the eval interval is a durable sleep | fault | the configured gap is recorded between evaluations and is not repeated on replay | `dur.07` fires the timer once |
| segment arithmetic | boundary | boundaries end at $N$, no repeats, $e = 0$ and $e \ge N$ | evaluation points are the same on every run |
| each segment gets the spec with its end | unit | only `steps` changes between segments | reproducibility lives in the spec |
| the kill loop resumes bitwise | fault | a crash at every position: same result and exactly the same final loss | the bitwise end-to-end check |
| a retry resumes from the latest checkpoint | fault | a segment that dies halfway resumes, no steps redone beyond the last checkpoint | hours of training are not repeated |
| cancel stops the run | fault | canceled after the current activity; no further segment; nothing deleted | a cancelled capstone keeps its checkpoints |
| corpus first | unit | CorpusBuild runs first; its failure stops training | no training on a half-built corpus |
| a bad spec fails fast | boundary | no name, no steps, negative `eval_every` or `eval_interval_seconds`: non-retryable, no activity | a typo does not burn a GPU-hour |
| one eval activity per suite | unit | one activity per suite, replay after a crash, the spec of one suite, a failed suite stops the rest | the zoo suite's crash keeps ppl's results |
| activity ids and options | unit | `train-000500`, `eval-000500-ppl`, `eval-zoo`; heartbeat timeouts set | idempotency keys are unique per segment and suite |

## 5. The solution, step by step

### 5.1 From first principles

1. **Evaluation must happen at the same steps on every run, exactly once each.** If evaluation were triggered from inside the trainer, a kill and resume could evaluate a step twice or skip one. Making each evaluation point a workflow step puts it in history, where replay answers it instead of rerunning it. So the run is a list of end steps, $b_i = \min(i \cdot e, N)$.
2. **Each train activity needs a distinct, stable identity.** The idempotency key `<workflow_id>/<activity_id>` names the work directory (`dur.09`), so the activity id must name the segment: `train-000500`. Zero-padding keeps ids sortable and fixed-width.
3. **A segment must stop at its boundary without the trainer knowing about segments.** The trainer already stops at `steps`; giving segment $i$ the run's spec with only `steps` changed to $b_i$ reuses that. Every other field (seed, data, optimizer) is untouched, so reproducibility lives in one spec.
4. **A segment must continue from the previous one.** The trainer resumes from `--resume` on a retry and from `<run>/ckpt/LATEST` otherwise, so segment 2 picks up step 500 without the workflow passing a path.
5. **Waiting between evaluations must survive a restart.** A wall-clock sleep inside workflow code would repeat on replay; `Runtime.Sleep` records a timer in history, so replay skips a fired timer and resumes an unfired one with its remaining time.
6. **A suite is the unit of retry for evaluation.** One activity per suite means a crash in `zoo` replays `ppl` from history; the first suite that fails for good stops the step, because a release gate must never read partial results as complete.

### 5.2 The code, annotated

**The segment plan.** A loop strictly below `steps` plus a final `steps` gives the shorter last segment without ever repeating a boundary:

```go
func Segments(steps, evalEvery int) []int {
	if evalEvery <= 0 || evalEvery >= steps {
		return []int{steps} // no periodic evaluation, or one period covers the run
	}
	var out []int
	for s := evalEvery; s < steps; s += evalEvery { // strictly below: 500, 1000 for N = 1200
		out = append(out, s)
	}
	return append(out, steps) // 1200, never 1000 twice when e divides N
}
```

For $N = 1000$, $e = 500$ the loop yields 500 only (1000 is not $< 1000$), and the append adds 1000: no duplicate.

**Activity options.** The id names the end step; the heartbeat timeout of one minute lets the `dur.09` runner beat every 20 s, and five attempts survive a few OOMs:

```go
func TrainOptions(until int) StepOptions {
	return StepOptions{ID: fmt.Sprintf("train-%06d", until), StartToClose: 24 * time.Hour, HeartbeatTimeout: time.Minute, MaxAttempts: 5}
}

func EvalOptions(tag, suite string) StepOptions {
	id := "eval-" + suite
	if tag != "" {
		id = fmt.Sprintf("eval-%s-%s", tag, suite) // "eval-000500-ppl"
	}
	return StepOptions{ID: id, StartToClose: 6 * time.Hour, HeartbeatTimeout: 2 * time.Minute, MaxAttempts: 3}
}
```

**Validate before any activity.** A spec error is a `*SpecError` whose `NonRetryable()` is true, returned before the corpus build, so a typo costs nothing:

```go
	var spec trainSpec
	if err := json.Unmarshal(in.Spec, &spec); err != nil {
		return TrainRunResult{}, &SpecError{Reason: "spec is not a JSON object: " + err.Error()}
	}
	if spec.Name == "" || spec.Steps < 1 || spec.EvalEvery < 0 || spec.EvalIntervalSeconds < 0 {
		return TrainRunResult{}, &SpecError{Reason: /* ... */}
	}
	res := TrainRunResult{Name: spec.Name, Steps: spec.Steps}
	if len(in.Corpus) > 0 {
		cb, err := CorpusBuild(rt, CorpusBuildInput{Config: in.Corpus}) // a child sequence in the same run
		if err != nil {
			return res, fmt.Errorf("corpus: %w", err) // no training on a half-built corpus
		}
		res.Corpus = &cb
	}
```

**The segment loop.** `withField` rewrites only `steps` in the raw spec (decoding into a `map[string]json.RawMessage` keeps every other field's value as raw bytes, untouched). Errors are wrapped with `%w`, so a cancel's `ErrCanceled` still matches at the top:

```go
	segments := Segments(spec.Steps, spec.EvalEvery)
	for i, until := range segments {
		seg, err := withField(in.Spec, "steps", until)
		// ...
		var done StageDone
		if err := rt.ExecuteActivity(ActivityTrain, json.RawMessage(seg), TrainOptions(until), &done); err != nil {
			return res, fmt.Errorf("train to step %d: %w", until, err) // a cancel ends here: no further segment
		}
		if len(done.Outputs) == 0 {
			return res, fmt.Errorf("train to step %d: no checkpoint in the result", until)
		}
		s := Segment{Until: until, Checkpoint: done.Outputs[0]} // DONE.json's first output
		if len(in.EvalSuites) > 0 {
			ev, err := EvalSuite(rt, EvalSuiteInput{
				Tag: fmt.Sprintf("%06d", until), Suites: in.EvalSuites, Seed: in.EvalSeed,
				Subjects: []EvalSubject{{ID: fmt.Sprintf("%s@%d", spec.Name, until), Model: s.Checkpoint}},
			})
			if err != nil {
				return res, fmt.Errorf("eval at step %d: %w", until, err)
			}
			s.Evals = &ev
			// A durable timer, and none after the last segment.
			if spec.EvalIntervalSeconds > 0 && i+1 < len(segments) {
				if err := rt.Sleep(time.Duration(spec.EvalIntervalSeconds) * time.Second); err != nil {
					return res, fmt.Errorf("eval cadence after step %d: %w", until, err)
				}
			}
		}
		res.Segments = append(res.Segments, s)
		res.Checkpoint = s.Checkpoint
	}
	return res, nil
```

There is no cleanup on cancel: the checkpoints are work done, and the `dur.09` runner has already made the activity checkpoint on its way out.

**EvalSuite.** One activity per suite, each with a spec of that suite only, stopping at the first failure:

```go
func EvalSuite(rt Runtime, in EvalSuiteInput) (EvalSuiteResult, error) {
	if len(in.Suites) == 0 || len(in.Subjects) == 0 {
		return EvalSuiteResult{}, &SpecError{Reason: "an eval suite needs suites and subjects"}
	}
	name := in.Name
	if name == "" {
		name = "evalsuite"
	}
	var res EvalSuiteResult
	for _, suite := range in.Suites {
		spec := evalSpec{Name: name, Suites: []string{suite}, Subjects: in.Subjects, Seed: in.Seed}
		applyAdvancedSpecOptions(&spec, in) // scorers, judge, A/B, thresholds (ag.12)
		var done StageDone
		if err := rt.ExecuteActivity(ActivityEval, spec, EvalOptions(in.Tag, suite), &done); err != nil {
			return res, fmt.Errorf("suite %s: %w", suite, err) // later suites never run
		}
		res.Suites = append(res.Suites, SuiteOutputs{Suite: suite, Outputs: done.Outputs})
	}
	return res, nil
}
```

`MarshalSpec` builds the same per-suite spec for the worker and the CLI's dry run. Full solutions: [`train_run.go`](../../../../course/solutions/go/workflows/train_run.go), [`eval_suite.go`](../../../../course/solutions/go/workflows/eval_suite.go).

**Why the resume is bitwise.** The workflow only guarantees that the right segment is redelivered with the right `--resume`. Bit equality comes from the trainer ([`cli_activity.py`](../../../../course/solutions/entry/python/tinyllm/cli_activity.py)), which restores everything the next step depends on:

```python
    if act.resume:
        m = re.search(r"step-(\d+)$", act.resume.rstrip("/"))
        # ...
        resume = load_checkpoint(str(under(act, act.resume).parent), int(m.group(1)))
    elif (ckpt / "LATEST").is_file():
        resume = load_checkpoint(str(ckpt))  # the previous segment's last checkpoint
    if resume is not None:
        model.load_state_dict(resume.model)   # weights
        opt.load_state_dict(resume.opt)       # AdamW moments and step count
        step, tokens_seen = int(resume.step), int(resume.extra["tokens_seen"])
        cur = resume.extra["data_cursor"]
        stream.restore({"shard": cur["shard"], "offset": cur["offset"], "rng": rng_tuple(resume.rng_state)})  # same batches next
```

### 5.3 Why this solution

- **Cost.** A run of $N$ steps with period $e$ records $\lceil N/e \rceil$ train activities plus one eval activity per suite per segment, and at most $\lceil N/e \rceil - 1$ timers: history grows with the number of segments, not with steps. A kill wastes at most the steps since the last checkpoint (`ckpt_every`), independent of $e$.
- **Segments over one long activity with in-trainer evaluation.** One activity is simpler, but evaluation would then live outside history: a resume could re-evaluate or skip a step, and `dur.12` could not find "the evaluation of step 1000" as a recorded result.
- **Segments over one activity per checkpoint.** Finer activities mean more history and more process start-ups (loading the model, the corpus) for no gain; the segment size is the evaluation period because that is where the workflow needs to act.
- **Accepted tradeoffs.** Evaluation of segment $i$ blocks segment $i+1$ (sequential, not overlapped with training); a failed suite fails the whole run, even though training could continue.

### 5.4 Explain it back

1. Why does `Segments` loop with `s < steps` and then append `steps`, rather than looping with `s <= steps`?
   <details><summary>Answer</summary>With `<=` and $e$ not dividing $N$ the loop would stop at 1000 and drop the shorter last segment (the run ends at 1000 of 1200). Appending `steps` unconditionally fixes that, and the strict bound keeps $N$ from appearing twice when $e$ divides it.</details>
2. What goes wrong if every segment's activity id were just `train`?
   <details><summary>Answer</summary>All segments would share one idempotency key and so one work directory. Segment 2 would find segment 1's `DONE.json` and return at once as a duplicate, and a replay could answer one segment's call with another's result.</details>
3. Why is `eval_interval_seconds` implemented with `rt.Sleep` and not `time.Sleep`?
   <details><summary>Answer</summary>Workflow code is replayed after every restart. `time.Sleep` would wait again on each replay and is invisible to history; `rt.Sleep` records a timer, so a fired timer is skipped on replay and an unfired one resumes with its remaining time.</details>
4. Where does the "bitwise-equal final loss" guarantee actually come from?
   <details><summary>Answer</summary>Not from the workflow, which only redelivers the right segment with `--resume`. It comes from the checkpoint holding weights, optimizer moments, the shuffle RNG state, and the data cursor, so the resumed trainer takes the same steps on the same batches with the same arithmetic.</details>
5. Why does `EvalSuite` stop at the first failed suite instead of collecting what it can?
   <details><summary>Answer</summary>Its result feeds a release gate (`dur.12`). A result missing a suite would look like a complete evaluation; failing makes the gap explicit, and the finished suites are still in history for a retry of the run.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. the last, shorter segment dropped or a boundary repeated | the run stops at 1000 of 1200 steps, or evaluates step 1200 twice | `Segments` loops strictly below `steps`, then appends `steps` once |
| 2. activity ids that do not name the segment or suite | segments share one work directory; a replay answers the wrong one | `train-%06d` and `eval-<tag>-<suite>` |
| 3. every segment trains to the end | the first segment runs the whole run; evaluations see the final model | `withField` sets `steps` to the segment's end |
| 4. no retries for train | one OOM ends a day-long run | `TrainOptions` sets `MaxAttempts: 5` |
| 5. a cancel treated as "move on" | the cancelled run trains every remaining segment | any `ExecuteActivity` error returns at once, wrapped with `%w` |
| 6. a failed corpus build ignored | training starts on missing token files | `CorpusBuild` runs first and its error returns |
| 7. no spec validation | a typo fails after the corpus build, or never | name, steps, and intervals are checked before any activity, as a non-retryable `SpecError` |
| 8. all suites in one activity | a crash in the last suite reruns every suite | `EvalSuite` gives each suite its own activity and spec |
| 9. continuing after a failed suite | a release gate sees partial results as complete | `EvalSuite` returns at the first error |
| 10. no heartbeat timeout on train | a dead trainer is noticed after a day | `HeartbeatTimeout: time.Minute` against a 24 h `StartToClose` |
| 11. using a wall-clock sleep or skipping the durable cadence | a worker restart repeats or loses the evaluation wait | `rt.Sleep`, only between segments |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `dur.08` | `Runtime`, `ErrCanceled` |
| Back | `data.09` | CorpusBuild before training |
| Back | `dur.09`, `L0.6` | the subprocess runner's `--resume` and the checkpoints it resumes from |
| Forward | `C1` | the capstone run is a TrainRun on the TinyStories config |
| Forward | `dur.12` | ModelRelease evaluates the candidate with EvalSuite |
| Forward | `ag.12` | paired experiments extend EvalSuite with scorers and A/B options |
| Forward | `obs.05` | one TrainRun is one trace down to `train.step` |

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| segments with evaluation | Kubeflow Pipelines, Flyte, Metaflow | DAGs of steps with caching by input hash, artifact lineage | Flyte "cached tasks" |
| resume from LATEST | PyTorch Lightning fault-tolerant training, TorchElastic | restart on node loss, rendezvous of many workers, sharded checkpoints | `torch.distributed.elastic` |
| one eval activity per suite | lm-evaluation-harness, HELM | thousands of tasks, cached model outputs, aggregate leaderboards | EleutherAI `lm-evaluation-harness` |
