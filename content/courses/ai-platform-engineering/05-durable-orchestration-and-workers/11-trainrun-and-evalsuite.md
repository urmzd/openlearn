# Platform workflows: TrainRun and EvalSuite

The system can already train ([Training loop and the autograd bigram](../../build-an-llm/p00-foundations/05-training-loop-and-the-autograd-bigram.md), [Safetensors for every dtype, atomic checkpoints, and the token stream](../../build-an-llm/p00-foundations/06-safetensors-checkpoints-and-token-streams.md)) and evaluate ([LM evaluation harness and the model zoo](../../build-an-llm/p06-objectives/07-lm-evaluation-harness-and-the-model-zoo.md)) from the command line, all in *Build Your Own LLM System*. And the durable engine can now run Python as activities ([Subprocess activities](09-subprocess-activities.md)). But a training run started by hand still dies with the terminal, evaluation is something you remember to run, and nothing records which checkpoint was evaluated with which result. The [TinyStories capstone](../../build-an-llm/capstones/01-tinystories.md) trains for hours and is released by a workflow ([ModelRelease](12-model-release.md)) that must find evaluated checkpoints.

This section turns training and evaluation into platform workflows. `TrainRun` and `EvalSuite` are started by `{ctl}`, resumed after any kill, cancellable, and evaluate at fixed points of the run. End to end, `{ctl} train --spec specs/tiny.json` survives a worker kill with a bitwise-equal final loss. The workflows are short; the interesting part is the argument for why their structure, a list of segments with stable ids, gives exactly-once evaluation and exact resumption.

The workflows are written against the `Runtime` seam and `ErrCanceled` from [Signals, cancellation, and sagas](08-signals-cancellation-and-sagas.md), and `TrainRun` builds its corpus first with CorpusBuild ([Pipeline as a durable workflow CorpusBuild](../../data-engineering/05-corpus-pipeline/09-corpusbuild-as-a-durable-workflow.md) in *Data Engineering*). The reference implementation is [`go/workflows/train_run.go`](../../../../course/solutions/go/workflows/train_run.go) (`TrainRun`, `Segments`, `TrainOptions`) and [`go/workflows/eval_suite.go`](../../../../course/solutions/go/workflows/eval_suite.go) (`EvalSuite`, `EvalOptions`, `MarshalSpec`). The contracts are the schemas [`train-spec.schema.json`](../../../../course/contracts/formats/train-spec.schema.json), [`eval-spec.schema.json`](../../../../course/contracts/formats/eval-spec.schema.json), and [`eval-result.schema.json`](../../../../course/contracts/formats/eval-result.schema.json), plus the [subprocess activity spec](../../../../course/contracts/spec/subprocess-activity.md).

> **Learning objectives**
> - Derive the segment plan $b_i = \min(i \cdot e, N)$ and implement it without dropping the last segment or repeating a boundary.
> - Explain why naming each activity by its end step makes a kill replay finished segments and redeliver only the one in flight.
> - State what a checkpoint must hold for a resumed run to reach exactly the uninterrupted run's loss.
> - Structure evaluation as one activity per suite so a crash in one suite never reruns another.
> - Use a durable timer, not a wall-clock sleep, for the wait between evaluations.

## The entry points

Two entry points drive the workflows, and both are part of the solution. `{tinyllm} train --spec <dir>/spec.json --progress <file> [--resume <ckpt>]` runs under `tinyllm.io.activity.run` and resumes from `--resume`, or else from `<run>/ckpt/LATEST`; `{tinyllm} eval --spec` runs one eval spec. Both live in the activity form of the CLI, [`entry/python/tinyllm/cli_activity.py`](../../../../course/solutions/entry/python/tinyllm/cli_activity.py). The worker registers them as the activities `train` and `eval` over the subprocess runner. On the operator's side, `{ctl} train --spec <file> --id <id> [--wait]` ([`entry/go/cmd/ctl/train.go`](../../../../course/solutions/entry/go/cmd/ctl/train.go)) starts `TrainRun` with that workflow id. A spec without `out` trains in `runs/<id>`, so every segment of one run resumes its own checkpoints and another run of the same spec starts fresh.

## Segments

The first requirement is that evaluation happens at the same steps on every run, exactly once each. If evaluation were triggered from inside the trainer, a kill and resume could evaluate a step twice or skip one. Making each evaluation point a workflow step puts it in history, where replay answers it instead of rerunning it. So the run is a list of end steps, with this notation:

| Symbol | Meaning |
|---|---|
| $N$ | the spec's `steps` |
| $e$ | the spec's `eval_every` (0: no periodic evaluation) |
| $b_i = \min(i \cdot e, N)$ | the end step of segment $i$, for $i = 1, 2, \dots$ until $b_i = N$ |

`Segments(N, e)` is $[e, 2e, \dots, N]$, ending exactly at $N$ even when $e$ does not divide it; $e = 0$ or $e \ge N$ gives one segment.

A segment must stop at its boundary without the trainer knowing about segments. The trainer already stops at `steps`, so segment $i$ is one `train` activity whose spec is the run's spec with `steps` set to $b_i$ and nothing else changed. Every other field (seed, data, optimizer) is untouched, so reproducibility lives in one spec. A segment must also continue from the previous one, and here the trainer helps again: it resumes from `--resume` on a retry and from `<run>/ckpt/LATEST` otherwise, so segment 2 picks up at step 500 without the workflow passing a path.

After each segment, when the input names suites, EvalSuite evaluates the checkpoint at $b_i$. The optional `eval_interval_seconds` field then waits durably between that evaluation and the next segment, through `Runtime.Sleep`; it has no effect after the last segment. Waiting must survive a restart, which is why it is a durable timer: a wall-clock sleep inside workflow code would repeat on replay, while `Runtime.Sleep` records a timer in history, so replay skips a fired timer and resumes an unfired one with its remaining time. The run is optionally preceded by `CorpusBuild` in the same workflow, so training never starts on a missing corpus.

Each train activity also needs a distinct, stable identity. The idempotency key `<workflow_id>/<activity_id>` names the work directory, so the activity id must name the segment: `train-000500`. Zero-padding keeps ids sortable and fixed-width.

## Kills and resume

With segments named this way, every kind of failure has a clear outcome.

| What dies | What happens |
|---|---|
| the worker between activities | the workflow replays: finished segments and evaluations come from history |
| the worker during `train` | the activity times out (heartbeat) and is redelivered with attempt 2 and `--resume` the last heartbeated checkpoint |
| the server | it recovers every run from the WAL ([the event log](01-event-log.md), [the workflow service](02-workflow-service.md)) and re-enqueues what was pending |

A checkpoint holds everything the next step depends on: weights, optimizer moments, the shuffle RNG state, and the token stream's cursor. So the resumed run takes the same steps on the same batches, and its final loss equals the uninterrupted run's in every bit. The only waste of a kill is the steps since the last checkpoint.

## Cancel

`CancelWorkflow` tells the train activity in flight to stop. The runner SIGTERMs it, it checkpoints and exits 130, and TrainRun returns an error wrapping `ErrCanceled`. TrainRun has nothing to compensate: the checkpoints are work done, and a later run with the same spec continues from them.

## EvalSuite

Evaluation has its own unit of retry, the suite. EvalSuite runs one activity per suite, in the order given, each with the id `eval-<tag>-<suite>` (TrainRun's tag is the segment's end step) and a spec of that one suite. A crash in `zoo` therefore replays `ppl` from history. The first suite that fails for good fails the step, because a release gate must never read partial results as complete. Each activity writes `evals/<suite>/<run>/results.jsonl` and `summary.json` (in the format of `eval-result.schema.json`) and returns them as its outputs.

## Working it by hand

Take the spec `{"name": "tiny", "steps": 1200, "eval_every": 500, "eval_interval_seconds": 30, "seed": 7, ...}` with suites `["ppl"]`. `Segments(1200, 500)` gives $b_1 = 500$, $b_2 = 1000$, $b_3 = \min(1500, 1200) = 1200$. The workflow records a 30 second durable wait after the first two evaluations. The commands, in order, are:

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

That is 1200 optimizer steps in all, with each checkpoint evaluated once. If the worker dies during a timer wait, replay reads the recorded sleep command and resumes with the same remaining timer, without repeating the evaluation. If it dies at step 740 of `train-001000` (last checkpoint 700), the activity is redelivered with `--resume runs/tiny/ckpt/step-000700`; steps 701 to 740 are redone, the run ends at the same final loss, and `eval-000500-ppl` is not rerun.

The code below shows where each of these steps comes from.

## Designing the API

Listing 1 is the whole workflow API.

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

## Building it step by step

### The segment plan

A loop strictly below `steps` plus a final `steps` gives the shorter last segment without ever repeating a boundary (Listing 2).

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

For $N = 1000$, $e = 500$ the loop yields 500 only (1000 is not $< 1000$), and the append adds 1000, so there is no duplicate.

### Activity options

The id names the end step. A heartbeat timeout of one minute lets the subprocess runner beat every 20 s, and five attempts survive a few OOMs (Listing 3).

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

### Validate before any activity

A spec error is a `*SpecError` whose `NonRetryable()` is true, returned before the corpus build, so a typo costs nothing (Listing 4).

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

### The segment loop

`withField` rewrites only `steps` in the raw spec; decoding into a `map[string]json.RawMessage` keeps every other field's value as raw bytes, untouched. Errors are wrapped with `%w`, so a cancel's `ErrCanceled` still matches at the top (Listing 5).

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

There is no cleanup on cancel: the checkpoints are work done, and the subprocess runner has already made the activity checkpoint on its way out.

### EvalSuite

One activity per suite, each with a spec of that suite only, stopping at the first failure (Listing 6).

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

`MarshalSpec` builds the same per-suite spec for the worker and the CLI's dry run. The full solutions are [`train_run.go`](../../../../course/solutions/go/workflows/train_run.go) and [`eval_suite.go`](../../../../course/solutions/go/workflows/eval_suite.go).

### Why the resume is bitwise

The workflow only guarantees that the right segment is redelivered with the right `--resume`. Bit equality comes from the trainer ([`cli_activity.py`](../../../../course/solutions/entry/python/tinyllm/cli_activity.py)), which restores everything the next step depends on (Listing 7).

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

### Why this design

A run of $N$ steps with period $e$ records $\lceil N/e \rceil$ train activities plus one eval activity per suite per segment, and at most $\lceil N/e \rceil - 1$ timers, so history grows with the number of segments, not with steps. A kill wastes at most the steps since the last checkpoint (`ckpt_every`), independent of $e$.

One long activity with evaluation inside the trainer would be simpler, but evaluation would then live outside history: a resume could re-evaluate or skip a step, and ModelRelease could not find "the evaluation of step 1000" as a recorded result. Going the other way, one activity per checkpoint, means more history and more process start-ups (loading the model, the corpus) for no gain. The segment size is the evaluation period because that is where the workflow needs to act.

Two tradeoffs are accepted. Evaluation of segment $i$ blocks segment $i+1$, sequential rather than overlapped with training. And a failed suite fails the whole run, even though training could continue.

## What can go wrong

These are the cases a correct solution must handle, with the downstream reason for each.

| Case | Kind of check | What must hold | Why it matters downstream |
|---|---|---|---|
| the worked example's segments and evaluations | unit | the example's ids, checkpoints, 1200 steps, one evaluation per checkpoint | ModelRelease finds every checkpoint's evaluation |
| the eval interval is a durable sleep | fault | the configured gap is recorded between evaluations and is not repeated on replay | the durable timer fires once |
| segment arithmetic | boundary | boundaries end at $N$, no repeats, $e = 0$ and $e \ge N$ | evaluation points are the same on every run |
| each segment gets the spec with its end | unit | only `steps` changes between segments | reproducibility lives in the spec |
| the kill loop resumes bitwise | fault | a crash at every position: same result and exactly the same final loss | the bitwise end-to-end check |
| a retry resumes from the latest checkpoint | fault | a segment that dies halfway resumes, no steps redone beyond the last checkpoint | hours of training are not repeated |
| cancel stops the run | fault | canceled after the current activity; no further segment; nothing deleted | a cancelled capstone keeps its checkpoints |
| corpus first | unit | CorpusBuild runs first; its failure stops training | no training on a half-built corpus |
| a bad spec fails fast | boundary | no name, no steps, negative `eval_every` or `eval_interval_seconds`: non-retryable, no activity | a typo does not burn a GPU-hour |
| one eval activity per suite | unit | one activity per suite, replay after a crash, the spec of one suite, a failed suite stops the rest | the zoo suite's crash keeps ppl's results |
| activity ids and options | unit | `train-000500`, `eval-000500-ppl`, `eval-zoo`; heartbeat timeouts set | idempotency keys are unique per segment and suite |

The mistakes that break these cases are mostly small. Dropping the last, shorter segment or repeating a boundary makes the run stop at 1000 of 1200 steps, or evaluate step 1200 twice; `Segments` loops strictly below `steps` and then appends `steps` once. Training every segment to the end makes the first segment run the whole run, so every evaluation sees the final model; `withField` sets `steps` to the segment's end.

> **Warning:** Activity ids that do not name the segment or suite make segments share one work directory, and a replay can answer one segment's call with another's result. Use `train-%06d` and `eval-<tag>-<suite>`.

Retry and timeout settings matter for long runs. With no retries for train, one OOM ends a day-long run, so `TrainOptions` sets `MaxAttempts: 5`. With no heartbeat timeout, a dead trainer is noticed only after a day; `HeartbeatTimeout: time.Minute` stands against a 24 h `StartToClose`.

Control flow is the rest. Treating a cancel as "move on" trains every remaining segment of a cancelled run; any `ExecuteActivity` error returns at once, wrapped with `%w`. Ignoring a failed corpus build starts training on missing token files; `CorpusBuild` runs first and its error returns. Without spec validation, a typo fails after the corpus build, or never; name, steps, and intervals are checked before any activity, as a non-retryable `SpecError`. Putting all suites in one activity makes a crash in the last suite rerun every suite; `EvalSuite` gives each suite its own activity and spec. Continuing after a failed suite lets a release gate see partial results as complete; `EvalSuite` returns at the first error. And using a wall-clock sleep, or skipping the durable cadence, means a worker restart repeats or loses the evaluation wait; use `rt.Sleep`, only between segments.

## Where this leads

These workflows sit on the `Runtime` and `ErrCanceled` of [Signals, cancellation, and sagas](08-signals-cancellation-and-sagas.md), CorpusBuild from *Data Engineering*, and the subprocess runner's `--resume` together with the checkpoints of [Safetensors for every dtype, atomic checkpoints, and the token stream](../../build-an-llm/p00-foundations/06-safetensors-checkpoints-and-token-streams.md) that it resumes from.

Going forward, the [TinyStories capstone](../../build-an-llm/capstones/01-tinystories.md) runs as a TrainRun on the TinyStories config through `{ctl}`. [ModelRelease](12-model-release.md), the next section, evaluates the candidate with EvalSuite. [Paired A/B experiments](../09-llm-evaluation/12-ab-experiments.md) extend EvalSuite with scorers and A/B options. And in [Observability for the control plane](../../systems/04-observability/05-control-plane-observability.md) in *Systems and Architecture*, one TrainRun is one trace down to `train.step`.

Production systems take each piece further. Kubeflow Pipelines, Flyte, and Metaflow generalize segments with evaluation into DAGs of steps with caching by input hash and artifact lineage (see Flyte's "cached tasks"). PyTorch Lightning's fault-tolerant training and TorchElastic (`torch.distributed.elastic`) extend resume-from-LATEST with restart on node loss, rendezvous of many workers, and sharded checkpoints. And lm-evaluation-harness and HELM scale one-activity-per-suite evaluation to thousands of tasks, cached model outputs, and aggregate leaderboards (EleutherAI `lm-evaluation-harness`).

### Further reading

- [Temporal: long-running activities and heartbeating](https://docs.temporal.io/encyclopedia/detecting-activity-failures).
- [PyTorch: saving and loading a general checkpoint](https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint_for_inference_and_training.html).

## Key terms

- **Segment**: one `train` activity that trains from the previous checkpoint to the end step $b_i = \min(i \cdot e, N)$.
- **Evaluation period**: the spec's `eval_every`, the number of steps between evaluations and therefore the segment size.
- **Eval cadence**: the optional durable wait, `eval_interval_seconds`, between an evaluation and the next segment.
- **Bitwise resume**: resuming from a checkpoint that holds weights, optimizer moments, RNG state, and data cursor, so the final loss equals the uninterrupted run's exactly.
- **Suite activity**: one `eval` activity per evaluation suite, the unit of retry for evaluation.

## Summary

- A training run is a workflow of segments: train to step $e$, evaluate that checkpoint, train to $2e$, and so on. Evaluation happens at the same steps on every run, and each evaluation runs once.
- Every segment's activity id names its end step (`train-000500`), so a kill replays finished segments and redelivers the one in flight, which resumes from its last heartbeated checkpoint.
- Resuming from a checkpoint that holds the model, the optimizer, the RNG state, and the data cursor gives exactly the loss of an uninterrupted run, the property the end-to-end kill loop checks bit for bit.
- EvalSuite runs one activity per suite: a crash in the middle of the zoo suite does not rerun the finished ppl suite.
- A cancel stops the run after the activity in flight checkpointed; nothing is deleted.

## Review questions

1. Why does `Segments` loop with `s < steps` and then append `steps`, rather than looping with `s <= steps`?
   <details><summary>Answer</summary>With `<=` and $e$ not dividing $N$ the loop would stop at 1000 and drop the shorter last segment (the run ends at 1000 of 1200). Appending `steps` unconditionally fixes that, and the strict bound keeps $N$ from appearing twice when $e$ divides it.</details>
2. What goes wrong if every segment's activity id were just `train`?
   <details><summary>Answer</summary>All segments would share one idempotency key and so one work directory. Segment 2 would find segment 1's `DONE.json` and return at once as a duplicate, and a replay could answer one segment's call with another's result.</details>
3. Why is `eval_interval_seconds` implemented with `rt.Sleep` and not `time.Sleep`?
   <details><summary>Answer</summary>Workflow code is replayed after every restart. `time.Sleep` would wait again on each replay and is invisible to history; `rt.Sleep` records a timer, so a fired timer is skipped on replay and an unfired one resumes with its remaining time.</details>
4. Where does the "bitwise-equal final loss" guarantee actually come from?
   <details><summary>Answer</summary>Not from the workflow, which only redelivers the right segment with `--resume`. It comes from the checkpoint holding weights, optimizer moments, the shuffle RNG state, and the data cursor, so the resumed trainer takes the same steps on the same batches with the same arithmetic.</details>
5. Why does `EvalSuite` stop at the first failed suite instead of collecting what it can?
   <details><summary>Answer</summary>Its result feeds a release gate (ModelRelease). A result missing a suite would look like a complete evaluation; failing makes the gap explicit, and the finished suites are still in history for a retry of the run.</details>
