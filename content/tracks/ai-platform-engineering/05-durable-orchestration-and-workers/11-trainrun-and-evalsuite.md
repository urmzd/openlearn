<!-- ol:module dur.11 -->
# Platform workflows TrainRun and EvalSuite

## Overview

| | |
|---|---|
| **Module** | `dur.11` · build · Go · Pass 8 · 4 to 6 h |
| **You build** | `go/workflows/train_run.go`: `TrainRun`, `Segments`, `TrainOptions`; `go/workflows/eval_suite.go`: `EvalSuite`, `EvalOptions`, `MarshalSpec`; then the activity form of your `{tinyllm}` CLI (`train --spec` and `eval --spec`, entry-point territory) and `{ctl} train --spec`, `{ctl} eval --suite` |
| **Contract** | [`course/contracts/formats/train-spec.schema.json`](../../../../course/contracts/formats/train-spec.schema.json), [`course/contracts/formats/eval-spec.schema.json`](../../../../course/contracts/formats/eval-spec.schema.json), [`course/contracts/formats/eval-result.schema.json`](../../../../course/contracts/formats/eval-result.schema.json), and [`course/contracts/spec/subprocess-activity.md`](../../../../course/contracts/spec/subprocess-activity.md) |
| **Tests** | `course/tests/go/dur_11/` (TrainRun and EvalSuite through the replaying simulator, over fake train and eval activities with a deterministic loss; section 4) |
| **Needs** | `dur.08` (the `Runtime` seam, `ErrCanceled`), `data.09` (TrainRun builds its corpus with CorpusBuild); reading: `dur.09` (the subprocess runner and `--resume`), `L0.6` (atomic checkpoints, the token cursor), `L6.7` (the eval suites) |
| **Used by** | `dur.12` (ModelRelease evaluates the candidate with EvalSuite); `ag.12` upgrades EvalSuite with paired experiments; `C1` runs the capstone as a TrainRun through `{ctl}` |
| **Milestone** | MS-durable (`{ctl} train --spec specs/tiny.json` survives a worker kill with a bitwise-equal final loss) |
| **Optional depth** | [Temporal: long-running activities and heartbeating](https://docs.temporal.io/encyclopedia/detecting-activity-failures) (free); [PyTorch: saving and loading a general checkpoint](https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint_for_inference_and_training.html) (free) |

## Key Takeaways

- A training run is a workflow of **segments**: train to step $e$, evaluate that checkpoint, train to $2e$, and so on. Evaluation happens at the **same steps on every run**, and each evaluation runs **once**.
- Every segment's activity id names its end step (`train-000500`), so a kill **replays** finished segments and **redelivers** the one in flight, which resumes from its last heartbeated checkpoint.
- Resuming from a checkpoint that holds the model, the optimizer, the RNG state, and the data cursor gives **exactly the loss** of an uninterrupted run: the property MS-durable checks bit for bit.
- EvalSuite runs **one activity per suite**: a crash in the middle of the zoo suite does not rerun the finished ppl suite.
- A cancel stops the run after the activity in flight **checkpointed**; nothing is deleted.

## How to work this chapter

```bash
ol start dur.11          # stubs go/workflows/{train_run,eval_suite}.go
ol tests dur.11
ol check dur.11
ol diff  dur.11
```

Then wire your entries: `{tinyllm} train --spec <dir>/spec.json --progress <file> [--resume <ckpt>]` under `tinyllm.io.activity.run` (resume from `--resume`, else from `<run>/ckpt/LATEST`), `{tinyllm} eval --spec`, the activities `train` and `eval` in your worker over the dur.09 runner, and `{ctl} train --spec <file>` starting `TrainRun` with workflow id `train/<name>`.

---

## 1. Why now

You can train (`L0.5`, `L0.6`) and evaluate (`L6.7`) from the command line, and your durable engine can run Python as activities (`dur.09`). But a training run started by hand still dies with the terminal, evaluation is something you remember to run, and nothing records which checkpoint was evaluated with which result. The capstone (`C1`) trains for hours and is released by a workflow (`dur.12`) that must find evaluated checkpoints. `TrainRun` and `EvalSuite` make training and evaluation platform workflows: started by `{ctl}`, resumed after any kill, cancellable, and with evaluation at fixed points of the run.

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

This is `TestHandExampleSegmentsAndEvals`.

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

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `TestHandExampleSegmentsAndEvals` | unit | section 3's ids, checkpoints, 1200 steps, one evaluation per checkpoint | `dur.12` finds every checkpoint's evaluation |
| `TestEvalIntervalUsesDurableSleepAndReplaysOnce` | fault | the configured gap is recorded between evaluations and is not repeated on replay | `dur.07` fires the timer once |
| `TestSegmentsMath` | boundary | boundaries end at $N$, no repeats, $e = 0$ and $e \ge N$ | evaluation points are the same on every run |
| `TestEachSegmentGetsTheSpecWithItsEnd` | unit | only `steps` changes between segments | reproducibility lives in the spec |
| `TestKillLoopResumesBitwise` | fault | a crash at every position: same result and exactly the same final loss | MS-durable's bitwise check |
| `TestRetryResumesFromLatest` | fault | a segment that dies halfway resumes, no steps redone beyond the last checkpoint | hours of training are not repeated |
| `TestCancelStopsTheRun` | fault | canceled after the current activity; no further segment; nothing deleted | a cancelled capstone keeps its checkpoints |
| `TestCorpusFirst` | unit | CorpusBuild runs first; its failure stops training | no training on a half-built corpus |
| `TestBadSpecFailsFast` | boundary | no name, no steps, negative `eval_every` or `eval_interval_seconds`: non-retryable, no activity | a typo does not burn a GPU-hour |
| `TestEvalSuiteOneActivityPerSuite` | unit | one activity per suite, replay after a crash, the spec of one suite, a failed suite stops the rest | the zoo suite's crash keeps ppl's results |
| `TestOptionsIDs` | unit | `train-000500`, `eval-000500-ppl`, `eval-zoo`; heartbeat timeouts set | idempotency keys are unique per segment and suite |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. the last, shorter segment dropped or a boundary repeated | the run stops at 1000 of 1200 steps, or evaluates step 1200 twice | `TestHandExampleSegmentsAndEvals`, `TestSegmentsMath` (mutants `s01`, `s04`) |
| 2. activity ids that do not name the segment or suite | segments share one work directory; a replay answers the wrong one | `TestHandExampleSegmentsAndEvals`, `TestOptionsIDs` (mutants `s02`, `s03`) |
| 3. every segment trains to the end | the first segment runs the whole run; evaluations see the final model | `TestEachSegmentGetsTheSpecWithItsEnd` (mutant `s05`) |
| 4. no retries for train | one OOM ends a day-long run | `TestRetryResumesFromLatest` (mutant `s06`) |
| 5. a cancel treated as "move on" | the cancelled run trains every remaining segment | `TestCancelStopsTheRun` (mutant `s07`) |
| 6. a failed corpus build ignored | training starts on missing token files | `TestCorpusFirst` (mutant `s08`) |
| 7. no spec validation | a typo fails after the corpus build, or never | `TestBadSpecFailsFast` (mutant `s09`) |
| 8. all suites in one activity | a crash in the last suite reruns every suite | `TestEvalSuiteOneActivityPerSuite` (mutant `s10`) |
| 9. continuing after a failed suite | a release gate sees partial results as complete | `TestEvalSuiteOneActivityPerSuite` (mutant `s11`) |
| 10. no heartbeat timeout on train | a dead trainer is noticed after a day | `TestOptionsIDs` (mutant `s12`) |
| 11. using a wall-clock sleep or skipping the durable cadence | a worker restart repeats or loses the evaluation wait | `TestEvalIntervalUsesDurableSleepAndReplaysOnce` (mutant `s13`) |

## 6. Where it's used next

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

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| segments with evaluation | Kubeflow Pipelines, Flyte, Metaflow | DAGs of steps with caching by input hash, artifact lineage | Flyte "cached tasks" |
| resume from LATEST | PyTorch Lightning fault-tolerant training, TorchElastic | restart on node loss, rendezvous of many workers, sharded checkpoints | `torch.distributed.elastic` |
| one eval activity per suite | lm-evaluation-harness, HELM | thousands of tasks, cached model outputs, aggregate leaderboards | EleutherAI `lm-evaluation-harness` |
