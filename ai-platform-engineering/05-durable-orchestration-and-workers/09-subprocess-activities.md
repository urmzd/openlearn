<!-- ol:module dur.09 -->
# Subprocess activity runner (Go) and the Python activity helper

## Overview

| | |
|---|---|
| **Module** | `dur.09` · build · Go and Python · Pass 8 · 5 to 7 h |
| **You build** | `go/activities/subprocess.go`: `Subprocess.Run`, `Task`, `Heartbeat`, `Failure`, `WorkDir`, `Tail`, `Classify`; `python/tinyllm/io/activity.py`: `Activity`, `run`, the exit codes; `python/tinyllm/io/telemetry.py`: `Tracer`, `parse_traceparent`, OTLP/HTTP JSON export |
| **Contract** | [`course/contracts/spec/subprocess-activity.md`](../../course/contracts/spec/subprocess-activity.md) (the interface between the two halves), [`course/contracts/py/tinyllm/io/activity.pyi`](../../course/contracts/py/tinyllm/io/activity.pyi), [`course/contracts/py/tinyllm/io/telemetry.pyi`](../../course/contracts/py/tinyllm/io/telemetry.pyi), and the spans of [`otel/semconv.md`](../../course/contracts/otel/semconv.md) |
| **Tests** | `course/tests/go/dur_09/` (the runner, against a scripted fake child) and `course/tests/dur.09/` (the Python helper and the exporter); what they check: section 4 |
| **Needs** | `L0.6` atomic checkpoints (`verify_step_dir` decides when a checkpoint may be reported); reading: `lang.02` exit codes and signals, the worker and activity SDK of `dur.04` and `dur.05` (the runner is plugged into them by your worker's composition root) |
| **Used by** | `data.09` (every corpus stage runs through the runner) · `dur.11` (`train` and `eval` activities) · `dep.06` (the worker image carries these units) · `obs.05` (runs `telemetry.py` as the Python end of the control-plane trace); later `dur.12` (`export`) and `C1` (the capstone run is a `TrainRun`) |
| **Milestone** | MS-durable (`<system> train --spec specs/tiny.json` survives a worker kill) |
| **Optional depth** | [Temporal: activity heartbeats and cancellation](https://docs.temporal.io/encyclopedia/detecting-activity-failures) (free); [W3C Trace Context](https://www.w3.org/TR/trace-context/) (free); [OTLP specification, JSON encoding](https://opentelemetry.io/docs/specs/otlp/#json-protobuf-encoding) (free); W. Richard Stevens, *Advanced Programming in the UNIX Environment*, ch. 9 and 10 (process groups, signals) |

## Key Takeaways

- Python never speaks gRPC to the platform (D9). A Go worker runs it as a **child process** and the whole interface is files, environment variables, signals, and an **exit code**.
- The exit code is the verdict: `65` never retries, `75` and crashes retry, `130` is a cancellation. A **worker drain is not a cancellation**, so the runner reports it as retryable even when the child exits 130.
- A checkpoint becomes a **heartbeat detail** the moment it is complete, and the next attempt gets it as `--resume`. Until a newer one exists, every heartbeat keeps sending the old one.
- Every attempt shares one **work directory** named by the idempotency key. Outputs are published by **rename**, `DONE.json` last, and an **exclusive lock** keeps an orphaned child from writing beside its successor.
- The Python spans continue the activity's trace from **TRACEPARENT**, honor its sampled flag, and a dead collector never fails training.

## How to work this chapter

```bash
ol start dur.09         # stubs go/activities/subprocess.go and python/tinyllm/io/{activity,telemetry}.py
ol tests dur.09         # read the test catalog first
ol check dur.09         # Go tests (fake child) and Python tests; the exit code is the verdict
ol diff  dur.09         # after passing: your code against the reference
```

Then wire it in your worker's composition root (`go/cmd/worker`, learner territory). For each Python verb, register an activity that adapts the delivery to the runner:

```go
r := activities.Subprocess{Entry: tinyllmEntry, Artifacts: "/artifacts", ServiceName: name + "-python",
	OTLPEndpoint: os.Getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
	Canceled: func(cause error) bool { return errors.Is(cause, worker.ErrCancelRequested) }}
w.RegisterActivity("train", func(ctx context.Context, spec []byte) ([]byte, error) {
	inf, _ := worker.InfoFrom(ctx)
	task := activities.Task{IdempotencyKey: inf.IdempotencyKey, Attempt: inf.Attempt,
		LastHeartbeat: inf.HeartbeatDetails, TraceContext: inf.TraceContext, HeartbeatTimeout: hbTimeout}
	return r.Run(ctx, task, []string{"train"}, spec, func(_ context.Context, d []byte) (bool, error) {
		return false, worker.Heartbeat(ctx, d) // a cancel arrives as ctx's cause, read by r.Canceled
	})
})
```

In your Python entries, wrap `train`, `eval`, `export`, and `corpus run --stage` in `activity.run`.

---

## 1. Why now

Your durable engine can run Go activities with retries, heartbeats, and fenced leases (`dur.03` to `dur.05`), but everything that matters to the platform's users is Python: the corpus stages from Pass 3 and the trainer from Pass 2. Today `{tinyllm} train` is a command you type. If the laptop sleeps at step 9,000, the run is gone, and nothing records which checkpoint was last good. If you start it from a Go activity with `exec.Command` and wait, three things break at once: the server sees no heartbeats and fences the activity mid-run, a worker restart kills the child without a checkpoint (or worse, leaves it running), and a retry starts again from step 0 in a fresh directory. This module is the bridge: a Go runner and a Python helper that keep one written contract, so `CorpusBuild` (`data.09`) and `TrainRun` (`dur.11`) can run Python work durably.

## 2. Principles

### 2.1 The boundary is a process

A **subprocess activity** is one attempt at one activity, run as a child process of the worker. The runner gives the child an argv, an environment, and a work directory; the child gives back a progress file, maybe some outputs, and an exit code. Nothing else crosses: no shared memory, no RPC. That is why the Python side stays testable without a cluster (you can run `{tinyllm} train --spec ... --progress ...` by hand) and why one durable protocol (gRPC, Go only) is enough.

| Symbol | Meaning | Type |
|---|---|---|
| $k$ | the idempotency key `<workflow_id>/<activity_id>` | string |
| $a$ | the attempt number, starting at 1 | int |
| $T_{hb}$ | the activity's heartbeat timeout | duration |
| $h = T_{hb}/3$ | the longest gap allowed between heartbeats | duration |
| $g$ | the grace period between SIGTERM and SIGKILL, 30 s by default | duration |

### 2.2 One work directory per activity, shared by every attempt

The work directory is `<TL_ARTIFACTS>/activities/<k>/`, each `/`-separated segment of $k$ a directory (`WorkDir`). A segment must match `[A-Za-z0-9._-]+` and must not be `.` or `..`: workflow ids are caller-chosen text, and a key like `../../etc` would otherwise escape the artifact root. Every attempt $a = 1, 2, \dots$ of one activity lands in the same directory, which is what makes a retry a **resume** rather than a restart.

Three rules make the directory safe to share:

1. **Publish by rename.** An output is written to `name.tmp`, flushed and fsynced, then renamed to `name`, and the directory is fsynced. `rename` is atomic within one filesystem: a reader sees the old file or the whole new one. The runner writes `spec.json` the same way.
2. **`DONE.json` last.** The last act of a successful run publishes `DONE.json` (`{"outputs": [...]}`), then emits the `done` event, then exits 0. A run that starts and finds `DONE.json` re-emits `done` and exits 0 without doing anything: a duplicate delivery has one effect.
3. **One attempt at a time.** The helper takes an exclusive, non-blocking `flock` on `<dir>/.lock` before any work. If a previous attempt still holds it (a child orphaned by a SIGKILLed worker), the new attempt exits 75 and is retried later.

### 2.3 Progress, checkpoints, and heartbeats

The child appends one JSON object per line to `--progress` and **flushes after each line**: the runner reads the file while it grows, so an event stuck in Python's buffer is invisible exactly when it matters (the worker is killed one second later). The runner's `Tail` keeps a line back until its newline arrives: half a line is not an event.

A `ckpt` event names a checkpoint step directory, relative to `TL_ARTIFACTS`. The helper emits it **only after** `L0.6`'s `verify_step_dir` says the directory is complete (its `MANIFEST.json` lists every file with the right size and hash). The runner treats each new `ckpt` path as the **heartbeat details** and heartbeats it at once; it also heartbeats at least every $h$ so the server does not fence the activity between checkpoints (training checkpoints every few minutes; heartbeat timeouts are tens of seconds). On a retry the server hands back the last details, and the runner passes them as `--resume <ckpt>`. Until the child reports a newer checkpoint, every heartbeat repeats the old one: an empty heartbeat would erase the resume point on the server.

### 2.4 Exit codes

| Code | Meaning | `Failure` |
|---|---|---|
| `0` and `DONE.json` present | success | none: the result is the bytes of `DONE.json` (at most 2 MiB) |
| `0` without `DONE.json` | a broken entry | `MissingDone`, retryable |
| `65` (`EX_DATAERR`) | the input is wrong: invalid spec, unlicensed source | `ExitCode65`, **non-retryable** |
| `75` (`EX_TEMPFAIL`) | transient: a flaky download, an OOM the next attempt may avoid | `ExitCode75`, retryable |
| `130` | cancelled | `Canceled`, non-retryable |
| any other code, or a signal | a crash | `ExitCode<n>` or `Signal<n>`, retryable |

`Failure.message` is the last 4 KiB of the child's stderr, where a traceback ends. On the Python side `run(main)` maps exceptions to the same codes: `SpecError` 65, `RetryableError` 75, `Cancelled` 130, anything else 1.

### 2.5 Stopping a child: process groups, SIGTERM, SIGKILL

The runner starts the child with `Setpgid`, so the child leads a new **process group** whose id is its pid. Signals go to the whole group (`kill(-pgid, sig)`), which reaches anything the child started (a tokenizer pool, a data loader worker), and a signal meant for the worker's own group does not reach the child behind the runner's back.

There are two reasons to stop a child, and they are **different results**:

| Why | How the runner learns it | Result |
|---|---|---|
| the activity is cancelled (a workflow cancel) | a heartbeat answer with `cancel_requested` | `Canceled`, non-retryable |
| the worker is going away (SIGTERM drain on a deploy, or a lost lease) | `ctx` ends | `WorkerShutdown`, **retryable**: another worker resumes from the checkpoint |

The worker SDK of `dur.04` reports a `cancel_requested` heartbeat answer by cancelling the activity's context with its own cause; `Subprocess.Canceled` tells the runner which causes mean "cancelled" (everything else is a shutdown). Either way the runner sends SIGTERM to the group, waits up to $g$, then sends SIGKILL. The Python handler only sets a flag; the training loop notices it at its next step, saves a checkpoint, emits its `ckpt` event, and raises `Cancelled` (exit 130). The runner heartbeats that last checkpoint even after `ctx` has ended (on a fresh context with a short timeout) and returns it in `Failure.Details` for `FailActivityRequest.last_heartbeat_details`. Track the last acknowledged checkpoint separately from the last one read: a stopping poll or failed heartbeat can leave an unreported checkpoint even when the final file read is empty.

A worker that dies outright cannot stop its child. The child notices instead: its parent pid changes (it is reparented), and `Activity.cancelled` becomes true.

### 2.6 Telemetry: one trace from the CLI to the training step

The worker passes the activity span's W3C context in `TRACEPARENT`: `00-<trace-id>-<parent-id>-<flags>`, lowercase hex, 32 and 16 digits, neither all zeros, flags bit 0 meaning **sampled**. The exporter makes `train.run` (or `corpus.stage <s>`) a child of that span in the same trace, `train.step` a child of `train.run` every 50th step, and posts them as OTLP/HTTP JSON to `OTEL_EXPORTER_OTLP_ENDPOINT/v1/traces` (gauges such as `tl.train.loss` to `/v1/metrics`: a subprocess cannot be scraped). Three rules:

- An invalid `TRACEPARENT` starts a **new root** trace; it is never parsed leniently.
- An **unsampled** parent means export nothing: the decision belongs to the root of the trace.
- Export failures return `False` and drop the batch. Telemetry never raises into the work it observes.

## 3. Worked example by hand

Activity `train-1/3` (workflow `train-1`, activity id `3`) with `TL_ARTIFACTS=/artifacts`, heartbeat timeout 150 ms, so $h$ = 50 ms.

**Attempt 1.** The runner computes the work directory `/artifacts/activities/train-1/3/`, publishes `spec.json` there, and execs

```
<entry> train --spec /artifacts/activities/train-1/3/spec.json --progress /artifacts/activities/train-1/3/progress.jsonl
```

with `TL_IDEMPOTENCY_KEY=train-1/3`, `TL_ATTEMPT=1`. No `--resume`: the task has no heartbeat details. The child writes

```
{"ts":...,"kind":"step","step":500,"loss":2.31,"lr":0.0009,"tokens":8192000}
{"ts":...,"kind":"ckpt","step":500,"ckpt":"runs/train-1/ckpt/step-000500"}
```

The tail reads the `ckpt` line within one poll, and the runner heartbeats with details `runs/train-1/ckpt/step-000500`. Then the child runs out of memory and exits 75 with `CUDA out of memory` on stderr. The result is `Failure{Type: "ExitCode75", NonRetryable: false, Message: "...CUDA out of memory..."}`: retry.

**Attempt 2.** The server redelivers with `attempt = 2` and `last_heartbeat_details = runs/train-1/ckpt/step-000500`. The runner uses the same directory and execs the same argv plus `--resume runs/train-1/ckpt/step-000500`, with `TL_ATTEMPT=2`. The helper (clock fixed at 1760000000.0) writes exactly

```
{"ts":1760000000.0,"kind":"step","step":600,"loss":2.25,"lr":0.0009,"tokens":9830400}
{"ts":1760000000.0,"kind":"ckpt","step":1000,"ckpt":"runs/train-1/ckpt/step-001000"}
{"ts":1760000000.0,"kind":"done","outputs":["runs/train-1/ckpt/step-001000"]}
```

and `DONE.json` is the 45 bytes `{"outputs":["runs/train-1/ckpt/step-001000"]}`, which become the activity's result. A third delivery (the worker died before `CompleteActivityTask`) finds `DONE.json`, re-emits `done`, and exits 0 without training.

**Telemetry.** With `TRACEPARENT=00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01` (the W3C example) and span ids drawn from a counter, `train.run` gets span id `0000000000000001` and parent `00f067aa0ba902b7`; the sampled step 50 (step 49 is not sampled) gets `0000000000000002` with parent `0000000000000001`; both carry trace id `4bf92f3577b34da6a3ce929d0e0e4736`. In the OTLP JSON body, `tl.train.step = 50` is `{"intValue": "50"}`: a string, because JSON numbers lose precision past $2^{53}$.

These numbers are `TestHandExampleResumeFlow`, `test_hand_example_progress_lines`, and `test_hand_example_span_tree`.

## 4. The interface

```go
// go/activities/subprocess.go
type Task struct {
	IdempotencyKey   string
	Attempt          int
	LastHeartbeat    []byte
	TraceContext     map[string]string
	HeartbeatTimeout time.Duration
}
type Heartbeat func(ctx context.Context, details []byte) (cancelRequested bool, err error)
type Subprocess struct {
	Entry []string; Artifacts, OTLPEndpoint, ServiceName string
	Grace, Poll time.Duration; Env []string; Dir string
	Canceled func(cause error) bool // is this end of ctx a cancel (not a shutdown)?
}
func (s Subprocess) Run(ctx context.Context, t Task, args []string, spec []byte, hb Heartbeat) ([]byte, error) // *Failure
type Failure struct { Type, Message string; NoRetry bool; ExitCode, Signal int; Details []byte }
func (f *Failure) FailureType() string  // what the worker SDK reports as Failure.type
func (f *Failure) NonRetryable() bool   // and as Failure.non_retryable
func WorkDir(artifacts, key string) (string, error)
type Tail struct{ Path string }        // Read() ([]Progress, error): complete lines only
func Classify(state *os.ProcessState, stderr []byte) *Failure
```

```python
# python/tinyllm/io/activity.py
class Activity:            # Activity(argv, env=None, clock=time.time)
    def load_spec(self, validate=None) -> dict: ...
    def step(self, step, loss, lr, tokens) -> None: ...
    def checkpoint(self, step, path) -> str: ...
    def publish(self, name, data) -> str: ...
    def done(self, outputs) -> bytes: ...
    @property
    def cancelled(self) -> bool: ...
def run(main, argv=None, env=None) -> int: ...

# python/tinyllm/io/telemetry.py
class Tracer:              # Tracer.from_env(); with tracer.span("train.run", {...}) as s: ...
    def flush(self) -> bool: ...
def parse_traceparent(value) -> SpanContext | None: ...
def should_sample_step(step, every=50) -> bool: ...
```

The runner takes a `Task` and a `Heartbeat` function instead of importing the worker SDK, so it is tested here against a scripted fake child, and your worker's composition root adapts each delivery to it (a dozen lines per verb).

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `TestHandExampleResumeFlow` | unit | section 3: attempt 1 heartbeats its checkpoint and fails retryable; attempt 2 gets `--resume`, `TL_ATTEMPT=2`, the same directory, and returns `DONE.json` | `TrainRun` resumes from the last checkpoint after a kill (`dur.11`) |
| `TestWorkDirFromKey` | boundary | key segments become directories; `..`, `.`, empty, and odd characters are rejected, non-retryable | a workflow id can never write outside `/artifacts/activities` |
| `TestEnvironmentContract` | unit | `TL_*`, `TRACEPARENT`, OTEL variables set per task; the worker's own copies never leak | Python spans land in the right trace (`obs.05`) |
| `TestExitCodeTable` | unit | every row of the table in section 2.4, signals included | a poison input fails fast; a transient one retries |
| `TestSuccessNeedsDoneFile` | boundary | exit 0 without `DONE.json` is a retryable `MissingDone`; over 2 MiB is non-retryable; exactly 2 MiB is fine | results fit in a gRPC message and a history event |
| `TestTailKeepsPartialLines` | unit | half a line waits for its newline; garbage lines are skipped | heartbeats never carry half a path |
| `TestHeartbeatCarriesNewCheckpoints` | unit | each new checkpoint is heartbeated while the child runs, even a line written in two pieces | a kill one second after a checkpoint loses nothing |
| `TestResumePointSurvivesUntilANewCheckpoint` | fault | a retry's heartbeats repeat the old resume point until a newer one exists | a kill before the retry's first checkpoint does not restart from step 0 |
| `TestHeartbeatEveryThirdOfTimeout` | unit | at least one heartbeat per $T_{hb}/3$ | the server does not fence a long step |
| `TestCancelRequestedStopsTheChild` | fault | `cancel_requested` sends SIGTERM; the run ends `Canceled` with the exit checkpoint heartbeated | `dur.08` cancel reaches Python |
| `TestGraceThenSIGKILL` | fault | a child ignoring SIGTERM is SIGKILLed after the grace, grandchild included | a stuck child never holds a worker slot |
| `TestOwnProcessGroup` | unit | the child leads its own group | group signals reach grandchildren only |
| `TestWorkerShutdownIsRetryable` | fault | `ctx` ending gives a retryable `WorkerShutdown` with `Details` = the exit checkpoint | a deploy does not end a training run for good |
| `TestShutdownFlushesUnacknowledgedCheckpoint` | fault | A failed checkpoint heartbeat is retried on shutdown even when the final tail read is empty | resume state reaches the server |
| `TestCancelCauseFromContext` | fault | with `Subprocess.Canceled` set, a ctx cancelled with the SDK's cancel cause ends `Canceled`; any other cause stays `WorkerShutdown` | the worker SDK's cancel reaches the child as a cancel |
| `TestSpecWrittenAtomically` | unit | `spec.json` replaced by rename, never rewritten in place | a retry never shows half a spec |
| `TestStderrTailIsBounded` | boundary | the last 4 KiB of stderr, traceback included | failures stay small in the history |
| `test_hand_example_progress_lines` | unit | section 3's three progress lines and `DONE.json`, byte for byte | the runner and the schema read these bytes |
| `test_flags_and_environment` | unit | `--spec`, `--progress`, `--resume` (both spellings), `TL_*` defaults; missing flags are `SpecError` | the entry and the runner agree on argv |
| `test_each_event_is_one_flushed_line` | unit | each event is on disk as one line when `emit` returns | the tail sees checkpoints at once |
| `test_rejects_bad_events` | boundary | unknown kinds and NaN or inf values raise, nothing written | a NaN loss never corrupts the progress file |
| `test_checkpoint_only_when_complete` | unit | no event for an incomplete directory; paths relative to `TL_ARTIFACTS`; outside paths rejected | the resume point is always loadable (`L0.6`) |
| `test_publish_never_exposes_a_partial_file` | fault | when the rename never happens, the final name does not exist | `DONE.json` exists only for finished work |
| `test_outputs_stay_in_the_work_dir` | boundary | `..`, absolute, and empty names rejected | one activity cannot overwrite another's files |
| `test_run_exit_codes` | unit | `SpecError` 65, `RetryableError` 75, `Cancelled` 130, crash 1, success 0 with `DONE.json` | the Go side's table has the right inputs |
| `test_spec_validation_is_exit_65` | unit | validation errors and non-object or bad JSON specs are 65, before any output | invalid specs do not burn retries |
| `test_rerun_after_done_is_a_noop` | regression | a second run finds `DONE.json`, skips `main`, re-emits `done` | a duplicate delivery has one effect |
| `test_sigterm_checkpoints_and_exits_130` | fault | SIGTERM, then a checkpoint as the last event, then exit 130 within 5 s | cancel and drain keep the work done so far |
| `test_second_attempt_is_locked_out` | fault | while one attempt holds the lock, another exits 75 without running | no two children write one directory |
| `test_orphaned_child_stops` | fault | a child whose parent died sees `cancelled` | an orphan does not train on beside its successor |
| `test_hand_example_span_tree` | conformance | section 3's ids, parents, kinds, and JSON encoding at a real HTTP receiver | `obs.05` finds `train.step` under `activity train` |
| `test_parse_traceparent` | boundary | valid, unsampled, future-version, and nine invalid headers | invalid input starts a new trace |
| `test_format_roundtrip` | unit | `format_traceparent` and `parse_traceparent` agree | contexts can be handed on |
| `test_unsampled_parent_exports_nothing` | boundary | flags `00`: no spans, no gauges sent | no orphan spans |
| `test_no_traceparent_starts_a_root` | boundary | no or bad `TRACEPARENT`: a fresh trace id and no parent | hand runs still trace |
| `test_from_env` | unit | endpoint, service name, resource attributes, parent from the environment | the runner's variables are the only input |
| `test_dead_collector_never_fails_the_work` | fault | connection refused: `flush` returns `False` within its timeout, queue dropped | training survives a collector outage |
| `test_collector_error_status_is_false` | fault | a 500 answer is `False` | drops are visible |
| `test_error_marks_the_span` | unit | an exception sets status ERROR and `error.type`, and propagates | failed stages show in the trace |
| `test_step_sampling` | unit | every 50th step | span volume stays bounded |
| `test_gauges_are_pushed` | unit | the latest gauge value per name and attributes goes to `/v1/metrics` | `tl.train.loss` reaches Prometheus |
| `test_shutdown_stops_export` | unit | nothing is queued after `shutdown` | clean exit |
| `test_body_shape_without_parent` | unit | a root span has no `parentSpanId`; attribute value types | the collector accepts the body |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. checkpoints never reach a heartbeat | after a kill, the retry starts from step 0 | `TestHandExampleResumeFlow`, `TestHeartbeatCarriesNewCheckpoints` (mutant `s01`) |
| 2. the key's segments are not checked | a workflow id `../../x` writes outside `/artifacts` | `TestWorkDirFromKey` (mutant `s02`) |
| 3. `--resume` is not passed | retries ignore the checkpoint | `TestHandExampleResumeFlow` (mutant `s03`) |
| 4. the worker's `TRACEPARENT` is inherited | Python spans join an unrelated trace | `TestEnvironmentContract` (mutant `s04`) |
| 5. a wrong row in the exit-code table | a poison spec retries until the dead-letter queue; a cancel restarts | `TestExitCodeTable`, `test_run_exit_codes` (mutants `s05`, `s06`, `s24`) |
| 6. exit 0 trusted without `DONE.json` | a broken entry "succeeds" with no outputs | `TestSuccessNeedsDoneFile` (mutant `s07`) |
| 7. half a progress line is parsed | a checkpoint event is lost forever | `TestTailKeepsPartialLines` (mutant `s08`) |
| 8. a retry heartbeats empty details | the resume point is erased before the first new checkpoint | `TestResumePointSurvivesUntilANewCheckpoint` (mutant `s09`) |
| 9. heartbeats only on checkpoints | the server fences the activity mid-step | `TestHeartbeatEveryThirdOfTimeout` (mutant `s10`) |
| 10. `cancel_requested` ignored | a cancelled workflow keeps training | `TestCancelRequestedStopsTheChild` (mutant `s11`) |
| 11. no process group, or no SIGKILL | grandchildren and stuck children outlive the activity | `TestGraceThenSIGKILL`, `TestOwnProcessGroup` (mutants `s12`, `s13`) |
| 12. a drain reported as a cancel, or a cancel as a drain | every deploy ends a training run for good; a cancelled run is retried | `TestWorkerShutdownIsRetryable`, `TestCancelCauseFromContext` (mutants `s14`, `s40`) |
| 13. `spec.json` written in place | a reader sees a truncated spec | `TestSpecWrittenAtomically` (mutant `s15`) |
| 14. all of stderr kept | a chatty child blows the 4 MiB message limit | `TestStderrTailIsBounded` (mutant `s16`) |
| 15. progress not flushed | the runner sees nothing until exit | `test_each_event_is_one_flushed_line` (mutant `s17`) |
| 16. absolute paths in events or `DONE.json` | a path that is right on one pod is wrong on the next | `test_hand_example_progress_lines`, `test_checkpoint_only_when_complete` (mutants `s18`, `s21`) |
| 17. a NaN loss written as `NaN` | the progress line is not JSON | `test_rejects_bad_events` (mutant `s19`) |
| 18. a ckpt event before the checkpoint is complete | the retry cannot load its resume point | `test_checkpoint_only_when_complete` (mutant `s20`) |
| 19. outputs written under their final name | a kill leaves a truncated `DONE.json` that looks finished | `test_publish_never_exposes_a_partial_file` (mutant `s22`) |
| 20. output names not confined | one activity overwrites another's files | `test_outputs_stay_in_the_work_dir` (mutant `s23`) |
| 21. spec validation errors ignored | a bad spec trains on garbage, or retries | `test_spec_validation_is_exit_65` (mutant `s25`) |
| 22. no `DONE.json` check on start | a duplicate delivery trains again | `test_rerun_after_done_is_a_noop` (mutant `s26`) |
| 23. the SIGTERM handler exits at once | the work since the last checkpoint is lost | `test_sigterm_checkpoints_and_exits_130` (mutant `s27`) |
| 24. no work-dir lock | an orphan and its successor write one directory | `test_second_attempt_is_locked_out` (mutant `s28`) |
| 25. orphans not detected | a child trains on after its worker died | `test_orphaned_child_stops` (mutant `s29`) |
| 26. the remote parent ignored | Python spans form a separate trace | `test_hand_example_span_tree` (mutant `s30`) |
| 27. `intValue` as a JSON number | collectors reject or round the attribute | `test_hand_example_span_tree`, `test_body_shape_without_parent` (mutant `s31`) |
| 28. lenient `traceparent` parsing | all-zero ids glue unrelated runs together | `test_parse_traceparent` (mutant `s32`) |
| 29. the sampled flag ignored | orphan spans whose parents were never recorded | `test_unsampled_parent_exports_nothing` (mutant `s33`) |
| 30. `OTEL_SERVICE_NAME` ignored | every system's Python is `tinyllm-python` | `test_from_env` (mutant `s34`) |
| 31. export errors raised | a collector outage kills training | `test_dead_collector_never_fails_the_work`, `test_collector_error_status_is_false` (mutant `s35`) |
| 32. exceptions swallowed by the span | a failing stage looks successful | `test_error_marks_the_span` (mutant `s36`) |
| 33. sampling off by one | step 0 never traced, step 1 always | `test_step_sampling` (mutant `s37`) |
| 34. the first gauge value kept | dashboards show the loss of step 1 | `test_gauges_are_pushed` (mutant `s38`) |
| 35. spans queued after shutdown | memory grows at exit; spans half sent | `test_shutdown_stops_export` (mutant `s39`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L0.6` | `verify_step_dir` gates every `ckpt` event; `--resume` loads with `load_checkpoint` |
| Back | `dur.04`, `dur.05` | the worker and activity SDK your composition root adapts to `Task` and `Heartbeat` |
| Forward | `data.09` | `CorpusBuild` runs each corpus stage as `{corpus} run --stage <s>` through the runner |
| Forward | `dur.11` | `TrainRun` and `EvalSuite` run `{tinyllm} train` and `eval`; kills resume from the heartbeated checkpoint |
| Forward | `obs.05` | the control-plane trace reaches `train.step` through this exporter |
| Forward | `dep.06` | the worker image carries the activity helper and the exporter |
| Forward | `dur.12`, `C1` | `export` and the capstone run use the same contract |

If you skip this module, `ol check data.09` fails with `needs dur.09: build it, or pass --ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `Subprocess.Run` | Temporal activity heartbeats with details | heartbeat throttling, details as typed payloads, async completion for work finished elsewhere | Temporal Go SDK `activity.RecordHeartbeat`, `activity.GetHeartbeatDetails` |
| the exit-code table | Kubernetes Job `podFailurePolicy` | exit-code rules that fail the Job, ignore the failure, or count it | Kubernetes docs, "Handling retriable and non-retriable pod failures" |
| process groups and grace | systemd `KillMode=control-group`, Kubernetes `terminationGracePeriodSeconds` | cgroups instead of process groups: no child can escape by calling `setsid` | systemd.kill(5) |
| orphan detection by parent pid | Linux `PR_SET_PDEATHSIG` | the kernel signals the child when the parent thread dies | prctl(2) |
| `telemetry.py` | the OpenTelemetry Python SDK with the OTLP/HTTP exporter | batching span processor, retries with backoff, protobuf encoding | `opentelemetry-exporter-otlp-proto-http` |
