# Process contract and recovery

This part of [Subprocess activities: the Go runner and the Python helper](README.md) develops the worked mechanisms below.

> **Learning objectives**
> - Define subprocess inputs, durable progress, termination, and telemetry.

## The boundary is a process

A platform design rule says Python never speaks gRPC to the platform. Instead, a **subprocess activity** is one attempt at one activity, run as a child process of a Go worker. The runner gives the child an argv, an environment, and a work directory; the child gives back a progress file, maybe some outputs, and an exit code. Nothing else crosses: no shared memory, no RPC. That is why the Python side stays testable without a cluster (you can run `{tinyllm} train --spec ... --progress ...` by hand) and why one durable protocol, gRPC in Go only, is enough.

A few quantities recur through the rest of the section, so it helps to name them now.

| Symbol | Meaning | Type |
|---|---|---|
| $k$ | the idempotency key `<workflow_id>/<activity_id>` | string |
| $a$ | the attempt number, starting at 1 | int |
| $T_{hb}$ | the activity's heartbeat timeout | duration |
| $h = T_{hb}/3$ | the longest gap allowed between heartbeats | duration |
| $g$ | the grace period between SIGTERM and SIGKILL, 30 s by default | duration |

The runner does not import the worker SDK. The worker's composition root ([`entry/go/cmd/worker/main.go`](../../../../../course/solutions/entry/go/cmd/worker/main.go)) builds one `Subprocess` per Python entry and adapts each delivery to it (Listing 1).

```go
canceled := func(cause error) bool { return errors.Is(cause, worker.ErrCancelRequested) }
py := activities.Subprocess{Entry: tinyllm, Artifacts: artifacts, OTLPEndpoint: full.OTel.Endpoint, ServiceName: os.Getenv("OTEL_SERVICE_NAME"), Canceled: canceled}

// subprocess adapts a dur.09 runner to an activity: the delivery's Info is
// the runner's Task, and its heartbeats go through the worker SDK (a cancel
// arrives as the context's cause).
func subprocess(r activities.Subprocess, args func(in []byte) []string) worker.ActivityFunc {
	return func(ctx context.Context, in []byte) ([]byte, error) {
		inf, _ := worker.InfoFrom(ctx)
		t := activities.Task{
			IdempotencyKey: inf.IdempotencyKey, Attempt: inf.Attempt,
			LastHeartbeat: inf.HeartbeatDetails, TraceContext: inf.TraceContext,
		}
		hb := func(ctx context.Context, details []byte) (bool, error) {
			err := worker.Heartbeat(ctx, details)
			return errors.Is(context.Cause(ctx), worker.ErrCancelRequested), err
		}
		return r.Run(ctx, t, args(in), in, hb)
	}
}
```

`train`, `eval`, and `export` are registered as `subprocess(py, ...)`, and `corpus run --stage <s>` as `subprocess(cp, ...)`. On the Python side each of those entries wraps its body in `activity.run`.

## One work directory, shared by every attempt

A retry must resume, not restart, and that forces the first design choice: every attempt must find the previous attempt's files. The work directory is therefore a pure function of the key: `<TL_ARTIFACTS>/activities/<k>/`, with each `/`-separated segment of $k$ a directory (`WorkDir`). A segment must match `[A-Za-z0-9._-]+` and must not be `.` or `..`, because workflow ids are caller-chosen text and a key like `../../etc` would otherwise escape the artifact root. Every attempt $a = 1, 2, \dots$ of one activity lands in the same directory, which is what makes a retry a **resume**.

A shared directory with retries means duplicates and orphans, and three rules make it safe:

1. **Publish by rename.** An output is written to `name.tmp`, flushed and fsynced, then renamed to `name`, and the directory is fsynced. `rename` is atomic within one filesystem, so a reader sees the old file or the whole new one. The runner writes `spec.json` the same way.
2. **`DONE.json` last.** The last act of a successful run publishes `DONE.json` (`{"outputs": [...]}`), then emits the `done` event, then exits 0. A run that starts and finds `DONE.json` re-emits `done` and exits 0 without doing anything, so a duplicate delivery has one effect.
3. **One attempt at a time.** The helper takes an exclusive, non-blocking `flock` on `<dir>/.lock` before any work. If a previous attempt still holds it (a child orphaned by a SIGKILLed worker), the new attempt exits 75 and is retried later.

## Progress, checkpoints, and heartbeats

The directory lets a retry find the files. The second half of resuming is that the server must remember the last good checkpoint, and the runner learns about checkpoints only by reading a file another process is still writing.

The child appends one JSON object per line to `--progress` and **flushes after each line**. The runner reads the file while it grows, so an event stuck in Python's buffer is invisible exactly when it matters, when the worker is killed one second later. The runner's `Tail` keeps a line back until its newline arrives, because half a line is not an event. Neither half of this agreement is enough alone.

A `ckpt` event names a checkpoint step directory, relative to `TL_ARTIFACTS`. The helper emits it **only after** `verify_step_dir` says the directory is complete (its `MANIFEST.json` lists every file with the right size and hash). The runner treats each new `ckpt` path as the **heartbeat details** and heartbeats it at once. It also heartbeats at least every $h$, so the server does not fence the activity between checkpoints: training checkpoints every few minutes, while heartbeat timeouts are tens of seconds. On a retry the server hands back the last details, and the runner passes them as `--resume <ckpt>`. Until the child reports a newer checkpoint, every heartbeat repeats the old one, because an empty heartbeat would erase the resume point on the server.

## Exit codes are the verdict

The platform needs a verdict per attempt, and the only thing every process produces is an exit code. So the exit code is the verdict, and its meanings are fixed in advance: data errors are never worth retrying, transient errors and crashes are, and 130 is a deliberate stop. Everything else, the result and the reason, travels through files whose names the runner already knows. The table is the whole mapping.

| Code | Meaning | `Failure` |
|---|---|---|
| `0` and `DONE.json` present | success | none: the result is the bytes of `DONE.json` (at most 2 MiB) |
| `0` without `DONE.json` | a broken entry | `MissingDone`, retryable |
| `65` (`EX_DATAERR`) | the input is wrong: invalid spec, unlicensed source | `ExitCode65`, **non-retryable** |
| `75` (`EX_TEMPFAIL`) | transient: a flaky download, an OOM the next attempt may avoid | `ExitCode75`, retryable |
| `130` | cancelled | `Canceled`, non-retryable |
| any other code, or a signal | a crash | `ExitCode<n>` or `Signal<n>`, retryable |

`Failure.message` is the last 4 KiB of the child's stderr, which is where a traceback ends. On the Python side `run(main)` maps exceptions to the same codes: `SpecError` 65, `RetryableError` 75, `Cancelled` 130, anything else 1.

## Stopping a child: process groups, SIGTERM, SIGKILL

Exit codes say how a child ended on its own. Often, though, the runner has to end it. Signals must reach the whole process tree and must end in a guaranteed kill. The runner starts the child with `Setpgid`, so the child leads a new **process group** whose id is its pid. Signals go to the whole group (`kill(-pgid, sig)`), which reaches anything the child started (a tokenizer pool, a data loader worker), and a signal meant for the worker's own group does not reach the child behind the runner's back.

There are two reasons to stop a child, and they need **opposite verdicts**:

| Why | How the runner learns it | Result |
|---|---|---|
| the activity is cancelled (a workflow cancel) | a heartbeat answer with `cancel_requested` | `Canceled`, non-retryable |
| the worker is going away (SIGTERM drain on a deploy, or a lost lease) | `ctx` ends | `WorkerShutdown`, **retryable**: another worker resumes from the checkpoint |

The worker SDK reports a `cancel_requested` heartbeat answer by cancelling the activity's context with its own cause; `Subprocess.Canceled` tells the runner which causes mean "cancelled", and everything else is a shutdown. The runner records *why* it sent SIGTERM (`stopReason`) and overrides the child's exit code with that reason, rather than trusting 130 to mean "cancelled". Either way the runner sends SIGTERM to the group, waits up to $g$, then sends SIGKILL.

The Python handler only sets a flag. The training loop notices it at its next step, saves a checkpoint, emits its `ckpt` event, and raises `Cancelled` (exit 130). The runner heartbeats that last checkpoint even after `ctx` has ended, on a fresh context with a short timeout, and returns it in `Failure.Details` for `FailActivityRequest.last_heartbeat_details`. Track the last acknowledged checkpoint separately from the last one read: a stopping poll or a failed heartbeat can leave an unreported checkpoint even when the final file read is empty.

A worker that dies outright cannot stop its child. The child notices instead: its parent pid changes (it is reparented), and `Activity.cancelled` becomes true.

## Telemetry: one trace from the CLI to the training step

The last piece of the contract is observability, and the rule for it is that telemetry is an observer. The worker passes the activity span's W3C context in `TRACEPARENT`: `00-<trace-id>-<parent-id>-<flags>`, lowercase hex, 32 and 16 digits, neither all zeros, with flags bit 0 meaning **sampled**. The exporter makes `train.run` (or `corpus.stage <s>`) a child of that span in the same trace, and `train.step` a child of `train.run` every 50th step. It posts them as OTLP/HTTP JSON to `OTEL_EXPORTER_OTLP_ENDPOINT/v1/traces`, and pushes gauges such as `tl.train.loss` to `/v1/metrics`, because a subprocess cannot be scraped. Three rules keep it honest:

- An invalid `TRACEPARENT` starts a **new root** trace; it is never parsed leniently.
- An **unsampled** parent means export nothing: the decision belongs to the root of the trace.
- Export failures return `False` and drop the batch. Telemetry never raises into the work it observes.

## Working it by hand

Now follow one activity through two attempts. It is `train-1/3` (workflow `train-1`, activity id `3`) with `TL_ARTIFACTS=/artifacts` and a heartbeat timeout of 150 ms, so $h$ = 50 ms.

**Attempt 1.** The runner computes the work directory `/artifacts/activities/train-1/3/`, publishes `spec.json` there, and execs

```
<entry> train --spec /artifacts/activities/train-1/3/spec.json --progress /artifacts/activities/train-1/3/progress.jsonl
```

with `TL_IDEMPOTENCY_KEY=train-1/3` and `TL_ATTEMPT=1`. There is no `--resume`, since the task has no heartbeat details. The child writes

```
{"ts":...,"kind":"step","step":500,"loss":2.31,"lr":0.0009,"tokens":8192000}
{"ts":...,"kind":"ckpt","step":500,"ckpt":"runs/train-1/ckpt/step-000500"}
```

The tail reads the `ckpt` line within one poll, and the runner heartbeats with details `runs/train-1/ckpt/step-000500`. Then the child runs out of memory and exits 75 with `CUDA out of memory` on stderr. The result is `Failure{Type: "ExitCode75", NonRetryable: false, Message: "...CUDA out of memory..."}`, so the activity is retried.

**Attempt 2.** The server redelivers with `attempt = 2` and `last_heartbeat_details = runs/train-1/ckpt/step-000500`. The runner uses the same directory and execs the same argv plus `--resume runs/train-1/ckpt/step-000500`, with `TL_ATTEMPT=2`. Attempt 2 starts from `step-000500` only because attempt 1 heartbeated that path. The helper (clock fixed at 1760000000.0) writes exactly

```
{"ts":1760000000.0,"kind":"step","step":600,"loss":2.25,"lr":0.0009,"tokens":9830400}
{"ts":1760000000.0,"kind":"ckpt","step":1000,"ckpt":"runs/train-1/ckpt/step-001000"}
{"ts":1760000000.0,"kind":"done","outputs":["runs/train-1/ckpt/step-001000"]}
```

and `DONE.json` is the 45 bytes `{"outputs":["runs/train-1/ckpt/step-001000"]}`, which become the activity's result. A third delivery (say the worker died before `CompleteActivityTask`) finds `DONE.json`, re-emits `done`, and exits 0 without training.

**Telemetry.** With `TRACEPARENT=00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01` (the W3C example) and span ids drawn from a counter, `train.run` gets span id `0000000000000001` and parent `00f067aa0ba902b7`. The sampled step 50 (step 49 is not sampled) gets `0000000000000002` with parent `0000000000000001`. Both carry trace id `4bf92f3577b34da6a3ce929d0e0e4736`. In the OTLP JSON body, `tl.train.step = 50` is `{"intValue": "50"}`, a string, because JSON numbers lose precision past $2^{53}$.

These numbers are worth checking against the code below: they are exactly what the runner and the helper produce.

## Designing the interface

Both halves have a small surface. Listing 2 is the Go runner; Listing 3 is the Python helper and exporter.

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

The runner takes a `Task` and a `Heartbeat` function instead of importing the worker SDK, so it can be exercised against a scripted fake child, and the worker's composition root adapts each delivery to it in a dozen lines per verb, as Listing 1 showed.

## Review questions

1. Why does a process exit code have to be interpreted with cancellation state?

   <details><summary>Answer</summary>

   The same signal-related exit can mean a requested cancellation or an unexpected failure, with different retry behavior.

   </details>

[Contents](README.md) · [Next: The Go subprocess runner](02-go-runner.md)
