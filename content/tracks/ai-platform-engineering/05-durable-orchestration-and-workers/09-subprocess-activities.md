# Subprocess activity runner (Go) and the Python activity helper

## Overview

| | |
|---|---|
| **Module** | `dur.09` · build · Go and Python · Pass 8 · reading time 2 to 3 h |
| **Solution** | [`go/activities/subprocess.go`](../../../../course/solutions/go/activities/subprocess.go): `Subprocess.Run`, `Task`, `Heartbeat`, `Failure`, `WorkDir`, `Tail`, `Classify`; [`python/tinyllm/io/activity.py`](../../../../course/solutions/python/tinyllm/io/activity.py): `Activity`, `run`, the exit codes; [`python/tinyllm/io/telemetry.py`](../../../../course/solutions/python/tinyllm/io/telemetry.py): `Tracer`, `parse_traceparent`, OTLP/HTTP JSON export; wired into the worker by [`entry/go/cmd/worker/main.go`](../../../../course/solutions/entry/go/cmd/worker/main.go) |
| **Contract** | [`course/contracts/spec/subprocess-activity.md`](../../../../course/contracts/spec/subprocess-activity.md) (the interface between the two halves), [`course/contracts/py/tinyllm/io/activity.pyi`](../../../../course/contracts/py/tinyllm/io/activity.pyi), [`course/contracts/py/tinyllm/io/telemetry.pyi`](../../../../course/contracts/py/tinyllm/io/telemetry.pyi), and the spans of [`otel/semconv.md`](../../../../course/contracts/otel/semconv.md) |
| **Needs** | `L0.6` atomic checkpoints (`verify_step_dir` decides when a checkpoint may be reported); reading: `lang.02` exit codes and signals, the worker and activity SDK of `dur.04` and `dur.05` (the runner is plugged into them by the worker's composition root) |
| **Used by** | `data.09` (every corpus stage runs through the runner) · `dur.11` (`train` and `eval` activities) · `dep.06` (the worker image carries these units) · `obs.05` (runs `telemetry.py` as the Python end of the control-plane trace); later `dur.12` (`export`) and `C1` (the capstone run is a `TrainRun`). With this module in place, `<system> train --spec specs/tiny.json` survives a worker kill. |
| **Optional depth** | [Temporal: activity heartbeats and cancellation](https://docs.temporal.io/encyclopedia/detecting-activity-failures) (free); [W3C Trace Context](https://www.w3.org/TR/trace-context/) (free); [OTLP specification, JSON encoding](https://opentelemetry.io/docs/specs/otlp/#json-protobuf-encoding) (free); W. Richard Stevens, *Advanced Programming in the UNIX Environment*, ch. 9 and 10 (process groups, signals) |

## Key Takeaways

- Python never speaks gRPC to the platform (D9). A Go worker runs it as a **child process** and the whole interface is files, environment variables, signals, and an **exit code**.
- The exit code is the verdict: `65` never retries, `75` and crashes retry, `130` is a cancellation. A **worker drain is not a cancellation**, so the runner reports it as retryable even when the child exits 130.
- A checkpoint becomes a **heartbeat detail** the moment it is complete, and the next attempt gets it as `--resume`. Until a newer one exists, every heartbeat keeps sending the old one.
- Every attempt shares one **work directory** named by the idempotency key. Outputs are published by **rename**, `DONE.json` last, and an **exclusive lock** keeps an orphaned child from writing beside its successor.
- The Python spans continue the activity's trace from **TRACEPARENT**, honor its sampled flag, and a dead collector never fails training.

## How to read this chapter

Sections 1 to 3 build the contract from first principles and walk one activity through two attempts by hand. Section 4 fixes the interface of both halves. Section 5 walks through the Go runner, the Python helper, and the exporter, and explains why each choice is the right one. Try to predict each step of the solution before reading it.

The runner does not import the worker SDK. The worker's composition root ([`entry/go/cmd/worker/main.go`](../../../../course/solutions/entry/go/cmd/worker/main.go)) builds one `Subprocess` per Python entry and adapts each delivery to it:

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

---

## 1. Why now

The durable engine can run Go activities with retries, heartbeats, and fenced leases (`dur.03` to `dur.05`), but everything that matters to the platform's users is Python: the corpus stages from Pass 3 and the trainer from Pass 2. Today `{tinyllm} train` is a command you type. If the laptop sleeps at step 9,000, the run is gone, and nothing records which checkpoint was last good. If you start it from a Go activity with `exec.Command` and wait, three things break at once: the server sees no heartbeats and fences the activity mid-run, a worker restart kills the child without a checkpoint (or worse, leaves it running), and a retry starts again from step 0 in a fresh directory. This module is the bridge: a Go runner and a Python helper that keep one written contract, so `CorpusBuild` (`data.09`) and `TrainRun` (`dur.11`) can run Python work durably.

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

These numbers are worth checking against the code in section 5: they are exactly what the runner and the helper produce.

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

The runner takes a `Task` and a `Heartbeat` function instead of importing the worker SDK, so it can be exercised against a scripted fake child, and the worker's composition root adapts each delivery to it (a dozen lines per verb).

### What a correct solution must handle

| Case | Kind of check | What must hold | Why it matters downstream |
|---|---|---|---|
| the hand example's resume flow | unit | section 3: attempt 1 heartbeats its checkpoint and fails retryable; attempt 2 gets `--resume`, `TL_ATTEMPT=2`, the same directory, and returns `DONE.json` | `TrainRun` resumes from the last checkpoint after a kill (`dur.11`) |
| work dir from the key | boundary | key segments become directories; `..`, `.`, empty, and odd characters are rejected, non-retryable | a workflow id can never write outside `/artifacts/activities` |
| the environment contract | unit | `TL_*`, `TRACEPARENT`, OTEL variables set per task; the worker's own copies never leak | Python spans land in the right trace (`obs.05`) |
| the exit-code table | unit | every row of the table in section 2.4, signals included | a poison input fails fast; a transient one retries |
| success needs `DONE.json` | boundary | exit 0 without `DONE.json` is a retryable `MissingDone`; over 2 MiB is non-retryable; exactly 2 MiB is fine | results fit in a gRPC message and a history event |
| partial progress lines | unit | half a line waits for its newline; garbage lines are skipped | heartbeats never carry half a path |
| new checkpoints are heartbeated | unit | each new checkpoint is heartbeated while the child runs, even a line written in two pieces | a kill one second after a checkpoint loses nothing |
| the resume point survives | fault | a retry's heartbeats repeat the old resume point until a newer one exists | a kill before the retry's first checkpoint does not restart from step 0 |
| heartbeat cadence | unit | at least one heartbeat per $T_{hb}/3$ | the server does not fence a long step |
| `cancel_requested` stops the child | fault | `cancel_requested` sends SIGTERM; the run ends `Canceled` with the exit checkpoint heartbeated | `dur.08` cancel reaches Python |
| grace, then SIGKILL | fault | a child ignoring SIGTERM is SIGKILLed after the grace, grandchild included | a stuck child never holds a worker slot |
| own process group | unit | the child leads its own group | group signals reach grandchildren only |
| worker shutdown is retryable | fault | `ctx` ending gives a retryable `WorkerShutdown` with `Details` = the exit checkpoint | a deploy does not end a training run for good |
| shutdown flushes an unacknowledged checkpoint | fault | a failed checkpoint heartbeat is retried on shutdown even when the final tail read is empty | resume state reaches the server |
| cancel cause from the context | fault | with `Subprocess.Canceled` set, a ctx cancelled with the SDK's cancel cause ends `Canceled`; any other cause stays `WorkerShutdown` | the worker SDK's cancel reaches the child as a cancel |
| atomic `spec.json` | unit | `spec.json` replaced by rename, never rewritten in place | a retry never shows half a spec |
| bounded stderr tail | boundary | the last 4 KiB of stderr, traceback included | failures stay small in the history |
| the hand example's progress lines | unit | section 3's three progress lines and `DONE.json`, byte for byte | the runner and the schema read these bytes |
| flags and environment | unit | `--spec`, `--progress`, `--resume` (both spellings), `TL_*` defaults; missing flags are `SpecError` | the entry and the runner agree on argv |
| one flushed line per event | unit | each event is on disk as one line when `emit` returns | the tail sees checkpoints at once |
| bad events rejected | boundary | unknown kinds and NaN or inf values raise, nothing written | a NaN loss never corrupts the progress file |
| checkpoint only when complete | unit | no event for an incomplete directory; paths relative to `TL_ARTIFACTS`; outside paths rejected | the resume point is always loadable (`L0.6`) |
| publish never exposes a partial file | fault | when the rename never happens, the final name does not exist | `DONE.json` exists only for finished work |
| outputs stay in the work dir | boundary | `..`, absolute, and empty names rejected | one activity cannot overwrite another's files |
| `run` exit codes | unit | `SpecError` 65, `RetryableError` 75, `Cancelled` 130, crash 1, success 0 with `DONE.json` | the Go side's table has the right inputs |
| spec validation is exit 65 | unit | validation errors and non-object or bad JSON specs are 65, before any output | invalid specs do not burn retries |
| rerun after done is a no-op | regression | a second run finds `DONE.json`, skips `main`, re-emits `done` | a duplicate delivery has one effect |
| SIGTERM checkpoints and exits 130 | fault | SIGTERM, then a checkpoint as the last event, then exit 130 within 5 s | cancel and drain keep the work done so far |
| a second attempt is locked out | fault | while one attempt holds the lock, another exits 75 without running | no two children write one directory |
| an orphaned child stops | fault | a child whose parent died sees `cancelled` | an orphan does not train on beside its successor |
| the hand example's span tree | conformance | section 3's ids, parents, kinds, and JSON encoding at a real HTTP receiver | `obs.05` finds `train.step` under `activity train` |
| `parse_traceparent` | boundary | valid, unsampled, future-version, and nine invalid headers | invalid input starts a new trace |
| format round trip | unit | `format_traceparent` and `parse_traceparent` agree | contexts can be handed on |
| unsampled parent exports nothing | boundary | flags `00`: no spans, no gauges sent | no orphan spans |
| no `TRACEPARENT` starts a root | boundary | no or bad `TRACEPARENT`: a fresh trace id and no parent | hand runs still trace |
| `Tracer.from_env` | unit | endpoint, service name, resource attributes, parent from the environment | the runner's variables are the only input |
| a dead collector never fails the work | fault | connection refused: `flush` returns `False` within its timeout, queue dropped | training survives a collector outage |
| a collector error status is `False` | fault | a 500 answer is `False` | drops are visible |
| an error marks the span | unit | an exception sets status ERROR and `error.type`, and propagates | failed stages show in the trace |
| step sampling | unit | every 50th step | span volume stays bounded |
| gauges are pushed | unit | the latest gauge value per name and attributes goes to `/v1/metrics` | `tl.train.loss` reaches Prometheus |
| shutdown stops export | unit | nothing is queued after `shutdown` | clean exit |
| body shape without a parent | unit | a root span has no `parentSpanId`; attribute value types | the collector accepts the body |

## 5. The solution, step by step

### 5.1 From first principles

1. **The platform needs a verdict per attempt, and the only thing every process produces is an exit code.** So the exit code is the verdict, and its meanings must be fixed in advance (section 2.4): data errors (65) are never worth retrying, transient errors (75) and crashes are, and 130 is a deliberate stop. Everything else (the result, the reason) travels through files the runner already knows the names of.
2. **A retry must resume, not restart.** That forces two things: every attempt must find the previous attempt's files, so the work directory is a pure function of the idempotency key (`WorkDir`), and the server must remember the last good checkpoint, so the checkpoint path becomes the heartbeat details. In the hand example, attempt 2 starts from `step-000500` only because attempt 1 heartbeated that path.
3. **The runner learns about checkpoints only by reading a file another process is still writing.** So events must be complete lines, written in one call and flushed (Python side), and the reader must hold back a line without a newline (`Tail`). Neither half alone is enough.
4. **A shared directory with retries means duplicates and orphans.** A duplicate delivery after success must be a no-op, so success is a file (`DONE.json`) published last by rename; an orphaned child from a killed worker must not write beside its successor, so the helper takes a non-blocking `flock` and gives up with 75 if it cannot.
5. **There are two reasons to stop a child and they need opposite verdicts.** A workflow cancel ends the activity for good; a worker drain must hand it to another worker. The runner therefore records *why* it sent SIGTERM (`stopReason`) and overrides the child's exit code with that reason, rather than trusting 130 to mean "cancelled".
6. **Signals must reach the whole tree and must end in a guaranteed kill.** A data-loader pool started by the child is a grandchild; only a process group (`Setpgid`, `kill(-pgid)`) reaches it, and only SIGKILL after a grace period bounds how long a stuck child can hold a slot.
7. **Telemetry is an observer.** It continues the trace it is given (`TRACEPARENT`), respects the root's sampling decision, and never raises: `flush` returns a boolean.

### 5.2 The code, annotated

**The work directory.** Each segment of the key is validated before it is joined, so `filepath.Join` can never climb out of `<artifacts>/activities`:

```go
var keySegment = regexp.MustCompile(`^[A-Za-z0-9._-]+$`)

func WorkDir(artifacts, key string) (string, error) {
	if artifacts == "" {
		return "", errors.New("activities: no artifacts root")
	}
	segs := strings.Split(key, "/")
	for _, s := range segs {
		// "." and ".." match the character class, so they are rejected by name;
		// an empty segment (a leading or doubled "/") fails the regexp.
		if s == "." || s == ".." || !keySegment.MatchString(s) {
			return "", fmt.Errorf("activities: idempotency key %q has a bad segment %q", key, s)
		}
	}
	// The same key always gives the same directory: that is what makes a retry a resume.
	return filepath.Join(append([]string{artifacts, "activities"}, segs...)...), nil
}
```

**Tailing a growing file.** `Tail` remembers a byte offset and the unfinished tail of the last read. Only text up to a newline is parsed:

```go
func (t *Tail) Read() ([]Progress, error) {
	f, err := os.Open(t.Path)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil // the child has not written anything yet
	}
	// ...
	if _, err := f.Seek(t.offset, io.SeekStart); err != nil {
		return nil, err
	}
	b, err := io.ReadAll(f)
	// ...
	t.offset += int64(len(b))
	buf := append(t.partial, b...) // the half line from last time comes first
	var out []Progress
	for {
		i := bytes.IndexByte(buf, '\n')
		if i < 0 {
			break // no newline: the rest is still being written
		}
		line := bytes.TrimSpace(buf[:i])
		buf = buf[i+1:]
		var p Progress
		// A line that is not a JSON object with a kind is skipped, never fatal.
		if len(line) > 0 && json.Unmarshal(line, &p) == nil && p.Kind != "" {
			out = append(out, p)
		}
	}
	t.partial = append([]byte(nil), buf...) // copy: buf aliases the read buffer
	return out, nil
}
```

**The exit-code table.** `Classify` is the table of section 2.4 as code. A signal is checked first because a signalled process has no meaningful exit code:

```go
func Classify(state *os.ProcessState, stderr []byte) *Failure {
	msg := string(stderr)
	if ws, ok := state.Sys().(syscall.WaitStatus); ok && ws.Signaled() {
		sig := int(ws.Signal())
		return &Failure{Type: "Signal" + strconv.Itoa(sig), Message: msg, ExitCode: -1, Signal: sig}
	}
	code := state.ExitCode()
	switch code {
	case ExitOK:
		return nil
	case ExitDataErr: // the input is wrong: retrying cannot help
		return &Failure{Type: "ExitCode65", Message: msg, NoRetry: true, ExitCode: code}
	case ExitCanceled:
		return &Failure{Type: "Canceled", Message: msg, NoRetry: true, ExitCode: code}
	default: // 75 and every crash: retryable
		return &Failure{Type: "ExitCode" + strconv.Itoa(code), Message: msg, ExitCode: code}
	}
}
```

**The child's environment.** `childEnv` removes inherited copies of every contract variable before setting this delivery's values. Without the `drop` set, a worker started under its own `TRACEPARENT` would leak it into an activity that has none, and the Python spans would join an unrelated trace:

```go
func (s Subprocess) childEnv(t Task) []string {
	// ...
	set := map[string]string{
		"TL_IDEMPOTENCY_KEY": t.IdempotencyKey,
		"TL_ATTEMPT":         strconv.Itoa(max(t.Attempt, 1)),
		"TL_ARTIFACTS":       s.Artifacts,
	}
	if tp := t.TraceContext["traceparent"]; tp != "" {
		set["TRACEPARENT"] = tp
		// ... TRACESTATE, OTEL_EXPORTER_OTLP_ENDPOINT, OTEL_SERVICE_NAME likewise
	}
	// Trace variables are always dropped, even when this task sets none.
	drop := map[string]bool{"TRACEPARENT": true, "TRACESTATE": true, "OTEL_EXPORTER_OTLP_ENDPOINT": true}
	for k := range set {
		drop[k] = true
	}
	// ... copy base minus drop, then append set in a fixed order
}
```

**Starting the attempt.** `Run` publishes the spec by rename, builds the argv the contract names, and starts the child as the leader of a new process group. stderr goes both to the attempt log and to a 4 KiB ring, so a chatty child cannot inflate the failure message:

```go
	specPath := filepath.Join(dir, "spec.json")
	progPath := filepath.Join(dir, "progress.jsonl")
	if err := writeAtomic(specPath, spec); err != nil { // spec.json.tmp, fsync, rename
		return nil, &Failure{Type: "WorkDir", Message: err.Error(), ExitCode: -1}
	}

	argv := append(append(append([]string{}, s.Entry...), args...), "--spec", specPath, "--progress", progPath)
	if len(t.LastHeartbeat) > 0 {
		argv = append(argv, "--resume", string(t.LastHeartbeat)) // the server's last checkpoint
	}
	cmd := exec.Command(argv[0], argv[1:]...)
	cmd.Env = s.childEnv(t)
	cmd.Dir = s.Dir
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true} // pgid == child's pid
	// ... open attempt-<n>.log
	tail := &ring{n: stderrTail}
	cmd.Stdout = logf
	cmd.Stderr = io.MultiWriter(logf, tail)

	// Start tailing at the current end: earlier attempts' events are history.
	tl := &Tail{Path: progPath}
	if fi, err := os.Stat(progPath); err == nil {
		tl.offset = fi.Size()
	}
```

Starting the tail at the file's current size means each attempt reads only its own events: attempt 1's lines (step 500 in the hand example) are still in `progress.jsonl` when attempt 2 starts, but they are history, and the resume point they produced already arrived through `LastHeartbeat`.

**The state of one attempt.** Three pieces of state drive the loop. `last` is the resume point, seeded from the task so a retry never heartbeats empty details; `acknowledged` is the last value the server accepted; `stopReason` records why the runner sent SIGTERM:

```go
	last := append([]byte(nil), t.LastHeartbeat...)
	lastBeat := time.Time{}
	var stopReason string // "" running, "Canceled", "WorkerShutdown"
	var kill <-chan time.Time
	acknowledged := string(last) // checkpoint already recorded by the durable server

	stop := func(reason string) {
		if stopReason != "" {
			return // the first reason wins; one SIGTERM, one grace timer
		}
		stopReason = reason
		_ = syscall.Kill(-pgid, syscall.SIGTERM) // negative pid: the whole group
		kill = time.After(grace)
	}
	beat := func() {
		cancel, err := hb(ctx, last)
		lastBeat = time.Now()
		if err == nil && cancel {
			stop("Canceled")
		}
		if err == nil {
			acknowledged = string(last) // a failed heartbeat is not an acknowledgement
		}
	}
	drain := func() bool {
		evs, _ := tl.Read()
		fresh := false
		for _, e := range evs {
			if e.Kind == "ckpt" && e.Ckpt != "" && e.Ckpt != string(last) {
				last = []byte(e.Ckpt)
				fresh = true
			}
		}
		return fresh
	}
```

**The loop.** One `select` serves four events. A fresh checkpoint triggers a heartbeat at once; otherwise the heartbeat fires every $T_{hb}/3$ (or `DefaultHeartbeat`), which leaves two missed beats of slack before the server fences:

```go
	ctxDone := ctx.Done()
	var waitErr error
loop:
	for {
		select {
		case waitErr = <-exited:
			break loop
		case <-ctxDone:
			ctxDone = nil // a closed channel would fire forever
			if s.Canceled != nil && s.Canceled(context.Cause(ctx)) {
				stop("Canceled")
			} else {
				stop("WorkerShutdown")
			}
		case <-kill:
			kill = nil
			_ = syscall.Kill(-pgid, syscall.SIGKILL) // the grace is over
		case <-ticker.C:
			fresh := drain()
			if stopReason == "" && (fresh || time.Since(lastBeat) >= every) {
				beat()
			}
		}
	}
```

**After the exit.** The group is SIGKILLed once more (a grandchild may outlive its parent), the file is drained one last time, and any checkpoint the server has not acknowledged is flushed on a context that survives the attempt's own cancellation. Then the stop reason overrides the child's verdict:

```go
	_ = syscall.Kill(-pgid, syscall.SIGKILL)
	fresh := drain()
	if fresh || string(last) != acknowledged {
		fctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
		_, _ = hb(fctx, last)
		cancel()
	}
	// ... f = Classify(cmd.ProcessState, tail.Bytes()); f.Details = last
	switch {
	case f == nil:
		// exit 0: the work finished, even if a stop raced with it.
	case stopReason == "WorkerShutdown":
		// Not a cancel: the work is unfinished and another worker resumes it.
		return nil, &Failure{Type: "WorkerShutdown", Message: "the worker stopped the activity: " + string(tail.Bytes()), ExitCode: f.ExitCode, Signal: f.Signal, Details: last}
	case stopReason == "Canceled":
		f.Type, f.NoRetry = "Canceled", true
		return nil, f
	default:
		return nil, f
	}
	done, err := os.ReadFile(filepath.Join(dir, "DONE.json"))
	if err != nil {
		return nil, &Failure{Type: "MissingDone", Message: "exit 0 without DONE.json: " + err.Error(), ExitCode: 0}
	}
	if len(done) > MaxResult { // 2 MiB: the result must fit a history event
		return nil, &Failure{Type: "ResultTooLarge", /* ... */ NoRetry: true}
	}
	return done, nil
```

Full solution: [`subprocess.go`](../../../../course/solutions/go/activities/subprocess.go).

**The Python helper: one event, one line.** `emit` validates before it writes, so a rejected event leaves the file untouched, and it writes one complete line then flushes:

```python
    def emit(self, kind: str, **fields: object) -> None:
        if kind not in _KINDS:
            raise ValueError(f"progress kind {kind!r} is not one of {_KINDS}")
        for k, v in fields.items():
            if isinstance(v, float) and not math.isfinite(v):
                raise ValueError(f"progress field {k}={v} is not finite (JSON has no NaN)")
        if self._progress is None:
            os.makedirs(self.work_dir, exist_ok=True)
            self._progress = open(self.progress_path, "a", encoding="utf-8")
        event = {"ts": self._clock(), "kind": kind, **fields}
        # One write of one complete line, then flush: the runner tails the
        # file and must never parse half an event.
        self._progress.write(json.dumps(event, separators=(",", ":"), allow_nan=False) + "\n")
        self._progress.flush()
```

`separators=(",", ":")` is what makes the hand example's lines byte for byte `{"ts":1760000000.0,"kind":"step",...}`.

**Checkpoints and outputs.** A checkpoint path is made relative to `TL_ARTIFACTS` (a path that is right on one pod must be right on the next) and is announced only after `verify_step_dir` finds nothing wrong. Outputs are published through a temp file, fsync, rename, and a directory fsync:

```python
    def checkpoint(self, step: int, path: str) -> str:
        rel = self._artifact_path(path)  # relative, and never outside TL_ARTIFACTS
        problems = verify_step_dir(os.path.join(self.artifacts, rel))
        if problems:
            raise ValueError(f"checkpoint {rel} is not complete: " + "; ".join(problems))
        self.emit("ckpt", step=int(step), ckpt=rel)  # only now may it become a resume point
        return rel

    def publish(self, name: str, data: bytes) -> str:
        final = self.output(name)  # rejects absolute names and anything outside the work dir
        os.makedirs(os.path.dirname(final), exist_ok=True)
        tmp = final + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, final)              # atomic: old file or whole new file
        _fsync_dir(os.path.dirname(final))  # make the rename itself durable
        return final

    def done(self, outputs: Sequence[str]) -> bytes:
        rel = [self._artifact_path(p) for p in outputs]
        data = json.dumps({"outputs": rel}, separators=(",", ":")).encode()
        self.publish("DONE.json", data)  # DONE.json first, then the event
        self.emit("done", outputs=rel)
        return data
```

**Cancellation inside Python.** The signal handler only sets a flag; the training loop calls `check_cancel()` at a safe point, after saving a checkpoint. The orphan check compares the parent pid with the one recorded at start:

```python
    @property
    def cancelled(self) -> bool:
        # A child whose worker was SIGKILLed is reparented (to init or a
        # subreaper): nobody will read its result, and the next attempt is
        # about to start in the same work directory. Stop as if cancelled.
        if not self._cancelled and os.getppid() != self._ppid:
            self._cancelled = True
        return self._cancelled

    def install_signal_handlers(self) -> None:
        def handler(signum, frame):  # noqa: ARG001
            self.request_cancel()  # no exit here: the loop checkpoints first
        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)
```

**`run`: exceptions to exit codes.** The order of the steps is the whole protocol: lock, then the `DONE.json` check, then `main`, then `done`:

```python
def run(main, argv=None, env=None) -> int:
    try:
        act = Activity(sys.argv[1:] if argv is None else argv, env)
    except SpecError as e:          # missing --spec/--progress, bad TL_ATTEMPT
        print(f"activity: {e}", file=sys.stderr)
        return EXIT_DATAERR
    act.install_signal_handlers()
    try:
        if not act._lock():         # flock(LOCK_EX | LOCK_NB) on <dir>/.lock
            print(f"activity: another attempt holds {act.work_dir}/.lock", file=sys.stderr)
            return EXIT_TEMPFAIL    # try again once the orphan is gone
        if act.is_done():           # a duplicate delivery: one effect only
            with open(os.path.join(act.work_dir, "DONE.json"), "rb") as f:
                outputs = json.loads(f.read()).get("outputs", [])
            act.emit("done", outputs=outputs)
            return EXIT_OK
        outputs = main(act)
        act.done(list(outputs or []))
        return EXIT_OK
    except SpecError as e:
        # ...
        return EXIT_DATAERR
    except RetryableError as e:
        # ...
        return EXIT_TEMPFAIL
    except Cancelled as e:
        # ...
        return EXIT_CANCELED
    except Exception:  # a crash: retryable, with the traceback for Failure.message
        traceback.print_exc()
        return 1
    finally:
        act.close()  # closes progress, releases the lock
```

Full solution: [`activity.py`](../../../../course/solutions/python/tinyllm/io/activity.py).

**The exporter.** `parse_traceparent` is strict: version `ff` is invalid, version `00` must have exactly four parts, and all-zero ids are rejected. The sampled flag is bit 0 of the flags byte:

```python
def _hex(s: str, n: int) -> bool:
    return len(s) == n and bool(_HEX.match(s)) and s != "0" * n  # lowercase hex, not all zeros

def parse_traceparent(value: Optional[str]) -> Optional[SpanContext]:
    if not value:
        return None
    parts = value.strip().split("-")
    if len(parts) < 4:
        return None
    version, trace_id, span_id, flags = parts[:4]
    if not (len(version) == 2 and _HEX.match(version)) or version == "ff":
        return None
    if version == "00" and len(parts) != 4:  # future versions may append fields; 00 may not
        return None
    if not (_hex(trace_id, 32) and _hex(span_id, 16)):
        return None
    if not (len(flags) == 2 and _HEX.match(flags)):
        return None
    return SpanContext(trace_id, span_id, bool(int(flags, 16) & 1))
```

The tracer takes its sampling decision from the remote parent, encodes integers as strings, and never raises out of `flush`:

```python
        # A remote parent decided whether this trace is recorded; a new root trace is.
        self.sampled = parent.sampled if parent is not None else True
        self._trace_id = parent.trace_id if parent is not None else self._new_id(16)

def _value(v: object) -> dict:
    # bool before int: bool is a subclass of int in Python.
    if isinstance(v, bool):
        return {"boolValue": v}
    if isinstance(v, int):
        return {"intValue": str(v)}  # OTLP JSON: int64 as a string, exact past 2**53
    # ... doubleValue, stringValue

    def _post(self, path: str, body: dict) -> bool:
        # ...
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return 200 <= resp.status < 300
        except (urllib.error.URLError, OSError, ValueError):
            return False  # telemetry never fails the work it observes

def should_sample_step(step: int, every: int = 50) -> bool:
    # ...
    return step % every == 0  # step 0, 50, 100, ...: step 50 yes, step 49 no
```

`end_span` queues a span only when `self.sampled and not self._closed`, which gives both "unsampled exports nothing" and "nothing queued after shutdown"; `gauge` overwrites by `(name, attributes)`, so the latest loss wins. Full solution: [`telemetry.py`](../../../../course/solutions/python/tinyllm/io/telemetry.py).

### 5.3 Why this solution

- **Cost.** Each poll reads only the bytes appended since the last one, so tailing is $O(\text{new bytes})$ per tick, not $O(\text{file})$; the stderr ring is bounded at 4 KiB however much the child prints. The runner does a heartbeat RPC at most once per poll, and typically once per $T_{hb}/3$.
- **Files over RPC.** A gRPC client inside Python would duplicate the durable protocol in a second language and couple the trainer to the platform. With files and exit codes, `{tinyllm} train --spec ... --progress ...` runs by hand with no cluster, and the protocol has one implementation.
- **Polling over inotify.** `inotify` would cut latency from 200 ms to near zero, but it is Linux-only, misbehaves on network filesystems where `/artifacts` often lives, and still needs the partial-line logic. A 200 ms poll is far below any heartbeat timeout.
- **Process groups over cgroups.** A child that calls `setsid` escapes its group; cgroups (systemd `KillMode=control-group`) would close that hole but need privileges a worker pod usually lacks. The group plus SIGKILL covers every well-behaved tree.
- **Overriding the exit code with the stop reason.** Trusting the child's 130 would make every deploy cancel every training run for good. The runner knows why it sent SIGTERM; the child does not.
- **Accepted tradeoffs.** Work since the last checkpoint is lost on SIGKILL; a duplicate delivery during a run is turned away with 75 and retried later rather than joined; an orphan is noticed only at its next `cancelled` check.

### 5.4 Explain it back

1. Why does the runner seed `last` from `t.LastHeartbeat` instead of starting it empty?
   <details><summary>Answer</summary>Every heartbeat sends `last` as the details, and the server stores whatever it receives. If a retry started with empty details, its first periodic heartbeat (before any new checkpoint) would erase the resume point, and a kill at that moment would restart from step 0.</details>
2. The child exits 130 after a worker drain. Why does `Run` return a retryable `WorkerShutdown` and not `Canceled`?
   <details><summary>Answer</summary>130 only says "I was asked to stop". The runner knows who asked: `ctx` ended without the SDK's cancel cause, so the worker is going away and the work is unfinished. Reporting `Canceled` (non-retryable) would end the run for good on every deploy; `WorkerShutdown` lets another worker resume from `Details`.</details>
3. Why is there a final heartbeat on `context.WithoutCancel(ctx)` after the loop, and why compare against `acknowledged` rather than only checking `fresh`?
   <details><summary>Answer</summary>The child writes its exit checkpoint after SIGTERM, when `ctx` may already be done, so a heartbeat on `ctx` would fail immediately. `fresh` covers a checkpoint read in the final drain; `acknowledged` covers one read earlier whose heartbeat failed or was skipped because `stopReason` was set. Either way the server must learn the latest resume point.</details>
4. Why does `run` take the lock before checking `DONE.json`, not after?
   <details><summary>Answer</summary>Checked outside the lock, the answer can be stale: a new attempt sees no `DONE.json` while an orphan is still finishing, then gets the lock the moment the orphan publishes `DONE.json` and exits, and trains a second time. Under the lock, the check and the work form one critical section, so whoever holds the lock sees the final state of the directory.</details>
5. Why is `tl.train.step = 50` encoded as `{"intValue": "50"}`?
   <details><summary>Answer</summary>OTLP's JSON mapping follows protobuf's, where int64 is a string: JSON numbers are doubles in most parsers and lose exactness past $2^{53}$. A collector may reject or round a bare number.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. checkpoints never reach a heartbeat | after a kill, the retry starts from step 0 | `drain` turns each new `ckpt` into `last` and the loop heartbeats at once when `fresh` |
| 2. the key's segments are not checked | a workflow id `../../x` writes outside `/artifacts` | `WorkDir` rejects `.`, `..`, empty, and non-`[A-Za-z0-9._-]` segments before joining |
| 3. `--resume` is not passed | retries ignore the checkpoint | `Run` appends `--resume <LastHeartbeat>` whenever the task carries details |
| 4. the worker's `TRACEPARENT` is inherited | Python spans join an unrelated trace | `childEnv` always drops the trace variables from the base environment |
| 5. a wrong row in the exit-code table | a poison spec retries until the dead-letter queue; a cancel restarts | `Classify` and `run` implement section 2.4 row for row |
| 6. exit 0 trusted without `DONE.json` | a broken entry "succeeds" with no outputs | exit 0 still reads `DONE.json`; its absence is `MissingDone` |
| 7. half a progress line is parsed | a checkpoint event is lost forever | `Tail` keeps bytes after the last newline in `partial` |
| 8. a retry heartbeats empty details | the resume point is erased before the first new checkpoint | `last` starts as a copy of `t.LastHeartbeat` |
| 9. heartbeats only on checkpoints | the server fences the activity mid-step | the loop also beats when `time.Since(lastBeat) >= T_hb/3` |
| 10. `cancel_requested` ignored | a cancelled workflow keeps training | `beat` calls `stop("Canceled")` when the answer says so |
| 11. no process group, or no SIGKILL | grandchildren and stuck children outlive the activity | `Setpgid`, signals to `-pgid`, SIGKILL after the grace and again after exit |
| 12. a drain reported as a cancel, or a cancel as a drain | every deploy ends a training run for good; a cancelled run is retried | `stopReason` from `Subprocess.Canceled(context.Cause(ctx))` overrides the child's code |
| 13. `spec.json` written in place | a reader sees a truncated spec | `writeAtomic`: temp file, fsync, rename |
| 14. all of stderr kept | a chatty child blows the 4 MiB message limit | stderr is teed into a 4 KiB `ring` |
| 15. progress not flushed | the runner sees nothing until exit | `emit` flushes after every line |
| 16. absolute paths in events or `DONE.json` | a path that is right on one pod is wrong on the next | `_artifact_path` makes every path relative to `TL_ARTIFACTS` |
| 17. a NaN loss written as `NaN` | the progress line is not JSON | `emit` rejects non-finite floats and dumps with `allow_nan=False` |
| 18. a ckpt event before the checkpoint is complete | the retry cannot load its resume point | `checkpoint` emits only when `verify_step_dir` returns no problems |
| 19. outputs written under their final name | a kill leaves a truncated `DONE.json` that looks finished | `publish` writes `name.tmp` and renames |
| 20. output names not confined | one activity overwrites another's files | `output` rejects absolute names and anything outside the work dir |
| 21. spec validation errors ignored | a bad spec trains on garbage, or retries | `load_spec` raises `SpecError` (exit 65) for bad JSON, non-objects, and validation errors |
| 22. no `DONE.json` check on start | a duplicate delivery trains again | `run` re-emits `done` and exits 0 when `DONE.json` exists |
| 23. the SIGTERM handler exits at once | the work since the last checkpoint is lost | the handler only sets a flag; the loop checkpoints, then raises `Cancelled` |
| 24. no work-dir lock | an orphan and its successor write one directory | `run` takes a non-blocking `flock` and exits 75 if it cannot |
| 25. orphans not detected | a child trains on after its worker died | `cancelled` compares `os.getppid()` with the pid recorded at start |
| 26. the remote parent ignored | Python spans form a separate trace | `Tracer` takes trace id and parent span id from `TRACEPARENT` |
| 27. `intValue` as a JSON number | collectors reject or round the attribute | `_value` encodes ints as strings |
| 28. lenient `traceparent` parsing | all-zero ids glue unrelated runs together | `parse_traceparent` rejects all-zero ids, version `ff`, and extra fields on `00` |
| 29. the sampled flag ignored | orphan spans whose parents were never recorded | `self.sampled` comes from the parent; `end_span` and `flush` honour it |
| 30. `OTEL_SERVICE_NAME` ignored | every system's Python is `tinyllm-python` | `from_env` reads it and falls back only when unset |
| 31. export errors raised | a collector outage kills training | `_post` catches network errors and returns `False` |
| 32. exceptions swallowed by the span | a failing stage looks successful | `span` marks the error and re-raises |
| 33. sampling off by one | step 0 never traced, step 1 always | `should_sample_step` is `step % every == 0` |
| 34. the first gauge value kept | dashboards show the loss of step 1 | `gauge` overwrites the entry for `(name, attributes)` |
| 35. spans queued after shutdown | memory grows at exit; spans half sent | `shutdown` sets `_closed`; `end_span` stops queueing |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L0.6` | `verify_step_dir` gates every `ckpt` event; `--resume` loads with `load_checkpoint` |
| Back | `dur.04`, `dur.05` | the worker and activity SDK the composition root adapts to `Task` and `Heartbeat` |
| Forward | `data.09` | `CorpusBuild` runs each corpus stage as `{corpus} run --stage <s>` through the runner |
| Forward | `dur.11` | `TrainRun` and `EvalSuite` run `{tinyllm} train` and `eval`; kills resume from the heartbeated checkpoint |
| Forward | `obs.05` | the control-plane trace reaches `train.step` through this exporter |
| Forward | `dep.06` | the worker image carries the activity helper and the exporter |
| Forward | `dur.12`, `C1` | `export` and the capstone run use the same contract |

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `Subprocess.Run` | Temporal activity heartbeats with details | heartbeat throttling, details as typed payloads, async completion for work finished elsewhere | Temporal Go SDK `activity.RecordHeartbeat`, `activity.GetHeartbeatDetails` |
| the exit-code table | Kubernetes Job `podFailurePolicy` | exit-code rules that fail the Job, ignore the failure, or count it | Kubernetes docs, "Handling retriable and non-retriable pod failures" |
| process groups and grace | systemd `KillMode=control-group`, Kubernetes `terminationGracePeriodSeconds` | cgroups instead of process groups: no child can escape by calling `setsid` | systemd.kill(5) |
| orphan detection by parent pid | Linux `PR_SET_PDEATHSIG` | the kernel signals the child when the parent thread dies | prctl(2) |
| `telemetry.py` | the OpenTelemetry Python SDK with the OTLP/HTTP exporter | batching span processor, retries with backoff, protobuf encoding | `opentelemetry-exporter-otlp-proto-http` |
