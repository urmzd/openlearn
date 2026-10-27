# Failure tests and integration

This part of [Subprocess activities: the Go runner and the Python helper](README.md) develops the worked mechanisms below.

> **Learning objectives**
> - Challenge retry, timeout, checkpoint, and process-cleanup behavior.

## Why this design

The cost is low. Each poll reads only the bytes appended since the last one, so tailing is $O(\text{new bytes})$ per tick, not $O(\text{file})$, and the stderr ring is bounded at 4 KiB however much the child prints. The runner does a heartbeat RPC at most once per poll, and typically once per $T_{hb}/3$.

Files beat RPC because a gRPC client inside Python would duplicate the durable protocol in a second language and couple the trainer to the platform. With files and exit codes, `{tinyllm} train --spec ... --progress ...` runs by hand with no cluster, and the protocol has one implementation. Polling beats inotify for a similar reason: `inotify` would cut latency from 200 ms to near zero, but it is Linux-only, misbehaves on network filesystems where `/artifacts` often lives, and still needs the partial-line logic. A 200 ms poll is far below any heartbeat timeout.

Process groups are chosen over cgroups knowingly. A child that calls `setsid` escapes its group; cgroups (systemd `KillMode=control-group`) would close that hole but need privileges a worker pod usually lacks. The group plus SIGKILL covers every well-behaved tree. Overriding the exit code with the stop reason is essential: trusting the child's 130 would make every deploy cancel every training run for good. The runner knows why it sent SIGTERM; the child does not.

Some tradeoffs are accepted outright. Work since the last checkpoint is lost on SIGKILL. A duplicate delivery during a run is turned away with 75 and retried later rather than joined. And an orphan is noticed only at its next `cancelled` check.

## What can go wrong

A correct pair of implementations must handle every case below; each exists because some later workflow depends on it.

| Case | Kind of check | What must hold | Why it matters downstream |
|---|---|---|---|
| the worked example's resume flow | unit | attempt 1 heartbeats its checkpoint and fails retryable; attempt 2 gets `--resume`, `TL_ATTEMPT=2`, the same directory, and returns `DONE.json` | `TrainRun` resumes from the last checkpoint after a kill |
| work dir from the key | boundary | key segments become directories; `..`, `.`, empty, and odd characters are rejected, non-retryable | a workflow id can never write outside `/artifacts/activities` |
| the environment contract | unit | `TL_*`, `TRACEPARENT`, OTEL variables set per task; the worker's own copies never leak | Python spans land in the right trace |
| the exit-code table | unit | every row of the exit-code table, signals included | a poison input fails fast; a transient one retries |
| success needs `DONE.json` | boundary | exit 0 without `DONE.json` is a retryable `MissingDone`; over 2 MiB is non-retryable; exactly 2 MiB is fine | results fit in a gRPC message and a history event |
| partial progress lines | unit | half a line waits for its newline; garbage lines are skipped | heartbeats never carry half a path |
| new checkpoints are heartbeated | unit | each new checkpoint is heartbeated while the child runs, even a line written in two pieces | a kill one second after a checkpoint loses nothing |
| the resume point survives | fault | a retry's heartbeats repeat the old resume point until a newer one exists | a kill before the retry's first checkpoint does not restart from step 0 |
| heartbeat cadence | unit | at least one heartbeat per $T_{hb}/3$ | the server does not fence a long step |
| `cancel_requested` stops the child | fault | `cancel_requested` sends SIGTERM; the run ends `Canceled` with the exit checkpoint heartbeated | a workflow cancel reaches Python |
| grace, then SIGKILL | fault | a child ignoring SIGTERM is SIGKILLed after the grace, grandchild included | a stuck child never holds a worker slot |
| own process group | unit | the child leads its own group | group signals reach grandchildren only |
| worker shutdown is retryable | fault | `ctx` ending gives a retryable `WorkerShutdown` with `Details` = the exit checkpoint | a deploy does not end a training run for good |
| shutdown flushes an unacknowledged checkpoint | fault | a failed checkpoint heartbeat is retried on shutdown even when the final tail read is empty | resume state reaches the server |
| cancel cause from the context | fault | with `Subprocess.Canceled` set, a ctx cancelled with the SDK's cancel cause ends `Canceled`; any other cause stays `WorkerShutdown` | the worker SDK's cancel reaches the child as a cancel |
| atomic `spec.json` | unit | `spec.json` replaced by rename, never rewritten in place | a retry never shows half a spec |
| bounded stderr tail | boundary | the last 4 KiB of stderr, traceback included | failures stay small in the history |
| the worked example's progress lines | unit | the three progress lines and `DONE.json`, byte for byte | the runner and the schema read these bytes |
| flags and environment | unit | `--spec`, `--progress`, `--resume` (both spellings), `TL_*` defaults; missing flags are `SpecError` | the entry and the runner agree on argv |
| one flushed line per event | unit | each event is on disk as one line when `emit` returns | the tail sees checkpoints at once |
| bad events rejected | boundary | unknown kinds and NaN or inf values raise, nothing written | a NaN loss never corrupts the progress file |
| checkpoint only when complete | unit | no event for an incomplete directory; paths relative to `TL_ARTIFACTS`; outside paths rejected | the resume point is always loadable |
| publish never exposes a partial file | fault | when the rename never happens, the final name does not exist | `DONE.json` exists only for finished work |
| outputs stay in the work dir | boundary | `..`, absolute, and empty names rejected | one activity cannot overwrite another's files |
| `run` exit codes | unit | `SpecError` 65, `RetryableError` 75, `Cancelled` 130, crash 1, success 0 with `DONE.json` | the Go side's table has the right inputs |
| spec validation is exit 65 | unit | validation errors and non-object or bad JSON specs are 65, before any output | invalid specs do not burn retries |
| rerun after done is a no-op | regression | a second run finds `DONE.json`, skips `main`, re-emits `done` | a duplicate delivery has one effect |
| SIGTERM checkpoints and exits 130 | fault | SIGTERM, then a checkpoint as the last event, then exit 130 within 5 s | cancel and drain keep the work done so far |
| a second attempt is locked out | fault | while one attempt holds the lock, another exits 75 without running | no two children write one directory |
| an orphaned child stops | fault | a child whose parent died sees `cancelled` | an orphan does not train on beside its successor |
| the worked example's span tree | conformance | the example's ids, parents, kinds, and JSON encoding at a real HTTP receiver | control-plane observability finds `train.step` under `activity train` |
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

The mistakes people make in the Go runner mostly lose the resume point. If checkpoints never reach a heartbeat, the retry after a kill starts from step 0; `drain` turns each new `ckpt` into `last`, and the loop heartbeats at once when `fresh`. If `--resume` is not passed, retries ignore the checkpoint; `Run` appends `--resume <LastHeartbeat>` whenever the task carries details. If half a progress line is parsed, a checkpoint event is lost forever; `Tail` keeps the bytes after the last newline in `partial`. And heartbeating only on checkpoints lets the server fence the activity mid-step, so the loop also beats when `time.Since(lastBeat) >= T_hb/3`.

> **Warning:** A retry that starts `last` empty erases the resume point on its first periodic heartbeat, before the child has written any new checkpoint. Seed `last` with a copy of `t.LastHeartbeat`.

Other runner pitfalls concern isolation and verdicts. Unchecked key segments let a workflow id `../../x` write outside `/artifacts`; `WorkDir` rejects `.`, `..`, empty, and non-`[A-Za-z0-9._-]` segments before joining. An inherited `TRACEPARENT` makes Python spans join an unrelated trace; `childEnv` always drops the trace variables from the base environment. A wrong row in the exit-code table makes a poison spec retry until the dead-letter queue, or a cancel restart; `Classify` and `run` implement the table row for row. Trusting exit 0 without `DONE.json` lets a broken entry "succeed" with no outputs, so exit 0 still reads `DONE.json` and its absence is `MissingDone`. Ignoring `cancel_requested` keeps a cancelled workflow training; `beat` calls `stop("Canceled")` when the answer says so. Without a process group or a SIGKILL, grandchildren and stuck children outlive the activity; the runner uses `Setpgid`, signals `-pgid`, and SIGKILLs after the grace and again after exit. Writing `spec.json` in place lets a reader see a truncated spec, which `writeAtomic` (temp file, fsync, rename) prevents. Keeping all of stderr lets a chatty child blow the 4 MiB message limit, so stderr is teed into a 4 KiB `ring`.

> **Warning:** Reporting a drain as a cancel ends every training run for good on every deploy; reporting a cancel as a drain retries a run the operator stopped. Take `stopReason` from `Subprocess.Canceled(context.Cause(ctx))` and let it override the child's exit code.

On the Python side, the pitfalls break the file agreement. Unflushed progress means the runner sees nothing until exit; `emit` flushes after every line. Absolute paths in events or `DONE.json` are right on one pod and wrong on the next; `_artifact_path` makes every path relative to `TL_ARTIFACTS`. A NaN loss written as `NaN` makes the line not JSON; `emit` rejects non-finite floats and dumps with `allow_nan=False`. A `ckpt` event before the checkpoint is complete leaves the retry unable to load its resume point; `checkpoint` emits only when `verify_step_dir` returns no problems. Writing outputs under their final name lets a kill leave a truncated `DONE.json` that looks finished; `publish` writes `name.tmp` and renames. Unconfined output names let one activity overwrite another's files; `output` rejects absolute names and anything outside the work dir. Ignoring spec validation errors trains on garbage, or retries; `load_spec` raises `SpecError` (exit 65) for bad JSON, non-objects, and validation errors. Skipping the `DONE.json` check on start makes a duplicate delivery train again; `run` re-emits `done` and exits 0 when the file exists. A SIGTERM handler that exits at once loses the work since the last checkpoint; the handler only sets a flag, and the loop checkpoints and then raises `Cancelled`. Without a work-dir lock, an orphan and its successor write one directory; `run` takes a non-blocking `flock` and exits 75 if it cannot. And undetected orphans train on after their worker died; `cancelled` compares `os.getppid()` with the pid recorded at start.

The exporter has its own set. Ignoring the remote parent makes Python spans a separate trace; `Tracer` takes the trace id and parent span id from `TRACEPARENT`. Encoding `intValue` as a JSON number makes collectors reject or round the attribute; `_value` encodes ints as strings. Lenient `traceparent` parsing lets all-zero ids glue unrelated runs together; `parse_traceparent` rejects all-zero ids, version `ff`, and extra fields on `00`. Ignoring the sampled flag produces orphan spans whose parents were never recorded; `self.sampled` comes from the parent, and `end_span` and `flush` honour it. Ignoring `OTEL_SERVICE_NAME` makes every system's Python report as `tinyllm-python`; `from_env` reads it and falls back only when unset. Raising export errors lets a collector outage kill training; `_post` catches network errors and returns `False`. Swallowing exceptions in the span makes a failing stage look successful; `span` marks the error and re-raises. Sampling off by one never traces step 0 but always traces step 1; `should_sample_step` is `step % every == 0`. Keeping the first gauge value makes dashboards show the loss of step 1; `gauge` overwrites the entry for `(name, attributes)`. And queueing spans after shutdown grows memory at exit and half-sends spans; `shutdown` sets `_closed`, and `end_span` stops queueing.

## Where this leads

This section stands on two earlier pieces. `verify_step_dir` from [Safetensors for every dtype, atomic checkpoints, and the token stream](../../../ai-engineering/06-numerics-and-autograd/07-safetensors-checkpoints-and-token-streams.md) gates every `ckpt` event, and `--resume` loads with `load_checkpoint`. The worker and activity SDK from [the task protocol and workers](../04-task-protocol-and-workers.md) and [activities](../05-activities.md) are what the composition root adapts to `Task` and `Heartbeat`.

Everything Python on the platform now runs through it. `CorpusBuild` runs each corpus stage as `{corpus} run --stage <s>` through the runner ([CorpusBuild: the pipeline as a durable workflow](../../../data-engineering/05-corpus-pipeline/09-corpusbuild-as-a-durable-workflow.md)). `TrainRun` and `EvalSuite` run `{tinyllm} train` and `eval`, and kills resume from the heartbeated checkpoint ([Platform workflows: TrainRun and EvalSuite](../11-trainrun-and-evalsuite.md)). The control-plane trace reaches `train.step` through this exporter ([Observability for the control plane](../../11-observability/06-control-plane-observability.md) in *Systems Engineering*). The worker image carries the activity helper and the exporter ([Durable and worker images and charts](../../10-containers-kubernetes/07-durable-and-worker-charts.md) in *Systems Engineering*). And the `export` step of [ModelRelease](../12-model-release.md) and the [TinyStories capstone](../../../ai-engineering/capstones/01-tinystories.md), whose run is a `TrainRun`, use the same contract.

Production systems extend each piece. Temporal's activity heartbeats with details (Go SDK `activity.RecordHeartbeat`, `activity.GetHeartbeatDetails`) add heartbeat throttling, details as typed payloads, and async completion for work finished elsewhere. Kubernetes Job `podFailurePolicy` generalizes the exit-code table into rules that fail the Job, ignore the failure, or count it (Kubernetes docs, "Handling retriable and non-retriable pod failures"). systemd `KillMode=control-group` and Kubernetes `terminationGracePeriodSeconds` use cgroups instead of process groups, so no child can escape by calling `setsid` (systemd.kill(5)). Linux `PR_SET_PDEATHSIG` replaces orphan detection by parent pid: the kernel signals the child when the parent thread dies (prctl(2)). And the OpenTelemetry Python SDK with the OTLP/HTTP exporter (`opentelemetry-exporter-otlp-proto-http`) adds a batching span processor, retries with backoff, and protobuf encoding.

### Further reading

- [Temporal: activity heartbeats and cancellation](https://docs.temporal.io/encyclopedia/detecting-activity-failures).
- [W3C Trace Context](https://www.w3.org/TR/trace-context/).
- [OTLP specification, JSON encoding](https://opentelemetry.io/docs/specs/otlp/#json-protobuf-encoding).
- W. Richard Stevens, *Advanced Programming in the UNIX Environment*, chapters 9 and 10 (process groups, signals).

## Key terms

- **Subprocess activity**: one attempt at one activity, run as a child process of a Go worker, whose interface is files, environment variables, signals, and an exit code.
- **Work directory**: `<TL_ARTIFACTS>/activities/<key>/`, shared by every attempt of one activity so that a retry resumes.
- **Heartbeat details**: the bytes sent with each heartbeat and returned to the next attempt; here, the last complete checkpoint path.
- **Publish by rename**: writing to a temp file, fsyncing, and renaming, so a reader sees the old file or the whole new one.
- **Process group**: the set of processes led by the child, signalled together with `kill(-pgid, sig)`.
- **Worker shutdown**: a retryable stop caused by the worker going away, as opposed to a non-retryable cancel.
- **TRACEPARENT**: the W3C header that carries the parent span's trace id, span id, and sampled flag into the child.

## Summary

- Python never speaks gRPC to the platform. A Go worker runs it as a child process, and the whole interface is files, environment variables, signals, and an exit code.
- The exit code is the verdict: `65` never retries, `75` and crashes retry, `130` is a cancellation. A worker drain is not a cancellation, so the runner reports it as retryable even when the child exits 130.
- A checkpoint becomes a heartbeat detail the moment it is complete, and the next attempt gets it as `--resume`. Until a newer one exists, every heartbeat keeps sending the old one.
- Every attempt shares one work directory named by the idempotency key. Outputs are published by rename, `DONE.json` last, and an exclusive lock keeps an orphaned child from writing beside its successor.
- The Python spans continue the activity's trace from `TRACEPARENT`, honour its sampled flag, and a dead collector never fails training.

## Review questions

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
6. Why does the tail of a new attempt start at the current end of `progress.jsonl` rather than at byte 0?
   <details><summary>Answer</summary>Earlier attempts' events are history. Their resume point already arrived through `LastHeartbeat`, so rereading them would only replay old checkpoints; starting at the file's current size means each attempt reads only its own events.</details>

[Contents](README.md)
