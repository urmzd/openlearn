# Subprocess activity contract

<!-- modules: dur.09 (Go runner go/activities/subprocess.go; Python helper python/tinyllm/io/activity.py and telemetry.py), data.09, dur.11, dur.12, C1
     conformance: activity/ (a fake Python script drives the runner) -->

Python never speaks gRPC to the platform (D9). A Go worker runs Python work as a **subprocess activity**: it execs one of the system's Python entries with a spec file and a progress file, watches it, and turns its exit code into the activity's result. This page is the whole interface between the two.

## Invocation

```
{tinyllm} <verb> --spec <dir>/spec.json --progress <dir>/progress.jsonl [--resume <ckpt dir>]
{corpus}  <verb> --spec <dir>/spec.json --progress <dir>/progress.jsonl
```

`{tinyllm}` and `{corpus}` are the system's Python entries from `system.toml` (spec/cli-roles.md). `<dir>` is the activity's work directory (below); paths in the spec and in progress events are relative to `/artifacts` (`TL_ARTIFACTS`).

| Entry, verb | `spec.json` validates against | Produces |
|---|---|---|
| `{tinyllm} train` | [`formats/train-spec.schema.json`](../formats/train-spec.schema.json) | checkpoints under the run directory ([checkpoint.md](../formats/checkpoint.md)) |
| `{tinyllm} eval` | [`formats/eval-spec.schema.json`](../formats/eval-spec.schema.json) | `evals/<suite>/<run_id>/results.jsonl` and `summary.json` |
| `{tinyllm} export` | [`formats/export-spec.schema.json`](../formats/export-spec.schema.json) | a released model directory |
| `{corpus} run --stage <s>` | [`formats/corpus-config.schema.json`](../formats/corpus-config.schema.json) (the runner writes the parsed TOML as JSON) | that stage's outputs (`fetch`, `filter`, `dedup_exact`, `dedup_near`, `pii`, `shard`, `tokenize`) |
| `{corpus} tokenize` | same | token streams ([tokens-bin.md](../formats/tokens-bin.md)) |

A spec that fails validation exits `65` before writing anything.

## Environment

| Variable | Value |
|---|---|
| `TRACEPARENT` | W3C context of the activity span (from `ActivityTask.trace_context`); Python makes its spans children of it |
| `TL_IDEMPOTENCY_KEY` | `ActivityTask.idempotency_key`, `<workflow_id>/<activity_id>`, stable across attempts |
| `TL_ATTEMPT` | `ActivityTask.attempt`, 1-based |
| `TL_ARTIFACTS` | `/artifacts` (`[paths].artifacts`) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | the collector (unset: export nothing) |
| `OTEL_SERVICE_NAME` | `<system>-python` |

## Work directory and idempotency

`<dir>` is `/artifacts/activities/<TL_IDEMPOTENCY_KEY>/`: the key's `/`-separated segments become directories (each segment matches `[A-Za-z0-9._-]+` and is not `.` or `..`, which workflow and activity ids guarantee). The same directory serves every attempt, which is what makes a retry resume rather than restart.

- Every output is written as `<name>.tmp` and published with an atomic `rename`.
- The last act of a successful run writes `<dir>/DONE.json` (`{"outputs": [...]}`, paths relative to `/artifacts`) the same way, then emits the `done` event and exits `0`.
- A run that starts and finds `DONE.json` re-emits its `done` event and exits `0` at once: a rerun after success is a no-op, so an activity delivered twice has one effect.
- The activity's result (`CompleteActivityTask.result`) is the bytes of `DONE.json`, at most 2 MiB.

## Progress and heartbeats

The child appends one JSON object per line to `--progress` and flushes after each line ([`formats/progress.schema.json`](../formats/progress.schema.json)):

```json
{"ts": 1760000000.5, "kind": "step", "step": 500, "loss": 2.31, "lr": 0.0009, "tokens": 8192000}
{"ts": 1760000001.0, "kind": "ckpt", "step": 500, "ckpt": "runs/train-1/ckpt/step-000500"}
{"ts": 1760000002.0, "kind": "metric", "name": "val_loss", "value": 2.40}
{"ts": 1760000003.0, "kind": "done", "outputs": ["runs/train-1/ckpt/step-020000"]}
```

A `ckpt` event is emitted only after the checkpoint is durable (checkpoint.md step 5). The Go runner tails the file and calls `RecordHeartbeat` at least every `heartbeat_timeout / 3` with `details` = the UTF-8 bytes of the last `ckpt` path (empty before the first). On a later attempt it passes `--resume <ckpt>` when `ActivityTask.last_heartbeat_details` is non-empty. A heartbeat answer with `cancel_requested` starts cancellation.

## Exit codes

| Code | Meaning | Runner action |
|---|---|---|
| `0` | success | `CompleteActivityTask` with `DONE.json` |
| `75` (`EX_TEMPFAIL`) | retryable failure (a flaky download, an OOM the next attempt may avoid) | `FailActivityTask`, retryable |
| `65` (`EX_DATAERR`) | the input is wrong (invalid spec, unlicensed source, poisoned shard) | `FailActivityTask` with `non_retryable` |
| `130` | cancelled (SIGTERM honored) | `FailActivityTask`, type `Canceled`, non-retryable |
| anything else, or killed by a signal | crash | `FailActivityTask`, retryable |

`Failure.type` is `ExitCode<n>` (or `Signal<n>`), and `Failure.message` the last 4 KiB of the child's stderr.

## Cancellation and shutdown

The runner starts the child in its own process group. To cancel (workflow cancel, heartbeat `cancel_requested`, or the worker's own SIGTERM drain), it sends SIGTERM to the group; the child checkpoints if it can, emits a `ckpt` event for it, and exits `130` within 30 s. After 30 s the runner sends SIGKILL to the group. A worker that dies outright (SIGKILL) leaves the activity to time out; the next attempt resumes from the last heartbeated checkpoint.
