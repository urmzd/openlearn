# Subprocess activities: the Go runner and the Python helper

The durable engine can now run Go activities with retries, heartbeats, and fenced leases ([the task queue](../03-task-queue.md), [the task protocol and workers](../04-task-protocol-and-workers.md), [activities](../05-activities.md)). But everything that matters to the platform's users is Python: the corpus stages of [the corpus pipeline](../../../data-engineering/05-corpus-pipeline/README.md) in *Data Engineering* and the trainer from *[Build an LLM](../../../ai-engineering/README.md)*. Today `{tinyllm} train` is a command you type. If the laptop sleeps at step 9,000, the run is gone, and nothing records which checkpoint was last good.

The obvious fix, starting the trainer from a Go activity with `exec.Command` and waiting, breaks three things at once. The server sees no heartbeats and fences the activity mid-run. A worker restart kills the child without a checkpoint, or worse, leaves it running. And a retry starts again from step 0 in a fresh directory. This section builds the bridge that avoids all three: a Go runner and a Python helper that keep one written contract, so that `CorpusBuild` and `TrainRun` can run Python work durably. With it in place, `<system> train --spec specs/tiny.json` survives a worker kill.

You will need the atomic checkpoints of [Safetensors for every dtype, atomic checkpoints, and the token stream](../../../ai-engineering/06-numerics-and-autograd/07-safetensors-checkpoints-and-token-streams.md) (its `verify_step_dir` decides when a checkpoint may be reported) and a working knowledge of exit codes and signals from [Shell, git, make, processes](../../../programming-and-algorithms/01-language-primers/02-shell-git-make.md) in *Programming and Algorithms*. The interface between the two halves is written down in [`course/contracts/spec/subprocess-activity.md`](../../../../../course/contracts/spec/subprocess-activity.md), with Python stubs in [`activity.pyi`](../../../../../course/contracts/py/tinyllm/io/activity.pyi) and [`telemetry.pyi`](../../../../../course/contracts/py/tinyllm/io/telemetry.pyi) and the span names in [`otel/semconv.md`](../../../../../course/contracts/otel/semconv.md). The reference implementation is [`go/activities/subprocess.go`](../../../../../course/solutions/go/activities/subprocess.go) (`Subprocess.Run`, `Task`, `Heartbeat`, `Failure`, `WorkDir`, `Tail`, `Classify`), [`python/tinyllm/io/activity.py`](../../../../../course/solutions/python/tinyllm/io/activity.py) (`Activity`, `run`, the exit codes), and [`python/tinyllm/io/telemetry.py`](../../../../../course/solutions/python/tinyllm/io/telemetry.py) (`Tracer`, `parse_traceparent`, OTLP/HTTP JSON export), wired into the worker by [`entry/go/cmd/worker/main.go`](../../../../../course/solutions/entry/go/cmd/worker/main.go).

> **Learning objectives**
> - Explain why the boundary between the platform and Python is a process, and what crosses it: files, environment variables, signals, and an exit code.
> - Map every exit code to a retry verdict, and explain why a worker drain is not a cancellation even when the child exits 130.
> - Turn checkpoints into heartbeat details so that a retry resumes instead of restarting, and keep the resume point alive until a newer one exists.
> - Share one work directory across attempts safely with publish-by-rename, `DONE.json` last, and an exclusive lock.
> - Continue the activity's trace into Python from `TRACEPARENT` without ever letting telemetry fail the work.

## In this part

1. [Process contract and recovery](01-process-contract-and-recovery.md): Define subprocess inputs, durable progress, termination, and telemetry.
2. [The Go subprocess runner](02-go-runner.md): Trace process-group startup, monitoring, shutdown, and retry state in the runner.
3. [Python helper and exporter](03-python-helper-and-exporter.md): Connect application progress/checkpoints to durable orchestration and export.
4. [Failure tests and integration](04-failure-tests-and-integration.md): Challenge retry, timeout, checkpoint, and process-cleanup behavior.
