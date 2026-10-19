# What the System Does After Pass 8

After Pass 8 the data pipeline and training run as workflows (CorpusBuild, TrainRun, and EvalSuite) on the course's own durable engine, with durable and worker charts autoscaled by KEDA and control-plane traces. The durable engine runs workflows to completion through the CLI, locally and on kind. As optional depth, the durable log survives leader kills and partitions on three Raft replicas. Everything from earlier passes keeps working. This page gives the commands the finished solution runs, what each produces, and why each piece is needed.

The engine is the durable-orchestration chapter of *AI Platform Engineering*, from [the append-only event log](../../../ai-platform-engineering/05-durable-orchestration-and-workers/01-event-log.md), [the workflow service](../../../ai-platform-engineering/05-durable-orchestration-and-workers/02-workflow-service.md), and [the task queue](../../../ai-platform-engineering/05-durable-orchestration-and-workers/03-task-queue.md) through [workers](../../../ai-platform-engineering/05-durable-orchestration-and-workers/04-task-protocol-and-workers.md), [activities](../../../ai-platform-engineering/05-durable-orchestration-and-workers/05-activities.md), [deterministic replay](../../../ai-platform-engineering/05-durable-orchestration-and-workers/06-deterministic-replay.md), [timers](../../../ai-platform-engineering/05-durable-orchestration-and-workers/07-durable-timers.md), [signals and sagas](../../../ai-platform-engineering/05-durable-orchestration-and-workers/08-signals-cancellation-and-sagas.md), and [subprocess activities](../../../ai-platform-engineering/05-durable-orchestration-and-workers/09-subprocess-activities.md), to the workflows [CorpusBuild](../../../data-engineering/05-corpus-pipeline/09-corpusbuild-as-a-durable-workflow.md) and [TrainRun and EvalSuite](../../../ai-platform-engineering/05-durable-orchestration-and-workers/11-trainrun-and-evalsuite.md). Around it are [the durable and worker charts](../../../infrastructure/01-containers-kubernetes/06-durable-and-worker-charts.md), [control-plane observability](../../../systems/04-observability/05-control-plane-observability.md), [resilience tests](../../../software-craftsmanship/03-testing-mentality/07-resilience-tests.md), and two drills: [the durable server SIGKILLed in a loop](../../../systems/05-incident-response-and-chaos/02-durable-kill9.md) and [a poison task](../../../systems/05-incident-response-and-chaos/03-poison-task.md). The optional depth is [Raft high availability](../../../ai-platform-engineering/05-durable-orchestration-and-workers/10-raft-ha.md) with the [Raft partition drill](../../../systems/05-incident-response-and-chaos/12-raft-partition.md).

## Workflows run to completion, locally and on kind

The durable server, the worker, and the Python activities the worker execs run workflows to completion through the system's own CLI. A start is idempotent, the history is recorded, CorpusBuild produces the same corpus as the direct `{corpus} run` of Pass 3, and a short TrainRun finishes. On kind, the durable StatefulSet and the KEDA-scaled workers carry a queue burst with control-plane traces, and the drills durable-kill9 and poison-task are resolved.

The point of the pass is that failures do not change results. Under a kill loop (500 `tl.test.Append` activities with SIGKILLs of `{durable}`, `{worker}`, and the Python children), every effect lands exactly once in an effects sink. CorpusBuild under kills of both services produces the same corpus as an uninterrupted run, and TrainRun under kills of the worker reaches the same steps and final loss. The same invariants hold one level down, in the sections themselves: the event log is tested under a kill loop, the task queue with a poison task and a stale lease, workers and activities under a kill loop with the effects sink, deterministic replay with the server killed mid-workflow and replayed, and subprocess activities with a worker SIGKILLed mid-activity that resumes from its checkpoint.

Why this matters: corpus builds and training runs take hours. Running them as durable workflows means that a crash, a deploy, or a preempted node costs a retry from the last recorded step, not the whole run, and the history says exactly what happened.

## The workflow verbs

The solution defines the Pass 8 verbs of the `ctl` role under "Verbs of later passes" in [`cli-roles.md`](../../../../../course/contracts/spec/cli-roles.md). Every verb keeps the rules of that page: exit 2 on a usage error, exit 1 on an RPC error with its gRPC code and message on stderr, and the last stdout line is one JSON object. `--durable <host:port>` names the server (default `$TL_DURABLE_ADDR`, else `127.0.0.1:7233`). Status names drop the proto prefix: `RUNNING`, `COMPLETED`, `FAILED`, `CANCELED`, `CONTINUED_AS_NEW`.

```text
{ctl} wf start <type> --id <workflow id> [--input-json <json>] [--queue <q>] [--wait] [--json]
    StartWorkflow (task queue default "default", input the JSON bytes,
    default null); with --wait, DescribeWorkflow until the run is not
    RUNNING. Final line: {"workflow_id", "run_id", "started", "status",
    "result"} where started is StartWorkflowResponse.started and result
    is the run's result parsed as JSON (null until COMPLETED).
    SleepDemo is registered with the durable timers: input JSON is an
    integer duration in milliseconds and its result is the fired time.
    tl.test.Batch (input N) runs N tl.test.Append activities with the
    SDK's activity ids 1 to N, so the effects sink receives the keys
    "<workflow id>/1" to "<workflow id>/N"; each append holds 100 ms
    after its effect (a kill then lands between effect and completion);
    its result is N. --wait polls through a server that answers
    Unavailable while it restarts, for up to 2 minutes in a row.

{ctl} wf describe <workflow id> --json
    DescribeWorkflow. Final line: {"workflow_id", "run_id", "status",
    "history_length", "pending_activities", "dead_lettered"} (the last two
    are counts).

{ctl} wf history <workflow id> --json
    GetHistory, every page: one JSON line per event {"event_id", "type"}
    (type is the HistoryEvent attribute name, e.g.
    "WorkflowExecutionStarted"), then a final line {"events": N,
    "first": "<type>", "last": "<type>"}.

{ctl} wf dlq list <queue> --json
    ListDeadLetters, every page: one JSON line per dead letter
    {"task_id", "workflow_id", "activity_type", "attempts"}, then
    {"queue", "count"}.

{ctl} wf dlq redrive <queue> (--all | <task id>...) --json
    RedriveDeadLetter. Final line: {"queue", "redriven"}.

{ctl} data build --config <corpus toml> --id <workflow id> [--wait] [--json]
    Starts CorpusBuild on the config. With --wait, the final line is the
    CorpusBuild result: Pass 3's RUN-<dataset>-<version> object (the same
    keys and values as {corpus} run) plus "workflow_id" and "status".

{ctl} train --spec <file> --id <workflow id> [--wait] [--json]
    Starts TrainRun on a train spec (formats/train-spec.schema.json).
    A spec without "out" gets out = "runs/<workflow id>", so a run
    resumes only its own checkpoints. With --wait, the final line is
    {"workflow_id", "status", "steps", "final_loss", "ckpt"}; the loss is
    the run directory's final.json, the same after a retry that resumed
    from the final checkpoint.
```

## How the solution is wired

The entry points are under [`go/cmd`](../../../../../course/solutions/entry/go/cmd/durable/), and the charts under [`deploy/helm`](../../../../../course/solutions/entry/deploy/helm/):

```text
[entry].durable, [entry].worker, [entry].ctl
[services.durable]  health = "http://127.0.0.1:{health_port}/healthz"; its
                    config template puts the WAL under {data}
[services.worker]   after = ["durable"], argv --queue default --durable
                    127.0.0.1:{durable.grpc_port} --test-activities plus
                    queue data for CorpusBuild, and an artifacts root
                    of {data} (the subprocess contract's /artifacts), so
                    the files land in {worker.data}
specs/tiny.json     the TrainRun spec: a few hundred thousand parameters,
                    20 steps, its data the tokens CorpusBuild wrote
                    (paths relative to the artifacts root:
                    "tokens/bytes/small/train-00000.bin")
[deploy]            as for Pass 7's production deploy, plus services.durable =
                    "statefulset/<system>-durable" and services.worker =
                    "deploy/<system>-worker"; the durable NodePort is 30733
                    (set up with the kind cluster's port mappings)
```

The port mapping comes from [the kind cluster and local registry](../../../infrastructure/01-containers-kubernetes/02-kind-cluster-and-local-registry.md), and the StatefulSet and worker Deployment from [the durable and worker charts](../../../infrastructure/01-containers-kubernetes/06-durable-and-worker-charts.md).

## Optional depth: three Raft replicas survive leader kills and partitions

The durable server with `--replicas 3`, three nodes of the Raft implementation talking `tl.raft.v1`, carries the durable kill loop: workflows run while the leader is killed and partitioned away, every acknowledged append survives, and the client history is linearizable.

It is worth being precise about what is demonstrated and what is not. The Raft section's tests check the raft package in-process: a seeded simulation with partitions, drops, reordering, crash-restarts, and a linearizability check over the client history, plus a leader kill with 20 acknowledged appends. On kind, the StatefulSet runs three Ready replicas that complete a workflow through the NodePort. A full kill loop against three local replicas (killing a named replica and cutting links between them while workflows run) is not part of the solution yet.

On kind, the durable chart renders `replicas: 3` with `[durable].raft.enabled`, `raft.id` taken from the pod ordinal, and the peers `<system>-durable-<i>.<system>-durable-headless:7234`, using the headless Service of the durable chart. A follower answers a client RPC with `UNAVAILABLE` and the trailer `tl-raft-leader` (defined in `proto/tl/raft/v1/raft.proto`). The `ctl` retries against the leader it names, so `--durable 127.0.0.1:30733` reaches whichever pod the NodePort picks.

[Pass 9](../course-p09-capstone-training/README.md) uses these workflows to train and release the capstone.
