# What the system does after Pass 8

**Pass result**: data pipeline and training run as workflows (CorpusBuild, TrainRun, EvalSuite) on their durable engine; durable and worker charts with KEDA; control-plane traces.

Each capability below is built from the listed modules. The sections that follow give the commands the finished solution runs, what each one produces, and why each piece is needed. Everything from earlier passes keeps working.

| What the system does | Built from |
|---|---|
| The durable engine runs workflows to completion through the CLI, locally and on kind. | `dur.01`, `dur.02`, `dur.03`, `dur.04`, `dur.05`, `dur.06`, `dur.07`, `dur.08`, `dur.09`, `data.09`, `dur.11`, `dep.06`, `obs.05`, `craft.21`, `ops.02`, `ops.03` |
| Durable workflows: data and training runs survive worker failures | - |
| The durable log survives leader kills and partitions on three Raft replicas. | `dur.10`, `MS-durable`, `ops.12` |


## Details

## The durable engine runs workflows to completion through the CLI, locally and on kind

"MS-durable", 5.7 catalog). MS-P8 includes it. Page:
content/courses/build-an-llm/paths/course-p08-durable/milestone.md (owned by the pass gate's author).

The durable server, the worker, and the Python activities it execs run
workflows to completion through its own CLI: a start is idempotent, the
history is recorded, CorpusBuild produces the same corpus as the direct
`{corpus} run` of MS-corpus, and a short TrainRun finishes. On kind the
durable StatefulSet and the KEDA-scaled workers (dep.06) carry a queue
burst with control-plane traces (obs.05), and the drills durable-kill9
(ops.02) and poison-task (ops.03) are resolved.

The design's kill loop (500 tl.test.Append activities with SIGKILLs of
{durable}, {worker}, and the Python children, exactly-once effects in the
testkit `effects` sink, then CorpusBuild and TrainRun under kills with
outputs equal to uninterrupted runs) uses the runner's `chaos` and `sink`
step keys and the `effects` matcher (DESIGN 5.7). Section 4 runs all
three: CorpusBuild under kills of both services, 500 appends of the
worker's tl.test.Batch under kills of both, exactly once in the sink, and
TrainRun under kills of the worker, with the steps and final loss of the
uninterrupted run. The same invariants are checked one level down by the
course tests in `requires`: dur.01 (KillLoop over the log), dur.03 (poison
task, stale lease), dur.04 and dur.05 (KillLoop over a worker with the
effects sink), dur.06 (kill the server mid-workflow and replay), dur.09
(SIGKILL a worker mid-activity, resume from the checkpoint).

The solution defines the Pass 8 verbs of the `ctl` role ([`cli-roles.md`](../../../../../course/contracts/spec/cli-roles.md),
"Verbs of later passes"). Every verb keeps the rules of that page: exit 2
on a usage error, exit 1 on an RPC error with its gRPC code and message on
stderr, the last stdout line is one JSON object. `--durable <host:port>`
names the server (default $TL_DURABLE_ADDR, else 127.0.0.1:7233). Status
names drop the proto prefix: RUNNING, COMPLETED, FAILED, CANCELED,
CONTINUED_AS_NEW.

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
      Starts CorpusBuild (data.09) on the config. With --wait, the final
      line is the CorpusBuild result: MS-corpus's RUN-<dataset>-<version>
      object (the same keys and values as `{corpus} run`) plus
      "workflow_id" and "status".
  {ctl} train --spec <file> --id <workflow id> [--wait] [--json]
      Starts TrainRun (dur.11) on a train spec (formats/train-spec.schema.json).
      A spec without "out" gets out = "runs/<workflow id>", so a run
      resumes only its own checkpoints. With --wait, the final line is
      {"workflow_id", "status", "steps", "final_loss", "ckpt"}; the loss is
      the run directory's final.json, the same after a retry that resumed
      from the final checkpoint.

The system.toml must declare:
  [entry].durable, [entry].worker, [entry].ctl
  [services.durable]  health = "http://127.0.0.1:{health_port}/healthz"; its
                      config template puts the WAL under {data}
  [services.worker]   after = ["durable"], argv `--queue default --durable
                      127.0.0.1:{durable.grpc_port} --test-activities` plus
                      queue `data` for CorpusBuild, and an artifacts root
                      of {data} (the subprocess contract's /artifacts), so
                      the files land in {worker.data}
  specs/tiny.json     the TrainRun spec: a few hundred thousand parameters,
                      20 steps, its data the tokens CorpusBuild wrote
                      (paths relative to the artifacts root:
                      "tokens/bytes/small/train-00000.bin")
  [deploy]            as for MS-prod, plus services.durable =
                      "statefulset/<system>-durable" and services.worker =
                      "deploy/<system>-worker" (dep.06); the durable NodePort
                      is 30733 (dep.02)

## Durable workflows: data and training runs survive worker failures

MS-durable covers durable workflows, corpus and training activities, charts,
observability, and the required drills. The optional high-availability
milestone MS-durable-ha is intentionally not a pass-gate dependency.
Earlier pass-gate smoke steps rerun through the spiral invariant.

## The durable log survives leader kills and partitions on three Raft replicas

"MS-durable-ha", 5.7 catalog). Not part of the MS-P8 gate.

The durable server with `--replicas 3` (three nodes of the dur.10 Raft,
talking tl.raft.v1) carries the durable kill loop: workflows run while the
leader is killed and partitioned away, every acknowledged append survives,
and the client history is linearizable.

What runs today, and what does not. A local three-replica run needs every
replica's raft port before the first starts (the runner allocates ports one
service at a time and starts each before the next exists), and the kill
loop needs the runner to SIGKILL a named service, cut links between
services (testkit chaosproxy), record the history, and check it. None of
that exists yet (DEVIATIONS B103-05), so this file holds:
  - the steps that run now: dur.10's course tests carry the same checks
    against the raft package in-process (a seeded simulation with
    partitions, drops, reordering, crash-restarts, and a linearizability
    check over the client history; a leader kill with 20 acknowledged
    appends), and on kind the StatefulSet runs three Ready replicas that
    complete a workflow through the NodePort;
  - the proposed kill-loop steps, commented at the end, as the target of
    the harness change.

The system.toml, besides MS-durable's: on kind, the durable chart renders
replicas: 3 with [durable].raft.enabled, raft.id from the pod ordinal, and
the peers `<system>-durable-<i>.<system>-durable-headless:7234` (dep.06's
headless Service). A follower answers a client RPC with UNAVAILABLE and the
trailer `tl-raft-leader` (proto/tl/raft/v1/raft.proto); the `ctl` retries
against the leader it names, so `--durable 127.0.0.1:30733` reaches
whichever pod the NodePort picks.