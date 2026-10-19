# Pass 8: Durable Execution

Corpus builds and training runs take hours, and over hours something always fails: a process crashes, a deploy restarts a pod, a node is preempted. This pass builds the course's own durable execution engine so that a failure costs a retry from the last recorded step rather than the whole run, and so that the history says exactly what happened. It starts with an append-only event log and a task queue with fenced leases, adds a workflow service, a worker SDK, activities with retries and heartbeats, deterministic replay, timers, signals, and sagas, and then runs the Pass 3 corpus pipeline and training as workflows.

The point of the pass is that failures do not change results. Every piece is tested under kill loops, and the end-to-end checks require that a corpus build or training run killed repeatedly produces exactly what an uninterrupted run produces. On kind, the durable server runs as a StatefulSet with KEDA-scaled workers and control-plane traces, and two drills (the server killed in a loop, and a poison task) are resolved.

> **Learning objectives**
> - Explain how the event log, task queue, and leases make every effect land exactly once.
> - Explain how deterministic replay rebuilds a workflow's state from its history.
> - Explain how retries, backoff, heartbeats, and idempotency keys make activities safe to repeat.
> - Explain how CorpusBuild and TrainRun reproduce their direct-run results under kills.

## Before you begin

This pass needs [Pass 7](../course-p07-serving-platform/README.md), whose kind deployment and observability stack the durable services join.

## In this pass

1. [Append-only segmented event log](../../../ai-platform-engineering/05-durable-orchestration-and-workers/01-event-log.md). Afterwards you can explain how the append-only segmented event log writes, syncs, rolls over, and recovers.
2. [Task queue: visibility timeout, fenced leases, retries to DLQ, long poll](../../../ai-platform-engineering/05-durable-orchestration-and-workers/03-task-queue.md). Afterwards you can explain how the task queue uses visibility timeouts and fenced leases, retries to a DLQ, and long-polls.
3. [Workflow service + idempotent start, describe, list, paged history](../../../ai-platform-engineering/05-durable-orchestration-and-workers/02-workflow-service.md). Afterwards you can explain how the workflow service makes start idempotent and serves describe, list, and paged history.
4. [Task gRPC protocol, Go worker SDK, worker pool](../../../ai-platform-engineering/05-durable-orchestration-and-workers/04-task-protocol-and-workers.md). Afterwards you can explain the task gRPC protocol and how the Go worker SDK and worker pool poll and complete tasks.
5. [Activities: retries, backoff + jitter, timeouts, heartbeats, idempotency keys](../../../ai-platform-engineering/05-durable-orchestration-and-workers/05-activities.md). Afterwards you can explain how activities retry with backoff and jitter, time out, heartbeat, and use idempotency keys.
6. [Deterministic replay workflows, ContinueAsNew, history paging, payload limits](../../../ai-platform-engineering/05-durable-orchestration-and-workers/06-deterministic-replay.md). Afterwards you can explain why workflows must be deterministic for replay and how ContinueAsNew, history paging, and payload limits keep history bounded.
7. [Durable timers and the test clock](../../../ai-platform-engineering/05-durable-orchestration-and-workers/07-durable-timers.md). Afterwards you can explain how durable timers survive restarts and how the test clock drives them.
8. [Signals, cancellation, sagas (compensation)](../../../ai-platform-engineering/05-durable-orchestration-and-workers/08-signals-cancellation-and-sagas.md). Afterwards you can explain how signals and cancellation reach a workflow and how a saga compensates completed steps.
9. [Subprocess activity runner (Go) and the Python activity helper](../../../ai-platform-engineering/05-durable-orchestration-and-workers/09-subprocess-activities.md). Afterwards you can explain how the subprocess activity runner launches Python activities and how the helper reports results.
10. [Pipeline as a durable workflow CorpusBuild](../../../data-engineering/05-corpus-pipeline/09-corpusbuild-as-a-durable-workflow.md). Afterwards you can explain how CorpusBuild runs the data pipeline as a durable workflow and resumes after a crash.
11. [Platform workflows TrainRun and EvalSuite](../../../ai-platform-engineering/05-durable-orchestration-and-workers/11-trainrun-and-evalsuite.md). Afterwards you can explain how the TrainRun and EvalSuite workflows orchestrate training and evaluation.
12. [Durable and worker images and charts: WAL PVC, StatefulSet, KEDA autoscaling of workers](../../../infrastructure/01-containers-kubernetes/06-durable-and-worker-charts.md). Afterwards you can explain why the durable server is a StatefulSet with a WAL PVC and how KEDA autoscales the workers.
13. [Observability for the control plane: workflow and activity spans, queue and DLQ panels](../../../systems/04-observability/05-control-plane-observability.md). Afterwards you can explain the workflow and activity spans and what the queue and DLQ panels show.
14. [Resilience tests](../../../software-craftsmanship/03-testing-mentality/07-resilience-tests.md). Afterwards you can explain how resilience tests inject crashes and partitions and what they assert.
15. [The durable server is SIGKILLed in a loop during a corpus build](../../../systems/05-incident-response-and-chaos/02-durable-kill9.md). Afterwards you can explain why a corpus build survives the durable server being SIGKILLed in a loop, and diagnose it when it does not.
16. [A poison task lands in the dead-letter queue](../../../systems/05-incident-response-and-chaos/03-poison-task.md). Afterwards you can explain how a poison task reaches the dead-letter queue and how it is diagnosed and redriven.
17. [What the system does after Pass 8](milestone.md). Afterwards you can explain how CorpusBuild, TrainRun, and EvalSuite run to completion on the durable engine, locally and on kind, and survive crashes.

## Going deeper

One optional section replicates the durable log with Raft:

18. [Raft HA for the durable log (optional)](../../../ai-platform-engineering/05-durable-orchestration-and-workers/10-raft-ha.md). Afterwards you can explain how Raft elects a leader and replicates the durable log so it survives leader loss.

The stages are also in this pass's manifest, [`path.tsv`](path.tsv). [Pass 9](../course-p09-capstone-training/README.md) uses these workflows to train and release the course's first model worth serving.
