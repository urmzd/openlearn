# Incident Response and Recovery

Every system fails; what differs is how long it stays failed. Time to mitigate is mostly time spent working out what is wrong, so the work that pays is making the next diagnosis shorter: telemetry that shows the cause, runbooks that say where to look, and rollouts that can be undone in one command. This chapter teaches that work the way it is best learned, by causing incidents on purpose. Each section is a drill: it injects a known fault into the running system you built across these books, and walks through how the fault is detected, mitigated, and recorded.

Three habits run through every drill. The first is triage on Kubernetes, from the outside in: which workload is unhealthy, why its last container exited (exit code and reason), what its previous log says, and what changed (rollout history). Exit codes below 128 mean the program chose to exit; `128 + n` means signal `n` (137 is SIGKILL, usually OOMKilled; 143 is SIGTERM). A CrashLoopBackOff waits 10 s between restarts, doubling to 5 minutes. The second habit is mitigation before root cause: roll back the change that started it (`kubectl rollout undo`, `helm rollback`), then fix forward through the chart, remembering that readiness probes and the rollout strategy decide whether a bad rollout becomes an outage. The third is learning from the incident. A runbook is written before the incident, for the person who did not build the system; a postmortem is written after, without blame, in fixed sections (Summary, Impact, Timeline, Root cause, Detection, Resolution, Action items), with action items that prevent, detect, or mitigate the next occurrence, each with an owner.

The arc of the chapter follows the system's growth. The first drill has no alerting at all, so a person notices the crashloop. From the second drill on, burn-rate alerts do the noticing, and time to detect and time to mitigate can be measured from Prometheus. The first four sections are full worked incidents, each with a timeline worked by hand, a model runbook, and a model postmortem. The remaining nine are shorter drills built around the runbook a correct response requires: planned migrations and upgrades that can go wrong, regressions, bad data, noisy tenants, runaway agents, a full disk, and, optionally, a Raft partition. These practices come from the industry: Google's SRE incident command and blameless postmortems, Netflix's chaos engineering against production, and the habit, common to any good on-call team, of linking a runbook from every alert and running game days.

> **Learning objectives**
> - Triage a failing Kubernetes workload from the outside in, reading exit codes, previous logs, and rollout history.
> - Mitigate before finding the root cause, and choose between waiting, scaling, rolling back, and fixing forward.
> - Compute when a burn-rate alert fires for a given fault, and measure time to detect and time to mitigate.
> - Verify recovery by the SLO and by the system's output, not by pod status.
> - Write runbooks for symptoms and blameless postmortems whose root causes name system properties.

## Before you begin

The drills act on the Kubernetes objects of [Cloud Native](../09-cloud-native/README.md) and are detected by the SLOs, burn-rate alerts, and dashboards of [Observability](../11-observability/README.md), so read both chapters first. Runbooks and decision records are covered in depth in the [documentation writing chapter](../../software-engineering/12-documentation/README.md) of *Software Engineering*. Each drill names the local deployment sections and the companion *AI Engineering* sections that built the component it breaks. Before you read a drill's solution, predict the diagnosis commands and their output, then compare.

## In this chapter

1. [First drill: the engine crashloops](01-first-drill.md). A bad rollout points the engine at a missing model directory. With no alerting, you triage the crashloop by hand, roll back, and write your first runbook and postmortem.
2. [A decode worker dies mid-stream](02-kill-decode.md). One of two decode pods is deleted under load. The TTFT budget burns, the gateway's commit rule keeps streams honest, and the root cause is capacity during replacement.
3. [The durable server is SIGKILLed in a loop during a corpus build](03-durable-kill9.md). Five kills cost time but not correctness: leases redeliver, idempotent activities make the second delivery harmless, and the output must equal a clean run's.
4. [A poison task lands in the dead-letter queue](04-poison-task.md). A test-only failpoint reaches production and one task fails every attempt. You find what failed and what changed, fix, then redrive.
5. [The KV format v2 migration](05-kv-v2-migration.md). A planned format change where readers must upgrade before writers, with block-hash evidence and a one-flag rollback.
6. [The API v2 migration](06-api-v2-migration.md). v1 and v2 served side by side, selected per request by a header, with every signal split by version.
7. [A breaking dependency upgrade](07-dependency-upgrade.md). One dependency at a time, gated on contract suites and a like-for-like load comparison against the last known good image.
8. [A performance regression bisect](08-performance-regression.md). Control the measurement, fix the threshold in advance, and let `git bisect run` find the first bad commit.
9. [A data incident](09-data-incident.md). Bad data leaves every service green; quarantine, trace forward through the ledger, and rebuild from a verified snapshot.
10. [A noisy neighbor](10-noisy-neighbor.md). One tenant's flood raises everyone's TTFT; attribute it per tenant and contain only the offender.
11. [A runaway agent](11-runaway-agent.md). An agent loop bounded on every axis, with propagated cancellation and a terminal reason that names the budget.
12. [The event log disk fills up](12-eventlog-disk-full.md). A WAL quota turns a full disk into a clean `RESOURCE_EXHAUSTED` that never loses acknowledged work.
13. [A Raft partition (optional)](13-raft-partition.md). For the three-replica durable log: minority writes fail, the majority elects a new leader, and you heal before any membership change.

## Key terms

- **Outside-in triage**: diagnosing a failing Kubernetes workload from the outside in: which workload is unhealthy, why its last container exited, what its previous log says, and what changed in its rollout history.
- **Signal exit code**: a process exit code of 128 + n, meaning the process was killed by signal n (137 is SIGKILL, often OOMKilled; 143 is SIGTERM); codes below 128 are the program's own choice.
- **Mitigation before root cause**: restoring service first, usually by rolling back the change that started the incident, and finding the cause afterwards, because diagnosis under pressure is slow.
- **Fix forward**: correcting an incident by shipping a new change rather than reverting, done after mitigation and through the normal release path.
- **Game day**: a scheduled exercise that injects a known fault into a running system so the team practises detection, mitigation, and runbooks before a real incident.

## Further reading

- *Site Reliability Engineering* (Google, [free](https://sre.google/sre-book/table-of-contents/)): ch. 12 Effective Troubleshooting, ch. 14 Managing Incidents, ch. 15 Postmortem Culture. Read one chapter after each drill and compare its advice with the model response.
- *The Site Reliability Workbook* ([free](https://sre.google/workbook/table-of-contents/)): ch. 9 Incident Response and ch. 10 Postmortem Culture.
- Rosenthal and Jones, *Chaos Engineering* (O'Reilly).
- Kubernetes docs, [Debug Applications](https://kubernetes.io/docs/tasks/debug/debug-application/) (free).
