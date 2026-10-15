# Incident Response and Chaos

## Overview

- **Primary reference**: [*Site Reliability Engineering*](https://sre.google/sre-book/table-of-contents/) (Google, free): ch. 12 Effective Troubleshooting, ch. 14 Managing Incidents, ch. 15 Postmortem Culture
- **Supplementary**: [*The Site Reliability Workbook*](https://sre.google/workbook/table-of-contents/) (free), ch. 9 Incident Response and ch. 10 Postmortem Culture; *Chaos Engineering* by Rosenthal and Jones (O'Reilly); Kubernetes docs, [Debug Applications](https://kubernetes.io/docs/tasks/debug/debug-application/) (free)
- **Prerequisites**: [Cloud Native](../03-cloud-native/), [Observability](../04-observability/)
- **Estimated time**: one drill per pass of the course, 1 to 2 h each; the full set in Pass 11

## Key Takeaways

- Incidents are learned by causing them on purpose: a drill injects a known fault into your own running system and grades how you detect, mitigate, and record it.
- Mitigate first (roll back, fail over, shed load), find the root cause second, write it down third.
- A runbook is written before the incident, for the person who did not build the system; a postmortem is written after, without blame.
- Detection moves from a person noticing (Pass 1) to burn-rate alerts (Pass 7), and `ol drill end` measures time to detect and time to mitigate from Prometheus.

## How to Study

- Do each drill when its pass reaches it, against your own kind cluster, with the time limit on.
- Write the runbook during the drill, not after: the commands you actually typed are the Diagnosis section.
- Read one chapter of the SRE book after each drill and compare its advice with what you did.

---

# Concepts & Techniques

## Core Insight

Every system fails; what differs is how long it stays failed. Time to mitigate is mostly time spent working out what is wrong, so the work that pays is making the next diagnosis shorter: telemetry that shows the cause, runbooks that say where to look, and rollouts that can be undone in one command.

## 1. Triage on Kubernetes

**Key ideas**:
- **Outside in**: which workload is unhealthy, why its last container exited (exit code and reason), what its previous log says, what changed (rollout history).
- **Exit codes**: below 128 the program chose to exit; `128 + n` means signal `n` (137 SIGKILL, usually OOMKilled; 143 SIGTERM).
- **CrashLoopBackOff**: restarts wait 10 s, doubling to 5 minutes.

## 2. Mitigation

**Key ideas**:
- **Roll back** the change that started it (`kubectl rollout undo`, `helm rollback`), then fix forward through the chart.
- **Readiness and rollout strategy** decide whether a bad rollout becomes an outage.

## 3. Learning from incidents

**Key ideas**:
- **Blameless postmortems**: Summary, Impact, Timeline, Root cause, Detection, Resolution, Action items.
- **Action items** that prevent, detect, or mitigate the next occurrence, each with an owner.

## Chapters

<!-- ol:chapters -->
| # | Module | Chapter | Kind | Pass |
|---|---|---|---|---|
| 1 | `ops.00` | [First drill: the engine crashloops](00-first-drill.md) | drill | 1 |
| 2 | `ops.01` | [A decode worker dies mid-stream](01-kill-decode.md) | drill | 7 |
| 3 | `ops.02` | [The durable server is SIGKILLed in a loop during a corpus build](02-durable-kill9.md) | drill | 8 |
| 4 | `ops.03` | [A poison task lands in the dead-letter queue](03-poison-task.md) | drill | 8 |
| 5 | `ops.04` | [Drill: KV v2 migration](04-kv-v2-migration.md) | drill | 11 |
| 6 | `ops.05` | [Drill: API v2 migration](05-api-v2-migration.md) | drill | 11 |
| 7 | `ops.06` | [Drill: breaking dependency upgrade](06-dependency-upgrade.md) | drill | 11 |
| 8 | `ops.07` | [Drill: performance regression bisect](07-performance-regression.md) | drill | 11 |
| 9 | `ops.08` | [Drill: data incident](08-data-incident.md) | drill | 11 |
| 10 | `ops.09` | [Drill: noisy neighbor](09-noisy-neighbor.md) | drill | 11 |
| 11 | `ops.10` | [Drill: runaway agent](10-runaway-agent.md) | drill | 11 |
| 12 | `ops.11` | [Drill: event log disk full](11-eventlog-disk-full.md) | drill | 11 |
| 13 | `ops.12` | [Drill: Raft partition (optional)](12-raft-partition.md) | drill | 11 |
<!-- /ol:chapters -->

## Connections to Other Tracks

| Track | Connection |
|---|---|
| [Observability](../04-observability/) | the SLOs and burn-rate alerts that detect what these drills inject |
| [Cloud Native](../03-cloud-native/) | the Kubernetes objects every drill acts on |
| [Software Craftsmanship: documentation](../../software-craftsmanship/06-documentation-writing/) | runbooks and decision records |

## Company Relevance

| Company | Practice |
|---|---|
| Google | SRE incident command and blameless postmortems |
| Netflix | chaos engineering against production |
| Any on-call team | runbooks linked from alerts; game days |
