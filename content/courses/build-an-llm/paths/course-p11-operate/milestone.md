# What the system does after Pass 11

**Pass result**: v1.0.0: migrations (KV v2, API v2 as build modules), upgrades, bisect, chaos, security, docs, reviews, mock engagement, defense.

The system is released as v1.0.0 and operated: its storage and API formats migrate without downtime, dependencies upgrade safely, a performance regression is bisected, the core incidents are rehearsed with runbooks and postmortems, its security is reviewed, and the whole design is defended in a review and in a mock customer engagement. Everything from earlier passes keeps working.

| What the system does | Built from |
|---|---|
| Core incident drills are detected, resolved, and documented. | `ops.01`, `ops.02`, `ops.03`, `ops.04`, `ops.05`, `ops.06`, `ops.07`, `ops.08`, `ops.09`, `ops.10`, `ops.11`, `ops.12` |
| v1.0.0 is released, migrated, reviewed, and handed off. | `craft.09`, `craft.10`, `craft.11`, `craft.12`, `craft.13`, `craft.14`, `craft.15`, `craft.16`, `craft.17`, `craft.18`, `craft.19`, `review.02`, `review.03`, `ethics.06`, `field.01`, `field.02`, `field.03`, `field.04`, `field.05`, `field.06`, `field.07`, `iv.01` |

## Details

## Core incident drills are detected, resolved, and documented

Each drill injects one failure into the running system: a decode pod killed under load, the durable server killed with `kill -9`, a poison task, the KV v1 to v2 and API v1 to v2 migrations, a dependency upgrade, a performance regression, a data incident, a noisy neighbour, a runaway agent, a full event-log disk, and a Raft partition. For each, the evidence is a runbook for the alert that fires and, where an incident happened, a blameless postmortem. The solution's runbooks are in [`docs/runbooks`](../../../../../course/solutions/docs/runbooks/RaftPartition.md) (the Raft partition runbook explains the linearizability guarantee the operator must preserve) and its postmortems in [`docs/postmortems`](../../../../../course/solutions/docs/postmortems/2026-10-09-kill-decode.md).

Why it is needed: a system is only operable if the people on call have rehearsed its failures. A runbook written before the incident, and a postmortem written after it, are what turn one person's debugging into the team's knowledge.

## v1.0.0 is released, migrated, reviewed, and handed off

The release follows the procedure in [`releases/v1.0.0.md`](../../../../../course/solutions/docs/releases/v1.0.0.md): a complete changelog entry, a clean tree with green CI, an annotated `v1.0.0` tag, and publication only after the tag workflow passes, with a written rollback. Around it: semantic versioning and a deprecation policy (craft.11, craft.12), the KV and API migrations as build modules (craft.13, craft.14), dependency upgrades and a bisected performance regression (craft.15, craft.16), a STRIDE threat model, an SBOM, and a secrets and authorization review (craft.17 to craft.19, see [`THREAT_MODEL.md`](../../../../../course/solutions/docs/THREAT_MODEL.md)), design reviews of the control plane and the whole system (review.02, review.03), professional responsibility (ethics.06), a mock customer engagement from discovery to handoff (field.01 to field.07), and a defense of the system (iv.01).

Why it is needed: v1.0.0 is a promise to users that the API and formats will change only through announced migrations. The migrations, the deprecation policy, and the security and design reviews are what make that promise keepable, and the engagement and the defense show the system can be explained to the people who will rely on it.
