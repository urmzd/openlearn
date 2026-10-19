# Pass 11: Operating the System

A system that works is not yet a system that can be operated. This pass releases the system as v1.0.0, a promise that its API and formats change only through announced migrations, and then does the work that makes the promise keepable: a deprecation policy, migrations of the KV format and the API built as modules and rehearsed as drills, dependency upgrades, a bisected performance regression, and drills for data incidents, noisy neighbours, runaway agents, and a full event-log disk. Each drill leaves a runbook and, where an incident happened, a blameless postmortem.

The pass also reviews the system from outside: a STRIDE threat model, an SBOM and supply-chain review, a secrets and authorization review, design reviews of the control plane and the whole system, and a mock customer engagement from discovery to handoff. It ends with a defense of the system, where you explain every decision in it.

> **Learning objectives**
> - Document the system with C4 diagrams, runbooks, and Diataxis docs.
> - Release v1.0.0 with semantic versioning and a deprecation policy, and migrate formats without downtime.
> - Detect, resolve, and document each core incident drill.
> - Review the system's security and design, and explain it to the people who will rely on it.

## Before you begin

This pass needs [Pass 10](../course-p10-agents/README.md), and with it the whole system the earlier passes built.

## In this pass

1. [C4 diagrams in D2](../../../diagramming/01-diagramming-c4/01-c4-diagrams-in-d2.md). Afterwards you can explain how the C4 diagrams in D2 describe the system at each level.
2. [Runbooks and Diataxis docs](../../../diagramming/02-documentation-writing/02-runbooks-and-diataxis.md). Afterwards you can explain how a runbook is structured and how Diataxis sorts documentation.
3. [Commits, semver, and releases](../../../software-craftsmanship/06-releases-and-deprecation/01-commits-semver-and-releases.md). Afterwards you can explain how conventional commits drive semver and how v1.0.0 is released.
4. [Deprecation policy](../../../software-craftsmanship/06-releases-and-deprecation/02-deprecation-policy.md). Afterwards you can explain the deprecation policy and how it protects users.
5. [Interface migration: KV format v1 to v2](../../../software-craftsmanship/07-maintenance/01-kv-format-v1-to-v2.md). Afterwards you can explain how the KV format migrates from v1 to v2 without downtime.
6. [The KV format v2 migration](../../../systems/05-incident-response-and-chaos/05-kv-v2-migration.md). Afterwards you can diagnose a failure during the KV v2 migration, explain the fix, and explain how it is prevented.
7. [Interface migration: API v1 to v2](../../../software-craftsmanship/07-maintenance/02-api-v1-to-v2.md). Afterwards you can explain how the API migrates from v1 to v2 while v1 clients keep working.
8. [The API v2 migration](../../../systems/05-incident-response-and-chaos/06-api-v2-migration.md). Afterwards you can diagnose a failure during the API v2 migration, explain the fix, and explain how it is prevented.
9. [Dependency upgrades](../../../software-craftsmanship/07-maintenance/03-dependency-upgrades.md). Afterwards you can explain how a dependency upgrade is planned, tested, and rolled out.
10. [A breaking dependency upgrade](../../../systems/05-incident-response-and-chaos/07-dependency-upgrade.md). Afterwards you can diagnose a breaking dependency upgrade, explain the fix, and explain how it is prevented.
11. [Performance regression bisect](../../../software-craftsmanship/07-maintenance/04-performance-regression-bisect.md). Afterwards you can explain how a performance regression is bisected to one commit.
12. [A performance regression bisect](../../../systems/05-incident-response-and-chaos/08-performance-regression.md). Afterwards you can bisect a performance regression from its symptoms, explain the fix, and explain how it is prevented.
13. [A data incident](../../../systems/05-incident-response-and-chaos/09-data-incident.md). Afterwards you can diagnose the data incident, explain the fix, and explain how it is prevented.
14. [A noisy neighbor](../../../systems/05-incident-response-and-chaos/10-noisy-neighbor.md). Afterwards you can diagnose the noisy neighbor, explain the fix, and explain how it is prevented.
15. [A runaway agent](../../../systems/05-incident-response-and-chaos/11-runaway-agent.md). Afterwards you can diagnose the runaway agent, explain the fix, and explain how it is prevented.
16. [The event log disk fills up](../../../systems/05-incident-response-and-chaos/12-eventlog-disk-full.md). Afterwards you can diagnose the full event-log disk, explain the fix, and explain how it is prevented.
17. [Threat modeling with STRIDE](../../../software-craftsmanship/08-security/01-threat-model-stride.md). Afterwards you can explain the STRIDE threat model of the system's data-flow diagram.
18. [SBOM and supply chain](../../../software-craftsmanship/08-security/02-sbom-and-supply-chain.md). Afterwards you can explain how the SBOM is generated and what supply-chain checks it enables.
19. [Secrets and authorization review](../../../software-craftsmanship/08-security/03-secrets-and-authorization-review.md). Afterwards you can explain how secrets are handled and how authorization is reviewed.
20. [Design review: the control plane](../../../systems/01-system-design/02-design-review-control-plane.md). Afterwards you can explain the design decisions of the control plane and the tradeoffs a design review raises.
21. [Design review: the whole system](../../../systems/01-system-design/03-design-review-whole-system.md). Afterwards you can explain the design of the whole system and defend its tradeoffs in a review.
22. [Professional responsibility and disclosure](../../../responsible-ai/06-professional-responsibility/01-professional-responsibility.md). Afterwards you can explain the professional responsibilities and disclosure duties that apply to the system.
23. [Writing the discovery record](../../../field-engineering/08-mock-engagement/01-discovery.md). Afterwards you can explain how discovery uncovers the customer's needs and constraints.
24. [Qualifying and sizing the engagement](../../../field-engineering/08-mock-engagement/02-qualification-and-sizing.md). Afterwards you can explain how the opportunity is qualified and the deployment sized.
25. [Designing the performance test protocol](../../../field-engineering/08-mock-engagement/03-performance-engagement.md). Afterwards you can explain how the performance engagement measures and improves the customer's workload.
26. [Pre-registering the POC](../../../field-engineering/08-mock-engagement/04-poc-and-evaluation.md). Afterwards you can explain how the POC is scoped and evaluated against success criteria.
27. [Planning the staged migration](../../../field-engineering/08-mock-engagement/05-migration-plan.md). Afterwards you can explain the migration plan and its risks.
28. [Answering the security questionnaire](../../../field-engineering/08-mock-engagement/06-commercials-and-security.md). Afterwards you can explain the commercials and how the security review is answered.
29. [Handing off operations](../../../field-engineering/08-mock-engagement/07-escalation-and-handoff.md). Afterwards you can explain how escalations are handled and how the engagement is handed off.
30. [Defend your system](../../../interview-preparation/06-defend-your-system/01-defend-your-system.md). Afterwards you can defend the system's design and tradeoffs under questioning.
31. [What the System Does After Pass 11](milestone.md). Afterwards you can explain how v1.0.0 is released, migrated, operated through its incident drills, reviewed, and handed off.

## Going deeper

One optional drill exercises the Raft-replicated durable log from [Pass 8](../course-p08-durable/README.md):

32. [A Raft partition (optional)](../../../systems/05-incident-response-and-chaos/13-raft-partition.md). Afterwards you can diagnose a Raft partition, explain what the cluster guarantees during it, and explain the fix.

The stages are also in this pass's manifest, [`path.tsv`](path.tsv). The core course ends here at v1.0.0; [Pass 12](../course-p12-multimodal/README.md) extends the system to images and audio.
