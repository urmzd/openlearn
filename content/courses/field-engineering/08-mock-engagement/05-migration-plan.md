# Planning the staged migration

A successful evaluation still leaves a production transition to plan. The customer's existing clients and data must keep working throughout, and the customer needs a clear exit path at every step. A migration plan is the document that makes both true: it stages the change, defines what evidence allows each widening of exposure, and decides in advance what forces a rollback.

This section writes that plan for Northstar Commerce, following [Migration and Cutover](../05-migration-and-cutover/README.md) from earlier in this book. It takes the passed POC from [Pre-registering the POC](04-poc-and-evaluation.md) as its first precondition, and it leans on two versioned procedures from *Software Engineering*: the [KV format v1 to v2 migration](../../software-engineering/08-maintenance/01-kv-format-v1-to-v2.md) and its hash verification, and the [API v1 to v2 rollout](../../software-engineering/08-maintenance/02-api-v1-to-v2.md) and its compatibility window. The model answer is [`docs/engagement/migration.md`](../../../../course/solutions/docs/engagement/migration.md). Try drafting the stage table yourself before reading the walkthrough.

> **Learning objectives**
> - List the preconditions a rollout needs before its first stage.
> - Design stages whose cohorts widen only after a gate the previous stage's data can pass.
> - Write rollback triggers, a fixed rollback sequence, and resume conditions.
> - Keep mixed versions compatible during the rollout and assign approvals on both sides.

## What a migration must protect

Stage the changes, preserve compatibility during the agreed window, define rollback triggers, and assign owners on both sides, including contract, artifact, and training-data dependencies. The principles behind those rules follow from one observation: the current workflow is the safety net.

So the plan's first job is to keep that net working: a tested route back to the manual workflow or the previous API, available until the last stage.

Exposure should grow only with evidence. Each stage widens the cohort (no one, five internal users, 10%, 50%, default) and needs a gate that the previous stage's data can pass.

The first stage should have no user impact at all. Shadow traffic validates logs, dashboards, and data boundaries before anyone sees a response.

During a rollout two versions coexist, so compatibility is a precondition: the KV format migration procedure with hash verification, and the API v1 compatibility window during a v2 rollout.

A gate needs data to pass. "No errors observed" because nothing was logged is not a pass, and the plan says so explicitly.

Rollback must be decided in advance, as triggers (error budget breached, a confirmed critical safety issue, data in an unapproved place, the incident commander says stop) and a fixed sequence of actions.

## The shape of a staged rollout

The worked pattern applies to any versioned change, whether a KV format or an API version: a canary cohort, mixed-version checks, health thresholds, an expansion schedule, and a backward route to the known-good version.

## What the plan must contain

A correct migration plan covers these cases:

| Case | What must hold | Why it matters downstream |
|---|---|---|
| Preconditions | POC passed, data flow approved, rollback tested, compatibility checks passed, on-call rehearsed | the rollout does not start on unproven ground |
| Compatibility | API and KV format changes follow their versioned procedures, with the old version supported for the agreed window | mixed versions during rollout do not break clients |
| Stages with cohorts, durations, and gates | each stage widens exposure only after a named gate passes | blast radius grows only with evidence |
| Missing data fails a gate | no stage advances because nothing was measured | silence is not mistaken for health |
| Rollback triggers and actions | what forces a rollback, what the rollback does, and who can call it | a bad stage is undone in minutes |
| Resume conditions | what must be true before expansion restarts | the same failure is not rolled out twice |
| Approvals | who owns expansion and who owns data and security sign-off | every decision has an owner |

## Reading the model answer

The full plan is [`docs/engagement/migration.md`](../../../../course/solutions/docs/engagement/migration.md).

Its status is "draft pending architecture and security approval", and it names owners on both sides: the customer owner, the migration lead, and Northstar platform operations.

The preconditions must all hold before stage 0. The POC passed its approved gates and every open risk has an owner. The customer approved the production data flow, retention, access scopes, and incident contacts. Rollback to the manual workflow and a known-good serving revision is tested. API and KV compatibility checks pass: a KV format change uses "the versioned migration and hash verification procedure", and an API v2 rollout preserves "the documented v1 compatibility window". And on-call has rehearsed the rollback "without the project team".

The staged rollout is the heart of the plan:

| Stage | Cohort and duration | Expansion gate |
|---|---|---|
| 0: shadow | 1 day, no customer-visible responses | No data boundary violations; logs and dashboards validated |
| 1: internal canary | 5 support users, 2 days | Error rate below approved limit; p95 latency within target; support lead approves samples |
| 2: limited use | 10% of eligible requests, 3 days | No unresolved critical quality or safety event; capacity headroom confirmed |
| 3: broad use | 50%, 5 days | Operations accepts alert load and runbook rehearsal; sponsor signs expansion |
| 4: default route | gradual, at least 1 day per step | Customer approves each step and manual fallback remains available |

Each gate tests what the previous stage can reveal. Shadow tests plumbing; the canary tests quality with people who will report problems; 10% tests capacity; 50% tests operational load. Then the plan adds: "A gate is not passed by missing data", and every decision is recorded with times, revision, cohort, and approver.

The last part is rollback and recovery. On any trigger, route new requests to the known-good path immediately; freeze expansion; preserve logs under the approved retention; notify Maya and Priya; open an incident. "Resume only after the cause is understood, a fix is reviewed, and the prior stage's gate passes again."

## Why this shape and not another

A big-bang cutover is faster and simpler to coordinate, but it turns any defect into a full outage and leaves no data to decide with. Percent-based canaries alone (1%, 10%, 100%) test capacity but not quality; the internal canary with five support users exists because people who know the domain notice wrong answers that metrics do not. Rollback by redeploying the old version is slower than routing, so keeping the old path live and switching traffic makes rollback a configuration change.

The tradeoff accepted is time: at least eleven days from shadow to broad use, plus a gradual default route. The plan trades speed for a gate at every widening of exposure.

## What can go wrong

> **Warning:** A decision without its evidence cannot be audited or reproduced. Each stage decision is recorded with revision, cohort, times, and approver.

> **Warning:** An assumption written as a fact lets a rollout proceed on unproven ground. The preconditions require a passed POC and a tested rollback, and missing data fails a gate.

> **Warning:** Without owners and gates, work stalls or proceeds unchecked. Maya owns expansion, Priya owns data and security, and the plan stays a draft until both approve.

## Where this leads

The plan's first precondition is the passed POC from [Pre-registering the POC](04-poc-and-evaluation.md), and its compatibility rules come from the [KV format v1 to v2 migration](../../software-engineering/08-maintenance/01-kv-format-v1-to-v2.md) and the [API v1 to v2 rollout](../../software-engineering/08-maintenance/02-api-v1-to-v2.md). Next, [Answering the security questionnaire](06-commercials-and-security.md) answers questions about this data flow, and [Handing off operations](07-escalation-and-handoff.md) hands over this rollback path.

Further reading:

- The [chapter opening](README.md), to see how the evidence and open decisions travel into each later customer-facing artifact.

## Key terms

- **Shadow traffic**: real requests sent to the new path without showing its responses to users.
- **Canary**: a small cohort exposed first, chosen so problems surface early.
- **Expansion gate**: the evidence a stage must produce before the cohort widens.
- **Compatibility window**: the period during which the old version stays supported alongside the new.
- **Rollback trigger**: a pre-agreed condition that forces traffic back to the known-good path.

## Summary

- The current workflow is the safety net; keep a tested route back to it until the last stage.
- Widen exposure in stages (shadow, internal canary, 10%, 50%, default), each behind a gate.
- Missing data fails a gate; silence is not health.
- Decide rollback triggers and actions in advance, and route rather than redeploy.
- After a rollback, resume only when the prior stage's gate passes again.

## Review questions

1. What can stage 0 (shadow) find that the canary cannot, and vice versa?
   <details><summary>Answer</summary>Shadow finds plumbing problems (logging, dashboards, data going somewhere unapproved) with zero user impact. It cannot find answer-quality problems, because nobody reads the responses; the canary's support users can.</details>
2. Why must the old API version stay supported during the rollout?
   <details><summary>Answer</summary>Clients migrate at different times and rollback may route traffic back; while two versions coexist, any client must work against either. The compatibility window makes that a guarantee rather than a hope.</details>
3. Why does "a gate is not passed by missing data" need saying?
   <details><summary>Answer</summary>A gate phrased as "no critical events" passes trivially when events are not recorded. Requiring positive evidence (logs validated in stage 0, metrics present) closes that hole.</details>
4. After a rollback, why must the prior stage's gate pass again before resuming?
   <details><summary>Answer</summary>The fix is a change; it has not been through the gate. Re-passing the gate at the smaller cohort checks that the fix works before exposure widens again.</details>
