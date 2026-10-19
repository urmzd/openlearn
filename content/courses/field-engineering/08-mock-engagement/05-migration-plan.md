# Mock engagement: migration plan

## Overview

| | |
|---|---|
| **Module** | `field.05` · practice · docs · Pass 11 · about 30 min reading |
| **Solution** | model answer: [`docs/engagement/migration.md`](../../../../course/solutions/docs/engagement/migration.md) (the staged production migration plan for Northstar Commerce) |
| **Needs** | none |
| **Used by** | no code call site; the security review (`field.06`) and the handoff (`field.07`) assume its rollback and owners |

## Key Takeaways

- Tie each claim to the exact evidence behind it: a passed gate, a rehearsed rollback, an approval.
- State the owner, success condition, and recovery action before declaring the work complete.
- Record unresolved assumptions so the next review can test them.

## How to read this chapter

Sections 1 to 3 set out what a migration must protect and how a staged rollout works. Section 4 lists what a correct migration plan must contain. Section 5 walks through the model answer for Northstar Commerce and explains why each stage and trigger is there. Try to draft the stage table yourself before reading section 5.

---

## 1. Why now

A successful evaluation still leaves a production transition to plan. The migration must protect existing clients and data while giving the customer a clear exit path.

## 2. Principles

Stage changes, preserve compatibility during the agreed window, define rollback triggers, and assign owners on both sides. Include contract, artifact, and training-data dependencies.

## 3. Worked example

Plan a KV or API version rollout with a canary cohort, mixed-version checks, health thresholds, expansion schedule, and a backward route to the known-good version.

## 4. The artifact

### What a correct solution must handle

| Case | What must hold | Why it matters downstream |
|---|---|---|
| Preconditions | POC passed, data flow approved, rollback tested, compatibility checks passed, on-call rehearsed | the rollout does not start on unproven ground |
| Compatibility | API and KV format changes follow their versioned procedures, with the old version supported for the agreed window | mixed versions during rollout do not break clients |
| Stages with cohorts, durations, and gates | each stage widens exposure only after a named gate passes | blast radius grows only with evidence |
| Missing data fails a gate | no stage advances because nothing was measured | silence is not mistaken for health |
| Rollback triggers and actions | what forces a rollback, what the rollback does, and who can call it | a bad stage is undone in minutes |
| Resume conditions | what must be true before expansion restarts | the same failure is not rolled out twice |
| Approvals | who owns expansion and who owns data and security sign-off | every decision has an owner |

## 5. The solution, step by step

### 5.1 From first principles

1. **The current workflow is the safety net**, so the plan's first job is to keep it working: a tested route back to the manual workflow or previous API, available until the last stage.
2. **Exposure should grow only with evidence.** Each stage widens the cohort (no one, five internal users, 10%, 50%, default) and needs a gate that the previous stage's data can pass.
3. **The first stage should have no user impact at all.** Shadow traffic validates logs, dashboards, and data boundaries before anyone sees a response.
4. **During a rollout two versions coexist**, so compatibility is a precondition: the KV format migration procedure with hash verification, and the API v1 compatibility window during a v2 rollout.
5. **A gate needs data to pass.** "No errors observed" because nothing was logged is not a pass; the plan says so explicitly.
6. **Rollback must be decided in advance**, as triggers (error budget breached, confirmed critical safety issue, data in an unapproved place, the incident commander says stop) and a fixed sequence of actions.

### 5.2 The model answer, annotated

The full plan is [`docs/engagement/migration.md`](../../../../course/solutions/docs/engagement/migration.md).

**Status: "draft pending architecture and security approval".** Owners on both sides: the customer owner, the migration lead, and Northstar platform operations.

**Preconditions** (all must hold before stage 0): the POC passed its approved gates and every open risk has an owner; the customer approved the production data flow, retention, access scopes, and incident contacts; rollback to the manual workflow and a known-good serving revision is tested; API and KV compatibility checks pass, a KV format change using "the versioned migration and hash verification procedure" and an API v2 rollout preserving "the documented v1 compatibility window"; and on-call has rehearsed the rollback "without the project team".

**Staged rollout.**

| Stage | Cohort and duration | Expansion gate |
|---|---|---|
| 0: shadow | 1 day, no customer-visible responses | No data boundary violations; logs and dashboards validated |
| 1: internal canary | 5 support users, 2 days | Error rate below approved limit; p95 latency within target; support lead approves samples |
| 2: limited use | 10% of eligible requests, 3 days | No unresolved critical quality or safety event; capacity headroom confirmed |
| 3: broad use | 50%, 5 days | Operations accepts alert load and runbook rehearsal; sponsor signs expansion |
| 4: default route | gradual, at least 1 day per step | Customer approves each step and manual fallback remains available |

Each gate tests what the previous stage can reveal: shadow tests plumbing, the canary tests quality with people who will report problems, 10% tests capacity, 50% tests operational load. Then: "A gate is not passed by missing data", and every decision is recorded with times, revision, cohort, and approver.

**Rollback and recovery.** Route new requests to the known-good path immediately on any trigger; freeze expansion; preserve logs under the approved retention; notify Maya and Priya; open an incident. "Resume only after the cause is understood, a fix is reviewed, and the prior stage's gate passes again."

### 5.3 Why this answer

- **A big-bang cutover** is faster and simpler to coordinate, but turns any defect into a full outage and leaves no data to decide with.
- **Percent-based canaries alone** (1%, 10%, 100%) test capacity but not quality: the internal canary with five support users exists because people who know the domain notice wrong answers that metrics do not.
- **Rollback by redeploying the old version** is slower than routing; keeping the old path live and switching traffic makes rollback a configuration change.
- **Tradeoff accepted.** At least eleven days from shadow to broad use, plus a gradual default route. The plan trades speed for a gate at every widening of exposure.

### 5.4 Explain it back

1. What can stage 0 (shadow) find that the canary cannot, and vice versa?
   <details><summary>Answer</summary>Shadow finds plumbing problems (logging, dashboards, data going somewhere unapproved) with zero user impact. It cannot find answer-quality problems, because nobody reads the responses; the canary's support users can.</details>
2. Why must the old API version stay supported during the rollout?
   <details><summary>Answer</summary>Clients migrate at different times and rollback may route traffic back; while two versions coexist, any client must work against either. The compatibility window makes that a guarantee rather than a hope.</details>
3. Why does "a gate is not passed by missing data" need saying?
   <details><summary>Answer</summary>A gate phrased as "no critical events" passes trivially when events are not recorded. Requiring positive evidence (logs validated in stage 0, metrics present) closes that hole.</details>
4. After a rollback, why must the prior stage's gate pass again before resuming?
   <details><summary>Answer</summary>The fix is a change; it has not been through the gate. Re-passing the gate at the smaller cohort checks that the fix works before exposure widens again.</details>

## 6. Pitfalls

| # | Pitfall | Symptom | How the solution avoids it |
|---|---|---|---|
| 1 | The deliverable records a conclusion without its customer evidence | Another engineer cannot reproduce the sizing or recommendation. | each stage decision is recorded with revision, cohort, times, and approver |
| 2 | An assumption is written as a measured fact | The engagement plan can promise capacity or security the system has not demonstrated. | preconditions require a passed POC and a tested rollback; missing data fails a gate |
| 3 | The next owner and decision gate are missing | Work stalls at handoff even when the technical work is complete. | Maya owns expansion, Priya owns data and security; the plan stays a draft until both approve |

## 7. Where it's used next

| Direction | Module | Connection |
|---|---|---|
| Back | `field.04` | the passed POC that is the first precondition |
| Back | `craft.13` | the KV format v1 to v2 migration and its hash verification |
| Back | `craft.14` | the API v1 to v2 rollout and its compatibility window |
| Forward | `field.06` | the security review answers questions about this data flow |
| Forward | `field.07` | the handoff hands over this rollback path |

## Going further

| Resource | What to inspect |
|---|---|
| [Mock engagement sequence](README.md) | Carry the evidence and open decisions into the next customer-facing artifact in the sequence. |
