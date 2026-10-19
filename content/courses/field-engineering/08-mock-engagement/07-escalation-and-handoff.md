# Escalation and handoff

## Overview

| | |
|---|---|
| **Module** | `field.07` · practice · docs · Pass 11 · about 30 min reading |
| **Solution** | model answer: [`docs/engagement/handoff.md`](../../../../course/solutions/docs/engagement/handoff.md) (the operational handoff for Northstar Commerce) |
| **Needs** | none |
| **Used by** | no code call site; this is the last artifact of the mock engagement, and the [capstone](capstone.md) applies the whole sequence to one customer |

## Key Takeaways

- Tie each responsibility to the exact runbook, dashboard, or owner that carries it.
- State the owner, success condition, and recovery action before declaring the work complete.
- Record unresolved assumptions so the next review can test them.

## How to read this chapter

Sections 1 to 3 set out when a handoff is complete and how to test it with a rehearsal. Section 4 lists what a correct handoff document must contain. Section 5 walks through the model answer for Northstar Commerce and explains why each part is there. Try writing the escalation sequence yourself before reading section 5.

---

## 1. Why now

A handoff is complete when another team can own normal operation and escalation without relying on the implementation team's memory.

## 2. Principles

Transfer service boundaries, dashboards, runbooks, access process, known limitations, escalation contacts, and recovery authority. Have the receiving team demonstrate the critical procedure.

## 3. Worked example

Walk the customer through a simulated availability alert. Ask them to locate impact, follow the runbook, decide when to escalate, and verify recovery. Capture unclear steps and revise the docs.

## 4. The artifact

### What a correct solution must handle

| Case | What must hold | Why it matters downstream |
|---|---|---|
| The service boundary | which components are in scope, and who owns what (operations, defect triage, fallback) | no alert lands on nobody |
| Normal operation | which panels to check and which runbooks apply | responders start from the same place |
| Escalation sequence | numbered steps from alert to recovery, with who decides what | the incident has a commander and a path |
| Data handling during incidents | what may be preserved and what may not be replayed | an incident does not become a data exposure |
| Recovery criteria | what must be verified, and by whom, before the incident closes | "it looks fine" is not recovery |
| A rehearsal | the receiving team demonstrates the critical procedure; gaps are recorded with owners | the handoff is tested, not asserted |
| Completion condition | the handoff is incomplete while any critical step needs undocumented knowledge or a private credential | ownership really transfers |

## 5. The solution, step by step

### 5.1 From first principles

1. **Ownership transfers only if knowledge does.** Anything the project team knows that is not written down (a dashboard's meaning, a credential, a contact) is a single point of failure after handoff.
2. **Boundaries come first.** The document names the components (gateway, engine, tokenizer and model assets, durable workflow services) and assigns each kind of work: Northstar operations owns cluster health, access, paging, and fallback; the engineering team owns defect triage during the support window.
3. **Responders need a starting point.** Normal operation lists the panels (gateway requests, errors, latency; engine queue and capacity; workflow backlog; current revision) and the runbooks (engine crash loops, TTFT budget burn, durable redeliveries, dead letters).
4. **An incident is a decision sequence**, so escalation is a numbered list: record, route to fallback and declare, assign customer updates and security leads, preserve the right evidence, resume only on verified recovery.
5. **A written procedure is a hypothesis until someone else runs it.** The rehearsal is the test, with time, participants, gaps, and owners recorded.

### 5.2 The model answer, annotated

The full handoff is [`docs/engagement/handoff.md`](../../../../course/solutions/docs/engagement/handoff.md).

**Header**: customer service owner (Northstar platform operations), escalation owner (Maya Chen), engineering contact (Alex Rivera), and the state: "rehearsal required before production cutover".

**Service boundary and normal operation.** The components and the split of responsibility above, with the note that "these boundaries must be reflected in the final service inventory and customer contract"; the panels and runbooks to use; and one data rule: "Do not replay customer prompts into an unapproved environment."

**Escalation and recovery**, five steps:

1. the on-call responder records the alert, affected cohort, start time, and current revision;
2. if responses are incorrect, unsafe, or unavailable, route to the agreed fallback and declare an incident;
3. the incident commander owns customer updates; Maya coordinates product and security decisions; Alex supports diagnosis; Priya leads any suspected data exposure;
4. preserve approved logs and trace identifiers, "not raw prompt contents unless the retention policy explicitly permits them";
5. resume only after the rollback or repair is verified, the customer owner accepts the recovery evidence, and the incident commander records the decision.

Step 1 records the revision first because it is what the migration plan's rollback needs; step 2 makes fallback the first action, not diagnosis; step 4 keeps the incident from creating a second data problem.

**Readiness rehearsal.** Operations must show they can find the dashboards, identify the serving revision, execute the fallback, page the right owner, and verify recovery; the rehearsal's date, participants, elapsed time, gaps, and owners are recorded. Then the completion condition:

> The handoff is incomplete while any critical step depends on undocumented knowledge or a private credential held by the project team.

The next action schedules a 45-minute rehearsal; Alex prepares a clean runbook bundle without unapproved contact details; Maya signs operational readiness once gaps are closed.

### 5.3 Why this answer

- **A handoff meeting with slides** transfers awareness, not capability; the rehearsal is what shows the receiving team can act alone.
- **Diagnosing before falling back** feels efficient to engineers and extends customer impact; the sequence restores service first.
- **Keeping the project team on the pager "for a while"** postpones the handoff indefinitely; the completion condition makes the remaining dependencies explicit so they can be removed.
- **Tradeoff accepted.** The cutover waits for a rehearsal and for every critical gap to close, which may delay production by days.

### 5.4 Explain it back

1. Why is routing to the fallback step 2, before diagnosis?
   <details><summary>Answer</summary>Customer impact continues for as long as diagnosis takes. The fallback (the manual workflow or the previous path) is known to work, so restoring service first and diagnosing afterwards minimizes harm.</details>
2. Why must raw prompts not be preserved by default during an incident?
   <details><summary>Answer</summary>Prompts may contain customer data; copying them into incident records creates a new store outside the approved retention policy. Trace identifiers and approved logs are enough to diagnose.</details>
3. What does "a private credential held by the project team" fail, and why does it block the handoff?
   <details><summary>Answer</summary>It means a critical step cannot be performed by the receiving team. When the project team is unavailable, the step cannot happen at all, so ownership has not actually transferred.</details>
4. What is the rehearsal testing that the document alone cannot?
   <details><summary>Answer</summary>Whether someone other than the author can follow it: find the panels, identify the revision, execute the fallback, page the right person, and verify recovery. Every place they hesitate is a gap in the document.</details>

## 6. Pitfalls

| # | Pitfall | Symptom | How the solution avoids it |
|---|---|---|---|
| 1 | The deliverable records a conclusion without its customer evidence | Another engineer cannot reproduce the sizing or recommendation. | the rehearsal is recorded with date, participants, elapsed time, and gaps |
| 2 | An assumption is written as a measured fact | The engagement plan can promise capacity or security the system has not demonstrated. | readiness is demonstrated by the receiving team, not asserted by the project team |
| 3 | The next owner and decision gate are missing | Work stalls at handoff even when the technical work is complete. | named service, escalation, and acceptance owners; incomplete while any critical step depends on the project team |

## 7. Where it's used next

| Direction | Module | Connection |
|---|---|---|
| Back | `field.06` | the incident and access answers this handoff puts into practice |
| Back | `craft.10` | the runbooks the escalation sequence links |

## Going further

| Resource | What to inspect |
|---|---|
| [Mock engagement sequence](README.md) | Carry the evidence and open decisions into the next customer-facing artifact in the sequence. |
