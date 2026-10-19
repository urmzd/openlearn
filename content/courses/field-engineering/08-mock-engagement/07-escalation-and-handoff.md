# Handing off operations

A handoff is complete when another team can own normal operation and escalation without relying on the implementation team's memory. Until then, every alert that only the project team can interpret is a single point of failure, and the engagement has not really ended. The handoff document is where that knowledge transfers, and the rehearsal is where you find out whether it did.

This is the last artifact of the Northstar Commerce engagement. It follows [Escalation and Handoff](../07-escalation-and-handoff/README.md) from earlier in this book, puts into practice the incident and access answers from [Answering the security questionnaire](06-commercials-and-security.md), and links the runbooks described in [Runbooks and Diataxis docs](../../diagramming/02-documentation-writing/02-runbooks-and-diataxis.md) from *Diagramming and Documentation*. The model answer is [`docs/engagement/handoff.md`](../../../../course/solutions/docs/engagement/handoff.md). Try writing the escalation sequence yourself before reading the walkthrough.

> **Learning objectives**
> - Define the service boundary and assign each kind of operational work to an owner.
> - Write a numbered escalation sequence that restores service before diagnosing it.
> - State what data may and may not be preserved during an incident.
> - Test the handoff with a rehearsal and state its completion condition.

## When ownership really transfers

Transfer the service boundaries, dashboards, runbooks, access process, known limitations, escalation contacts, and recovery authority, and have the receiving team demonstrate the critical procedure. The reasoning behind that list starts from one fact: ownership transfers only if knowledge does. Anything the project team knows that is not written down (a dashboard's meaning, a credential, a contact) is a single point of failure after handoff.

Boundaries come first. The document names the components (gateway, engine, tokenizer and model assets, durable workflow services) and assigns each kind of work: Northstar operations owns cluster health, access, paging, and fallback; the engineering team owns defect triage during the support window.

Responders need a starting point. Normal operation lists the panels (gateway requests, errors, and latency; engine queue and capacity; workflow backlog; current revision) and the runbooks (engine crash loops, TTFT budget burn, durable redeliveries, dead letters).

An incident is a decision sequence, so escalation is a numbered list: record, route to fallback and declare, assign customer updates and security leads, preserve the right evidence, and resume only on verified recovery.

And a written procedure is a hypothesis until someone else runs it. The rehearsal is the test, with time, participants, gaps, and owners recorded.

## Rehearsing an alert

The worked exercise is a simulated availability alert. Walk the customer through it and ask them to locate the impact, follow the runbook, decide when to escalate, and verify recovery. Capture every unclear step and revise the docs.

## What the handoff must contain

A correct handoff document covers these cases:

| Case | What must hold | Why it matters downstream |
|---|---|---|
| The service boundary | which components are in scope, and who owns what (operations, defect triage, fallback) | no alert lands on nobody |
| Normal operation | which panels to check and which runbooks apply | responders start from the same place |
| Escalation sequence | numbered steps from alert to recovery, with who decides what | the incident has a commander and a path |
| Data handling during incidents | what may be preserved and what may not be replayed | an incident does not become a data exposure |
| Recovery criteria | what must be verified, and by whom, before the incident closes | "it looks fine" is not recovery |
| A rehearsal | the receiving team demonstrates the critical procedure; gaps are recorded with owners | the handoff is tested, not asserted |
| Completion condition | the handoff is incomplete while any critical step needs undocumented knowledge or a private credential | ownership really transfers |

## Reading the model answer

The full handoff is [`docs/engagement/handoff.md`](../../../../course/solutions/docs/engagement/handoff.md).

Its header names the customer service owner (Northstar platform operations), the escalation owner (Maya Chen), the engineering contact (Alex Rivera), and the state: "rehearsal required before production cutover".

The service boundary and normal operation follow: the components and the split of responsibility described above, with the note that "these boundaries must be reflected in the final service inventory and customer contract"; the panels and runbooks to use; and one data rule: "Do not replay customer prompts into an unapproved environment."

Escalation and recovery is five steps:

1. The on-call responder records the alert, affected cohort, start time, and current revision.
2. If responses are incorrect, unsafe, or unavailable, route to the agreed fallback and declare an incident.
3. The incident commander owns customer updates; Maya coordinates product and security decisions; Alex supports diagnosis; Priya leads any suspected data exposure.
4. Preserve approved logs and trace identifiers, "not raw prompt contents unless the retention policy explicitly permits them".
5. Resume only after the rollback or repair is verified, the customer owner accepts the recovery evidence, and the incident commander records the decision.

Step 1 records the revision first because it is what the [migration plan's](05-migration-plan.md) rollback needs. Step 2 makes fallback the first action, not diagnosis. Step 4 keeps the incident from creating a second data problem.

The readiness rehearsal comes last. Operations must show they can find the dashboards, identify the serving revision, execute the fallback, page the right owner, and verify recovery, and the rehearsal's date, participants, elapsed time, gaps, and owners are recorded. Then the completion condition:

> The handoff is incomplete while any critical step depends on undocumented knowledge or a private credential held by the project team.

The next action schedules a 45-minute rehearsal. Alex prepares a clean runbook bundle without unapproved contact details, and Maya signs operational readiness once the gaps are closed.

## Why this shape and not another

A handoff meeting with slides transfers awareness, not capability; the rehearsal is what shows the receiving team can act alone. Diagnosing before falling back feels efficient to engineers and extends customer impact, so the sequence restores service first. Keeping the project team on the pager "for a while" postpones the handoff indefinitely, and the completion condition makes the remaining dependencies explicit so they can be removed.

The tradeoff accepted is that cutover waits for a rehearsal and for every critical gap to close, which may delay production by days.

## What can go wrong

> **Warning:** A claim of readiness without evidence cannot be checked. The rehearsal is recorded with its date, participants, elapsed time, and gaps.

> **Warning:** An assumption written as a fact, here "operations can handle it", is not readiness. Readiness is demonstrated by the receiving team, not asserted by the project team.

> **Warning:** Without named owners, work stalls at exactly this point. The handoff names service, escalation, and acceptance owners, and stays incomplete while any critical step depends on the project team.

## Where this leads

The handoff puts into practice the incident and access answers from [Answering the security questionnaire](06-commercials-and-security.md) and links the runbooks of [Runbooks and Diataxis docs](../../diagramming/02-documentation-writing/02-runbooks-and-diataxis.md). It closes the mock engagement. The [capstone](08-capstone.md) then applies the whole sequence, from discovery to escalation, to a single new customer.

Further reading:

- The [chapter opening](README.md), to see the whole engagement sequence in one place.

## Key terms

- **Service boundary**: the components in scope and the owner of each kind of work on them.
- **Fallback**: the known-good path (the manual workflow or the previous route) that restores service during an incident.
- **Incident commander**: the person who owns customer updates and records decisions during an incident.
- **Readiness rehearsal**: a run of the critical procedure by the receiving team, recorded with its gaps.
- **Completion condition**: the rule that the handoff is incomplete while any critical step depends on the project team.

## Summary

- Ownership transfers only when knowledge does; anything unwritten is a single point of failure.
- Name the service boundary and the owner of each kind of work.
- In an incident, record the revision, fall back first, then diagnose.
- Preserve approved logs and trace identifiers, not raw prompts.
- Test the handoff with a rehearsal; it is incomplete while any critical step needs the project team.

## Review questions

1. Why is routing to the fallback step 2, before diagnosis?
   <details><summary>Answer</summary>Customer impact continues for as long as diagnosis takes. The fallback (the manual workflow or the previous path) is known to work, so restoring service first and diagnosing afterwards minimizes harm.</details>
2. Why must raw prompts not be preserved by default during an incident?
   <details><summary>Answer</summary>Prompts may contain customer data; copying them into incident records creates a new store outside the approved retention policy. Trace identifiers and approved logs are enough to diagnose.</details>
3. What does "a private credential held by the project team" fail, and why does it block the handoff?
   <details><summary>Answer</summary>It means a critical step cannot be performed by the receiving team. When the project team is unavailable, the step cannot happen at all, so ownership has not actually transferred.</details>
4. What is the rehearsal testing that the document alone cannot?
   <details><summary>Answer</summary>Whether someone other than the author can follow it: find the panels, identify the revision, execute the fallback, page the right person, and verify recovery. Every place they hesitate is a gap in the document.</details>
