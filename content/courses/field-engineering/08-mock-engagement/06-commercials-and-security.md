# Answering the security questionnaire

Commercial and security reviews decide whether a customer can adopt the platform at all. The customer's security reviewer and procurement team will read every answer you give, and some of those answers will end up in a contract or in front of an auditor. That makes the questionnaire a different kind of document from the ones before it: each answer is a claim someone may check, so each must be traceable to a control or a current artifact.

This section answers the commercial and security questionnaire for Northstar Commerce, following [Commercials and Security](../06-commercials-and-security/README.md) from earlier in this book. The evidence comes from artifacts built elsewhere in the library: the [STRIDE threat model](../../software-engineering/09-security/01-threat-model-stride.md) and the [SBOM and supply-chain](../../software-engineering/09-security/02-sbom-and-supply-chain.md) work in *Software Engineering*, and the [usage policy at the gateway](../../ai-engineering/37-usage-policy/01-usage-policy-at-the-gateway.md) in *AI Engineering*. The data flow being reviewed is the one in [Planning the staged migration](05-migration-plan.md). The model answer is [`docs/engagement/security-questionnaire.md`](../../../../course/solutions/docs/engagement/security-questionnaire.md). Try answering the questions yourself before reading the walkthrough.

> **Learning objectives**
> - Answer each security question from a named control or artifact, or say there is none yet.
> - Keep implemented controls apart from configuration still to be done for this customer.
> - Give every open answer an owner, and mark the document's status.
> - Refuse to state a price or saving until its volume and service assumptions are agreed.

## Answers bounded by evidence

Answer only what the evidence supports. Distinguish implemented controls from planned work. State subprocessors, data flows, retention, access, incident notification, and exceptions plainly. Each of those rules follows from what a security answer is.

A security answer is a claim someone may audit, so every answer must point at a control or artifact (the threat model, an SBOM, a runbook, an access policy) or say there is none yet.

The course platform is not the customer's production deployment. Answers that describe what the platform does must not imply what Northstar's environment will do; where configuration remains, the answer says so and names who will do it.

Every gap needs an owner. The third column of the questionnaire is "Status / owner", because each answer that is not final will be closed by a specific person.

Commercial numbers are outputs of assumptions. A price or saving is only meaningful with the volume, retention, support hours, footprint, and service levels it assumes, and until those are agreed the right answer is "no estimate approved".

Finally, the document has a status. It is an internal draft until security and the commercial sponsor approve it, and it says what remains before external delivery.

## Where the answers come from

The worked approach is to use the threat model, the image SBOMs, and the usage policy to answer how prompts are protected, how dependencies are tracked, how access is scoped, and how incidents are escalated, linking each answer to its evidence and an owner.

## What the questionnaire must contain

A correct questionnaire covers these cases:

| Case | What must hold | Why it matters downstream |
|---|---|---|
| Each question answered from evidence | the answer says what exists and what it rests on | an answer the customer's auditors cannot trace is worthless |
| Implemented vs planned | controls that must still be configured or verified say so | "we will" is never read as "we do" |
| Data processed, stored, retained | data classes, location, and retention, or the statement that they are not yet approved | residency and retention are often contractual |
| Access and secrets | how access is scoped, where credentials live, who rotates and revokes | the most common audit findings |
| Dependencies | dependency allowlists and per-image SBOMs, with exceptions signed | supply-chain questions get an artifact, not an assurance |
| Incidents and subprocessors | the runbook path and the terms still to agree; subprocessors inventoried, none assumed | notification obligations and third parties are named |
| Commercial assumptions | no price or savings without stated volume and service assumptions; who owns which part | no number escapes without its inputs |
| Status and open items | a draft until approved, with a numbered list of what must happen before external use | the document is not mistaken for a certification |

## Reading the model answer

The full questionnaire is [`docs/engagement/security-questionnaire.md`](../../../../course/solutions/docs/engagement/security-questionnaire.md). Four of its rows show the patterns the rest follow:

| Question | Evidence-based answer | Status / owner |
|---|---|---|
| Where is data stored and for how long? | The POC retention period and storage location are not yet approved. No claim about production residency can be made from the current course configuration. | Priya to approve before data transfer |
| Who can access the system? | The deployment uses scoped API keys and role-based operational access. Customer-specific identities, rotation, and audit retention must be configured and verified before production. | Northstar IAM owner to supply mapping |
| How are dependencies tracked? | The course requires dependency allowlists and image SBOM evidence. The production image inventory and vulnerability review must be attached to the release record. | Alex attaches generated SBOM; security signs exceptions |
| Which subprocessors are involved? | None are assumed by this local POC. The production hosting, telemetry, and model asset sources require a current inventory. | Procurement/security to validate before contracting |

The storage row refuses to answer beyond the evidence: "no claim about production residency can be made". The access row states what is implemented (scoped API keys, role-based access) and, separately, what must still be configured and verified. The dependency row points at artifacts (allowlists, SBOMs) and names who attaches them and who signs exceptions. The subprocessor row avoids the common trap of answering "none" for a system whose production hosting is not yet decided.

The other rows follow the same pattern. Data processed: the redacted, approved sample only, with the exact fields to confirm. Secrets: in the approved secret manager, never in source, logs, fixtures, or tickets, with rotation owners to name. Incidents: the runbook covers detection, containment, rollback, evidence, and escalation, with notification timing and contacts to agree in the contract.

The commercial assumptions section opens with "No price or savings estimate is approved." Any proposal must state request volume, retained data, support hours, deployment footprint, and service level assumptions. The customer owns the procurement terms and the service level; the engineering team owns only the technical estimates and their evidence.

The questionnaire ends with numbered open items before external delivery: confirm data classes, geography, and retention; attach the production threat model, per-image SBOM, dependency exceptions, and access review; confirm escalation contacts and notification terms; and have procurement validate the hosting and subprocessor statements. The closing line is: "Until those approvals, this document is an internal draft, not a security certification."

## Why this shape and not another

Answering every question "yes" is what a questionnaire seems to invite, and it is what gets a vendor removed at audit; answers bounded by evidence survive scrutiny. Leaving unanswerable questions blank looks like evasion, whereas stating what is not yet known, and who will answer it, shows a process. Including a price to move the deal forward would commit the team to numbers whose inputs (volume, retention, service level) nobody has agreed, and which belong to the customer's procurement.

The tradeoff accepted is that the questionnaire is less reassuring to read than a list of confident answers. It is far more useful to the customer's security reviewer.

## What can go wrong

> **Warning:** An answer without its evidence cannot be audited. Each answer here names its control or artifact: an SBOM, a runbook, the access model.

> **Warning:** An assumption written as a fact lets the engagement promise security the system has not demonstrated. Implemented controls and remaining configuration are stated separately, and there is no residency or price claim without evidence.

> **Warning:** Without owners and a decision gate, open answers never close. The questionnaire has a "Status / owner" column, numbered open items, and named approvers for the security and commercial statements.

## Where this leads

This review answers questions about the data flow and rollback from [Planning the staged migration](05-migration-plan.md), drawing on the [STRIDE threat model](../../software-engineering/09-security/01-threat-model-stride.md) for the data and access answers, the [SBOM and supply-chain](../../software-engineering/09-security/02-sbom-and-supply-chain.md) work for the dependency answer, and the [usage policy at the gateway](../../ai-engineering/37-usage-policy/01-usage-policy-at-the-gateway.md) for the data-handling answers. Next, [Handing off operations](07-escalation-and-handoff.md) relies on its incident and access answers.

Further reading:

- The [chapter opening](README.md), to see how the evidence and open decisions travel into each later customer-facing artifact.

## Key terms

- **Evidence-based answer**: an answer that names the control or artifact it rests on, or states that none exists yet.
- **Implemented vs planned**: the distinction between a control in place and configuration still to be done.
- **SBOM**: a software bill of materials, the inventory of components in an image.
- **Subprocessor**: a third party that processes customer data on the vendor's behalf.
- **Data residency**: where data is stored and processed, often a contractual requirement.

## Summary

- Every security answer points at a control or artifact, or says there is none yet.
- Keep what the platform does apart from what the customer's production deployment will do.
- Give every open answer an owner, and list the open items before external delivery.
- No price or saving without agreed volume, retention, footprint, and service assumptions.
- The questionnaire is an internal draft, not a certification, until it is approved.

## Review questions

1. Why does the storage answer say no claim about production residency can be made?
   <details><summary>Answer</summary>The only evidence is the course configuration, which is not Northstar's production deployment, and the POC location is not yet approved. Any residency claim would be unsupported and could become a contractual misstatement.</details>
2. What distinguishes the two sentences of the access answer?
   <details><summary>Answer</summary>The first describes an implemented control (scoped keys, role-based access); the second lists work still required for this customer (identities, rotation, audit retention). Keeping them apart is the "implemented vs planned" rule.</details>
3. Why is "none" the wrong answer to the subprocessor question even though the local POC uses none?
   <details><summary>Answer</summary>The question is about the system the customer will adopt. Production hosting, telemetry, and model sources may all involve third parties; the right answer is that none are assumed by the POC and production needs an inventory.</details>
4. Why does the document call itself "not a security certification"?
   <details><summary>Answer</summary>It is a draft prepared by the engineering team. Only the customer's security reviewer can approve its technical statements, and only after the open items are closed.</details>
