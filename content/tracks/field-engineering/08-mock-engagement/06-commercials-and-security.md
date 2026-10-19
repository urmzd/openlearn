# Commercials and security review

## Overview

| | |
|---|---|
| **Module** | `field.06` · practice · docs · Pass 11 · about 30 min reading |
| **Solution** | model answer: [`docs/engagement/security-questionnaire.md`](../../../../course/solutions/docs/engagement/security-questionnaire.md) (the commercial and security questionnaire for Northstar Commerce) |
| **Needs** | none |
| **Used by** | no code call site; the handoff (`field.07`) relies on its incident and access answers |

## Key Takeaways

- Tie each answer to the exact control or artifact behind it, and say when there is none yet.
- State the owner, success condition, and recovery action before declaring the work complete.
- Record unresolved assumptions so the next review can test them.

## How to read this chapter

Sections 1 to 3 set out why security and commercial answers must be traceable and which artifacts they come from. Section 4 lists what a correct questionnaire must contain. Section 5 walks through the model answer for Northstar Commerce and explains how each answer was phrased and why. Try answering the questions yourself before reading section 5.

---

## 1. Why now

Commercial and security reviews shape whether a customer can adopt the platform. Answers should be traceable to controls and current artifacts.

## 2. Principles

Answer only what evidence supports. Distinguish implemented controls from planned work. State subprocessors, data flows, retention, access, incident notification, and exceptions plainly.

## 3. Worked example

Use the threat model, image SBOMs, and usage policy to answer how prompts are protected, dependencies tracked, access scoped, and incidents escalated. Link each answer to evidence and owner.

## 4. The artifact

### What a correct solution must handle

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

## 5. The solution, step by step

### 5.1 From first principles

1. **A security answer is a claim someone may audit**, so every answer must point at a control or artifact (the threat model, an SBOM, a runbook, an access policy) or say there is none yet.
2. **The course platform is not the customer's production deployment.** Answers that describe what the platform does must not imply what Northstar's environment will do; where configuration remains, the answer says so and names who will do it.
3. **Every gap needs an owner.** The third column of the questionnaire is "Status / owner", because each answer that is not final will be closed by a specific person.
4. **Commercial numbers are outputs of assumptions.** A price or saving is only meaningful with the volume, retention, support hours, footprint, and service levels it assumes; until those are agreed, the right answer is "no estimate approved".
5. **The document has a status.** It is an internal draft until security and the commercial sponsor approve it, and it says what remains before external delivery.

### 5.2 The model answer, annotated

The full questionnaire is [`docs/engagement/security-questionnaire.md`](../../../../course/solutions/docs/engagement/security-questionnaire.md). Selected rows, and the pattern each shows:

| Question | Evidence-based answer | Status / owner |
|---|---|---|
| Where is data stored and for how long? | The POC retention period and storage location are not yet approved. No claim about production residency can be made from the current course configuration. | Priya to approve before data transfer |
| Who can access the system? | The deployment uses scoped API keys and role-based operational access. Customer-specific identities, rotation, and audit retention must be configured and verified before production. | Northstar IAM owner to supply mapping |
| How are dependencies tracked? | The course requires dependency allowlists and image SBOM evidence. The production image inventory and vulnerability review must be attached to the release record. | Alex attaches generated SBOM; security signs exceptions |
| Which subprocessors are involved? | None are assumed by this local POC. The production hosting, telemetry, and model asset sources require a current inventory. | Procurement/security to validate before contracting |

- The **storage** row refuses to answer beyond the evidence: "no claim about production residency can be made".
- The **access** row states what is implemented (scoped API keys, role-based access) and, separately, what must still be configured and verified.
- The **dependency** row points at artifacts (allowlists, SBOMs) and names who attaches them and who signs exceptions.
- The **subprocessor** row avoids the common trap of answering "none" for a system whose production hosting is not yet decided.

The other rows follow the same pattern: data processed (redacted, approved sample only; exact fields to confirm), secrets (in the approved secret manager, never in source, logs, fixtures, or tickets; rotation owners to name), incidents (the runbook covers detection, containment, rollback, evidence, escalation; notification timing and contacts to agree in the contract).

**Commercial assumptions.** "No price or savings estimate is approved." Any proposal must state request volume, retained data, support hours, deployment footprint, and service level assumptions; the customer owns procurement terms and the service level, the engineering team only the technical estimates and their evidence.

**Open items before external delivery**, numbered: confirm data classes, geography, and retention; attach the production threat model, per-image SBOM, dependency exceptions, and access review; confirm escalation contacts and notification terms; have procurement validate hosting and subprocessor statements. The closing line: "Until those approvals, this document is an internal draft, not a security certification."

### 5.3 Why this answer

- **Answering every question "yes"** is what a questionnaire seems to invite and what gets a vendor removed at audit. Answers bounded by evidence survive scrutiny.
- **Leaving unanswerable questions blank** looks like evasion; stating what is not yet known, and who will answer it, shows a process.
- **Including a price to move the deal forward** would commit the team to numbers whose inputs (volume, retention, service level) nobody has agreed, and which belong to the customer's procurement.
- **Tradeoff accepted.** The questionnaire is less reassuring to read than a list of confident answers; it is far more useful to the customer's security reviewer.

### 5.4 Explain it back

1. Why does the storage answer say no claim about production residency can be made?
   <details><summary>Answer</summary>The only evidence is the course configuration, which is not Northstar's production deployment, and the POC location is not yet approved. Any residency claim would be unsupported and could become a contractual misstatement.</details>
2. What distinguishes the two sentences of the access answer?
   <details><summary>Answer</summary>The first describes an implemented control (scoped keys, role-based access); the second lists work still required for this customer (identities, rotation, audit retention). Keeping them apart is the "implemented vs planned" rule.</details>
3. Why is "none" the wrong answer to the subprocessor question even though the local POC uses none?
   <details><summary>Answer</summary>The question is about the system the customer will adopt. Production hosting, telemetry, and model sources may all involve third parties; the right answer is that none are assumed by the POC and production needs an inventory.</details>
4. Why does the document call itself "not a security certification"?
   <details><summary>Answer</summary>It is a draft prepared by the engineering team. Only the customer's security reviewer can approve its technical statements, and only after the open items are closed.</details>

## 6. Pitfalls

| # | Pitfall | Symptom | How the solution avoids it |
|---|---|---|---|
| 1 | The deliverable records a conclusion without its customer evidence | Another engineer cannot reproduce the sizing or recommendation. | each answer names its control or artifact (SBOM, runbook, access model) |
| 2 | An assumption is written as a measured fact | The engagement plan can promise capacity or security the system has not demonstrated. | implemented controls and remaining configuration are stated separately; no residency or price claim without evidence |
| 3 | The next owner and decision gate are missing | Work stalls at handoff even when the technical work is complete. | a "Status / owner" column, numbered open items, and approvers for security and commercial statements |

## 7. Where it's used next

| Direction | Module | Connection |
|---|---|---|
| Back | `field.05` | the data flow and rollback this review answers questions about |
| Back | `craft.17` | the threat model behind the data and access answers |
| Back | `craft.18` | the SBOMs behind the dependency answer |
| Back | `ethics.05` | the usage policy behind the data-handling answers |
| Forward | `field.07` | the handoff relies on the incident and access answers |

## Going further

| Resource | What to inspect |
|---|---|
| [Mock engagement sequence](README.md) | Carry the evidence and open decisions into the next customer-facing artifact in the sequence. |
