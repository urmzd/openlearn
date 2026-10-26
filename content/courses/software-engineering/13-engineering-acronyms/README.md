# Engineering Acronyms

Software craftsmanship includes making your reasoning understandable to teammates. Acronyms compress conversations, but expanding the letters is only the first step. You also need to know which action, document, or piece of evidence a term calls for. This chapter teaches 14 terms used in proposals, decisions, planning, reliability, incidents, and vendor engagements.

Start with **RFC, ADR, OKR, and SLO** for everyday engineering conversations. Use the reference table when another term appears, then work through the examples before revealing the review answers.

> **Learning objectives**
> - Expand each acronym and explain when to use it in plain language.
> - Distinguish a proposal from a decision, a goal from a metric, and a reliability signal from a target or agreement.
> - Check a numerical reliability or recovery claim against its stated requirement.

## Before you begin

No coding prerequisite is required. Read this chapter alongside the [software craft reading route](../reading-guides/craft.md). For deeper treatment of decision records, continue to [Documentation Writing](../12-documentation/README.md). The [source register](../../../references/sources.md#engineering-acronyms) records the primary references and their scope.

## Quick reference

| Acronym | Stands for | Use |
|---|---|---|
| RFC | Request for Comments | Propose a change and invite feedback before deciding |
| ADR | Architecture Decision Record | Preserve a decision, its rationale, and its consequences |
| RFP | Request for Proposal | Ask vendors to submit proposals against requirements |
| SOW | Statement of Work | Define contracted scope, deliverables, and acceptance criteria |
| OKR | Objectives and Key Results | Set a goal with measurable outcomes for a defined period |
| KPI | Key Performance Indicator | Track an ongoing measure of performance |
| SLA | Service Level Agreement | State agreed service commitments and consequences for missing them |
| SLO | Service Level Objective | Set a measurable service reliability target |
| SLI | Service Level Indicator | Measure a particular aspect of service behavior |
| RTO | Recovery Time Objective | Set the maximum acceptable time to restore service |
| RPO | Recovery Point Objective | Set the maximum acceptable data loss, expressed as a time window |
| PIR | Post-Incident Review | Learn from an incident and track corrective actions |
| NDA | Non-Disclosure Agreement | Define confidentiality obligations for shared information |
| POC | Proof of Concept | Run a small experiment to test feasibility |

These are working meanings for team conversations. An SLO is often an internal operating target, but it can also be published or included in an SLA. An ADR can begin in a proposed state. The distinction is the artifact's purpose and status, rather than a universal rule about when its file can exist.

## The four everyday terms

Consider a team improving an online lesson reader. Each term answers a different question.

| Term | Question | Example |
|---|---|---|
| RFC | What should we change, and what feedback do we need? | Propose caching lesson pages; compare freshness, latency, and failure behavior before choosing |
| ADR | What did we choose, and why? | Record the accepted cache design, its invalidation rule, alternatives, and consequences |
| OKR | What outcome should improve this quarter? | Make lesson access dependable; reduce failed eligible page requests from 0.4% to at most 0.1% by quarter end |
| SLO | What level of service are we targeting? | At least 99.9% of eligible lesson page requests succeed over each rolling 28-day window |

### RFC and ADR: proposal and decision

Write an RFC when a change needs discussion across people or boundaries. State the problem, constraints, proposed approach, alternatives, risks, and open questions. Feedback can reveal that the proposal should change or be rejected. An accepted RFC can remain the historical design document; a separate ADR is a team convention, not a mandatory second file.

When recording a decision in an ADR, include its status, context, choice, and consequences. Link the discussion that informed it. An accepted record should make clear what the team actually chose. A document titled "ADR" whose status is "proposed" does not establish approval. The [worked ADR section](../12-documentation/01-architecture-decision-records.md) shows a complete example.

### OKR and KPI: change and ongoing observation

The example OKR names an outcome and a deadline. Its key result can be checked against request measurements. "Install a cache" names an activity; it does not establish that readers experience fewer failures. A KPI such as weekly lesson completion rate can remain on the dashboard across several quarters, even when the current OKR changes.

The same measure can inform a KPI, a key result, or an SLO. Its role depends on the question: ongoing performance, desired improvement, or acceptable service behavior. An ambitious planning goal does not automatically become a customer commitment.

## SLI, SLO, and SLA: measurement, target, and agreement

For a worked example, define eligible requests as valid requests for published lesson pages received at the service edge. Count one as successful only when it returns the requested page successfully. Include failed and timed-out eligible requests in the denominator, and use the same definition throughout the window.

Suppose 998,500 of 1,000,000 eligible requests succeed during a rolling 28-day window:

```text
Observed SLI = successful eligible requests / all eligible requests
             = 998,500 / 1,000,000 = 99.85%
SLO          = at least 99.9% over that window
Result       = target missed
```

An SLA adds an agreement with the customer, including how compliance is measured and what happens when commitments are missed. The observed SLI alone cannot tell you whether an SLA was breached: check that agreement's scope, window, exclusions, and threshold. Continue with [Observability](../../systems/11-observability/README.md) for operational practice.

## RTO and RPO: downtime and data loss

Suppose a service has an RTO of 30 minutes and an RPO of 5 minutes. It fails at 10:00, returns to usable service at 10:20, and its restored data includes every committed write through 09:57.

```text
Observed recovery time = 10:20 - 10:00 = 20 minutes <= 30-minute RTO
Unrecovered data window = 10:00 - 09:57 = 3 minutes <= 5-minute RPO
```

Both targets are met in this scenario. Fast restoration alone does not establish acceptable data loss. Validate both with a recovery drill that checks service usability and the recovered writes. Backup frequency alone does not prove the RPO, because a backup can be missing, corrupt, or impossible to restore. See [Incident Response and Recovery](../../systems/12-incident-response-and-chaos/README.md).

## Vendor work and incident learning

In a hypothetical vendor engagement, an **NDA** defines confidentiality obligations before sensitive information is shared. An **RFP** asks vendors how they would meet the requirements. A **POC** checks a narrow feasibility claim, such as whether a representative dataset can be imported correctly. A **SOW** specifies the contracted deliverables and how acceptance will be judged. The exact order depends on the engagement; a POC may itself be contracted under a SOW.

After an incident, a **PIR** records impact, timeline, contributing conditions, recovery, and follow-up actions with owners. Use evidence to improve the system. A successful POC does not establish production readiness, and a completed PIR does not establish that its actions were implemented. [Field Engineering](../../field-engineering/README.md) applies these terms to customer engagements.

## Review questions

Commit to an answer and explain the evidence before opening each solution.

### A decision is still disputed

A teammate proposes replacing the database. They create a document called "ADR: Replace the database" with status "proposed." Has the team made the decision? What should happen next?

<details>
<summary>Worked answer</summary>

No. Its status says the choice is still proposed. Gather feedback on constraints, alternatives, migration risks, and the evidence needed to decide, using the team's RFC or decision-review process. Record the outcome and accepted status when the team decides. The filename alone grants no authority.

</details>

### A dashboard shows 99.85%

Using the request counts above, how many failed requests occurred? How many would a 99.9% target allow at the same request volume? Does the result establish an SLA breach?

<details>
<summary>Worked answer</summary>

There were 1,500 failed requests. A 99.9% target allows at most 1,000 failures out of 1,000,000, so the result exceeds that allowance by 500. It misses this SLO. An SLA conclusion needs the agreement's actual terms and matching measurements.

</details>

### Recovery is fast, but the data is older

Keep the 10:00 failure, 10:20 restoration, 30-minute RTO, and 5-minute RPO. Change the recovered data cutoff to 09:45. Which target is met, and what evidence would you collect in a drill?

<details>
<summary>Worked answer</summary>

The 20-minute recovery meets the RTO. The 15-minute data loss window misses the RPO. Record the interruption and usable-service timestamps, then compare known committed writes with restored records to establish the recovery point. Restoring a process successfully does not establish either result without these checks.

</details>

### A POC is declared ready for production

A vendor imports a sample dataset successfully. Someone proposes "Complete the POC" as the only key result for "Make imports dependable." What is missing?

<details>
<summary>Worked answer</summary>

The POC supports feasibility for that sample. It does not validate production scale, failures, recovery, or ongoing reliability. Define an outcome with a baseline, target, deadline, and representative measurement, such as reducing failed eligible imports from 2% to at most 0.5% by quarter end. Put agreed deliverables and acceptance criteria in the SOW when contracting the work.

</details>

## Key terms

- **Request for Comments (RFC)**: in a team design process, a proposal that invites feedback on a change before a decision is made.
- **Architecture Decision Record (ADR)**: a record of one architectural decision, its status, rationale, and consequences.
- **Request for Proposal (RFP)**: a request for vendors to propose how they would meet a set of requirements.
- **Statement of Work (SOW)**: a document specifying contracted scope, deliverables, and acceptance criteria.
- **Objectives and Key Results (OKR)**: a goal-setting framework pairing an objective with measurable outcomes for a defined period.
- **Key Performance Indicator (KPI)**: an ongoing measure selected to track performance against a business or operational concern.
- **Service Level Agreement (SLA)**: an agreement specifying service commitments, how they are measured, and consequences when they are missed.
- **Service Level Objective (SLO)**: a target for a defined measure of service behavior over a specified window.
- **Service Level Indicator (SLI)**: a quantitative measure of a particular aspect of service behavior, such as successful eligible requests divided by all eligible requests.
- **Recovery Time Objective (RTO)**: the maximum acceptable delay from service interruption to restoration of usable service.
- **Recovery Point Objective (RPO)**: the maximum acceptable data loss expressed as a time window before an interruption.
- **Post-Incident Review (PIR)**: an evidence-based review of incident impact, contributing conditions, recovery, and corrective actions with owners.
- **Non-Disclosure Agreement (NDA)**: an agreement defining confidentiality obligations for information shared between parties.
- **Proof of Concept (POC)**: a limited experiment that tests a specific feasibility claim under stated conditions.
