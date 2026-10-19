# Mock engagement: discovery

## Overview

| | |
|---|---|
| **Module** | `field.01` · practice · docs · Pass 11 · about 30 min reading |
| **Solution** | model answer: [`docs/engagement/discovery.md`](../../../../course/solutions/docs/engagement/discovery.md) (discovery notes for the mock customer Northstar Commerce) |
| **Needs** | none |
| **Used by** | no code call site; qualification and sizing (`field.02`) starts from these notes, and every later artifact of the engagement carries their open questions forward |

## Key Takeaways

- Tie each claim to the exact evidence behind it: an interview note, a customer report, a measurement. Say which it is.
- State the owner, success condition, and next action before declaring discovery complete.
- Record unresolved assumptions so the next review can test them.

## How to read this chapter

Sections 1 to 3 set out what discovery is for and how a discovery call runs. Section 4 lists what a correct discovery record must contain. Section 5 walks through the model answer for the mock customer Northstar Commerce and explains why each part is there. Before reading section 5, try writing the record yourself from the worked example, then compare.

---

## 1. Why now

A useful engagement begins with the customer's operating problem and decision process, not with a product tour. Discovery establishes context before anyone promises a solution.

## 2. Principles

Ask open questions, separate stated symptoms from measured impact, identify stakeholders and data restrictions, and agree on a written definition of success.

## 3. Worked example

Interview a mock platform team reporting slow inference. Ask for request mix, baseline, users affected, error budget, deployment constraints, and acceptable evidence. Reflect back the problem and ask them to correct your summary.

## 4. The artifact

### What a correct solution must handle

| Case | What must hold | Why it matters downstream |
|---|---|---|
| The problem in the customer's words | the stated symptom is recorded as stated, separately from any measured impact | sizing and testing start from what is known, not what was assumed |
| Evidence and confidence per finding | each answer names its source (interview, report, measurement) and how far it can be trusted | the next engineer can tell a fact from a guess |
| Stakeholders and roles | sponsor, security approver, and budget owner are named, or explicitly marked unknown | qualification (`field.02`) needs a decision maker |
| Data restrictions | what data may be shared, under whose approval | no test set is built on data the customer cannot release |
| A decision and a date | what the customer will decide, and when | the engagement has an end, not just a start |
| No promises | no latency, capacity, or price commitment | a promise made before measurement is a liability |
| Success conditions for the next step, and the next action | numbered, each with an owner | work does not stall between meetings |

## 5. The solution, step by step

### 5.1 From first principles

1. **The engagement exists to support a customer decision**, so discovery must find out what decision, who makes it, and by when. Without that there is nothing to qualify.
2. **A symptom is not an impact.** "Responses take too long" is what the customer says; minutes lost or staff affected is what a measurement would show. Writing them separately keeps the later sizing honest.
3. **Every later artifact will be checked against evidence**, so each finding gets a source and a confidence level now, while it is cheap to record.
4. **Data access gates everything technical.** A benchmark or evaluation needs a sample; the person who can approve releasing it must be identified in discovery.
5. **Unknowns are findings too.** A missing budget owner or baseline is written down as open, with an owner who will close it, rather than left out.
6. **Discovery ends with conditions for the next meeting, not with a solution.**

### 5.2 The model answer, annotated

The full record is [`docs/engagement/discovery.md`](../../../../course/solutions/docs/engagement/discovery.md). Its parts, and why each is there:

**Header: status, date, owners on both sides.**

```markdown
**Status:** discovery complete, qualification pending
**Customer owner:** Maya Chen, platform lead
```

The status says what stage comes next, so nobody mistakes discovery for a qualified opportunity. An owner on each side makes every later action assignable.

**Customer problem: symptom, impact, and the gap between them.**

> The stated symptom is "responses take too long"; no customer trace or production percentile report has been supplied yet. [...] We have not measured the number of affected staff or minutes lost.

The record quotes the symptom and states plainly what has not been measured. It then records the decision (fund a bounded proof of concept, by 2026-10-23), who owns the problem statement, who must approve the data sample (Priya Shah, security), and that the budget owner is not yet identified.

**Questions answered, with evidence and confidence.**

| Topic | Current answer | Evidence / confidence |
|---|---|---|
| Request mix | Mostly order lookup and policy questions; prompt and output lengths are unknown | Maya's interview notes, low confidence |
| Baseline | No agreed latency or error baseline | Open; requires gateway/load report |
| Constraints | No raw customer identifiers in a shared test set | Priya's verbal requirement; written policy pending |

The third column is the point of the table: "low confidence", "not instrumented", and "verbal requirement; written policy pending" tell the next engineer exactly which statements need confirming.

**What was not promised.**

> We did not promise a latency improvement or production capacity. The course platform's load reports and runbooks show how to gather relevant evidence, but they are not measurements of Northstar's workload.

This sentence protects the engagement: the platform's own measurements are evidence of method, not of the customer's workload.

**Success conditions for the next meeting, then the next action and what is unresolved.** Four numbered conditions (a redacted sample with a data dictionary and length distribution; written approval of transfer and retention; an agreed baseline and peak window; a named budget owner), followed by one owner-and-date action and an explicit list of unresolved items: budget owner, measured impact, retention period, production data boundary.

### 5.3 Why this answer

- **A narrative summary** ("Northstar has latency problems at peak; we will help") is shorter, but loses which claims are measured and who owns each gap; the next stage would have to rediscover them.
- **Proposing a solution in discovery** (a bigger GPU, a different model) would anchor the customer on an answer before the workload is known, and any later measurement that disagrees becomes a credibility problem.
- **Leaving out unknowns** makes the record look complete and is the most common way an engagement stalls at qualification: nobody was asked to find the budget owner.
- **The tradeoff accepted** is that the record ends with no technical content at all. That is correct for discovery; the numbers belong to `field.02` and `field.03`, once a sample exists.

### 5.4 Explain it back

1. Why does the model answer record "responses take too long" separately from the impact on support staff?
   <details><summary>Answer</summary>The first is a stated symptom, the second would be a measured impact, and none has been measured yet. Keeping them apart stops later sizing or ROI arguments from treating a complaint as data.</details>
2. Why is the security approver named in a discovery record about latency?
   <details><summary>Answer</summary>Every later technical step (load test, evaluation) needs a sample of real requests, and the customer will only release one with security approval. Naming Priya now puts the slowest dependency on the critical path early.</details>
3. What is the purpose of the "Evidence / confidence" column?
   <details><summary>Answer</summary>It lets anyone reading the record tell which findings are facts (confirmed in writing or measured) and which are interview impressions, so the next review knows what to test.</details>
4. Why does the record say explicitly that no latency improvement was promised?
   <details><summary>Answer</summary>Customers remember promises. Stating in writing that none was made, and that the platform's numbers are not measurements of this workload, keeps the engagement anchored on evidence still to be gathered.</details>

## 6. Pitfalls

| # | Pitfall | Symptom | How the solution avoids it |
|---|---|---|---|
| 1 | The deliverable records a conclusion without its customer evidence | Another engineer cannot reproduce the sizing or recommendation. | every row of the findings table names its evidence and confidence |
| 2 | An assumption is written as a measured fact | The engagement plan can promise capacity or security the system has not demonstrated. | the symptom and the unmeasured impact are separated; nothing is promised |
| 3 | The next owner and decision gate are missing | Work stalls at handoff even when the technical work is complete. | the record ends with numbered success conditions, a dated next action, and owners |

## 7. Where it's used next

| Direction | Module | Connection |
|---|---|---|
| Back | `craft.09` | the C4 diagrams used to show the customer the platform's boundaries |
| Back | `review.03` | the whole-system design review whose questions discovery reuses |
| Forward | `field.02` | qualification starts from these findings and open questions |

## Going further

| Resource | What to inspect |
|---|---|
| [Mock engagement sequence](README.md) | Carry the evidence and open decisions into the next customer-facing artifact in the sequence. |
