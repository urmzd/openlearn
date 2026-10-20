# Writing the discovery record

A useful engagement begins with the customer's operating problem and the way they make decisions, not with a product tour. Discovery is where you establish that context, before anyone has promised a solution. It is tempting to treat it as a warm-up call before the real technical work; in practice, everything you do later (the sizing, the benchmark, the POC, the migration) is only as good as the record you write at the end of discovery.

This section opens the mock engagement that runs through the rest of the chapter. You will play the field engineer for a mock customer, Northstar Commerce, whose platform team reports slow inference. The techniques come from [Discovery](../01-discovery/README.md), earlier in this book; here you apply them to a platform you understand to the bottom, and the result is a written discovery record that the next stage, [qualification and sizing](02-qualification-and-sizing.md), can start from. The model answer is [`docs/engagement/discovery.md`](../../../../course/solutions/docs/engagement/discovery.md). Try writing your own record from the worked example before you read the walkthrough, then compare.

> **Learning objectives**
> - Run a discovery interview that separates a stated symptom from a measured impact.
> - Record each finding with its source and a confidence level.
> - Name the stakeholders, data restrictions, and decision date that gate every later step.
> - End discovery with numbered success conditions and an owned next action, and with no promises.

## What discovery is for

The engagement exists to support a customer decision, so the first job of discovery is to find out what that decision is, who makes it, and by when. Without those three facts there is nothing to qualify, and the engagement has a start but no end.

To get there, ask open questions, keep the customer's stated symptoms apart from any measured impact, identify the stakeholders and the restrictions on their data, and agree on a written definition of success. Each of those habits protects a later stage. "Responses take too long" is what the customer says; minutes lost or staff affected is what a measurement would show. Writing them separately keeps the later sizing honest, because sizing and testing then start from what is known rather than from what was assumed.

Every later artifact will be checked against evidence, so each finding gets a source and a confidence level now, while it is cheap to record. Data access gates everything technical: a benchmark or an evaluation needs a sample of real requests, and the person who can approve releasing it must be identified in discovery. Unknowns are findings too. A missing budget owner or a missing baseline is written down as open, with an owner who will close it, rather than left out. And discovery ends with conditions for the next meeting, not with a solution.

## Running the interview

Picture the call. A mock platform team tells you inference is slow. Rather than reaching for a fix, you ask for the request mix, the current baseline, how many users are affected, the error budget, the deployment constraints, and what evidence they would accept as proof of improvement. Then you reflect the problem back in your own words and ask them to correct your summary. The correction is often the most valuable thing you hear: it is where the customer tells you what you misunderstood.

## What the record must contain

That interview produces raw notes. Turning them into a record that another engineer can act on means covering a fixed set of cases, each of which matters downstream:

| Case | What must hold | Why it matters downstream |
|---|---|---|
| The problem in the customer's words | the stated symptom is recorded as stated, separately from any measured impact | sizing and testing start from what is known, not what was assumed |
| Evidence and confidence per finding | each answer names its source (interview, report, measurement) and how far it can be trusted | the next engineer can tell a fact from a guess |
| Stakeholders and roles | sponsor, security approver, and budget owner are named, or explicitly marked unknown | [qualification](02-qualification-and-sizing.md) needs a decision maker |
| Data restrictions | what data may be shared, under whose approval | no test set is built on data the customer cannot release |
| A decision and a date | what the customer will decide, and when | the engagement has an end, not just a start |
| No promises | no latency, capacity, or price commitment | a promise made before measurement is a liability |
| Success conditions for the next step, and the next action | numbered, each with an owner | work does not stall between meetings |

## Reading the model answer

The full record is [`docs/engagement/discovery.md`](../../../../course/solutions/docs/engagement/discovery.md). Walking through its parts in order shows why each one is there.

It opens with a header giving status, date, and owners on both sides:

```markdown
**Status:** discovery complete, qualification pending
**Customer owner:** Maya Chen, platform lead
```

The status says what stage comes next, so nobody mistakes discovery for a qualified opportunity. An owner on each side makes every later action assignable.

Next comes the customer problem: the symptom, the impact, and the gap between them.

> The stated symptom is "responses take too long"; no customer trace or production percentile report has been supplied yet. [...] We have not measured the number of affected staff or minutes lost.

The record quotes the symptom and states plainly what has not been measured. It then records the decision (fund a bounded proof of concept, by 2026-10-23), who owns the problem statement, who must approve the data sample (Priya Shah, security), and that the budget owner is not yet identified.

The questions answered so far appear in a table with a column for evidence and confidence:

| Topic | Current answer | Evidence / confidence |
|---|---|---|
| Request mix | Mostly order lookup and policy questions; prompt and output lengths are unknown | Maya's interview notes, low confidence |
| Baseline | No agreed latency or error baseline | Open; requires gateway/load report |
| Constraints | No raw customer identifiers in a shared test set | Priya's verbal requirement; written policy pending |

The third column is the point of the table. "Low confidence", "not instrumented", and "verbal requirement; written policy pending" tell the next engineer exactly which statements need confirming.

Then the record says what was not promised:

> We did not promise a latency improvement or production capacity. The course platform's load reports and runbooks show how to gather relevant evidence, but they are not measurements of Northstar's workload.

This sentence protects the engagement. The platform's own measurements are evidence of method, not of the customer's workload.

The record closes with success conditions for the next meeting, the next action, and what is unresolved. There are four numbered conditions: a redacted sample with a data dictionary and length distribution; written approval of transfer and retention; an agreed baseline and peak window; and a named budget owner. They are followed by one action with an owner and a date, and an explicit list of unresolved items: budget owner, measured impact, retention period, production data boundary.

## Why this shape and not another

It helps to see the alternatives the model answer rejects. A narrative summary ("Northstar has latency problems at peak; we will help") is shorter, but it loses which claims are measured and who owns each gap, so the next stage would have to rediscover them. Proposing a solution in discovery, such as a bigger GPU or a different model, would anchor the customer on an answer before the workload is known, and any later measurement that disagrees becomes a credibility problem. Leaving out unknowns makes the record look complete, and it is the most common way an engagement stalls at qualification: nobody was asked to find the budget owner.

The tradeoff the record accepts is that it ends with no technical content at all. That is correct for discovery. The numbers belong to [qualification and sizing](02-qualification-and-sizing.md) and the [performance engagement](03-performance-engagement.md), once a sample exists.

## What can go wrong

Three failures recur in discovery records, and the model answer is built to avoid each.

> **Warning:** A record that states a conclusion without its customer evidence cannot be reproduced; another engineer cannot rebuild the sizing or recommendation from it. Every row of the findings table names its evidence and confidence.

> **Warning:** An assumption written as a measured fact lets the engagement plan promise capacity or security the system has not demonstrated. The model answer keeps the symptom and the unmeasured impact apart and promises nothing.

> **Warning:** When the next owner and decision gate are missing, work stalls at handoff even when the technical work is complete. The record ends with numbered success conditions, a dated next action, and owners.

## Where this leads

Discovery reuses tools from elsewhere in the library. The [C4 diagrams in D2](../../software-engineering/11-diagramming/01-c4-diagrams-in-d2.md) from *Software Engineering* are how you show the customer the platform's boundaries, and the questions from the [whole-system design review](../../systems/01-system-design/03-design-review-whole-system.md) in *Systems Engineering* are the ones discovery asks. Going forward, [qualification and sizing](02-qualification-and-sizing.md) starts from these findings and carries the open questions with it; every later artifact of the engagement does the same.

Further reading:

- The [chapter opening](README.md), to see how the evidence and open decisions travel into each later customer-facing artifact.

## Key terms

- **Discovery record**: the written output of discovery, listing findings with evidence, stakeholders, constraints, the decision and its date, and the next action.
- **Stated symptom**: the problem as the customer describes it, before any measurement.
- **Measured impact**: the effect of the problem as shown by data, such as minutes lost or staff affected.
- **Confidence level**: how far a finding can be trusted, given its source.
- **Success condition**: a numbered, owned condition that must be met before the next meeting or stage.

## Summary

- Discovery exists to find the customer's decision, its owner, and its date.
- Record the symptom as stated and the impact as measured, separately; if the impact is unmeasured, say so.
- Give every finding a source and a confidence level, and write unknowns down as owned open items.
- Identify the data approver early, because every technical step needs a sample.
- End with numbered success conditions and a dated next action, and promise nothing.

## Review questions

1. Why does the model answer record "responses take too long" separately from the impact on support staff?
   <details><summary>Answer</summary>The first is a stated symptom, the second would be a measured impact, and none has been measured yet. Keeping them apart stops later sizing or ROI arguments from treating a complaint as data.</details>
2. Why is the security approver named in a discovery record about latency?
   <details><summary>Answer</summary>Every later technical step (load test, evaluation) needs a sample of real requests, and the customer will only release one with security approval. Naming Priya now puts the slowest dependency on the critical path early.</details>
3. What is the purpose of the "Evidence / confidence" column?
   <details><summary>Answer</summary>It lets anyone reading the record tell which findings are facts (confirmed in writing or measured) and which are interview impressions, so the next review knows what to test.</details>
4. Why does the record say explicitly that no latency improvement was promised?
   <details><summary>Answer</summary>Customers remember promises. Stating in writing that none was made, and that the platform's numbers are not measurements of this workload, keeps the engagement anchored on evidence still to be gathered.</details>
