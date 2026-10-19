# Qualifying and sizing the engagement

Discovery becomes an engagement only when the problem, the sponsor, the resources, and the decision are clear. Qualification is the check that they are; sizing is the first piece of technical work, translating the customer's workload into a capacity estimate that a test can confirm or refute. Both belong in one record, because a sizing that nobody can act on is wasted, and a qualified engagement with no numbers has nothing to test.

This section continues the Northstar Commerce engagement from [Writing the discovery record](01-discovery.md) and applies the methods of [Qualification and Sizing](../02-qualification-and-sizing/README.md) from earlier in this book. The capacity arithmetic is the same as in [Counting: parameters, FLOPs, KV bytes, and memory plans](../../mathematics-for-computing/06-discrete-math-1/01-counting-params-flops-kv-bytes.md) in *Mathematics for Computing*, and the estimate must eventually be checked against the reports of the [open-loop load generator](../../build-an-llm/11-serving/11-load-generator.md) from *Build Your Own LLM System*. The model answer is [`docs/engagement/qualification.md`](../../../../course/solutions/docs/engagement/qualification.md). Try the sizing arithmetic yourself before reading the walkthrough.

> **Learning objectives**
> - Qualify an engagement on urgency, authority, data access, budget, and decision date, with an owner for each gap.
> - Derive offered load and KV memory from a workload and a model configuration, showing every step and unit.
> - State what a first-pass estimate leaves out, so it is not mistaken for a deployment size.
> - Turn the estimate into a bounded next step with gates and a fallback.

## Qualification: can a decision happen?

Qualification asks whether a decision can happen at the end of the work: is there pain, an owner, someone who can pay, data to test with, and a date? Each of those is a row with an owner, because each can block. An unqualified engagement consumes engineering time with no decision at the end.

Sizing has a different job. It must be testable, so it starts from the workload (arrival rate, input and output lengths) and the model's dimensions, and it shows every multiplication along with an uncertainty range. The same discipline applies to compute, memory, throughput, and integration effort. The worked method is to take the customer's arrival rate, prompt and output lengths, and the model dimensions, estimate KV memory and concurrency, compare the estimate with load reports, and name the variables that could change it.

## Working the numbers by hand

Northstar's workload, as supplied, is 12 requests per second, each with 1,200 input tokens and 300 output tokens.

Offered load is rate times length. That gives $12 \times 1200 = 14{,}400$ input tokens/s and $12 \times 300 = 3{,}600$ output tokens/s.

KV memory per token is fixed by the architecture: keys and values (a factor of 2), for each layer, each KV head, and each element of the head dimension, at the width of the KV dtype. For a model with 24 layers, 16 KV heads, head dimension 64, stored in bf16 (2 bytes):

$$2 \times 24 \times 16 \times 64 \times 2 = 98{,}304 \text{ bytes per token.}$$

Concurrency multiplies it. Each sequence holds 1,500 tokens (1,200 in plus 300 out), so 12 concurrent sequences need

$$1500 \times 12 \times 98{,}304 = 1{,}769{,}472{,}000 \text{ bytes},$$

about 1.77 GB, or 1.65 GiB. Twelve concurrent sequences is itself an assumption. By Little's law, concurrency is arrival rate times time in system, so it is 12 only if each request takes about a second end to end.

Finally, the estimate is a lower bound on one component, not a deployment size. Weights, block-allocator overhead, runtime buffers, and headroom come on top, and bursts raise concurrency above the average.

## What the record must contain

Those numbers go into a record alongside the qualification findings. A correct record covers these cases:

| Case | What must hold | Why it matters downstream |
|---|---|---|
| Qualification questions | urgency, sponsor, decision maker, data access, decision date, each with a finding and an owner | an unqualified engagement consumes engineering time with no decision at the end |
| Gates, not assumptions | data approval and the budget owner are stated as conditions to proceed | the POC cannot start on data nobody approved |
| Workload numbers with their source | arrival rate and lengths are labelled customer-supplied and unvalidated | the [performance engagement](03-performance-engagement.md) knows what to verify |
| Arithmetic shown | offered tokens/s and KV bytes per token derived step by step, with units | anyone can recheck or redo it with other inputs |
| What the estimate leaves out | weights, allocator overhead, runtime state, margin, arrival burstiness | the estimate is not mistaken for a deployment size |
| A bounded next step | what will be tested, what success means, which thresholds are still open | the performance engagement has a protocol to write |

## Reading the model answer

The full record is [`docs/engagement/qualification.md`](../../../../course/solutions/docs/engagement/qualification.md).

Its status line reads "qualified for a bounded test, subject to data approval". The qualification is conditional, and the condition is named.

The qualification table follows:

| Question | Finding | Owner / evidence |
|---|---|---|
| Is the problem urgent? | Peak-hour delay causes manual fallback, but lost staff time is not yet measured | Maya to quantify from one week of support records |
| Is there a decision maker? | Budget owner remains unknown | Maya to identify before POC approval |
| Can data be used? | Only a redacted sample after security approval | Priya Shah; approval is a gate, not assumed |

Every "no" or "not yet" has an owner who will close it, and the data row says "a gate, not assumed": the record refuses to proceed on hope.

The first-pass sizing is labelled "explicitly an estimate". The model answer marks the workload numbers as "customer supplied but not yet validated", computes the offered load (14,400 input and 3,600 output tokens/s), and insists that the test "replay the observed arrival pattern rather than assume a constant rate". It then derives KV bytes per token with the formula above and the concurrency product, and closes with what is not counted:

> [...] before allocator/block overhead, weights, runtime state, and safety margin. This estimate is not a deployment commitment. Validate the actual config and concurrency against load and memory reports.

> **Note:** The model answer writes the product as "about 1.77 GiB". The figure is 1.77 GB in decimal units and 1.65 GiB in binary units. Either way it is a small fraction of one accelerator's memory, which is the useful conclusion.

The record ends with the decision and its gates. Proceed with a one-day offline capacity test if Priya approves the sample and Maya names the budget owner. Success is defined in terms of customer-approved p95 TTFT and TPOT limits and an error threshold, and the record says the numeric limits are still open and must be set by the customer before the run. The next action has owners and dates, and a fallback: if approval is late, use synthetic prompts "and label the result as a harness check only".

## Why this shape and not another

Sizing from a vendor table ("one GPU serves N users") is faster, but it is untestable: it hides the workload assumptions the customer could correct. Rounding the estimate up to a deployment (a node count, a price) before the capacity test would turn an estimate into a commitment; the model answer stops at the component it can derive. A constant-rate assumption understates peaks, because queues form under bursts even when the average fits, and requiring a replay of the observed arrival pattern is cheaper than discovering this in production.

The tradeoff accepted is that the record proceeds without the budget owner, but only to a bounded, one-day test. Anything beyond that needs the gate closed.

## What can go wrong

> **Warning:** A conclusion without its customer evidence cannot be reproduced. Here every number is labelled with its source and the arithmetic is shown, so another engineer can redo the sizing with different inputs.

> **Warning:** An assumption written as a measured fact invites promises of capacity the system has not demonstrated. The sizing is titled "explicitly an estimate" and lists what it leaves out.

> **Warning:** Without a next owner and a decision gate, work stalls even when the technical work is done. The decision section names the gates, the owners, and a dated next action.

## Where this leads

This record qualifies the findings and open questions from [Writing the discovery record](01-discovery.md). Its estimate rests on the arithmetic of [Counting: parameters, FLOPs, KV bytes, and memory plans](../../mathematics-for-computing/06-discrete-math-1/01-counting-params-flops-kv-bytes.md) and must be validated against reports from the [open-loop load generator](../../build-an-llm/11-serving/11-load-generator.md). Next, [Designing the performance test protocol](03-performance-engagement.md) tests the estimate, and the POC inherits its gates.

Further reading:

- The [chapter opening](README.md), to see how the evidence and open decisions travel into each later customer-facing artifact.

## Key terms

- **Qualification**: checking that an engagement has pain, an owner, a budget, usable data, and a decision date.
- **Offered load**: the token rate the workload asks of the system, arrival rate times tokens per request.
- **KV bytes per token**: $2 \times \text{layers} \times \text{KV heads} \times \text{head dim} \times \text{bytes per value}$.
- **Little's law**: average concurrency equals arrival rate times average time in system.
- **Gate**: a condition that must be met before the engagement proceeds, stated rather than assumed.

## Summary

- Qualify on urgency, authority, data access, budget, and date, and give every gap an owner.
- Size from the workload and the model configuration, and show every multiplication with units.
- For Northstar: 14,400 input and 3,600 output tokens/s offered; 98,304 KV bytes per token; about 1.77 GB of KV at 12 concurrent sequences.
- Concurrency is an assumption grounded in Little's law; bursts and slower requests raise it.
- An estimate is a lower bound on one component, not a deployment commitment; the next step is a bounded, gated test.

## Review questions

1. Where do the factors 2, 24, 16, 64, and 2 in $98{,}304$ bytes come from?
   <details><summary>Answer</summary>Keys and values (2), one pair per layer (24), per KV head (16), per element of the head dimension (64), and 2 bytes per bf16 value.</details>
2. Why is "12 concurrent requests" an assumption, and what would change it?
   <details><summary>Answer</summary>Concurrency is arrival rate times time in system (Little's law). Twelve concurrent at 12 requests/s means about one second per request; slower generation or bursty arrivals raise it, and with it the KV memory.</details>
3. Why does the record call data approval "a gate, not assumed"?
   <details><summary>Answer</summary>Without approved data there is no valid test; proceeding as if approval will come would schedule work that may have to be discarded, or worse, use data the customer never released.</details>
4. What does the synthetic-prompt fallback buy, and what can it not show?
   <details><summary>Answer</summary>It keeps the harness work moving while approval is pending, but it cannot say anything about Northstar's workload, which is why it must be labelled a harness check only.</details>
