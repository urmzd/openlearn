# Mock engagement: qualification and sizing

## Overview

| | |
|---|---|
| **Module** | `field.02` · practice · docs · Pass 11 · about 30 min reading |
| **Solution** | model answer: [`docs/engagement/qualification.md`](../../../../course/solutions/docs/engagement/qualification.md) (qualification and a first-pass capacity estimate for Northstar Commerce) |
| **Needs** | none |
| **Used by** | no code call site; the performance engagement (`field.03`) tests the estimate, and the POC (`field.04`) inherits its gates |

## Key Takeaways

- Tie each claim to the exact evidence behind it, and mark estimates as estimates.
- State the owner, success condition, and next action before declaring the work complete.
- Record unresolved assumptions so the next review can test them.

## How to read this chapter

Sections 1 to 3 set out what qualification and sizing are for and how the arithmetic goes. Section 4 lists what a correct qualification record must contain. Section 5 walks through the model answer for Northstar Commerce, re-derives its numbers, and explains why each part is there. Try the sizing arithmetic yourself before reading section 5.

---

## 1. Why now

Discovery becomes an engagement only when the problem, sponsor, resources, and decision are clear. Sizing translates workload into a testable capacity estimate.

## 2. Principles

Qualify urgency, authority, data access, budget, and decision date. Show arithmetic and uncertainty ranges for compute, memory, throughput, and integration effort.

## 3. Worked example

Use the customer's arrival rate, prompt and output lengths, and model dimensions to estimate KV memory and concurrency. Compare the estimate with load reports; name the variables that could change it.

## 4. The artifact

### What a correct solution must handle

| Case | What must hold | Why it matters downstream |
|---|---|---|
| Qualification questions | urgency, sponsor, decision maker, data access, decision date, each with a finding and an owner | an unqualified engagement consumes engineering time with no decision at the end |
| Gates, not assumptions | data approval and the budget owner are stated as conditions to proceed | the POC cannot start on data nobody approved |
| Workload numbers with their source | arrival rate and lengths are labelled customer-supplied and unvalidated | the test in `field.03` knows what to verify |
| Arithmetic shown | offered tokens/s and KV bytes per token derived step by step, with units | anyone can recheck or redo it with other inputs |
| What the estimate leaves out | weights, allocator overhead, runtime state, margin, arrival burstiness | the estimate is not mistaken for a deployment size |
| A bounded next step | what will be tested, what success means, which thresholds are still open | `field.03` has a protocol to write |

## 5. The solution, step by step

### 5.1 From first principles

1. **Qualification asks whether a decision can happen**: is there pain, an owner, someone who can pay, data to test with, and a date. Each is a row with an owner, because each can block.
2. **Sizing must be testable**, so it starts from the workload (arrival rate, input and output lengths) and the model's dimensions, and shows every multiplication.
3. **Offered load is rate times length.** At 12 requests/s, 1,200 input and 300 output tokens each: $12 \times 1200 = 14{,}400$ input tokens/s and $12 \times 300 = 3{,}600$ output tokens/s.
4. **KV memory per token is fixed by the architecture**: keys and values ($\times 2$), for each layer, each KV head, each head dimension, at the KV dtype's width. For 24 layers, 16 KV heads, head dimension 64, bf16 (2 bytes): $2 \times 24 \times 16 \times 64 \times 2 = 98{,}304$ bytes per token.
5. **Concurrency multiplies it.** 1,500 tokens per sequence (1,200 in plus 300 out) times 12 concurrent sequences times 98,304 bytes is $1{,}769{,}472{,}000$ bytes: about 1.77 GB, or 1.65 GiB. Twelve concurrent sequences is itself an assumption: by Little's law concurrency is arrival rate times time in system, so it is 12 only if each request takes about a second end to end.
6. **The estimate is a lower bound on one component**, not a deployment size: weights, block-allocator overhead, runtime buffers, and headroom come on top, and bursts raise concurrency above the average.

### 5.2 The model answer, annotated

The full record is [`docs/engagement/qualification.md`](../../../../course/solutions/docs/engagement/qualification.md).

**Status line: "qualified for a bounded test, subject to data approval".** The qualification is conditional, and the condition is named.

**Qualification table.**

| Question | Finding | Owner / evidence |
|---|---|---|
| Is the problem urgent? | Peak-hour delay causes manual fallback, but lost staff time is not yet measured | Maya to quantify from one week of support records |
| Is there a decision maker? | Budget owner remains unknown | Maya to identify before POC approval |
| Can data be used? | Only a redacted sample after security approval | Priya Shah; approval is a gate, not assumed |

Every "no" or "not yet" has an owner who will close it, and the data row says "a gate, not assumed": the record refuses to proceed on hope.

**First-pass sizing, "explicitly an estimate".** The model answer labels the workload numbers "customer supplied but not yet validated", computes the offered load (14,400 input and 3,600 output tokens/s), and insists that the test "replay the observed arrival pattern rather than assume a constant rate". It then derives KV bytes per token with the formula of step 4 and the concurrency product of step 5, and closes with what is not counted:

> [...] before allocator/block overhead, weights, runtime state, and safety margin. This estimate is not a deployment commitment. Validate the actual config and concurrency against load and memory reports.

(The model answer writes the product as "about 1.77 GiB"; the figure is 1.77 GB in decimal units, 1.65 GiB in binary units. Either way it is a small fraction of one accelerator's memory, which is the useful conclusion.)

**Decision and gates.** Proceed with a one-day offline capacity test if Priya approves the sample and Maya names the budget owner. Success is defined in terms of customer-approved p95 TTFT and TPOT limits and an error threshold, and the record says the numeric limits are still open and must be set by the customer before the run. The next action has owners and dates, and a fallback: if approval is late, use synthetic prompts "and label the result as a harness check only".

### 5.3 Why this answer

- **Sizing from a vendor table** ("one GPU serves N users") is faster but untestable: it hides the workload assumptions the customer can correct.
- **Rounding the estimate up to a deployment** (a node count, a price) before the capacity test would turn an estimate into a commitment. The model answer stops at the component it can derive.
- **A constant-rate assumption** understates peaks: queues form under bursts even when the average fits. Requiring a replay of the observed arrival pattern is cheaper than discovering this in production.
- **Tradeoff accepted.** The record proceeds without the budget owner, but only to a bounded, one-day test; anything beyond needs the gate closed.

### 5.4 Explain it back

1. Where do the factors 2, 24, 16, 64, and 2 in $98{,}304$ bytes come from?
   <details><summary>Answer</summary>Keys and values (2), one pair per layer (24), per KV head (16), per element of the head dimension (64), and 2 bytes per bf16 value.</details>
2. Why is "12 concurrent requests" an assumption, and what would change it?
   <details><summary>Answer</summary>Concurrency is arrival rate times time in system (Little's law). Twelve concurrent at 12 requests/s means about one second per request; slower generation or bursty arrivals raise it, and with it the KV memory.</details>
3. Why does the record call data approval "a gate, not assumed"?
   <details><summary>Answer</summary>Without approved data there is no valid test; proceeding as if approval will come would schedule work that may have to be discarded, or worse, use data the customer never released.</details>
4. What does the synthetic-prompt fallback buy, and what can it not show?
   <details><summary>Answer</summary>It keeps the harness work moving while approval is pending, but it cannot say anything about Northstar's workload, which is why it must be labelled a harness check only.</details>

## 6. Pitfalls

| # | Pitfall | Symptom | How the solution avoids it |
|---|---|---|---|
| 1 | The deliverable records a conclusion without its customer evidence | Another engineer cannot reproduce the sizing or recommendation. | every number is labelled with its source and the arithmetic is shown |
| 2 | An assumption is written as a measured fact | The engagement plan can promise capacity or security the system has not demonstrated. | the sizing is titled "explicitly an estimate" and lists what it leaves out |
| 3 | The next owner and decision gate are missing | Work stalls at handoff even when the technical work is complete. | the decision section names the gates, the owners, and a dated next action |

## 7. Where it's used next

| Direction | Module | Connection |
|---|---|---|
| Back | `field.01` | the discovery findings and open questions this record qualifies |
| Back | `M05.1` | the capacity arithmetic behind the KV estimate |
| Back | `load.01` | the load reports the estimate must be validated against |
| Forward | `field.03` | the performance engagement tests this estimate |

## Going further

| Resource | What to inspect |
|---|---|
| [Mock engagement sequence](README.md) | Carry the evidence and open decisions into the next customer-facing artifact in the sequence. |
