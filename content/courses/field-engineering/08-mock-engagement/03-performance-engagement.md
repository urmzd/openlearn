# Mock engagement: performance engagement

## Overview

| | |
|---|---|
| **Module** | `field.03` · practice · docs · Pass 11 · about 30 min reading |
| **Solution** | model answer: [`docs/engagement/performance.md`](../../../../course/solutions/docs/engagement/performance.md) (the performance test protocol for Northstar Commerce) |
| **Needs** | none |
| **Used by** | no code call site; the POC (`field.04`) measures latency and reliability with this protocol |

## Key Takeaways

- Tie each claim to the exact evidence behind it: a run, its configuration, and its spread.
- State the owner, success condition, and recovery action before declaring the work complete.
- Record unresolved assumptions so the next review can test them.

## How to read this chapter

Sections 1 to 3 set out what a performance engagement must establish and how the runs are done. Section 4 lists what a correct test protocol must contain. Section 5 walks through the model answer for Northstar Commerce and explains why each rule in it is there. Try to list the rules yourself before reading section 5.

---

## 1. Why now

A performance engagement should answer a customer question with repeatable measurements. It must separate bottlenecks from noisy environmental effects.

## 2. Principles

Agree on workload, baseline, success threshold, system configuration, and stop conditions before testing. Preserve reproducibility and explain statistical spread.

## 3. Worked example

Run the same open-loop request mix against the baseline and candidate. Report throughput, TTFT, TPOT, errors, and saturation. Use the perf bisect when a change creates a meaningful regression.

## 4. The artifact

### What a correct solution must handle

| Case | What must hold | Why it matters downstream |
|---|---|---|
| The question and its scope | what the test decides, and what it explicitly does not establish | a load test is not mistaken for production readiness |
| Like-for-like comparison | baseline and candidate share weights, tokenizer, hardware class, prompts, arrival schedule, warm-up | a difference can be attributed to the change |
| Open-loop arrivals | requests are sent on a schedule, not when the previous one returns | no coordinated omission: slow responses do not hide queueing |
| Metrics | offered and completed rate, tokens/s, TTFT and TPOT percentiles, errors, queue depth, utilization, memory | saturation can be located, not just observed |
| Repetition and spread | several runs per condition, each reported, with the median and the spread | noise is not reported as a result |
| Stop conditions | when to abort the test, and who can stop it | the test never harms the customer's environment |
| Regression procedure | what to do when the candidate is slower | a regression is bisected, not tuned away mid-run |

## 5. The solution, step by step

### 5.1 From first principles

1. **A measurement answers one question**, so the protocol starts by stating it (can the candidate handle the weekday peak within customer limits?) and listing what it cannot answer.
2. **A comparison is valid only if one thing changes.** Everything else (weights, tokenizer, hardware, prompt sample, arrival schedule, warm-up) is fixed and recorded with hashes and revisions.
3. **Closed-loop clients lie under load.** A client that waits for each response before sending the next slows down exactly when the server does, so the measured latency omits the queueing a real user would see. An open-loop schedule sends at the agreed times regardless.
4. **Latency has a shape over a run**: ramp, steady state, drain. Only the steady window says anything about capacity; the ramp and drain are reported but not used for the verdict.
5. **One run is an anecdote.** Three runs per condition, all reported with the median and spread, separate the effect from noise; an invalidated run is kept with its reason.
6. **A test on someone else's system needs a brake**: stop conditions agreed in advance, with the customer able to call stop.
7. **A regression is a finding to locate**, with the run metadata preserved and a bisect, not an invitation to tune during a measured run.

### 5.2 The model answer, annotated

The full protocol is [`docs/engagement/performance.md`](../../../../course/solutions/docs/engagement/performance.md).

**Question and scope.** The test "will determine whether the candidate serving configuration can handle Northstar's weekday peak request mix within customer-defined latency and error limits. It will not establish production readiness, model quality for all users, or performance on unrepresented prompts." The second sentence is as important as the first.

**Controls.** Same weights, tokenizer, hardware class, prompt sample, arrival schedule, and warm-up; the sample is the redacted one Priya approved; the sample hash, config, software revision, and load-generator version are kept with the report; arrivals are open-loop "to avoid coordinated omission".

**Measurement protocol**, five numbered rules:

1. a 30-minute baseline and a 30-minute candidate run after an agreed warm-up;
2. the approved peak pattern replayed with a 10-minute ramp, a 20-minute steady window, and a 5-minute drain;
3. offered and completed requests/s, input and output tokens/s, p50/p95/p99 TTFT and TPOT, end-to-end latency, timeout and error rate, queue depth, GPU/CPU utilization, and memory high-water mark;
4. three repeats per condition, each run and the median reported, spread included, invalidated runs kept with the reason;
5. stop if the error rate exceeds 2% for two consecutive minutes, memory reaches 90% of the limit, or customer operations asks.

Then a line that keeps the protocol honest:

> The 2% stop limit is a proposed safety guard for the test, not an agreed success threshold. [...] A report that lacks those approvals is exploratory only.

**Analysis and regressions.** Compare like-for-like runs and find saturation before recommending changes; a regression beyond the agreed 5% budget triggers the performance bisect of `craft.16` with the run metadata preserved; no tuning during a measured run; customer traffic stays isolated. The record ends with the next action and a recovery step: stop load, return the environment to baseline, notify operations.

### 5.3 Why this answer

- **A single "peak throughput" number** from a closed-loop benchmark is the common shortcut; it overstates capacity and hides tail latency, the thing the customer complained about.
- **Tuning while measuring** produces a configuration that fits one run's noise; separating measurement from change keeps every reported number attributable.
- **Testing on production traffic** would give the most realistic mix, but risks the customer's users; a replayed, approved sample in an isolated environment is the safe approximation.
- **Tradeoff accepted.** Three 30-minute runs per condition cost several hours of environment time; fewer runs would not show spread.

### 5.4 Explain it back

1. What is coordinated omission, and how does the protocol avoid it?
   <details><summary>Answer</summary>A closed-loop load generator waits for each response before sending the next request, so when the server slows it also sends less, and the latencies of the requests it never sent are never measured. An open-loop schedule sends at fixed times regardless of responses.</details>
2. Why is the 2% error stop rule explicitly "not an agreed success threshold"?
   <details><summary>Answer</summary>It protects the environment during the test; success thresholds are the customer's to set. Treating a safety guard as a target would let the engineering team define success on the customer's behalf.</details>
3. Why only the 20-minute steady window for the verdict?
   <details><summary>Answer</summary>During the ramp the system is filling caches and queues, and during the drain it is emptying them; neither represents sustained peak behaviour.</details>
4. Why record the sample hash and load-generator version with the report?
   <details><summary>Answer</summary>So the run can be reproduced exactly, and so a later difference can be attributed to the change under test rather than to a different input or tool.</details>

## 6. Pitfalls

| # | Pitfall | Symptom | How the solution avoids it |
|---|---|---|---|
| 1 | The deliverable records a conclusion without its customer evidence | Another engineer cannot reproduce the sizing or recommendation. | sample hash, config, revisions, and tool version travel with every report |
| 2 | An assumption is written as a measured fact | The engagement plan can promise capacity or security the system has not demonstrated. | the scope says what the test cannot establish; unapproved runs are "exploratory only" |
| 3 | The next owner and decision gate are missing | Work stalls at handoff even when the technical work is complete. | owners for the test, the environment, and the thresholds; a dated next action and a recovery step |

## 7. Where it's used next

| Direction | Module | Connection |
|---|---|---|
| Back | `field.02` | the estimate this protocol tests |
| Back | `load.01` | the load reports and generator behind the measurements |
| Back | `craft.16` | the performance bisect used on a regression |
| Forward | `field.04` | the POC measures latency and reliability with this protocol |

## Going further

| Resource | What to inspect |
|---|---|
| [Mock engagement sequence](README.md) | Carry the evidence and open decisions into the next customer-facing artifact in the sequence. |
