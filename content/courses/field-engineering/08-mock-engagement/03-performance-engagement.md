# Designing the performance test protocol

A performance engagement should answer one customer question with repeatable measurements. That sounds simple, but most benchmarks run for customers fail in one of two ways: they measure something other than what the customer asked, or they cannot separate a real bottleneck from noise in the environment. The cure is a protocol, written and agreed before any load is sent.

This section takes the estimate from [Qualifying and sizing the engagement](02-qualification-and-sizing.md) and writes the protocol that will test it, applying the methods of [Performance Testing Engagements](../03-performance-testing-engagements/README.md) from earlier in this book. The measurements come from the [open-loop load generator](../../build-an-llm/p10-serving/10-load-generator.md) in *Build Your Own LLM System*, and regressions are located with the [perf regression bisect](../../software-craftsmanship/07-maintenance/04-performance-regression-bisect.md) from *Software Craftsmanship*. The model answer is [`docs/engagement/performance.md`](../../../../course/solutions/docs/engagement/performance.md). Try listing the protocol's rules yourself before reading the walkthrough.

> **Learning objectives**
> - State the question a test decides and, just as explicitly, what it cannot establish.
> - Design a like-for-like comparison with open-loop arrivals that avoids coordinated omission.
> - Specify metrics, repetitions, and spread so a result can be told apart from noise.
> - Agree stop conditions and a regression procedure before the first run.

## What a valid measurement needs

Before testing, agree on the workload, the baseline, the success threshold, the system configuration, and the stop conditions. Preserve reproducibility, and be ready to explain statistical spread. Each of those rests on a principle worth spelling out.

A measurement answers one question, so the protocol starts by stating it (can the candidate handle the weekday peak within customer limits?) and listing what it cannot answer.

A comparison is valid only if one thing changes. Everything else (weights, tokenizer, hardware, prompt sample, arrival schedule, warm-up) is fixed and recorded, with hashes and revisions.

Closed-loop clients lie under load. A client that waits for each response before sending the next slows down exactly when the server does, so the measured latency omits the queueing a real user would see. This is called coordinated omission. An open-loop schedule sends requests at the agreed times regardless of when responses return.

Latency also has a shape over a run: ramp, steady state, drain. Only the steady window says anything about capacity. The ramp and drain are reported but not used for the verdict.

One run is an anecdote. Three runs per condition, all reported with the median and spread, separate the effect from noise, and an invalidated run is kept along with its reason.

A test on someone else's system needs a brake: stop conditions agreed in advance, with the customer able to call stop. And a regression is a finding to locate, with the run metadata preserved and a bisect, not an invitation to tune during a measured run.

## The shape of a run

Put together, the worked procedure is this: run the same open-loop request mix against the baseline and the candidate; report throughput, TTFT, TPOT, errors, and saturation; and use the perf bisect when a change creates a meaningful regression.

## What the protocol must contain

A correct protocol covers the following cases:

| Case | What must hold | Why it matters downstream |
|---|---|---|
| The question and its scope | what the test decides, and what it explicitly does not establish | a load test is not mistaken for production readiness |
| Like-for-like comparison | baseline and candidate share weights, tokenizer, hardware class, prompts, arrival schedule, warm-up | a difference can be attributed to the change |
| Open-loop arrivals | requests are sent on a schedule, not when the previous one returns | no coordinated omission: slow responses do not hide queueing |
| Metrics | offered and completed rate, tokens/s, TTFT and TPOT percentiles, errors, queue depth, utilization, memory | saturation can be located, not just observed |
| Repetition and spread | several runs per condition, each reported, with the median and the spread | noise is not reported as a result |
| Stop conditions | when to abort the test, and who can stop it | the test never harms the customer's environment |
| Regression procedure | what to do when the candidate is slower | a regression is bisected, not tuned away mid-run |

## Reading the model answer

The full protocol is [`docs/engagement/performance.md`](../../../../course/solutions/docs/engagement/performance.md).

It begins with the question and its scope. The test "will determine whether the candidate serving configuration can handle Northstar's weekday peak request mix within customer-defined latency and error limits. It will not establish production readiness, model quality for all users, or performance on unrepresented prompts." The second sentence is as important as the first.

The controls come next: the same weights, tokenizer, hardware class, prompt sample, arrival schedule, and warm-up. The sample is the redacted one Priya approved. The sample hash, config, software revision, and load-generator version are kept with the report, and arrivals are open-loop "to avoid coordinated omission".

The measurement protocol itself is five numbered rules:

1. A 30-minute baseline and a 30-minute candidate run after an agreed warm-up.
2. The approved peak pattern replayed with a 10-minute ramp, a 20-minute steady window, and a 5-minute drain.
3. Offered and completed requests/s, input and output tokens/s, p50/p95/p99 TTFT and TPOT, end-to-end latency, timeout and error rate, queue depth, GPU/CPU utilization, and memory high-water mark.
4. Three repeats per condition, each run and the median reported, spread included, invalidated runs kept with the reason.
5. Stop if the error rate exceeds 2% for two consecutive minutes, memory reaches 90% of the limit, or customer operations asks.

Then comes a line that keeps the protocol honest:

> The 2% stop limit is a proposed safety guard for the test, not an agreed success threshold. [...] A report that lacks those approvals is exploratory only.

The final part covers analysis and regressions. Compare like-for-like runs and find saturation before recommending changes. A regression beyond the agreed 5% budget triggers the [performance bisect](../../software-craftsmanship/07-maintenance/04-performance-regression-bisect.md) with the run metadata preserved. There is no tuning during a measured run, and customer traffic stays isolated. The record ends with the next action and a recovery step: stop load, return the environment to baseline, notify operations.

## Why this shape and not another

A single "peak throughput" number from a closed-loop benchmark is the common shortcut. It overstates capacity and hides tail latency, which is the very thing the customer complained about. Tuning while measuring produces a configuration that fits one run's noise; separating measurement from change keeps every reported number attributable. Testing on production traffic would give the most realistic mix, but risks the customer's users, so a replayed, approved sample in an isolated environment is the safe approximation.

The tradeoff accepted is cost: three 30-minute runs per condition consume several hours of environment time. Fewer runs would not show spread.

## What can go wrong

> **Warning:** A conclusion without its evidence cannot be reproduced. The sample hash, config, revisions, and tool version travel with every report.

> **Warning:** An assumption written as a measured fact lets the plan promise capacity that has not been demonstrated. The scope says what the test cannot establish, and unapproved runs are "exploratory only".

> **Warning:** Without owners and a decision gate, work stalls at handoff. The protocol names owners for the test, the environment, and the thresholds, and gives a dated next action and a recovery step.

## Where this leads

This protocol tests the estimate from [Qualifying and sizing the engagement](02-qualification-and-sizing.md), using the generator and reports of the [open-loop load generator](../../build-an-llm/p10-serving/10-load-generator.md) and, on a regression, the [perf regression bisect](../../software-craftsmanship/07-maintenance/04-performance-regression-bisect.md). Next, [Pre-registering the POC](04-poc-and-evaluation.md) measures latency and reliability with this same protocol.

Further reading:

- The [chapter opening](README.md), to see how the evidence and open decisions travel into each later customer-facing artifact.

## Key terms

- **Open-loop load**: requests sent on a fixed schedule, independent of when responses return.
- **Coordinated omission**: the error a closed-loop client makes by sending less when the server slows, so the latency of unsent requests is never measured.
- **Steady window**: the part of a run between ramp and drain, the only part used for the verdict.
- **Stop condition**: a pre-agreed rule for aborting a test to protect the environment.
- **Like-for-like comparison**: baseline and candidate runs that differ in exactly one factor.

## Summary

- State the question and what the test does not establish before sending any load.
- Fix and record everything except the change under test, and use open-loop arrivals.
- Judge only the steady window, run three repeats per condition, and report spread.
- Safety stop limits are not success thresholds; success thresholds belong to the customer.
- Bisect regressions with the metadata preserved; never tune during a measured run.

## Review questions

1. What is coordinated omission, and how does the protocol avoid it?
   <details><summary>Answer</summary>A closed-loop load generator waits for each response before sending the next request, so when the server slows it also sends less, and the latencies of the requests it never sent are never measured. An open-loop schedule sends at fixed times regardless of responses.</details>
2. Why is the 2% error stop rule explicitly "not an agreed success threshold"?
   <details><summary>Answer</summary>It protects the environment during the test; success thresholds are the customer's to set. Treating a safety guard as a target would let the engineering team define success on the customer's behalf.</details>
3. Why only the 20-minute steady window for the verdict?
   <details><summary>Answer</summary>During the ramp the system is filling caches and queues, and during the drain it is emptying them; neither represents sustained peak behaviour.</details>
4. Why record the sample hash and load-generator version with the report?
   <details><summary>Answer</summary>So the run can be reproduced exactly, and so a later difference can be attributed to the change under test rather than to a different input or tool.</details>
