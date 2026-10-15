# POC and Evaluation

How to write a **proof of concept (POC)** plan that can fail, prove eval parity against the customer's incumbent model, and run the kickoff and readout calls so the POC ends in a decision.

> Parent track: [Field Engineering](../). Builds on the hypothesis from [02](../02-qualification-and-sizing/) and the benchmark method from [03](../03-performance-testing-engagements/). Next: [Migration and Cutover](../05-migration-and-cutover/).

## Overview

- **Primary reference**: [LLM Evaluation](../../ai-platform-engineering/09-llm-evaluation/) (free, in repo): eval set design, judges, regression testing
- **Supplementary**: [*Implementing SLOs*](https://sre.google/workbook/implementing-slos/), *The Site Reliability Workbook* (free): turning an SLO into a measurable criterion; [Serving & Load](../../ml/04-llm-systems/serving-and-load/) (free, in repo) for the latency criteria
- **Prerequisites**: [Discovery](../01-discovery/), [Qualification and Sizing](../02-qualification-and-sizing/), [Performance Testing Engagements](../03-performance-testing-engagements/)
- **Estimated time**: 3 days at 6-8 hrs/week
- **Usually led by**: FDE (or SE in pre-sales POCs), with the customer's engineering owner; MLE joins for eval design

## Key Takeaways

- A POC exists to make a decision. "Evaluate latency" cannot fail; "TTFT p95 < 500 ms at 50 req/s open loop for 30 minutes, measured client-side" can.
- Every criterion has a **threshold**, a **measurement method**, and an **owner**. The customer owns quality; the vendor owns latency and cost evidence.
- **Eval parity** is a per-case diff, not an aggregate score. A 0.5-point average drop can hide ten broken golden cases.
- Write the **exit criteria** before the POC starts: success, partial, stop. Agreeing them later is negotiating.
- The readout leads with the verdict per criterion and the config that produced it.

## How to Study

- Take the template below and fill it for the support copilot in [02](../02-qualification-and-sizing/). Then try to "pass" each criterion with a misleading measurement; tighten the method until you cannot.
- Read [LLM Evaluation](../../ai-platform-engineering/09-llm-evaluation/) with one question: which of these metrics would the customer's VP accept as "same quality"?
- Rehearse the readout agenda aloud with a peer playing a skeptical CTO.

---

# Concepts & Techniques

## Core Insight

A POC is a small contract: both sides agree in advance what evidence will trigger which decision. Quality is usually the deciding criterion, and most quality failures after a model swap are formatting and template issues, not model capability. So the plan pairs a latency benchmark (owned by the vendor) with an eval harness on the customer's data (owned by the customer), and defines the move that follows each outcome.

## 1. Writing Criteria That Can Fail

**Key ideas**:
- **Quality**: score on the customer's eval set with their judge and rubric, a tolerance versus the incumbent baseline, and zero regressions on golden cases.
- **Latency**: the SLO metric, percentile, threshold, arrival rate, load model, duration, and measurement point ([03](../03-performance-testing-engagements/)).
- **Reliability**: error rate (non-2xx, timeouts) during the latency run, rate limits recorded separately.
- **Cost**: cost per 1M tokens at the measured operating point versus a target derived from their current spend ([06](../06-commercials-and-security/)).
- **Features**: tool calls and structured output parse on 100% of eval cases that use them.
- **Compliance**: the residency and retention constraints verified, not assumed (a named owner on the customer's security side signs off).

## 2. POC Plan Template

```markdown
# POC plan: <customer> / <workload>

## Decision this POC informs
<e.g. migrate the support copilot from <incumbent> to a dedicated deployment>

## Success criteria
| # | Criterion | Threshold | Measured how | Owner |
|---|---|---|---|---|
| 1 | Quality | >= baseline - 1.0 pt on 400-ticket eval, no golden-set regressions | Customer eval harness, same judge, same rubric | Customer MLE |
| 2 | TTFT p95 | < 500 ms at 50 req/s | Open-loop Poisson load, 30 min steady state, measured client-side | FDE |
| 3 | TPOT p95 | < 40 ms at 50 req/s | Same run | FDE |
| 4 | Error rate | < 0.1% non-2xx | Same run | FDE |
| 5 | Cost | <= $X per 1M tokens at measured operating point | GPU-hours x rate / tokens served | FDE + AE |
| 6 | Features | Tool calls and JSON schema parse on 100% of eval cases | Eval harness | Customer eng |

## Workload under test
- Request sample: <source, size, date range, sanitized how>
- Token distributions: input p50/p95/max, output p50/p95/max
- Shared prefix: <length, verified byte-identical>

## Configuration under test
- Model, revision, quantization, engine and version, GPU type and count, TP/PP/EP, key flags

## Benchmarks (method in topic 03)
| Run | Mode | Load | Purpose |
|---|---|---|---|
| A | Closed loop | Concurrency sweep 1, 8, 32, 128, 256 | Find saturation throughput per replica |
| B | Open loop | Poisson at 15, 35, 50, 65 req/s | SLO attainment at average, peak, and 30% over peak |
| C | Open loop | Replay of a real traffic hour | Realistic burstiness and length mix |
| D | Cold start | Scale 1 -> 3 replicas | Time to absorb the morning ramp |

## Timeline
| Week | Milestone |
|---|---|
| 1 | Access, data, serverless smoke test, eval baseline on incumbent |
| 2 | Dedicated deployment, runs A and B, eval on candidate config |
| 3 | Tuning, runs C and D, cost model |
| 4 | Readout |

## Exit criteria
- Success: criteria 1-4 and 6 met; 5 within 10% -> recommend migration
- Partial: quality met, latency missed -> one week extension with performance team
- Stop: quality gap > 3 pts after one prompt-adaptation iteration -> recommend fine-tune scoping or stop

## Owners and cadence
- Customer: <eng owner>, <decision maker>; Vendor: FDE, AE, perf contact
- Check-ins: Tuesdays and Fridays, 20 min, written update after each
```

Start the security review ([06](../06-commercials-and-security/)) in week 1 if production data enters the POC; it is usually the longest pole.

## 3. Eval Parity Against the Incumbent

**Key ideas**:
- **Baseline first**: run the customer's eval on the incumbent in week 1, with the exact production prompts and parameters. Without a fresh baseline, "matched quality" has nothing to match.
- **Identical inputs**: same eval cases, same system prompt, same judge model and rubric version, sampling parameters pinned explicitly on both sides.
- **Diff per case**: list every case where the candidate scores lower; a human reads each regression and labels its cause (format, refusal, factual, truncation, tool call).
- **Golden cases** never regress. Agree the list before the run.
- **Separate format from capability**: most regressions after a swap are chat template, stop tokens, or schema handling ([05](../05-migration-and-cutover/#2-template-and-parameter-differences)). Fix those before concluding the model is weaker.
- **Quantization delta**: if the candidate config is FP8 or lower, run BF16 on the same cases once so the quantization cost is measured, not guessed.
- **Prompt adaptation, then fine-tune**: one round of prompt changes first; fine-tune only if prompt changes plateau ([Training & Post-Training](../../ml/07-training-and-post-training/)).
- **Judge bias**: an LLM judge from the incumbent's model family may prefer the incumbent's style. Spot-check judge verdicts with a human sample.

Eval parity report table:

| Slice | Cases | Incumbent | Candidate | Delta | Regressions read | Top cause |
|---|---|---|---|---|---|---|
| All | | | | | | |
| Golden | | | | | | |
| Long inputs (> p95 length) | | | | | | |
| Tool-using | | | | | | |

## 4. Kickoff and Readout Calls

**POC kickoff (45 min)**

| Min | Item |
|---|---|
| 0-10 | Walk the written POC plan, line by line |
| 10-25 | Success criteria: confirm each metric, threshold, and how it is measured |
| 25-35 | Access, data transfer, environments, security sign-off |
| 35-45 | Owners, check-in cadence, readout date |

**POC readout (45 min)**

| Min | Item |
|---|---|
| 0-5 | One-slide verdict: criteria met, missed, or partial |
| 5-25 | Results per criterion with raw numbers and the config that produced them |
| 25-35 | What we would change for production; cost per 1M tokens at the measured operating point |
| 35-45 | Decision and next step: migrate, extend, or stop |

Send the readout document 24 hours before the call so the meeting is for the decision, not the reading.

## Connections to Other Tracks

| Concept | Connected Track | Application |
|---------|-----------------|-------------|
| Eval sets, LLM judges, regression suites | [LLM Evaluation](../../ai-platform-engineering/09-llm-evaluation/) | Criterion 1 |
| Load testing, goodput | [Serving & Load](../../ml/04-llm-systems/serving-and-load/) | Criteria 2 to 4 |
| Quantization accuracy trade-offs | [Quantization](../../ml/04-llm-systems/quantization/) | Quantization delta |
| SFT, LoRA, DPO scoping | [Training & Post-Training](../../ml/07-training-and-post-training/) | The "stop" path to fine-tune scoping |
| Routing and cascades | [Model Routing & Cascades](../../ai-platform-engineering/11-model-routing-and-cascades/) | When only part of the traffic passes |

## Company Relevance

| Company | How This Appears | Difficulty |
|---------|-----------------|------------|
| Fireworks AI / Together AI / Baseten | Paid or free POCs for dedicated deployments and fine-tunes | Advanced |
| OpenAI / Anthropic (applied AI, solutions) | Customer evals during enterprise adoption | Advanced |
| Databricks / Snowflake | Platform POCs with success plans signed by the customer | Intermediate |
| Palantir | Bootcamp-style POCs measured on a customer outcome | Intermediate |

## Exercise

**Deliverable**: a **POC plan for Lexa** ([brief](../01-discovery/#exercise)), using the template, your hypothesis from [02](../02-qualification-and-sizing/#exercise), and your benchmark plan from [03](../03-performance-testing-engagements/#exercise). Include:

- The deployment hypothesis summary: model size class, tier for the online path, tier for the nightly job, GPU count range, quantization, and whether prefix caching applies.
- Success criteria for quality on their 150 labelled sections, JSON schema validity, latency, EU residency and no retention, and cost per 1M tokens against their current $60k/month spend.
- An eval parity plan: baseline on the incumbent, slices (long sections, golden cases), and who reads regressions.
- The point at which you involve the CISO, and what you bring ([06](../06-commercials-and-security/)).

### Done when

- Every success criterion has a threshold, a measurement method, and an owner.
- The hypothesis states its token rates, compute estimate, and cost comparison, each with the formula visible.
- The plan names what you would do if the quality criterion fails.
- The nightly job is priced on a batch or async surface, not on the online deployment.
- A peer reading only your documents could run the POC without asking you a question.
