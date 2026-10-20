# POC and Evaluation

A proof of concept (POC) is a small contract. Both sides agree, in advance, what evidence will trigger which decision. That framing changes how you write one. A POC exists to make a decision, so its criteria must be able to fail: "evaluate latency" cannot fail, while "TTFT p95 < 500 ms at 50 req/s open loop for 30 minutes, measured client-side" can. Every criterion therefore has a **threshold**, a **measurement method**, and an **owner**.

Quality is usually the deciding criterion, and most quality failures after a model swap turn out to be formatting and template issues rather than model capability. So a good POC plan pairs two pieces of evidence: a latency benchmark on the customer's traffic shape, owned by the vendor, and an eval harness on the customer's own data, owned by the customer. It then defines the move that follows each outcome before the POC starts, because agreeing exit criteria afterwards is negotiating.

This chapter builds on the hypothesis from [Qualification and Sizing](../02-qualification-and-sizing/README.md) and the benchmark method from [Performance Testing Engagements](../03-performance-testing-engagements/README.md). You will learn to write criteria that can fail, fill a complete POC plan, prove eval parity against the incumbent model case by case, and run the kickoff and readout calls so the POC ends in a decision.

> **Learning objectives**
> - Write POC success criteria for quality, latency, reliability, cost, features, and compliance, each with a threshold, a method, and an owner.
> - Fill a POC plan with workload, configuration, benchmark runs, timeline, and exit criteria (success, partial, stop) agreed in advance.
> - Run an eval parity comparison against the incumbent as a per-case diff, separating format problems from capability gaps.
> - Run a POC kickoff and a readout that leads with the verdict per criterion and ends in a decision.

## Before you begin

You should have worked through [Discovery](../01-discovery/README.md), [Qualification and Sizing](../02-qualification-and-sizing/README.md), and [Performance Testing Engagements](../03-performance-testing-engagements/README.md). The primary reference is [LLM Evaluation](../../ai-engineering/29-evaluation/) in *AI Engineering*: eval set design, judges, and regression testing. Read it with one question in mind: which of these metrics would the customer's VP accept as "same quality"? Two free supplements help: [*Implementing SLOs*](https://sre.google/workbook/implementing-slos/) from *The Site Reliability Workbook*, on turning an SLO into a measurable criterion, and [Serving & Load](../../ai-engineering/20-inference-engines/04-serving-and-load/) for the latency criteria. In practice an FDE (or an SE, for pre-sales POCs) leads with the customer's engineering owner, and an MLE joins for eval design.

## Writing criteria that can fail

A POC plan is only as useful as its criteria, and each kind of criterion has a standard form.

**Quality** is the score on the customer's eval set, with their judge and rubric, within a tolerance of the incumbent's baseline, and with zero regressions on golden cases. **Latency** names the SLO metric, percentile, threshold, arrival rate, load model, duration, and measurement point, exactly as in [Performance Testing Engagements](../03-performance-testing-engagements/README.md). **Reliability** is the error rate (non-2xx responses and timeouts) during the latency run, with rate limits recorded separately. **Cost** is the cost per 1M tokens at the measured operating point, against a target derived from the customer's current spend (see [Commercials and Security](../06-commercials-and-security/README.md)). **Features** means tool calls and structured output parse on 100% of the eval cases that use them. **Compliance** means the residency and retention constraints are verified rather than assumed, with a named owner on the customer's security side signing off.

Ownership follows evidence: the customer owns quality, and the vendor owns latency and cost evidence. A good test of your criteria is to try to "pass" each one with a misleading measurement and then tighten the method until you cannot.

## The POC plan

The criteria go into a plan that also fixes the workload, the configuration, the runs, the timeline, and what happens at the end. Here is a complete template, filled with the support-copilot numbers from [Qualification and Sizing](../02-qualification-and-sizing/README.md):

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

The benchmark runs are the standard matrix from [Performance Testing Engagements](../03-performance-testing-engagements/README.md), with the open-loop rates set to the customer's average, peak, and 30% over peak. The exit criteria are the part teams most often leave vague, and they are the part that turns the POC into a decision.

> **Tip:** If production data will enter the POC, start the security review (see [Commercials and Security](../06-commercials-and-security/README.md)) in week 1. It is usually the longest pole.

## Eval parity against the incumbent

The quality criterion needs its own method, because "matched quality" is easy to claim and hard to prove. Eval parity is a per-case diff, not an aggregate score: a 0.5-point average drop can hide ten broken golden cases.

Start with a **baseline**. Run the customer's eval on the incumbent in week 1, with the exact production prompts and parameters. Without a fresh baseline, "matched quality" has nothing to match.

Keep **inputs identical** on both sides: the same eval cases, the same system prompt, the same judge model and rubric version, and sampling parameters pinned explicitly.

**Diff per case.** List every case where the candidate scores lower, and have a human read each regression and label its cause: format, refusal, factual, truncation, or tool call. **Golden cases** never regress, and the list of golden cases is agreed before the run.

**Separate format from capability.** Most regressions after a swap are caused by the chat template, stop tokens, or schema handling (see [Template and parameter differences](../05-migration-and-cutover/README.md#template-and-parameter-differences)). Fix those before concluding the model is weaker.

Measure the **quantization delta**. If the candidate configuration is FP8 or lower, run BF16 on the same cases once, so the cost of quantization is measured rather than guessed.

Adapt the prompt **before fine-tuning**. Do one round of prompt changes first, and fine-tune only if prompt changes plateau (see [Training & Post-Training](../../ai-engineering/19-training-methods/) in *AI Engineering*).

Watch for **judge bias**. An LLM judge from the incumbent's model family may prefer the incumbent's style, so spot-check the judge's verdicts against a human sample.

The results go into an eval parity report, sliced so that an aggregate cannot hide a broken subset:

| Slice | Cases | Incumbent | Candidate | Delta | Regressions read | Top cause |
|---|---|---|---|---|---|---|
| All | | | | | | |
| Golden | | | | | | |
| Long inputs (> p95 length) | | | | | | |
| Tool-using | | | | | | |

## Kickoff and readout calls

The plan is agreed and the results are delivered in two calls, each with a fixed agenda. The POC kickoff takes 45 minutes and walks the written plan:

| Min | Item |
|---|---|
| 0-10 | Walk the written POC plan, line by line |
| 10-25 | Success criteria: confirm each metric, threshold, and how it is measured |
| 25-35 | Access, data transfer, environments, security sign-off |
| 35-45 | Owners, check-in cadence, readout date |

The readout also takes 45 minutes and leads with the verdict per criterion and the configuration that produced it:

| Min | Item |
|---|---|
| 0-5 | One-slide verdict: criteria met, missed, or partial |
| 5-25 | Results per criterion with raw numbers and the config that produced them |
| 25-35 | What we would change for production; cost per 1M tokens at the measured operating point |
| 35-45 | Decision and next step: migrate, extend, or stop |

Send the readout document 24 hours before the call, so the meeting is for the decision, not the reading. Rehearse the readout aloud with a peer playing a skeptical CTO before you give it for real.

## Where this leads

Each criterion draws on another book. Criterion 1 depends on the eval sets, LLM judges, and regression suites of [LLM Evaluation](../../ai-engineering/29-evaluation/). Criteria 2 to 4 use the load testing and goodput of [Serving & Load](../../ai-engineering/20-inference-engines/04-serving-and-load/). The quantization delta rests on the accuracy trade-offs in [Quantization](../../ai-engineering/20-inference-engines/01-quantization/). The "stop" path to fine-tune scoping uses SFT, LoRA, and DPO from [Training & Post-Training](../../ai-engineering/19-training-methods/), and when only part of the traffic passes, [Model Routing & Cascades](../../ai-engineering/24-model-routing/) offers a way forward.

POCs are a core part of the job across the industry. Fireworks AI, Together AI, and Baseten run paid or free POCs for dedicated deployments and fine-tunes, and applied AI and solutions teams at OpenAI and Anthropic run customer evals during enterprise adoption; both are advanced work. Databricks and Snowflake run platform POCs with success plans signed by the customer, and Palantir runs bootcamp-style POCs measured on a customer outcome.

When the POC passes, the next chapter, [Migration and Cutover](../05-migration-and-cutover/README.md), moves production traffic.

### Further reading

- [LLM Evaluation](../../ai-engineering/29-evaluation/), in *AI Engineering*.
- [*Implementing SLOs*](https://sre.google/workbook/implementing-slos/), *The Site Reliability Workbook* (free).

## Exercise

Start by filling the template for the support copilot from [Qualification and Sizing](../02-qualification-and-sizing/README.md), then try to pass each criterion with a misleading measurement and tighten the method until you cannot.

The deliverable is a **POC plan for Lexa** (see the [brief](../01-discovery/README.md#exercise)), using the template, your hypothesis from the [Qualification and Sizing exercise](../02-qualification-and-sizing/README.md#exercise), and your benchmark plan from the [Performance Testing Engagements exercise](../03-performance-testing-engagements/README.md#exercise). Include:

- The deployment hypothesis summary: model size class, tier for the online path, tier for the nightly job, GPU count range, quantization, and whether prefix caching applies.
- Success criteria for quality on their 150 labelled sections, JSON schema validity, latency, EU residency and no retention, and cost per 1M tokens against their current $60k/month spend.
- An eval parity plan: baseline on the incumbent, slices (long sections, golden cases), and who reads regressions.
- The point at which you involve the CISO, and what you bring (see [Commercials and Security](../06-commercials-and-security/README.md)).

### Done when

- Every success criterion has a threshold, a measurement method, and an owner.
- The hypothesis states its token rates, compute estimate, and cost comparison, each with the formula visible.
- The plan names what you would do if the quality criterion fails.
- The nightly job is priced on a batch or async surface, not on the online deployment.
- A peer reading only your documents could run the POC without asking you a question.

## Key terms

- **Proof of concept (POC)**: a time-boxed test whose agreed criteria decide whether to migrate, extend, or stop.
- **Exit criteria**: the success, partial, and stop conditions and the move that follows each, agreed before the POC starts.
- **Eval parity**: evidence that the candidate matches the incumbent on the customer's eval, shown as a per-case diff.
- **Golden case**: an eval case that must never regress, agreed before the run.
- **Quantization delta**: the measured quality difference between a quantized configuration and BF16 on the same cases.

## Summary

- A POC exists to make a decision, so every criterion must be able to fail.
- Every criterion has a threshold, a measurement method, and an owner; the customer owns quality, the vendor owns latency and cost evidence.
- Eval parity is a per-case diff, not an aggregate; a small average drop can hide broken golden cases.
- Most post-swap regressions are template and format problems; fix them before blaming the model.
- Write exit criteria before the POC starts, and lead the readout with the verdict per criterion and the config that produced it.

## Review questions

1. Why is "evaluate latency" a bad POC criterion, and what would you write instead?
   <details><summary>Answer</summary>It cannot fail, so it cannot inform a decision. A criterion needs a metric, percentile, threshold, load, duration, and measurement point, for example "TTFT p95 < 500 ms at 50 req/s open loop for 30 minutes, measured client-side."</details>
2. The candidate's average eval score is 0.5 points below the incumbent's, inside the 1.0-point tolerance. Is quality met?
   <details><summary>Answer</summary>Not necessarily. The criterion also requires zero golden-case regressions, and a small average drop can hide several broken cases. Read the per-case diff, label each regression's cause, and check the golden slice.</details>
3. Why run BF16 once on the same cases when the candidate uses FP8?
   <details><summary>Answer</summary>To measure the quality cost of quantization directly. Without it, a regression could be blamed on the model or on FP8 without evidence either way.</details>
4. Why send the readout document 24 hours ahead?
   <details><summary>Answer</summary>So the meeting is spent on the decision (migrate, extend, or stop) rather than on reading results for the first time.</details>
