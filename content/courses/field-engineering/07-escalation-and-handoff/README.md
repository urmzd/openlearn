# Escalation and Handoff

Specialists, the performance engineers, MLEs, and researchers, are the scarcest people in an inference company, and every round trip with them costs a day. The field engineer's leverage is to arrive with the environment, the traffic shape, the measurement, and the hypotheses already ruled out, so the specialist starts at diagnosis instead of interrogation. Escalate when you have done the work you can do, and bring **evidence, not a narrative**. A handoff is accepted when the receiver can act **without calling you**.

The same discipline runs in the other direction. Findings from research only help a customer when they are translated into a decision the customer can make, and when you take over work from a specialist you should ask for the same artifacts you would hand over. If they do not exist, writing them is your first deliverable.

This chapter gives you the tools for both directions. You will learn an escalation report template, what evidence each counterpart needs, how to run an incident call with a fixed update cadence, and how to relay research back to a customer in layers. It uses the benchmark evidence from [Performance Testing Engagements](../03-performance-testing-engagements/README.md) and the failure-mode table from [Migration and Cutover](../05-migration-and-cutover/README.md#common-failure-modes). In practice an FDE leads, and an SE escalates pre-sales blockers through the same template.

> **Learning objectives**
> - Write an escalation report with environment, traffic shape, observed versus expected values with sources, evidence, what was ruled out, and a clear ask.
> - Match each counterpart (performance, MLE, research, product, SA, AE) to the situation that warrants them and the evidence they need.
> - Run an incident call and send updates on a committed cadence, mitigating before the root cause is known.
> - Translate a research finding into a consequence, a decision, and a plan change for the right audience.

## Before you begin

You should have worked through [Performance Testing Engagements](../03-performance-testing-engagements/README.md) and [Migration and Cutover](../05-migration-and-cutover/README.md). The primary reference is [*Postmortem Culture: Learning from Failure*](https://sre.google/sre-book/postmortem-culture/), chapter 15 of *Site Reliability Engineering* (free). Supplements are [*Managing Incidents*](https://sre.google/sre-book/managing-incidents/), chapter 14 of the same book (free), and [Serving & Load](../../machine-learning/04-llm-systems/serving-and-load/) in *Machine Learning and AI* for the engine metrics an escalation cites.

## The escalation report

An escalation is a document, and its first line is a symptom with a number. The template below is what a performance engineer needs to start diagnosis on the first read:

```markdown
# Escalation: <one-line symptom with a number>
e.g. "TPOT p95 doubles to 80 ms above 40 req/s on <model> dedicated, customer <X>"

Impact: <customer, users affected, SLO breached, since when>
Severity: <sev level and why>

## Environment
- Engine + version/commit: <...>
- Launch flags (full, verbatim): <...>
- Model + revision + quantization: <...>
- GPU type, count, TP/PP/EP, region, node IDs: <...>

## Traffic shape
- Rate: <req/s avg/peak>, arrival pattern: <steady/bursty>
- Input tokens p50/p95/max, output tokens p50/p95/max
- Prefix sharing: <length, hit rate if known>
- Streaming: <yes/no>; features: <tools, JSON schema, LoRA>

## Observed vs expected
| Metric | Expected | Observed | Source |

## Evidence
- Benchmark command and raw output: <link>
- Engine metrics over the window (queue depth, running/waiting seqs, KV usage, preemptions): <link>
- Trace or profile: <link>
- Minimal repro: <script that reproduces with synthetic or sanitized data>

## What I ruled out
- <hypothesis> -> <how ruled out>

## Ask
<what you need: root cause, config recommendation, patch, ETA>
```

Two sections carry most of the weight. The "expected" column needs a source: the [benchmark report](../03-performance-testing-engagements/README.md#the-benchmark-report-template), the hypothesis math from [Qualification and Sizing](../02-qualification-and-sizing/README.md), or a previous run. "It feels slow" is not an expected value. And the **"what I ruled out"** section is what separates an escalation from a forwarded complaint. Escalations that list what was ruled out get acted on; forwarded complaints get meetings.

To practice, write an escalation for a problem you can reproduce locally, for example `loadgen.py --mock` with `--mock-max-running` set low so TTFT explodes. Hand it to a peer who has not seen your setup and time how long it takes before they ask a question.

## What each counterpart needs

Not every problem goes to the performance team. Each counterpart is right for a particular situation and needs particular evidence:

| To | When | Bring |
|---|---|---|
| **MTS / performance** | Latency or throughput misses SLO after config tuning you can do yourself | Engine version and commit, full launch flags, GPU type and count, traffic shape (QPS, input/output token histograms, prefix share), benchmark command and raw output, a trace or profile, minimal repro script |
| **MLE** | Quality gap after migration or fine-tune not converging | Eval set (or a sanitized sample), metric definition, baseline score versus current score, prompts with chat template applied, sampling params, training config and loss curves |
| **Research** | A customer problem no current recipe solves (a new architecture, a domain where speculators fail, a quantization that breaks a capability) | The pattern across customers, the measured gap, why the existing recipes were ruled out, the business value of solving it |
| **Product** | A missing feature or a recurring workaround blocks deals | Accounts affected, the workaround and its cost, revenue at stake, the smallest change that removes the blocker |
| **Solutions architect** | A pattern repeats across three or more accounts | The three accounts, the common integration, what you built each time |
| **Account executive** | Scope or commercial change: new workload, capacity request, SLO change | Revised deployment hypothesis with GPU count and cost per 1M tokens, risk list, timeline |

The reverse direction matters too: when you take over from an MLE or MTS, ask for the same artifacts, and if they do not exist, write them first.

Before escalating to performance, walk the [failure-mode table](../05-migration-and-cutover/README.md#common-failure-modes) and record each first check in "what I ruled out." Most escalations that bounce back fail on one of four things: no engine version, no traffic shape, a closed-loop benchmark offered as SLO evidence, or prefix-cache hits in the repro.

## The incident and escalation call

Some problems cannot wait for a written exchange. The incident call is 30 minutes, can be called within the hour, and follows a fixed agenda:

| Min | Item |
|---|---|
| 0-5 | Impact: what users see, since when, how many requests |
| 5-15 | Timeline and what changed (their deploy, our deploy, traffic shift) |
| 15-25 | Current mitigation and next diagnostic step, with owners |
| 25-30 | Next update time; who is on point on each side |

Four principles govern the call. Keep **one voice to the customer**: name the person who sends updates, so specialists work the problem instead of fielding status questions. Prefer **cadence over cause**: during an incident the customer needs an update cadence more than a root cause, so commit to "next update at 14:30" and keep it, even if the update is "no change." **Mitigate first**: roll back, shift traffic, add replicas, or fall back to the incumbent path (see [Shadow, canary, cutover, rollback](../05-migration-and-cutover/README.md#shadow-canary-cutover-rollback)) before the root cause is known. And write a **blameless postmortem afterwards**, with the timeline, impact, root cause, contributing factors, and action items with owners, as described in [Postmortem Culture](https://sre.google/sre-book/postmortem-culture/).

Every update uses the same short format, so the customer can scan the thread:

```markdown
[<time, timezone>] <customer> / <symptom> / status: investigating | mitigated | resolved
Impact now: <what users see, % of requests>
Since last update: <what was done, what was learned>
Next step: <action, owner>
Next update: <time>
```

A good exercise is to read one public postmortem a day for a week and rewrite its summary in this format.

## Relaying research back to the customer

Research and performance findings arrive as mechanisms, for example "the speculator's acceptance rate drops on legal text because the draft was trained on chat data." The customer needs a decision. Translating one into the other means moving through four layers, each for a different audience:

| Layer | Example | Audience |
|---|---|---|
| **Finding** | Speculative decoding acceptance rate is 0.35 on their traffic versus 0.7 on chat benchmarks | Their MLE, in a written note |
| **Consequence** | TPOT gain at low load is about 10%, not the 2x on the vendor blog | Their engineering lead |
| **Decision** | Keep speculative decoding off at peak; test a draft fine-tuned on their domain next quarter, or accept current TPOT | Their decision maker |
| **What changes in the plan** | Hypothesis row updated; no change to cost; new risk logged | Everyone, in the next notes |

Four rules keep the relay safe. Translate rather than forward: a research thread pasted into a customer channel leaks internal context and confuses the reader. State what was measured and what was not, and carry any "not confirmed" marker through. Ask research before sharing anything unpublished, and never use another customer's data or name as evidence. Finally, close the loop internally by telling research what the customer decided, so the pattern counts toward their priorities. Practice relaying a technical finding at two levels: one sentence for the customer's VP and one paragraph for their engineer.

## Where this leads

The evidence section of an escalation cites engine metrics such as queue depth, KV usage, and preemptions from [Serving & Load](../../machine-learning/04-llm-systems/serving-and-load/), and the traces, dashboards, and alerting behind them from [Observability](../../systems/04-observability/) in *Systems and Architecture*. Blameless write-ups are treated in depth in [Lessons from Practice](../../software-craftsmanship/04-lessons-from-practice/) in *Software Craftsmanship*, and quality escalations to an MLE bring the eval artifacts of [LLM Evaluation](../../ai-platform-engineering/09-llm-evaluation/).

FDE-to-performance-team escalations, and "how would you escalate this?" interview prompts, are advanced material at Fireworks AI, Together AI, and Baseten. Google and every SRE organization practice incident command, update cadence, and blameless postmortems, and Palantir runs field-to-product feedback loops from its forward deployed teams.

This chapter completes the engagement. The book's final chapter, [Mock Engagement](../08-mock-engagement/README.md), applies all seven chapters to a platform you built yourself.

### Further reading

- [*Postmortem Culture: Learning from Failure*](https://sre.google/sre-book/postmortem-culture/), *Site Reliability Engineering* ch. 15 (free).
- [*Managing Incidents*](https://sre.google/sre-book/managing-incidents/), *Site Reliability Engineering* ch. 14 (free).

## Exercise

**Scenario.** Lexa (see the [brief](../01-discovery/README.md#exercise)) is two weeks past cutover (see the [Migration and Cutover exercise](../05-migration-and-cutover/README.md#exercise)). Every weekday around 9:30 CET, TTFT p95 for contract review rises from about 1.2 s to 6 s for twenty minutes, then recovers. The error rate is flat. Their nightly job finished late twice this week.

The deliverable has three parts:

1. An **escalation report** to the performance team using the template, with plausible values from your earlier documents and at least four hypotheses in "what I ruled out," each with the check you ran, drawn from the failure-mode table.
2. The **first two incident updates** to Lexa in the update format.
3. A **relay note** for Lexa's VP of Engineering, assuming performance finds that autoscaling reacts to GPU utilization instead of queue depth and that the nightly job overran into the morning.

### Done when

- A performance engineer could start diagnosis from the report without a question (test with a peer).
- The report links the overrunning nightly job to the morning symptom as a hypothesis with evidence, not a conclusion.
- Each incident update has a next-update time.
- The relay note ends in a decision Lexa can make, with its cost.

## Key terms

- **Escalation report**: a written handoff with symptom, environment, traffic shape, observed versus expected values with sources, evidence, what was ruled out, and an ask.
- **What I ruled out**: the list of hypotheses already tested, each with the check that eliminated it.
- **Update cadence**: a committed schedule of status updates during an incident, kept even when nothing has changed.
- **Mitigation**: an action that reduces user impact (rollback, traffic shift, more replicas, fallback) before the root cause is known.
- **Blameless postmortem**: a written incident analysis focused on systems and contributing factors, with owned action items.

## Summary

- Escalate when you have done the work you can do, and bring evidence, not a narrative.
- A handoff is accepted when the receiver can act without calling you; "what I ruled out" separates an escalation from a forwarded complaint.
- Match the counterpart to the problem, and bring the evidence they need; ask for the same artifacts when you take work over.
- During an incident, keep one voice and a committed update cadence, and mitigate before the root cause is known.
- Relay research as finding, consequence, decision, and plan change, translated for each audience.

## Review questions

1. Why does an escalation's "expected" column need a source?
   <details><summary>Answer</summary>Without one, the gap between observed and expected cannot be checked. A source (the benchmark report, the hypothesis math, or a previous run) tells the specialist what the system should do and why, rather than "it feels slow."</details>
2. Name the four things that most often make an escalation to performance bounce back.
   <details><summary>Answer</summary>No engine version, no traffic shape, a closed-loop benchmark offered as SLO evidence, and prefix-cache hits in the repro.</details>
3. An hour into an incident, nothing has changed and the next update is due. What do you send?
   <details><summary>Answer</summary>The update, on time, in the standard format, saying there is no change, what was tried since the last update, the next step and owner, and the next update time. Cadence matters more than cause.</details>
4. Why not paste the research team's thread into the customer's channel?
   <details><summary>Answer</summary>It leaks internal context, may include unpublished work or other customers' data, and gives the customer a mechanism instead of a decision. Translate it into the consequence and the decision they can make.</details>
