# Escalation and Handoff

How to hand a problem to the people who can fix it so they act on the first read: the escalation report for the performance team, the evidence each counterpart needs, the incident call, and how to relay research findings back to a customer.

> Parent track: [Field Engineering](../). Uses the benchmark evidence from [03](../03-performance-testing-engagements/) and the failure-mode table from [05](../05-migration-and-cutover/#4-common-failure-modes).

## Overview

- **Primary reference**: [*Postmortem Culture: Learning from Failure*](https://sre.google/sre-book/postmortem-culture/), *Site Reliability Engineering* ch. 15 (free)
- **Supplementary**: [*Managing Incidents*](https://sre.google/sre-book/managing-incidents/), *SRE* ch. 14 (free); [Serving & Load](../../ml/04-llm-systems/serving-and-load/) (free, in repo) for the engine metrics an escalation cites
- **Prerequisites**: [Performance Testing Engagements](../03-performance-testing-engagements/), [Migration and Cutover](../05-migration-and-cutover/)
- **Estimated time**: 2 days at 6-8 hrs/week
- **Usually led by**: FDE; SE escalates pre-sales blockers through the same template

## Key Takeaways

- Escalate when you have done the work you can do, and bring **evidence, not a narrative**.
- A handoff is accepted when the receiver can act **without calling you**.
- The **"what I ruled out"** section separates an escalation from a forwarded complaint.
- During an incident the customer needs an **update cadence** more than a root cause. Commit to a time and keep it.
- When taking over work, ask for the same artifacts you would hand over. If they do not exist, writing them is your first deliverable.

## How to Study

- Write an escalation for a problem you can reproduce locally (for example, `loadgen.py --mock` with `--mock-max-running` set low so TTFT explodes). Hand it to a peer who has not seen your setup and time how long until they ask a question.
- Read one public postmortem per day for a week and rewrite its summary in the incident update format below.
- Practice relaying a technical finding at two levels: one sentence for the customer's VP, one paragraph for their engineer.

---

# Concepts & Techniques

## Core Insight

Specialists (performance engineers, MLEs, researchers) are the scarcest people in an inference company, and every round trip with them costs a day. The field engineer's leverage is to arrive with the environment, the traffic shape, the measurement, and the hypotheses already ruled out, so the specialist starts at diagnosis instead of interrogation. The same discipline runs the other way: findings from research only help a customer when they are translated into a decision the customer can make.

## 1. Escalation Report Template

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

The "expected" column needs a source: the benchmark report ([03](../03-performance-testing-engagements/#8-benchmark-report-template)), the hypothesis math ([02](../02-qualification-and-sizing/)), or a previous run. "It feels slow" is not an expected value.

## 2. What Each Counterpart Needs

| To | When | Bring |
|---|---|---|
| **MTS / performance** | Latency or throughput misses SLO after config tuning you can do yourself | Engine version and commit, full launch flags, GPU type and count, traffic shape (QPS, input/output token histograms, prefix share), benchmark command and raw output, a trace or profile, minimal repro script |
| **MLE** | Quality gap after migration or fine-tune not converging | Eval set (or a sanitized sample), metric definition, baseline score versus current score, prompts with chat template applied, sampling params, training config and loss curves |
| **Research** | A customer problem no current recipe solves (a new architecture, a domain where speculators fail, a quantization that breaks a capability) | The pattern across customers, the measured gap, why the existing recipes were ruled out, the business value of solving it |
| **Product** | A missing feature or a recurring workaround blocks deals | Accounts affected, the workaround and its cost, revenue at stake, the smallest change that removes the blocker |
| **Solutions architect** | A pattern repeats across three or more accounts | The three accounts, the common integration, what you built each time |
| **Account executive** | Scope or commercial change: new workload, capacity request, SLO change | Revised deployment hypothesis with GPU count and cost per 1M tokens, risk list, timeline |

The reverse direction matters too. When you take over from an MLE or MTS, ask for the same artifacts. If they do not exist, writing them is your first deliverable.

**Before escalating to performance**, walk the [failure-mode table](../05-migration-and-cutover/#4-common-failure-modes) and record each first check in "what I ruled out". Most escalations that bounce back fail one of these: no engine version, no traffic shape, a closed-loop benchmark offered as SLO evidence, or prefix-cache hits in the repro.

## 3. Incident and Escalation Call

**Agenda (30 min, can be called within the hour)**

| Min | Item |
|---|---|
| 0-5 | Impact: what users see, since when, how many requests |
| 5-15 | Timeline and what changed (their deploy, our deploy, traffic shift) |
| 15-25 | Current mitigation and next diagnostic step, with owners |
| 25-30 | Next update time; who is on point on each side |

**Key ideas**:
- **One voice to the customer.** Name the person who sends updates; specialists work the problem, they do not field status questions.
- **Cadence over cause.** Commit to "next update at 14:30" and keep it, even if the update is "no change".
- **Mitigate first.** Roll back, shift traffic, add replicas, or fall back to the incumbent path ([05](../05-migration-and-cutover/#3-shadow-canary-cutover-rollback)) before root cause is known.
- **Blameless write-up after.** Timeline, impact, root cause, contributing factors, action items with owners ([Postmortem Culture](https://sre.google/sre-book/postmortem-culture/)).

Update format:

```markdown
[<time, timezone>] <customer> / <symptom> / status: investigating | mitigated | resolved
Impact now: <what users see, % of requests>
Since last update: <what was done, what was learned>
Next step: <action, owner>
Next update: <time>
```

## 4. Relaying Research Back to the Customer

Research and performance findings arrive as mechanisms ("the speculator's acceptance rate drops on legal text because the draft was trained on chat data"). The customer needs a decision.

| Layer | Example | Audience |
|---|---|---|
| **Finding** | Speculative decoding acceptance rate is 0.35 on their traffic versus 0.7 on chat benchmarks | Their MLE, in a written note |
| **Consequence** | TPOT gain at low load is about 10%, not the 2x on the vendor blog | Their engineering lead |
| **Decision** | Keep speculative decoding off at peak; test a draft fine-tuned on their domain next quarter, or accept current TPOT | Their decision maker |
| **What changes in the plan** | Hypothesis row updated; no change to cost; new risk logged | Everyone, in the next notes |

**Rules**:
- Translate, do not forward. A research thread pasted into a customer channel leaks internal context and confuses the reader.
- State what was measured and what was not; carry the "not confirmed" marker through.
- Ask research before sharing anything unpublished, and never share another customer's data or name as the evidence.
- Close the loop internally: tell research what the customer decided, so the pattern counts toward their priorities.

## Connections to Other Tracks

| Concept | Connected Track | Application |
|---------|-----------------|-------------|
| Engine metrics: queue depth, KV usage, preemptions | [Serving & Load](../../ml/04-llm-systems/serving-and-load/) | Evidence section |
| Traces, dashboards, alerting | [Observability](../../systems/04-observability/) | Engine metrics and profiles |
| Postmortems, lessons learned | [Software Craftsmanship: Lessons from Practice](../../software-craftsmanship/04-lessons-from-practice/) | Blameless write-ups |
| Eval artifacts for MLE handoffs | [LLM Evaluation](../../ai-platform-engineering/09-llm-evaluation/) | Quality escalations |

## Company Relevance

| Company | How This Appears | Difficulty |
|---------|-----------------|------------|
| Fireworks AI / Together AI / Baseten | FDE-to-performance-team escalations; "how would you escalate this?" interview prompts | Advanced |
| Google / any SRE organization | Incident command, update cadence, blameless postmortems | Intermediate |
| Palantir | Field-to-product feedback loops from forward deployed teams | Intermediate |

## Exercise

**Scenario**: Lexa ([brief](../01-discovery/#exercise)) is two weeks past cutover ([05](../05-migration-and-cutover/#exercise)). Every weekday around 9:30 CET, TTFT p95 for contract review rises from about 1.2 s to 6 s for twenty minutes, then recovers. Error rate is flat. Their nightly job finished late twice this week.

**Deliverable**:

1. An **escalation report** to the performance team using the template, with plausible values from your earlier documents and at least four hypotheses in "what I ruled out" (each with the check you ran, drawn from the failure-mode table).
2. The **first two incident updates** to Lexa in the update format.
3. A **relay note** for Lexa's VP of Engineering, assuming performance finds that autoscaling reacts to GPU utilization instead of queue depth and the nightly job overran into the morning.

### Done when

- A performance engineer could start diagnosis from the report without a question (test with a peer).
- The report links the overrunning nightly job to the morning symptom as a hypothesis with evidence, not a conclusion.
- Each incident update has a next-update time.
- The relay note ends in a decision Lexa can make, with its cost.
