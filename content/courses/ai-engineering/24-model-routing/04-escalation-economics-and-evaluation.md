# Escalation economics and evaluation

This part of [Model Routing and Cascades](README.md) develops the mechanisms below. Work the numerical examples or trace the code before checking the answer.

> **Learning objectives**
> - Evaluate a cascade using total cost and final task quality.

## Escalation policy

Every routing design needs a rule for leaving the cheap path. That rule is the escalation policy, and it is worth writing down before you build anything, including what happens when the router itself errors. There are four ways to trigger escalation, and they fail differently:

| Trigger | How it works | Fails when |
|---------|--------------|------------|
| **Confidence** | Escalate when the model or a scorer reports low confidence | Confidence is miscalibrated, or the input manipulates it |
| **Category** | Named categories always escalate, whatever the model thinks | The category list is incomplete |
| **Verifier** | A deterministic check rejects the output | No verifier exists for the task |
| **Human override** | A person corrects the route | Volume exceeds attention |

Escalate on category or on a verifier, not on model confidence alone. Confidence is produced by the very thing you are trying to check, and untrusted input can move it. If the content being routed can contain instructions, a confidence score is attacker-influenced. A rule such as "security and schema changes always get the strong reviewer" cannot be argued with, because no model evaluates it. This is the same principle as the deterministic gate in the [Grounded SQL Agent](../30-case-study-sql-agent/): the control sits outside the model.

You must also decide the failure direction explicitly, and the right answer depends on what the system does:

| System | Router errors, then | Why |
|--------|--------------------|-----|
| Review or analysis fan-out | **Fail open**: run everything | The worst case is a known cost, never silent loss of coverage |
| Anything that executes or writes | **Fail closed**: refuse | The worst case must not be an unreviewed action |

Treat the escalation rate as a metric. A cost model that assumes 20% of calls escalate is wrong the day the rate becomes 40%, so alert on it and cap it with a budget. And remember that an override is also a label: when a person corrects a route, record the pair. The escape hatch and the training data are the same mechanism.

## Economics: the small decision that steers the bill

With the policy in place, you can reason about cost. Work from tokens, not from a price list. Every per-request figure is tokens times a rate, so every figure can be re-derived when a rate changes.

Consider a system with six specialised subagents, of which two are relevant to a typical request. With illustrative rates, the individual calls cost:

| Call | Tokens in / out | Rate per 1M in / out | Cost |
|------|-----------------|----------------------|------|
| Frontier subagent | 15K / 1K | $2.50 / $15.00 | $0.0525 |
| Small tuned subagent | 9K / 1K | $0.20 flat | $0.0020 |
| Prompted classifier, small model | 19K / 0.1K | $0.10 flat | $0.0019 |
| Tuned classifier, small model | 2K / 0.1K | $0.10 flat | $0.0002 |

Combining them into three configurations gives very different per-request costs:

| Configuration | Per request | Derivation |
|---------------|-------------|------------|
| Run all six on the frontier model | $0.315 | 6 × $0.0525 |
| Prompt-routed, two frontier subagents | $0.107 | $0.0019 + 2 × $0.0525 |
| Tuned router, cascade, 20% escalate | $0.024 | $0.0002 + 2 × (0.8 × $0.0020 + 0.2 × $0.0525) |

Three independent levers produced that range. **Routing** decides which calls run at all. **Tier** decides which model runs them. **Context** decides how many tokens each call carries; instructions and examples re-sent on every call are exactly what tuning moves into weights. The router is about 1% of the final figure and determines the other 99%. That makes precision the adoption metric and the labelled set the durable asset.

Precision pays twice. A wrong route costs the call and also produces a wrong output that someone has to read. At 20 outputs a week, 90% precision is two wrong ones every week, and 99% is one every five weeks. That is the difference between a tool people read and a tool people mute. Coverage and precision have to land together, though: fixing coverage alone means more calls, a larger bill, and more wrong outputs at once.

> **Warning:** Check how a tuned model is served before you trust a per-token number. Some platforms serve adapters only on dedicated deployments billed by GPU time. At $7 per GPU-hour, that is about $5,040 a month whether or not traffic arrives, and it changes the break-even completely.

## Tuning the router: a ladder

The tuned classifier in that example had to come from somewhere. Two choices are independent when you tune a router: how much of the model changes, and what signal it trains against. On the first axis:

| How much changes | What it is | Use when |
|------------------|-----------|----------|
| **LoRA** | Base frozen, small low-rank adapters trained; artifact in megabytes | Narrow specialisation |
| **Full-parameter** | Every weight updated; a full checkpoint | New capability or domain |

On the second axis:

| Signal | Needs | Example for a router |
|--------|-------|----------------------|
| **SFT** | Input and correct output pairs | Request and the route a person chose |
| **DPO** | Chosen and rejected pairs | The route that was overridden against its replacement |
| **RFT** | A scorer that returns a reward from 0 to 1 | Did the downstream task succeed on this route? |

Climb the ladder in order, and only when the harness shows the rung below has stopped paying. Start with a **prompted classifier**: no weights change and no training data is required, and it sets the baseline. Move to **SFT on historical labels** as soon as labelled routes exist. Then add **preference and reward tuning**, DPO on overrides and RFT on outcomes; these need live traffic, so they cannot be pulled forward.

RFT deserves a closer look. You supply prompts and an evaluator, and training increases the score. Gold completions are optional, which makes RFT the right fit when a task is verifiable and you lack reference answers. The evaluator can be rules, tests, a judge model, or a mix. The risk is reward hacking, where the model learns to satisfy the evaluator instead of the task, so inspect rollouts and stop when quality regresses or the evaluator saturates. A good eval harness is already most of a reward function.

Be honest about label quality. Labels that come from the subset of users who bother to label skew toward trained users, and some are wrong. Measure both before training on them.

## Evaluating a router

Everything above depends on measuring the router well. Five metrics cover what matters:

| Metric | Asks |
|--------|------|
| **Coverage** | Did a route run at all? |
| **Recall** | When a specialist was needed, was it selected? |
| **Precision** | When a route fired, was it correct? |
| **Latency** | Did the answer arrive while it was still useful? |
| **Cost per successful task** | Including retries, cache misses, and escalations |

A fair comparison follows a few rules. Compare against the human baseline, measured: if people route by hand today, the bar is their error rate, not perfection. Remember that agreement is not correctness; precision against human labels measures agreement, so send disagreements to adjudication. Evaluate whole sessions, held out from anything the router trained on. Version the policy: router version, threshold, candidate models, prompts, and fallback order are all part of the configuration under test, and an untracked policy change invalidates the comparison. And replay history first, since a harness that replays past traffic gives a baseline before any production change.

What survives a model swap is the harness, the labelled set, and the baseline. A tuned adapter is tied to a base model. Build the measurement first and keep it provider-agnostic.

## Summary

Evaluate a cascade using total cost and final task quality. State the assumptions before applying the result to a different workload.

## Review questions

1. Why evaluate the complete cascade instead of each model alone?

   <details><summary>Answer</summary>

   The routing decisions change which examples reach each model and add latency, retries, and selection error.

   </details>

[Chapter contents](README.md) · [Next: Routing implementations and selection](05-routing-implementations-and-selection.md)
