# Routing implementations and selection

This part of [Model Routing and Cascades](README.md) develops the mechanisms below. Work the numerical examples or trace the code before checking the answer.

> **Learning objectives**
> - Distinguish an implementation example from evidence for a new workload.

## A worked product: FireRouter

To see how these ideas appear in a shipping product, consider FireRouter, a managed router from Fireworks, reviewed from its public documentation:

| Aspect | Documented behaviour |
|--------|----------------------|
| Granularity | Chooses a model for each new user turn; that model handles the turn, including its tool calls |
| Model IDs | `firerouter`, a family alias such as `firerouter/opus`, or an exact pin such as `firerouter/claude-opus-5-5` |
| Custom routes | Up to eight models, `firerouter/{model1}/{model2}/...`; the first is primary |
| Preferences | `max-intelligence`, `more-intelligence`, `balanced`, `more-savings`, `max-savings`; support varies by harness |
| Reporting | The response `model` field names the model that served the request |
| Billing | You pay for the model that served each turn; closed models bill through your own provider credential; cached input bills at the serving model's cached rate |

What is not published, as reviewed, is the features the router uses, its training data, and a measured comparison against the oracle. Read it as an instance of a session-level, cache-aware router, and apply the evaluation rules from the previous section before adopting it. Notice the design choices it makes against the earlier sections: per-turn granularity, cache awareness, and a user-set preference in place of a hidden threshold.

## System One decision models

A decision is not a generation. Routing, gating, and scoring all have a closed answer set, and a model that returns a typed choice with a probability fits that job better than one that writes text you then parse. A new category of models is built on that observation. It was named on September 15, 2026, when TypeSafe AI announced Jev, and the name borrows Kahneman's fast, intuitive System 1. The category was days old at the time of this review, so treat every number here as vendor-reported and unreplicated.

A **decision model** takes content plus a typed question and returns a typed answer, a probability for every option, and a confidence value. It does not write text; it produces all outputs in one forward pass instead of a token-by-token loop. It offers three primitives:

| Primitive | Question shape | Returns |
|-----------|----------------|---------|
| **Choice** | Pick one of a listed set, up to 255 options | The selection and a probability per option |
| **Score** | Place content on an ordered scale of 2 to 10 levels | A probability-weighted position |
| **Noul** | Yes or no | One probability from 0 to 1 |

It belongs in a discussion of routing because every router family above is a decision with a closed answer set. There are three ways to make that decision, in order of how much machinery runs:

| Approach | What runs | What you get |
|----------|-----------|--------------|
| LLM, free text | Full generation | A string to parse and validate |
| LLM, structured output | Generation constrained to a schema | A valid shape; correctness not guaranteed |
| Decision model | One pass, no generation | A typed value and a probability to threshold on |

Structured output guarantees the shape, but it does not give a trained, calibrated probability. The claimed advantage of a decision model lives in the probabilities, not in the types.

Calibration is the product. Jev is described as trained with RLCD, a method that rewards honest probabilities: when the model says 0.8, it should be right about 80% of the time. Calibration is what makes confidence escalation workable: act above a threshold, escalate below it. Measure it yourself with a reliability plot on your own labelled data before trusting a threshold.

A decision model has clear limits. It cannot invent a value outside your option list, but it can pick the wrong one. Documented weak spots for Jev 1.13 include arithmetic and counting, dates treated as text instead of ordered quantities, and literal reading of questions, where scoping words and double negatives change the answer. It reads supplied text as data, and prompt injection is listed as a weakness, so the escalation rule still applies: on untrusted input, category rules outrank any probability.

An independent directory lists the field as follows:

| Model | Vendor | Announced | Access |
|-------|--------|-----------|--------|
| Jev 1.13 | TypeSafe AI | Sep 15, 2026 | Hosted API |
| Kev (0.8B, 4B, 9B) | Jared Palmer | Sep 17, 2026 | Apache 2.0 |
| Laya | Convai Innovations | Sep 18, 2026 | Apache 2.0 |
| Tev1-4B | Together AI | Sep 23, 2026 | Hosted API |
| CLM-8B | Contrastive-LM | Sep 23, 2026 | Apache 2.0 |
| GLiNER2.5-Decide (340M) | Fastino Labs | Sep 24, 2026 | Apache 2.0 |

The reported figures for Jev are $0.042 per million input tokens with output free, and 70 to 500 ms per decision. TypeSafe has not published Jev's architecture, its training data, or the RLCD method. The open reproductions use their own training recipes, so they test the interface, not the method.

To evaluate one, use the same harness as for any router, plus calibration. Compare against two baselines you already own: a small LLM with structured output, and a fine-tuned encoder classifier. If a 300M-parameter classifier trained on your labels matches it, you did not need a new category.

## What goes in the pool: specialised models

A router is only as good as its candidates. The open-weight-only oracle reached 90.3%, which says the cheap tier is no longer a fallback. The pool is filling with specialised models: open bases post-trained on a verifiable reward, sitting beside general frontier models and often the cheap tier worth routing to. The figures below are vendor-reported, from Fireworks' own posts:

| Model | Base | How it was built | Reported result |
|-------|------|------------------|-----------------|
| **Ember-1** (research preview, Sep 23, 2026) | Kimi K3 | Over 50 training experiments and 200 evaluations | About 40% fewer tokens than the base at comparable quality; 75.2% against 66.4% on DeepSWE 1.1 |
| **Gen-1 Slides** (with Genspark) | MiniMax M3 | SFT on curated decks, then RL on a curriculum of growing context length | Low-rated decks cut from 18% to 3.6%; about 90% lower cost per deck than Opus 5 |
| **dfs-large1** (depthfirst) | GLM 5.2 | Post-trained with RL | Reported as a new Pareto frontier on a security benchmark |
| **FARE-20B** | gpt-oss-20B | Rejection-sampled SFT on a multi-domain set | A multi-task evaluator: pairwise, step-level, reference-based and reference-free scoring |
| **FireFunction** | Open base | Additional training for tool use | Function calling |

Four patterns repeat across that table. **SFT comes first and RL second**: supervised data teaches the format, and a reward teaches judgment, which is the tuning ladder applied to the worker instead of the router. **Token efficiency is a training target**: Ember-1 optimises how many tokens a task consumes, not only whether it succeeds, so cost per successful task falls even when accuracy is flat. **The evaluator is a model too**: FARE-20B exists because every reward and every eval needs a scorer, and a tuned evaluator is infrastructure. And **long-horizon RL is an engineering problem**: the Gen-1 write-up names sequence-level rewards, alignment of tokenization between inference and training, and numerical stability on episodes over 100,000 tokens.

Benchmark scores overstate deployed value. The Specialized Intelligence Index cites a METR review of 296 AI-generated pull requests in which maintainer acceptance averaged 24.2 points below the automated benchmark score. The [Grounded SQL Agent](../30-case-study-sql-agent/) case study makes the same point: a score is a capability claim, and a business outcome is a separate measurement.

> **Warning:** Check how a specialised model is served. Several of these are on-demand only, which means dedicated capacity billed by GPU time. The GPU-hour warning from the economics discussion applies before any of them enters a per-token cost model.

## Choosing an approach

When you face a routing decision, this mapping from situation to approach summarizes the chapter:

| Situation | Reach for |
|-----------|-----------|
| A deterministic feature already separates the traffic | Rules |
| Most requests are easy, a few need a strong model | Predictive router with a tuned threshold |
| A cheap check can verify the answer | Verifier-gated cascade |
| Long agent sessions with large context | Session-level routing; price the cache |
| Untrusted content decides the route | Category escalation, never confidence alone |
| No labelled routes yet | Prompted classifier plus a replay harness |
| Labelled routes exist | SFT on them, LoRA first |
| A verifiable outcome and no gold answers | RFT with your eval as the reward |
| A vendor shows an oracle chart | Ask for the deployed router's numbers on matched tasks |
| A closed-set decision at high volume, thresholded in code | A decision model or a tuned classifier; verify calibration |
| A narrow task with a verifiable reward and steady volume | A specialised post-trained model in the pool |

## Where these ideas show up in practice

Fireworks runs session-level, cache-aware routing across open and closed models in FireRouter. OpenRouter's Auto Router is a gateway with provider selection, fallbacks, and optional model selection. LMSYS and Anyscale built preference-trained predictive routing in RouteLLM, and Stanford's FrugalGPT is the learned cascade with a scoring function. TypeSafe AI and Together AI ship typed decisions with calibrated probabilities in Jev and Tev1. Fireworks and its partners take an open base through SFT then RL against a domain reward, as in Ember-1 and Gen-1 Slides. And almost any agent product uses category escalation and fail-open fan-out, with specialised subagents behind a classifier.

The chapter also connects to the rest of the library. [LLM Evaluation](../29-evaluation/) supplies the harness design, precision and recall, and judges for measuring a routing policy. [LLM Systems and Inference](../20-inference-engines/) explains through prefill, the KV cache, and serving cost why a switch is expensive, and [LLM Serving Platforms](../20-inference-engines/05-serving-platforms.md) maps the gateways, engines, provider selection, and reproducibility concerns a routed request passes through. [Training and Frameworks](../16-training-frameworks/) covers the SFT, DPO, LoRA, and small models used to tune the router and the tiers. [Retrieval and RAG](../27-retrieval-and-rag/) matters because routing across hundreds of specialists is a ranking problem. The [Grounded SQL Agent](../30-case-study-sql-agent/) shows deterministic gates and bounded retry, verifier-gated escalation in a real build, and the [Agent Evaluation Harness](../32-case-study-agent-evals/) shows a checker proved by oracles and mutants, which is what a reward function has to survive before RFT trains on it. [Authorization and Access Control](../../software-engineering/10-authorization/) explains, through least privilege and untrusted input, why confidence is not a control.

## Key takeaways

- Routing is three decisions: which model answers, which endpoint serves it, and what happens after a failure.
- The oracle proves opportunity, not a router; upper bounds, best-case benchmarks, and vendor charts are all statements about opportunity.
- Real routers fail mainly on model recall, not difficulty estimation; measure before you route, against the best single model, a random split, and a simple rule.
- Most of the value is in two or three complementary, curated models.
- A model switch abandons the prompt cache; price the state you abandon.
- Put the control outside the model: escalate on category or a verifier, and choose the failure direction explicitly.
- The router is a small share of the cost and decides the rest, so precision is the adoption metric and the labelled set is the durable asset.
- Match the machinery to the answer space: for a closed set, decide; for an open set, generate. Do not pay for generation to pick one of six labels.

## Key terms

- **Model selection**: the routing decision of which model should answer a request, made by rules, a classifier, or a learned scorer; distinct from choosing the endpoint that serves the model and from fallback after an error.
- **Provider selection**: the routing decision of which endpoint serves an already chosen model, usually made by a gateway on price, availability, or latency; pinning a model name does not pin the provider.
- **Oracle router**: a policy that assigns each task to the model with the best measured outcome after seeing every result, breaking ties on cost; it is an upper bound on what routing could gain, not a router that can be deployed.
- **Success matrix**: a table of task-by-model outcomes on an eval set, from which the best single model, the oracle, and the best pair of models can be computed to size the routing opportunity.
- **Model recall**: how often a router selects the model that would have succeeded when that model was in the pool; the main reason real routers fall short of the oracle.
- **Verifier-gated cascade**: a cascade that escalates to a stronger model only when a deterministic check, such as passing tests or a validating schema, rejects the cheap model's output.
- **Session-level routing**: choosing a model once per task or user turn and holding it, which keeps tool loops coherent and the prompt cache warm at the cost of not correcting mid-turn.
- **Prompt caching**: a provider feature that bills previously seen prompt-prefix tokens at a reduced rate; caches are per model and usually per provider, so switching models discards them.
- **Cache-aware routing**: routing that prices the prompt cache a model switch would abandon, since a full uncached prefill on the new model can cost more than the switch saves.
- **Escalation policy**: the written rule for when a request leaves the cheap path, triggered by confidence, a named category, a verifier, or a human override; category and verifier triggers resist manipulation by untrusted input.
- **Fail open**: on an error in a control such as a router, let everything run; suited to review or analysis fan-out, where the worst case is a known cost rather than silent loss of coverage.
- **Reinforcement fine-tuning (RFT)**: tuning a model against an evaluator that returns a reward from 0 to 1 for each output, with gold completions optional; it fits tasks that are verifiable but lack reference answers.
- **Reward hacking**: a model learning to satisfy its evaluator or reward function instead of the intended task, caught by inspecting rollouts and stopping when quality regresses.
- **Decision model / System One model**: a model that takes content plus a typed question and returns one of a closed set of answers with a calibrated probability per option in a single forward pass, without generating text.
- **Cost per successful task**: total spend, including retries, cache misses, and escalations, divided by the number of tasks completed correctly; the cost figure that a routing comparison should report.

## Further reading

- The [RouteLLM paper](https://arxiv.org/abs/2406.18665) and the [LMSYS write-up](https://www.lmsys.org/blog/2024-07-01-routellm/) (free).
- [FrugalGPT](https://arxiv.org/abs/2305.05176) (free).
- [LLMRouterBench](https://arxiv.org/abs/2601.07206) (free; Findings of ACL 2026).
- The [FireRouter documentation](https://docs.fireworks.ai/ecosystem/firerouter/overview) and Fireworks' [*The Frontier Isn't a Model, It's a Router*](https://fireworks.ai/blog/the-frontier-isnt-a-model-its-a-router).
- [OpenRouter Auto Router](https://openrouter.ai/docs/guides/routing/routers/auto-router).
- [Fireworks reinforcement fine-tuning documentation](https://docs.fireworks.ai/fine-tuning/reinforcement-fine-tuning-models).
- [TypeSafe System One documentation](https://docs.typesafe.ai/concepts/system-one) and [System One Models](https://systemonemodels.org/), an independent directory.
- Fireworks' [Specialized Intelligence Index](https://fireworks.ai/blog/introducing-the-specialized-intelligence-index).

## Summary

Distinguish an implementation example from evidence for a new workload. State the assumptions before applying the result to a different workload.

## Review questions

1. What must be remeasured when the candidate model pool changes?

   <details><summary>Answer</summary>

   The quality labels, cost assumptions, routing accuracy, latency, fallback rate, and cache behavior.

   </details>

[Chapter contents](README.md)
