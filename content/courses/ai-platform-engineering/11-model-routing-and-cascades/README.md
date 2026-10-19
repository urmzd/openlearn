# Model Routing & Cascades

No single model is best at everything, and prices differ by more than an order of magnitude. If you could always send each request to the cheapest model that would succeed, you would get higher accuracy than the best model at a fraction of its cost. That statement is true and nearly useless, because it describes an oracle. Routing is the discipline of approximating the oracle with information available before the answer exists, and of knowing how much of the gap you actually closed.

This chapter builds that discipline in layers. It starts by separating the three decisions that all get called "routing", surveys the families of routers, and then uses an oracle analysis to show how much opportunity exists, followed by an independent benchmark that shows how little of it real routers capture and why. From there it turns to the practical forces that decide whether routing pays: the granularity of the decision, the prompt cache a switch abandons, the escalation policy, and the token economics. It closes with how to tune and evaluate a router, a documented product as a worked example, the new category of decision models, and the specialised models that increasingly fill the candidate pool.

> **Note:** Sources were reviewed on September 26, 2026. Vendor numbers and product behaviour change; treat every figure in this chapter as dated.

> **Learning objectives**
> - Distinguish model selection, provider selection, and fallback, and identify which one a product actually performs.
> - Build a success matrix, compute best single model, oracle, and best pair, and explain why the oracle is an upper bound rather than a router.
> - Price a model switch against the prompt cache it abandons.
> - Write an escalation policy that uses categories and verifiers on untrusted input and chooses its failure direction explicitly.
> - Evaluate a router against the best single model, a random split at equal cost, and a simple rule, on whole sessions.

## Before you begin

Routing is measured, so you need the harnesses and precision and recall from [LLM Evaluation](../09-llm-evaluation/). The cost arguments depend on prefill, the KV cache, and prompt caching from [LLM Systems and Inference](../../machine-learning/04-llm-systems/) in *Machine Learning*, and tuning the router uses SFT and LoRA from [Training and Frameworks](../01-training-and-frameworks/).

## Three decisions that share a name

Routing is three decisions, not one. Products blur them; keep them apart when you evaluate.

| Decision | Question | Typical owner |
|----------|----------|---------------|
| **Model selection** | Which model should answer this request? | A router: rules, a classifier, or a learned scorer |
| **Provider selection** | Which endpoint serves the chosen model? | A gateway, on price, availability, or latency |
| **Fallback** | What runs after an error or timeout? | An ordered list supplied by the caller |

Pinning a model name does not pin the provider, the quantization, or the cache. Using a gateway does not mean a learned router chose the model. When someone says "we use routing," ask which of the three they mean.

For where these decisions sit in a full serving stack, and how gateways, orchestrators, and inference engines differ, see [LLM Serving Platforms](../../machine-learning/04-llm-systems/serving-platforms.md). This chapter covers the decision itself.

## Router families

Within model selection, routers come in five families that differ in when they decide and what they pay to decide:

| Family | Decides | Signal | Extra cost | Main weakness |
|--------|---------|--------|------------|---------------|
| **Rules and categories** | Before generation | Deterministic features: file paths, tool names, request type | None | Rules drift and do not generalise |
| **Predictive router** | Before generation | A model scores the prompt | One small inference | Misjudges prompts unlike its training data |
| **Cascade** | After generation | A scorer judges the cheap answer, then escalates | The cheap attempt is paid even when discarded | Latency, and a scorer that can be wrong |
| **Verifier-gated** | After generation | A deterministic check: tests pass, query executes, schema validates | The cheap attempt plus the check | Only works where a verifier exists |
| **Session-level router** | Per task or per user turn | Task features plus cache state | One decision per turn | Early choices shape every later step |

RouteLLM is the standard example of **predictive routing**. A router estimates how much a strong model would beat a weak one on this prompt, and a threshold turns that score into a choice. The project trained four routers on Chatbot Arena preference data: similarity-weighted ranking, matrix factorization, a BERT classifier, and a causal LLM classifier. Against a GPT-4 baseline at 95% of its quality, the reported cost reductions were over 85% on MT Bench, 45% on MMLU, and 35% on GSM8K. The spread is the lesson: savings depend on the workload. Routers trained only on Arena data performed poorly on MMLU until about 1,500 in-domain examples were added, under 2% of the training set.

FrugalGPT is the standard example of a **cascade**. It calls models in order from cheap to expensive and stops when a scoring function accepts the answer. The paper reports matching GPT-4 with up to 98% cost reduction, or 4% better accuracy at equal cost, on its evaluated tasks. A cascade pays for every rejected attempt, so it wins only when the cheap model's answer is accepted most of the time.

A retry loop is a cascade in time. Returning an error to the same model and retrying, as in the [Grounded SQL Agent](../../case-studies/02-grounded-sql-agent/) case study, escalates effort, where a cascade escalates tier. Both need a hard budget.

## The oracle and what it proves

To know whether any of these families is worth building, you need to know how much there is to gain. An oracle router assigns each task to the model with the best measured outcome, breaking ties on cost. It needs the answer key, so it is an upper bound and nothing more. The oracle proves opportunity, not a router: it picks the best model after seeing every outcome, while a real router must choose before, and the distance between them is the whole engineering problem.

Fireworks published an oracle analysis on DeepSWE v1.1: 113 agentic software-engineering tasks, 18 models, and four rollouts per pair. The headline policies compare as follows:

| Policy | Success | Cost per task |
|--------|---------|---------------|
| Best fixed model (GPT-6 Astra) | 74.1% | $6.52 |
| Claude Opus 5, fixed | 73.8% | $11.84 |
| Oracle over open-weight models only | 90.3% | $1.45 |
| Oracle over all 18 models | 97.6% | $1.88 |

The same analysis reported portfolio results: the best pair added 13.1 points over the best single model, the best trio reached 91.2%, and going from three models to eighteen added only 6.4 more points.

Three things follow. First, **complementarity is real**: models fail on different tasks, so the union of a pool covers far more than its best member. Second, **curation beats count**: most of the gain arrives by the third model, so most of the value is in two or three complementary models. Third, **none of this measures a router**. The analysis modelled the simplest policy, one model chosen at the start of a task and held. Whether a deployed router captures the opportunity is a separate measurement.

The same post reports production numbers: 2,334 internal coding sessions over four weeks at $7.42 per session, against $15.81 for a fixed Claude Opus 5, a 53% reduction. That is a cost comparison. Before treating any routing claim as a quality claim, ask for task success on the same sessions.

You can run this analysis on your own data. Take an eval set you already have and run three models over it. Build the success matrix (task by model), then compute the best single model, the oracle, and the best pair by hand. The gap between the first two is your opportunity.

## Why real routers fall short

The oracle shows the ceiling; the next question is how close real routers get. LLMRouterBench re-evaluated the field under one framework, with over 400K instances, 21 datasets, 33 models, and 10 routing baselines, and its findings are the counterweight to every oracle chart. Many routing methods perform about the same once they are evaluated identically. Several recent approaches, including commercial routers, fail to reliably beat the **best single model**. A large gap to the oracle persists, driven mainly by **model-recall failures**: the right model was in the pool and the router did not choose it. And the embedding backbone matters little, while larger pools show diminishing returns against careful curation.

The common failure, then, is model recall, not difficulty estimation. Predicting that a prompt is hard does not tell you which model handles it. A router needs to learn what each model is good at, which is why in-domain labelled outcomes matter more than general preference data.

Every router must beat three baselines: the best single model, a random split at the same cost, and a rule a person could write in ten minutes. If it does not beat all three on your traffic, it is complexity without a return. A good first exercise is to write the simplest router that could work, a rule on one input feature, and measure how often it selects the non-default model when that model was the only one that would have succeeded. That number is model recall.

## Granularity

Once you decide to route, you must decide what unit of work each decision covers. The three choices trade flexibility against cache reuse and consistency:

| Unit | Example | Trade-off |
|------|---------|-----------|
| Per request | Classify each API call | Most flexible; breaks cache and consistency in multi-step work |
| Per user turn | One model handles the turn and its tool calls | Keeps a tool loop coherent; cannot correct mid-turn |
| Per task or session | Pick once, hold | Best cache reuse; a wrong first choice is expensive |

In agent workloads, early decisions change later inputs. A model that reads the wrong file produces a different context for every following step, so per-step accuracy numbers do not compose; evaluate complete trajectories. A router that scores only the last message has a specific blind spot here: when that message is "continue" or a tool result, it carries almost no signal about the task.

## Cache-aware routing

Granularity matters so much because of the prompt cache. Prompt caching bills previously seen prefix tokens at a reduced rate, and caches are per model and usually per provider. Switching models discards the cache and pays full prefill on the new one, so a model switch can cost more than it saves.

Some illustrative arithmetic, with made-up round rates, shows how:

| Option for the next turn, 100K tokens of context | Input rate | Input cost |
|---|---|---|
| Stay on the strong model, cache hit | $0.50 per 1M (cached) | $0.05 |
| Switch to a cheap model, cache miss | $0.50 per 1M (uncached) | $0.05 |
| Switch back to the strong model later, cache expired | $5.00 per 1M (uncached) | $0.50 |

The cheap model saved nothing on input for that turn, and the round trip cost ten times the turn it tried to optimise. A router has to price the state it abandons, not only the model it selects. A router that ignores cache state optimises the sticker price and raises the bill. This is why session-level routers hold a choice across a turn, and why "cheapest model per request" is often the most expensive policy for long contexts.

To see it on real numbers, price one long session twice: once staying on a single model with a warm cache, and once switching models halfway, using real cached-input rates. In every evaluation, record cache hits, cache writes, the resolved model, and the resolved provider.

## Escalation policy

Every routing design needs a rule for leaving the cheap path. That rule is the escalation policy, and it is worth writing down before you build anything, including what happens when the router itself errors. There are four ways to trigger escalation, and they fail differently:

| Trigger | How it works | Fails when |
|---------|--------------|------------|
| **Confidence** | Escalate when the model or a scorer reports low confidence | Confidence is miscalibrated, or the input manipulates it |
| **Category** | Named categories always escalate, whatever the model thinks | The category list is incomplete |
| **Verifier** | A deterministic check rejects the output | No verifier exists for the task |
| **Human override** | A person corrects the route | Volume exceeds attention |

Escalate on category or on a verifier, not on model confidence alone. Confidence is produced by the very thing you are trying to check, and untrusted input can move it. If the content being routed can contain instructions, a confidence score is attacker-influenced. A rule such as "security and schema changes always get the strong reviewer" cannot be argued with, because no model evaluates it. This is the same principle as the deterministic gate in the [Grounded SQL Agent](../../case-studies/02-grounded-sql-agent/): the control sits outside the model.

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

Benchmark scores overstate deployed value. The Specialized Intelligence Index cites a METR review of 296 AI-generated pull requests in which maintainer acceptance averaged 24.2 points below the automated benchmark score. The [Grounded SQL Agent](../../case-studies/02-grounded-sql-agent/) case study makes the same point: a score is a capability claim, and a business outcome is a separate measurement.

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

The chapter also connects to the rest of the library. [LLM Evaluation](../09-llm-evaluation/) supplies the harness design, precision and recall, and judges for measuring a routing policy. [LLM Systems and Inference](../../machine-learning/04-llm-systems/) explains through prefill, the KV cache, and serving cost why a switch is expensive, and [LLM Serving Platforms](../../machine-learning/04-llm-systems/serving-platforms.md) maps the gateways, engines, provider selection, and reproducibility concerns a routed request passes through. [Training and Frameworks](../01-training-and-frameworks/) covers the SFT, DPO, LoRA, and small models used to tune the router and the tiers. [Retrieval and RAG](../07-retrieval-and-rag/) matters because routing across hundreds of specialists is a ranking problem. The [Grounded SQL Agent](../../case-studies/02-grounded-sql-agent/) shows deterministic gates and bounded retry, verifier-gated escalation in a real build, and the [Agent Evaluation Harness](../../case-studies/05-agent-eval-harness/) shows a checker proved by oracles and mutants, which is what a reward function has to survive before RFT trains on it. [Authorization and Access Control](../08-authorization-and-access-control/) explains, through least privilege and untrusted input, why confidence is not a control.

## Key takeaways

- Routing is three decisions: which model answers, which endpoint serves it, and what happens after a failure.
- The oracle proves opportunity, not a router; upper bounds, best-case benchmarks, and vendor charts are all statements about opportunity.
- Real routers fail mainly on model recall, not difficulty estimation; measure before you route, against the best single model, a random split, and a simple rule.
- Most of the value is in two or three complementary, curated models.
- A model switch abandons the prompt cache; price the state you abandon.
- Put the control outside the model: escalate on category or a verifier, and choose the failure direction explicitly.
- The router is a small share of the cost and decides the rest, so precision is the adoption metric and the labelled set is the durable asset.
- Match the machinery to the answer space: for a closed set, decide; for an open set, generate. Do not pay for generation to pick one of six labels.

## Further reading

- The [RouteLLM paper](https://arxiv.org/abs/2406.18665) and the [LMSYS write-up](https://www.lmsys.org/blog/2024-07-01-routellm/) (free).
- [FrugalGPT](https://arxiv.org/abs/2305.05176) (free).
- [LLMRouterBench](https://arxiv.org/abs/2601.07206) (free; Findings of ACL 2026).
- The [FireRouter documentation](https://docs.fireworks.ai/ecosystem/firerouter/overview) and Fireworks' [*The Frontier Isn't a Model, It's a Router*](https://fireworks.ai/blog/the-frontier-isnt-a-model-its-a-router).
- [OpenRouter Auto Router](https://openrouter.ai/docs/guides/routing/routers/auto-router).
- [Fireworks reinforcement fine-tuning documentation](https://docs.fireworks.ai/fine-tuning/reinforcement-fine-tuning-models).
- [TypeSafe System One documentation](https://docs.typesafe.ai/concepts/system-one) and [System One Models](https://systemonemodels.org/), an independent directory.
- Fireworks' [Specialized Intelligence Index](https://fireworks.ai/blog/introducing-the-specialized-intelligence-index).
