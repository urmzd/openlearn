# Model Routing and Cascades

No single model is best at everything, and prices differ by more than an order of magnitude. If you could always send each request to the cheapest model that would succeed, you would get higher accuracy than the best model at a fraction of its cost. That statement is true and nearly useless, because it describes an oracle. Routing is the discipline of approximating the oracle with information available before the answer exists, and of knowing how much of the gap you actually closed.

This chapter builds that discipline in layers. It starts by separating the three decisions that all get called "routing", surveys the families of routers, and then uses an oracle analysis to show how much opportunity exists, followed by an independent benchmark that shows how little of it real routers capture and why. From there it turns to the practical forces that decide whether routing pays: the granularity of the decision, the prompt cache a switch abandons, the escalation policy, and the token economics. It closes with how to tune and evaluate a router, a documented product as a worked example, the new category of decision models, and the specialised models that increasingly fill the candidate pool.

> **Note:** Sources were reviewed on September 26, 2026. Vendor numbers and product behaviour change; treat every figure in this chapter as dated.

> **Learning objectives**
> - Distinguish model selection, provider selection, and fallback, and identify which one a product actually performs.
> - Build a success matrix, compute best single model, oracle, and best pair, and explain why the oracle is an upper bound rather than a router.
> - Price a model switch against the prompt cache it abandons.
> - Write an escalation policy that uses categories and verifiers on untrusted input and chooses its failure direction explicitly.
> - Evaluate a router against the best single model, a random split at equal cost, and a simple rule, on whole sessions.
> - Design provider adapters and cache-aware switching around an explicit application contract.

## Before you begin

Routing is measured, so you need the harnesses and precision and recall from [LLM Evaluation](../29-evaluation/). The cost arguments depend on prefill, the KV cache, and prompt caching from [LLM Systems and Inference](../20-inference-engines/) in *AI Engineering*, and tuning the router uses SFT and LoRA from [Training and Frameworks](../16-training-frameworks/).

## In this chapter

1. [Designing a model router across APIs, caches, and applications](01-router-api-cache-application-design.md): define the application contract and compatible execution plans.
2. [Routing decisions and oracle bounds](02-routing-decisions-and-oracles.md): Separate request routing, model selection, and expert routing.
3. [Routing errors and cache locality](03-routing-errors-and-cache-locality.md): Account for routing uncertainty, granularity, and prefix reuse.
4. [Escalation economics and evaluation](04-escalation-economics-and-evaluation.md): Evaluate a cascade using total cost and final task quality.
5. [Routing implementations and selection](05-routing-implementations-and-selection.md): Distinguish an implementation example from evidence for a new workload.
