# Routing errors and cache locality

This part of [Model Routing and Cascades](README.md) develops the mechanisms below. Work the numerical examples or trace the code before checking the answer.

> **Learning objectives**
> - Account for routing uncertainty, granularity, and prefix reuse.

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

Granularity matters so much because of the prompt cache. Prompt caching can bill previously seen prefix tokens at a reduced rate, with eligibility and scope determined by the model, provider, and deployment. Switching models does not carry the old cache into the new endpoint, so a cold switch can cost more than it saves. The old cache may remain reusable if you return before it expires; a switch does not necessarily delete it.

Some illustrative arithmetic, with made-up round rates, shows how:

| Option for the next turn, 100K tokens of context | Input rate | Input cost |
|---|---|---|
| Stay on the strong model, cache hit | $0.50 per 1M (cached) | $0.05 |
| Switch to a cheap model, cache miss | $0.50 per 1M (uncached) | $0.05 |
| Switch back to the strong model later, cache expired | $5.00 per 1M (uncached) | $0.50 |

The cheap model saved nothing on input for that turn, and the round trip cost ten times the turn it tried to optimise. A router has to price the state it abandons, not only the model it selects. A router that ignores cache state optimises the sticker price and raises the bill. This is why session-level routers hold a choice across a turn, and why "cheapest model per request" is often the most expensive policy for long contexts.

To see it on real numbers, price one long session twice: once staying on a single model with a warm cache, and once switching models halfway, using real cached-input rates. In every evaluation, record cache hits, cache writes, the resolved model, and the resolved provider.

## Summary

Account for routing uncertainty, granularity, and prefix reuse. State the assumptions before applying the result to a different workload.

## Review questions

1. Can choosing the cheapest model increase end-to-end cost?

   <details><summary>Answer</summary>

   Yes. Retries, escalation, cache misses, longer responses, and errors can outweigh a lower per-token price.

   </details>

[Chapter contents](README.md) · [Next: Escalation economics and evaluation](04-escalation-economics-and-evaluation.md)
