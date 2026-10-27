# Routing decisions and oracle bounds

This part of [Model Routing and Cascades](README.md) develops the mechanisms below. Work the numerical examples or trace the code before checking the answer.

> **Learning objectives**
> - Separate request routing, model selection, and expert routing.

## Three decisions that share a name

Routing is three decisions, not one. Products blur them; keep them apart when you evaluate.

| Decision | Question | Typical owner |
|----------|----------|---------------|
| **Model selection** | Which model should answer this request? | A router: rules, a classifier, or a learned scorer |
| **Provider selection** | Which endpoint serves the chosen model? | A gateway, on price, availability, or latency |
| **Fallback** | What runs after an error or timeout? | An ordered list supplied by the caller |

Pinning a model name does not pin the provider, the quantization, or the cache. Using a gateway does not mean a learned router chose the model. When someone says "we use routing," ask which of the three they mean.

For where these decisions sit in a full serving stack, and how gateways, orchestrators, and inference engines differ, see [LLM Serving Platforms](../20-inference-engines/05-serving-platforms.md). This chapter covers the decision itself.

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

A retry loop is a cascade in time. Returning an error to the same model and retrying, as in the [Grounded SQL Agent](../30-case-study-sql-agent/) case study, escalates effort, where a cascade escalates tier. Both need a hard budget.

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

## Summary

Separate request routing, model selection, and expert routing. State the assumptions before applying the result to a different workload.

## Review questions

1. What does an oracle router establish?

   <details><summary>Answer</summary>

   It establishes an upper bound under its candidate models, labels, and cost assumptions, not the performance of a deployable router.

   </details>

[Chapter contents](README.md) · [Next: Routing errors and cache locality](03-routing-errors-and-cache-locality.md)
