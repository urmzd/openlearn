# LLM Evaluation

## Overview

- **Primary references**: [MacKay — *Information Theory, Inference, and Learning*](http://www.inference.org.uk/mackay/itila/) (free; cross-entropy, perplexity, compression), [Stanford HELM](https://crfm.stanford.edu/helm/) (free), [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (free)
- **Supplementary**: [Ragas docs](https://docs.ragas.io/) (RAG metrics, free), [Chatbot Arena / LMSYS paper](https://arxiv.org/abs/2403.04132) (free), [*The Pile* paper](https://arxiv.org/abs/2101.00027) (bits-per-byte, free), [Hendrycks MMLU](https://arxiv.org/abs/2009.03300) (free), and **[saige](https://github.com/urmzd/saige)**'s `eval/` package as a reference for a composable `Scorer` framework
- **Prerequisites**: [Information Theory](../../math/11-information-theory/) (entropy, KL divergence, cross-entropy), [Deep Learning](../../ml/02-deep-learning/) (the training loss), [Retrieval & RAG](../07-retrieval-and-rag/) (what RAG eval measures)
- **Estimated time**: 2-3 weeks at 8-10 hrs/week

## Key Takeaways

- **The training loss *is* an evaluation.** A language model is trained to minimize **cross-entropy** — the average bits needed to encode the next token. Lower loss = better prediction = better compression of text. Eval starts here, in information theory.
- **Perplexity, bits-per-token, and bits-per-byte are the same quantity in different clothes.** Perplexity is the exponential of cross-entropy (the effective branching factor); **bits-per-byte (bpb)** normalizes that loss per *byte* so you can compare models with different tokenizers.
- **Intrinsic ≠ extrinsic.** Low perplexity says the model predicts text well; it does *not* say the model is helpful, correct, or safe. You need task and behavioral evals on top.
- **Benchmarks measure capability; LLM-as-judge measures preference; humans are the ground truth.** Each is a different evaluator with different biases — triangulate, don't trust one.
- **RAG and agents need their own metrics.** Faithfulness/grounding and context precision/recall for RAG; task success, tool-call accuracy, and latency (TTFT/TTLT) for agents.
- **Eval is a pipeline you build, not a number you read.** A composable scorer framework (dataset × model × metric × aggregation) is platform infrastructure.

## How to Study

- Compute cross-entropy, perplexity, and bits-per-byte for a small model on a held-out text by hand from the per-token log-probs — prove to yourself they're one quantity.
- Take two models with *different tokenizers* and show why perplexity is unfair but bits-per-byte is comparable.
- Trace a tiny eval harness end to end in the [eval runner](09-eval-runner.md) and [scorers](10-scorers.md) chapters: a `Scorer` interface, a few metrics, and a dataset runner that scores a model and aggregates; predict each aggregate before reading the solution's.
- Set up an LLM-as-judge for a subjective task, then measure the judge's bias (position, verbosity) and its agreement with human labels.

## Agent evaluation solution chapters

| Module | Chapter |
|---|---|
| `ag.09` Eval runner | [Run suites and write reports](09-eval-runner.md) |
| `ag.10` Scorers | [Score text, retrieval, timing, tools, and state changes](10-scorers.md) |
| `ag.11` LLM judge | [Use rubrics and detect position bias](11-llm-judge.md) |
| `ag.12` A/B experiments | [Compare paired runs with confidence intervals](12-ab-experiments.md) |

---

# Concepts & Techniques

## Core Insight

Evaluating a generative model is hard because there's no single right answer to compare against. So evaluation splits into two fundamentally different questions. The first is **intrinsic**: how well does the model predict text? That has a clean, information-theoretic answer — cross-entropy / perplexity / bits-per-byte — and it's literally the training objective, which is why "scaling = lower loss" works. The second is **extrinsic**: is the output actually *good* — correct, helpful, grounded, safe? That has no closed form, so we approximate it with benchmarks, model judges, and humans, each a noisy proxy. Mature evaluation is knowing which question you're asking and never mistaking a low perplexity for a good product.

## 1. Cross-Entropy: The Loss That Is an Eval

**Why "bit loss" is the foundation**

**Key ideas**:
- A language model outputs a probability distribution over the next token. It's trained to minimize **cross-entropy loss**:

  `L = − (1/N) Σ_i log p(x_i | x_<i)`

  the average negative log-probability the model assigned to the *actual* next token. Use log base 2 and the unit is **bits**; natural log gives **nats**.
- **Information-theoretic meaning** (see [Information Theory](../../math/11-information-theory/)): cross-entropy `H(p, q) = −Σ p(x) log q(x)` is the average number of bits to encode samples from the true distribution `p` using the model's distribution `q`. The model's loss is the cross-entropy between real text and its predictions. **Minimizing loss = learning the true distribution = compressing text** — a better language model is literally a better compressor (Shannon).
- This is why eval *begins* at training: the loss curve is your first, cheapest, most honest signal.

## 2. Perplexity, Bits-per-Token, Bits-per-Byte

**One quantity, three normalizations — this is the "bp" the question is about**

| Metric | Definition | Reads as | Tokenizer-dependent? |
|--------|-----------|----------|----------------------|
| **Cross-entropy** | avg −log₂ p (bits/token) | bits to predict the next token | Yes |
| **Bits-per-token** | = cross-entropy in bits | same thing, named for the unit | Yes |
| **Perplexity (PPL)** | `2^(cross-entropy in bits)` = `exp(loss)` | "effective # of equally-likely choices" — the branching factor | Yes |
| **Bits-per-byte (bpb)** | total bits / total UTF-8 **bytes** of text | compression rate, tokenizer-free | **No** |
| **Bits-per-character (bpc)** | total bits / total characters | older char-level variant | No |

**Key ideas**:
- **Perplexity** is the most intuitive: PPL = 100 means the model is, on average, as uncertain as if choosing uniformly among 100 tokens. Lower is better.
- **The tokenizer problem**: perplexity and bits-per-token depend on *how text is split*. A model with a bigger vocabulary has fewer, "easier" tokens and looks better on per-token metrics — an unfair comparison.
- **Bits-per-byte fixes it**: divide the *total* cross-entropy (summed bits over the sequence) by the number of **bytes** of the underlying text, not the number of tokens. Now it's a tokenizer-independent compression rate you can compare across models (used by The Pile, language-modeling leaderboards). This is the "bit loss"/bpb metric — the rigorous way to compare base models.
- **Conversions**: `bpb = (loss_in_nats / ln 2) × (tokens / bytes)` — same loss, re-normalized from per-token to per-byte.

## 3. Intrinsic vs Extrinsic Evaluation

- **Intrinsic** (the above): measures *modeling quality* on held-out text. Cheap, automatic, great for pretraining and tracking scaling — but **blind to usefulness**. A model can have great perplexity and still be unhelpful, biased, or wrong.
- **Extrinsic**: measures *task performance* — does it answer correctly, follow instructions, stay grounded, refuse appropriately? No closed form, so we approximate with benchmarks, judges, and humans. **This is what users feel.**
- **The discipline**: use intrinsic metrics to steer training; use extrinsic metrics to decide if it's *good*. Never ship on perplexity alone.

## 4. Benchmarks & Capability Evals

**Key ideas**:
- **Task benchmarks**: MMLU (knowledge), GSM8K/MATH (reasoning), HumanEval/MBPP (code), HellaSwag (commonsense), and aggregators like **HELM** and the lm-evaluation-harness that run many tasks with standardized prompts/scoring.
- **Scoring modes**: exact match / multiple-choice log-likelihood (pick the highest-probability option), or generation + a checker.
- **Contamination** is the central threat: if benchmark data leaked into training, the score is memorization, not capability. Prefer fresh, private, or held-out evals; treat public-benchmark gains skeptically.
- **Capability ≠ deployment fit**: a high MMLU score doesn't mean the model is good at *your* task — build a domain eval set.

## 5. LLM-as-Judge & Preference Evaluation

**Using a model (or humans) to score open-ended outputs**

**Key ideas**:
- **LLM-as-judge**: prompt a strong model to grade an output (pointwise score) or pick the better of two (**pairwise**). Scalable and cheap; the default for subjective quality.
- **Judge biases to control**: **position** bias (favoring the first/second answer — swap and average), **verbosity** bias (longer looks better), **self-preference** (favoring its own family), and limited discrimination on hard cases. Always validate the judge against human labels.
- **Pairwise → ranking**: aggregate many pairwise verdicts into **Elo / Bradley-Terry** ratings (LMSYS **Chatbot Arena**). Humans voting on pairs is the closest thing to ground truth for general quality.
- **Rubrics & reference-guided judging**: give the judge criteria or a gold answer to reduce variance.

## 6. RAG Evaluation

**Separate the retriever from the generator** (ties to [Retrieval & RAG](../07-retrieval-and-rag/))

| Stage | Metric | Asks |
|-------|--------|------|
| Retrieval | **Context precision / recall**, **MRR**, **NDCG**, Recall@k | Did we fetch the right chunks, ranked well? |
| Generation | **Faithfulness / groundedness** | Is every claim supported by the retrieved context (no hallucination)? |
| Generation | **Answer relevance / correctness** | Does it actually answer, and match the truth? |

**Key ideas**:
- **Decompose the failure**: a bad RAG answer is either a *retrieval* miss (right context not fetched) or a *generation* miss (context fetched but ignored/contradicted). Faithfulness vs context-recall tells you which — so you fix the right stage.
- **Frameworks**: Ragas and similar use LLM-as-judge to compute these; **saige's `eval/` package** exposes exactly this set (ContextPrecision, ContextRecall, NDCG, MRR, Faithfulness, AnswerCorrectness) behind a composable `Scorer` interface.

## 7. Agent & System Evaluation

**Key ideas**:
- **Task success / goal completion**: did the agent accomplish the objective end-to-end (the metric that matters)?
- **Tool-call accuracy**: right tool, right arguments, valid schema; recovery from tool errors.
- **Trajectory metrics**: turn count, redundant steps, loop detection.
- **Operational metrics**: **TTFT** (time-to-first-token) and **TTLT** (time-to-last-token), cost per task, tokens per task — quality is meaningless if it's too slow or too expensive (ties to [Streaming](../03-streaming-sse/) and serving SLOs). saige's agent scorers track exactly these (TTFT/TTLT, tool success, turn count).

## 8. The Eval Harness Pattern

**Evaluation as platform infrastructure, not a notebook**

**Key ideas**:
- **Composable `Scorer` interface**: a metric is a pure function `(input, output, reference?) → score` ([a functional pattern](../06-coding-and-design-patterns/)); the harness runs `dataset × model × [scorers] → aggregated report`. saige's universal framework (an `Observation` + pluggable `Scorer`s across RAG/agent/KG) is this pattern.
- **Eval datasets are versioned assets**: curate, freeze, and review them like code; guard against contamination and drift.
- **Regression gating**: run the eval suite in CI on every model/prompt change; block on regressions — the eval analogue of unit tests ([Software Craftsmanship](../../software-craftsmanship/)).
- **Online + offline**: offline evals on fixed sets before ship; online signals (user feedback, A/B, acceptance rate) after — close the loop back to training data.

---

## Decision Cheat Sheet

| Question | Metric |
|----------|--------|
| How well does the base model predict text? | Cross-entropy / perplexity |
| Compare base models with different tokenizers | **Bits-per-byte (bpb)** |
| General capability | Benchmarks (MMLU, GSM8K, HumanEval) — watch contamination |
| Open-ended quality, at scale | LLM-as-judge (control position/verbosity bias) |
| General "which model is better" | Pairwise + Elo (human arena) |
| Did RAG retrieve the right context? | Context precision/recall, NDCG, MRR |
| Did RAG hallucinate? | Faithfulness / groundedness |
| Is the agent actually working? | Task success + tool accuracy + TTFT/TTLT |
| Ship-gate a change | Versioned eval suite in CI |

## Patterns Worth Internalizing

- **The loss is information, literally.** Cross-entropy = bits to predict the next token = compression; perplexity and bits-per-byte are just re-normalizations. Understand one and you understand all.
- **Normalize per byte to compare fairly** — tokenizer-dependent metrics flatter bigger vocabularies; bpb doesn't.
- **Intrinsic steers, extrinsic decides** — never confuse low perplexity with a good product.
- **Decompose before you judge** — separate retrieval from generation, capability from preference, quality from cost/latency.
- **Every evaluator is biased** — benchmarks leak, judges have position/verbosity bias, humans are noisy; triangulate.
- **Make eval a versioned, CI-gated harness** — a composable `Scorer` over frozen datasets, not a one-off notebook.

## Connections to Other Tracks

| Concept | Connected Track | Application |
|---------|-----------------|-------------|
| Entropy, cross-entropy, KL, compression | [Information Theory](../../math/11-information-theory/) | Why the loss measures bits |
| The training objective, scaling laws | [Deep Learning](../../ml/02-deep-learning/) | Loss as the first eval |
| Context precision/recall, faithfulness | [Retrieval & RAG](../07-retrieval-and-rag/) | Evaluating the pipeline |
| TTFT/TTLT, cost, serving SLOs | [LLM Systems](../../ml/04-llm-systems/) / [Streaming & SSE](../03-streaming-sse/) | Operational eval |
| Scorer as a pure function, composition | [Coding & Design Patterns](../06-coding-and-design-patterns/) | The harness design |
| Eval suites as tests, CI gating | [Software Craftsmanship](../../software-craftsmanship/) | Regression-gating model changes |
| Grading an agent that changes state | [Agent Evaluation Harness](../../../case-studies/05-agent-eval-harness/) | Replay, set-based checks, oracles and mutants in a worked build |

## How Companies Apply These Patterns

| Company | The pattern they lean on | Instance |
|---------|--------------------------|----------|
| OpenAI / Anthropic | Held-out loss + capability + safety evals | bits-per-byte, internal evals, red-teaming |
| LMSYS | Human pairwise → Elo | Chatbot Arena |
| Stanford CRFM | Standardized multi-metric benchmarking | HELM |
| EleutherAI | Reproducible benchmark harness | lm-evaluation-harness |
| Ragas / eval vendors | LLM-as-judge for RAG metrics | Faithfulness, context precision/recall |
| saige (reference) | Composable `Scorer` across subsystems | RAG/agent/KG metrics |

## Chapters

| # | Module | Chapter | Kind | Pass |
|---|---|---|---|---|
| 1 | `ag.09` | [Eval runner](09-eval-runner.md) | build | 10 |
| 2 | `ag.10` | [Agent evaluation scorers](10-scorers.md) | build | 10 |
| 3 | `ag.11` | [LLM judge with position control](11-llm-judge.md) | build | 10 |
| 4 | `ag.12` | [Paired A/B experiments](12-ab-experiments.md) | build | 10 |
