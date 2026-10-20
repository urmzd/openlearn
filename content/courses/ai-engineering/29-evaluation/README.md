# LLM Evaluation

Evaluating a generative model is hard because there is no single right answer to compare against. So evaluation splits into two fundamentally different questions. The first is **intrinsic**: how well does the model predict text? That has a clean, information-theoretic answer (cross-entropy, perplexity, bits-per-byte), and it is literally the training objective, which is why "scaling means lower loss" works. The second is **extrinsic**: is the output actually *good*, meaning correct, helpful, grounded, and safe? That has no closed form, so we approximate it with benchmarks, model judges, and humans, each a noisy proxy. Mature evaluation is knowing which question you are asking and never mistaking a low perplexity for a good product.

This chapter starts with the intrinsic side, because the training loss is the first and cheapest evaluation you have, and then moves outward to benchmarks, LLM-as-judge, RAG metrics, and agent metrics. It ends by treating evaluation as what it really is on a platform: a pipeline you build, not a number you read. The four sections that follow the overview build that pipeline for agents in Go, from the runner that executes suites to the paired experiments that tell you whether a change helped.

> **Learning objectives**
> - Compute cross-entropy, perplexity, and bits-per-byte from per-token log-probabilities, and explain why they are one quantity.
> - Explain why per-token metrics are unfair across tokenizers and why bits-per-byte is comparable.
> - Choose the right evaluator (benchmark, LLM judge, human, code scorer) for a question, and name each one's biases.
> - Decompose RAG and agent failures into retrieval, generation, tool use, and latency metrics.
> - Build an eval harness that runs suites, scores outputs, reports confidence intervals, and compares two versions with a paired test.

## Before you begin

You should be comfortable with entropy, KL divergence, and cross-entropy from [Information Theory](../../mathematics/17-information-theory/) in *Mathematics*, and with the training loss from [Deep Learning](../02-deep-learning/) in *AI Engineering*. The RAG metrics assume you know what a retrieval pipeline does, from [Retrieval and RAG](../27-retrieval-and-rag/). The agent sections use the agent SDK from [the agent SDK chapter](../28-agents/README.md) and the durable `EvalSuite` workflow from [Durable orchestration and workers](../../systems/07-durable-orchestration/README.md).

## Cross-entropy: the loss that is an eval

A language model outputs a probability distribution over the next token, and it is trained to minimize the **cross-entropy loss**

$$L = -\frac{1}{N} \sum_i \log p(x_i \mid x_{<i}),$$

the average negative log-probability the model assigned to the *actual* next token. With log base 2 the unit is **bits**; with the natural log it is **nats**.

The information-theoretic meaning is what makes this the foundation. Cross-entropy $H(p, q) = -\sum_x p(x) \log q(x)$ is the average number of bits needed to encode samples from the true distribution $p$ using the model's distribution $q$. The model's loss is the cross-entropy between real text and its predictions. Minimizing loss is learning the true distribution, which is compressing text: a better language model is literally a better compressor, as Shannon showed. This is why evaluation begins at training. The loss curve is your first, cheapest, and most honest signal.

## Perplexity, bits-per-token, and bits-per-byte

That one quantity shows up under several names, normalized in different ways. The comparison below is the crux:

| Metric | Definition | Reads as | Tokenizer-dependent? |
|--------|-----------|----------|----------------------|
| **Cross-entropy** | average $-\log_2 p$ (bits per token) | bits to predict the next token | Yes |
| **Bits-per-token** | cross-entropy in bits | the same thing, named for the unit | Yes |
| **Perplexity (PPL)** | $2^{\text{cross-entropy in bits}} = e^{\text{loss in nats}}$ | the effective number of equally likely choices: the branching factor | Yes |
| **Bits-per-byte (bpb)** | total bits / total UTF-8 **bytes** of text | compression rate, tokenizer-free | **No** |
| **Bits-per-character (bpc)** | total bits / total characters | an older character-level variant | No |

Perplexity is the most intuitive: a PPL of 100 means the model is, on average, as uncertain as if it were choosing uniformly among 100 tokens. Lower is better.

The trouble is that perplexity and bits-per-token depend on how text is split. A model with a bigger vocabulary has fewer, "easier" tokens and looks better on per-token metrics, which makes the comparison unfair. Bits-per-byte fixes this by dividing the *total* cross-entropy, summed in bits over the sequence, by the number of **bytes** of the underlying text rather than the number of tokens. The result is a tokenizer-independent compression rate you can compare across models; The Pile and language-modeling leaderboards use it, and it is the rigorous way to compare base models. The conversion is

$$\text{bpb} = \frac{\text{loss in nats}}{\ln 2} \times \frac{\text{tokens}}{\text{bytes}},$$

the same loss, renormalized from per-token to per-byte. A good exercise is to take two models with different tokenizers, compute both metrics on the same held-out text from per-token log-probabilities, and watch perplexity flatter the larger vocabulary while bits-per-byte does not.

## Intrinsic versus extrinsic evaluation

Everything so far is intrinsic: it measures modeling quality on held-out text. It is cheap, automatic, and ideal for pretraining and tracking scaling, but it is blind to usefulness. A model can have great perplexity and still be unhelpful, biased, or wrong. Extrinsic evaluation measures task performance: does the model answer correctly, follow instructions, stay grounded, and refuse appropriately? There is no closed form, so it is approximated with benchmarks, judges, and humans, and it is what users actually feel. The discipline is simple to state: use intrinsic metrics to steer training, use extrinsic metrics to decide whether the model is good, and never ship on perplexity alone.

## Benchmarks and capability evals

Task benchmarks are the first extrinsic tool. MMLU measures knowledge, GSM8K and MATH measure reasoning, HumanEval and MBPP measure code, and HellaSwag measures commonsense. Aggregators such as **HELM** and EleutherAI's lm-evaluation-harness run many tasks with standardized prompts and scoring. A benchmark scores either by exact match or multiple-choice log-likelihood (pick the highest-probability option), or by generating an answer and running a checker.

The central threat is **contamination**: if benchmark data leaked into training, the score measures memorization, not capability. Prefer fresh, private, or held-out evals, and treat gains on public benchmarks skeptically. And remember that capability is not deployment fit. A high MMLU score does not mean the model is good at *your* task; for that you need a domain eval set.

## LLM-as-judge and preference evaluation

For open-ended outputs, you can prompt a strong model to grade an output (a pointwise score) or to pick the better of two (**pairwise**). This is scalable and cheap, and it is the default for subjective quality. But judges have biases you must control: **position** bias (favouring the first or second answer, which you control by asking in both orders and reconciling the verdicts), **verbosity** bias (longer looks better), **self-preference** (favouring its own model family), and limited discrimination on hard cases. Always validate the judge against human labels. Rubrics and reference-guided judging, which give the judge criteria or a gold answer, reduce variance.

Many pairwise verdicts can be aggregated into **Elo** or **Bradley-Terry** ratings, as LMSYS **Chatbot Arena** does. Humans voting on pairs is the closest thing to ground truth for general quality. The point to carry forward is that benchmarks measure capability, LLM judges measure preference, and humans are the ground truth; each evaluator has different biases, so triangulate rather than trusting one.

## RAG evaluation

A RAG system has two stages, and its evaluation should separate them. The metrics divide cleanly by stage:

| Stage | Metric | Asks |
|-------|--------|------|
| Retrieval | **Context precision / recall**, **MRR**, **NDCG**, Recall@k | Did we fetch the right chunks, ranked well? |
| Generation | **Faithfulness / groundedness** | Is every claim supported by the retrieved context (no hallucination)? |
| Generation | **Answer relevance / correctness** | Does it actually answer, and match the truth? |

The value of this split is that it decomposes failure. A bad RAG answer is either a retrieval miss (the right context was not fetched) or a generation miss (the context was fetched but ignored or contradicted). Comparing faithfulness with context recall tells you which, so you fix the right stage. Frameworks such as Ragas use LLM-as-judge to compute these metrics, and saige's `eval/` package exposes exactly this set (ContextPrecision, ContextRecall, NDCG, MRR, Faithfulness, AnswerCorrectness) behind a composable `Scorer` interface.

## Agent and system evaluation

Agents need their own metrics on top of these. **Task success**, whether the agent accomplished the objective end to end, is the metric that matters most. **Tool-call accuracy** asks whether it chose the right tool with the right arguments and a valid schema, and whether it recovered from tool errors. **Trajectory metrics** track turn count, redundant steps, and loops. **Operational metrics** include TTFT (time to first token), TTLT (time to last token), cost per task, and tokens per task; quality is meaningless if the answer is too slow or too expensive, which ties evaluation to [Streaming and SSE](../../systems/04-streaming/) and to serving SLOs. saige's agent scorers track exactly these: TTFT and TTLT, tool success, and turn count.

## The eval harness pattern

All of this adds up to evaluation as platform infrastructure rather than a notebook. The core is a composable `Scorer` interface: a metric is a pure function from input, output, and optional reference to a score, which is [a functional pattern](../../software-engineering/04-design-patterns/), and the harness runs dataset times model times scorers into an aggregated report. saige's framework, an `Observation` plus pluggable `Scorer`s across RAG, agent, and knowledge-graph subsystems, is this pattern, and so is what you build in this chapter.

Eval datasets are versioned assets: curate, freeze, and review them like code, and guard them against contamination and drift. Run the eval suite in CI on every model or prompt change and block on regressions, the evaluation analogue of unit tests discussed in [Software Engineering](../../software-engineering/). And combine offline and online signals: offline evals on fixed sets before you ship, online signals (user feedback, A/B tests, acceptance rate) after, closing the loop back to training data.

When you need to pick a metric quickly, this mapping from question to measurement summarizes the chapter so far:

| Question | Metric |
|----------|--------|
| How well does the base model predict text? | Cross-entropy / perplexity |
| Compare base models with different tokenizers | **Bits-per-byte (bpb)** |
| General capability | Benchmarks (MMLU, GSM8K, HumanEval); watch contamination |
| Open-ended quality, at scale | LLM-as-judge (control position and verbosity bias) |
| General "which model is better" | Pairwise plus Elo (human arena) |
| Did RAG retrieve the right context? | Context precision/recall, NDCG, MRR |
| Did RAG hallucinate? | Faithfulness / groundedness |
| Is the agent actually working? | Task success plus tool accuracy plus TTFT/TTLT |
| Ship-gate a change | Versioned eval suite in CI |

## Where these ideas show up in practice

The same patterns appear across the industry. OpenAI and Anthropic combine held-out loss with capability and safety evals: bits-per-byte, internal evals, and red-teaming. LMSYS turns human pairwise votes into Elo ratings in Chatbot Arena. Stanford CRFM standardizes multi-metric benchmarking in HELM, and EleutherAI provides a reproducible benchmark harness in lm-evaluation-harness. Ragas and other eval vendors use LLM-as-judge for RAG metrics such as faithfulness and context precision and recall, and saige, the reference implementation for this chapter, applies one composable `Scorer` across RAG, agent, and knowledge-graph metrics.

The ideas also connect to other parts of the library. Entropy, KL divergence, and compression in [Information Theory](../../mathematics/17-information-theory/) explain why the loss measures bits. [Deep Learning](../02-deep-learning/) covers the training objective and scaling laws behind the loss as the first eval. [Retrieval and RAG](../27-retrieval-and-rag/) is the pipeline that context precision, recall, and faithfulness evaluate. [LLM Systems](../20-inference-engines/) and [Streaming and SSE](../../systems/04-streaming/) cover the TTFT, TTLT, cost, and serving SLOs behind operational evaluation. [Coding and Design Patterns](../../software-engineering/04-design-patterns/) motivates the scorer as a composable pure function, [Software Engineering](../../software-engineering/) treats eval suites as CI-gated tests, and the [agent eval harness case study](../32-case-study-agent-evals/) works through grading an agent that changes state, with replay, set-based checks, oracles, and mutants.

## In this chapter

The four sections build the agent evaluation pipeline in Go, each plugging into the one before it. As you read, try to predict each aggregate and each step before you read the solution's.

1. [Eval runner](01-eval-runner.md): run every case of a suite through a subject concurrently and deterministically, exclude and count failed scores, and report means with cluster bootstrap intervals in a schema-defined file.
2. [Agent evaluation scorers](02-scorers.md): deterministic scorers for text, schema, timing, tool use, retrieval ranking, citations, and state changes graded as sets.
3. [LLM judge with position control](03-llm-judge.md): a rubric judge with strict parsing, a pairwise comparison that detects position bias, Cohen's kappa against human labels, and deterministic sampling.
4. [Paired A/B experiments](04-ab-experiments.md): compare two versions on the same cases with paired deltas, a paired bootstrap interval, and a sign permutation p-value.

## Further reading

- MacKay, [*Information Theory, Inference, and Learning Algorithms*](http://www.inference.org.uk/mackay/itila/) (free), for cross-entropy, perplexity, and compression.
- [Stanford HELM](https://crfm.stanford.edu/helm/) (free).
- [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (free).
- [Ragas documentation](https://docs.ragas.io/) (free), for RAG metrics.
- The [Chatbot Arena paper](https://arxiv.org/abs/2403.04132) (free).
- [*The Pile*](https://arxiv.org/abs/2101.00027) (free), for bits-per-byte.
- Hendrycks et al., [MMLU](https://arxiv.org/abs/2009.03300) (free).
- [saige](https://github.com/urmzd/saige)'s `eval/` package, as a reference for a composable `Scorer` framework.
