# Pass 10: Agents

With a served model and tool calls in the engine, the system can act as well as answer. This pass builds an agent SDK in Go on the engine's tool-call API: provider types, a tool registry with schema validation, the agent loop, a tool gate with a prompt-injection suite, and durable agent runs. It adds retrieval over the system's own documentation, so the agent can answer questions about the system and cite its sources, and an evaluation runner with scorers, an LLM judge, and paired A/B experiments. The gateway gains a usage policy enforced once, at the edge, for every model and client.

An optional second half post-trains the capstone model: supervised fine-tuning, DPO, GRPO with verifiable rewards, and distillation.

> **Learning objectives**
> - Explain how the agent loop calls tools through the engine and why the tool gate blocks injected instructions.
> - Explain how retrieval, context assembly, and citations make an answer checkable.
> - Explain how the eval runner, scorers, judge, and A/B experiments compare two models with an interval.
> - Explain how the gateway enforces a usage policy for every request.

## Before you begin

This pass needs [Pass 9](../course-p09-capstone-training/README.md): the served capstone and the engine's tool calls from [Pass 7](../course-p07-serving-platform/README.md).

## In this pass

1. [Types, OpenAI-compatible provider, retry wrapper](../../../ai-platform-engineering/13-agent-sdk/01-types-and-provider.md). Afterwards you can explain the agent SDK's types, the OpenAI-compatible provider, and which errors the retry wrapper retries.
2. [Tools, registry, JSON Schema argument validation](../../../ai-platform-engineering/13-agent-sdk/02-tools-and-schemas.md). Afterwards you can explain how tools register and how arguments are validated against JSON Schema.
3. [Agent loop](../../../ai-platform-engineering/13-agent-sdk/03-agent-loop.md). Afterwards you can trace the agent loop from model call to tool execution and explain when it stops.
4. [Tool gate, SQL safety gate, prompt-injection suite](../../../ai-platform-engineering/13-agent-sdk/04-tool-gate-and-injection.md). Afterwards you can explain how the tool gate and SQL safety gate block unsafe calls and what the prompt-injection suite tests.
5. [Durable agent runs (AgentRun)](../../../ai-platform-engineering/13-agent-sdk/05-durable-agent-runs.md). Afterwards you can explain how AgentRun makes an agent run durable and replayable.
6. [RAG ingest: crawler with SSRF allowlist, chunker, embedder, store](../../../ai-platform-engineering/07-retrieval-and-rag/06-rag-ingest.md). Afterwards you can explain how RAG ingest crawls behind an SSRF allowlist, chunks, embeds, and stores documents.
7. [Retrieval: BM25, flat and IVF vectors, RRF, MMR](../../../ai-platform-engineering/07-retrieval-and-rag/07-retrieval.md). Afterwards you can explain how BM25, flat and IVF vector search, RRF, and MMR retrieve and rerank.
8. [Context assembly, citations, search_docs tool](../../../ai-platform-engineering/07-retrieval-and-rag/08-context-assembly.md). Afterwards you can explain how context assembly fits retrieved chunks into budget, attaches citations, and exposes search_docs.
9. [Eval runner](../../../ai-platform-engineering/09-llm-evaluation/09-eval-runner.md). Afterwards you can explain how the eval runner executes a suite and records results.
10. [Agent evaluation scorers](../../../ai-platform-engineering/09-llm-evaluation/10-scorers.md). Afterwards you can explain what each agent evaluation scorer measures.
11. [LLM judge with position control](../../../ai-platform-engineering/09-llm-evaluation/11-llm-judge.md). Afterwards you can explain how the LLM judge controls for position bias.
12. [Paired A/B experiments](../../../ai-platform-engineering/09-llm-evaluation/12-ab-experiments.md). Afterwards you can explain how a paired A/B experiment estimates a difference with a confidence interval.
13. [Usage policy at the gateway](../../../responsible-ai/05-usage-policy/01-usage-policy-at-the-gateway.md). Afterwards you can explain the usage policy and why it is enforced at the gateway.
14. [Usage policy enforcement](../../../ai-platform-engineering/12-gateway/08-usage-policy-enforcement.md). Afterwards you can explain how the gateway enforces the usage policy and returns 451.
15. [Agent image and chart](../../../infrastructure/01-containers-kubernetes/07-agent-image-and-chart.md). Afterwards you can explain how the agent image and chart are built and deployed.
16. [Agent evals as tests](../../../software-craftsmanship/03-testing-mentality/23-agent-evals-as-tests.md). Afterwards you can explain how agent evals are written as tests and what they assert.
17. [What the system does after Pass 10](milestone.md). Afterwards you can explain how the durable agent answers with citations and passes evals, policy checks, and prompt-injection cases.

## Going deeper: post-training

These optional sections post-train the TinyStories model and end with a second capstone:

18. [Optimization problem set, part b: constrained optimization, duality, and DPO](../../../mathematics-for-computing/10-optimization/91-problem-set-b.md). Afterwards you can work every problem in the set and explain each answer.
19. [Supervised fine-tuning: chat rendering, assistant-only loss, packing](../../p12-post-training/01-supervised-fine-tuning.md). Afterwards you can explain how SFT renders chats, masks the loss to assistant tokens, and packs sequences.
20. [Direct preference optimization and IPO](../../p12-post-training/02-direct-preference-optimization.md). Afterwards you can derive the DPO and IPO losses from the preference objective.
21. [Group relative policy optimization with verifiable rewards](../../p12-post-training/03-grpo-with-verifiable-rewards.md). Afterwards you can explain how GRPO computes group-relative advantages from verifiable rewards.
22. [Distillation with forward and reverse KL](../../p12-post-training/04-distillation.md). Afterwards you can explain how distillation with forward KL differs from reverse KL.
23. [Capstone 2: post-trained TinyStories model](../../capstones/02-post-trained.md). Afterwards you can explain how the post-trained TinyStories model is built and why it beats the base model on held-out preference tasks.

The stages are also in this pass's manifest, [`path.tsv`](path.tsv). [Pass 11](../course-p11-operate/README.md) releases the system as v1.0.0 and operates it.
