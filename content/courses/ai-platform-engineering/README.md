# AI Platform Engineering

This book is about the infrastructure underneath AI products, taught as patterns rather than products: caches, consistent hashing, idempotent retry, event-sourced replay, the worker pool, backpressure. Popular tools appear as instances of those patterns (vLLM is continuous batching plus a paged cache; Temporal is event-sourced durable execution plus a worker pool), so the next tool is something you already understand. It is written for ML engineers who want to see the platform their model runs on, backend engineers moving into AI infrastructure, and anyone building a GenAI or retrieval product. You should already know how transformers are trained and served.

## What you will be able to do

- Tell distributed training, distributed data, and distributed orchestration apart, and reach for the right family of patterns for each.
- Choose a caching pattern by when its data goes stale, and defend it against stampedes, herds, and penetration.
- Build durable execution: persist every step, replay history to recover, and run the code on a pool of workers.
- Build hybrid retrieval with RRF and reranking, and filter it by permission during the search.
- Evaluate a model or router with a harness you trust, then build the gateway and agent SDK that sit in front of a pool of models.

## Ways through this book

Reading every chapter takes about 25 to 35 weeks at eight to ten hours a week; the chapters form a graph, so you rarely need them strictly in order.

- **Coming from ML or research** (about 12 weeks): [Chapter 1: Training and Frameworks](01-training-and-frameworks/README.md), then [Chapter 4: Distributed Data and Caching](04-distributed-data-and-caching/README.md), then [Chapter 5: Durable Orchestration and Workers](05-durable-orchestration-and-workers/README.md), to see the platform around your model.
- **Coming from backend or distributed systems** (about 17 weeks): [Chapter 2: RPC and Protocols](02-rpc-and-protocols/README.md), [Chapter 3: Streaming and SSE](03-streaming-sse/README.md), Chapters 4 and 5, then loop back to Chapter 1. If you want the vocabulary first, start with [Chapter 6: Coding and Design Patterns](06-coding-and-design-patterns/README.md), which the other chapters are written in.
- **Building a RAG product** (about 11 weeks): [Chapter 7: Retrieval and RAG](07-retrieval-and-rag/README.md) with [embedding models from Chapter 1](01-training-and-frameworks/README.md#creating-and-hosting-embedding-models), then Chapter 3 and [4.1 Response cache](04-distributed-data-and-caching/01-response-cache.md), then [Chapter 8: Authorization and Access Control](08-authorization-and-access-control/README.md) for multi-tenant retrieval and [Chapter 9: LLM Evaluation](09-llm-evaluation/README.md) to measure it.
- **Cutting an inference bill** (about 6 weeks): Chapter 9 first, through [9.4 Paired A/B experiments](09-llm-evaluation/04-ab-experiments.md), then [Chapter 11: Model Routing & Cascades](11-model-routing-and-cascades/README.md), [small language models in Chapter 1](01-training-and-frameworks/README.md#small-language-models), and [12.4 Routing, cascades, and failover](12-gateway/04-routing.md). Build the harness before you change which model answers.

As you read, ask of every tool a chapter names: what pattern is this an instance of? That pattern is the part that transfers.

## Before you start

You should be comfortable with [Deep Learning](../machine-learning/02-deep-learning/README.md) and [LLM Systems and Inference](../machine-learning/04-llm-systems/README.md) in *Machine Learning and AI*, [System Design](../systems/01-system-design/README.md) and [Cloud Native](../systems/03-cloud-native/README.md) in *Systems and Architecture*, and the OLTP, storage, and replication material in [*Data Engineering*](../data-engineering/README.md). The build chapters (12 and 13, and the build sections of Chapters 7 and 9) work against the engine from [*Build Your Own LLM System*](../build-an-llm/README.md), and [*Programming Languages*](../programming-languages/README.md) has the Go, Python, and protocol primers they rely on. For the messaging side of workers, see [*Infrastructure*](../infrastructure/README.md). New to these books? [How to read these books](../../references/how-to-read.md) explains how sections, review questions, and solutions work.

## Three kinds of "distributed"

"Distributed X" is three problems, and the book keeps them in separate chapters because the patterns differ.

| Problem | Chapter | Pattern family |
|---------|---------|----------------|
| One model across many GPUs | [1. Training and Frameworks](01-training-and-frameworks/README.md) | parallelism and sharding (data, tensor, pipeline; FSDP and ZeRO) |
| One dataset across many nodes | [4. Distributed Data and Caching](04-distributed-data-and-caching/README.md) | partitioning, replication, caching, invalidation |
| One workflow across many failures | [5. Durable Orchestration and Workers](05-durable-orchestration-and-workers/README.md) | durable execution, checkpoints, the worker pool |
