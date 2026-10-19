# A reading plan for AI Platform Engineering

This plan paces the book over roughly 25 to 35 weeks at eight to ten hours a week. It is a suggestion, not a schedule to keep: the chapters form a graph, and the [front matter](README.md) offers shorter reading paths if you have a specific goal. Whatever pace you choose, read with one question in mind: for every tool a chapter names, what pattern is it an instance of? That pattern is the part that transfers.

Before you start, make sure you are comfortable with [Deep Learning](../machine-learning/02-deep-learning/README.md) and [LLM Systems and Inference](../machine-learning/04-llm-systems/README.md) from *Machine Learning and AI*, [System Design](../systems/01-system-design/README.md) from *Systems and Architecture*, and the foundations of *Data Engineering* ([its front matter](../data-engineering/README.md)), especially OLTP versus OLAP and replication. [Coding and Design Patterns](06-coding-and-design-patterns/README.md) can be read first as vocabulary if you like; the other chapters are written in its terms.

## Training and frameworks (about six weeks)

Spend the first six weeks in [Training and Frameworks](01-training-and-frameworks/README.md). Begin with a PyTorch deep dive (eager execution, autograd, `nn.Module`, `torch.compile`), then JAX (`jit`, `grad`, `vmap`, `pmap`, Flax and Optax) and a comparison of the two. The next two weeks are distributed training: first data parallelism with DDP, all-reduce, and mixed precision, then FSDP and ZeRO with tensor, pipeline, and expert parallelism as in Megatron and DeepSpeed. Finish with embedding models (contrastive fine-tuning, hosting with TEI, vector stores) and small language models (distillation, LoRA and QLoRA with PEFT, quantization, and serving with vLLM). The framework documentation, the Ultra-Scale Playbook, the FSDP tutorial, DeepSpeed's docs, Sentence-Transformers, pgvector, and HF PEFT are good companions.

## RPC, protocols, and streaming (about five weeks)

[RPC and Protocols](02-rpc-and-protocols/README.md) takes about three weeks: the RPC model, partial failure, and serialization formats (JSON, Protobuf, Avro, Thrift), with chapter 4 of *Designing Data-Intensive Applications* alongside; then gRPC over HTTP/2 with its four call types, streaming, interceptors, and gateways; then schema evolution, choosing a protocol per boundary, and Arrow Flight. Follow it with two weeks of [Streaming and SSE](03-streaming-sse/README.md): the SSE protocol and `EventSource`, how SSE compares with WebSockets, long-polling, and gRPC streaming, and then end-to-end token streaming with cancellation, backpressure, proxies, and scaling.

## Distributed data and caching (about four weeks)

[Distributed Data and Caching](04-distributed-data-and-caching/README.md) opens with OLTP versus OLAP and sharding by hash, range, and consistent hashing, along with hot shards and rebalancing (chapters 5 and 6 of *Designing Data-Intensive Applications*, plus Citus and Vitess). The second week covers in-memory stores and the caching patterns (cache-aside, write-through, write-back) and invalidation, with the Redis docs and *Scaling Memcache at Facebook*. The third covers caching failure modes (stampedes, herds, penetration) and Cassandra's LSM trees, ring, and quorums. The last week turns to knowledge graphs (Neo4j, Apache AGE on Postgres) and bitemporal data.

## Orchestration and patterns (about three weeks)

Read [Durable Orchestration and Workers](05-durable-orchestration-and-workers/README.md) over two weeks: first durable execution and checkpointing, event-sourced replay, and the worker pattern; then reliability patterns (idempotency, sagas and compensation, the outbox) together with distributed observability and profiling, using the Temporal and DBOS docs, OpenTelemetry, and Brendan Gregg's work. Use the third week for [Coding and Design Patterns](06-coding-and-design-patterns/README.md) if you have not read it yet: decorator, facade, closures, currying, and composition.

## Retrieval, authorization, and evaluation (about seven weeks)

[Retrieval and RAG](07-retrieval-and-rag/README.md) deserves four weeks. Start with encoders (bi- versus cross-encoders), embedding retrieval, and approximate nearest neighbors with HNSW; then chunking (fixed, recursive, semantic, parent-child) and metadata; then lexical retrieval (bag of words, TF-IDF, BM25), hybrid fusion with RRF, and reranking; and finally multimodal handling (document, section, variant), GraphRAG, and production retrieval.

Spend two weeks on [Authorization and Access Control](08-authorization-and-access-control/README.md): RBAC, ABAC, ReBAC, and NGAC, the pushdown complexity ladder, and Zanzibar with OpenFGA; then secure RAG with permission-filtered retrieval and multi-tenancy, PDP and PEP, and policy as code with OPA.

Close this stretch with a week of [LLM Evaluation](09-llm-evaluation/README.md) on cross-entropy, perplexity, and bits per byte, and on benchmarks and contamination, reading MacKay, HELM, and The Pile.

## Edge, realtime, and on-device inference (about three weeks)

In [Edge, Realtime and On-Device Inference](10-edge-realtime-inference/README.md), first walk the deployment spectrum and run llama.cpp with GGUF end to end (convert, quantize, `llama-server`), along with Ollama and on-device runtimes such as MLX and ExecuTorch. Next, study realtime and streaming encoders: offline versus causal and chunked, speech with Whisper offline and Conformer or RNN-T streaming, and the real-time factor. Finish with the efficiency architectures in the Mistral clinic: sliding-window attention, GQA, Mixtral's mixture of experts, Codestral Mamba's state-space model, and Voxtral.

## Model routing and cascades (about two weeks)

[Model Routing and Cascades](11-model-routing-and-cascades/README.md) takes two weeks. The first covers the three routing decisions, the router families (rules, predictive, cascade, verifier-gated, session-level), the gap between the oracle and a deployed router, and model recall, with RouteLLM, FrugalGPT, and LLMRouterBench. The second covers cache-aware switching, escalation policy (category, verifier, confidence), token-basis economics, the SFT, DPO, and RFT ladder, evaluating a router, System One decision models, and specialized models in the pool.

## The gateway and the agent SDK

The last two chapters are build chapters that sit on top of the engine from *Build Your Own LLM System*. Read [Gateway](12-gateway/README.md) once you have the streaming chapter behind you, and then [Agent SDK](13-agent-sdk/README.md), which uses the gateway as its model endpoint and the durable-orchestration chapter's ideas for its durable runs. Budget about two weeks for each if you work through the solution code.

## Going deeper

If you have time left (weeks 31 to 35 in a long plan), return to [LLM Evaluation](09-llm-evaluation/README.md) for the optional depth: LLM-as-judge and Elo arenas, RAG metrics (faithfulness, context precision and recall, NDCG, MRR), agent metrics (time to first and last token, tool success), and a composable `Scorer` evaluation harness gated in CI. Ragas, Chatbot Arena, lm-evaluation-harness, and saige's `eval/` are the references.

## How to read each week

Four habits make the reading stick. Name the pattern before the tool. Build rather than only read: train it, shard it, cache it, stream it, orchestrate it, retrieve it, authorize it, score it. Measure (GPU memory, tokens per second, bytes on the wire, time to first token, cache hit rate, retrieval recall, bits per byte), because numbers beat intuition. And break things on purpose: kill a worker in the middle of a workflow, expire a hot cache key, reuse a Protobuf tag, leak a document past an ACL, and learn the failure modes firsthand.
