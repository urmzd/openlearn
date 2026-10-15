# AI Platform Engineering (25-35 weeks)

~8-10 hours per week. **Patterns-first**: the goal is to internalize the fundamental patterns (like math, everything downstream is derivative) and treat each trendy tool as an instance. Requires [Deep Learning](../../tracks/ml/02-deep-learning) and [LLM Systems](../../tracks/ml/04-llm-systems), plus [System Design](../../tracks/systems/01-system-design) and [Data Engineering](../../tracks/data-engineering) foundations (OLTP vs OLAP, replication). Tip: [Coding & Design Patterns](../../tracks/ai-platform-engineering/06-coding-and-design-patterns) can be read first as vocabulary.

### Weeks 1-6: Training & Frameworks

| Week | Focus | Reference |
|------|-------|-----------|
| 1 | PyTorch deep-dive: eager, autograd, `nn.Module`, `torch.compile` | PyTorch docs |
| 2 | JAX: `jit`/`grad`/`vmap`/`pmap`, Flax/Optax, PyTorch vs JAX | JAX docs |
| 3 | Distributed training I: data parallel (DDP), all-reduce, mixed precision | PyTorch DDP + Ultra-Scale Playbook |
| 4 | Distributed training II: FSDP/ZeRO, tensor/pipeline/expert parallelism, Megatron/DeepSpeed | FSDP tutorial + DeepSpeed docs |
| 5 | Embedding models: contrastive fine-tuning, hosting (TEI), vector stores | Sentence-Transformers + pgvector |
| 6 | Small language models: distillation, LoRA/QLoRA (PEFT), quantize, serve with vLLM | HF PEFT + vLLM docs |

### Weeks 7-9: RPC & Protocols

| Week | Focus | Reference |
|------|-------|-----------|
| 7 | RPC model, partial failure, serialization formats (JSON/Protobuf/Avro/Thrift) | DDIA Ch 4 + protobuf.dev |
| 8 | gRPC: HTTP/2, the four call types, streaming, interceptors, gateways | gRPC docs |
| 9 | Schema evolution, choosing the protocol per boundary, Arrow Flight | Avro spec + gRPC docs |

### Weeks 10-11: Streaming & SSE

| Week | Focus | Reference |
|------|-------|-----------|
| 10 | SSE protocol, `EventSource`, SSE vs WebSockets vs long-polling vs gRPC streaming | MDN SSE + HTML spec |
| 11 | End-to-end token streaming, cancellation/backpressure, proxies and scaling | OpenAI streaming + vLLM docs |

### Weeks 12-15: Distributed Data & Caching

| Week | Focus | Reference |
|------|-------|-----------|
| 12 | OLTP vs OLAP, sharding (hash/range/consistent hashing), hot shards, rebalancing | DDIA Ch 5-6 + Citus/Vitess |
| 13 | In-memory stores & caching patterns (cache-aside/through/back), invalidation | Redis docs + Scaling Memcache |
| 14 | Caching failure modes (stampede, herd, penetration); Cassandra (LSM, ring, quorum) | Redis + Cassandra docs |
| 15 | Knowledge graphs (Neo4j, Apache AGE on Postgres), bitemporal/temporal data | AGE + Neo4j docs |

### Weeks 16-18: Durable Orchestration, Workers & Patterns

| Week | Focus | Reference |
|------|-------|-----------|
| 16 | Durable execution & checkpointing, event-sourced replay, the worker pattern | Temporal + DBOS docs |
| 17 | Reliability patterns (idempotency, saga/compensation, outbox); distributed observability + profiling | Temporal docs + OpenTelemetry + Gregg |
| 18 | Coding & design patterns: decorator, facade, closures, currying, composition | Refactoring Guru + Mostly Adequate Guide |

### Weeks 19-25: Retrieval, Authorization & Evaluation

| Week | Focus | Reference |
|------|-------|-----------|
| 19 | Encoders (bi- vs cross-encoder), embedding retrieval, ANN/HNSW | RAG paper + HNSW + sbert |
| 20 | Chunking patterns (fixed/recursive/semantic, parent-child), metadata | saige RAG + pgvector |
| 21 | Lexical retrieval (BM25, bag-of-words, TF-IDF); hybrid fusion (RRF) + reranking | BM25 review + RRF paper |
| 22 | Multimodal handling (Document→Section→Variant), GraphRAG, production retrieval | GraphRAG + saige |
| 23 | Authorization: RBAC/ABAC/ReBAC/NGAC, the pushdown complexity ladder, Zanzibar | NIST + Zanzibar + OpenFGA |
| 24 | Secure RAG (permission-filtered retrieval, multi-tenancy); PDP/PEP, policy-as-code | NIST ABAC + OPA |
| 25 | LLM evaluation: cross-entropy, perplexity, **bits-per-byte**; benchmarks & contamination | MacKay + HELM + The Pile |

### Weeks 26-28: Edge, Realtime & On-Device Inference

| Week | Focus | Reference |
|------|-------|-----------|
| 26 | The deployment spectrum; **llama.cpp + GGUF** end-to-end (convert → quantize → `llama-server`); Ollama; MLX/ExecuTorch on-device | llama.cpp + GGUF + Ollama |
| 27 | Realtime/streaming encoders: offline vs causal/chunked; speech (Whisper offline, Conformer/RNN-T streaming); real-time factor | Whisper + Conformer papers |
| 28 | Efficiency architectures (the Mistral clinic): sliding-window attention, GQA, Mixtral MoE, Codestral Mamba/SSM, Voxtral | Mistral 7B + Mixtral + Mamba papers |

### Weeks 29-30: Model Routing & Cascades

| Week | Focus | Reference |
|------|-------|-----------|
| 29 | The three routing decisions; router families (rules, predictive, cascade, verifier-gated, session-level); oracle vs deployed router; model recall | RouteLLM + FrugalGPT + LLMRouterBench |
| 30 | Cache-aware switching, escalation policy (category, verifier, confidence), token-basis economics, the SFT/DPO/RFT ladder, evaluating a router, System One decision models, specialised models in the pool | FireRouter docs + RFT docs + TypeSafe docs |

### Optional deep dive (weeks 31-35)

LLM-as-judge & Elo arenas, RAG metrics (faithfulness, context precision/recall, NDCG/MRR), agent metrics (TTFT/TTLT, tool success), and building a composable `Scorer` eval harness gated in CI. References: Ragas, Chatbot Arena, lm-evaluation-harness, saige `eval/`.

### Daily Routine (AI Platform Engineering)

1. **Name the pattern, then the tool** — for every tool, ask "what pattern is this an instance of?" That's the transferable knowledge.
2. **Build, don't just read** — train it, shard it, cache it, stream it, orchestrate it, retrieve it, authorize it, score it.
3. **Measure** — GPU memory, tokens/sec, bytes-on-wire, TTFT, cache hit rate, retrieval recall, bits-per-byte; numbers beat intuition.
4. **Break it on purpose** — kill a worker mid-workflow, expire a hot cache key, reuse a Protobuf tag, leak a doc past an ACL; learn the failure modes.

---
