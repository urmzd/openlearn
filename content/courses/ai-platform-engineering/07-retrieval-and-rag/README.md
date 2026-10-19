# Retrieval & RAG

## Overview

- **Primary references**: [RAG paper (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401) (free), [BM25 / *The Probabilistic Relevance Framework* (Robertson & Zaragoza)](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) (free), [pgvector](https://github.com/pgvector/pgvector) (free)
- **Supplementary**: [Sentence-Transformers](https://www.sbert.net/) + [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) (free), [Reciprocal Rank Fusion (Cormack et al., 2009)](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) (free), [HNSW paper (Malkov & Yashunin)](https://arxiv.org/abs/1603.09320) (free), [GraphRAG (Microsoft)](https://arxiv.org/abs/2404.16130) (free), and **[saige](https://github.com/urmzd/saige)** as a Go reference implementation (multi-retriever fusion, chunking, multimodal variants, KG retrieval)
- **Prerequisites**: [Training & Frameworks](../01-training-and-frameworks/) (encoders, embeddings), [Distributed Data & Caching](../04-distributed-data-and-caching/) (vector stores, knowledge graphs)
- **Estimated time**: 3-4 weeks at 8-10 hrs/week

## Key Takeaways

- **RAG = retrieve, then generate.** Retrieval quality is the ceiling on answer quality — a model can't ground on what you didn't fetch. Most "the LLM is wrong" bugs are retrieval bugs.
- **Hybrid beats either half.** Dense (semantic) retrieval and lexical (BM25) retrieval fail in opposite ways; fuse them — Reciprocal Rank Fusion is the simple, strong default — and add graph retrieval when relationships matter.
- **Chunking is the highest-leverage knob.** How you split documents (size, overlap, recursive vs semantic boundaries) decides what *can* be retrieved. Get this wrong and no reranker saves you.
- **Encoders map meaning to geometry.** A bi-encoder embeds query and document separately (fast, indexable); a cross-encoder scores a pair jointly (accurate, slow) — so you retrieve with the first and rerank with the second.
- **Metadata is a first-class retrieval signal.** Filtering by source, time, type, and *permissions* before/with vector search is what makes retrieval correct and secure in production (ties to [Authorization](../08-authorization-and-access-control/)).
- **Multimodal retrieval reduces to text + a unifying record.** Normalize heterogeneous media (image, table, audio, PDF) into a common searchable representation so one index serves them all.

## How to Study

- Build the minimal RAG loop: chunk → embed → store (pgvector) → retrieve top-k → stuff context → generate. Then measure how answer quality moves as you change *only* the retriever.
- Implement BM25 from the formula and compare it to dense retrieval on the same queries — find the queries where each wins (exact terms/IDs vs paraphrases).
- Add a cross-encoder reranker over the fused candidates; measure NDCG/MRR before and after.
- Re-chunk the same corpus three ways (fixed, recursive, semantic) and watch retrieval recall change.

---

# Concepts & Techniques

## Core Insight

A language model knows what was in its training data, frozen and unattributed. RAG turns it into an open-book system: at query time you *retrieve* the relevant evidence and put it in the context window, so the model reasons over fresh, private, citable facts instead of its parametric memory. That reframes most of "AI engineering" as an **information retrieval** problem wearing a neural coat — and IR has 50 years of patterns. The pipeline is always the same shape (ingest → chunk → encode → index → retrieve → rerank → assemble → generate → evaluate); the skill is knowing which knob to turn, and retrieval — not the LLM — is almost always the bottleneck.

## 1. The RAG Pipeline (the shape)

```
ingest → chunk → encode (embed) → index ──┐
                                          ├─ retrieve → rerank → assemble context → generate → cite
query ──────────── encode (embed) ────────┘
```
Every production RAG system is a specialization of this. The rest of this topic is the patterns at each stage.

## 2. How Encoders Work

**Mapping text (and other media) to vectors**

**Key ideas**:
- **Transformer encoder**: a bidirectional stack (BERT-family) reads the whole input at once (no causal mask) and emits a contextual vector per token. **Pooling** (CLS token or mean over tokens) collapses them into one fixed-length **embedding**. (Foundations in [Neural Architectures](../../machine-learning/06-neural-architectures/) and [Deep Learning](../../machine-learning/02-deep-learning/).)
- **The geometry**: training arranges the space so **semantic similarity ≈ cosine similarity**. Retrieval is then "nearest neighbors in this space."
- **Bi-encoder vs cross-encoder** — the central distinction of the topic:
  - **Bi-encoder**: encode query and document *separately* into vectors; compare by dot product. Documents can be embedded once and indexed → **fast, scalable retrieval**. Slightly less accurate (no query-document interaction).
  - **Cross-encoder**: feed query+document *together* through the model, output one relevance score. Sees the interaction → **more accurate**, but must run per candidate pair at query time → only feasible for **reranking** a short candidate list.
  - **The production pattern**: bi-encoder to *retrieve* hundreds, cross-encoder to *rerank* to the top few.
- **Matryoshka embeddings**: train so a truncated prefix of the vector is still usable — store short vectors cheaply, expand when needed.

## 3. Chunking Patterns

**Splitting documents into retrievable units — the highest-leverage decision**

| Strategy | How | When |
|----------|-----|------|
| **Fixed-size** | N tokens with overlap | Baseline; simple, ignores structure |
| **Recursive / structural** | Split on a hierarchy of separators (¶ → sentence → word) to respect boundaries, with overlap | The strong default (saige's recursive splitter) |
| **Semantic** | Split where embedding similarity between adjacent sentences drops | Coherent topical chunks; more compute |
| **Document-aware** | Split on headings/sections/code blocks | Structured docs, markdown, code |

**Key ideas**:
- **The tension**: chunks too *large* dilute the embedding and waste context; too *small* lose the context needed to be meaningful. There's a sweet spot per corpus (often a few hundred tokens).
- **Overlap** carries context across boundaries so a fact split between chunks is still findable.
- **Parent/child (small-to-big)**: retrieve on small precise chunks, but feed the *parent* (larger surrounding) chunk to the model — precision in search, context in generation.
- **Carry metadata onto every chunk** (source, section, timestamp, permissions) — you'll filter and cite on it.

## 4. Indexing & Approximate Nearest Neighbor (ANN)

**Key ideas**:
- Exact nearest-neighbor over millions of vectors is too slow; **ANN** trades a little recall for orders-of-magnitude speed.
- **HNSW** (hierarchical navigable small-world graphs): the dominant ANN index — a multi-layer proximity graph you greedily descend. Great recall/latency; tunable via `M` and `efSearch`.
- **IVF / IVF-PQ**: cluster vectors, search only nearby clusters; product quantization compresses vectors for memory.
- **Where it lives**: pgvector (Postgres), Qdrant, Milvus, Weaviate, FAISS, Redis — see [Distributed Data & Caching](../04-distributed-data-and-caching/). The index is sharded, replicated, and cached like any other data.

## 5. Lexical Retrieval (BM25 & Bag-of-Words)

**The other half of hybrid — don't skip it**

**Key ideas**:
- **Bag-of-words**: represent text as term counts, ignoring order. Sparse, high-dimensional, exact-term.
- **TF-IDF**: weight a term by its frequency in the doc (TF) × its rarity across the corpus (IDF) — common words count less, distinctive words more.
- **BM25**: the refined, dominant lexical ranker — TF-IDF with **saturation** (`k1`: extra occurrences matter less and less) and **length normalization** (`b`: don't reward long docs for having more words). Decades old, still a brutally strong baseline.
- **Why keep lexical at all**: dense retrieval misses **exact matches** — IDs, error codes, names, rare jargon, acronyms — that BM25 nails. They fail in opposite directions, which is exactly why you fuse them.

## 6. Hybrid Retrieval & Fusion

**Combining dense + lexical (+ graph) into one ranked list**

**Key ideas**:
- **Reciprocal Rank Fusion (RRF)**: combine multiple ranked lists by summing `1/(k + rank)` across them. No score calibration needed, robust, embarrassingly simple — the default fusion (and what **saige** uses to fuse vector + BM25 + graph retrievers).
- **Score-based fusion**: normalize and weight each retriever's scores — more tunable, more fragile.
- **Graph retrieval / GraphRAG**: pull a connected subgraph of facts from a [knowledge graph](../04-distributed-data-and-caching/), resolving entities to documents — adds multi-hop reasoning and relationships that flat chunks can't express. saige fuses this as a third retriever and tracks **temporal validity** (`ValidAt`/`InvalidAt`) on relations so retrieval is point-in-time correct.
- **The pattern**: retrieve broadly from several complementary sources, fuse, then rerank — breadth first, precision last.

## 7. Metadata & Production-Grade Retrieval

**What separates a demo from production**

**Key ideas**:
- **Metadata filtering**: constrain retrieval by structured fields — source, type, recency, language, tenant, and **access permissions** — combined with vector/lexical search (pre-filter or post-filter). Often the difference between a right and wrong answer.
- **Authorization-aware retrieval**: a user must only retrieve chunks they're allowed to see. Filter by permission *at retrieval time* (carry ACLs/labels on each chunk) — see [Authorization & Access Control](../08-authorization-and-access-control/). This is "secure RAG," and it's a hard, mandatory requirement in the enterprise.
- **Context assembly**: dedupe, order, and fit candidates into the context budget; add **citations** (which chunk supported which claim) and optionally **compress** context with an LLM (saige does both).
- **Freshness & invalidation**: re-embed and re-index on document change; pin embedding-model versions (changing the encoder invalidates the whole index — a [caching](../04-distributed-data-and-caching/) problem in disguise).
- **Query transforms**: rewriting, expansion, HyDE (embed a hypothetical answer), and multi-query — improve recall before retrieval even runs.

## 8. Multimodal Handling

**One index over heterogeneous media**

**Key ideas**:
- **The unifying pattern**: model content as a hierarchy — **Document → Sections → ContentVariants** (saige's data model) — where every variant, whatever its medium (image, table, audio, PDF), carries a **`.Text` representation** (a caption, transcript, OCR, or extracted text). Now *one* text/vector index searches across all of them uniformly.
- **Joint embedding spaces**: models like CLIP embed images and text into the *same* space, so a text query can retrieve images directly (true multimodal retrieval) — see [Foundation Models](../../machine-learning/05-foundation-models/).
- **Ingestion plumbing**: URI resolution (`file://`, `s3://`), content negotiation, and per-type extractors (OCR, ASR, table parsing) turn raw files into searchable variants.
- **Generation**: feed retrieved text variants (and, for VLMs, the original media) into a multimodal model.

## 9. Evaluating Retrieval

Retrieval has its own metrics, separate from generation (full treatment in [LLM Evaluation](../09-llm-evaluation/)):
- **Recall@k / Precision@k**: did the relevant chunk make the top-k?
- **MRR** (mean reciprocal rank): how high was the first relevant hit?
- **NDCG**: rank-quality with graded relevance and position discounting.
- **Faithfulness / context-precision / context-recall** (RAG-specific, à la Ragas / saige's RAG scorers): is the answer grounded in the retrieved context, and was the right context retrieved?

---

## Decision Cheat Sheet

| Need | Reach for |
|------|-----------|
| Semantic / paraphrase matching | Dense bi-encoder + HNSW |
| Exact terms, IDs, rare jargon | BM25 (lexical) |
| Best general retrieval | Hybrid (dense + BM25) fused with RRF |
| Multi-hop / relationship questions | + Graph retrieval (GraphRAG) |
| Squeeze accuracy from candidates | Cross-encoder reranker |
| Split documents well | Recursive/structural chunking + overlap |
| Precise search, rich context | Parent/child (small-to-big) chunks |
| Secure, tenant-correct results | Metadata + permission filtering |
| Mixed media | Document→Section→Variant with a `.Text` per variant |

## Patterns Worth Internalizing

- **Retrieval is the ceiling.** Fix retrieval before touching the prompt or the model.
- **Fuse complementary retrievers, then rerank** — breadth (dense + lexical + graph via RRF) first, precision (cross-encoder) last.
- **Bi-encode to find, cross-encode to rank** — separate the fast indexable step from the accurate scoring step.
- **Chunk for what's retrievable; carry metadata on every chunk** — including permissions, timestamps, and source for filtering and citation.
- **Normalize every medium to a common searchable record** — heterogeneous data, one index.
- **Changing the encoder invalidates the index** — treat embeddings like a cache with a version key.

## Connections to Other Tracks

| Concept | Connected Track | Application |
|---------|-----------------|-------------|
| Encoders, embeddings, fine-tuning, CLIP | [Training & Frameworks](../01-training-and-frameworks/) / [Foundation Models](../../machine-learning/05-foundation-models/) | The models that embed |
| Vector stores, ANN, knowledge graphs, caching | [Distributed Data & Caching](../04-distributed-data-and-caching/) | Where the index lives |
| Permission-filtered retrieval | [Authorization & Access Control](../08-authorization-and-access-control/) | Secure / multi-tenant RAG |
| Retrieval & faithfulness metrics | [LLM Evaluation](../09-llm-evaluation/) | Measuring the pipeline |
| Token streaming of the answer | [Streaming & SSE](../03-streaming-sse/) | Delivering the generation |

## How Companies Apply These Patterns

| Company | The pattern they lean on | Instance |
|---------|--------------------------|----------|
| Microsoft | Graph + vector hybrid | GraphRAG over knowledge graphs |
| Perplexity / You.com | Hybrid retrieval + reranking + citations | Answer engines |
| Elastic / Vespa | Lexical + dense in one engine | BM25 + ANN, fusion built in |
| Cohere / Voyage | Bi-encoder embeddings + cross-encoder rerankers | Retrieval-as-a-service |
| Glean / enterprise search | Permission-filtered, metadata-rich RAG | Secure multi-tenant retrieval |
| saige (reference) | RRF over vector + BM25 + graph, multimodal variants | Go SDK for agents/KG/RAG |

## Chapters

| # | Module | Chapter | Kind | Pass |
|---|---|---|---|---|
| 1 | `ag.06` | [RAG ingest: crawler with SSRF allowlist, chunker, embedder, store](06-rag-ingest.md) | build | 10 |
| 2 | `ag.07` | [Retrieval: BM25, flat and IVF vectors, RRF, MMR](07-retrieval.md) | build | 10 |
| 3 | `ag.08` | [Context assembly, citations, search_docs tool](08-context-assembly.md) | build | 10 |
