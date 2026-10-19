# Retrieval and RAG

A language model knows what was in its training data, frozen and unattributed. Retrieval-augmented generation (RAG) turns it into an open-book system: at query time you *retrieve* the relevant evidence and put it in the context window, so the model reasons over fresh, private, citable facts instead of its parametric memory. That reframes much of "AI engineering" as an **information retrieval** problem wearing a neural coat, and information retrieval has fifty years of patterns to draw on.

The pipeline always has the same shape: ingest, chunk, encode, index, retrieve, rerank, assemble, generate, evaluate. The skill is knowing which knob to turn, and the first lesson of this chapter is that retrieval, not the LLM, is almost always the bottleneck. A model cannot ground on what you did not fetch, so most "the LLM is wrong" bugs are retrieval bugs. The chapter first surveys the patterns at each stage of the pipeline, then builds three of them for real: an ingest pipeline that produces an index, a hybrid retriever that searches it, and a context assembler that turns results into cited excerpts behind a `search_docs` tool. By the end, the agent from the [Agent SDK](../13-agent-sdk/) chapter answers questions about the course docs with citations that resolve to stored chunks.

> **Learning objectives**
> - Describe each stage of a RAG pipeline and the main design choices at that stage.
> - Explain why dense and lexical retrieval fail in opposite ways and how fusion and reranking combine them.
> - Build an idempotent ingest pipeline with token-budgeted chunking and an SSRF-safe crawler.
> - Implement BM25, flat and IVF vector search, RRF fusion, and MMR diversification over a shared index format.
> - Assemble retrieved chunks into a budgeted, cited context and verify the citations a model writes.

## Before you begin

This chapter assumes the encoders and embeddings of [Training & Frameworks](../01-training-and-frameworks/) and the vector stores, caches, and knowledge graphs of [Distributed Data & Caching](../04-distributed-data-and-caching/). The three build sections plug into the agent of the [Agent SDK](../13-agent-sdk/) chapter, so its [provider](../13-agent-sdk/01-types-and-provider.md) and [tools](../13-agent-sdk/02-tools-and-schemas.md) sections are worth reading first. Background on transformers comes from [Neural Architectures](../../machine-learning/06-neural-architectures/) and [Deep Learning](../../machine-learning/02-deep-learning/) in *Machine Learning and AI*.

## The RAG pipeline

Every production RAG system is a specialization of one diagram. Documents flow through ingest into an index; queries are encoded with the same encoder; the two meet at retrieval:

```
ingest → chunk → encode (embed) → index ──┐
                                          ├─ retrieve → rerank → assemble context → generate → cite
query ──────────── encode (embed) ────────┘
```

The rest of this introduction walks the stages in order, starting with the model that maps text to vectors.

## How encoders work

A **transformer encoder** is a bidirectional stack (the BERT family) that reads the whole input at once, with no causal mask, and emits a contextual vector per token. **Pooling** (the CLS token, or a mean over tokens) collapses those into one fixed-length **embedding**. Training arranges the space so that semantic similarity is approximately cosine similarity, and retrieval then becomes "nearest neighbors in this space."

The central distinction of the chapter is between two ways of using an encoder. A **bi-encoder** encodes query and document *separately* into vectors and compares them by dot product. Documents can be embedded once and indexed, which makes retrieval fast and scalable, at the price of slightly lower accuracy, because the model never sees query and document together. A **cross-encoder** feeds query and document *together* through the model and outputs one relevance score. It sees the interaction, so it is more accurate, but it must run per candidate pair at query time, which makes it feasible only for **reranking** a short candidate list. The production pattern follows directly: a bi-encoder to *retrieve* hundreds of candidates, a cross-encoder to *rerank* them down to the top few.

One refinement is worth knowing: **Matryoshka embeddings** are trained so that a truncated prefix of the vector is still usable, letting you store short vectors cheaply and expand when needed.

## Chunking

Before anything is embedded, documents must be split into retrievable units, and this is the highest-leverage decision in the pipeline. How you split decides what *can* be retrieved; get it wrong and no reranker saves you. Four strategies cover most practice:

| Strategy | How | When |
|----------|-----|------|
| **Fixed-size** | N tokens with overlap | Baseline; simple, ignores structure |
| **Recursive / structural** | Split on a hierarchy of separators (paragraph, then sentence, then word) to respect boundaries, with overlap | The strong default (saige's recursive splitter) |
| **Semantic** | Split where embedding similarity between adjacent sentences drops | Coherent topical chunks; more compute |
| **Document-aware** | Split on headings, sections, code blocks | Structured docs, Markdown, code |

The underlying tension is size. Chunks that are too *large* dilute the embedding and waste context; chunks that are too *small* lose the context needed to be meaningful. There is a sweet spot per corpus, often a few hundred tokens. **Overlap** carries context across boundaries, so a fact split between chunks is still findable. The **parent/child** (small-to-big) pattern retrieves on small, precise chunks but feeds the model the larger parent chunk around them: precision in search, context in generation. And every chunk should carry its metadata (source, section, timestamp, permissions), because you will filter and cite on it.

## Indexing and approximate nearest neighbors

Exact nearest-neighbor search over millions of vectors is too slow, so production systems use **approximate nearest neighbor** (ANN) indexes that trade a little recall for orders-of-magnitude speed. **HNSW** (hierarchical navigable small-world graphs) is the dominant ANN index: a multi-layer proximity graph you descend greedily, with excellent recall and latency, tuned through `M` and `efSearch`. **IVF** and **IVF-PQ** cluster vectors and search only nearby clusters, with product quantization compressing vectors to save memory. The index itself lives in pgvector (Postgres), Qdrant, Milvus, Weaviate, FAISS, or Redis (see [Distributed Data & Caching](../04-distributed-data-and-caching/)), and it is sharded, replicated, and cached like any other data.

## Lexical retrieval

Dense retrieval is half the story; the other half of hybrid search is lexical, and it should not be skipped. **Bag-of-words** represents text as term counts, ignoring order: sparse, high-dimensional, and exact-term. **TF-IDF** weights a term by its frequency in the document times its rarity across the corpus, so common words count less and distinctive words more. **BM25** is the refined, dominant lexical ranker: TF-IDF with **saturation** (`k1`, so extra occurrences matter less and less) and **length normalization** (`b`, so long documents are not rewarded just for having more words). It is decades old and still a brutally strong baseline.

The reason to keep lexical retrieval is that dense retrieval misses **exact matches**: IDs, error codes, names, rare jargon, acronyms. BM25 nails those. The two fail in opposite directions, which is exactly why you fuse them.

## Hybrid retrieval and fusion

Fusion combines dense, lexical, and sometimes graph results into one ranked list. **Reciprocal Rank Fusion** (RRF) sums $1/(k + \text{rank})$ across the lists. It needs no score calibration, is robust, and is embarrassingly simple, which makes it the default; it is what saige uses to fuse its vector, BM25, and graph retrievers. **Score-based fusion** normalizes and weights each retriever's scores instead, which is more tunable and more fragile.

**Graph retrieval** (GraphRAG) pulls a connected subgraph of facts from a [knowledge graph](../04-distributed-data-and-caching/), resolving entities to documents. It adds multi-hop reasoning and relationships that flat chunks cannot express. saige fuses it as a third retriever and tracks **temporal validity** (`ValidAt`/`InvalidAt`) on relations so retrieval is correct at a point in time. The general pattern is breadth first, precision last: retrieve broadly from several complementary sources, fuse, then rerank.

## What production retrieval adds

Several concerns separate a demo from production. **Metadata filtering** constrains retrieval by structured fields (source, type, recency, language, tenant, and access permissions), combined with vector or lexical search as a pre-filter or post-filter; it is often the difference between a right and a wrong answer. **Authorization-aware retrieval** goes further: a user must retrieve only the chunks they are allowed to see, so permissions are carried as ACLs or labels on each chunk and applied at retrieval time (see [Authorization & Access Control](../08-authorization-and-access-control/)). This "secure RAG" is a hard, mandatory requirement in the enterprise.

**Context assembly** dedupes, orders, and fits candidates into the context budget, adds **citations** saying which chunk supported which claim, and can optionally **compress** context with an LLM (saige does both). **Freshness** means re-embedding and re-indexing when documents change and pinning embedding-model versions, because changing the encoder invalidates the whole index; treat embeddings like a cache with a version key, a [caching](../04-distributed-data-and-caching/) problem in disguise. Finally, **query transforms** (rewriting, expansion, HyDE, which embeds a hypothetical answer, and multi-query) improve recall before retrieval even runs.

## Multimodal retrieval

Multimodal retrieval mostly reduces to text plus a unifying record. Model content as a hierarchy of **Document, Sections, and ContentVariants** (saige's data model), where every variant, whatever its medium (image, table, audio, PDF), carries a **`.Text` representation**: a caption, transcript, OCR output, or extracted text. Then *one* text and vector index searches all of them uniformly. **Joint embedding spaces** such as CLIP's go further, embedding images and text into the *same* space so a text query can retrieve images directly (see [Foundation Models](../../machine-learning/05-foundation-models/)). Getting there takes ingestion plumbing: URI resolution (`file://`, `s3://`), content negotiation, and per-type extractors (OCR, ASR, table parsing) that turn raw files into searchable variants. At generation time you feed the retrieved text variants, and for vision-language models the original media, into a multimodal model.

## Evaluating retrieval

Retrieval has its own metrics, separate from generation; the full treatment is in [LLM Evaluation](../09-llm-evaluation/). **Recall@k** and **Precision@k** ask whether the relevant chunk made the top $k$. **MRR** (mean reciprocal rank) asks how high the first relevant hit was. **NDCG** measures rank quality with graded relevance and position discounting. RAG-specific metrics such as **faithfulness**, **context precision**, and **context recall** (as in Ragas or saige's RAG scorers) ask whether the answer is grounded in the retrieved context and whether the right context was retrieved. A good habit is to change *only* the retriever and measure how answer quality moves.

## Choosing a technique

The survey condenses into a lookup from need to technique:

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

These patterns are everywhere in industry. Microsoft's GraphRAG is a graph and vector hybrid over knowledge graphs. Answer engines such as Perplexity and You.com combine hybrid retrieval, reranking, and citations. Elastic and Vespa put lexical and dense search in one engine with fusion built in. Cohere and Voyage sell bi-encoder embeddings and cross-encoder rerankers as a service. Glean and other enterprise search products depend on permission-filtered, metadata-rich RAG. And [saige](https://github.com/urmzd/saige), a Go SDK for agents, knowledge graphs, and RAG, serves as a reference implementation of RRF over vector, BM25, and graph retrievers with multimodal variants.

The ideas also connect across this book and its neighbors: encoders, fine-tuning, and CLIP come from [Training & Frameworks](../01-training-and-frameworks/) and [Foundation Models](../../machine-learning/05-foundation-models/); the index lives in the stores of [Distributed Data & Caching](../04-distributed-data-and-caching/); permission-filtered retrieval is the subject of [Authorization & Access Control](../08-authorization-and-access-control/); retrieval and faithfulness metrics are measured in [LLM Evaluation](../09-llm-evaluation/); and the generated answer is delivered token by token with [Streaming & SSE](../03-streaming-sse/).

## In this chapter

The three sections build a complete, minimal RAG stack in Go, in pipeline order.

1. [RAG Ingest: Crawler, Chunker, Embedder, and Store](01-rag-ingest.md) builds the index: sources that yield clean text, a chunker that packs to a true token budget, an embedder that cannot misalign rows, fingerprint dedup so re-ingest is cheap, and a crawler hardened against SSRF.
2. [Retrieval: BM25, Flat and IVF Vectors, RRF, and MMR](02-retrieval.md) searches that index lexically and densely, fuses the two lists by rank, diversifies the result, and treats the index file as a byte-exact contract between programs.
3. [Context Assembly, Citations, and the search_docs Tool](03-context-assembly.md) packs hits into a budgeted context of numbered excerpts, numbers citations with one ledger per agent run, reads them back from the answer, and exposes it all as a tool.

## Further reading

- Lewis et al., [the RAG paper](https://arxiv.org/abs/2005.11401) (2020, free).
- Robertson and Zaragoza, [*The Probabilistic Relevance Framework: BM25 and Beyond*](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) (free).
- [pgvector](https://github.com/pgvector/pgvector) (free).
- [Sentence-Transformers](https://www.sbert.net/) and the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) (free).
- Cormack et al., [*Reciprocal Rank Fusion*](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) (2009, free).
- Malkov and Yashunin, [the HNSW paper](https://arxiv.org/abs/1603.09320) (free).
- Microsoft, [GraphRAG](https://arxiv.org/abs/2404.16130) (free).
- [saige](https://github.com/urmzd/saige), a Go reference implementation of multi-retriever fusion, chunking, multimodal variants, and knowledge-graph retrieval.
