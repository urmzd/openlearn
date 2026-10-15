# Distributed Data & Caching

## Overview

- **Primary references**: [Redis docs](https://redis.io/docs/latest/) (free), [Apache Cassandra docs](https://cassandra.apache.org/doc/latest/) (free), DDIA Ch 5-7 (replication, partitioning, transactions)
- **Supplementary**: [Apache AGE](https://age.apache.org/) (graphs on Postgres, free), [pgvector](https://github.com/pgvector/pgvector), [Neo4j docs](https://neo4j.com/docs/), [Citus](https://www.citusdata.com/) / [Vitess](https://vitess.io/) (sharding), [Valkey](https://valkey.io/) (open-source Redis fork), "[Scaling Memcache at Facebook](https://www.usenix.org/system/files/conference/nsdi13/nsdi13-final170_update.pdf)" (the canonical caching-at-scale paper)
- **Prerequisites**: [Data Engineering](../../data-engineering/) (OLTP vs OLAP, storage engines), [System Design](../../systems/01-system-design/) (replication, CAP, consistency)
- **Estimated time**: 3-4 weeks at 8-10 hrs/week

## Key Takeaways

- **Data outgrows one node along three axes**: too *big* (partition it — sharding), too *important to lose* (replicate it), too *slow to read repeatedly* (cache it). Each axis has its own pattern family.
- **OLTP and OLAP are different machines.** Row-oriented for transactions, column-oriented for analytics. Storage layout follows access pattern — this is the foundational dichotomy.
- **Sharding is partitioning by a key**, and the whole difficulty is the key: a bad one creates hot shards; cross-shard queries scatter-gather; rebalancing must move minimal data (consistent hashing).
- **Caching is a deliberate lie for speed, and invalidation is the cost.** Cache-aside, write-through, write-back, and TTL are four different answers to "when does the cached copy stop being trusted?"
- **The engine encodes the question**: Cassandra for write-heavy/always-on, Redis for sub-millisecond in-memory access, Postgres (+ AGE for graphs, + pgvector for embeddings) for transactional and relational, a knowledge graph when the *relationships are the data*.

## How to Study

- Shard a table by a key in Citus/Vitess; write one query that hits a single shard and one that fans out — feel the cost difference, then deliberately create a hot shard.
- Put Redis in front of a slow query with **cache-aside**; measure hit rate and p99. Then break it: cause a stale read, a cache stampede, and a thundering herd, and fix each with the matching pattern.
- Model one domain three ways — relational (Postgres), wide-column (Cassandra), graph (Apache AGE/Cypher) — and run the query each is good at.
- Profile a read path end to end (app → cache → DB) and find where the time actually goes before optimizing.

---

# Concepts & Techniques

## Core Insight

At scale, data outgrows a single box along three independent axes, and each is a separate pattern family: it gets too *big* (partition — sharding), too *important to lose* (replicate), and too *expensive to recompute or re-read* (cache). The art is matching the physical layout to the access pattern — row vs column, hashed vs ranged, durable vs in-memory — and then accepting that every copy you make (a replica, a cache) is a consistency problem you've chosen to take on for performance. Caching is the sharpest version of that bargain: you trade correctness-over-time for latency, and *invalidation* is the bill.

## 1. OLTP vs OLAP

**The foundational storage dichotomy**

| | OLTP | OLAP |
|---|------|------|
| Workload | Many small reads/writes | Few huge scans/aggregations |
| Layout | **Row-oriented** | **Column-oriented** |
| Optimized for | Point lookups, updates, transactions | Bulk scans, group-by, joins |
| Latency | Milliseconds, per-request | Seconds-minutes, per-query |
| Examples | Postgres, MySQL, Cassandra | Snowflake, BigQuery, ClickHouse, DuckDB |

**Pattern**: columnar wins analytics because an aggregation reads a few columns over millions of rows — store each column contiguously and you read only what you need, compress it well, and vectorize. Row wins transactions because an entity is read/written whole. **Don't run analytics on the OLTP store**; ELT into a warehouse (see [Data Engineering](../../data-engineering/)).

## 2. Sharding (Horizontal Partitioning)

**Splitting data across nodes by a key**

**Patterns**:
- **Hash sharding**: `hash(key) → shard`. Even spread, kills range scans.
- **Range sharding**: contiguous key ranges. Great for ranges, risks hot ranges.
- **Consistent hashing**: nodes and keys on a ring so adding/removing a node moves *minimal* data — the basis of Cassandra/Dynamo and of distributed caches.

**The hard parts (all key-choice problems)**:
- **Hot shards / skew**: a low-cardinality or celebrity key overloads one shard — pick a high-cardinality, evenly-distributed key, sometimes salted.
- **Cross-shard queries**: spanning shards forces scatter-gather and merge — co-locate data queried together.
- **Rebalancing**: virtual nodes / hash slots make adding capacity incremental and online.
- **Cross-shard transactions** need 2PC or sagas — expensive; avoid spanning shards in one transaction.
- **Tooling**: Citus (Postgres), Vitess (MySQL), native in Cassandra/Mongo/CockroachDB/Spanner.

## 3. Replication & Consistency (Brief)

- **Replication** = copies of a partition for availability and read-scaling. Leader-follower (Postgres) vs leaderless quorum (Cassandra/Dynamo).
- **Quorum**: `R + W > N` for strong-ish consistency; relax for speed.
- **CAP / PACELC**: under a partition, choose consistency *or* availability; even without one, trade latency vs consistency. Depth in [System Design](../../systems/01-system-design/) and DDIA.

## 4. In-Memory Data Stores & Caching (Redis)

**Trading durability and memory for sub-millisecond latency**

**Why in-memory**: RAM is ~100-1000× faster than disk/SSD and far faster than recomputing a result. An in-memory store (**Redis**, Valkey, Memcached) keeps hot data in RAM so reads skip the slow path entirely.

**Redis is "a network-attached data-structure server"**, not just a key-value cache:
- **Structures**: strings, hashes, lists, sets, sorted sets (leaderboards), bitmaps, HyperLogLog (cardinality), streams (a log), geospatial. Picking the right structure *is* the optimization.
- **Beyond caching**: rate limiters (token bucket via `INCR`+TTL), distributed locks (`SETNX` / Redlock — with caveats), session stores, pub/sub, queues (streams), and a vector index (Redis as a vector store for RAG).
- **Persistence is optional**: RDB snapshots + AOF log let Redis survive restarts, but it's tuned for speed, not durability — treat it as a cache or fast tier, not the system of record.
- **Eviction policies**: when memory fills, evict by `LRU`, `LFU`, `TTL`, or `random` — choosing this wrong silently tanks hit rate.

## 5. Caching Patterns

**"There are only two hard things in CS: cache invalidation and naming things."**

**Read patterns**:
- **Cache-aside (lazy loading)**: app checks cache; on miss, reads DB and populates cache. The default. Simple, resilient (cache down ≠ app down), but first read is slow and data can go stale.
- **Read-through**: the cache itself loads from the DB on miss (library/provider does it). Cleaner app code, couples cache to DB.

**Write patterns** (these *are* the invalidation strategy):
- **Write-through**: write cache and DB synchronously. Cache always fresh; writes are slower.
- **Write-back (write-behind)**: write cache now, flush to DB asynchronously. Fast writes, risk of loss on crash before flush.
- **Write-around**: write only the DB, let the cache populate on the next read. Avoids caching write-only data, at the cost of a guaranteed first-read miss.

**The invalidation question** — every pattern is a different answer to "when does the cached copy stop being trusted?":
- **TTL (expiry)**: trust it for N seconds, then re-fetch. Simple, eventually consistent, the workhorse — bounds staleness but doesn't eliminate it.
- **Explicit invalidation**: delete/update the key on write. Precise but easy to miss a write path → permanent staleness (the classic bug).
- **Versioning / key-by-content**: bake a version or content hash into the key; old versions just age out. Sidesteps invalidation entirely.

## 6. Caching Failure Modes (and Their Patterns)

| Failure | What happens | Pattern fix |
|---------|--------------|-------------|
| **Cache stampede / dogpile** | A hot key expires; thousands of requests miss and hit the DB at once | Request coalescing (single-flight), probabilistic early expiry, locks |
| **Thundering herd** | Many keys expire together (same TTL) | Jitter the TTLs |
| **Cache penetration** | Queries for non-existent keys always miss → DB hammered | Cache the negative result; Bloom filter in front |
| **Hot key** | One key gets disproportionate traffic, overloads its shard | Local (client) cache layer, replicate the key |
| **Stale read** | Cache not invalidated after a write | Match the write pattern to the consistency need; TTL as a backstop |
| **Cold start** | Empty cache after deploy → all misses | Warm the cache; gradual rollout |

**Meta-pattern**: a cache changes your consistency model whether you admit it or not. Decide explicitly how stale you can tolerate, then pick the pattern — don't discover it in an incident.

## 7. Cassandra (Wide-Column, Write-Optimized)

- **Architecture pattern**: masterless ring + consistent hashing — no SPOF, linear scaling, multi-DC replication.
- **Storage pattern**: **LSM-tree** (memtable → SSTables → compaction) — sequential, write-optimized ingest (same family as [DE storage](../../data-engineering/02-storage-warehousing/)).
- **Modeling pattern**: **query-first** — design tables around the queries (no ad-hoc joins). The **partition key** is your shard key.
- **Tunable consistency** per query (`ONE`/`QUORUM`/`ALL`). Use for time-series, event logs, feature stores, always-on multi-region writes. Avoid for ad-hoc analytics or multi-row transactions. Relatives: ScyllaDB, DynamoDB, Bigtable/HBase.

## 8. Knowledge Graphs & Graph Databases

**When the relationships *are* the data**

- **Graph model**: nodes + typed edges + properties; traversals ("friends-of-friends", "what depends on this") are index-free adjacency, not many self-joins.
- **Knowledge graph**: entities + facts (often RDF triples) encoding a domain — powers entity resolution, recommendations, and **GraphRAG** (retrieve a connected subgraph of facts for multi-hop, grounded answers — see [Retrieval & RAG](../07-retrieval-and-rag/)).
- **Engines**: **Neo4j** (native, Cypher), **Apache AGE** (Cypher *inside Postgres*, alongside relational tables and pgvector — one DB for relational + vector + graph), RDF triple stores (SPARQL).

## 9. Postgres as the Swiss-Army Database

One reason Postgres dominates platform stacks: extensions make it many databases — **pgvector** (embeddings + ANN for RAG), **Apache AGE** (graphs/Cypher), **Citus** (sharding), **TimescaleDB** (time-series) — atop solid OLTP transactions. Pragmatic default: start with Postgres + extensions + Redis for caching; reach for Cassandra (write-scale/HA), a dedicated vector DB (Qdrant/Milvus), or Neo4j (deep graph) only when Postgres's limits actually bite.

## 10. Temporal Data Modeling

**Modeling time itself** (distinct from the Temporal *workflow* engine in [topic 05](../05-durable-orchestration-and-workers/))

- **Bitemporal data**: track **valid time** (true in the world) vs **transaction time** (recorded in the system) — answer "what did we believe on March 1, as of what we knew Feb 28?"
- **Slowly Changing Dimensions (SCD Type 2)**: keep history via validity-range rows, not overwrites.
- **Event sourcing**: store the immutable event log; current state is a fold over it — gives audit, replay, and time-travel for free (and is exactly how durable orchestration recovers).

---

## Patterns Worth Internalizing

These outlive every product in this topic:

- **Partition by a high-cardinality key; consistent-hash to rebalance cheaply.** (Sharding, Cassandra, distributed caches all reuse this.)
- **A cache is a consistency tradeoff, not a free speedup.** Pick read pattern (cache-aside) × write pattern (through/back/around) × invalidation (TTL/explicit/versioned) deliberately.
- **Coalesce, jitter, and guard against the empty case** — stampede, thundering herd, and penetration are the three caching incidents you will meet.
- **Layout follows access pattern** — row vs column, in-memory vs on-disk, normalized vs query-first. Choose the engine that makes your dominant query cheap.
- **Every copy is a liability** — replicas and caches buy performance with a consistency obligation; name the staleness budget up front.

## Connections to Other Tracks

| Concept | Connected Track | Application |
|---------|-----------------|-------------|
| OLTP/OLAP, storage engines, ELT, SCDs | [Data Engineering](../../data-engineering/) | Where this storage knowledge is rooted |
| Replication, CAP, partitioning, consensus | [System Design](../../systems/01-system-design/) | The distributed-systems foundations |
| Bloom filters, HyperLogLog, consistent hashing | [Probabilistic Structures](../../algorithms/15-probabilistic-structures/) | Cache penetration guards, cardinality |
| Vector stores, embeddings, ANN | [Training & Frameworks](../01-training-and-frameworks/) | Where embeddings live and are queried |
| Graph traversal (BFS/DFS, shortest path) | [Algorithms — Graphs](../../algorithms/06-graphs/) | What graph databases execute |
| Durable execution, event sourcing, replay | [Orchestration & Workers](../05-durable-orchestration-and-workers/) | The log as source of truth |
| Profiling the read path, tail latency | [Observability](../../systems/04-observability/) | Finding the real bottleneck |

## How Companies Apply These Patterns

| Company | The pattern they lean on | Instance |
|---------|--------------------------|----------|
| Meta | Look-aside cache at scale + invalidation | Memcache tier ("Scaling Memcache") |
| Discord / X | Wide-column + consistent hashing | Cassandra/ScyllaDB for messages/timelines |
| Uber / Netflix | Sharding + multi-region replication | Cassandra, sharded datastores |
| Stripe | Sharded OLTP + idempotent writes | Sharded Postgres/Mongo |
| Palantir | Knowledge graph + bitemporal | Entity resolution, point-in-time |
| Twitter (historic) | Caching the timeline (fan-out on write) | Redis-backed timelines |

## Chapters

<!-- ol:chapters -->
| # | Module | Chapter | Kind | Pass |
|---|---|---|---|---|
| 1 | `gw.06` | [Response cache: LRU + TTL, singleflight, tenant-scoped keys](01-response-cache.md) | build | 7 |
<!-- /ol:chapters -->
