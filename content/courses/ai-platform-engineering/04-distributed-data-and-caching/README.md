# Distributed Data and Caching

At scale, data outgrows a single machine along three independent axes, and each axis has its own family of patterns. Data gets too *big* for one node, so you partition it (sharding). It gets too *important to lose*, so you replicate it. And it gets too *expensive to recompute or re-read*, so you cache it. The art is matching the physical layout to the access pattern (row or column, hashed or ranged, durable or in-memory) and then accepting that every copy you make, whether a replica or a cache entry, is a consistency problem you have chosen to take on in exchange for performance.

Caching is the sharpest version of that bargain. A cache is a deliberate lie told for speed: you trade correctness over time for latency, and *invalidation* is the bill. Every caching pattern is a different answer to one question, "when does the cached copy stop being trusted?", and the incidents you will meet in production (stampedes, thundering herds, penetration, stale reads) are what happens when that question was answered by accident.

The chapter starts with the foundational storage split between OLTP and OLAP, then covers sharding, replication, in-memory stores such as Redis, the caching patterns, and caching's failure modes. It then surveys the engines that encode different questions, Cassandra, graph databases, and Postgres with its extensions, and closes with temporal data modeling. The section that follows the chapter opening turns the caching ideas into running code: a tenant-scoped response cache for the gateway.

> **Learning objectives**
> - Explain why OLTP stores are row-oriented and OLAP stores column-oriented, and why analytics should not run on the transactional store.
> - Choose a shard key and partitioning scheme (hash, range, consistent hashing), and name the problems a bad key causes.
> - Compare cache-aside, read-through, write-through, write-back, and write-around, and the TTL, explicit, and versioned invalidation strategies.
> - Diagnose and fix the classic caching failures: stampede, thundering herd, penetration, hot keys, stale reads, and cold starts.
> - Pick an engine (Postgres with extensions, Redis, Cassandra, a graph database) from the dominant query.

## Before you begin

This chapter assumes the storage background in [Data Engineering](../../data-engineering/): OLTP versus OLAP and how storage engines work. It also assumes the distributed-systems foundations of replication, CAP, and consistency from [System Design](../../systems/01-system-design/) in *Systems & Architecture*. The response-cache section additionally builds on the gateway's [Server skeleton](../12-gateway/01-server-skeleton.md) and on [API keys and scopes](../08-authorization-and-access-control/01-api-keys-and-scopes.md), which supply the request chain and the tenant.

## In this chapter

The chapter opening below covers the concepts. One section then builds a production component from them:

- [Response cache: LRU, TTL, singleflight, and tenant-scoped keys](01-response-cache.md): the gateway's cache for deterministic requests, with a canonical request key, per-tenant isolation, request coalescing, and byte-exact stream replay.

## OLTP and OLAP

The foundational storage dichotomy is between transactional and analytical workloads, and they are different machines:

| | OLTP | OLAP |
|---|------|------|
| Workload | Many small reads/writes | Few huge scans/aggregations |
| Layout | **Row-oriented** | **Column-oriented** |
| Optimized for | Point lookups, updates, transactions | Bulk scans, group-by, joins |
| Latency | Milliseconds, per-request | Seconds-minutes, per-query |
| Examples | Postgres, MySQL, Cassandra | Snowflake, BigQuery, ClickHouse, DuckDB |

The layout follows the access pattern. Columnar storage wins analytics because an aggregation reads a few columns over millions of rows: store each column contiguously and you read only what you need, compress it well, and vectorize the computation. Row storage wins transactions because an entity is read and written whole. The practical rule is not to run analytics on the OLTP store; extract and load into a warehouse instead, as [Data Engineering](../../data-engineering/) describes.

## Sharding

When the data is too big for one node, you split it across nodes by a key. There are three basic schemes. **Hash sharding** maps `hash(key)` to a shard, which spreads data evenly but destroys range scans. **Range sharding** assigns contiguous key ranges to shards, which is great for range queries but risks hot ranges. **Consistent hashing** places both nodes and keys on a ring so that adding or removing a node moves only a minimal amount of data; it is the basis of Cassandra and Dynamo and of distributed caches.

The whole difficulty of sharding is the choice of key, and every hard part is a key-choice problem. **Hot shards and skew** come from a low-cardinality or celebrity key that overloads one shard; pick a high-cardinality, evenly distributed key, sometimes salted. **Cross-shard queries** force a scatter-gather and merge, so co-locate data that is queried together. **Rebalancing** should be incremental and online, which virtual nodes or hash slots make possible. **Cross-shard transactions** need two-phase commit or sagas, both expensive, so avoid spanning shards in one transaction.

You rarely build this yourself. Citus shards Postgres, Vitess shards MySQL, and sharding is native in Cassandra, MongoDB, CockroachDB, and Spanner.

## Replication and consistency

Replication keeps copies of each partition for availability and read scaling. Postgres uses a leader-follower design; Cassandra and Dynamo are leaderless and use quorums. With $N$ replicas, writes acknowledged by $W$ of them, and reads consulting $R$, choosing $R + W > N$ guarantees that every read overlaps the latest write, giving strong-ish consistency; relax it for speed. CAP and PACELC frame the tradeoff: under a network partition you choose consistency *or* availability, and even without a partition you trade latency against consistency. The depth is in [System Design](../../systems/01-system-design/) and in Kleppmann's *Designing Data-Intensive Applications*.

## In-memory stores and Redis

The third axis, data too slow to re-read, is where caching begins, and the tool is memory. RAM is roughly 100 to 1000 times faster than disk or SSD, and far faster than recomputing a result. An in-memory store such as **Redis**, Valkey, or Memcached keeps hot data in RAM so reads skip the slow path entirely, trading durability and memory for sub-millisecond latency.

Redis is best understood as a network-attached data-structure server, not just a key-value cache. It offers strings, hashes, lists, sets, sorted sets (for leaderboards), bitmaps, HyperLogLog (for cardinality estimates), streams (a log), and geospatial indexes, and picking the right structure is the optimization. That range of structures lets Redis do much more than cache: rate limiters (a token bucket built from `INCR` and a TTL), distributed locks (`SETNX` or Redlock, with caveats), session stores, pub/sub, queues (streams), and even a vector index, making Redis a vector store for RAG.

Two operational facts follow. **Persistence is optional**: RDB snapshots and the AOF log let Redis survive restarts, but it is tuned for speed, not durability, so treat it as a cache or fast tier, never the system of record. And **eviction policy** decides what happens when memory fills: evict by LRU, LFU, TTL, or at random. Choosing wrongly silently tanks the hit rate.

## Caching patterns

"There are only two hard things in computer science: cache invalidation and naming things." The patterns divide into how you read, how you write, and how you decide a copy is no longer trustworthy.

On the read side, **cache-aside** (lazy loading) is the default: the application checks the cache, and on a miss it reads the database and populates the cache. It is simple and resilient, because the cache being down does not take the application down, but the first read is slow and data can go stale. **Read-through** moves the loading into the cache itself (a library or provider loads from the database on a miss), which makes application code cleaner but couples the cache to the database.

The write patterns are, in effect, the invalidation strategy. **Write-through** writes the cache and the database synchronously, so the cache is always fresh but writes are slower. **Write-back** (write-behind) writes the cache now and flushes to the database asynchronously, which makes writes fast but risks losing data if the cache crashes before the flush. **Write-around** writes only the database and lets the cache populate on the next read, which avoids caching data that is written but never read, at the cost of a guaranteed first-read miss.

Underneath both sits the invalidation question itself, and there are three answers. **TTL expiry** trusts the copy for $N$ seconds and then re-fetches; it is simple, eventually consistent, and the workhorse, and it bounds staleness without eliminating it. **Explicit invalidation** deletes or updates the key on every write; it is precise, but it is easy to miss one write path, which produces permanent staleness, the classic caching bug. **Versioning**, or keying by content, bakes a version or content hash into the key so old versions simply age out, sidestepping invalidation entirely. The response cache in the next section uses this last approach, folding a configuration revision into every key.

## When caches fail

Each caching incident you will meet has a matching pattern fix:

| Failure | What happens | Pattern fix |
|---------|--------------|-------------|
| **Cache stampede / dogpile** | A hot key expires; thousands of requests miss and hit the DB at once | Request coalescing (single-flight), probabilistic early expiry, locks |
| **Thundering herd** | Many keys expire together (same TTL) | Jitter the TTLs |
| **Cache penetration** | Queries for non-existent keys always miss → DB hammered | Cache the negative result; Bloom filter in front |
| **Hot key** | One key gets disproportionate traffic, overloads its shard | Local (client) cache layer, replicate the key |
| **Stale read** | Cache not invalidated after a write | Match the write pattern to the consistency need; TTL as a backstop |
| **Cold start** | Empty cache after deploy → all misses | Warm the cache; gradual rollout |

The three you are most certain to meet are stampede, thundering herd, and penetration: coalesce identical misses, jitter expiry times, and guard against the empty case. The meta-pattern is that a cache changes your consistency model whether you admit it or not. Decide explicitly how much staleness you can tolerate, then pick the pattern; do not discover your staleness budget in an incident.

## Cassandra

Different engines encode different questions, and Cassandra's question is how to take writes everywhere, always. Architecturally it is a masterless ring with consistent hashing, so there is no single point of failure, it scales linearly, and it replicates across data centers. Its storage is an **LSM tree**: writes land in a memtable, flush to SSTables, and are merged by compaction, giving sequential, write-optimized ingest from the same family of engines covered in [Storage & Warehousing](../../data-engineering/02-storage-warehousing/) in *Data Engineering*.

Modeling in Cassandra is **query-first**: you design tables around the queries you will run, because there are no ad hoc joins, and the partition key is your shard key. Consistency is tunable per query (`ONE`, `QUORUM`, `ALL`). Use Cassandra for time series, event logs, feature stores, and always-on multi-region writes; avoid it for ad hoc analytics or multi-row transactions. Its relatives are ScyllaDB, DynamoDB, and Bigtable and HBase.

## Graph databases and knowledge graphs

Some data is mostly relationships, and when the relationships *are* the data, a graph model fits. A graph is nodes, typed edges, and properties; traversals such as "friends of friends" or "what depends on this" use index-free adjacency rather than many self-joins. A **knowledge graph** encodes a domain as entities and facts, often RDF triples, and powers entity resolution, recommendations, and **GraphRAG**, which retrieves a connected subgraph of facts to give multi-hop, grounded answers; see [Retrieval and RAG](../07-retrieval-and-rag/).

The main engines are **Neo4j** (native, queried with Cypher), **Apache AGE** (Cypher inside Postgres, alongside relational tables and pgvector, so one database serves relational, vector, and graph queries), and RDF triple stores queried with SPARQL.

## Postgres as the Swiss Army database

Apache AGE hints at one reason Postgres dominates platform stacks: extensions make it many databases at once. **pgvector** adds embeddings and approximate-nearest-neighbor search for RAG, **Apache AGE** adds graphs and Cypher, **Citus** adds sharding, and **TimescaleDB** adds time series, all on top of solid OLTP transactions. The pragmatic default is to start with Postgres plus extensions, with Redis for caching, and reach for Cassandra (for write scale and high availability), a dedicated vector database (Qdrant or Milvus), or Neo4j (for deep graph work) only when Postgres's limits actually bite.

## Temporal data modeling

The last topic is modeling time itself, which is distinct from the Temporal *workflow engine* in [Durable Orchestration and Workers](../05-durable-orchestration-and-workers/). **Bitemporal data** tracks two times: **valid time**, when a fact was true in the world, and **transaction time**, when the system recorded it. With both, you can answer "what did we believe on March 1, as of what we knew on February 28?" **Slowly changing dimensions (SCD Type 2)** keep history with validity-range rows instead of overwriting. **Event sourcing** stores the immutable log of events and computes current state as a fold over it, which gives audit, replay, and time travel for free, and is exactly how durable orchestration recovers after a crash.

## The patterns that last

The products in this chapter will change; these patterns will not. Partition by a high-cardinality key and use consistent hashing to rebalance cheaply; sharding, Cassandra, and distributed caches all reuse that idea. Treat a cache as a consistency tradeoff, not a free speedup, choosing the read pattern, the write pattern, and the invalidation strategy deliberately. Coalesce, jitter, and guard against the empty case. Let layout follow the access pattern, and choose the engine that makes your dominant query cheap. And remember that every copy is a liability: replicas and caches buy performance with a consistency obligation, so name the staleness budget up front.

Industry examples bear this out. Meta runs a look-aside cache with invalidation at enormous scale, described in "Scaling Memcache at Facebook." Discord and X use wide-column stores with consistent hashing (Cassandra and ScyllaDB) for messages and timelines, and Uber and Netflix combine sharding with multi-region replication. Stripe pairs sharded OLTP with idempotent writes. Palantir builds knowledge graphs with bitemporal, point-in-time data for entity resolution, and Twitter historically cached timelines in Redis with fan-out on write.

## Where this leads

The [response cache](01-response-cache.md) that follows applies this chapter's caching ideas to the gateway: versioned keys instead of invalidation, singleflight against stampedes, and TTLs to bound staleness. Elsewhere in this book, [Training and Frameworks](../01-training-and-frameworks/) produces the embeddings that vector stores hold, and [Durable Orchestration and Workers](../05-durable-orchestration-and-workers/) treats the event log as the source of truth for durable execution and replay. In other books, [Data Engineering](../../data-engineering/) roots the storage knowledge here (OLTP and OLAP, storage engines, ELT, SCDs); [System Design](../../systems/01-system-design/) covers replication, CAP, partitioning, and consensus; [Probabilistic Data Structures Patterns](../../algorithms/15-probabilistic-structures/) in *Algorithm Mastery* covers the Bloom filters, HyperLogLog, and consistent hashing used against cache penetration and for cardinality; the [graphs chapter](../../algorithms/06-graphs/) of the same book covers the traversals and shortest paths that graph databases execute; and [Observability](../../systems/04-observability/) covers profiling the read path to find the real bottleneck.

### Further reading

- [Redis docs](https://redis.io/docs/latest/) and [Apache Cassandra docs](https://cassandra.apache.org/doc/latest/) (free), and Kleppmann, *Designing Data-Intensive Applications*, Chapters 5 to 7 (replication, partitioning, transactions): the primary references.
- [Apache AGE](https://age.apache.org/) for graphs on Postgres (free), [pgvector](https://github.com/pgvector/pgvector), and the [Neo4j docs](https://neo4j.com/docs/).
- [Citus](https://www.citusdata.com/) and [Vitess](https://vitess.io/) for sharding.
- [Valkey](https://valkey.io/), the open-source Redis fork.
- "[Scaling Memcache at Facebook](https://www.usenix.org/system/files/conference/nsdi13/nsdi13-final170_update.pdf)", the canonical paper on caching at scale.

## Key terms

- **OLTP / OLAP**: transactional workloads of many small row-oriented operations / analytical workloads of large column-oriented scans.
- **Shard key**: the field whose value decides which partition holds a record.
- **Consistent hashing**: placing nodes and keys on a ring so membership changes move minimal data.
- **Quorum**: requiring $W$ write and $R$ read acknowledgements out of $N$ replicas, with $R + W > N$ for overlapping reads and writes.
- **Cache-aside**: the application reads the cache, and on a miss loads from the database and fills the cache.
- **Write-through / write-back / write-around**: write cache and database together / write cache now and database later / write only the database.
- **Cache stampede**: many simultaneous misses on one expired hot key, all hitting the backing store.
- **Cache penetration**: repeated lookups of keys that do not exist, which always miss.
- **Bitemporal data**: records that carry both valid time and transaction time.

## Summary

- Data outgrows one node along three axes: too big (shard it), too important to lose (replicate it), too slow to re-read (cache it).
- OLTP and OLAP are different machines; storage layout follows the access pattern.
- Sharding's difficulty is the key: bad keys create hot shards, cross-shard queries scatter-gather, and rebalancing should use consistent hashing.
- A cache is a deliberate lie for speed; cache-aside, write-through, write-back, write-around, TTL, and versioned keys are different answers to when the copy stops being trusted, and stampede, herd, and penetration are the incidents to design against.
- The engine encodes the question: Cassandra for write-heavy, always-on data; Redis for sub-millisecond access; Postgres with AGE and pgvector for relational, graph, and vector work; a knowledge graph when relationships are the data.

## Review questions

1. Why does a columnar store answer `SELECT avg(price) FROM orders` faster than a row store?
   <details><summary>Answer</summary>The query needs one column over many rows. A columnar store reads only that column, stored contiguously and compressed well, and can vectorize the aggregation; a row store reads every field of every row.</details>

2. You shard user events by `country`. What goes wrong, and what is a better key?
   <details><summary>Answer</summary>`country` is low-cardinality and skewed, so a few countries create hot shards. A high-cardinality, evenly distributed key such as `user_id` (salted if some users are extremely active) spreads the load.</details>

3. With $N = 3$ replicas, which $(R, W)$ choices give overlapping reads and writes?
   <details><summary>Answer</summary>Any with $R + W > 3$: for example $R = 2, W = 2$, or $R = 1, W = 3$, or $R = 3, W = 1$. $R = 1, W = 1$ does not overlap and can return stale data.</details>

4. A popular cached answer expires and the database falls over. Name the failure and two fixes.
   <details><summary>Answer</summary>A cache stampede. Coalesce the identical misses so one request refetches while the rest wait (single-flight), or refresh probabilistically before expiry, or take a lock on refetch.</details>

5. Why is versioning the key often better than explicit invalidation?
   <details><summary>Answer</summary>Explicit invalidation must run on every write path, and missing one leaves permanently stale data. With the version or content hash in the key, a change makes old entries unreachable at once, and they age out of the cache without any invalidation step.</details>

## Exercises

- Shard a table by a key in Citus or Vitess, write one query that hits a single shard and one that fans out, and compare their cost; then deliberately create a hot shard.
- Put Redis in front of a slow query with cache-aside and measure the hit rate and p99. Then cause a stale read, a cache stampede, and a thundering herd, and fix each with its matching pattern.
- Model one domain three ways (relational in Postgres, wide-column in Cassandra, graph in Apache AGE with Cypher) and run the query each is best at.
- Profile a read path end to end (application, cache, database) and find where the time actually goes before optimizing.
