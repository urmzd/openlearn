# System Design

System design is about tradeoffs, not optimal solutions: every choice has costs, and the skill is in naming them. Almost every design problem reduces to three questions. What are the access patterns? What are the scale requirements? Which consistency and availability tradeoffs are acceptable? The answers to those three questions determine most of the architecture, and the rest of this chapter is about asking them well.

The chapter has two halves. This opening surveys the building blocks you combine to answer those questions: scaling, databases, distributed-systems primitives, messaging, and API design, followed by the classic design problems that exercise them. The three sections that follow put the building blocks to work on a real system. Each is a design review of the system built across these books, judged against the same eight-line rubric: first the serving platform, then the control plane, then the whole system. Every design starts with requirements (functional and non-functional) and back-of-envelope arithmetic, and the reviews show what that looks like when the arithmetic is wrong by a factor of sixteen.

System design is also the interview topic asked at every Staff+ level, and those interviews test your ability to navigate ambiguity and make justified decisions rather than recall memorized architectures. The same is true of a real design review.

> **Learning objectives**
> - Choose between vertical and horizontal scaling, caching layers, and sharding schemes for a given access pattern.
> - Explain CAP, replication topologies, partitioning, consensus, and distributed transactions, and when each applies.
> - Pick a messaging model, an API style, a rate-limiting scheme, and a pagination strategy with stated tradeoffs.
> - Write and review a design document against an eight-line rubric, with capacity and error-budget arithmetic.
> - Trace failure modes across layers and back each detection with evidence.

## Before you begin

The building blocks assume the data structures and analysis from *[Programming and Algorithms](../../programming-and-algorithms/README.md)*. The design reviews additionally assume you know the system they review: the serving engine from *AI Engineering*, the gateway and durable control plane from *AI Engineering*, and the deployment and observability from *Systems Engineering* and the [Observability](../11-observability/README.md) chapter of this book. Each review lists exactly which sections it draws on.

## Scaling

There are two ways to add capacity. **Vertical scaling** means a bigger machine: simple, but it has a ceiling, and it is a good fit for databases. **Horizontal scaling** means more machines: more complex, but effectively unlimited, and a good fit for stateless services. Once you have more than one instance, a **load balancer** spreads traffic across them, using round-robin, least connections, or consistent hashing, at layer 4 (TCP) or layer 7 (HTTP).

Caching sits at every level of the stack: browser, CDN, reverse proxy, application, database. Adding a cache is easy; cache invalidation is the hard part. A **CDN** serves static content from edge locations, either pulling from the origin on demand or having content pushed to it, and is controlled by `Cache-Control` headers. When a single database can no longer hold the data or the write load, you **shard** it, partitioning horizontally by range, by hash, or through a directory. Resharding later is painful, so the choice deserves care up front.

All of this rests on back-of-envelope numbers you should know cold: QPS for common services (100K to 1M), storage per record (about 1 KB), monthly storage growth, and the read/write ratio.

## Databases

SQL databases give you ACID transactions, joins, and strong consistency; PostgreSQL is the default choice. NoSQL stores trade those for BASE semantics, eventual consistency, and horizontal scaling, and come in several shapes: key-value (DynamoDB), document (MongoDB), wide-column (Cassandra), and graph (Neo4j). The rule of thumb is SQL for transactions and complex queries, NoSQL for massive scale, flexible schemas, or specific access patterns.

The **CAP theorem** frames the central tradeoff: during a network partition, you choose consistency (CP, refuse stale reads) or availability (AP, serve stale reads). You cannot have both. **Replication** comes in three topologies: single-leader (simple, one write path), multi-leader (for multiple data centers), and leaderless (Dynamo-style quorum reads and writes). **Partitioning** by range is good for scans but risks hot spots; partitioning by hash spreads load evenly but gives up range queries.

## Distributed-systems primitives

Beyond a single database, a handful of primitives recur. **Consensus** protocols guarantee agreement among nodes: Raft is the understandable one, Paxos the original, and they underlie etcd and ZooKeeper. **Distributed transactions** come in two styles: two-phase commit, which blocks and makes its coordinator a single point of failure, and sagas, which use compensating transactions and accept eventual consistency. **Vector clocks** and **Lamport timestamps** establish causal ordering without synchronized clocks. **Consistent hashing** distributes keys across nodes so that adding or removing a node remaps only about $K/N$ of the $K$ keys; DynamoDB and Cassandra use it. **Gossip protocols** propagate state epidemic-style until every node has it, as in Cassandra and SWIM.

## Messaging and events

Message queues decouple producers from consumers. Kafka is log-based and ordered, RabbitMQ is a traditional broker with rich routing, and SQS is managed. Two patterns build on them. **Event sourcing** stores events rather than state and rebuilds state by replaying them, which gives you an audit log and time travel for free. **CQRS** separates the read and write models: writes go to an event store and are projected into read-optimized views. Exactly-once delivery is impossible in general; you achieve effectively-once processing with idempotent consumers plus deduplication, which is exactly the pattern the control-plane review checks.

## API design

REST is resource-oriented, uses HTTP verbs, and is stateless; it suits CRUD but suffers from over-fetching and under-fetching. gRPC uses a binary protocol (protobuf) with streaming and strong typing, and suits internal service-to-service calls. GraphQL lets the client specify exactly the data it needs, which helps with complex nested data at the cost of complexity in server-side resolvers.

Three cross-cutting concerns apply to any style. **Rate limiting** uses a token bucket or a sliding window, keyed per user, per IP, or per API key, and returns 429 with a `Retry-After` header. **Idempotency** for POST uses an idempotency key, which is critical for payments: in Stripe's approach the client generates the key and the server deduplicates. **Pagination** should be cursor-based, which stays consistent under concurrent writes, rather than offset-based, which skips items when the data changes underneath it; keyset pagination is the database form.

## Classic design problems

The classic problems are worth studying two or three a week, each the same way: estimate scale first, then design, then discuss tradeoffs, and practice explaining the result in fifteen minutes. Each one stresses a different combination of the building blocks above.

A **URL shortener** turns on hash-based versus counter-based ID generation, base62 encoding, read-heavy caching, and an analytics pipeline. A **chat system** such as WhatsApp needs WebSocket connections, a message queue per user, delivery receipts, a presence service, and end-to-end encryption. A **news feed** such as Twitter's uses fan-out on write (push) for users with few followers and fan-out on read (pull) for celebrities, and in practice a hybrid of the two. **Video streaming** such as YouTube needs an upload pipeline that transcodes to multiple resolutions, a CDN for delivery, and adaptive bitrate streaming (HLS or DASH). A **payment system** such as Stripe's depends on idempotency, exactly-once processing, a double-entry ledger, reconciliation, and PCI compliance. A **key-value store** combines consistent hashing, replication, quorum reads and writes, conflict resolution (last-writer-wins or vector clocks), and compaction. A **search engine** needs a crawler, an inverted index, ranking (TF-IDF, then PageRank, then learned models), query processing, and caching.

Different companies favor different problems, which tells you where their systems hurt. The following table summarizes the usual pattern.

| Company | Favorite problems | Focus |
|---|---|---|
| Google | Distributed systems, search, Bigtable-style storage | Scale, consistency |
| Amazon | E-commerce, inventory, microservices | Availability, partition tolerance |
| Meta | Social graph, news feed, real-time messaging | Fan-out, caching |
| Netflix | Streaming, recommendation, resilience | CDN, fault tolerance |
| Stripe | Payment processing, ledger, API design | Correctness, idempotency |
| Anthropic | ML inference serving, API rate limiting | GPU scheduling, latency |
| Uber | Real-time dispatch, geospatial indexing | Low latency, high throughput |

## Choosing a building block

Each building block solves one problem and introduces another. The following table pairs each with when to use it and what it costs.

| Building block | Use when | Tradeoff |
|---|---|---|
| Load balancer | Multiple instances of a service | Added latency, complexity |
| Cache (Redis) | Read-heavy, latency-sensitive | Stale data, cache invalidation |
| Message queue | Async processing, decoupling | Eventual consistency, ordering |
| CDN | Static content, global users | Cache invalidation, cost |
| SQL database | Transactions, complex queries | Scaling writes is hard |
| NoSQL database | Massive scale, simple access patterns | No joins, eventual consistency |
| Consistent hashing | Distributed data partitioning | Uneven load with few nodes |
| Consensus (Raft) | Distributed state machine | Latency, availability during partition |

## In this chapter

1. [Design review: the tracer and the serving platform](01-design-review-tracer-and-serving-platform.md): the eight-line rubric, the capacity and error-budget arithmetic a serving design needs, six worked problems, and a model design document with its review.
2. [Design review: the control plane](02-design-review-control-plane.md): what is persisted before each external effect, traced through a corpus build and an approved agent write, and a review that turns every missing answer into an owned note.
3. [Design review: the whole system](03-design-review-whole-system.md): a request and a model release traced through every layer, with failure modes backed by drills and the cross-layer findings no single review owned.

## Further reading

- Martin Kleppmann, *Designing Data-Intensive Applications*: the deepest treatment of the building blocks, worth reading cover to cover.
- *ByteByteGo System Design*, for worked interview-style problems.
- The [System Design Primer](https://github.com/donnemartin/system-design-primer) (free).
