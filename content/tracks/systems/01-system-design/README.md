# System Design

## Overview

- **Primary references**:
  - *ByteByteGo System Design* (recommended purchase)
  - *Designing Data-Intensive Applications* by Martin Kleppmann (recommended purchase -- the industry bible)
  - [System Design Primer](https://github.com/donnemartin/system-design-primer) (free)
- **Prerequisites**: [Algorithms](../../algorithms/)
- **Estimated time**: 4-5 weeks at 8-10 hrs/week

## Key Takeaways

- System design is about trade-offs, not optimal solutions -- every choice has costs
- Start every design with requirements (functional + non-functional) and back-of-envelope math
- The most important distributed systems concepts: CAP theorem, consistency models, and failure modes
- Staff+ interviews test your ability to navigate ambiguity and make justified decisions, not memorize architectures

## How to Study

- Study 2-3 classic design problems per week
- For each problem: estimate scale first, then design, then discuss trade-offs
- Practice explaining your design to someone in 15 minutes (interview format)
- Read DDIA cover-to-cover for deep understanding of the building blocks

---

# Concepts & Techniques

## Core Insight

Every system design problem reduces to: (1) what are the access patterns? (2) what are the scale requirements? (3) what consistency/availability trade-offs are acceptable? The answer to these three questions determines 90% of the architecture.

## 1. Scalability Fundamentals

**Key ideas**:
- **Vertical scaling**: bigger machine. Simple but has a ceiling. Good for databases.
- **Horizontal scaling**: more machines. Complex but unlimited. Good for stateless services.
- **Load balancing**: round-robin, least connections, consistent hashing. L4 (TCP) vs L7 (HTTP).
- **Caching**: browser → CDN → reverse proxy → application → database. Cache invalidation is the hard part.
- **CDN**: serve static content from edge locations. Pull vs push CDN. Cache-Control headers.
- **Database sharding**: horizontal partitioning. Range-based, hash-based, directory-based. Resharding is painful.

**Back-of-envelope**: know these numbers -- QPS for common services (100K-1M), storage per record (~1KB), monthly storage growth, read/write ratio.

## 2. Database Design

**Key ideas**:
- **SQL**: ACID transactions, joins, strong consistency. PostgreSQL is the default choice.
- **NoSQL**: BASE, eventual consistency, horizontal scaling. DynamoDB (K-V), MongoDB (document), Cassandra (wide-column), Neo4j (graph).
- **CAP theorem**: in a network partition, choose Consistency (CP: refuse stale reads) or Availability (AP: serve stale reads). You can't have both.
- **Replication**: single-leader (simple, one write path), multi-leader (multi-DC), leaderless (Dynamo-style quorum reads/writes).
- **Partitioning**: range (good for scans, hot spots risk), hash (even distribution, no range queries).
- **When to use what**: SQL for transactions and complex queries; NoSQL for massive scale, flexible schema, or specific access patterns.

## 3. Distributed Systems Primitives

**Key ideas**:
- **Consensus**: Raft (understandable) or Paxos (original). Guarantees agreement among nodes. Used by etcd, ZooKeeper.
- **Distributed transactions**: 2PC (blocking, coordinator is SPOF), Saga (compensating transactions, eventual consistency).
- **Vector clocks / Lamport timestamps**: establish causal ordering without synchronized clocks.
- **Consistent hashing**: distribute keys across nodes; adding/removing a node only remaps K/N keys. Used by DynamoDB, Cassandra.
- **Gossip protocol**: eventually propagate state across all nodes; epidemic-style. Used by Cassandra, SWIM.

## 4. Messaging & Event Systems

**Key ideas**:
- **Message queues**: decouple producers and consumers. Kafka (log-based, ordered), RabbitMQ (traditional, routing), SQS (managed).
- **Event sourcing**: store events, not state. Rebuild state by replaying events. Enables audit log and time travel.
- **CQRS**: separate read and write models. Write to event store, project to read-optimized views.
- **Exactly-once semantics**: impossible in general; achieve effectively-once via idempotent consumers + deduplication.

## 5. API Design

**Key ideas**:
- **REST**: resource-oriented, HTTP verbs, stateless. Good for CRUD. Over-fetching/under-fetching problems.
- **gRPC**: binary protocol (protobuf), streaming, strongly typed. Good for internal service-to-service.
- **GraphQL**: client specifies exactly what data it needs. Good for complex, nested data. Complexity in server-side resolvers.
- **Rate limiting**: token bucket, sliding window. Per-user, per-IP, per-API-key. Return 429 with Retry-After header.
- **Idempotency**: POST with idempotency key. Critical for payment systems. Stripe's approach: client generates key, server deduplicates.
- **Pagination**: cursor-based (consistent with concurrent writes) > offset-based (skips under mutation). Keyset pagination for databases.

## 6. Classic Design Problems

**URL shortener**: hash-based vs counter-based ID generation, base62 encoding, read-heavy caching, analytics pipeline.

**Chat system (WhatsApp)**: WebSocket connections, message queue per user, delivery receipts, presence service, end-to-end encryption.

**News feed (Twitter)**: fan-out on write (push) for users with few followers, fan-out on read (pull) for celebrities. Hybrid approach.

**Video streaming (YouTube)**: upload pipeline (transcode to multiple resolutions), CDN for delivery, adaptive bitrate streaming (HLS/DASH).

**Payment system (Stripe)**: idempotency, exactly-once processing, ledger with double-entry bookkeeping, reconciliation, PCI compliance.

**Key-value store**: consistent hashing, replication, quorum reads/writes, conflict resolution (LWW or vector clocks), compaction.

**Search engine**: web crawler, inverted index, ranking (TF-IDF → PageRank → ML), query processing, caching.

---

## Technique Catalog

| Building Block | Use When | Trade-off |
|---------------|----------|-----------|
| Load balancer | Multiple instances of a service | Added latency, complexity |
| Cache (Redis) | Read-heavy, latency-sensitive | Stale data, cache invalidation |
| Message queue | Async processing, decoupling | Eventual consistency, ordering |
| CDN | Static content, global users | Cache invalidation, cost |
| SQL database | Transactions, complex queries | Scaling writes is hard |
| NoSQL database | Massive scale, simple access patterns | No joins, eventual consistency |
| Consistent hashing | Distributed data partitioning | Uneven load with few nodes |
| Consensus (Raft) | Distributed state machine | Latency, availability during partition |

## Company Relevance

| Company | Favorite Problems | Focus |
|---------|------------------|-------|
| Google | Distributed systems, search, bigtable-style storage | Scale, consistency |
| Amazon | E-commerce, inventory, microservices | Availability, partition tolerance |
| Meta | Social graph, news feed, real-time messaging | Fan-out, caching |
| Netflix | Streaming, recommendation, resilience | CDN, fault tolerance |
| Stripe | Payment processing, ledger, API design | Correctness, idempotency |
| Anthropic | ML inference serving, API rate limiting | GPU scheduling, latency |
| Uber | Real-time dispatch, geospatial indexing | Low latency, high throughput |

## Chapters

| # | Module | Chapter | Kind | Pass |
|---|---|---|---|---|
| 1 | `review.01` | [Design review: tracer and serving platform](01-design-review-tracer-and-serving-platform.md) | proof | 7 |
| 2 | `review.02` | [Design review: control plane](02-design-review-control-plane.md) | proof | 11 |
| 3 | `review.03` | [Design review: the whole system](03-design-review-whole-system.md) | proof | 11 |
