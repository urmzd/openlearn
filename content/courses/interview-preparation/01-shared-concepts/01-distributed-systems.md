# Distributed systems fundamentals

Almost every system design interview, whatever the company, eventually becomes a conversation about distributed systems. Why does this read return stale data? What happens when the leader dies? How do you shard this table, and what breaks when one shard gets hot? The interviewer is not looking for a recitation of definitions but for the judgment to pick the right trade-off and name what it costs.

This section collects the core concepts that recur across top-tier engineering interviews, in roughly the order a design conversation reaches them. It starts with the fundamental trade-off between consistency and availability, moves through consistency models, consensus, replication, and partitioning, then covers distributed transactions, load balancing, caching, messaging, and rate limiting, and finishes with the operational concerns of microservices, observability, and back-of-envelope estimation. The company chapters, such as [Google](../02-big-tech-and-ai-labs/02-google/README.md), [Meta](../02-big-tech-and-ai-labs/05-meta/README.md), and [Netflix](../03-infrastructure-and-data-platforms/01-netflix/README.md), apply this material to their own design prompts.

> **Learning objectives**
> - State the CAP theorem precisely and classify real systems as CP, AP, or tunable.
> - Choose a consistency model, replication scheme, and partitioning strategy for a given workload and justify it.
> - Explain Raft's leader election and log replication, and contrast two-phase commit with sagas.
> - Pick caching, messaging, and rate limiting strategies and describe their failure modes.
> - Do back-of-envelope latency and capacity estimates from memorised reference numbers.

## Consistency, availability, and partitions

The CAP theorem says that a distributed data store can guarantee at most two of three properties. Consistency means every read sees the most recent write. Availability means every request gets a response. Partition tolerance means the system keeps working despite network partitions between its nodes. In practice partitions happen whether you like it or not, so the real choice is what to do during one: a CP system stays consistent but may reject requests, and an AP system stays available but may return stale data.

Real systems sit at different points on this spectrum:

| System | Type | Trade-off |
|--------|------|-----------|
| ZooKeeper | CP | Rejects writes during partition |
| Cassandra | AP | May return stale data, eventual consistency |
| Spanner | CP | Uses TrueTime for global consistency, may increase latency |
| DynamoDB | Tunable | Choose between strong and eventual consistency per read |

## Consistency models

"Consistent" is not one thing. Consistency models range from strongest to weakest, and each step down buys performance:

| Model | Guarantee | Performance |
|-------|-----------|-------------|
| Linearizability | Every read sees the latest write globally | Slowest |
| Sequential consistency | All processes see operations in same order | Slow |
| Causal consistency | Causally related operations are ordered | Medium |
| Eventual consistency | All replicas converge eventually | Fastest |

The skill is matching the model to the data. Linearizability is worth its cost for a bank account balance, leader election, and distributed locks, where acting on a stale value is a correctness bug. Causal consistency suits social media feeds and collaborative editing, where a reply must never appear before the post it answers but unrelated posts can arrive in any order. Eventual consistency is enough for DNS, caches, analytics counters, and recommendations, where brief staleness is harmless.

## Consensus

Linearizable operations, leader election, and locks all rest on consensus: getting a group of nodes to agree on a value despite failures.

### Raft

Raft is the most interview-relevant consensus algorithm, because it was designed to be understandable. A single leader replicates a log to followers, and an entry is committed once a majority has acknowledged it, three of five in this picture:

```
[Leader] ---AppendEntries--> [Follower 1]
    |                        [Follower 2]
    |                        [Follower 3]
    |                        [Follower 4]
    |
    +--- Committed when majority (3/5) acknowledges
```

Leader election works like this. A follower that hears no heartbeat within its timeout becomes a candidate, votes for itself, and requests votes from the others. If it wins a majority it becomes leader, and it sends heartbeats to maintain its authority. Log replication then proceeds in four steps: the leader appends an entry to its log, sends it to all followers, marks it committed when a majority acknowledges, and applies it to the state machine.

Three properties make Raft safe and practical. There is at most one leader per term, because a candidate needs a majority and two majorities always overlap. Committed entries survive leader changes, because a new leader must have every committed entry to win an election. And the algorithm is simple enough to implement correctly.

### Paxos

Know Paxos conceptually. It is more general than Raft and harder to understand. It has three roles, proposer, acceptor, and learner, and runs in two phases, prepare and then accept. Multi-Paxos runs repeated rounds of consensus to agree on a sequence of values, which makes it similar in effect to Raft.

## Replication

Consensus is one way to keep copies in agreement. Replication more broadly comes in three shapes, distinguished by who may accept writes.

In single-leader replication, all writes go through one leader, which forwards them to followers:

```
[Client] --> [Leader] --> [Follower 1]
                     --> [Follower 2]
                     --> [Follower 3]
```

Followers replicate asynchronously, or synchronously when you need strong consistency. If the leader fails, a follower is promoted in its place.

In multi-leader replication, several leaders accept writes, typically one per data centre:

```
[Client A] --> [Leader DC1] <--> [Leader DC2] <-- [Client B]
```

This gives low write latency in a multi-region deployment, at the price of conflicts: two leaders can accept conflicting writes to the same record, so you need a resolution strategy such as last-writer-wins, merging, or custom resolution logic.

In leaderless, Dynamo-style replication, the client sends each write to several nodes directly:

```
[Client] --> [Node 1] (W)
         --> [Node 2] (W)
         --> [Node 3] (W)
```

Consistency then comes from quorums. With $N$ replicas, a write waits for $W$ acknowledgements and a read queries $R$ nodes. If $W + R > N$, every read set overlaps every write set, so a read sees the latest write. With $N = 3$, choosing $W = 2, R = 2$ gives strong consistency; choosing $W = 1, R = 1$ is fastest but only eventually consistent. A background anti-entropy process syncs replicas that have drifted.

## Partitioning

Replication copies data; partitioning, or sharding, splits it, so that no single node has to hold or serve everything. Four strategies are common:

| Strategy | Description | Pros | Cons |
|----------|-------------|------|------|
| Hash partitioning | hash(key) % N | Even distribution | Range queries span all partitions |
| Range partitioning | Key ranges per partition | Efficient range queries | Hot spots possible |
| Directory-based | Lookup table maps key to partition | Flexible | Lookup table is bottleneck |
| Consistent hashing | Hash ring with virtual nodes | Easy rebalancing | Less control over placement |

Whatever the strategy, some keys will be hotter than others: a celebrity's profile, a viral post. You can mitigate a hot partition by adding a random suffix to hot keys to scatter their reads and writes across partitions, by replicating the hot partition, by caching hot data at the application level, or by splitting partitions dynamically as load grows.

## Distributed transactions

Once data is partitioned, an operation that touches several partitions or services needs some way to commit atomically, or at least to fail cleanly.

### Two-phase commit

Two-phase commit (2PC) uses a coordinator. In the prepare phase it asks every participant whether it can commit, and each participant that says yes locks its resources and promises to commit if told to. In the commit phase, if everyone said yes, the coordinator tells them all to commit:

```
Phase 1 (Prepare):
  Coordinator --> "Can you commit?" --> Participant A: "Yes"
                                    --> Participant B: "Yes"

Phase 2 (Commit):
  Coordinator --> "Commit" --> Participant A: Done
                           --> Participant B: Done
```

The problem is blocking. If the coordinator fails after sending prepare but before sending commit, the participants are stuck holding locks, unable to commit or abort on their own.

### Sagas

For long-running distributed transactions, the saga pattern avoids locks entirely. A saga is a sequence of local steps, each paired with a compensating action that undoes it:

```
Step 1: Create Order     | Compensate: Cancel Order
Step 2: Reserve Payment  | Compensate: Release Payment
Step 3: Reserve Inventory| Compensate: Release Inventory
Step 4: Ship             | Compensate: Return
```

If any step fails, the compensating actions for the completed steps run in reverse order. The result is non-blocking and eventually consistent. Sagas come in two variants: in choreography, each service emits events and the next service reacts to them; in orchestration, a central coordinator drives the steps.

## Load balancing

With data partitioned and replicated, requests must be spread across servers. The common algorithms suit different workloads:

| Algorithm | Description | Use Case |
|-----------|-------------|----------|
| Round Robin | Distribute sequentially | Equal-capacity servers |
| Weighted Round Robin | Proportional to capacity | Mixed server sizes |
| Least Connections | Send to least-busy server | Variable request duration |
| Consistent Hashing | Hash-based routing | Stateful services, caching |
| Random | Random selection | Simple, surprisingly effective |

Load balancers also differ in which layer they work at. A layer 4 (TCP) balancer is fast and low-overhead but cannot inspect content. A layer 7 (HTTP) balancer can route on URL, headers, and cookies, which makes it more flexible at some cost.

## Caching

Caching is the most common answer to a latency or load problem, and also a common source of bugs. The strategies differ in who reads and writes the backing store, and so in the consistency they offer:

| Strategy | Read Path | Write Path | Consistency |
|----------|-----------|------------|-------------|
| Cache-aside | App checks cache, misses hit DB | App updates DB, invalidates cache | Eventual |
| Read-through | Cache fetches from DB on miss | App writes to cache, cache writes to DB | Eventual |
| Write-through | Read from cache | Write to cache and DB simultaneously | Strong |
| Write-behind | Read from cache | Write to cache, async write to DB | Weak |

Cache invalidation is famously the hardest problem in computer science, after naming things. Three approaches cover most designs. A TTL is simple, but data can be stale for up to the TTL. Event-based invalidation removes entries on write events, giving low staleness at the cost of complexity. Version-based keys include a version in the cache key, so a cached value is never stale, but you must track versions.

Large systems cache at several levels, each catching what the one before it missed:

```
[Client] --> [Browser Cache] --> [CDN] --> [API Gateway Cache]
                                               --> [Application Cache (Redis)]
                                                       --> [Database]
```

## Message queues

Queues decouple producers from consumers in time, which is how systems absorb bursts and survive a consumer being down. The patterns differ in how many consumers see each message:

| Pattern | Description | Delivery | Use Case |
|---------|-------------|----------|----------|
| Point-to-Point | One producer, one consumer | At-most-once or at-least-once | Task processing |
| Pub/Sub | One producer, many consumers | Fan-out | Event notification |
| Competing Consumers | Many consumers, each message processed once | At-least-once | Load distribution |
| Event Sourcing | Log of all state changes | Exactly-once (with dedup) | Audit trail, replay |

Delivery guarantees are the other axis. At-most-once delivery sends and forgets; it is fastest but may lose messages. At-least-once delivery retries until acknowledged; it is the most common, but may deliver duplicates. Exactly-once requires deduplication plus idempotent processing, and is the hardest and slowest. In practice, design for at-least-once delivery with idempotent consumers, which gives exactly-once effects without exactly-once machinery.

## Rate limiting

Rate limiting protects a service from overload and enforces fair use. Five algorithms are standard:

| Algorithm | Description | Pros | Cons |
|-----------|-------------|------|------|
| Token Bucket | Tokens added at rate R, consumed per request | Allows bursts, smooth | State per client |
| Leaky Bucket | Requests drain at fixed rate | Smooth output | No burst tolerance |
| Fixed Window | Count requests per time window | Simple | Boundary burst (2x at window edge) |
| Sliding Window Log | Track timestamp of each request | Accurate | Memory-intensive |
| Sliding Window Counter | Weighted average of current and previous window | Low memory, accurate | Approximate |

The fixed window's weakness is worth being able to explain: a client can send a full window's quota at the end of one window and another full quota at the start of the next, so twice the limit passes in a short span. The [OpenAI](../02-big-tech-and-ai-labs/04-openai/README.md) chapter implements the token bucket and sliding window log.

When there are many API servers, the limit must be shared. There are two options:

```
Option 1: Centralized (Redis)
  [Client] --> [API Server] --> [Redis: INCR + EXPIRE] --> Allow/Deny

Option 2: Local + Sync
  Each server has local counter, periodically syncs to central store
  Allows slightly over limit but no network hop per request
```

## Microservices patterns

Systems built from many services need to decide how those services talk and how they survive each other's failures. The communication styles trade latency against coupling:

| Pattern | Latency | Coupling | Reliability |
|---------|---------|----------|-------------|
| Synchronous REST | Low | High | Request fails if service down |
| Synchronous gRPC | Lower | High | Same, but with protobuf efficiency |
| Async messaging | Higher | Low | Queue buffers if service down |
| Event-driven | Variable | Lowest | Most resilient |

For resilience, five patterns recur. A circuit breaker stops calling a failing service and fails fast instead (the [Netflix](../03-infrastructure-and-data-platforms/01-netflix/README.md) chapter implements one). Retries use exponential backoff with jitter. Bulkheads isolate resources per dependency, for example with separate thread pools, so that one slow dependency cannot exhaust them all. Timeouts are always set, because a call that waits indefinitely is a resource leak. And fallbacks degrade gracefully, to a cached response or a default value.

A service mesh moves these concerns out of application code into sidecar proxies managed by a control plane:

```
[Service A] --> [Sidecar Proxy A] --> [Sidecar Proxy B] --> [Service B]
                     |                       |
              [Control Plane (Istio/Linkerd)]
```

The mesh handles load balancing, mTLS, retries, circuit breaking, and observability without application code changes.

## Observability

You cannot operate what you cannot see, and interviewers increasingly ask how you would monitor a design. Observability rests on three pillars:

| Pillar | What It Captures | Tools |
|--------|-----------------|-------|
| **Logs** | Discrete events | ELK, Loki, CloudWatch |
| **Metrics** | Aggregated measurements | Prometheus, Datadog, CloudWatch |
| **Traces** | Request flow across services | Jaeger, Zipkin, X-Ray |

For request-serving systems, the RED method names the metrics that matter: rate (requests per second), errors (the percentage of requests that fail), and duration (the latency distribution at p50, p95, and p99).

Those metrics feed service level objectives. A service level indicator (SLI) is the measurement, an objective (SLO) is the target, and the error budget is what the target leaves over:

```
SLI (Indicator): p99 latency < 200ms
SLO (Objective): 99.9% of requests meet the SLI over 30 days
Error Budget: 0.1% of requests can violate (43.2 minutes/month)
```

The 43.2 minutes is $0.1\%$ of a 30-day month ($0.001 \times 30 \times 24 \times 60$), the equivalent in full downtime. When the error budget is exhausted, the team freezes deployments and focuses on reliability.

## Back-of-envelope estimation

Every design round asks you to estimate, and good estimates rest on a few memorised numbers. These are the standard latency references:

| Operation | Time |
|-----------|------|
| L1 cache reference | 1 ns |
| L2 cache reference | 4 ns |
| Main memory reference | 100 ns |
| SSD random read | 16 μs |
| HDD seek | 4 ms |
| Send 1 KB over 1 Gbps network | 10 μs |
| Read 1 MB from memory | 250 μs |
| Read 1 MB from SSD | 1 ms |
| Read 1 MB from HDD | 20 ms |
| Roundtrip same datacenter | 0.5 ms |
| Roundtrip CA -> Netherlands | 150 ms |

Capacity math leans on one approximation, that a day is about $10^5$ seconds:

```
1 day = 86,400 seconds ≈ 100K seconds
1 million requests/day ≈ 12 requests/second
1 billion requests/day ≈ 12,000 requests/second

1 KB * 1 billion = 1 TB
1 MB * 1 million = 1 TB
```

## Where this leads

The remaining sections of this chapter build on these foundations for ML systems: [Full-stack AI engineering](02-full-stack-ai.md) applies caching, queues, and streaming to model serving, and [ML infrastructure patterns](03-ml-infrastructure.md) applies replication, partitioning, and fault tolerance to training. The company chapters show the same concepts under each interviewer's preferred prompts.

### Further reading

- Martin Kleppmann, *Designing Data-Intensive Applications*, the standard reference for nearly every topic in this section.
- Diego Ongaro and John Ousterhout, "In Search of an Understandable Consensus Algorithm", the Raft paper.

## Key terms

- **CAP theorem**: during a network partition, a distributed store must choose between consistency and availability.
- **Linearizability**: the strongest consistency model, in which every read sees the latest write globally.
- **Quorum**: a write acknowledged by $W$ nodes and a read from $R$ nodes, with $W + R > N$ guaranteeing overlap.
- **Two-phase commit**: an atomic commit protocol with prepare and commit phases that blocks if the coordinator fails.
- **Saga**: a sequence of local transactions with compensating actions, run in reverse on failure.
- **Idempotent consumer**: a consumer for which processing the same message twice has the same effect as once.
- **Error budget**: the fraction of requests (or time) an SLO allows to fail.

## Summary

- Partitions are unavoidable, so the CAP choice is between rejecting requests (CP) and serving stale data (AP).
- Match the consistency model to the data: linearizable for money and locks, causal for feeds, eventual for caches and counters.
- Raft gives one leader per term and durable committed entries through overlapping majorities; quorums do the same for leaderless stores.
- Prefer sagas over 2PC for long-running cross-service work, and at-least-once delivery with idempotent consumers over exactly-once.
- Back every design with estimates from the reference latency numbers and the "one day is about $10^5$ seconds" rule.

## Review questions

1. Why is "CA" not a realistic choice for a distributed database?

   <details><summary>Answer</summary>

   Network partitions happen in any real distributed system, so partition tolerance is not optional. The only real choice is what to give up while a partition lasts: consistency (serve possibly stale data) or availability (reject some requests).

   </details>

2. With $N = 5$ replicas, which values of $W$ and $R$ guarantee that a read sees the latest write, and what does each choice favour?

   <details><summary>Answer</summary>

   Any pair with $W + R > 5$. For example $W = 3, R = 3$ balances reads and writes; $W = 5, R = 1$ makes reads fast at the cost of slow, fragile writes; $W = 1, R = 5$ does the opposite.

   </details>

3. What exactly goes wrong in two-phase commit if the coordinator crashes after the prepare phase?

   <details><summary>Answer</summary>

   Participants that voted yes have promised to commit if told to and are holding locks. Without the coordinator's decision they can neither commit (others may have voted no) nor abort (the coordinator may have decided commit), so they block until it recovers.

   </details>

4. Why does a fixed-window rate limiter allow up to twice its limit in a short period?

   <details><summary>Answer</summary>

   A client can send the full quota at the very end of one window and the full quota again at the very start of the next. Both windows are within their limits, but twice the limit has passed in a span much shorter than one window.

   </details>

5. Roughly how many requests per second is 500 million requests per day?

   <details><summary>Answer</summary>

   Using one day as about $10^5$ seconds, $5 \times 10^8 / 10^5 = 5000$ requests per second on average (about 5800 with 86,400 seconds). Peak traffic is typically a small multiple of the average.

   </details>
