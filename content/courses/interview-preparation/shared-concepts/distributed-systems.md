# Distributed Systems Fundamentals

Core distributed systems concepts that appear across all top-tier engineering interviews.

## CAP Theorem

You can only guarantee two of three:
- **Consistency**: Every read sees the most recent write
- **Availability**: Every request gets a response
- **Partition Tolerance**: System works despite network partitions

In practice, partitions happen, so you're choosing between **CP** (consistent but may reject requests) and **AP** (available but may return stale data).

| System | Type | Trade-off |
|--------|------|-----------|
| ZooKeeper | CP | Rejects writes during partition |
| Cassandra | AP | May return stale data, eventual consistency |
| Spanner | CP | Uses TrueTime for global consistency, may increase latency |
| DynamoDB | Tunable | Choose between strong and eventual consistency per read |

## Consistency Models

From strongest to weakest:

| Model | Guarantee | Performance |
|-------|-----------|-------------|
| Linearizability | Every read sees the latest write globally | Slowest |
| Sequential consistency | All processes see operations in same order | Slow |
| Causal consistency | Causally related operations are ordered | Medium |
| Eventual consistency | All replicas converge eventually | Fastest |

### When to Use What

- **Linearizability**: Bank account balance, leader election, distributed locks
- **Causal consistency**: Social media feeds, collaborative editing
- **Eventual consistency**: DNS, caches, analytics counters, recommendations

## Consensus Algorithms

### Raft (Most Interview-Relevant)

```
[Leader] ---AppendEntries--> [Follower 1]
    |                        [Follower 2]
    |                        [Follower 3]
    |                        [Follower 4]
    |
    +--- Committed when majority (3/5) acknowledges
```

**Leader election**:
1. Follower times out (no heartbeat)
2. Becomes candidate, votes for self, requests votes
3. Wins with majority, becomes leader
4. Sends heartbeats to maintain authority

**Log replication**:
1. Leader appends entry to log
2. Sends to all followers
3. Committed when majority acknowledges
4. Applied to state machine

**Key properties**:
- At most one leader per term
- Committed entries survive leader changes
- Simple enough to implement correctly

### Paxos (Know Conceptually)

- More general than Raft, harder to understand
- Proposer, Acceptor, Learner roles
- Two-phase: Prepare -> Accept
- Multi-Paxos for repeated consensus (similar to Raft)

## Replication

### Single-Leader Replication

```
[Client] --> [Leader] --> [Follower 1]
                     --> [Follower 2]
                     --> [Follower 3]
```

- All writes go through leader
- Followers replicate asynchronously (or synchronously for strong consistency)
- Failover: Promote a follower to leader if leader fails

### Multi-Leader Replication

```
[Client A] --> [Leader DC1] <--> [Leader DC2] <-- [Client B]
```

- Multiple leaders accept writes (e.g., one per data center)
- Conflict resolution needed: last-writer-wins, merge, custom resolution
- Use case: Multi-region deployment for low write latency

### Leaderless Replication (Dynamo-Style)

```
[Client] --> [Node 1] (W)
         --> [Node 2] (W)
         --> [Node 3] (W)
```

- Client sends writes to multiple nodes
- **Quorum**: W + R > N ensures read sees latest write
  - N=3, W=2, R=2: Strong consistency
  - N=3, W=1, R=1: Fastest, eventual consistency
- Anti-entropy: Background process syncs replicas

## Partitioning (Sharding)

### Strategies

| Strategy | Description | Pros | Cons |
|----------|-------------|------|------|
| Hash partitioning | hash(key) % N | Even distribution | Range queries span all partitions |
| Range partitioning | Key ranges per partition | Efficient range queries | Hot spots possible |
| Directory-based | Lookup table maps key to partition | Flexible | Lookup table is bottleneck |
| Consistent hashing | Hash ring with virtual nodes | Easy rebalancing | Less control over placement |

### Hot Partition Mitigation

- Add random suffix to hot keys (scatter reads/writes)
- Replicate hot partitions
- Application-level caching for hot data
- Dynamic partition splitting

## Distributed Transactions

### Two-Phase Commit (2PC)

```
Phase 1 (Prepare):
  Coordinator --> "Can you commit?" --> Participant A: "Yes"
                                    --> Participant B: "Yes"

Phase 2 (Commit):
  Coordinator --> "Commit" --> Participant A: Done
                           --> Participant B: Done
```

**Problem**: Blocking. If coordinator fails after sending "prepare" but before "commit", participants are stuck holding locks.

### Saga Pattern

For long-running distributed transactions:

```
Step 1: Create Order     | Compensate: Cancel Order
Step 2: Reserve Payment  | Compensate: Release Payment
Step 3: Reserve Inventory| Compensate: Release Inventory
Step 4: Ship             | Compensate: Return
```

If any step fails, execute compensating actions in reverse. Non-blocking, eventually consistent.

### Two Variants

- **Choreography**: Each service emits events, next service reacts
- **Orchestration**: Central coordinator drives the saga steps

## Load Balancing

### Algorithms

| Algorithm | Description | Use Case |
|-----------|-------------|----------|
| Round Robin | Distribute sequentially | Equal-capacity servers |
| Weighted Round Robin | Proportional to capacity | Mixed server sizes |
| Least Connections | Send to least-busy server | Variable request duration |
| Consistent Hashing | Hash-based routing | Stateful services, caching |
| Random | Random selection | Simple, surprisingly effective |

### Layer 4 vs. Layer 7

- **Layer 4 (TCP)**: Fast, low overhead, no content inspection
- **Layer 7 (HTTP)**: Can route based on URL, headers, cookies. More flexible.

## Caching

### Caching Strategies

| Strategy | Read Path | Write Path | Consistency |
|----------|-----------|------------|-------------|
| Cache-aside | App checks cache, misses hit DB | App updates DB, invalidates cache | Eventual |
| Read-through | Cache fetches from DB on miss | App writes to cache, cache writes to DB | Eventual |
| Write-through | Read from cache | Write to cache and DB simultaneously | Strong |
| Write-behind | Read from cache | Write to cache, async write to DB | Weak |

### Cache Invalidation

The hardest problem in computer science (after naming things):

- **TTL**: Simple, but data can be stale up to TTL
- **Event-based**: Invalidate on write events. Low staleness but complex.
- **Version-based**: Include version in cache key. Never stale but requires version tracking.

### Multi-Level Caching

```
[Client] --> [Browser Cache] --> [CDN] --> [API Gateway Cache]
                                               --> [Application Cache (Redis)]
                                                       --> [Database]
```

## Message Queues

### Patterns

| Pattern | Description | Delivery | Use Case |
|---------|-------------|----------|----------|
| Point-to-Point | One producer, one consumer | At-most-once or at-least-once | Task processing |
| Pub/Sub | One producer, many consumers | Fan-out | Event notification |
| Competing Consumers | Many consumers, each message processed once | At-least-once | Load distribution |
| Event Sourcing | Log of all state changes | Exactly-once (with dedup) | Audit trail, replay |

### Delivery Guarantees

- **At-most-once**: Send and forget. May lose messages. Fastest.
- **At-least-once**: Retry until acknowledged. May duplicate. Most common.
- **Exactly-once**: Deduplication + idempotent processing. Hardest, slowest.

In practice, design for **at-least-once delivery + idempotent consumers**.

## Rate Limiting

### Algorithms

| Algorithm | Description | Pros | Cons |
|-----------|-------------|------|------|
| Token Bucket | Tokens added at rate R, consumed per request | Allows bursts, smooth | State per client |
| Leaky Bucket | Requests drain at fixed rate | Smooth output | No burst tolerance |
| Fixed Window | Count requests per time window | Simple | Boundary burst (2x at window edge) |
| Sliding Window Log | Track timestamp of each request | Accurate | Memory-intensive |
| Sliding Window Counter | Weighted average of current and previous window | Low memory, accurate | Approximate |

### Distributed Rate Limiting

```
Option 1: Centralized (Redis)
  [Client] --> [API Server] --> [Redis: INCR + EXPIRE] --> Allow/Deny

Option 2: Local + Sync
  Each server has local counter, periodically syncs to central store
  Allows slightly over limit but no network hop per request
```

## Microservices Patterns

### Service Communication

| Pattern | Latency | Coupling | Reliability |
|---------|---------|----------|-------------|
| Synchronous REST | Low | High | Request fails if service down |
| Synchronous gRPC | Lower | High | Same, but with protobuf efficiency |
| Async messaging | Higher | Low | Queue buffers if service down |
| Event-driven | Variable | Lowest | Most resilient |

### Resilience Patterns

- **Circuit Breaker**: Stop calling failing services, fail fast
- **Retry with Backoff**: Exponential backoff + jitter
- **Bulkhead**: Isolate resources per dependency (separate thread pools)
- **Timeout**: Always set timeouts. Never wait indefinitely.
- **Fallback**: Degrade gracefully (cached response, default value)

### Service Mesh

```
[Service A] --> [Sidecar Proxy A] --> [Sidecar Proxy B] --> [Service B]
                     |                       |
              [Control Plane (Istio/Linkerd)]
```

Handles: Load balancing, mTLS, retries, circuit breaking, observability -- without application code changes.

## Observability

### The Three Pillars

| Pillar | What It Captures | Tools |
|--------|-----------------|-------|
| **Logs** | Discrete events | ELK, Loki, CloudWatch |
| **Metrics** | Aggregated measurements | Prometheus, Datadog, CloudWatch |
| **Traces** | Request flow across services | Jaeger, Zipkin, X-Ray |

### Key Metrics (RED Method)

- **Rate**: Requests per second
- **Errors**: Error rate (percentage of failed requests)
- **Duration**: Latency distribution (p50, p95, p99)

### SLOs and Error Budgets

```
SLI (Indicator): p99 latency < 200ms
SLO (Objective): 99.9% of requests meet the SLI over 30 days
Error Budget: 0.1% of requests can violate (43.2 minutes/month)
```

When error budget is exhausted: Freeze deployments, focus on reliability.

## Back-of-Envelope Estimation

### Quick Reference Numbers

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

### Quick Capacity Math

```
1 day = 86,400 seconds ≈ 100K seconds
1 million requests/day ≈ 12 requests/second
1 billion requests/day ≈ 12,000 requests/second

1 KB * 1 billion = 1 TB
1 MB * 1 million = 1 TB
```
