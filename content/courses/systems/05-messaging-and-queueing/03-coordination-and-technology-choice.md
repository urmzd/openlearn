# Coordination and technology choice

This part of [Messaging and Distributed Queueing](README.md) develops the mechanisms below. Work the numerical examples or trace the code before checking the answer.

> **Learning objectives**
> - Separate metadata coordination from data delivery and compare mechanisms.

## Coordination and metadata: ZooKeeper, KRaft, and Redpanda

Everything so far assumed the broker cluster agrees on its own shape. It has to: a cluster of brokers needs a single, agreed answer to "who is the controller, which broker leads partition $p$, and what is the current ISR (in-sync replica set)?" That metadata must be **linearizable**, meaning everyone agrees on it, and it must survive node failure. This is a consensus problem, and the industry has answered it three ways.

### ZooKeeper and ZAB

Classic Kafka stored cluster metadata in an *external* ZooKeeper ensemble. ZooKeeper runs **ZAB (ZooKeeper Atomic Broadcast)**, a primary-backup protocol in which one leader sequences all writes and atomically broadcasts them to followers, and a quorum must acknowledge before a write commits. Kafka's controller watched ZooKeeper znodes and propagated changes to the brokers. It works, but it has costs: a second distributed system to operate, metadata changes bottlenecked through watches, and partition counts limited by how fast metadata could propagate (tens of thousands of partitions before it became painful).

### Kafka KRaft

[KIP-500](https://cwiki.apache.org/confluence/display/KAFKA/KIP-500%3A+Replace+ZooKeeper+with+a+Self-Managed+Metadata+Quorum) replaced ZooKeeper with **KRaft**: a dedicated quorum of controller nodes that run **Raft** and store all cluster metadata in an internal `__cluster_metadata` log. The Raft leader is the active controller, and metadata becomes just another replicated log that brokers *tail* (pull, again) to learn the current state.

Controller election is Raft leader election. Candidates bump a **term**, request votes, and a **majority quorum** elects exactly one leader per term; [`code/raft_election.go`](code/raft_election.go) lets you watch this happen. Metadata propagation becomes log replication: brokers replay the metadata log instead of reacting to ZooKeeper watches, so failover is faster and recovery is simply "catch up the log."

Removing ZooKeeper matters for four reasons. You run one system instead of two. Metadata as a log scales to *millions* of partitions. Controller failover is faster. And operations are simpler, with a smaller failure surface. ZooKeeper removal was completed in Apache Kafka 4.0, so KRaft is now the only mode.

### Redpanda

Redpanda is a clean-room, Kafka-API-compatible rewrite that takes the consensus idea further. It has no JVM, no ZooKeeper, and no separate coordinator. It is written in C++ on the [Seastar](https://seastar.io/) framework with a **thread-per-core (shard-per-core)** model: each CPU core owns a slice of partitions and its own memory, which avoids lock contention and garbage-collection pauses. And instead of one coordinator, it runs **Raft per partition**: every partition is its own Raft group managing its own replication and leader, so consensus is embedded everywhere rather than centralized. The result is lower tail latency and a single binary, the strongest contrast to the JVM-plus-ZooKeeper lineage.

### The frontier and the counterexample

Three further developments are worth knowing. **KIP-932 share groups**, "queues for Kafka," add queue-like competing-consumer semantics with per-message acknowledgement *on top of* the log, decoupling consumer count from partition count. That directly attacks the partition-equals-parallelism ceiling from the worker side. **KIP-405 tiered storage** offloads cold log segments to object storage such as S3 while hot data stays on local disk, which makes effectively infinite retention cheap and decouples storage from broker disk. And **Apache Pulsar** is the contrasting architecture. It separates *serving* (stateless brokers) from *storage* (the [Apache BookKeeper](https://bookkeeper.apache.org/) segment store). Brokers hold no data, so they scale and rebalance independently of storage, and segments stripe across bookies. Pulsar also offers native queue *and* stream semantics. It is the architectural opposite of Kafka's broker-owns-its-partitions model, and worth knowing as the counterexample.

## Choosing a technique

The chapter's ideas reduce to a set of choices, each with a situation where it is the right one. The table collects them for quick reference.

| Technique | When to apply |
|-----------|---------------|
| Log over queue (Kafka/Redpanda) | Replay, multiple independent readers, retention as source of truth |
| Classic queue (SQS/RabbitMQ) | Simple task distribution, per-message ack, no replay needed |
| Pull / long-poll consumption | When you want backpressure for free and lag-based scaling |
| Push + prefetch limits | Low-latency task delivery where you can bound in-flight work |
| Competing consumers (one group) | Scaling work across a pool |
| Publish-subscribe (multiple groups) | Independent systems reacting to the same events |
| Key by entity | When per-entity ordering matters (per-order, per-user) |
| At-least-once + idempotency | Default for anything touching external systems |
| KRaft / Raft quorum | Self-managed metadata, scale to many partitions, no ZooKeeper |
| Redpanda (thread-per-core, Raft/partition) | No-JVM, low-tail-latency, single-binary Kafka API |
| Tiered storage (KIP-405) | Cheap effectively-infinite retention |
| Share groups (KIP-932) | Queue semantics + parallelism beyond partition count |
| Pulsar (broker/storage split) | Independent scaling of serving vs storage |

## Where this leads

The consuming side of these dynamics (rebalancing, dead-letter queues, sizing the pool) continues in [Durable Orchestration and Workers](../07-durable-orchestration/README.md) in *Systems Engineering*. The data-engineering view of the same Kafka, with watermarks, compaction, and Kappa, is in [Batch and Streaming Processing](../../data-engineering/03-batch-streaming/README.md). Lag-based autoscaling with KEDA lives in [Containers, Kubernetes, and Workloads](../10-containers-kubernetes/README.md). The consensus and replicated-state-machine fundamentals behind Raft, ZAB, quorums, and linearizability are in [System Design](../01-system-design/README.md). And "exactly-once is idempotency" is a property you can test, which [The Testing Mentality](../../software-engineering/03-testing/README.md) in *Software Engineering* teaches you to do. The [Apache stack](../14-apache-stack/README.md) chapter later in this book places Kafka and Pulsar on the wider map.

These ideas show up across the industry. LinkedIn created Kafka and runs it at trillions of messages per day, which is where the log as a unifying abstraction came from. Confluent, the company built around Kafka, drove KRaft and tiered storage. Redpanda Data builds the Kafka-API-compatible C++ engine with thread-per-core and Raft per partition. Amazon offers SQS and SNS (queue plus pub-sub) and MSK (managed Kafka). StreamNative and Yahoo are behind Apache Pulsar and its BookKeeper-backed broker/storage split. For almost any backend or data role, async messaging behind a log is the default architecture, so queue versus log, push versus pull, and delivery semantics are core vocabulary.

### Further reading

- [Apache Kafka documentation](https://kafka.apache.org/documentation/): the Design, Implementation, and Replication sections.
- [The Log: What every software engineer should know about real-time data's unifying abstraction](https://engineering.linkedin.com/distributed-systems/log-what-every-software-engineer-should-know-about-real-time-datas-unifying-abstraction) by Jay Kreps: why a log subsumes queues and pub-sub.
- *Designing Data-Intensive Applications* (Kleppmann), Chapter 11, "Stream Processing."
- [Enterprise Integration Patterns](https://www.enterpriseintegrationpatterns.com/): Competing Consumers, Publish-Subscribe, Dead Letter Channel.
- [*In Search of an Understandable Consensus Algorithm*](https://raft.github.io/raft.pdf): the Raft paper, the consensus behind KRaft and Redpanda.
- [ZooKeeper: Wait-free coordination for Internet-scale systems](https://www.usenix.org/legacy/event/atc10/tech/full_papers/Hunt.pdf): ZAB and the coordinator Kafka shed.
- [KIP-500](https://cwiki.apache.org/confluence/display/KAFKA/KIP-500%3A+Replace+ZooKeeper+with+a+Self-Managed+Metadata+Quorum) (KRaft), [KIP-405](https://cwiki.apache.org/confluence/display/KAFKA/KIP-405%3A+Kafka+Tiered+Storage) (tiered storage), and [KIP-932](https://cwiki.apache.org/confluence/display/KAFKA/KIP-932%3A+Queues+for+Kafka) (share groups).

## Key terms

- **Log**: an append-only, ordered sequence of messages retained for a window regardless of who has read it.
- **Offset**: the monotonic position of a message within a partition; consumers record the offset they have processed up to.
- **Consumer group**: a set of consumers that divide a topic's partitions among themselves; different groups each see every message.
- **Consumer lag**: log-end offset minus committed offset; the backlog a group still has to process.
- **Backpressure**: the signal that tells a fast producer the consumers are behind; free under pull, built from prefetch limits under push.
- **At-least-once delivery**: committing after the work, so a crash between work and commit causes a redelivered duplicate but never a loss.
- **Idempotency**: the property that applying the same message twice has the same effect as applying it once.
- **KRaft**: Kafka's Raft-based controller quorum that stores cluster metadata in an internal log and replaced ZooKeeper.

## Summary

- A queue deletes on acknowledgement; a log retains and replays, and its consumers own their offsets. That is the deepest dividing line in messaging.
- Pull consumption gives backpressure for free: surplus stays in the log as lag, and lag is the signal autoscalers watch.
- Consumer groups give competing consumers within a group and publish-subscribe across groups, with partition count as the parallelism ceiling.
- "Exactly-once" is mostly idempotency; the crash window between work and commit makes at-least-once the honest default.
- Little's Law ($L = \lambda W$) and the lag integral turn queueing intuition into numbers.
- Someone must own the metadata: ZooKeeper did it externally with ZAB, KRaft folds it into Kafka with Raft, and Redpanda runs Raft per partition with no coordinator at all.

## Review questions

1. Why can a Kafka topic serve both a worker pool and an independent analytics job reading the same messages, when a classic SQS queue cannot do this without copies?
   <details><summary>Answer</summary>The log retains messages for a retention window regardless of who has read them, and each consumer group tracks its own committed offset. Two groups simply read the same partitions at different positions. A classic queue deletes a message once it is acknowledged, so a second reader would need its own fan-out copy.</details>
2. A slow pull consumer and a slow push consumer each receive a burst three times larger than they can process. What happens to each?
   <details><summary>Answer</summary>The pull consumer calls <code>poll()</code> less often, so the surplus stays in the log and shows up as growing consumer lag; nothing is dropped. The push consumer is sent messages faster than it can handle them; without prefetch or credit limits it is overwhelmed and messages can be dropped or pile up in its memory.</details>
3. A worker polls a message, charges a card, and then commits the offset. Where is the crash window and what is the consequence?
   <details><summary>Answer</summary>Between the charge completing and the commit landing. A crash there leaves the offset uncommitted, so the message is redelivered and the card can be charged twice. Kafka transactions do not help because the payment gateway is outside the transaction; the fix is an idempotency key the charge path deduplicates on.</details>
4. A pool of 8 consumers each holds at most 50 in-flight messages, and average processing latency is 0.2 seconds. What is the maximum sustainable arrival rate?
   <details><summary>Answer</summary>By Little's Law, $L \le c \cdot p = 8 \cdot 50 = 400$, so $\lambda \le c \cdot p / W = 400 / 0.2 = 2000$ messages per second.</details>
5. Why does a sixth consumer in a group reading a five-partition topic do nothing?
   <details><summary>Answer</summary>Within a group each partition is owned by exactly one consumer. With five partitions there are only five owners; the sixth consumer has no partition to own and sits idle. Partition count is the parallelism ceiling (KIP-932 share groups aim to lift it).</details>
6. What did Kafka gain by replacing ZooKeeper with KRaft, and how does Redpanda go further?
   <details><summary>Answer</summary>KRaft removes a second distributed system, stores metadata as a Raft-replicated log that brokers tail, scales to millions of partitions, and fails over faster. Redpanda drops the JVM and any central coordinator entirely, running a Raft group per partition on a thread-per-core C++ runtime for lower tail latency.</details>

[Chapter contents](README.md)
