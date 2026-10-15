# Side quest: Kafka consumer groups

## Overview

- **Primary references**:
  - [Apache Kafka documentation](https://kafka.apache.org/documentation/) (free) -- especially the Consumer and Consumer Group Protocol sections
  - [Confluent: Consumer group protocol](https://developer.confluent.io/courses/architecture/consumer-group-protocol/) (free) -- rebalancing in depth
  - *Designing Data-Intensive Applications* (Kleppmann) Ch 11 "Stream Processing" -- recommended
- **Supplementary**: [Enterprise Integration Patterns](https://www.enterpriseintegrationpatterns.com/) (Competing Consumers, Dead Letter Channel, Idempotent Receiver), [KIP-429 Incremental Cooperative Rebalancing](https://cwiki.apache.org/confluence/display/KAFKA/KIP-429%3A+Kafka+Consumer+Incremental+Rebalance+Protocol) (free), [microservices.io: Transactional Outbox](https://microservices.io/patterns/data/transactional-outbox.html) (free)
- **Prerequisites**: [Messaging & Distributed Queueing](../../../../infrastructure/02-messaging-and-queueing/) (the conceptual frame this topic operates), [Containers, Kubernetes & Workloads](../../../../infrastructure/01-containers-kubernetes/) (KEDA, SIGTERM, graceful shutdown)
- **Estimated time**: 1-2 weeks at 8-10 hrs/week

## Key Takeaways

- **A worker is a process that pulls work and does it.** This topic is the *operations*: how many run in parallel, what happens on failure, how you size and scale the pool. (For the messaging *dynamics* -- queue vs log, push vs pull, delivery semantics -- see [Messaging & Distributed Queueing](../../../../infrastructure/02-messaging-and-queueing/).)
- **Partition count is the parallelism ceiling.** Within one consumer group a partition is owned by exactly one consumer, so a 5th worker on a 4-partition topic sits idle.
- **Rebalancing is what bites you.** Eager (stop-the-world) reassignment stalls the whole group; a slow handler that misses `max.poll.interval.ms` triggers a false rebalance and reprocessing.
- **At-least-once means design for duplicates.** Idempotency keys, the transactional outbox, and commit-after-durable-work are the operational tools that make redelivery safe.
- **Lag is the master signal.** It drives KEDA scaling and your alerts; at the partition cap, you add partitions, not pods.

## How to Study

- Run a local Kafka or [Redpanda](https://redpanda.com/) via Docker. Produce 1000 messages to `orders`, start consumers one at a time, and watch partitions rebalance.
- Add a consumer beyond the partition count and confirm it sits idle: the topology diagram in [Depth: partitioned consumers](../../README.md#depth-partitioned-consumers), made real.
- Read [`consumer-go.go`](consumer-go.go), [`consumer-rust.rs`](consumer-rust.rs), and [`consumer-python.py`](consumer-python.py) -- the same at-least-once + idempotent + clean-shutdown pattern in three languages.
- Kill a worker mid-batch (don't let it commit) and confirm the message is redelivered to another member. Then add an idempotency guard and confirm the duplicate is absorbed.

---

The concepts behind this lab are in [Depth: partitioned consumers](../../README.md#depth-partitioned-consumers). This side quest has no course module: it is optional reading and practice outside the course spine (`sq.*` in course/DESIGN.md 4.7 lists the graded side quests).
