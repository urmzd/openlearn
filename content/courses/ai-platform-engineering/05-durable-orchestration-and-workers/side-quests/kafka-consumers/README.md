# Kafka consumer groups

This side quest is a hands-on lab for the other common worker shape, a pool of consumers reading a partitioned log. A worker is a process that pulls work and does it; this lab is about the *operations*: how many workers run in parallel, what happens on failure, and how you size and scale the pool. The concepts are explained in [Partitioned consumers](../../README.md#partitioned-consumers) in this chapter's opening, and the messaging *dynamics* behind them (queue versus log, push versus pull, delivery semantics) are in [Messaging and Distributed Queueing](../../../../infrastructure/02-messaging-and-queueing/README.md) in *Infrastructure*. The lab also uses KEDA, SIGTERM handling, and graceful shutdown from [Containers, Kubernetes, and Workloads](../../../../infrastructure/01-containers-kubernetes/README.md). It is optional and sits outside the main line of the book.

> **Learning objectives**
> - Show that partition count is the parallelism ceiling: within one consumer group a partition is owned by exactly one consumer, so a fifth worker on a four-partition topic sits idle.
> - Observe rebalancing, and explain why eager reassignment stalls the whole group and why a slow handler that misses `max.poll.interval.ms` triggers a false rebalance and reprocessing.
> - Make redelivery safe with idempotency keys, the transactional outbox, and commit-after-durable-work.
> - Use lag as the master signal for alerting and KEDA scaling, and explain why at the partition cap you add partitions, not pods.

## The code

The directory holds the same consumer in three languages: [`consumer-go.go`](consumer-go.go), [`consumer-rust.rs`](consumer-rust.rs), and [`consumer-python.py`](consumer-python.py). Each implements one pattern: at-least-once consumption with an idempotency guard, cooperative rebalancing, explicit offset commits after the handler returns, and a clean leave from the group on SIGTERM. Read them side by side; the differences are only in each client library's API. The diagram [`kafka-consumption.svg`](kafka-consumption.svg) (source [`kafka-consumption.d2`](kafka-consumption.d2)) shows the partition-to-consumer topology.

## The lab

Run a local Kafka or [Redpanda](https://redpanda.com/) via Docker. Produce 1000 messages to an `orders` topic, then start consumers one at a time and watch the partitions rebalance across them as each joins.

Next, add one consumer more than the partition count and confirm that it sits idle. This is the topology diagram from [Partitioned consumers](../../README.md#partitioned-consumers), made real: no partition is left for the extra member to own, so adding it buys nothing.

Finally, test redelivery. Kill a worker mid-batch, before it commits, and confirm that its messages are redelivered to another member of the group. Then add an idempotency guard and confirm the duplicate is absorbed: the second delivery sees the key already claimed, skips the work, and lets the offset advance.

## Summary

- Partition count is the parallelism ceiling; workers beyond it sit idle.
- Rebalancing is what bites you: use cooperative rebalancing, keep handlers inside `max.poll.interval.ms`, and leave the group cleanly on SIGTERM.
- At-least-once means designing for duplicates: idempotency keys, the transactional outbox, and committing only after durable work.
- Lag drives scaling and alerts; at the partition cap, add partitions, not pods.

## Further reading

- [Apache Kafka documentation](https://kafka.apache.org/documentation/), especially the Consumer and Consumer Group Protocol sections.
- Confluent, [Consumer group protocol](https://developer.confluent.io/courses/architecture/consumer-group-protocol/), for rebalancing in depth.
- Martin Kleppmann, *Designing Data-Intensive Applications*, chapter 11, "Stream Processing".
- [Enterprise Integration Patterns](https://www.enterpriseintegrationpatterns.com/): Competing Consumers, Dead Letter Channel, Idempotent Receiver.
- [KIP-429: Incremental Cooperative Rebalancing](https://cwiki.apache.org/confluence/display/KAFKA/KIP-429%3A+Kafka+Consumer+Incremental+Rebalance+Protocol).
- microservices.io, [Transactional Outbox](https://microservices.io/patterns/data/transactional-outbox.html).
