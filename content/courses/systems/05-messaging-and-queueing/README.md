# Messaging and Distributed Queueing

Most services start life synchronous: a request arrives, the handler does the work, and the response goes back on the same connection. That works until the work is slow, bursty, or unreliable, and then the caller is stuck waiting on whatever is slowest downstream. Asynchronous messaging breaks that coupling. It separates *receiving* a request from *doing* it: a producer drops a message into a broker and walks away, and a pool of workers consumes at its own pace. That one indirection is behind most of the event-driven architectures you will meet in practice.

This chapter is about the dynamics of that indirection. It begins with the deepest dividing line in the space, the difference between a queue that deletes messages on acknowledgement and a log that retains and replays them, and then works outward: who controls the pace of delivery (push, pull, or long-poll), how one stream can feed both a worker pool and many independent subscribers, where the crash window sits that makes "exactly-once" so slippery, and the two formulas (Little's Law and backpressure as an integral) that explain most queueing intuition. It closes with the question every broker cluster must answer, who owns the metadata, and the three answers the industry has given: ZooKeeper, KRaft, and Redpanda.

The running example is **Streamflow**, the event-driven order platform that threads through this book. Its API produces to an `orders` topic; worker pools and analytics jobs consume from it. The containers you scheduled in [Containers, Kubernetes, and Workloads](../10-containers-kubernetes/README.md) are the consumers you will reason about here.

> **Learning objectives**
> - Explain the difference between a delete-on-ack queue and a retained, replayable log, and choose between them for a given workload.
> - Explain why pull-based consumption gives backpressure for free, and what push brokers must add to match it.
> - Show how Kafka consumer groups provide both competing consumers and publish-subscribe at once.
> - Locate the crash window behind at-least-once delivery and design handlers that tolerate duplicates through idempotency.
> - Use Little's Law and the lag integral to reason about in-flight messages, latency, and sustainable throughput.
> - Compare ZooKeeper (ZAB), Kafka KRaft (Raft), and Redpanda (Raft per partition) as answers to cluster coordination.

## Before you begin

You should be comfortable with the replication, quorum, and consensus vocabulary from [System Design](../01-system-design/README.md) in this book. Knowing how a workload is deployed and scaled on Kubernetes, from [Containers, Kubernetes, and Workloads](../10-containers-kubernetes/README.md), helps when lag-based autoscaling comes up.

The chapter ships three small programs you can run while you read. [`code/backpressure_demo.py`](code/backpressure_demo.py) (plain `python3`, no dependencies) and its Go twin [`code/backpressure-demo.go`](code/backpressure-demo.go) (`go run`) simulate a push and a pull consumer side by side: watch the push model drop messages while the pull model converts the surplus into lag, with nothing lost. [`code/raft_election.go`](code/raft_election.go) (`go run`) runs a Raft leader election; trace the terms and votes and confirm that a minority partition can never elect a leader. That is the KRaft controller quorum in miniature. If you want hands-on time with a real broker, stand up a single-broker [Redpanda](https://redpanda.com/) (no ZooKeeper, Kafka-API compatible) and a vanilla [Kafka in KRaft mode](https://kafka.apache.org/documentation/#kraft), produce 1000 messages to `orders`, then start a second consumer *group* and replay from offset 0. You will have proved fan-out and retention for yourself.

## In this chapter

1. [Queues, logs, and consumers](01-queues-logs-and-consumers.md): Choose queue or log semantics from ownership and replay needs.
2. [Delivery, backpressure, and capacity](02-delivery-backpressure-and-capacity.md): Trace crash windows and apply backpressure when arrival rate exceeds service rate.
3. [Coordination and technology choice](03-coordination-and-technology-choice.md): Separate metadata coordination from data delivery and compare mechanisms.
