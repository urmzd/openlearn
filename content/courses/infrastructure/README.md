# Infrastructure

This book is about what happens to a service once it runs at scale, under failure, on shared infrastructure: the containers and orchestration that schedule it, the messaging layers that move its data, and the search engines underneath. It is for engineers who can ship software and now need to reason about the runtime, not just the code. The first chapters share one running example, **Streamflow**, an event-driven order platform.

## What you will be able to do

- Package and deploy a service on Kubernetes, with limits, probes, graceful shutdown, and disruption budgets that prevent outages instead of causing them.
- Design an asynchronous architecture and defend its choice of queue or log, push or pull, and delivery semantics, with numbers from Little's Law.
- Explain how cluster metadata is coordinated by ZooKeeper, KRaft, or Redpanda.
- Build an inverted index, rank it with BM25, and explain how Lucene keeps it fast under writes.
- Place any Apache data project on a map of the problems it solves.

## Ways through this book

Reading every chapter takes five to seven weeks at eight to ten hours a week; Chapter 4 is short enough to read first if all you need is the map.

- **The whole book, in order** (five to seven weeks): [Chapter 1: Containers, Kubernetes, and Workloads](01-containers-kubernetes/README.md), [Chapter 2: Messaging and Distributed Queueing](02-messaging-and-queueing/README.md), the [partitioned consumers](../ai-platform-engineering/05-durable-orchestration-and-workers/README.md#partitioned-consumers) part of Chapter 5 in *AI Platform Engineering* (rebalancing, idempotency, dead-letter queues), then [Chapter 3: Search and Indexing](03-search-and-indexing/README.md) and [Chapter 4: The Apache Stack](04-apache-stack/README.md).
- **Adding Kubernetes support to a service** (about two weeks): the opening of [Chapter 1](01-containers-kubernetes/README.md) for the model and the caveats, then [1.1 Tracer deploy](01-containers-kubernetes/01-tracer-deploy.md) through [1.6 CI for the platform repo](01-containers-kubernetes/06-ci-for-your-repo.md). Add [1.7](01-containers-kubernetes/07-durable-and-worker-charts.md) for stateful or queue-driven workloads.
- **Designing an async architecture** (about two weeks): [Chapter 2](02-messaging-and-queueing/README.md), then [partitioned consumers](../ai-platform-engineering/05-durable-orchestration-and-workers/README.md#partitioned-consumers) and its [Kafka consumers side quest](../ai-platform-engineering/05-durable-orchestration-and-workers/13-kafka-consumers/) in *AI Platform Engineering*, then [1.7](01-containers-kubernetes/07-durable-and-worker-charts.md) to scale the workers on queue depth with KEDA.
- **Building search** (about a week): [Chapter 3](03-search-and-indexing/README.md), then [Retrieval and RAG](../ai-platform-engineering/07-retrieval-and-rag/README.md) in *AI Platform Engineering* for the dense and hybrid side.

## Before you start

You need some experience shipping software and Linux basics. [Containers and Kubernetes](../programming-languages/02-language-and-tool-primers/07-containers-and-kubernetes.md) in *Programming Languages* introduces images, Pods, and Helm if they are new to you; [System Design](../systems/01-system-design/README.md) in *Systems and Architecture* supplies the replication and consensus vocabulary the messaging chapter assumes. Related books go further: [Cloud Native](../systems/03-cloud-native/README.md) in *Systems and Architecture* on orchestration, [Batch and Streaming Processing](../data-engineering/03-batch-streaming/README.md) in *Data Engineering* for the data-engineering view of the same Kafka, and [*Diagramming and Documentation*](../diagramming/README.md) for drawing deployments and consumer topologies honestly. New to these books? [How to read these books](../../references/how-to-read.md) explains how sections, review questions, and solutions work.
