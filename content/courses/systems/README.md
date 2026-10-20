# Systems Engineering

Understand a production system from its architecture and communication contracts to distributed execution, deployment, telemetry, and recovery. Worked examples connect services, queues, caches, search, and Kubernetes to their guarantees and failure modes. Streamflow supplies infrastructure examples; the companion model platform supplies design reviews, deployment exercises, and incident drills.

## What you will be able to do

- Derive a design from access patterns, scale, consistency, and failure requirements, then defend its tradeoffs.
- Choose protocols, streams, queues, caches, and indexes by their guarantees and measured behavior.
- Design durable execution with explicit retry, cancellation, and recovery semantics.
- Deploy services reproducibly and connect telemetry to SLOs and diagnosis.
- Predict failures, test recovery, and explain what the evidence establishes.

## Ways through this book

Work one chapter at a time, allowing extra time for experiments. The three parts cover design and communication, data and execution, and deployment and operations; the Apache ecosystem is an appendix.

- **Designing services:** begin with [System Design](01-system-design/README.md), [Software Architecture](02-software-architecture/README.md), [RPC and Protocols](03-rpc-and-protocols/README.md), and [Streaming](04-streaming/README.md). State each interface's invariant and predict its failure behavior.
- **Distributed execution:** follow [Messaging](05-messaging-and-queueing/README.md), [Distributed Data and Caching](06-distributed-data-and-caching/README.md), [Durable Orchestration](07-durable-orchestration/README.md), and [Search and Indexing](08-search-and-indexing/README.md). Compare consistency, replay, retries, and idempotency across the examples.
- **Operating a platform:** read [Cloud Native Architecture](09-cloud-native/README.md), [Kubernetes Workloads and Deployment](10-containers-kubernetes/README.md), [Observability](11-observability/README.md), and [Incident Response and Recovery](12-incident-response-and-chaos/README.md). Finish with the [Event API case study](13-case-study-event-api/README.md), demonstrating duplicate delivery and restart recovery.
- **The whole book:** follow the thirteen chapters in order. Use [The Apache Ecosystem](14-apache-stack/README.md) to locate specific projects. Defend a design, demonstrate a controlled failure, and report the limits of your checks.

## Before you start

Be comfortable with programming, Linux basics, and data structures from [Programming and Algorithms](../programming-and-algorithms/README.md). Its [Containers and Kubernetes primer](../programming-and-algorithms/01-language-primers/07-containers-and-kubernetes.md) prepares you for deployments.

[Software Engineering](../software-engineering/README.md) covers testing, patterns, authorization, diagrams, and runbooks. [Data Engineering](../data-engineering/README.md) covers pipelines and analytical data. [AI Engineering](../ai-engineering/README.md) applies systems mechanisms to models, gateways, and agents. These books link to the general systems chapters here.

Reference implementations and operational examples live in [`course/solutions/`](../../../course/solutions/). [How to read these books](../../references/how-to-read.md) explains the shared method.
