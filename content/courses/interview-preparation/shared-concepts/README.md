# Shared Concepts

The company chapters in this book differ in their coding styles, culture rounds, and pet problems, but their system design rounds keep returning to the same foundations. Whether the prompt is Google's recommendation system, Netflix's streaming platform, OpenAI's API, or NVIDIA's GPU cluster, you will be asked how data is replicated and partitioned, how the system behaves when a node fails, how a model gets from training to serving, and how you would know when something has gone wrong. This chapter collects that shared material once, so the company chapters can build on it.

The chapter moves from general to specific. It begins with the distributed systems concepts that underlie any large-scale design, then climbs the AI stack layer by layer from data to the user-facing application, and ends with the production ML infrastructure that automates training, promotion, monitoring, and retraining. Together these are the foundational skills for a full-stack engineer building and scaling AI systems.

> **Learning objectives**
> - Reason about consistency, replication, partitioning, consensus, and failure in a system design conversation.
> - Describe every layer of an AI system, from data pipelines and feature stores to serving and monitoring.
> - Design production ML infrastructure: registries, promotion pipelines, distributed training, drift detection, and retraining.
> - Support design choices with reference latency numbers and cost arithmetic.

## Before you begin

These sections assume you can read Python and are comfortable with basic data structures. Some familiarity with a deep learning framework helps for the ML sections; the [Deep Learning](../../machine-learning/02-deep-learning/README.md) chapter of *Machine Learning and AI* covers the background. Read this chapter before the company chapters if you are preparing for several companies, since most of them assume it.

## In this chapter

- [Distributed systems fundamentals](distributed-systems.md): the CAP theorem, consistency models, Raft and Paxos, replication and quorums, partitioning, distributed transactions, load balancing, caching, messaging, rate limiting, observability, and back-of-envelope estimation.
- [Full-stack AI engineering](full-stack-ai.md): the AI stack from bottom to top, with code for a data validator, an online feature store, an evaluator, a dynamically batching model server, and a drift detector.
- [ML infrastructure patterns](ml-infrastructure.md): MLOps maturity, point-in-time correctness, the model registry and promotion pipeline, distributed training and checkpointing, experimentation, drift detection, retraining, LLM infrastructure, and GPU cost.
