# Batch & Streaming Processing

## Overview

- **Primary reference**: [*Streaming Systems*](https://www.oreilly.com/library/view/streaming-systems/9781491983867/) by Akidau, Chernyak, Lax (the Dataflow model) -- foundational concepts also in [*The Dataflow Model*](https://research.google/pubs/pub43864/) paper (free)
- **Supplementary**: [Apache Spark docs](https://spark.apache.org/docs/latest/) (free), [Apache Kafka docs](https://kafka.apache.org/documentation/) (free), [Apache Flink docs](https://nightlies.apache.org/flink/flink-docs-stable/) (free), [DDIA](https://dataintensive.net/) Ch 10-11
- **Prerequisites**: [Foundations](../01-foundations/)
- **Estimated time**: 3-4 weeks at 8-10 hrs/week

## Key Takeaways

- Batch and streaming are not separate worlds -- batch is a bounded special case of streaming. The Dataflow model unifies them with four questions: *what, where, when, how*
- The hardest problems in streaming are **time** (event time vs processing time), **late/out-of-order data** (watermarks), and **correctness under failure** (exactly-once)
- **Kafka is the backbone**: a distributed, partitioned, replayable log that decouples producers from consumers
- Spark dominates batch + micro-batch; Flink is the reference for true low-latency stateful streaming

## How to Study

- Run the annotated [`spark/orders_bronze_to_silver.py`](spark/orders_bronze_to_silver.py) job in this directory: it reads one day of raw Streamflow order events, dedupes them with a windowed shuffle, and writes a partitioned silver table. Re-run it for the same date and confirm the output doesn't duplicate (dynamic partition overwrite = idempotency)
- Run a Spark job and a Kafka → consumer pipeline locally; watch partitioning and parallelism
- For any streaming question, separate *event time* (when it happened) from *processing time* (when you saw it) -- most confusion comes from conflating them
- Read the Dataflow paper -- it reframes everything and is the intellectual core of the field

---

# Concepts & Techniques

## Core Insight

All data processing is "take a collection of records, apply a transformation, produce a result." Batch does it over a *bounded* collection and finishes; streaming does it over an *unbounded* collection that never ends, so it must decide *when* a result is "complete enough" to emit. The entire complexity of streaming is dealing with the fact that data arrives late, out of order, and at unpredictable rates -- while still producing correct, exactly-once results.

## 1. Distributed Processing Model

**Key ideas**:
- **Partitioning**: split data into independent shards processed in parallel -- the source of all horizontal scale
- **Shuffle**: redistributing data across the cluster (for joins, group-bys) -- the most expensive operation; minimize it
- **MapReduce lineage**: map (transform per record) → shuffle (group by key) → reduce (aggregate). Spark generalizes this to a DAG of transformations
- **Data skew**: a few hot keys overload some partitions -- the most common cause of slow jobs. Mitigate with salting, repartitioning

## 2. Apache Spark (Batch + Micro-batch)

**Key ideas**:
- **RDDs → DataFrames/Datasets**: from low-level resilient distributed datasets to the optimized, columnar-aware DataFrame API
- **Lazy evaluation + DAG**: transformations build a plan; an *action* triggers execution. Lets Catalyst optimize the whole graph
- **Catalyst optimizer + Tungsten**: query optimization (predicate pushdown, join reordering) and code generation for CPU efficiency
- **Wide vs narrow transformations**: narrow (map, filter) need no shuffle; wide (groupBy, join) do -- the boundary defines stages
- **Structured Streaming**: streaming as an unbounded table; micro-batch (default) or continuous processing

**Worked example**: [`spark/orders_bronze_to_silver.py`](spark/orders_bronze_to_silver.py) is a runnable bronze→silver batch job. It makes three of these ideas concrete -- the dedup is a *wide* transformation (a shuffle via `Window.partitionBy`), the output is columnar Parquet *partitioned by date*, and the write uses dynamic partition overwrite for *idempotent* re-runs and backfills. It produces the silver table the [dbt project](../04-orchestration-modeling/dbt/) builds its marts on.

## 3. Apache Kafka (The Log)

**Key ideas**:
- **The unifying abstraction**: an append-only, partitioned, replicated commit log. Producers append; consumers read at their own offset
- **Topics & partitions**: ordering is guaranteed *within* a partition, not across; partition count sets max consumer parallelism
- **Consumer groups**: partitions are divided among group members for scale-out; offsets track progress
- **Retention & replay**: data persists for a retention window (or forever via compaction) -- consumers can rewind and reprocess. This is what makes Kafka a source of truth, not just a queue
- **Log compaction**: keep only the latest value per key -- turns a topic into a changelog/materialized table

## 4. Event Time, Watermarks & Windows

**The heart of streaming -- Streaming Systems book**

**Key ideas**:
- **Event time vs processing time**: when the event *occurred* vs when the system *processed* it. Network delays, retries, and outages make them diverge -- always aggregate on event time for correctness
- **Watermarks**: the system's estimate of "event time has progressed to T; no more data older than T is expected." Drives when windows can close
- **Windowing**: bound an unbounded stream into finite chunks --
  - *Tumbling*: fixed, non-overlapping (every 5 min)
  - *Sliding*: fixed size, overlapping (last 5 min, every 1 min)
  - *Session*: gap-based, dynamic (activity bursts)
- **Triggers & late data**: when to emit a window result, and what to do with data that arrives after the watermark (drop, or update via *allowed lateness* + retractions)

## 5. The Dataflow Model: What/Where/When/How

**Key ideas**:
- **What** results are computed → transformations (sum, count, join)
- **Where** in event time → windowing
- **When** in processing time → watermarks + triggers
- **How** refinements relate → accumulation mode (discard, accumulate, accumulate-and-retract)

This framework subsumes batch and streaming: batch is just streaming with a single global window and one trigger at the end.

## 6. Delivery Semantics & Fault Tolerance

**Key ideas**:
- **At-most-once**: may lose data, never duplicates (fire and forget)
- **At-least-once**: never loses, may duplicate (retries without dedup)
- **Exactly-once**: each record affects state once -- the gold standard
- **How exactly-once is achieved**: idempotent writes + transactional sinks, or checkpoint/snapshot of operator state (Flink's Chandy-Lamport distributed snapshots) tied to source offsets
- **Checkpointing**: periodically persist processing state + input position so a failed job resumes without reprocessing or dropping. The foundation of stateful streaming recovery

## 7. Apache Flink (True Streaming)

**Key ideas**:
- **Streaming-first**: processes events one at a time (not micro-batches) for true low latency
- **Stateful operators**: keyed state, managed and checkpointed; enables complex event processing and large aggregations
- **Exactly-once via snapshots**: aligned checkpoint barriers flow through the dataflow graph
- **When to choose**: sub-second latency, heavy stateful logic, CEP -- vs Spark for unified batch + simpler micro-batch streaming

## 8. Architectural Patterns

| Pattern | Idea | Tradeoff |
|---------|------|----------|
| Lambda | Separate batch + speed layers, merge at query | Correctness + freshness, but two codebases |
| Kappa | Stream-only; replay the log to "rebatch" | One codebase; needs replayable log (Kafka) |
| Medallion | Bronze (raw) → Silver (cleaned) → Gold (modeled) | Clear lineage, incremental refinement |
| CDC | Stream DB changes via the WAL | Low-impact, near-real-time ingestion from OLTP |

---

## Batch vs Streaming Decision

| Need | Choose | Why |
|------|--------|-----|
| Daily/hourly reports, backfills | Batch (Spark) | Simpler, cheaper, exact |
| Sub-minute freshness, alerting | Streaming (Flink/Spark SS) | Low latency |
| Decouple producers/consumers, replay | Kafka | Durable log |
| Fast DB-to-warehouse sync | CDC + stream | Near-real-time, low source impact |
| Unifying both | Kappa + Dataflow model | One pipeline, replay for batch |

## Connections to Other Tracks

| Concept | Connected Track | Application |
|---------|-----------------|-------------|
| Messaging, queues, event systems | [System Design](../../systems/01-system-design/) | Kafka, async architecture |
| Consensus, replication, snapshots | [System Design](../../systems/01-system-design/) | Exactly-once, checkpointing |
| Streaming feature pipelines | [LLM Systems](../../ml/04-llm-systems/) | Real-time features for models |
| MapReduce, DAG scheduling | [Graphs](../../algorithms/06-graphs/) | Stage/dependency graphs |

## Company Relevance

| Company | How This Appears | Difficulty |
|---------|-----------------|------------|
| Netflix | Massive Kafka + Flink streaming platform | Expert |
| Databricks | Spark (created here), Structured Streaming | Expert |
| LinkedIn | Kafka (created here), stream processing | Expert |
| Uber | Flink, real-time pricing/ETA, Kappa | Expert |
| Confluent | Kafka as a product, exactly-once semantics | Expert |
