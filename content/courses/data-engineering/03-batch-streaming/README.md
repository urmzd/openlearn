# Batch and Streaming Processing

All data processing has the same shape: take a collection of records, apply a transformation, produce a result. Batch processing does this over a *bounded* collection and then finishes. Streaming does it over an *unbounded* collection that never ends, so it must decide *when* a result is complete enough to emit. Almost all of the complexity of streaming comes from one fact: data arrives late, out of order, and at unpredictable rates, and you still have to produce correct, exactly-once results.

The reassuring insight of this chapter is that batch and streaming are not separate worlds. Batch is a bounded special case of streaming, and the Dataflow model unifies the two with four questions: *what*, *where*, *when*, and *how*. You will build up to that model from the distributed processing basics underneath every engine, then meet the three systems that dominate practice. Spark dominates batch and micro-batch work. Kafka is the backbone: a distributed, partitioned, replayable log that decouples producers from consumers. Flink is the reference for true low-latency stateful streaming. Along the way you will face the three hard problems of streaming: **time** (event time versus processing time), **late and out-of-order data** (watermarks), and **correctness under failure** (exactly-once).

The chapter is anchored by a runnable example, [`spark/orders_bronze_to_silver.py`](spark/orders_bronze_to_silver.py), the batch stage of the Streamflow orders pipeline that runs through this book.

> **Learning objectives**
> - Explain partitioning, shuffles, and data skew, and identify which Spark transformations force a shuffle.
> - Describe Kafka's partitioned log, consumer groups, retention, and log compaction, and explain why replay makes it a source of truth.
> - Distinguish event time from processing time and use watermarks, windows, and triggers to handle late data.
> - Frame any pipeline with the Dataflow model's what/where/when/how questions.
> - Compare at-most-once, at-least-once, and exactly-once delivery and explain how checkpointing and idempotent sinks achieve exactly-once.

## Before you begin

This chapter builds on [Foundations](../01-foundations/), especially idempotency and the batch-versus-streaming framing. The intellectual core of the field is Akidau, Chernyak, and Lax's [*Streaming Systems*](https://www.oreilly.com/library/view/streaming-systems/9781491983867/), whose ideas first appeared in [*The Dataflow Model*](https://research.google/pubs/pub43864/) paper (free); reading the paper reframes everything that follows. Chapters 10 and 11 of [*Designing Data-Intensive Applications*](https://dataintensive.net/) cover the same ground from the batch side. To get your hands dirty, run a Spark job and a Kafka-to-consumer pipeline locally and watch how partitioning drives parallelism. And whenever a streaming question confuses you, separate *event time* (when it happened) from *processing time* (when you saw it); most confusion comes from conflating them.

## The distributed processing model

Every engine in this chapter, batch or streaming, rests on the same few ideas.

**Partitioning** splits data into independent shards that are processed in parallel. It is the source of all horizontal scale: more partitions, more machines working at once.

A **shuffle** redistributes data across the cluster so that all records with the same key end up on the same machine, which joins and group-bys require. It moves data over the network and through disk, which makes it the most expensive operation in distributed processing. Minimizing shuffles is most of performance tuning.

These two ideas descend from **MapReduce**: a *map* step transforms each record independently, a *shuffle* groups records by key, and a *reduce* step aggregates each group. Spark generalizes this fixed three-step shape to an arbitrary DAG (directed acyclic graph) of transformations.

The most common cause of slow jobs is **data skew**: a few hot keys send far more records to some partitions than to others, so a handful of tasks run long after the rest have finished. You mitigate it by *salting* hot keys (appending a random suffix so one key spreads over several partitions, then combining the partial results) or by repartitioning.

## Apache Spark for batch and micro-batch

Spark is where those ideas become an API. Its original abstraction, the **RDD** (resilient distributed dataset), is a low-level partitioned collection. Most code today uses **DataFrames and Datasets** instead, which carry a schema and are columnar-aware, so the engine can optimize them.

Spark uses **lazy evaluation**. Transformations do not run when you call them; they build up a plan, a DAG of operations. Only an *action*, such as writing output or collecting a result, triggers execution. Because Spark sees the whole graph before running anything, its **Catalyst** optimizer can rewrite the plan (pushing predicates down toward the source, reordering joins), and its **Tungsten** engine generates efficient code for the CPU.

The distinction that matters most when reading a Spark plan is between narrow and wide transformations. A **narrow** transformation such as `map` or `filter` computes each output partition from one input partition and needs no shuffle. A **wide** transformation such as `groupBy` or `join` needs records from many input partitions and therefore shuffles. Shuffle boundaries are where Spark splits a job into *stages*.

Spark also handles streams through **Structured Streaming**, which treats a stream as an unbounded table that rows keep being appended to. It runs as micro-batches by default, with an optional continuous processing mode.

### Working through the bronze-to-silver job

The runnable job [`spark/orders_bronze_to_silver.py`](spark/orders_bronze_to_silver.py) makes three of these ideas concrete. It reads one day of raw Streamflow order events that were landed from Kafka as JSON, cleans and deduplicates them, and writes a partitioned silver table that the [dbt project](../04-orchestration-modeling/01-dbt/) in the next chapter builds its marts on.

The job starts by declaring an explicit schema rather than inferring one. That schema is a data contract with the producer: if raw events stop matching it, the job fails visibly instead of silently shifting columns and producing wrong numbers.

```python
RAW_ORDER_SCHEMA = StructType(
    [
        StructField("order_id", StringType(), nullable=False),
        StructField("customer_id", StringType(), nullable=False),
        StructField("status", StringType(), nullable=False),
        StructField("amount", DoubleType(), nullable=False),
        StructField("currency", StringType(), nullable=False),
        StructField("event_ts", TimestampType(), nullable=False),
    ]
)
```

It then reads only the landing directory for the run date (`ingest_date={run_date}`) and derives `order_date` from the event timestamp. Slicing by date is what makes each run independent and safely re-runnable.

The deduplication step is the *wide* transformation. Kafka delivers at least once, so the same `order_id` can appear more than once. The job keeps the latest event per order by ranking rows within a window partitioned by `order_id` and filtering to rank 1:

```python
latest_per_order = Window.partitionBy("order_id").orderBy(F.col("event_ts").desc())
deduped = (
    raw.withColumn("_rank", F.row_number().over(latest_per_order))
    .where(F.col("_rank") == 1)
    .drop("_rank")
)
```

To rank within each key, Spark must co-locate all rows for that key on one executor. That is the shuffle.

Next comes a clean-and-conform step: `status` is lowercased and trimmed, `currency` uppercased and trimmed, and rows with a negative `amount` (a known bad-data signal upstream) are dropped. This is the silver contract boundary: everything downstream trusts these types and values.

Finally, the write. The output is columnar Parquet *partitioned by* `order_date`, so analytical queries prune to the days they need and read only the columns they select. The session is configured with `spark.sql.sources.partitionOverwriteMode` set to `dynamic`, and the write uses overwrite mode:

```python
silver.write.mode("overwrite").partitionBy("order_date").parquet(args.silver)
```

With dynamic partition overwrite, writing the 2026-06-14 partition replaces *only* that partition; every other day is untouched. A retry or backfill for one date therefore cannot duplicate data or touch other days. Run the job twice for the same date and confirm the output does not duplicate. That is idempotency, the single most important property of a batch pipeline.

## Apache Kafka: the log

The bronze events that job reads came from Kafka, and Kafka deserves its own treatment because it is the unifying abstraction of streaming architecture: an append-only, partitioned, replicated commit log. Producers append records to the end; each consumer reads at its own *offset*, a position in the log.

A **topic** is split into **partitions**. Ordering is guaranteed *within* a partition, not across partitions, so records that must stay in order (all events for one order, say) should share a partition key. The partition count also sets the maximum consumer parallelism.

**Consumer groups** scale reading out. The partitions of a topic are divided among the members of a group, and committed offsets track each group's progress.

What separates Kafka from a traditional queue is **retention and replay**. Data persists for a retention window, or indefinitely, rather than disappearing when consumed, so consumers can rewind and reprocess. That is what makes Kafka a source of truth rather than just a queue. **Log compaction** goes further: it keeps only the latest value per key, which turns a topic into a changelog, effectively a materialized table.

## Event time, watermarks, and windows

Kafka gives you a durable, ordered (per partition) stream. It does not tell you when an aggregate over that stream is finished. This is the heart of streaming, and the central idea of *Streaming Systems*.

Every event has two times. **Event time** is when the event occurred. **Processing time** is when the system processed it. Network delays, retries, and outages make the two diverge, sometimes by hours. For correct results, always aggregate on event time; aggregating on processing time makes your numbers depend on how fast your pipeline happened to run.

But if you aggregate on event time, when can you be sure you have seen all the events for, say, 10:00 to 10:05? A **watermark** is the system's estimate that event time has progressed to $T$: no more data older than $T$ is expected. Watermarks drive when windows can close.

**Windowing** bounds an unbounded stream into finite chunks so that aggregates are defined. There are three common shapes:

- *Tumbling* windows are fixed and non-overlapping, for example every 5 minutes.
- *Sliding* windows have a fixed size and overlap, for example the last 5 minutes, recomputed every minute.
- *Session* windows are gap-based and dynamic, capturing bursts of activity separated by idle time.

Watermarks are estimates, so some data will arrive after them. **Triggers** decide when to emit a window's result (at the watermark, early and speculatively, or repeatedly), and a late-data policy decides what to do with data that arrives after the watermark: drop it, or keep the window open for an *allowed lateness* period and update the result, issuing retractions of the earlier value.

## The Dataflow model: what, where, when, how

Those mechanisms fit together into a single framework, the Dataflow model, which asks four questions of any pipeline:

- **What** results are computed? This is the transformation: sum, count, join.
- **Where** in event time are they computed? This is windowing.
- **When** in processing time are results emitted? This is watermarks plus triggers.
- **How** do refinements of a result relate to each other? This is the accumulation mode: discard previous results, accumulate into them, or accumulate and retract the old value.

This framework subsumes batch and streaming. Batch is just streaming with a single global window and one trigger that fires at the end, once all the (bounded) input has been seen.

## Delivery semantics and fault tolerance

The Dataflow model tells you what to compute. Failures decide whether you compute it correctly. Systems offer three delivery guarantees.

**At-most-once** delivery may lose data but never duplicates it; it is fire and forget. **At-least-once** never loses data but may duplicate it, typically because of retries without deduplication; this is why the Spark job above deduplicates. **Exactly-once** means each record affects state exactly once, and it is the gold standard.

Exactly-once is achieved in one of two ways. The first is idempotent writes combined with transactional sinks, so a replayed record overwrites rather than duplicates. The second is to snapshot operator state and tie each snapshot to the source offsets it reflects; Flink does this with distributed snapshots based on the Chandy-Lamport algorithm.

Both rely on **checkpointing**: periodically persisting processing state together with the input position, so that a failed job resumes from the last checkpoint without reprocessing (and double counting) or dropping data. Checkpointing is the foundation of recovery for stateful streaming.

## Apache Flink for true streaming

Flink is the system that takes those fault-tolerance ideas furthest. It is streaming-first: it processes events one at a time rather than in micro-batches, which gives true low latency. Its operators are stateful, with keyed state that Flink manages and checkpoints for you, which enables complex event processing (CEP) and large aggregations. It achieves exactly-once through snapshots: aligned checkpoint barriers flow through the dataflow graph with the data, and each operator snapshots its state when the barrier passes.

Choose Flink when you need sub-second latency, heavy stateful logic, or CEP. Choose Spark when you want one engine for batch and simpler micro-batch streaming.

## Architectural patterns

Engines are assembled into a handful of recurring architectures. The table below names the four you will meet most often.

| Pattern | Idea | Tradeoff |
|---------|------|----------|
| Lambda | Separate batch + speed layers, merge at query | Correctness + freshness, but two codebases |
| Kappa | Stream-only; replay the log to "rebatch" | One codebase; needs replayable log (Kafka) |
| Medallion | Bronze (raw) → Silver (cleaned) → Gold (modeled) | Clear lineage, incremental refinement |
| CDC | Stream DB changes via the WAL | Low-impact, near-real-time ingestion from OLTP |

Lambda was the first answer to "we need both correct and fresh": a batch layer recomputes exact results while a speed layer provides approximate fresh ones. Its cost is maintaining the same logic twice. Kappa removes the batch layer by relying on a replayable log: to recompute history, you replay the log through the same streaming code. The medallion pattern is the layering the Streamflow example uses, with the Spark job producing silver from bronze and dbt producing gold. CDC streams database changes from the write-ahead log (WAL).

Putting the choices side by side gives a practical decision guide:

| Need | Choose | Why |
|------|--------|-----|
| Daily/hourly reports, backfills | Batch (Spark) | Simpler, cheaper, exact |
| Sub-minute freshness, alerting | Streaming (Flink/Spark SS) | Low latency |
| Decouple producers/consumers, replay | Kafka | Durable log |
| Fast DB-to-warehouse sync | CDC + stream | Near-real-time, low source impact |
| Unifying both | Kappa + Dataflow model | One pipeline, replay for batch |

## Where this leads

The silver table this chapter's job produces is the input to the [orchestration and modeling](../04-orchestration-modeling/) chapter, where dbt builds tested marts on it and Airflow runs the whole pipeline as one DAG.

The ideas here connect widely. Messaging, queues, and event-driven architecture, along with the consensus, replication, and snapshot techniques behind exactly-once and checkpointing, are covered in [System Design](../../systems/01-system-design/) in *Systems Engineering*. Streaming feature pipelines that deliver real-time features to models appear in [LLM Systems](../../ai-engineering/20-inference-engines/). MapReduce stages and DAG scheduling are dependency graphs, studied in [Graphs](../../programming-and-algorithms/08-graphs/) in *Programming and Algorithms*.

In industry, Netflix runs a massive Kafka and Flink streaming platform. Spark was created by the team that went on to found Databricks, which also built Structured Streaming. Kafka was created at LinkedIn, and Confluent sells it as a product, with exactly-once semantics as a headline feature. Uber uses Flink and a Kappa architecture for real-time pricing and ETAs.

### Further reading

- Tyler Akidau, Slava Chernyak, and Reuven Lax, [*Streaming Systems*](https://www.oreilly.com/library/view/streaming-systems/9781491983867/).
- Akidau et al., [*The Dataflow Model*](https://research.google/pubs/pub43864/) (free).
- [Apache Spark docs](https://spark.apache.org/docs/latest/), [Apache Kafka docs](https://kafka.apache.org/documentation/), and [Apache Flink docs](https://nightlies.apache.org/flink/flink-docs-stable/) (all free).
- Martin Kleppmann, [*Designing Data-Intensive Applications*](https://dataintensive.net/), chapters 10 and 11.

## Key terms

- **Shuffle**: redistributing records across the cluster by key, required by joins and group-bys and the most expensive distributed operation.
- **Data skew**: uneven distribution of records over partitions caused by hot keys.
- **Narrow / wide transformation**: a transformation that needs no shuffle (map, filter) versus one that does (groupBy, join); wide transformations define stage boundaries.
- **Lazy evaluation**: building a plan from transformations and executing it only when an action runs.
- **Offset**: a consumer's position in a Kafka partition.
- **Log compaction**: retaining only the latest value per key in a Kafka topic.
- **Event time / processing time**: when an event occurred versus when the system processed it.
- **Watermark**: the system's estimate that no more data older than a given event time is expected.
- **Trigger**: the rule deciding when a window's result is emitted.
- **Exactly-once**: the guarantee that each record affects state exactly once.
- **Checkpoint**: a persisted snapshot of processing state and input position used for recovery.
- **Dynamic partition overwrite**: a write mode that replaces only the partitions being written, making per-date reruns idempotent.

## Summary

- Batch is a bounded special case of streaming; the Dataflow model's what/where/when/how questions unify them.
- Partitioning gives scale, shuffles are the main cost, and skew is the main cause of slow jobs.
- Spark's lazy DAG lets Catalyst optimize whole plans; wide transformations shuffle and define stages.
- Kafka is a partitioned, replicated, replayable log; replay and compaction make it a source of truth.
- Aggregate on event time, use watermarks and triggers to decide when to emit, and plan for late data.
- Exactly-once comes from idempotent or transactional sinks and checkpointed state tied to source offsets.

## Review questions

1. In the bronze-to-silver job, why does deduplication cause a shuffle, and why is deduplication needed at all?
   <details><summary>Answer</summary>Kafka delivers at least once, so the same `order_id` can arrive more than once. Ranking events within `Window.partitionBy("order_id")` requires all rows for a key to be on one executor, so Spark must redistribute rows by key, which is a shuffle.</details>

2. What makes rerunning the bronze-to-silver job for the same date safe?
   <details><summary>Answer</summary>It reads only that date's landing directory and writes in overwrite mode with `partitionOverwriteMode` set to `dynamic`, partitioned by `order_date`. Only the partition being written is replaced, so a rerun replaces that day without duplicating it or touching other days.</details>

3. Why should a streaming aggregation use event time rather than processing time?
   <details><summary>Answer</summary>Network delays, retries, and outages make processing time diverge from when events actually happened. Aggregating on processing time makes results depend on pipeline speed; event time gives the correct answer regardless of when data arrives.</details>

4. What is a watermark, and what happens to data that arrives after it?
   <details><summary>Answer</summary>A watermark is the system's estimate that event time has progressed to $T$ and no older data is expected; it lets windows close. Late data is either dropped or, with allowed lateness, used to update the window's result, with retractions of the earlier output.</details>

5. Express a nightly batch job in the terms of the Dataflow model.
   <details><summary>Answer</summary>What: the transformation it computes. Where: a single global window over the bounded input. When: one trigger that fires at the end, after all input is seen. How: there are no refinements, so the accumulation mode does not matter.</details>

6. Why is ordering in Kafka guaranteed only within a partition, and how do you keep related events in order?
   <details><summary>Answer</summary>Each partition is an independent append-only log consumed in parallel, so there is no global order across partitions. Giving related events the same partition key (for example, the order id) routes them to the same partition, where their order is preserved.</details>

7. When would you choose Kappa over Lambda?
   <details><summary>Answer</summary>When you have a replayable log such as Kafka and want one codebase: Kappa recomputes history by replaying the log through the same streaming code, avoiding Lambda's duplicated batch and speed layers.</details>
