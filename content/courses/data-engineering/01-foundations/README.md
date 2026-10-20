# Foundations

Data engineering is the work of taking data that was produced for one purpose and making it reliably available for a different one. An application database exists to run the application; a sensor emits readings because that is what sensors do; a web server writes logs so operators can debug it. None of these were designed for the analyst who wants weekly revenue by region, or for the ML engineer who needs a clean training set. The data engineer closes that gap. Every design decision in the field flows from the distance between how data is *generated* and how it will be *consumed*: transactional versus analytical, real-time versus batch, raw versus modeled.

It helps to be blunt about the purpose up front. Data engineering exists to make data *usable* downstream. It is plumbing in service of analytics and ML, not an end in itself, and most pipeline failures come from choosing tools for novelty instead of for the access pattern and service-level agreement (SLA) they actually have to meet.

This chapter gives you the vocabulary and the mental map that the rest of the book assumes. You will start with the data engineering lifecycle, which is the map of the whole field. You will then meet the single most important distinction in analytical systems, OLTP versus OLAP, and see how it drives storage layout, file formats, and the shift from ETL to ELT. The chapter closes with the reliability ideas (consistency, idempotency) and the batch-versus-streaming framing that later chapters develop in depth.

> **Learning objectives**
> - Place any data system on the lifecycle (generation, ingestion, storage, transformation, serving) and name the undercurrents that cut across it.
> - Explain why column-oriented storage wins for analytics and classify a system as OLTP or OLAP.
> - Contrast ETL and ELT and explain why ELT became the modern default.
> - Choose between row and columnar formats, schema-on-write and schema-on-read, and normalized and denormalized models for a given workload.
> - Define idempotency and explain why it makes retries and backfills safe.

## Before you begin

You should be comfortable writing SQL and have the basics of distributed systems from the [System Design](../../systems/01-system-design/) chapter of *Systems Engineering*. Chapters 1 and 2 of Martin Kleppmann's [*Designing Data-Intensive Applications*](https://dataintensive.net/) (DDIA) are the best companion reading: they introduce reliability, scalability, maintainability, and data models, which this book uses throughout. As you read, a useful habit is to classify every system you meet twice, once by where it sits on the lifecycle and once as OLTP or OLAP. And before you go deep on any one stage, it is worth building a tiny end-to-end pipeline yourself: ingest from an API, land the data in object storage, transform it with SQL, and query the result.

## The data engineering lifecycle

The lifecycle is the mental map for the entire field. Data moves through five stages, and every tool you will ever evaluate lives in one or more of them.

**Generation** is where data originates: application databases, APIs, event streams, IoT devices, files. The defining fact about source systems is that you usually don't control them. Their schemas change, their uptime is someone else's problem, and their owners did not design them with you in mind.

**Ingestion** gets the data in. It can be batch (scheduled pulls) or streaming (continuous). It can be push (the source sends) or pull (you fetch). For databases, the preferred technique is often change data capture (CDC), which reads the database's own change log so you can follow every insert, update, and delete without hammering the production tables with queries.

**Storage** is where data lands: object storage, data lakes, warehouses. It is the substrate that every other stage sits on, which is why the next chapter is devoted to it.

**Transformation** turns raw data into useful shapes by cleaning, joining, aggregating, and modeling it.

**Serving** delivers the result to its consumers: BI dashboards, ML features, and "reverse ETL," which pushes modeled data back into operational applications such as a CRM.

Running underneath all five stages are the *undercurrents*: security, data management and governance, DataOps, data architecture, orchestration, and software engineering. They are not stages you pass through once; they are concerns you carry through every stage.

## OLTP versus OLAP

With the map in hand, the next question is what kind of system sits at each point on it. The foundational split is between systems that *run* the business and systems that *analyze* it. The following table summarizes the two.

| | OLTP | OLAP |
|---|------|------|
| Purpose | Run the application | Analyze the business |
| Access | Many small reads/writes | Few large scans/aggregates |
| Layout | Row-oriented | Column-oriented |
| Latency | Milliseconds | Seconds to minutes |
| Examples | Postgres, MySQL, DynamoDB | Snowflake, BigQuery, Redshift, DuckDB |

Online transaction processing (OLTP) systems serve the application: fetch this user, update that order, many times per second, each touching a handful of rows. Online analytical processing (OLAP) systems serve questions about the business: total revenue by month over three years, which touches millions of rows but only a few columns.

That access pattern is exactly why columnar storage wins for analytics. An analytical query reads a few columns across many rows. If you store each column contiguously, the engine reads only the columns the query names and skips the rest entirely. Storing columns together also compresses far better, because similar values (all the dates, all the country codes) sit next to each other. This is the single most important idea in analytical storage, and it dictates storage layout, file format, and system choice for everything that follows.

## ETL versus ELT

Once you know the destination is a columnar analytical system, you have to decide *where* transformation happens. There are two answers.

**ETL** (extract, transform, load) transforms data *before* loading it, on a dedicated engine, and loads only the finished result. It was the old default when storage and compute were expensive and you could not afford to keep raw data around in the warehouse.

**ELT** (extract, load, transform) lands raw data first and transforms it *in* the warehouse with SQL. It is the modern default because cheap object storage and elastic compute make it cheaper to keep raw data and transform on demand.

ELT won for several reasons that reinforce one another. Storage and compute are decoupled, so you pay for each separately and scale each independently. Raw data is kept, so it is replayable: when you find a bug in a transformation, you fix the SQL and rebuild instead of re-extracting from sources that may no longer have the data. Transformation logic lives in version-controlled SQL (the role [dbt](../04-orchestration-modeling/) plays in the chapter on orchestration and modeling). And schema-on-read gives you flexibility to interpret raw data in new ways later.

## Storage formats

ELT puts files on object storage, so the format of those files matters. Formats fall into a few families.

Row formats such as Avro, JSON, and CSV store one record after another. They are good for write-heavy, record-at-a-time workloads, and Avro in particular handles schema evolution well. Columnar formats such as Parquet and ORC are the analytics workhorses. They support *column pruning* (read only the columns you need), *predicate pushdown* (skip chunks of the file whose statistics show they cannot match a filter), and dictionary and run-length compression.

Two more families sit alongside these. For compact, typed wire formats between services, Protobuf and Avro are the usual serialization choices. For in-memory interchange between tools, Apache Arrow provides a zero-copy columnar layout, so a DataFrame library and a query engine can share the same buffers without converting.

Whatever the format, you also choose a compression codec, which trades CPU for size: Snappy is fast, Zstd is balanced, and Gzip produces small files at higher CPU cost.

## Data models and schemas

Formats describe how bytes are laid out; models describe what the data *means* and when that meaning is enforced. Here again the choice follows the workload.

A warehouse typically uses **schema-on-write**: structure is enforced at load time, which is safe but rigid. A lake typically uses **schema-on-read**: data is stored raw and interpreted at query time, which is flexible but risky, since a malformed record is discovered only when someone reads it.

The same OLTP/OLAP split shapes table design. OLTP systems use **normalization** (third normal form, 3NF) to avoid update anomalies, where the same fact stored twice drifts out of sync. OLAP systems use **denormalization**, typically a star schema, to avoid the cost of joins at query time. You will build a star schema in the [storage and warehousing](../02-storage-warehousing/) chapter.

Finally, not all data is tabular. JSON and nested types are first-class in modern warehouses, so you can handle semi-structured data without flattening everything up front.

## Consistency and reliability

Storage and modeling decisions sit on top of distributed systems, and the guarantees those systems give you shape what your pipeline can promise. This material comes from DDIA and is shared with the [System Design](../../systems/01-system-design/) chapter of *Systems Engineering*.

The **CAP** theorem says that under a network partition, a system must choose between consistency and availability. **PACELC** extends it: even when there is no partition, you trade latency against consistency. Within that space you pick a **consistency model** per use case: strong consistency, eventual consistency, or intermediate guarantees such as read-your-writes.

DDIA argues that every data system is judged on three properties: **reliability** (it keeps working correctly under faults), **scalability** (it copes with growth), and **maintainability** (people can keep operating and changing it).

For pipelines, one property matters more than any other: **idempotency**. A step is idempotent if re-running it does not change the result. That is what makes retries and backfills safe. If a job that loads one day of data fails halfway and is retried, an idempotent job produces the same output as a single clean run; a non-idempotent one produces duplicates. You will see idempotency made concrete with dynamic partition overwrite in the [batch and streaming](../03-batch-streaming/) chapter and with date-sliced orchestration in the [orchestration](../04-orchestration-modeling/) chapter.

## Batch versus streaming

The last framing question is *when* processing happens. **Batch** processing works on bounded data on a schedule. It offers high throughput and simple semantics, and it should be your default unless you need freshness. **Streaming** works on unbounded data continuously. It offers low latency but harder semantics: ordering, late data, and exactly-once processing all become your problem.

The honest rule is that most "real-time" requirements are actually "fresh enough" requirements. A dashboard refreshed every fifteen minutes satisfies most people who asked for real time. Don't pay the complexity tax of streaming unless the latency SLA truly demands it. The [batch and streaming](../03-batch-streaming/) chapter shows why batch is a special case of streaming and how to reason about both with one model.

## Where this leads

The ideas in this chapter recur everywhere. CAP, replication, and consistency are the shared distributed-systems core with [System Design](../../systems/01-system-design/). The reason columnar data compresses so well is an entropy argument, developed in [Information Theory](../../mathematics/17-information-theory/) in *Mathematics*. Feature pipelines that feed models are covered in [LLM Systems](../../ai-engineering/20-inference-engines/).

In industry, these are not abstractions. The lakehouse, the lifecycle, and Spark are Databricks's product. Amazon's Redshift, S3, and Glue are a reference data lake architecture. Netflix runs a massive batch and streaming data platform. Palantir's business is data integration and modeling at scale. Snowflake built its cloud warehouse on ELT and the separation of storage and compute.

Next, the [storage and warehousing](../02-storage-warehousing/) chapter takes the OLTP/OLAP split down to the storage engine and up to the lakehouse.

### Further reading

- Andreas Kretz, [*The Data Engineering Cookbook*](https://github.com/andkret/Cookbook) (free), the primary reference for this chapter.
- Joe Reis and Matt Housley, *Fundamentals of Data Engineering*, the source of the lifecycle-and-undercurrents framing.
- Martin Kleppmann, [*Designing Data-Intensive Applications*](https://dataintensive.net/), chapters 1 and 2.
- [Awesome Data Engineering](https://github.com/igorbarinov/awesome-data-engineering) (free), a curated index of tools.

## Key terms

- **Data engineering lifecycle**: the five stages data passes through (generation, ingestion, storage, transformation, serving), with security, governance, DataOps, architecture, orchestration, and software engineering as undercurrents.
- **OLTP**: transactional systems with many small row-oriented reads and writes at millisecond latency.
- **OLAP**: analytical systems with few large column-oriented scans and aggregates.
- **ELT**: extract, load raw data, then transform it in the warehouse; the modern default with decoupled storage and compute.
- **Columnar format**: a file layout such as Parquet or ORC that stores each column contiguously for pruning and compression.
- **CDC (change data capture)**: capturing a database's changes as a stream for low-impact ingestion from OLTP systems.
- **Schema-on-write / schema-on-read**: enforcing structure at load time versus interpreting raw data at query time.
- **Idempotency**: the property that re-running a step does not change its result, which makes pipelines recoverable.

## Summary

- Data engineering makes data produced for one purpose usable for another; it serves analytics and ML rather than being an end in itself.
- The lifecycle (generation, ingestion, storage, transformation, serving, plus undercurrents) is the map of the field.
- OLTP versus OLAP is the most important distinction: it dictates storage layout, file format, and system choice, and columnar storage wins for analytics.
- ELT replaced ETL because cheap storage and elastic compute make keeping raw, replayable data the cheaper option.
- Idempotency is what makes retries and backfills safe.
- Prefer batch unless the latency SLA truly demands streaming, and choose components for the access pattern and SLA, not novelty.

## Review questions

1. Why does a column-oriented layout make analytical queries faster and cheaper than a row-oriented one?
   <details><summary>Answer</summary>Analytical queries touch a few columns over many rows. Storing each column contiguously lets the engine read only the columns the query names, and adjacent similar values compress far better (dictionary and run-length encoding), so far fewer bytes are read.</details>

2. Give three reasons ELT displaced ETL as the default.
   <details><summary>Answer</summary>Storage and compute are decoupled and elastic; keeping raw data makes it replayable, so a transformation bug is fixed by rebuilding rather than re-extracting; transformation logic lives in version-controlled SQL (dbt); and schema-on-read gives flexibility to reinterpret raw data later.</details>

3. Place CDC on the lifecycle and explain why it is preferred over repeatedly querying a production database.
   <details><summary>Answer</summary>CDC is an ingestion technique. It reads the database's change log, so it captures every insert, update, and delete as a stream with low impact on the source, instead of running heavy queries against production tables.</details>

4. What does it mean for a pipeline step to be idempotent, and why does that matter for backfills?
   <details><summary>Answer</summary>Re-running the step does not change the result. A backfill or retry re-runs steps for dates that may already have partial or complete output; an idempotent step overwrites cleanly instead of duplicating data.</details>

5. A stakeholder asks for a "real-time" dashboard. What should you ask before reaching for a streaming system?
   <details><summary>Answer</summary>What freshness they actually need. Most real-time requests are really "fresh enough"; if a batch refresh every few minutes or hours meets the SLA, batch is simpler, cheaper, and has easier semantics than streaming.</details>
