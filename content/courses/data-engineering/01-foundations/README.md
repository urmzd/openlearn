# Foundations

## Overview

- **Primary reference**: [*The Data Engineering Cookbook*](https://github.com/andkret/Cookbook) by Andreas Kretz (free)
- **Supplementary**: *Fundamentals of Data Engineering* by Reis & Housley (recommended), [*Designing Data-Intensive Applications*](https://dataintensive.net/) Ch 1-2 (recommended), [Awesome Data Engineering](https://github.com/igorbarinov/awesome-data-engineering) (free)
- **Prerequisites**: SQL, [System Design](../../systems/01-system-design/) basics
- **Estimated time**: 2-3 weeks at 8-10 hrs/week

## Key Takeaways

- Data engineering exists to make data *usable* downstream -- it is plumbing in service of analytics and ML, not an end in itself
- The data engineering lifecycle (generation → ingestion → storage → transformation → serving) is the mental map for the entire field
- The single most important distinction is **OLTP vs OLAP** -- it dictates storage layout, file format, and system choice
- Choose components for the *access pattern and SLA*, not for novelty; most failures come from mismatched tools

## How to Study

- Build a tiny end-to-end pipeline (ingest an API → land in object storage → transform with SQL → query) before going deep on any one stage
- For every system you learn, classify it on the lifecycle and as OLTP or OLAP
- Read DDIA Ch 1-2 for the vocabulary (reliability, scalability, maintainability, data models) that the rest of the track assumes

---

# Concepts & Techniques

## Core Insight

A data engineer's job is to take data that was produced for one purpose (running an application, a sensor, a log) and make it reliably available for an entirely different purpose (analytics, ML, reporting). Every design decision flows from the gap between how data is *generated* and how it will be *consumed*: transactional vs analytical, real-time vs batch, raw vs modeled.

## 1. The Data Engineering Lifecycle

**Key ideas**:
- **Generation**: source systems -- application databases, APIs, event streams, IoT, files. You usually don't control these
- **Ingestion**: get data in -- batch (scheduled pulls) or streaming (continuous). Push vs pull, CDC (change data capture) from databases
- **Storage**: where it lands -- object storage, lakes, warehouses. The substrate the other stages sit on
- **Transformation**: cleaning, joining, aggregating, modeling -- raw data into useful shapes
- **Serving**: deliver to consumers -- BI dashboards, ML features, reverse ETL back into apps
- **Undercurrents** (cut across all stages): security, data management/governance, DataOps, data architecture, orchestration, software engineering

## 2. OLTP vs OLAP

**The foundational split**

| | OLTP | OLAP |
|---|------|------|
| Purpose | Run the application | Analyze the business |
| Access | Many small reads/writes | Few large scans/aggregates |
| Layout | Row-oriented | Column-oriented |
| Latency | Milliseconds | Seconds to minutes |
| Examples | Postgres, MySQL, DynamoDB | Snowflake, BigQuery, Redshift, DuckDB |

**Why columnar wins for analytics**: queries touch a few columns over many rows; storing columns together means reading only what you need, plus far better compression (similar values adjacent). This is the single most important idea in analytical storage.

## 3. ETL vs ELT

**Key ideas**:
- **ETL** (extract → transform → load): transform *before* loading, on a dedicated engine. The old default when storage and compute were expensive
- **ELT** (extract → load → transform): land raw data first, transform *in* the warehouse with SQL. The modern default -- cheap object storage + elastic compute make it cheaper to keep raw data and transform on demand
- **Why ELT won**: decoupled storage/compute, raw data is replayable, transformation logic lives in version-controlled SQL (dbt), schema-on-read flexibility

## 4. Storage Formats

**Key ideas**:
- **Row formats**: Avro, JSON, CSV -- good for write-heavy, record-at-a-time, schema evolution (Avro)
- **Columnar formats**: Parquet, ORC -- the analytics workhorses; column pruning, predicate pushdown, dictionary/run-length compression
- **Serialization**: Protobuf/Avro for compact typed wire formats; Arrow for zero-copy in-memory columnar interchange between tools
- **Compression**: Snappy (fast), Zstd (balanced), Gzip (small) -- trade CPU for size

## 5. Data Models & Schemas

**Key ideas**:
- **Schema-on-write** (warehouse): enforce structure at load time -- safe, rigid
- **Schema-on-read** (lake): store raw, interpret at query time -- flexible, risky
- **Normalization** (3NF) for OLTP to avoid update anomalies; **denormalization** (star schema) for OLAP to avoid join cost
- **Semi-structured**: JSON/nested types are first-class in modern warehouses -- handle them without flattening everything

## 6. CAP, Consistency & Reliability

**From [DDIA](https://dataintensive.net/) -- shared with [System Design](../../systems/01-system-design/)**

**Key ideas**:
- **CAP / PACELC**: under partition, choose consistency or availability; even without partitions, latency vs consistency
- **Consistency models**: strong, eventual, read-your-writes -- pick per use case
- **Reliability, scalability, maintainability**: the three properties DDIA argues every data system is judged on
- **Idempotency**: re-running a step must not change the result -- the property that makes retries and backfills safe

## 7. Batch vs Streaming (Framing)

**Key ideas**:
- **Batch**: bounded data, process on a schedule, high throughput, simple semantics. Default unless you need freshness
- **Streaming**: unbounded data, process continuously, low latency, harder semantics (ordering, late data, exactly-once)
- **The honest rule**: most "real-time" requirements are actually "fresh enough" -- don't pay the complexity tax of streaming unless the latency SLA truly demands it

---

## Concept Catalog

| Concept | What it is | Why it matters |
|---------|-----------|----------------|
| Lifecycle | Generation → serving + undercurrents | The map of the whole field |
| OLTP/OLAP | Transactional vs analytical | Dictates storage and system choice |
| ELT | Load raw, transform in warehouse | Modern default, decoupled compute |
| Columnar | Parquet/ORC | Analytics speed + compression |
| CDC | Capture DB changes as a stream | Low-impact ingestion from OLTP |
| Idempotency | Safe re-execution | Makes pipelines recoverable |

## Connections to Other Tracks

| Concept | Connected Track | Application |
|---------|-----------------|-------------|
| CAP, replication, consistency | [System Design](../../systems/01-system-design/) | Shared distributed-systems core |
| Compression, entropy | [Information Theory](../../mathematics-for-computing/11-information-theory/) | Why columnar compresses well |
| Feature pipelines | [LLM Systems](../../machine-learning/04-llm-systems/) | Feeding data to models |

## Company Relevance

| Company | How This Appears | Difficulty |
|---------|-----------------|------------|
| Databricks | Lakehouse, lifecycle, Spark -- this is their product | Expert |
| Amazon | Redshift, S3, Glue, data lake architecture | Advanced |
| Netflix | Massive batch + streaming data platform | Expert |
| Palantir | Data integration and modeling at scale | Advanced |
| Snowflake | Cloud warehouse, ELT, separation of storage/compute | Advanced |
