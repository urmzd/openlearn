# Storage & Warehousing

## Overview

- **Primary reference**: [*Designing Data-Intensive Applications*](https://dataintensive.net/) Ch 3 (storage engines) + *The Data Warehouse Toolkit* by Kimball (dimensional modeling)
- **Supplementary**: [Apache Iceberg docs](https://iceberg.apache.org/docs/latest/) (free), [Delta Lake docs](https://docs.delta.io/) (free), [DuckDB docs](https://duckdb.org/docs/) (free), [BigQuery](https://cloud.google.com/bigquery/docs) / [Snowflake](https://docs.snowflake.com/) docs
- **Prerequisites**: [Foundations](../01-foundations/), SQL
- **Estimated time**: 3-4 weeks at 8-10 hrs/week

## Key Takeaways

- Storage engine choice is about access pattern: LSM-trees for write-heavy, B-trees for read/update-heavy, columnar for scan-heavy analytics
- The **data lake** (cheap raw storage) and **data warehouse** (structured, queryable) are converging into the **lakehouse** via open table formats
- **Partitioning and clustering** are the highest-leverage performance levers in any warehouse -- prune data before you scan it
- Table formats (Iceberg, Delta, Hudi) bring ACID transactions, time travel, and schema evolution to files on object storage

## How to Study

- Load the same dataset into Parquet-on-object-storage and a warehouse; compare query cost and speed
- Build a star schema for a toy domain (e.g. retail orders) by hand -- facts, dimensions, grain
- Read DDIA Ch 3 to understand *why* LSM-trees and B-trees behave differently under load

---

# Concepts & Techniques

## Core Insight

Where and how you lay data on disk determines everything downstream: a write-optimized engine makes analytics crawl, and a scan-optimized layout makes single-row updates painful. The art is matching the physical layout (row vs column, indexed vs partitioned, mutable vs append-only) to the dominant access pattern, then letting open table formats give you warehouse guarantees on cheap lake storage.

## 1. Storage Engines

**DDIA Ch 3**

**Key ideas**:
- **LSM-trees** (log-structured merge): buffer writes in memory, flush sorted segments, compact in background. Write-optimized; powers Cassandra, RocksDB, ScyllaDB
- **B-trees**: in-place updates, balanced tree on disk pages. Read/update-optimized; powers Postgres, MySQL, most OLTP
- **Tradeoff**: LSM has write amplification from compaction but high write throughput; B-trees have predictable reads but more write overhead
- **Column stores**: store each column contiguously; pair with compression and vectorized execution -- the basis of every analytical warehouse

## 2. Data Warehouse, Lake, and Lakehouse

| | Warehouse | Lake | Lakehouse |
|---|-----------|------|-----------|
| Stores | Structured, modeled | Raw, any format | Raw + table format |
| Schema | On write | On read | On read, enforced via table format |
| Engine | Snowflake, BigQuery, Redshift | S3/GCS + Spark | Databricks, Iceberg + engine |
| Strength | Fast SQL, governance | Cheap, flexible | Both -- ACID on cheap storage |

**Key shift**: separation of storage and compute. Data lives once in object storage; multiple elastic engines query it. No more copying data into a proprietary warehouse just to query it.

## 3. Open Table Formats

**Iceberg / Delta Lake / Hudi**

**Key ideas**:
- **The problem they solve**: a directory of Parquet files has no transactions, no schema evolution, no atomic updates, no consistent reads during writes
- **How**: a metadata layer (manifest files, transaction log) over the data files tracks which files make up a table version
- **What you get**: ACID transactions, **time travel** (query a past snapshot), schema and partition evolution, concurrent readers/writers, efficient upserts/deletes (MERGE)
- **Iceberg vs Delta vs Hudi**: Iceberg is engine-agnostic and the emerging open standard; Delta is Databricks-native; Hudi specializes in fast upserts and incremental processing

## 4. Partitioning & Clustering

**The biggest performance lever**

**Key ideas**:
- **Partitioning**: split a table by a column (usually date) so queries scan only relevant partitions -- *partition pruning*. Watch for skew and over-partitioning (too many tiny files)
- **Clustering / sorting**: physically order data within partitions on common filter columns so predicate pushdown skips row groups
- **Hidden partitioning** (Iceberg): partition transforms decouple the partition scheme from the query, avoiding the classic "forgot the partition filter" full scan
- **File sizing & compaction**: many small files kill performance; periodically compact into right-sized files (~100MB-1GB)

## 5. Dimensional Modeling

**Kimball -- The Data Warehouse Toolkit**

**Key ideas**:
- **Star schema**: a central *fact* table (measurements -- sales, clicks) surrounded by *dimension* tables (context -- customer, product, date)
- **Grain**: define the level of detail of a fact row *first* -- one row per order line, per session, etc. Everything else follows
- **Facts**: additive (sum across all dimensions), semi-additive (sum across some), non-additive (ratios)
- **Slowly Changing Dimensions (SCD)**: Type 1 (overwrite), Type 2 (new row + validity dates, preserves history), Type 3 (previous-value column)
- **Snowflake schema**: normalized dimensions -- saves space, costs joins; usually not worth it in columnar warehouses

## 6. Indexing & Acceleration for Analytics

**Key ideas**:
- **Zone maps / min-max stats**: per-block metadata lets the engine skip blocks that can't match a predicate
- **Bloom filters**: probabilistic membership test to skip files for point lookups (see [Probabilistic Structures](../../algorithms/15-probabilistic-structures/))
- **Materialized views**: precompute expensive aggregations; trade storage + refresh cost for query speed
- **Caching**: result cache and metadata cache in modern warehouses make repeated queries nearly free

## 7. Cost & Performance Tuning

**Key ideas**:
- **Pruning before scanning**: partition + cluster so the engine reads less -- bytes scanned is the cost unit in BigQuery/Snowflake
- **Avoid `SELECT *`**: columnar means you pay per column read
- **Right-size compute**: warehouse size / slot count vs query latency; autosuspend idle compute
- **Storage tiering**: hot (SSD) vs cold (object storage / Glacier) by access frequency

---

## Storage Decision Table

| Access pattern | Use | Why |
|----------------|-----|-----|
| High-volume writes, key lookups | LSM (Cassandra, RocksDB) | Write throughput |
| Transactional reads + updates | B-tree (Postgres, MySQL) | In-place updates, indexes |
| Analytical scans/aggregates | Columnar warehouse / Parquet | Column pruning, compression |
| Raw + ACID on cheap storage | Lakehouse (Iceberg/Delta) | Transactions on object storage |
| Local/embedded analytics | DuckDB | Zero-setup columnar OLAP |

## Connections to Other Tracks

| Concept | Connected Track | Application |
|---------|-----------------|-------------|
| Storage engines, replication, CAP | [System Design](../../systems/01-system-design/) | Shared distributed-storage core |
| Bloom filters | [Probabilistic Structures](../../algorithms/15-probabilistic-structures/) | File/block skipping |
| B-trees, sorting, hashing | [Trees](../../algorithms/05-trees/) | Index internals |
| Compression | [Information Theory](../../math/11-information-theory/) | Columnar encoding |
| Object storage as model store | [LLM Systems](../../ml/04-llm-systems/) | Loading weights from S3/GCS |

## Company Relevance

| Company | How This Appears | Difficulty |
|---------|-----------------|------------|
| Snowflake | Separation of storage/compute, micro-partitions | Expert |
| Databricks | Delta Lake, lakehouse, Iceberg | Expert |
| Google | BigQuery internals, Dremel, columnar | Expert |
| Amazon | Redshift, S3, Iceberg on Glue | Advanced |
| Netflix | Iceberg (created here), petabyte tables | Expert |
