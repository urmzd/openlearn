# Storage and Warehousing

Where and how you lay data on disk determines everything downstream. A write-optimized engine makes analytics crawl, and a scan-optimized layout makes single-row updates painful. The craft of storage is matching the physical layout (row or column, indexed or partitioned, mutable or append-only) to the dominant access pattern, and then using open table formats to get warehouse guarantees on cheap lake storage.

The [Foundations](../01-foundations/) chapter drew the line between OLTP and OLAP. This chapter follows that line down into the storage engines that implement each side, then back up through the architectures built on them: the data warehouse, the data lake, and the lakehouse that is collapsing the two. You will learn the open table formats that make the lakehouse possible, the partitioning and clustering techniques that are the highest-leverage performance levers in any warehouse, and Kimball's dimensional modeling, which is still how most analytical tables are designed. The chapter ends with the indexing, caching, and cost techniques that decide your monthly bill.

> **Learning objectives**
> - Explain why LSM-trees suit write-heavy workloads, B-trees suit read-and-update workloads, and column stores suit scans.
> - Compare warehouse, lake, and lakehouse architectures and explain what open table formats (Iceberg, Delta Lake, Hudi) add to files on object storage.
> - Use partitioning, clustering, and file compaction to prune data before it is scanned.
> - Design a star schema by declaring the grain first, classify facts by additivity, and choose a slowly-changing-dimension type.
> - Reduce query cost by reasoning about bytes scanned, compute sizing, and storage tiers.

## Before you begin

You need SQL and the [Foundations](../01-foundations/) chapter. The primary references are chapter 3 of [*Designing Data-Intensive Applications*](https://dataintensive.net/) (storage engines), which explains *why* LSM-trees and B-trees behave differently under load, and Ralph Kimball's *The Data Warehouse Toolkit* (dimensional modeling). Two exercises make this chapter stick. First, load the same dataset into Parquet on object storage and into a warehouse, and compare query cost and speed. Second, build a star schema for a toy domain such as retail orders by hand, deciding the facts, dimensions, and grain yourself.

## Storage engines

Start at the bottom, with the data structure that sits on disk.

A **log-structured merge tree (LSM-tree)** buffers writes in memory, flushes them to disk as sorted, immutable segments, and merges (compacts) those segments in the background. Every write is a sequential append, which makes LSM-trees write-optimized. They power Cassandra, RocksDB, and ScyllaDB.

A **B-tree** keeps a balanced tree of fixed-size disk pages and updates them in place. Reads follow one path from root to leaf, so read latency is predictable. B-trees are read- and update-optimized and power Postgres, MySQL, and most OLTP databases.

The tradeoff between them is about where the cost of a write lands. An LSM-tree accepts writes cheaply and pays later through *write amplification*: compaction rewrites the same data several times. In exchange it gets high write throughput. A B-tree pays at write time, updating pages and the write-ahead log, which costs more per write but keeps reads predictable.

Neither of these is what an analytical warehouse uses. A **column store** stores each column contiguously and pairs that layout with compression and *vectorized execution*, which processes a batch of values from one column per CPU instruction loop instead of one row at a time. That combination is the basis of every analytical warehouse.

The following table collects these choices by access pattern, along with the two higher-level options you will meet next.

| Access pattern | Use | Why |
|----------------|-----|-----|
| High-volume writes, key lookups | LSM (Cassandra, RocksDB) | Write throughput |
| Transactional reads + updates | B-tree (Postgres, MySQL) | In-place updates, indexes |
| Analytical scans/aggregates | Columnar warehouse / Parquet | Column pruning, compression |
| Raw + ACID on cheap storage | Lakehouse (Iceberg/Delta) | Transactions on object storage |
| Local/embedded analytics | DuckDB | Zero-setup columnar OLAP |

## Warehouse, lake, and lakehouse

Engines are building blocks. Architectures decide how an organization assembles them, and three have dominated in turn. The table below compares them.

| | Warehouse | Lake | Lakehouse |
|---|-----------|------|-----------|
| Stores | Structured, modeled | Raw, any format | Raw + table format |
| Schema | On write | On read | On read, enforced via table format |
| Engine | Snowflake, BigQuery, Redshift | S3/GCS + Spark | Databricks, Iceberg + engine |
| Strength | Fast SQL, governance | Cheap, flexible | Both: ACID on cheap storage |

A **data warehouse** stores structured, modeled data with schema enforced on write, and gives you fast SQL and strong governance. A **data lake** stores raw data in any format on cheap object storage and interprets it on read; it is cheap and flexible but offers few guarantees. The **lakehouse** keeps raw files on object storage but adds a table format on top that enforces schema and provides transactions, so you get both.

The key shift underneath all of this is the separation of storage and compute. Data lives once, in object storage, and multiple elastic engines query it. You no longer copy data into a proprietary warehouse just to be able to query it.

## Open table formats

The lakehouse depends on one piece of technology, so it deserves a closer look. Consider what a plain directory of Parquet files lacks. It has no transactions, so a reader can see a half-finished write. It has no schema evolution, no atomic updates, and no consistent reads while a writer is working.

Open table formats (Apache Iceberg, Delta Lake, and Apache Hudi) solve this by adding a metadata layer over the data files: manifest files or a transaction log that records exactly which files make up each version of the table. A write produces new data files and then atomically commits a new version of the metadata. Readers always see a complete version.

From that one mechanism you get a long list of warehouse features on lake storage: ACID transactions; **time travel**, which lets you query a past snapshot; schema and partition evolution; safe concurrent readers and writers; and efficient upserts and deletes through `MERGE`.

The three formats differ in emphasis. Iceberg is engine-agnostic and is the emerging open standard. Delta Lake is Databricks-native. Hudi specializes in fast upserts and incremental processing.

## Partitioning and clustering

With a table format in place, the next question is how to make queries against it fast. The biggest performance lever is not a faster engine; it is reading less data, which means pruning data before you scan it.

**Partitioning** splits a table by a column, usually a date, so that a query filtering on that column scans only the relevant partitions. This is *partition pruning*. Watch for two failure modes: skew, where a few partitions hold most of the data, and over-partitioning, where too fine a scheme produces many tiny files.

**Clustering** (or sorting) physically orders data *within* partitions on commonly filtered columns. Because each Parquet row group records min and max statistics, sorted data lets predicate pushdown skip whole row groups that cannot match.

Iceberg adds **hidden partitioning**. Instead of exposing a separate partition column that users must remember to filter on, the table declares a transform (for example, "day of `event_ts`") and the engine applies it automatically to filters on the source column. This decouples the partition scheme from the query and avoids the classic full scan caused by someone forgetting the partition filter.

Finally, file sizing matters. Many small files kill performance, because each file costs a metadata lookup and an open. Compact them periodically into right-sized files of roughly 100 MB to 1 GB.

## Dimensional modeling

Partitioning decides how a table is stored. Dimensional modeling, from Kimball's *The Data Warehouse Toolkit*, decides what tables to build in the first place.

The central pattern is the **star schema**: a *fact* table of measurements (sales, clicks) in the middle, surrounded by *dimension* tables that give context (customer, product, date). Analysts filter and group by dimension attributes and aggregate fact measures.

The first and most important design decision is the **grain**: the level of detail one fact row represents, such as one row per order line or one row per session. Declare the grain first. Everything else follows from it, and most modeling bugs are grain confusion.

Facts are classified by how they can be aggregated. *Additive* facts, such as sales amount, can be summed across every dimension. *Semi-additive* facts, such as an account balance, can be summed across some dimensions but not across time. *Non-additive* facts, such as ratios, cannot be summed at all; you recompute them from their additive parts.

Dimensions change over time, and how you record that change matters. A **slowly changing dimension (SCD)** of Type 1 simply overwrites the old value, losing history. Type 2 inserts a new row with validity dates, preserving full history. Type 3 adds a column holding the previous value, keeping one step of history.

You may also see the **snowflake schema**, which normalizes dimensions into sub-tables. It saves space but costs joins, and in columnar warehouses, where storage is cheap and compression strong, it is usually not worth it.

## Indexing and acceleration for analytics

Columnar warehouses do not use B-tree indexes the way OLTP databases do, but they have their own tools for avoiding work.

**Zone maps** (min-max statistics) store per-block metadata so the engine can skip any block whose range cannot match a predicate. **Bloom filters** provide a probabilistic membership test, which lets an engine skip files during point lookups; you can read how they work in [Probabilistic Structures](../../algorithms/15-probabilistic-structures/) in *Algorithms*. **Materialized views** precompute expensive aggregations, trading storage and refresh cost for query speed. And **caching**, both of query results and of metadata, makes repeated queries in modern warehouses nearly free.

## Cost and performance tuning

In a cloud warehouse, performance and cost are the same conversation. In BigQuery and Snowflake, bytes scanned is the cost unit, so everything in the partitioning section is also a cost technique: partition and cluster so the engine reads less. For the same reason, avoid `SELECT *`. In a columnar system you pay per column read, and selecting columns you do not need is paying for nothing.

On the compute side, right-size the warehouse: the warehouse size or slot count trades directly against query latency, and autosuspend stops you paying for idle compute. On the storage side, tier data by access frequency, keeping hot data on SSD and moving cold data to object storage or archival tiers such as Glacier.

## Where this leads

Storage engines, replication, and CAP are the shared distributed-storage core with [System Design](../../systems/01-system-design/) in *Systems*. Bloom filters for file and block skipping come from [Probabilistic Structures](../../algorithms/15-probabilistic-structures/), and the B-trees, sorting, and hashing behind index internals from [Trees](../../algorithms/05-trees/), both in *Algorithms*. Columnar encoding is an application of [Information Theory](../../mathematics-for-computing/11-information-theory/) in *Mathematics for Computing*. And object storage reappears as a model store in [LLM Systems](../../machine-learning/04-llm-systems/), where weights are loaded from S3 or GCS.

These ideas are the core products of several companies. Snowflake is built on the separation of storage and compute and on micro-partitions. Databricks builds on Delta Lake, the lakehouse, and Iceberg. Google's BigQuery descends from Dremel and its columnar design. Amazon offers Redshift, S3, and Iceberg on Glue. Netflix created Iceberg and runs petabyte-scale tables on it.

The [batch and streaming](../03-batch-streaming/) chapter turns from storing data to computing over it, and the [orchestration and modeling](../04-orchestration-modeling/) chapter builds dimensional marts on top of these tables.

### Further reading

- Martin Kleppmann, [*Designing Data-Intensive Applications*](https://dataintensive.net/), chapter 3.
- Ralph Kimball and Margy Ross, *The Data Warehouse Toolkit*.
- [Apache Iceberg docs](https://iceberg.apache.org/docs/latest/) and [Delta Lake docs](https://docs.delta.io/) (free).
- [DuckDB docs](https://duckdb.org/docs/) (free).
- [BigQuery](https://cloud.google.com/bigquery/docs) and [Snowflake](https://docs.snowflake.com/) documentation.

## Key terms

- **LSM-tree**: a write-optimized engine that buffers writes in memory, flushes sorted segments, and compacts them in the background.
- **B-tree**: a read- and update-optimized engine that keeps a balanced tree of disk pages updated in place.
- **Write amplification**: the extra writes an LSM-tree performs when compaction rewrites data.
- **Lakehouse**: raw files on object storage plus an open table format, giving ACID guarantees on cheap storage.
- **Open table format**: a metadata layer (Iceberg, Delta Lake, Hudi) that tracks which files make up each table version.
- **Time travel**: querying a past snapshot of a table.
- **Partition pruning**: skipping partitions that a query's filter rules out.
- **Hidden partitioning**: Iceberg's partition transforms applied automatically to filters on the source column.
- **Grain**: the level of detail a single fact row represents.
- **Slowly changing dimension (SCD)**: a policy for recording changes to dimension attributes (Type 1 overwrite, Type 2 new row with validity dates, Type 3 previous-value column).
- **Zone map**: per-block min-max statistics used to skip blocks that cannot match a predicate.

## Summary

- Choose the storage engine by access pattern: LSM-trees for write-heavy, B-trees for read and update, column stores for scans.
- The warehouse and the lake are converging into the lakehouse, enabled by the separation of storage and compute.
- Open table formats add ACID transactions, time travel, and schema evolution to files on object storage through a metadata layer.
- Partitioning, clustering, and right-sized files are the highest-leverage performance levers: prune before you scan.
- Dimensional modeling starts with the grain; star schemas usually beat snowflake schemas in columnar warehouses.
- Bytes scanned is the cost unit, so pruning, avoiding `SELECT *`, right-sizing compute, and tiering storage all cut cost.

## Review questions

1. Why does an LSM-tree sustain higher write throughput than a B-tree, and what does it pay for it?
   <details><summary>Answer</summary>It turns every write into an in-memory insert followed by sequential flushes of sorted segments, avoiding in-place page updates. It pays through write amplification, since background compaction rewrites data several times, and reads may have to consult several segments.</details>

2. What problems does a plain directory of Parquet files have that an open table format solves, and how?
   <details><summary>Answer</summary>It has no transactions, no schema evolution, no atomic updates, and no consistent reads during writes. A table format adds a metadata layer (manifests or a transaction log) that records which files make up each table version; writers commit new versions atomically, so readers always see a complete snapshot, and past snapshots enable time travel.</details>

3. What is hidden partitioning, and which classic mistake does it prevent?
   <details><summary>Answer</summary>Iceberg declares a partition transform on a source column (for example, day of a timestamp) and applies it automatically when queries filter on that column. Users no longer need to filter on a separate partition column, so forgetting the partition filter no longer causes a full scan.</details>

4. Why should you declare the grain of a fact table before anything else?
   <details><summary>Answer</summary>The grain fixes what one row means (one order line, one session). Which dimensions attach, which facts are valid and how they aggregate all follow from it, and mixing grains in one table produces double counting.</details>

5. An account balance is summed across all days in a month and the result is nonsense. What kind of fact is it, and how should it be aggregated?
   <details><summary>Answer</summary>It is semi-additive: it can be summed across some dimensions (such as accounts) but not across time. Across time, use the balance at a point (for example, end of month) or an average instead of a sum.</details>

6. Name three ways to reduce cost in a warehouse that bills by bytes scanned.
   <details><summary>Answer</summary>Partition and cluster tables so queries prune data; select only the columns you need instead of `SELECT *`; and use materialized views or caching for repeated expensive queries. Right-sizing and autosuspending compute and tiering cold storage also reduce cost.</details>
