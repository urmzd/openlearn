# Interviewing at Databricks

Databricks is widely considered the most selective company in tech outside the largest platforms, with acceptance rates comparable to the top quantitative trading firms. Only about 20% of candidates advance past the coding stage. Two things make its loop distinctive. The first is that your code must actually run: the coding rounds use CoderPad with real execution, so syntax errors, missing imports, and off-by-one bugs surface immediately. The second is the domain. Databricks builds the Lakehouse, and both the coding problems and the system design round have a strong data-engineering flavor, from processing log streams to designing a distributed SQL engine.

This chapter follows the loop. After the process and compensation, it sets out the themes that run through every round, then works through four reported coding problems in detail: a log-stream aggregation, a sharded key-value store on a consistent-hash ring, the core operators of a small query engine, and tumbling-window stream aggregation. It then covers data-platform system design (the Lakehouse and medallion architecture, a distributed SQL engine, a feature store, a streaming pipeline), Databricks' own products, the behavioral round, and a preparation plan.

> **Learning objectives**
> - Describe Databricks' five-round loop and why runnable code and data-engineering fluency matter.
> - Implement stream aggregation, a consistent-hashing key-value store, hash join and group-by, and tumbling windows, and explain each one's limitations.
> - Design a Lakehouse with the Bronze, Silver, and Gold medallion layers and Delta Lake's guarantees.
> - Explain how a distributed SQL engine plans and executes a query, including predicate pushdown, partition pruning, adaptive execution, and shuffle.
> - Design a feature store with point-in-time correctness and a streaming pipeline with exactly-once delivery and watermarks.

## Before you begin

The coding rounds assume LeetCode medium-to-hard fluency and comfort writing runnable Python. The design round assumes you know the data stack. *Data Engineering* covers it in depth, especially [Storage and Warehousing](../../data-engineering/02-storage-warehousing/README.md) and [Batch and Streaming Processing](../../data-engineering/03-batch-streaming/README.md), and [The Apache Stack](../../infrastructure/05-apache-stack/README.md) in *Infrastructure* places Spark, Parquet, Iceberg, and Delta on one map. For the ML topics, read [ML infrastructure](../shared-concepts/ml-infrastructure.md) in this book.

## The interview process

The process takes four to eight weeks, averaging about 26 days, and has five rounds after the recruiter screen:

| Round | Format | Duration | Focus |
|-------|--------|----------|-------|
| Recruiter Screen | Phone | 30 min | Background, role fit |
| Technical Phone Screen | CoderPad (runnable code) | 60 min | LC medium-hard, systems thinking |
| Onsite 1 | Coding | 60 min | Algorithms, data structures |
| Onsite 2 | Coding | 60 min | Applied coding, data engineering |
| Onsite 3 | System Design | 60 min | Data platform architecture |
| Onsite 4 | Behavioral | 45 min | Values alignment, collaboration |

## Compensation

For a senior engineer in the US, base salary is $190-230K, RSUs add roughly $200-500K a year, and the bonus is 10-20%. Total compensation is about $400-700K at senior level and $700K to $1M or more at Staff.

## What Databricks is looking for

Five themes run through the loop. **Data platform expertise** comes first: Databricks builds the Lakehouse, so you need a deep understanding of Spark, Delta Lake, and data engineering. **Runnable code is required**: unlike interviews held in a plain shared document, Databricks uses CoderPad, where your code must compile and run. **Scale with correctness**: data systems must be both fast and correct, and ACID semantics matter. **Open-source DNA**: the company grew out of Apache Spark and values open-source contributions and community thinking. And the **AI and ML platform** is central, with Unity Catalog, MLflow, and Mosaic ML (an acquisition) all part of the product.

## The coding rounds

There are two onsite coding rounds plus the phone screen. The difficulty is LeetCode medium to hard, but the problems are more applied than pure algorithmic puzzles. They often have a data-engineering flavor (processing records, streaming data, ETL-like transformations), and they test production thinking: error handling, edge cases, and time and space analysis. Above all, the code must run. Four reported problems follow.

### Aggregating a log stream

The first is a typical ETL problem: process a stream of log entries and compute per-user statistics. Each entry has a user ID, an action, a timestamp, and a duration in milliseconds, and for each user you return the total number of actions, the average duration, and the set of distinct actions.

The design point is that the input is an `Iterator`, so you get one pass and should not materialize it. Accumulate running totals per user in a `defaultdict`, keeping the *sum* of durations rather than a running average (which would accumulate rounding error and need the count anyway), and compute averages only at the end.

```python
from collections import defaultdict
from typing import Iterator

def process_log_stream(logs: Iterator[dict]) -> dict:
    """Process a stream of log entries, computing per-user statistics.

    Each log: {"user_id": str, "action": str, "timestamp": int, "duration_ms": int}
    Returns: {user_id: {"total_actions": int, "avg_duration": float, "unique_actions": set}}
    """
    stats = defaultdict(lambda: {"total_actions": 0, "total_duration": 0, "unique_actions": set()})

    for log in logs:
        uid = log["user_id"]
        stats[uid]["total_actions"] += 1
        stats[uid]["total_duration"] += log["duration_ms"]
        stats[uid]["unique_actions"].add(log["action"])

    result = {}
    for uid, s in stats.items():
        result[uid] = {
            "total_actions": s["total_actions"],
            "avg_duration": s["total_duration"] / s["total_actions"],
            "unique_actions": s["unique_actions"],
        }
    return result
```

The pass is $O(n)$ in time and $O(u + a)$ in memory for $u$ users and $a$ distinct user-action pairs. Because a user appears in `stats` only after their first log, `total_actions` is never zero and the division is safe. In the interview, mention what production data would add: malformed records (a missing `duration_ms` raises `KeyError`), and whether a set is an acceptable output type or should be converted to a sorted list for serialization.

### A sharded key-value store

The second asks for a sharded key-value store using consistent hashing. Each shard is placed on a hash ring at several points; a key belongs to the first ring point clockwise from the key's own hash. Placing each shard at several points smooths out the distribution, and when a shard is added or removed, only the keys adjacent to its points move, rather than nearly all keys as with `hash(key) % num_shards`.

The implementation keeps the ring as a sorted list of hashes, with a map from each ring hash to its shard, and uses `bisect_right` to find the successor of a key's hash, wrapping around with `% len(self.ring)`. Each shard has its own lock, so operations on different shards proceed in parallel.

```python
import hashlib
import bisect
from threading import Lock

class DistributedKV:
    """Sharded key-value store with consistent hashing."""

    def __init__(self, num_shards: int, replicas_per_shard: int = 3):
        self.shards: list[dict] = [{} for _ in range(num_shards)]
        self.locks: list[Lock] = [Lock() for _ in range(num_shards)]
        self.num_shards = num_shards

        # Build hash ring
        self.ring = []
        self.ring_to_shard = {}
        for shard_id in range(num_shards):
            for r in range(replicas_per_shard):
                h = self._hash(f"shard-{shard_id}-replica-{r}")
                bisect.insort(self.ring, h)
                self.ring_to_shard[h] = shard_id

    def _hash(self, key: str) -> int:
        return int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**32)

    def _get_shard(self, key: str) -> int:
        h = self._hash(key)
        idx = bisect.bisect_right(self.ring, h) % len(self.ring)
        return self.ring_to_shard[self.ring[idx]]

    def put(self, key: str, value: str) -> None:
        shard_id = self._get_shard(key)
        with self.locks[shard_id]:
            self.shards[shard_id][key] = value

    def get(self, key: str) -> str | None:
        shard_id = self._get_shard(key)
        with self.locks[shard_id]:
            return self.shards[shard_id].get(key)

    def delete(self, key: str) -> bool:
        shard_id = self._get_shard(key)
        with self.locks[shard_id]:
            return self.shards[shard_id].pop(key, None) is not None

    def scan(self, prefix: str) -> list[tuple[str, str]]:
        """Scan across all shards for keys matching prefix."""
        results = []
        for shard_id in range(self.num_shards):
            with self.locks[shard_id]:
                for k, v in self.shards[shard_id].items():
                    if k.startswith(prefix):
                        results.append((k, v))
        return sorted(results)
```

Two points are worth making unprompted. First, the parameter named `replicas_per_shard` actually controls *virtual nodes*, the number of ring positions per shard; it does not replicate data. True replication would write each key to the next $k$ distinct shards clockwise on the ring. Second, `scan` takes each shard's lock in turn, so it does not see a consistent snapshot: a key moved or updated between two shards' scans can be missed or seen in an unexpected state. It is also $O(n)$ over all keys; a range-partitioned or sorted store would make prefix scans cheaper.

### Query engine components

The third problem asks for the core operators of a SQL-like engine over in-memory tables: `SELECT` with filtering, projection, ordering, and a limit; a join; and a group-by with aggregations.

`select` applies the clauses in a fixed order: `WHERE`, then projection, then `ORDER BY`, then `LIMIT`. `join` is a **hash join**: it builds a hash table on the right table's join key, then probes it with each left row, which costs $O(|L| + |R|)$ instead of the $O(|L| \cdot |R|)$ of a nested-loop join. A left join keeps unmatched left rows. `group_by` buckets rows by the grouping column and then computes each requested aggregate per bucket.

```python
class SimpleQueryExecutor:
    """Execute simple SQL-like operations on in-memory tables."""

    def __init__(self):
        self.tables: dict[str, list[dict]] = {}

    def create_table(self, name: str, rows: list[dict]):
        self.tables[name] = rows

    def select(self, table: str, columns: list[str] = None,
               where: callable = None, order_by: str = None,
               limit: int = None) -> list[dict]:
        rows = self.tables.get(table, [])

        # WHERE
        if where:
            rows = [r for r in rows if where(r)]

        # SELECT (projection)
        if columns:
            rows = [{c: r.get(c) for c in columns} for r in rows]

        # ORDER BY
        if order_by:
            desc = order_by.startswith("-")
            key = order_by.lstrip("-")
            rows = sorted(rows, key=lambda r: r.get(key, ""), reverse=desc)

        # LIMIT
        if limit:
            rows = rows[:limit]

        return rows

    def join(self, left_table: str, right_table: str,
             left_key: str, right_key: str, join_type: str = "inner") -> list[dict]:
        """Hash join implementation."""
        left = self.tables.get(left_table, [])
        right = self.tables.get(right_table, [])

        # Build hash table on right side
        right_index = {}
        for row in right:
            key = row.get(right_key)
            if key not in right_index:
                right_index[key] = []
            right_index[key].append(row)

        results = []
        for left_row in left:
            key = left_row.get(left_key)
            matches = right_index.get(key, [])

            if matches:
                for right_row in matches:
                    merged = {**left_row, **right_row}
                    results.append(merged)
            elif join_type == "left":
                results.append(left_row)

        return results

    def group_by(self, table: str, group_col: str,
                 agg: dict[str, str]) -> list[dict]:
        """Group by with aggregations. agg: {column: "sum"|"count"|"avg"|"max"|"min"}"""
        rows = self.tables.get(table, [])
        groups: dict = {}

        for row in rows:
            key = row.get(group_col)
            if key not in groups:
                groups[key] = []
            groups[key].append(row)

        results = []
        for key, group_rows in groups.items():
            result = {group_col: key}
            for col, func in agg.items():
                values = [r[col] for r in group_rows if col in r]
                if func == "sum":
                    result[f"{func}_{col}"] = sum(values)
                elif func == "count":
                    result[f"{func}_{col}"] = len(values)
                elif func == "avg":
                    result[f"{func}_{col}"] = sum(values) / len(values) if values else 0
                elif func == "max":
                    result[f"{func}_{col}"] = max(values) if values else None
                elif func == "min":
                    result[f"{func}_{col}"] = min(values) if values else None
            results.append(result)

        return results
```

An interviewer will probe the edge cases, and this code has several worth knowing. Because projection happens before `ORDER BY`, you cannot order by a column you did not select; real engines resolve ordering against the pre-projection row. `if limit:` treats `limit=0` as "no limit." Sorting with a default of `""` fails if some rows have numeric values and others lack the key, since Python cannot compare `""` with a number. In the join, a column present in both tables is silently overwritten by the right side's value, and an unmatched left-join row lacks the right table's columns entirely rather than carrying them as nulls. Choosing which side to build the hash table on also matters at scale: build on the smaller table.

### Tumbling-window aggregation

The fourth problem aggregates events into fixed-size, non-overlapping time windows: a **tumbling window**. Each event's window start is its timestamp floored to a multiple of the window size. Each window holds per-key running aggregates (sum, count, min, max), from which averages can be derived. A window is complete once the current time has passed its end, and old windows are dropped after a retention period.

```python
from collections import deque
import time

class TumblingWindowAggregator:
    """Aggregate events in fixed-size time windows."""

    def __init__(self, window_size_sec: int):
        self.window_size = window_size_sec
        self.windows: dict[int, dict] = {}  # window_start -> aggregation state

    def _window_start(self, timestamp: float) -> int:
        return int(timestamp // self.window_size) * self.window_size

    def add_event(self, timestamp: float, key: str, value: float):
        ws = self._window_start(timestamp)
        if ws not in self.windows:
            self.windows[ws] = {}
        if key not in self.windows[ws]:
            self.windows[ws][key] = {"sum": 0, "count": 0, "min": float("inf"), "max": float("-inf")}

        agg = self.windows[ws][key]
        agg["sum"] += value
        agg["count"] += 1
        agg["min"] = min(agg["min"], value)
        agg["max"] = max(agg["max"], value)

    def get_window(self, timestamp: float) -> dict:
        ws = self._window_start(timestamp)
        return self.windows.get(ws, {})

    def get_completed_windows(self, current_time: float) -> list[tuple[int, dict]]:
        """Return all windows that have closed (timestamp past window end)."""
        completed = []
        for ws, data in sorted(self.windows.items()):
            if ws + self.window_size <= current_time:
                completed.append((ws, data))
        return completed

    def cleanup(self, current_time: float, retention_windows: int = 10):
        """Remove windows older than retention period."""
        cutoff = self._window_start(current_time) - (retention_windows * self.window_size)
        self.windows = {ws: data for ws, data in self.windows.items() if ws >= cutoff}
```

Adding an event is $O(1)$. `get_completed_windows` sorts all windows on every call and returns completed windows repeatedly rather than emitting each once, so a real pipeline would track which windows it has already emitted. More importantly, nothing stops a late event from being added to a window that has already been reported. Production stream processors handle this with a **watermark**, an estimate of how far event time has progressed, after which a window is finalized and later events are dropped or routed to a side output. (The `deque` and `time` imports are unused.) [Batch and Streaming Processing](../../data-engineering/03-batch-streaming/README.md) covers windows and watermarks in depth.

## The system design round

Databricks system design is specific to data platforms. Generic web prompts such as "design Twitter" are less relevant. You need to understand **Apache Spark internals** (jobs, stages, tasks, shuffle, partitioning); **Delta Lake** (ACID transactions on data lakes, time travel, Z-ordering); **Lakehouse architecture** (unified analytics over structured and unstructured data); **Parquet and ORC** (columnar storage, predicate pushdown, column pruning); and **Kafka** (streaming ingestion and exactly-once semantics). Four prompts are commonly reported.

### A Lakehouse platform

The Lakehouse organizes data into three refinement layers, the **medallion architecture**, and serves every consumer from the same tables:

```text
[Data Sources] --> [Ingestion Layer]
                        |
                  [Bronze Layer] (raw data, append-only)
                        |
                  [Silver Layer] (cleaned, validated, deduplicated)
                        |
                  [Gold Layer] (aggregated, business-ready)
                        |
              +---------+---------+
              |         |         |
         [BI/SQL]  [ML Training]  [Streaming]
```

The **Bronze** layer is raw ingestion: append-only, schema-on-read, with no transformations, so you can always reprocess from the source. The **Silver** layer is cleaned, typed, and deduplicated, with business logic applied. The **Gold** layer holds aggregated tables optimized for specific use cases.

**Delta Lake** supplies the guarantees that make this work on object storage. It provides ACID transactions through optimistic concurrency with conflict resolution; time travel, so you can query any past version of a table; schema enforcement and evolution; Z-ordering, which clusters data across several columns at once to speed up multi-dimensional queries; and `VACUUM`, which garbage-collects old files that no retained version still references.

### A distributed SQL query engine

A distributed SQL engine turns a query into a plan and the plan into tasks on workers:

```text
[SQL Query] --> [Parser] --> [Logical Plan] --> [Optimizer]
                                                    |
                                             [Physical Plan]
                                                    |
                                        [Task Scheduler (DAG)]
                                                    |
                                  +--------+--------+--------+
                                  |        |        |        |
                               [Worker] [Worker] [Worker] [Worker]
                                  |
                              [Shuffle]
                                  |
                              [Aggregation]
                                  |
                              [Result]
```

The concepts to explain are the **Catalyst optimizer**, Spark's combination of rule-based and cost-based optimization; **predicate pushdown**, which filters data at the storage level instead of in memory; **partition pruning**, which skips partitions that cannot match the query's predicates; **Adaptive Query Execution (AQE)**, which re-optimizes the plan during execution using runtime statistics; and **shuffle optimization**, which minimizes data movement across the network, since the shuffle is usually the most expensive stage.

### A feature store

For an ML feature store, cover the **online store** for low-latency serving (Redis or DynamoDB) and the **offline store** for high-throughput batch access (Delta Lake or S3); **point-in-time correctness**, which joins each training example only with feature values that existed at its timestamp and so prevents data leakage; **feature sharing** through a catalog that teams can search; and **lineage**, tracking the relationships from feature to model to prediction.

### A streaming data pipeline

For a streaming pipeline, cover the **source** (Kafka, Event Hubs, or Kinesis); **processing** with Spark Structured Streaming and its exactly-once guarantees; the **sink**, Delta Lake with ACID writes; **checkpointing** for reliable recovery from failures; **watermarking** to handle late-arriving events; and **monitoring** of throughput, latency, backlog size, and error rate.

### Databricks products to know

You will be expected to know Databricks' own platform:

| Concept | What to Know |
|---------|-------------|
| Unity Catalog | Centralized governance for data and AI assets |
| MLflow | Experiment tracking, model registry, deployment |
| Delta Sharing | Open protocol for secure data sharing |
| Photon | C++ vectorized query engine (faster than Spark JVM) |
| Serverless Compute | Auto-scaling, pay-per-query compute |
| Mosaic ML | LLM training infrastructure (acquired by Databricks) |

## The behavioral round

The behavioral round assesses five values:

| Value | What They Assess |
|-------|-----------------|
| Customer-obsessed | Start from customer pain, work backwards |
| Data-driven | Use data to make decisions, not opinions |
| Open ecosystem | Contribute to open-source, avoid vendor lock-in |
| High standards | Quality and correctness matter |
| Ownership | End-to-end accountability |

Expect questions such as "Why Databricks over other data companies?"; "Tell me about a time you built something from scratch to solve a data problem"; "Describe a time you had to make a tradeoff between speed and correctness"; "How do you approach debugging a data pipeline that's producing incorrect results?"; and "Tell me about a time you contributed to or used an open-source project."

## AI and ML at Databricks

For ML roles, know the ML infrastructure. **MLflow** covers the end-to-end ML lifecycle: tracking, registry, serving, and evaluation. **Mosaic ML** provides large-scale model training, including LLMs, on Databricks. **Model Serving** offers serverless model endpoints with autoscaling. The **Feature Store** is built on Delta Lake with point-in-time correctness. And **Vector Search** provides embedding-based retrieval for RAG applications.

The generative AI focus spans training custom LLMs on enterprise data, RAG with Unity Catalog governing the sources, AI-powered SQL generation from natural language, and fine-tuning and deploying foundation models.

## Preparing well

Learn Spark internals: how a job executes across the driver and executors, and how it breaks into stages, tasks, and shuffles; the Spark documentation is the place to start. Understand Delta Lake, including ACID on data lakes, time travel, `VACUUM`, and Z-ordering, and read the Delta Lake paper. Practice with real execution, on CoderPad or LeetCode, because syntax errors fail you. Practice ETL-style coding problems, not just pure algorithms. Read the Databricks Lakehouse whitepaper and understand the medallion architecture. Be familiar with MLflow, Delta Lake, and Apache Spark at a conceptual level. And study the Databricks engineering blog, which has excellent material on internals.

## Key terms

- **Lakehouse**: an architecture that provides warehouse guarantees (transactions, schemas, governance) over open files on cheap object storage.
- **Medallion architecture**: organizing lake data into Bronze (raw), Silver (cleaned), and Gold (aggregated) layers.
- **Consistent hashing**: placing shards and keys on a hash ring so that adding or removing a shard moves only nearby keys.
- **Virtual node**: one of several ring positions assigned to a single shard to balance load.
- **Hash join**: joining two tables by building a hash table on one side's key and probing it with the other side.
- **Tumbling window**: a fixed-size, non-overlapping time window for stream aggregation.
- **Watermark**: an estimate of event-time progress after which a window is finalized and late events are handled separately.
- **Point-in-time correctness**: joining training examples only with feature values as of each example's timestamp, preventing leakage.

## Summary

- Databricks admits about 20% of candidates past coding; code must run, and problems are applied and data-flavored.
- One-pass stream aggregation keeps sums and counts, not running averages; consistent hashing with virtual nodes balances shards and limits key movement.
- Query operators are filter, project, sort, and limit, plus hash join and group-by; know their edge cases, from projection order to join column collisions.
- Tumbling windows floor timestamps to window starts; production systems add watermarks for late data.
- Design answers are data-platform specific: medallion layers on Delta Lake, Catalyst with pushdown and AQE, feature stores with point-in-time correctness, and exactly-once streaming with checkpoints.

## Review questions

1. Why does consistent hashing move fewer keys than `hash(key) % num_shards` when a shard is added?
   <details><summary>Answer</summary>With modulo hashing, changing the shard count changes almost every key's assignment. On a ring, a new shard takes over only the arcs just before its points, so only those keys move.</details>
2. What does `replicas_per_shard` do in the `DistributedKV` class, and what would real replication require?
   <details><summary>Answer</summary>It sets the number of virtual nodes, the ring positions per shard, which balances load; it does not copy data. Real replication would write each key to the next $k$ distinct shards clockwise and read from a quorum of them.</details>
3. Why build the hash table on the smaller side of a hash join?
   <details><summary>Answer</summary>The build side must fit in memory and is hashed once; the probe side is streamed. Building on the smaller table minimizes memory and build time while keeping the join $O(|L| + |R|)$.</details>
4. In `SimpleQueryExecutor.select`, a caller selects only `name` and orders by `age`. What happens, and how do real engines avoid it?
   <details><summary>Answer</summary>Projection runs before sorting, so <code>age</code> is gone and every row sorts on the default value. Real engines resolve <code>ORDER BY</code> against the full row before the final projection.</details>
5. An event with timestamp 59 arrives after the window [0, 60) was reported as complete. What happens in `TumblingWindowAggregator`, and what would a production system do?
   <details><summary>Answer</summary>It is silently added to the already-reported window, changing a result that consumers already saw. A production system uses a watermark: once event time passes the window end plus an allowed lateness, the window is finalized and later events are dropped or sent to a side output.</details>
6. What is point-in-time correctness in a feature store, and what goes wrong without it?
   <details><summary>Answer</summary>Each training example is joined only with feature values as they were at its timestamp. Without it, training sees feature values from the future relative to the label, which leaks information and makes offline metrics look better than production performance.</details>

## Further reading

- [Databricks software engineer interview guide](https://www.interviewquery.com/interview-guides/databricks-software-engineer) (InterviewQuery)
- [Databricks system design interview](https://www.systemdesignhandbook.com/guides/databricks-system-design-interview/) (System Design Handbook)
- [Databricks interview prep](https://www.databricks.com/company/careers/interview-prep) (official)
- [Get a job at Databricks](https://www.tryexponent.com/blog/databricks-interview-process) (Exponent)
- [Ace the Databricks SWE interview](https://prepfully.com/interview-guides/databricks-software-engineer) (Prepfully)
- [Glassdoor: Databricks SWE interview questions](https://www.glassdoor.com/Interview/Databricks-Software-Engineer-Interview-Questions-EI_IE954734.0,10_KO11,28.htm)
- [Databricks Blog](https://www.databricks.com/blog)
- Community reports on r/dataengineering and Blind.
