# Databricks Software Engineer Interview Guide

Comprehensive preparation for Databricks SWE roles. Databricks is widely considered the **most selective non-FAANG company** in tech, with acceptance rates comparable to top quant firms.

## Interview Process Overview

Timeline: **4-8 weeks** (average 26 days), **5 rounds**

| Round | Format | Duration | Focus |
|-------|--------|----------|-------|
| Recruiter Screen | Phone | 30 min | Background, role fit |
| Technical Phone Screen | CoderPad (runnable code) | 60 min | LC medium-hard, systems thinking |
| Onsite 1 | Coding | 60 min | Algorithms, data structures |
| Onsite 2 | Coding | 60 min | Applied coding, data engineering |
| Onsite 3 | System Design | 60 min | Data platform architecture |
| Onsite 4 | Behavioral | 45 min | Values alignment, collaboration |

**Only ~20% of candidates advance past the coding stage.** The bar is extremely high.

## Compensation (Senior, US)

- **Base**: $190-230K
- **Equity**: RSUs ~$200-500K/yr (Databricks went public via direct listing)
- **Bonus**: 10-20%
- **Total Comp**: ~$400-700K (Senior), ~$700K-1M+ (Staff)

## Key Themes

1. **Data platform expertise** -- Databricks builds the Lakehouse. You need to understand Spark, Delta Lake, and data engineering deeply.
2. **Runnable code required** -- Unlike Google (Google Docs), Databricks uses CoderPad where your code must compile and run.
3. **Scale with correctness** -- Data systems must be both fast AND correct. ACID semantics matter.
4. **Open-source DNA** -- Databricks originated from Apache Spark. They value open-source contributions and community thinking.
5. **AI/ML platform** -- Unity Catalog, MLflow, Mosaic ML (acquired) -- ML infrastructure is central.

## Coding Rounds (2 rounds + phone screen)

### What Makes Databricks Coding Different

- Code must **run**. Syntax errors, missing imports, and off-by-one bugs are caught immediately.
- Problems often have a **data engineering flavor**: processing records, streaming data, ETL-like transformations.
- They test **production thinking**: error handling, edge cases, time/space analysis.
- LC medium-to-hard difficulty, but more applied than pure algorithmic puzzles.

### Reported Problem Types

#### 1. Data Processing / ETL Problems

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

#### 2. Distributed Key-Value Store

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

#### 3. SQL Query Engine Components

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

#### 4. Streaming Window Aggregation

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

## System Design Round

### What Makes Databricks SD Different

Databricks system design is **data-platform-specific**. Generic web system design (design Twitter, design Uber) is less relevant. You need to understand:

- **Apache Spark internals**: Jobs, stages, tasks, shuffle, partitioning
- **Delta Lake**: ACID transactions on data lakes, time travel, Z-ordering
- **Lakehouse architecture**: Unified analytics on structured + unstructured data
- **Parquet/ORC**: Columnar storage formats, predicate pushdown, column pruning
- **Kafka**: Streaming data ingestion, exactly-once semantics

### Common Topics

#### Design a Lakehouse Platform

```
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

**Medallion architecture** (Bronze/Silver/Gold):
- **Bronze**: Raw ingestion, schema-on-read, no transformations
- **Silver**: Cleaned, typed, deduplicated, business logic applied
- **Gold**: Aggregated tables optimized for specific use cases

**Delta Lake features**:
- ACID transactions (optimistic concurrency, conflict resolution)
- Time travel (query data as of any past version)
- Schema enforcement and evolution
- Z-ordering for multi-dimensional query optimization
- VACUUM for garbage collection of old files

#### Design a Distributed SQL Query Engine

```
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

Key concepts:
- **Catalyst optimizer** (Spark): Rule-based and cost-based optimization
- **Predicate pushdown**: Filter data at storage level, not in memory
- **Partition pruning**: Skip partitions that don't match query predicates
- **Adaptive Query Execution (AQE)**: Re-optimize plan during execution based on runtime statistics
- **Shuffle optimization**: Minimize data movement across the network

#### Design a Feature Store for ML

- **Online store**: Low-latency serving (Redis/DynamoDB)
- **Offline store**: High-throughput batch access (Delta Lake/S3)
- **Point-in-time correctness**: Prevent data leakage in training
- **Feature sharing**: Catalog for discovery across teams
- **Lineage**: Track feature -> model -> prediction relationships

#### Design a Streaming Data Pipeline

- **Source**: Kafka / Event Hubs / Kinesis
- **Processing**: Structured Streaming (Spark) with exactly-once guarantees
- **Sink**: Delta Lake with ACID writes
- **Checkpointing**: Reliable recovery from failures
- **Watermarking**: Handle late-arriving events
- **Monitoring**: Throughput, latency, backlog size, error rate

### Databricks-Specific Concepts

| Concept | What to Know |
|---------|-------------|
| Unity Catalog | Centralized governance for data and AI assets |
| MLflow | Experiment tracking, model registry, deployment |
| Delta Sharing | Open protocol for secure data sharing |
| Photon | C++ vectorized query engine (faster than Spark JVM) |
| Serverless Compute | Auto-scaling, pay-per-query compute |
| Mosaic ML | LLM training infrastructure (acquired by Databricks) |

## Behavioral Round

### Databricks Values

| Value | What They Assess |
|-------|-----------------|
| Customer-obsessed | Start from customer pain, work backwards |
| Data-driven | Use data to make decisions, not opinions |
| Open ecosystem | Contribute to open-source, avoid vendor lock-in |
| High standards | Quality and correctness matter |
| Ownership | End-to-end accountability |

### Common Questions

- "Why Databricks over other data companies?"
- "Tell me about a time you built something from scratch to solve a data problem."
- "Describe a time you had to make a trade-off between speed and correctness."
- "How do you approach debugging a data pipeline that's producing incorrect results?"
- "Tell me about a time you contributed to or used an open-source project."

## AI/ML at Databricks

### ML Infrastructure
- **MLflow**: End-to-end ML lifecycle (tracking, registry, serving, evaluation)
- **Mosaic ML**: Large-scale model training (training LLMs on Databricks)
- **Model Serving**: Serverless model endpoints with auto-scaling
- **Feature Store**: Built on Delta Lake with point-in-time correctness
- **Vector Search**: Embedding-based retrieval for RAG applications

### LLM/GenAI Focus
- Training custom LLMs on enterprise data
- RAG (Retrieval-Augmented Generation) with Unity Catalog
- AI-powered SQL generation (natural language to SQL)
- Foundation model fine-tuning and deployment

## Preparation Tips

1. **Learn Spark internals** -- Understand how a Spark job executes: driver, executors, stages, tasks, shuffle. Read the Spark documentation.
2. **Understand Delta Lake** -- ACID on data lakes, time travel, VACUUM, Z-ordering. Read the Delta Lake paper.
3. **Code must run** -- Practice on CoderPad or LeetCode with actual execution. Syntax errors fail you.
4. **Data engineering problems** -- Practice ETL-style coding problems, not just pure algorithms.
5. **Know the Lakehouse** -- Read the Databricks Lakehouse whitepaper. Understand medallion architecture.
6. **Open-source familiarity** -- Know MLflow, Delta Lake, Apache Spark at a conceptual level.
7. **Study the tech blog** -- databricks.com/blog has excellent technical content on internals.

## Sources

- [Databricks Software Engineer Interview Guide - InterviewQuery](https://www.interviewquery.com/interview-guides/databricks-software-engineer)
- [Databricks System Design Interview - System Design Handbook](https://www.systemdesignhandbook.com/guides/databricks-system-design-interview/)
- [Databricks Interview Prep - Official](https://www.databricks.com/company/careers/interview-prep)
- [Get a Job at Databricks - Exponent](https://www.tryexponent.com/blog/databricks-interview-process)
- [Ace the Databricks SWE Interview - Prepfully](https://prepfully.com/interview-guides/databricks-software-engineer)
- [Glassdoor - Databricks SWE Interview Questions](https://www.glassdoor.com/Interview/Databricks-Software-Engineer-Interview-Questions-EI_IE954734.0,10_KO11,28.htm)
- [Databricks Blog](https://www.databricks.com/blog)
- r/dataengineering, Blind (community reports)
