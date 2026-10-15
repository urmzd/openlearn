# Data Engineering (10-14 weeks)

~8-10 hours per week. Requires SQL and system design basics.

### Weeks 1-3: Foundations

| Week | Focus | Reference |
|------|-------|-----------|
| 1 | Data engineering lifecycle, OLTP vs OLAP, ETL vs ELT | Data Engineering Cookbook + DDIA Ch 1-2 |
| 2 | Storage formats (Parquet/Avro/Arrow), schemas, compression | Fundamentals of Data Engineering |
| 3 | Build an end-to-end mini pipeline (ingest → store → transform → query) | Hands-on |

### Weeks 4-7: Storage & Warehousing

| Week | Focus | Reference |
|------|-------|-----------|
| 4 | Storage engines (LSM vs B-tree), columnar execution | DDIA Ch 3 |
| 5 | Warehouse / lake / lakehouse, separation of storage & compute | Iceberg/Delta docs |
| 6 | Open table formats, partitioning, clustering, file sizing | Iceberg docs |
| 7 | Dimensional modeling (star schema, grain, SCDs) | The Data Warehouse Toolkit |

### Weeks 8-11: Batch & Streaming

| Week | Focus | Reference |
|------|-------|-----------|
| 8 | Distributed processing, partitioning, shuffle, Spark | Spark docs |
| 9 | Kafka -- log, partitions, consumer groups, replay | Kafka docs |
| 10 | Event time, watermarks, windowing, the Dataflow model | Streaming Systems |
| 11 | Delivery semantics, exactly-once, checkpointing, Flink | Flink docs + DDIA Ch 11 |

### Weeks 12-14: Orchestration & Modeling

| Week | Focus | Reference |
|------|-------|-----------|
| 12 | Orchestration (Airflow/Dagster), DAGs, idempotency, backfills | Airflow docs |
| 13 | dbt -- models, ref(), tests, materializations, lineage | dbt docs |
| 14 | Data quality, contracts, governance, data observability | Great Expectations docs |

---
