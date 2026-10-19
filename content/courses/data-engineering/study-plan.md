# Reading Plan

This plan paces the book over ten to fourteen weeks at eight to ten hours a week, which leaves room to run the examples and build a small pipeline of your own alongside the reading. It assumes you already know SQL and the basics of system design. Each block below covers one chapter; the weekly themes are suggestions for how to divide a chapter, not gates, so move faster through material you know and slow down where it is new. The [Corpus Pipeline](05-corpus-pipeline/README.md) chapter is not scheduled here: read it once you have finished the first four chapters, giving it three to four weeks of its own.

## Weeks 1 to 3: Foundations

Start with [Foundations](01-foundations/README.md). In the first week, learn the data engineering lifecycle, the distinction between OLTP and OLAP, and the move from ETL to ELT, reading *The Data Engineering Cookbook* and the first two chapters of *Designing Data-Intensive Applications* (DDIA) alongside. In the second week, turn to storage formats (Parquet, Avro, Arrow), schemas, and compression, with *Fundamentals of Data Engineering* as your companion text. Spend the third week building something: an end-to-end mini pipeline that ingests, stores, transforms, and queries a dataset. That hands-on week is what makes the rest of the book concrete.

## Weeks 4 to 7: Storage and warehousing

Move on to [Storage and Warehousing](02-storage-warehousing/README.md). In week four, study storage engines (LSM-trees versus B-trees) and columnar execution, reading chapter 3 of DDIA. In week five, compare the warehouse, the lake, and the lakehouse, and understand why storage and compute separated, using the Iceberg and Delta documentation. Week six goes deeper into open table formats, partitioning, clustering, and file sizing, again with the Iceberg docs open. Finish in week seven with dimensional modeling: star schemas, grain, and slowly changing dimensions, from Kimball's *The Data Warehouse Toolkit*.

## Weeks 8 to 11: Batch and streaming

Read [Batch and Streaming Processing](03-batch-streaming/README.md) over four weeks. Week eight covers distributed processing, partitioning, and the shuffle, with Spark as the working example and the Spark documentation as reference; run the Streamflow bronze-to-silver job while you read. Week nine is Kafka: the log, partitions, consumer groups, and replay. Week ten introduces event time, watermarks, windowing, and the Dataflow model, from *Streaming Systems*. Week eleven closes the chapter with delivery semantics, exactly-once processing, checkpointing, and Flink, using the Flink documentation and chapter 11 of DDIA.

## Weeks 12 to 14: Orchestration and modeling

Finish with [Orchestration and Modeling](04-orchestration-modeling/README.md). In week twelve, learn orchestration with Airflow and Dagster: DAGs, idempotent tasks, and backfills, and read the Streamflow DAG against the Airflow documentation. In week thirteen, work through dbt: models, `ref()`, tests, materializations, and lineage, running the Streamflow dbt project from the dbt docs. In week fourteen, cover data quality, data contracts, governance, and data observability, with the Great Expectations documentation as a practical reference.
