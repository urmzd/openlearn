# Data Engineering

Data is almost never produced for the people who need it. Data engineering builds the systems that move, store, and shape it so analysts, ML engineers, and applications can rely on it. This book is for software engineers who know SQL and the basics of system design and want to understand how a modern data platform fits together. It closes by building one demanding pipeline end to end: the pretraining corpus for a language model.

## What you will be able to do

- Place any data system on the lifecycle of generation, ingestion, storage, transformation, and serving, and choose storage by access pattern: row stores for OLTP, column stores for OLAP, open table formats for the lakehouse.
- Design ELT pipelines that load raw data cheaply and transform it in place with SQL.
- Build batch and streaming jobs with event time, watermarks, and windows, and state exactly which delivery guarantee a job provides.
- Make every pipeline idempotent and replayable, with tests, contracts, and backfills that work.
- Build a deterministic, licensed, deduplicated training corpus and explain every stage of it.

## Ways through this book

Reading every chapter takes about thirteen to eighteen weeks at eight to ten hours a week.

- **The whole book, in order** (13 to 18 weeks): [Chapter 1: Foundations](01-foundations/README.md) through [Chapter 5: Corpus Pipeline](05-corpus-pipeline/README.md). Chapters 2 and 3 can go in either order.
- **New to data, or coming from analytics and BI** (about 10 weeks): [Chapter 1: Foundations](01-foundations/README.md), then [Chapter 2: Storage and Warehousing](02-storage-warehousing/README.md), then [Chapter 4: Orchestration and Modeling](04-orchestration-modeling/README.md), where dbt lives.
- **Coming from backend or systems work** (about 8 weeks): skim [Chapter 1: Foundations](01-foundations/README.md), then [Chapter 3: Batch and Streaming Processing](03-batch-streaming/README.md), then [Chapter 4: Orchestration and Modeling](04-orchestration-modeling/README.md).
- **Building training data for ML** (about 10 weeks): [Chapter 1: Foundations](01-foundations/README.md), [Chapter 2: Storage and Warehousing](02-storage-warehousing/README.md) for columnar files, then [5.1 Async fetch](05-corpus-pipeline/01-async-fetch-resume-checksums.md) through [5.8 Licensing ledger verification](05-corpus-pipeline/08-ledger-and-datasheet.md). Read [Durable Orchestration and Workers](../ai-platform-engineering/05-durable-orchestration-and-workers/README.md) in *AI Platform Engineering* before [5.9 CorpusBuild](05-corpus-pipeline/09-corpusbuild-as-a-durable-workflow.md).

## Before you start

You should be comfortable writing SQL and know the basics of [System Design](../systems/01-system-design/) from *Systems and Architecture*. For the corpus chapter, the hashing and Bloom filters of *[Algorithms](../algorithms/)*, the asyncio primer in *[Programming Languages](../programming-languages/)*, and the licensing and PII policies in *[Responsible AI](../responsible-ai/)* all help. From here, *[Infrastructure](../infrastructure/)* covers the messaging and search stack data platforms run on, and *[Machine Learning and AI](../machine-learning/)* and *[Build Your Own LLM System](../build-an-llm/)* consume the corpus this book builds.

New to these books? [How to read these books](../../references/how-to-read.md) explains how sections, review questions, and solutions work.

## The running example

A small **Streamflow orders** pipeline runs through Chapters 3 and 4, locally on DuckDB with no cloud setup. It uses the same Streamflow platform as *[Infrastructure](../infrastructure/)* and *[Diagramming and Documentation](../diagramming/)*.

| Stage | File | Chapter |
|-------|------|---------|
| Raw to clean (Spark, bronze to silver) | [`spark/orders_bronze_to_silver.py`](03-batch-streaming/spark/orders_bronze_to_silver.py) | 3 |
| Clean to tested marts (dbt, silver to gold) | [`dbt/`](04-orchestration-modeling/01-dbt/) | 4 |
| Orchestrate the DAG (Airflow) | [`airflow/streamflow_orders_dag.py`](04-orchestration-modeling/airflow/streamflow_orders_dag.py) | 4 |

Solution code for the corpus pipeline lives in [`course/solutions/python/corpus/`](../../../course/solutions/python/corpus/), with the interface contracts each section implements under [`course/contracts/`](../../../course/contracts/).
