# Orchestration and Modeling

A data platform is a graph of transformations that must run in the right order, recover from failure, and produce trustworthy output. Two disciplines make that possible. **Orchestration** runs the right tasks in the right order at the right time, with retries and observability. **Modeling and quality** shape raw data into correct, tested, well-documented tables. Get these wrong and you have a fragile pile of cron jobs feeding wrong numbers to executives.

Orchestration is the difference between a pipeline and a cron job that breaks silently: it turns a pile of scripts into a reliable, observable, dependency-aware system. Pipelines are DAGs of tasks with dependencies that are scheduled, retried, and monitored, and every task should be idempotent so that retries and backfills are safe. On the modeling side, dbt brought software engineering (version control, tests, modularity, CI) to SQL transformation and now owns the "T" in ELT. And data quality is a first-class deliverable: untested pipelines produce confidently wrong dashboards, and tests, contracts, and observability are how you catch drift before consumers do.

This chapter takes those ideas in order. It starts with the fundamentals of orchestration and compares the major orchestrators, then turns to dbt and how to layer models for consumption. It finishes with data quality, contracts and governance, and data observability. Two runnable artifacts complete the Streamflow orders pipeline that began with the Spark job in [Batch and Streaming Processing](../03-batch-streaming/).

> **Learning objectives**
> - Explain how an orchestrator schedules a DAG, retries failures with backoff, and makes backfills safe through idempotent, date-partitioned tasks.
> - Compare task-centric and asset-centric orchestrators (Airflow, Dagster, Prefect, Temporal).
> - Build a dbt project in which `ref()` defines the dependency graph, with appropriate materializations and tests.
> - Layer models as staging, intermediate, and marts, and use a semantic layer to prevent metric drift.
> - Design data-quality tests, contracts, and observability checks that stop bad data before it reaches consumers.

## Before you begin

You should have read [Foundations](../01-foundations/) (idempotency) and [Storage and Warehousing](../02-storage-warehousing/) (dimensional modeling), and be comfortable with SQL. The primary references are the [dbt docs](https://docs.getdbt.com/) and the [Apache Airflow docs](https://airflow.apache.org/docs/), both free.

The best way to learn this material is to run it. Build an Airflow (or Dagster) DAG with three or four dependent tasks, make one fail, and watch the retry recover it. Convert a tangle of SQL scripts into a dbt project with `ref()`, tests, and a generated lineage graph. And for every transformation, write the data-quality test *before* you trust the output.

## In this chapter

The chapter's prose is accompanied by two pieces of working code:

- [The Streamflow dbt project](01-dbt/README.md): a minimal, runnable dbt project that turns the Spark-produced silver orders into a tested `daily_revenue` mart, illustrating `ref()`, materializations, medallion layering, and tests.
- [The Streamflow orders DAG](airflow/streamflow_orders_dag.py): an Airflow DAG that chains the Spark job to dbt with retries, exponential backoff, and date-partitioned backfills.

## Orchestration fundamentals

At the core of every orchestrator is **the DAG**: a directed acyclic graph of tasks whose edges are dependencies. The scheduler runs a task only after all of its upstream tasks have succeeded, which is a topological sort of the graph (see [Graphs](../../programming-and-algorithms/08-graphs/) in *Programming and Algorithms*).

**Scheduling** decides when a DAG runs. It can be time-based, using cron expressions, or event- and data-driven, running when upstream data lands.

**Retries and backoff** handle failure. Transient failures (a flaky network, a briefly overloaded service) are retried, with *exponential backoff* so that each retry waits longer than the last and a struggling dependency has time to recover. Permanent failures raise alerts.

**Idempotency and backfills** keep reruns safe. Re-running a task for a past date must overwrite that date's output cleanly, not duplicate it. You achieve that by partitioning by run date and writing deterministically.

**Observability** lets you see what happened: task status, run history, SLAs, lineage, and alerts on failure or lateness.

### Reading the Streamflow DAG

The DAG in [`airflow/streamflow_orders_dag.py`](airflow/streamflow_orders_dag.py) wires the whole pipeline as one graph of four tasks. `bronze_to_silver` runs the Spark job; `check_source_freshness` runs `dbt source freshness` to confirm the source actually landed recent data, failing early rather than building marts on nothing; `dbt_run` builds staging, intermediate, and mart models; and `dbt_test` runs the schema tests and the custom singular test as a quality gate after the build. A single line declares the dependency edges, and Airflow derives the run order from it:

```python
bronze_to_silver >> check_source_freshness >> dbt_run >> dbt_test
```

The reliability policy lives in `default_args`, which applies to every task unless overridden:

```python
default_args = {
    "owner": "data-eng",
    "retries": 3,
    "retry_delay": datetime.timedelta(minutes=2),
    # Exponential backoff: 2m, 4m, 8m
    "retry_exponential_backoff": True,
    "max_retry_delay": datetime.timedelta(minutes=15),
}
```

Three retries with exponential backoff wait roughly 2, 4, and 8 minutes, capped at 15, which gives a flaky dependency time to recover instead of hammering it three times in six minutes.

Idempotency comes from the run's logical date. The Spark task is invoked with `--run-date {{ ds }}`, so each run reads that day's landing partition and overwrites only that day's silver partition. Because every task is sliced by `{{ ds }}`, the DAG can safely set `catchup=True`, which tells Airflow to backfill every missing day since `start_date` (2026-06-01) as an independent run. It also sets `max_active_runs=1`, so two runs never write the same date partition concurrently. The schedule is `@daily`.

## Orchestrators compared

Airflow is the default, but not the only choice. The four main orchestrators differ in what they treat as the unit of work.

| Tool | Model | Strength |
|------|-------|----------|
| **Airflow** | Task-centric DAGs in Python | Mature, huge ecosystem, the default |
| **Dagster** | Asset-centric (software-defined assets) | Data-aware, testable, strong typing/lineage |
| **Prefect** | Dynamic Python flows | Flexible, light, good local DX |
| **Temporal** | Durable workflow execution | Long-running, stateful, exactly-once workflows |

The most important difference is the asset-centric shift. Airflow models *the tasks to run*. Dagster models *the data assets you want to exist*. Because the orchestrator then understands which assets depend on which, it knows lineage natively and can re-materialize just what is stale, rather than re-running a whole task graph.

## dbt and transformation as code

The orchestrator decides when work runs. dbt is how most teams write the transformation work itself, and it owns the "T" in ELT.

In dbt, a model is a SQL `SELECT` statement, templated with Jinja. dbt handles the DDL (creating tables and views), the dependencies, and the materialization. Models refer to each other with `ref('model')`, and dbt infers the dependency graph from those references and runs models in order. You never write the run order down.

Each model has a **materialization**: a *view* (cheap and always fresh), a *table* (precomputed for fast reads), *incremental* (process only new rows on each run), or *ephemeral* (inlined into downstream models as a common table expression).

dbt also makes testing routine. Built-in tests (`unique`, `not_null`, `accepted_values`, `relationships`) are declared in YAML next to the model, custom tests are written as SQL, and all of them run in CI. dbt generates documentation and a column-level lineage graph from the model DAG.

Why did this matter so much? Because it brought version control, modularity, testing, and CI/CD to analytics SQL, which until then had lived in ad hoc scripts. That shift is the foundation of the analytics engineering role.

The [The Streamflow dbt Project](01-dbt/README.md) implements exactly this staging, intermediate, and marts chain over the Spark-produced silver table, with built-in tests (`not_null`, `unique`, `accepted_values`), a source freshness check, and a custom singular test. Trace it end to end: `source('silver','orders')` feeds `stg_orders`, which feeds `int_orders_enriched`, which feeds `daily_revenue`. Then run `dbt build` against DuckDB, with zero infrastructure, and `dbt docs serve` to browse the lineage graph.

## Data modeling for consumption

The layering in that project is not arbitrary. It builds on the [dimensional modeling](../02-storage-warehousing/) of the previous chapter and maps the medallion pattern onto dbt.

**Medallion layering** runs from bronze (raw) through silver (cleaned and conformed) to gold (business-level marts), and maps cleanly onto dbt's staging, intermediate, and marts folders. **Staging models** are one per source and do only light cleaning and renaming; they are the contract boundary with raw data, and everything downstream refers to them rather than to the source. **Marts** are business-facing, dimensional, and denormalized for BI.

The final piece is a **semantic layer** or metrics layer, which defines each metric (for example, `revenue`) once, so that every dashboard computes it identically. This kills metric drift, the situation where three teams report three different revenue numbers. The Streamflow project shows the same idea at small scale: `int_orders_enriched` defines the revenue-recognition rule (only completed orders count) in exactly one place, and every mart inherits it.

## Data quality and testing

Well-layered models are only as good as the data in them, which is why quality deserves its own discipline.

Data quality has several dimensions: accuracy, completeness, consistency, timeliness, validity, and uniqueness. Tests target them in different ways. *Schema tests* check types and nullability. *Constraint tests* check ranges and accepted values. *Referential integrity* tests check that foreign keys resolve. *Freshness* tests ask whether the data is recent. *Volume and anomaly* tests ask whether row counts swung unexpectedly.

Tools such as dbt tests, [Great Expectations](https://docs.greatexpectations.io/), and Soda express these as assertions that fail the pipeline before bad data reaches consumers. A final technique, **reconciliation**, cross-checks totals against a source of truth to catch silent loss or duplication that row-level tests miss.

The Streamflow project's custom test, [`tests/assert_daily_revenue_is_sane.sql`](01-dbt/tests/assert_daily_revenue_is_sane.sql), is a good model of writing the test before trusting the output. It asserts that the mart's grain holds (at most one row per `order_date` and `currency`) and that revenue is never negative.

## Data contracts and governance

Tests catch problems inside your pipeline. Contracts and governance manage the boundaries with the people and systems around it.

A **data contract** is an explicit agreement on schema and SLA between a producer and its consumers. If a producer changes its schema unexpectedly, the build breaks rather than the dashboard. The Spark job's explicit schema and the dbt staging layer are both contract boundaries in this sense.

**Catalogs** such as DataHub, OpenMetadata, and Unity Catalog provide metadata and discovery so people can find datasets and know whether to trust them. **Lineage**, end-to-end and at column level, answers two questions you will be asked constantly: "what breaks if I change this?" and "where did this number come from?" **Governance** covers access control, PII handling, GDPR and CCPA right-to-delete requests, audit trails, and retention policies.

## Data observability

The last discipline applies the ideas of [Observability](../../systems/11-observability/) in *Systems Engineering* to data. Data observability watches five pillars: freshness, volume, schema, distribution, and lineage. Its workflow is detect, triage, resolve: alert on an anomaly, trace through lineage to its root cause, then fix it and backfill.

Mature teams set SLAs and SLOs for data, such as "the orders table is fresh within 1 hour, 99% of days," and treat data products like services.

## A reliability checklist

Everything in this chapter adds up to a short list of properties every pipeline should have, each with a concrete way to achieve it.

| Property | How to achieve it |
|----------|-------------------|
| Idempotent | Partition by run date, deterministic writes, MERGE/overwrite |
| Recoverable | Retries + backoff, checkpoints, safe backfills |
| Observable | Run status, lineage, freshness + volume alerts |
| Correct | Schema/constraint/freshness tests in CI |
| Documented | dbt docs, data catalog, ownership metadata |
| Governed | Access control, PII tagging, contracts |

## Where this leads

The concepts here connect across the library. Task and asset dependency ordering is topological sort from [Graphs](../../programming-and-algorithms/08-graphs/) in *Programming and Algorithms*. Data observability, SLOs, and alerting come from [Observability](../../systems/11-observability/) in *Systems Engineering*. The CI/CD, testing, and version-control practices that dbt brings to analytics are the subject of [*Software Engineering*](../../software-engineering/). Idempotency and retries were introduced in [Foundations](../01-foundations/). Feature freshness and lineage for trustworthy training and serving data reappear in [LLM Systems](../../ai-engineering/20-inference-engines/). The next chapter, [Corpus Pipeline](../05-corpus-pipeline/), applies all of this to building a training corpus, ending with the pipeline as a durable workflow.

In industry, dbt Labs builds dbt and its semantic layer and coined analytics engineering. Airflow was created at Airbnb, which also built the Minerva metrics platform for data quality and consistent metrics. Netflix runs orchestration and data observability at scale. Databricks offers Unity Catalog, lineage, and Delta Live Tables. Palantir focuses on data integration, lineage, and governance.

### Further reading

- [dbt docs](https://docs.getdbt.com/) and [Apache Airflow docs](https://airflow.apache.org/docs/) (free).
- [Dagster docs](https://docs.dagster.io/) (free).
- [Great Expectations docs](https://docs.greatexpectations.io/) (free).
- Joe Reis and Matt Housley, *Fundamentals of Data Engineering*, chapters 8 and 9.

## Key terms

- **DAG**: a directed acyclic graph of tasks whose edges are dependencies; the scheduler runs tasks in topological order.
- **Exponential backoff**: retrying with a delay that grows each attempt.
- **Backfill**: re-running a pipeline for past dates, safe only when tasks are idempotent.
- **Asset-centric orchestration**: modeling the data assets that should exist rather than the tasks to run.
- **`ref()`**: dbt's function for referencing another model, from which dbt infers the dependency graph.
- **Materialization**: how dbt persists a model: view, table, incremental, or ephemeral.
- **Staging model**: a one-per-source model doing light renaming and recasting; the contract boundary with raw data.
- **Semantic layer**: a single definition of each metric shared by all consumers.
- **Data contract**: an explicit schema and SLA agreement between a data producer and its consumers.
- **Data observability**: monitoring freshness, volume, schema, distribution, and lineage of data.

## Summary

- Orchestration turns scripts into a reliable, observable, dependency-aware DAG with scheduling, retries, and backoff.
- Idempotent, date-partitioned tasks make retries and backfills safe; the Streamflow DAG slices every task by `{{ ds }}`.
- Asset-centric orchestrators such as Dagster understand lineage and re-materialize only what is stale.
- dbt brought version control, tests, modularity, and CI to SQL; `ref()` builds the DAG and staging, intermediate, and marts layers map to the medallion pattern.
- Data quality is a deliverable: tests, reconciliation, contracts, and observability catch bad data before consumers do.

## Review questions

1. Why is `catchup=True` safe in the Streamflow DAG, and what would go wrong if the Spark task appended instead of overwriting its partition?
   <details><summary>Answer</summary>Every task is sliced by `{{ ds }}` and the Spark job overwrites only that date's partition, so each backfilled day is an independent idempotent run. If it appended, any retry or rerun would duplicate that day's orders.</details>

2. What does `max_active_runs=1` protect against?
   <details><summary>Answer</summary>Two runs writing the same date partition concurrently, which could interleave writes and corrupt or duplicate the partition.</details>

3. In the dbt project, you never declared that `stg_orders` runs before `daily_revenue`. How does dbt know?
   <details><summary>Answer</summary>Each model references its upstream with `ref()` (or `source()`), and dbt builds the dependency DAG from those references and runs models in topological order.</details>

4. What is the difference between a task-centric and an asset-centric orchestrator?
   <details><summary>Answer</summary>A task-centric orchestrator (Airflow) models the tasks to run and their order. An asset-centric one (Dagster) models the data assets that should exist, so it understands lineage directly and can re-materialize only stale assets.</details>

5. How does defining revenue once in `int_orders_enriched` relate to the semantic layer?
   <details><summary>Answer</summary>It is the same idea at small scale: the revenue-recognition rule (only completed orders count) lives in one place, and every mart inherits it, so dashboards cannot disagree about what revenue means. A semantic layer does this for metrics across all consumers.</details>

6. Name the five pillars of data observability.
   <details><summary>Answer</summary>Freshness, volume, schema, distribution, and lineage.</details>
