# Orchestration & Modeling

## Overview

- **Primary reference**: [dbt docs](https://docs.getdbt.com/) (free) + [Apache Airflow docs](https://airflow.apache.org/docs/) (free)
- **Supplementary**: [Dagster docs](https://docs.dagster.io/) (free), [Great Expectations docs](https://docs.greatexpectations.io/) (free), *Fundamentals of Data Engineering* Ch 8-9 (recommended)
- **Prerequisites**: [Foundations](../01-foundations/), [Storage & Warehousing](../02-storage-warehousing/), SQL
- **Estimated time**: 2-3 weeks at 8-10 hrs/week

## Key Takeaways

- Orchestration turns a pile of scripts into a *reliable, observable, dependency-aware* system -- it is the difference between a pipeline and a cron job that breaks silently
- Pipelines are **DAGs**: tasks with dependencies, scheduled, retried, monitored. Design every task to be **idempotent** so retries and backfills are safe
- **dbt** brought software engineering (version control, tests, modularity, CI) to SQL transformation -- it owns the "T" in ELT
- **Data quality is a first-class deliverable**: untested data pipelines produce confidently wrong dashboards. Tests, contracts, and observability catch drift before consumers do

## How to Study

- Read the runnable [`dbt/`](dbt/) project here: trace `source('silver','orders')` → `stg_orders` → `int_orders_enriched` → `daily_revenue` and notice you never wrote the run order down -- `ref()` builds the DAG. Then run `dbt build` (DuckDB, zero infra) and `dbt docs serve` to browse the lineage graph
- Read the [`airflow/streamflow_orders_dag.py`](airflow/streamflow_orders_dag.py) DAG: it chains the Spark job to `dbt run` → `dbt test`, with retries, exponential backoff, and `{{ ds }}` date-partitioning for idempotent backfills
- Build an Airflow (or Dagster) DAG with 3-4 dependent tasks, a failure, and a retry -- watch it recover
- Convert a tangle of SQL scripts into a dbt project with `ref()`, tests, and a generated lineage graph
- For every transformation, write the data-quality test *before* you trust the output (see [`tests/assert_daily_revenue_is_sane.sql`](dbt/tests/assert_daily_revenue_is_sane.sql))

---

# Concepts & Techniques

## Core Insight

A data platform is a graph of transformations that must run in the right order, recover from failure, and produce trustworthy output. Two disciplines make that possible: **orchestration** (running the right tasks in the right order at the right time, with retries and observability) and **modeling + quality** (shaping raw data into correct, tested, well-documented tables). Get these wrong and you have a fragile pile of cron jobs feeding wrong numbers to executives.

## 1. Orchestration Fundamentals

**Key ideas**:
- **The DAG**: a directed acyclic graph of tasks; edges are dependencies. The scheduler runs a task only after its upstreams succeed (see [Graphs](../../algorithms/06-graphs/) -- topological sort)
- **Scheduling**: time-based (cron) or event/data-driven (run when upstream data lands)
- **Retries & backoff**: transient failures retried with exponential backoff; permanent failures alert
- **Idempotency & backfills**: re-running for a past date must overwrite cleanly, not duplicate -- partition by run date and write deterministically
- **Observability**: task status, run history, SLAs, lineage, alerting on failure or lateness

**Worked example**: [`airflow/streamflow_orders_dag.py`](airflow/streamflow_orders_dag.py) wires the whole pipeline as one DAG -- `bronze_to_silver` (Spark) → `check_source_freshness` → `dbt_run` → `dbt_test` -- with retries + exponential backoff in `default_args` and `{{ ds }}` slicing so `catchup` backfills are safe.

## 2. Orchestrators Compared

| Tool | Model | Strength |
|------|-------|----------|
| **Airflow** | Task-centric DAGs in Python | Mature, huge ecosystem, the default |
| **Dagster** | Asset-centric (software-defined assets) | Data-aware, testable, strong typing/lineage |
| **Prefect** | Dynamic Python flows | Flexible, light, good local DX |
| **Temporal** | Durable workflow execution | Long-running, stateful, exactly-once workflows |

**Asset-centric shift**: Dagster models *the data assets you want to exist* rather than *the tasks to run* -- the orchestrator understands lineage and can re-materialize just what's stale.

## 3. dbt & Transformation-as-Code

**Owns the "T" in ELT**

**Key ideas**:
- **SQL + Jinja**: models are `SELECT` statements; dbt handles DDL, dependencies, and materialization
- **`ref()` and the DAG**: models reference each other with `ref('model')`; dbt infers the dependency graph and runs in order
- **Materializations**: view, table, incremental (only process new rows), ephemeral
- **Testing**: built-in tests (unique, not_null, accepted_values, relationships) + custom tests, run in CI
- **Documentation & lineage**: auto-generated docs and a column-level lineage graph from the model DAG
- **Why it mattered**: brought version control, modularity, testing, and CI/CD to analytics SQL -- the foundation of the "analytics engineering" role

**Worked example**: the [`dbt/`](dbt/) project implements exactly this staging → intermediate → marts chain over the Spark-produced silver table, with built-in tests (`not_null`, `unique`, `accepted_values`), a source freshness check, and a custom singular test. See its [README](dbt/README.md) for the rendered DAG and run instructions.

## 4. Data Modeling for Consumption

**Builds on [dimensional modeling](../02-storage-warehousing/)**

**Key ideas**:
- **Medallion layering**: bronze (raw) → silver (cleaned, conformed) → gold (business-level marts). Maps cleanly to dbt staging → intermediate → marts
- **Staging models**: one per source, light cleaning + renaming -- the contract boundary with raw data
- **Marts**: business-facing, dimensional, denormalized for BI
- **Semantic layer / metrics**: define metrics once (e.g. `revenue`) so every dashboard computes them identically -- kills metric drift across teams

## 5. Data Quality & Testing

**Key ideas**:
- **Dimensions of quality**: accuracy, completeness, consistency, timeliness, validity, uniqueness
- **Test types**: schema tests (types, nullability), constraint tests (ranges, accepted values), referential integrity, freshness (is the data recent?), volume/anomaly (did row counts swing?)
- **Tools**: dbt tests, [Great Expectations](https://docs.greatexpectations.io/), Soda -- assertions that fail the pipeline before bad data reaches consumers
- **Reconciliation**: cross-check totals against source-of-truth to catch silent loss/duplication

## 6. Data Contracts & Governance

**Key ideas**:
- **Data contracts**: explicit schema + SLA agreements between producers and consumers; break the build if a producer changes schema unexpectedly
- **Catalog & discovery**: metadata catalogs (DataHub, OpenMetadata, Unity Catalog) so people can find and trust datasets
- **Lineage**: end-to-end column-level provenance -- answer "what breaks if I change this?" and "where did this number come from?"
- **Governance**: access control, PII handling, GDPR/CCPA right-to-delete, audit trails, retention policies

## 7. Data Observability

**The [Observability](../../systems/04-observability/) discipline applied to data**

**Key ideas**:
- **Five pillars**: freshness, volume, schema, distribution, lineage
- **Detect → triage → resolve**: alert on anomalies, trace through lineage to the root cause, fix and backfill
- **SLAs/SLOs for data**: "the orders table is fresh within 1 hour, 99% of days" -- treat data products like services

---

## Pipeline Reliability Checklist

| Property | How to achieve it |
|----------|-------------------|
| Idempotent | Partition by run date, deterministic writes, MERGE/overwrite |
| Recoverable | Retries + backoff, checkpoints, safe backfills |
| Observable | Run status, lineage, freshness + volume alerts |
| Correct | Schema/constraint/freshness tests in CI |
| Documented | dbt docs, data catalog, ownership metadata |
| Governed | Access control, PII tagging, contracts |

## Connections to Other Tracks

| Concept | Connected Track | Application |
|---------|-----------------|-------------|
| DAGs, topological sort | [Graphs](../../algorithms/06-graphs/) | Task/asset dependency ordering |
| Observability, SLOs, alerting | [Observability](../../systems/04-observability/) | Data observability |
| CI/CD, testing, version control | [Software Craftsmanship](../../software-craftsmanship/) | dbt + analytics engineering |
| Idempotency, retries | [Foundations](../01-foundations/) | Safe re-execution |
| Feature freshness, lineage | [LLM Systems](../../ml/04-llm-systems/) | Trustworthy training/serving data |

## Company Relevance

| Company | How This Appears | Difficulty |
|---------|-----------------|------------|
| dbt Labs | dbt, semantic layer, analytics engineering | Expert |
| Airbnb | Airflow (created here), data quality (Minerva metrics) | Expert |
| Netflix | Orchestration at scale, data observability | Expert |
| Databricks | Unity Catalog, lineage, Delta Live Tables | Expert |
| Palantir | Data integration, lineage, governance | Advanced |
