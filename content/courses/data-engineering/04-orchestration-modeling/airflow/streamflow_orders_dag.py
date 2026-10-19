"""Airflow DAG: the end-to-end Streamflow orders pipeline, wired as one graph.

This is the orchestration layer -- it runs the right tasks in the right order,
retries transient failures, and makes every run safely re-runnable. It chains
the two earlier stages into a single DAG:

    bronze_to_silver (Spark, topic 03)
        -> check_source_freshness (dbt)
        -> dbt_run   (build staging -> intermediate -> marts)
        -> dbt_test  (schema + custom tests; fails the run on bad data)

Concepts made concrete:
  * The DAG: tasks + dependencies; the scheduler runs a task only after its
    upstreams succeed (topological order -- see content/courses/algorithms/06-graphs).
  * Idempotency: every task is sliced by the run's logical date ({{ ds }}), so a
    retry or a backfill for one day overwrites exactly that day -- no dupes.
  * Retries & backoff: transient failures retry with exponential backoff before
    the task is marked failed and alerts fire.
  * Quality gate: dbt_test runs AFTER the build; if an invariant breaks, the run
    fails loudly instead of publishing confidently-wrong marts.
"""

from __future__ import annotations

import datetime

from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator

# Where the project lives on the worker. In a real deploy these come from an
# Airflow Variable or Connection; constants keep the example readable.
DBT_DIR = "/opt/streamflow/dbt"
SPARK_JOB = "/opt/streamflow/spark/orders_bronze_to_silver.py"
WAREHOUSE = "/opt/streamflow/warehouse"

# default_args apply to every task unless overridden. This is where the
# reliability policy lives.
default_args = {
    "owner": "data-eng",
    "retries": 3,
    "retry_delay": datetime.timedelta(minutes=2),
    # Exponential backoff: 2m, 4m, 8m -- give a flaky dependency time to recover
    # instead of hammering it three times in six minutes.
    "retry_exponential_backoff": True,
    "max_retry_delay": datetime.timedelta(minutes=15),
}

with DAG(
    dag_id="streamflow_orders",
    description="bronze -> silver (Spark) -> tested marts (dbt)",
    default_args=default_args,
    schedule="@daily",
    start_date=datetime.datetime(2026, 6, 1),
    # catchup=True lets Airflow backfill every missing day since start_date,
    # each as an independent, idempotent run -- only safe BECAUSE tasks partition
    # by {{ ds }}.
    catchup=True,
    # One run per day at a time: a date partition must not be written by two runs
    # concurrently.
    max_active_runs=1,
    tags=["streamflow", "data-engineering"],
) as dag:
    # 1) STORE + TRANSFORM. Spark reads the {{ ds }} landing partition and writes
    #    that day's silver partition. Re-running overwrites only that partition.
    bronze_to_silver = BashOperator(
        task_id="bronze_to_silver",
        bash_command=(
            f"spark-submit {SPARK_JOB} "
            "--run-date {{ ds }} "
            f"--bronze {WAREHOUSE}/bronze/orders "
            f"--silver {WAREHOUSE}/silver/orders"
        ),
    )

    # 2) FRESHNESS GATE. Before transforming, confirm the source actually landed
    #    recent data. If it's stale, fail here rather than build marts on nothing.
    check_source_freshness = BashOperator(
        task_id="check_source_freshness",
        bash_command=f"cd {DBT_DIR} && dbt source freshness",
    )

    # 3) BUILD MARTS. dbt walks its own DAG (staging -> intermediate -> marts).
    dbt_run = BashOperator(
        task_id="dbt_run",
        bash_command=f"cd {DBT_DIR} && dbt run",
    )

    # 4) QUALITY GATE. Schema tests + the custom singular test. Zero rows = pass.
    dbt_test = BashOperator(
        task_id="dbt_test",
        bash_command=f"cd {DBT_DIR} && dbt test",
    )

    # The dependency edges. This single line IS the DAG -- Airflow derives the
    # run order from it.
    bronze_to_silver >> check_source_freshness >> dbt_run >> dbt_test
