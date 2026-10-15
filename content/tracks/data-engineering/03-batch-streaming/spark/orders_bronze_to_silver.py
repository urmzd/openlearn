"""Streamflow batch job: bronze (raw order events) -> silver (clean, deduped).

This is the "store + transform" stage of the end-to-end mini-pipeline. It reads
one day's raw order events landed from Kafka as JSON, applies the topic-03
distributed-processing concepts, and writes a partitioned, columnar silver table
that dbt (topic 04) builds its marts on top of.

Concepts made concrete here:
  * Partitioning & shuffle: dedup uses a window partitioned by order_id, which
    forces a shuffle (records for the same key must meet on one executor).
  * Idempotency: we overwrite ONLY the run-date partition (partitionOverwrite =
    dynamic), so a retry/backfill for 2026-06-14 cannot duplicate or touch other
    days. This is the single most important property of a batch pipeline.
  * Columnar storage: silver is Parquet, partitioned by order_date -- analytical
    queries prune to the days they need and read only the columns they select.
  * Schema-on-read: we declare an explicit schema instead of inferring it, so a
    malformed upstream event fails loudly instead of silently shifting columns.

Run locally (needs pyspark installed):
    python orders_bronze_to_silver.py \
        --run-date 2026-06-14 \
        --bronze ./warehouse/bronze/orders \
        --silver ./warehouse/silver/orders
"""

from __future__ import annotations

import argparse

from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

# Explicit schema = a data contract with the producer. If raw events stop
# matching this shape, we want a visible failure, not silently-wrong numbers.
RAW_ORDER_SCHEMA = StructType(
    [
        StructField("order_id", StringType(), nullable=False),
        StructField("customer_id", StringType(), nullable=False),
        StructField("status", StringType(), nullable=False),
        StructField("amount", DoubleType(), nullable=False),
        StructField("currency", StringType(), nullable=False),
        StructField("event_ts", TimestampType(), nullable=False),
    ]
)


def build_silver(spark: SparkSession, bronze_path: str, run_date: str):
    """Read one day of raw events and return the cleaned, deduped silver frame."""
    # 1) INGEST. Read only this run's landing directory. The pipeline is sliced
    #    by date so each run is independent and safely re-runnable.
    raw = (
        spark.read.schema(RAW_ORDER_SCHEMA)
        .json(f"{bronze_path}/ingest_date={run_date}")
        .withColumn("order_date", F.to_date("event_ts"))
    )

    # 2) DEDUP (a shuffle). At-least-once delivery from Kafka means the same
    #    order_id can appear more than once. Keep the LATEST event per order by
    #    ranking within each key's window, then filter to rank 1. Spark must
    #    co-locate all rows for a key on one executor -> this is the shuffle.
    latest_per_order = Window.partitionBy("order_id").orderBy(F.col("event_ts").desc())
    deduped = (
        raw.withColumn("_rank", F.row_number().over(latest_per_order))
        .where(F.col("_rank") == 1)
        .drop("_rank")
    )

    # 3) CLEAN / CONFORM. Normalize the few columns the marts depend on. This is
    #    the silver contract boundary: downstream (dbt) trusts these types/values.
    cleaned = (
        deduped.withColumn("status", F.lower(F.trim("status")))
        .withColumn("currency", F.upper(F.trim("currency")))
        # Defensive: a negative amount is a known bad-data signal upstream.
        .where(F.col("amount") >= 0)
    )

    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description="Streamflow bronze->silver orders job")
    parser.add_argument(
        "--run-date", required=True, help="partition to process, YYYY-MM-DD"
    )
    parser.add_argument("--bronze", required=True, help="bronze (raw JSON) root path")
    parser.add_argument("--silver", required=True, help="silver (Parquet) root path")
    args = parser.parse_args()

    spark = (
        SparkSession.builder.appName(f"orders-bronze-to-silver-{args.run_date}")
        # Dynamic partition overwrite = idempotency. Writing the 2026-06-14
        # partition replaces ONLY that partition; every other day is untouched.
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
        .getOrCreate()
    )

    silver = build_silver(spark, args.bronze, args.run_date)

    # 4) STORE. Columnar + partitioned by order_date. The combination of
    #    overwrite + dynamic mode makes a re-run of any single day a no-op-safe
    #    replace -- the heart of safe backfills.
    (silver.write.mode("overwrite").partitionBy("order_date").parquet(args.silver))

    spark.stop()


if __name__ == "__main__":
    main()
