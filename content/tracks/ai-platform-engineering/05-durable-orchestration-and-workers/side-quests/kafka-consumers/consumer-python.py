"""Streamflow order-worker consumer (Python) -- at-least-once + idempotent.

Illustrative: needs `confluent-kafka` (pip install confluent-kafka) and a broker.

    python consumer-python.py

Mirrors the Go and Rust examples: manual offset commit AFTER durable work
(at-least-once), idempotency guard to absorb redeliveries, and clean SIGTERM
handling so Kubernetes scale-down doesn't kill us mid-message
(Containers, Kubernetes & Workloads -- ../../01-containers-kubernetes/).
"""

from __future__ import annotations

import os
import signal
import sys

from confluent_kafka import Consumer, KafkaError, Message

_running = True


def _stop(*_: object) -> None:
    """SIGTERM handler: break the poll loop so we can close() cleanly."""
    global _running
    _running = False


def idempotent(key: str) -> bool:
    """True if this key was already processed (real impl: Redis SETNX + TTL)."""
    return False


def handle(msg: Message) -> None:
    key = msg.key().decode() if msg.key() else ""
    if idempotent(key):
        return  # duplicate redelivery -- skip, but the offset still advances
    # ... persist order state in a DB transaction; emit settlement.requested ...


def main() -> None:
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    consumer = Consumer(
        {
            "bootstrap.servers": os.environ["KAFKA_BROKERS"],
            "group.id": "order-workers",
            "enable.auto.commit": False,  # we own commits
            "partition.assignment.strategy": "cooperative-sticky",
            "auto.offset.reset": "latest",
            "max.poll.interval.ms": 300_000,
        }
    )
    consumer.subscribe(["orders"])

    try:
        while _running:
            msg = consumer.poll(0.1)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    print(f"consume error: {msg.error()}", file=sys.stderr)
                continue
            try:
                handle(msg)
            except Exception as exc:  # noqa: BLE001 -- illustrative
                print(f"handler failed @ {msg.offset()}: {exc}", file=sys.stderr)
                continue  # do NOT commit -> redelivery (retry/DLQ logic omitted)
            # Commit only after durable work -> at-least-once semantics.
            consumer.commit(msg, asynchronous=False)
    finally:
        consumer.close()  # leaves the group cleanly so partitions reassign fast


if __name__ == "__main__":
    main()
