//! Streamflow order-worker consumer (Rust) -- at-least-once + idempotent.
//!
//! Illustrative: needs `rdkafka` (with the `tokio` feature) + `tokio` and a broker.
//! This is the language the Dockerfile and C4 diagrams use for the worker.
//!
//! Same contract as the Go/Python examples: manual commit AFTER durable work
//! (at-least-once), an idempotency guard to absorb redeliveries, and a tokio
//! SIGTERM handler so Kubernetes scale-down drains cleanly
//! (Containers, Kubernetes & Workloads -- ../../01-containers-kubernetes/).
//!
//! ```ignore
//! cargo run --release
//! ```

use rdkafka::config::ClientConfig;
use rdkafka::consumer::{CommitMode, Consumer, StreamConsumer};
use rdkafka::message::Message;
use std::env;
use tokio::signal::unix::{signal, SignalKind};

/// True if this key was already processed (real impl: Redis SETNX + TTL).
fn idempotent(_key: &str) -> bool {
    false
}

async fn handle(msg: &impl Message) -> anyhow::Result<()> {
    let key = msg.key().map(|k| String::from_utf8_lossy(k).into_owned()).unwrap_or_default();
    if idempotent(&key) {
        return Ok(()); // duplicate redelivery -- skip; offset still advances
    }
    // ... persist order state in a DB transaction; emit settlement.requested ...
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let consumer: StreamConsumer = ClientConfig::new()
        .set("bootstrap.servers", env::var("KAFKA_BROKERS")?)
        .set("group.id", "order-workers")
        .set("enable.auto.commit", "false") // we own commits
        .set("partition.assignment.strategy", "cooperative-sticky")
        .set("auto.offset.reset", "latest")
        .set("max.poll.interval.ms", "300000")
        .create()?;
    consumer.subscribe(&["orders"])?;

    let mut sigterm = signal(SignalKind::terminate())?;

    loop {
        tokio::select! {
            // SIGTERM from Kubernetes: stop consuming. Dropping `consumer`
            // leaves the group cleanly so partitions reassign immediately.
            _ = sigterm.recv() => {
                eprintln!("SIGTERM received, draining...");
                break;
            }
            result = consumer.recv() => {
                let msg = match result {
                    Ok(m) => m,
                    Err(e) => { eprintln!("consume error: {e}"); continue; }
                };
                if let Err(e) = handle(&msg).await {
                    eprintln!("handler failed @ {}: {e}", msg.offset());
                    continue; // do NOT commit -> redelivery (retry/DLQ omitted)
                }
                // Commit only after durable work -> at-least-once semantics.
                if let Err(e) = consumer.commit_message(&msg, CommitMode::Async) {
                    eprintln!("commit failed: {e}");
                }
            }
        }
    }
    Ok(())
}
