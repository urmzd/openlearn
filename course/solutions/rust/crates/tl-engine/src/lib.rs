//! tl-engine: the inference engine (DESIGN 2.1, 4.3 L10).
//!
//! L8.4 creates this crate with `prefix`, the radix prefix cache the block
//! manager (L10.4) consults before allocating KV blocks for a prompt. Each
//! later module adds one `pub mod` line here (DESIGN 2.15, Q7):
//!
//! | Module | Course module | What |
//! |---|---|---|
//! | `prefix` | L8.4 | radix prefix cache over token ids |
//! | `quant`, `model`, `forward`, `runner`, `sample`, `kv` | L10.1 | f16 and int4, Candle-backed model math, the model directory, a Rust KV cache, the runner, the sampler and PCG32 |
//! | `sched` | L10.2 | continuous batching: admission, priority with aging, preemption |
//! | `chunk` | L10.3 | chunked prefill and the mixed prefill + decode batch |
//! | `block_manager` | L10.4 | KV blocks per request with a prefix cache (none, hash, radix) |
//! | `engine` | L10.5 | the step loop: runner + scheduler + block manager + sampling |
//!
//! Later modules (L10.6 `kv_transfer`, `heartbeat`; L10.8 `spec`; L10.9
//! `constrain`) join the same way.

pub mod block_manager;
pub mod chunk;
pub mod constrain;
pub mod engine;
pub mod forward;
pub mod heartbeat;
pub mod kv;
pub mod kv_transfer;
pub mod model;
pub mod prefix;
pub mod quant;
pub mod runner;
pub mod sample;
pub mod sched;
pub mod spec;

pub use forward::{ForwardBatch, ForwardSeq, Logits};
pub use runner::{EngineConfig, KvConfig, ModelRunner, PrefixCache, Quant, SchedPolicy, SharedPool};
pub use sample::{sample, Pcg32, SamplingParams};
