//! tl-ds: the Rust data structures of the system (DESIGN 2.14), each with a
//! caller in the tokenizer or the engine.
//!
//! | Module  | Course module | Caller |
//! |---|---|---|
//! | `robin` | ds.05 | `tl-tok` vocabulary and merge ranks (L1.5) |
//! | `heap`  | ds.06 | `tl-tok` merge queue (L1.5); the engine's waiting queue (L10.2) |
//! | `bloom` | ds.08 | exact dedup in the corpus pipeline (data.03), through the Rust API |
//!
//! ds.05 owns this crate root. The radix tree over token ids (`radix`, ds.07)
//! is available to the engine's prefix cache. std only.

pub mod bloom;
pub mod heap;
pub mod radix;
pub mod robin;

pub use bloom::Bloom;
pub use heap::{Handle, Heap, LazyHeap};
pub use robin::{FxBuild, RobinHoodMap};
