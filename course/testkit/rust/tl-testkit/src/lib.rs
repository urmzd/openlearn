//! The Rust half of the course fault and determinism kit (DESIGN 4.4).
//! Course tests (the `ol-tests` crate) depend on it; it is std only.
//!
//! - [`failpoint`]: `TL_FAILPOINTS` named failpoints, the same spec as the Go,
//!   Python, and C kits.
//! - [`clock`]: a `Clock` trait with a real and a fake implementation.
pub mod clock;
pub mod failpoint;
