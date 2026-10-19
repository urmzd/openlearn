//! Generated gRPC types, clients, and servers of the course contracts.
//! One module per proto package:
//!
//! | Module | Proto | Served by |
//! |---|---|---|
//! | [`tl::engine::v1`] | `proto/tl/engine/v1/engine.proto` | every engine (L10.6) |
//! | [`tl::kv::v1`] | `proto/tl/kv/v1/kv.proto` | decode engines (L10.6, craft.13) |
//! | [`tl::control::v1`] | `proto/tl/control/v1/control.proto` | the gateway (gw.05) |
//! | [`tl::durable::v1`] | `proto/tl/durable/v1/durable.proto` | the durable server (dur.02 to dur.05) |
//! | [`tl::raft::v1`] | `proto/tl/raft/v1/raft.proto` | durable replicas (dur.10, optional) |
//!
//! The files under `src/gen/` are generated; edit the `.proto` files and
//! regenerate them with prost and tonic instead of editing them by hand.
#![allow(clippy::all)]

pub mod tl {
    pub mod engine {
        pub mod v1 {
            include!("gen/tl.engine.v1.rs");
        }
    }
    pub mod kv {
        pub mod v1 {
            include!("gen/tl.kv.v1.rs");
        }
    }
    pub mod control {
        pub mod v1 {
            include!("gen/tl.control.v1.rs");
        }
    }
    pub mod durable {
        pub mod v1 {
            include!("gen/tl.durable.v1.rs");
        }
    }
    pub mod raft {
        pub mod v1 {
            include!("gen/tl.raft.v1.rs");
        }
    }
}

/// Every message is capped at 4 MiB. Pass it to
/// `max_decoding_message_size` and `max_encoding_message_size` on clients and
/// servers.
pub const MAX_MESSAGE_BYTES: usize = 4 * 1024 * 1024;
