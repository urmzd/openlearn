//! tl-serve: the engine's HTTP front.
//!
//! | Module | Course module | What |
//! |---|---|---|
//! | `http` | L10.0 | the tracer: std-only HTTP/1.1 + SSE for API v0 over the byte bigram |
//! | `server` | L10.5 | API v1 (engine tier) on tokio and hyper over tl-engine: admission, abort on disconnect, drain; v1 and v2 side by side from craft.14 |
//! | `openai` | L10.5 | request validation and the response and chunk documents |
//! | `sse` | L10.5 | SSE framing; incremental UTF-8 and stop strings |
//! | `template` | L10.5 | the chat-template Jinja subset |
//! | `metrics` | L10.7 | the otel/metrics.yaml instruments and the Prometheus exposition |
//! | `telemetry` | L10.7 | request spans and W3C trace context |
//! | `tools` | L10.9 | tool calls, tool_choice, and response_format |
//! | `api_version` | craft.14 | API v1 and v2 selection, `Deprecation` and `Sunset`, v2 error codes and usage |
//!
//! The binary, src/main.rs, is yours: `--model-dir --port --health-port`
//! (the tracer, spec/cli-roles.md) calls `http::serve`; `--config
//! <runtime.toml>` (from L10.5) calls `server::run`.

pub mod api_version;
pub mod http;
pub mod control;
pub mod metrics;
pub mod openai;
pub mod server;
pub mod sse;
pub mod telemetry;
pub mod template;
pub mod tools;
