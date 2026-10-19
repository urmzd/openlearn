//! mistralrs_serve.rs — turnkey Rust inference with
//! [mistral.rs](https://github.com/EricLBuehler/mistral.rs).
//!
//! Frameworks README §4: mistral.rs is the "batteries-included" Rust tier —
//! it borrows candle's kernels and adds a runtime, **ISQ** (in-situ
//! quantization: quantize an fp16 model to Q4K/Q8/etc. at load time, no
//! pre-quantized file needed), and an OpenAI-compatible HTTP server.
//!
//! ILLUSTRATIVE — not built in CI (downloads a model from the Hub, heavy deps,
//! async runtime). mistral.rs's builder API evolves; pin the version and check
//! the docs.
//!
//! Cargo.toml:
//!   [dependencies]
//!   mistralrs = "0.3"
//!   tokio = { version = "1", features = ["full"] }
//!   anyhow = "1"
//! Run:  cargo run --release

use mistralrs::{IsqType, TextMessageRole, TextMessages, TextModelBuilder};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load an fp16 model and quantize it in-situ to 4-bit (Q4K) at load time.
    // Backend (Metal / CUDA / CPU) is auto-selected.
    let model = TextModelBuilder::new("meta-llama/Llama-3.2-3B-Instruct")
        .with_isq(IsqType::Q4K)
        .with_logging()
        .build()
        .await?;

    let messages = TextMessages::new()
        .add_message(TextMessageRole::System, "You are a concise assistant.")
        .add_message(
            TextMessageRole::User,
            "In one sentence: why is LLM decode memory-bandwidth bound?",
        );

    let response = model.send_chat_request(messages).await?;
    println!(
        "{}",
        response.choices[0]
            .message
            .content
            .as_deref()
            .unwrap_or("(no content)")
    );

    // ── Serving instead of in-process ──────────────────────────────────────
    // For an OpenAI-compatible HTTP server (drop-in for vLLM / llama-server),
    // use the prebuilt binary instead of embedding the engine:
    //
    //   cargo install mistralrs-server
    //   mistralrs-server --isq Q4K --port 1234 plain -m meta-llama/Llama-3.2-3B-Instruct
    //
    // then POST to http://localhost:1234/v1/chat/completions like any OpenAI API.
    Ok(())
}
