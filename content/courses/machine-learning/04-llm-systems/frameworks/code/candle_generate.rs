//! candle_generate.rs — load a GGUF quantized model and generate text with
//! [candle](https://github.com/huggingface/candle), the Rust "PyTorch-lite".
//!
//! Frameworks README §4: candle is kernels + runtime in pure Rust (CPU / CUDA /
//! Metal / WASM) that loads the same GGUF files as llama.cpp — so a model
//! quantized by the llama.cpp ecosystem ([Quantization §5.5](../../quantization/))
//! runs here in a single safe static binary.
//!
//! ILLUSTRATIVE — not built in CI (needs a GGUF model + tokenizer.json on disk
//! and pulls heavy deps). candle's API moves; pin versions.
//!
//! Cargo.toml:
//!   [dependencies]
//!   candle-core = "0.8"
//!   candle-transformers = "0.8"
//!   tokenizers = "0.20"
//!   anyhow = "1"
//! Run:  cargo run --release -- model.gguf tokenizer.json "Once upon a time"

use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_llama::ModelWeights;
use tokenizers::Tokenizer;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args.get(1).map(String::as_str).unwrap_or("model.gguf");
    let tok_path = args.get(2).map(String::as_str).unwrap_or("tokenizer.json");
    let prompt = args.get(3).map(String::as_str).unwrap_or("Once upon a time");

    // Pick the best available backend — the same code runs on Metal, CUDA, or CPU.
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);

    // 1. Load the GGUF weights into a quantized-llama runtime.
    let mut file = std::fs::File::open(model_path)?;
    let content = gguf_file::Content::read(&mut file)?;
    let mut model = ModelWeights::from_gguf(content, &mut file, &device)?;

    // 2. Tokenize the prompt.
    let tokenizer = Tokenizer::from_file(tok_path).map_err(anyhow::Error::msg)?;
    let mut tokens = tokenizer
        .encode(prompt, true)
        .map_err(anyhow::Error::msg)?
        .get_ids()
        .to_vec();

    // 3. Greedy-ish sampling (temperature 0.8). LogitsProcessor owns the RNG.
    let mut logits_processor = LogitsProcessor::new(42, Some(0.8), Some(0.95));
    let eos = tokenizer.token_to_id("</s>").unwrap_or(2);

    // 4. Decode loop: forward the new tokens at their position, sample, repeat.
    let mut pos = 0usize;
    for _ in 0..256 {
        let input = Tensor::new(&tokens[pos..], &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, pos)?;
        let logits = logits.squeeze(0)?; // [vocab]
        let next = logits_processor.sample(&logits)?;

        pos = tokens.len();
        tokens.push(next);
        if next == eos {
            break;
        }
        if let Some(piece) = tokenizer.id_to_token(next) {
            print!("{}", piece.replace('▁', " "));
            use std::io::Write;
            std::io::stdout().flush().ok();
        }
    }
    println!();
    Ok(())
}
