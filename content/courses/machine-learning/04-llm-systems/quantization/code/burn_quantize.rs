//! burn_quantize.rs — quantization with the Burn framework (Rust).
//!
//! Mirrors README section 7. Burn exposes a backend-agnostic quantization API
//! in `burn::tensor::quantization`: a `QuantScheme` (config) selecting a
//! `QuantValue` (Q8S/Q4S/Q2S fixed-point symmetric, or E4M3/E5M2/E2M1 float),
//! a `QuantLevel` (per-tensor or per-block granularity), and `Calibration`
//! (min-max). `quantize_dynamic` calibrates from the tensor itself.
//!
//! NOTE: Burn is pre-1.0 and the quantization API is still moving — pin a
//! version (this targets the burn 0.18.x family) and check the docs:
//! https://burn.dev/docs/burn/tensor/quantization/
//!
//! Cargo.toml:
//!   [dependencies]
//!   burn = { version = "0.18", features = ["ndarray"] }
//!
//! Run:  cargo run --release

use burn::backend::NdArray;
use burn::tensor::Tensor;
use burn::tensor::quantization::{QuantScheme, QuantValue};

type B = NdArray<f32>;

/// Quantize a tensor with the given value type, dequantize, and report the
/// reconstruction MSE — the empirical distortion of README section 3.
fn round_trip<const D: usize>(name: &str, x: &Tensor<B, D>, value: QuantValue) {
    let scheme = QuantScheme::default().with_value(value);

    // Dynamic quantization: min-max calibration is computed from `x` itself,
    // so there is no separate calibration dataset (cf. TurboQuant's data-oblivious goal).
    let q = x.clone().quantize_dynamic(&scheme);
    let recovered = q.dequantize();

    let diff = x.clone().sub(recovered);
    let mse = diff.clone().mul(diff).mean().into_scalar();
    println!("{name:<22} reconstruction MSE = {mse:.3e}");
}

fn main() {
    let device = Default::default();

    // A weight-like tensor: mostly small values plus a couple of outliers,
    // exactly the case where per-tensor INT8 struggles and finer schemes help.
    let data: Vec<f32> = (0..1024)
        .map(|i| {
            let t = i as f32 * 0.01;
            let base = (t).sin() * 0.1 + (t * 0.3).cos() * 0.05;
            if i == 7 { 4.0 } else { base } // outlier inflates the per-tensor scale
        })
        .collect();
    let x = Tensor::<B, 1>::from_floats(data.as_slice(), &device);

    println!("Burn quantization round-trip (1024 weight-like values, one outlier):");
    round_trip("INT8 symmetric (Q8S)", &x, QuantValue::Q8S);
    round_trip("INT4 symmetric (Q4S)", &x, QuantValue::Q4S);
    round_trip("FP8 e4m3       (E4M3)", &x, QuantValue::E4M3);

    // Bytes saved: f32 -> int8 is 4x, f32 -> int4 is ~8x (plus per-block scales).
    println!(
        "\nfootprint: f32={} B, int8≈{} B (4x), int4≈{} B (8x) + scales",
        x.shape().num_elements() * 4,
        x.shape().num_elements(),
        x.shape().num_elements() / 2,
    );
    println!("(int4 with one big outlier shows why per-block QuantLevel beats per-tensor.)");
}
