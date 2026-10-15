//! lenet.rs — LeNet-5 (LeCun, 1998) in Rust with `candle` (README §4-§5).
//!
//! The original CNN that read handwritten digits on bank checks, and the
//! direct ancestor of AlexNet. Same primitives — conv, pooling, fully-connected
//! — in a systems language, to show they aren't Python magic. We use ReLU + max
//! pooling (the modern recipe); the 1998 paper used tanh + average pooling.
//!
//! Architecture (28×28 MNIST input):
//!   conv(1->6, 5x5, pad 2) -> relu -> maxpool 2
//!   conv(6->16, 5x5)       -> relu -> maxpool 2
//!   fc(16*5*5 -> 120) -> relu -> fc(120->84) -> relu -> fc(84->10)
//!
//! candle is pre-1.0 and its API moves; pin a version and check the docs:
//! https://github.com/huggingface/candle
//!
//! Cargo.toml:
//!   [dependencies]
//!   candle-core = "0.8"
//!   candle-nn   = "0.8"
//!
//! Run:
//!   cargo run --release

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{conv2d, linear, Conv2d, Conv2dConfig, Linear, Module, VarBuilder, VarMap};

struct LeNet {
    conv1: Conv2d,
    conv2: Conv2d,
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
}

impl LeNet {
    fn new(vb: VarBuilder) -> Result<Self> {
        let c1 = Conv2dConfig { padding: 2, ..Default::default() }; // keep 28x28
        Ok(Self {
            conv1: conv2d(1, 6, 5, c1, vb.pp("conv1"))?,
            conv2: conv2d(6, 16, 5, Default::default(), vb.pp("conv2"))?,
            fc1: linear(16 * 5 * 5, 120, vb.pp("fc1"))?,
            fc2: linear(120, 84, vb.pp("fc2"))?,
            fc3: linear(84, 10, vb.pp("fc3"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv1.forward(x)?.relu()?.max_pool2d(2)?; // 28 -> 14
        let x = self.conv2.forward(&x)?.relu()?.max_pool2d(2)?; // 14 -> 5 (10->5)
        let x = x.flatten_from(1)?; // (N, 16*5*5)
        let x = self.fc1.forward(&x)?.relu()?;
        let x = self.fc2.forward(&x)?.relu()?;
        self.fc3.forward(&x) // logits (N, 10)
    }
}

fn main() -> Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let net = LeNet::new(vb)?;

    // One batch of 8 fake 1x28x28 "images" to trace the shapes.
    let x = Tensor::zeros((8, 1, 28, 28), DType::F32, &device)?;
    let logits = net.forward(&x)?;
    println!("input  {:?}", x.dims());
    println!("logits {:?}", logits.dims()); // [8, 10]
    Ok(())
}
