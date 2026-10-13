//! The model runner (L10.1): loads a model directory, owns the KV pool, and
//! runs one forward step at a time. It knows nothing about requests,
//! scheduling, or sampling: the scheduler (L10.2) decides what runs, the
//! block manager (L10.4) decides which blocks hold it, and `sample` turns
//! each logits row into a token.
//!
//! The engine-wide configuration lives here too, because the runner is the
//! first piece that reads it (`[engine]` of runtime.toml, DESIGN 2.12).

use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard};

use anyhow::{anyhow, bail, Context, Result};
use candle_core::Device;

use crate::kv::{KvCfg, KvPool, TL_F16, TL_KV_FORMAT_V1};

use crate::forward::{bigram_forward, llama_forward, rope_inv_freq, ForwardBatch, ForwardSeq, Logits};
use crate::model::{load_weights, Arch, ModelConfig, SafeTensors, Weights};

fn engine_device() -> Device {
    // SOLUTION-BEGIN L10.1
    #[cfg(target_os = "macos")]
    {
        // Candle 0.10.2 indexes the device list before returning a Result.
        // Hosted macOS runners can have no Metal device at all.
        if candle_metal_kernels::metal::Device::all().is_empty() {
            return Device::Cpu;
        }
        return Device::new_metal(0).unwrap_or(Device::Cpu);
    }
    #[cfg(not(target_os = "macos"))]
    { Device::Cpu }
    // SOLUTION-END
}

/// `prefix_cache` of runtime.toml (L10.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefixCache {
    None,
    Hash,
    Radix,
}

/// Waiting-queue order (L10.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SchedPolicy {
    /// First come, first served.
    Fcfs,
    /// Higher `priority` first, with aging; ties first come, first served.
    Priority,
}

/// Weight quantization applied at load.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quant {
    /// int4, symmetric, `group` columns per f16 scale (L8.5's scheme).
    Int4 { group: usize },
}

/// The KV pool's size: `blocks` blocks of `block_size` token positions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvConfig {
    pub blocks: usize,
    pub block_size: usize,
}

/// `[engine]` of runtime.toml, the parts the engine library reads.
#[derive(Clone, Debug, PartialEq)]
pub struct EngineConfig {
    pub max_batch_tokens: usize,
    pub max_seqs: usize,
    /// 0: prefill whole prompts (L10.2); otherwise chunks of at most this
    /// many tokens (L10.3).
    pub prefill_chunk: usize,
    pub prefix_cache: PrefixCache,
    pub kv: KvConfig,
    pub policy: SchedPolicy,
    /// Kernel threads; 0 means the kernels' default (serial in this engine).
    pub threads: usize,
    pub quant: Option<Quant>,
}

impl Default for EngineConfig {
    fn default() -> Self {
        // SOLUTION-BEGIN L10.1
        EngineConfig {
            max_batch_tokens: 2048,
            max_seqs: 64,
            prefill_chunk: 0,
            prefix_cache: PrefixCache::None,
            kv: KvConfig { blocks: 512, block_size: 16 },
            policy: SchedPolicy::Fcfs,
            threads: 0,
            quant: None,
        }
        // SOLUTION-END
    }
}

/// The KV pool, shared by the runner (which writes K and V) and the block
/// manager (which allocates and frees blocks). kv_pool.h: one lock.
pub type SharedPool = Arc<Mutex<KvPool>>;

/// A loaded model plus its KV pool.
pub struct ModelRunner {
    cfg: ModelConfig,
    weights: Weights,
    inv_freq: Vec<f32>,
    pool: SharedPool,
    device: Device,
}

/// Locks the pool; a poisoned lock (a panic while holding it) is an error.
pub fn lock(pool: &SharedPool) -> Result<MutexGuard<'_, KvPool>> {
    // SOLUTION-BEGIN L10.1
    pool.lock().map_err(|_| anyhow!("the KV pool lock is poisoned"))
    // SOLUTION-END
}

impl ModelRunner {
    /// Reads `dir` (config.json, model.safetensors), checks the C library's
    /// ABI version, loads the weights (quantizing them when `cfg.quant` asks),
    /// and creates a KV pool of `cfg.kv.blocks` blocks. A bigram has no
    /// layers; its pool has one layer of width 1, so block accounting works
    /// the same for every model.
    pub fn load(dir: &Path, cfg: &EngineConfig) -> Result<ModelRunner> {
        // SOLUTION-BEGIN L10.1
        let mcfg = ModelConfig::load(dir).map_err(|e| anyhow!(e))?;
        let st = SafeTensors::open(&dir.join("model.safetensors")).map_err(|e| anyhow!(e))?;
        let group = cfg.quant.map(|Quant::Int4 { group }| group);
        let weights = load_weights(&st, &mcfg, group).map_err(|e| anyhow!(e))?;
        ModelRunner::from_parts(mcfg, weights, cfg)
        // SOLUTION-END
    }

    /// A runner over weights already in memory (tests build tiny models).
    pub fn from_parts(mcfg: ModelConfig, weights: Weights, cfg: &EngineConfig) -> Result<ModelRunner> {
        // SOLUTION-BEGIN L10.1
        if cfg.kv.blocks == 0 || cfg.kv.block_size == 0 {
            bail!("kv: blocks and block_size must be at least 1");
        }
        let (layers, heads, dim) = match mcfg.arch {
            Arch::Llama => (mcfg.n_layers, mcfg.n_kv_heads, mcfg.head_dim),
            Arch::Bigram => (1, 1, 1),
        };
        let kcfg = KvCfg {
            n_blocks: u32::try_from(cfg.kv.blocks)?,
            block_tokens: u32::try_from(cfg.kv.block_size)?,
            n_layers: u32::try_from(layers.max(1))?,
            n_kv_heads: u32::try_from(heads)?,
            head_dim: u32::try_from(dim)?,
            dtype: TL_F16,
            format: TL_KV_FORMAT_V1,
        };
        let pool = KvPool::new(kcfg).context("create Rust KV cache")?;
        let inv_freq = match mcfg.arch {
            Arch::Llama => rope_inv_freq(mcfg.rope_theta, mcfg.head_dim),
            Arch::Bigram => Vec::new(),
        };
        Ok(ModelRunner { cfg: mcfg, weights, inv_freq, pool: Arc::new(Mutex::new(pool)), device: engine_device() })
        // SOLUTION-END
    }

    /// The model's config.json.
    pub fn config(&self) -> &ModelConfig {
        // SOLUTION-BEGIN L10.1
        &self.cfg
        // SOLUTION-END
    }

    /// The weights.
    pub fn weights(&self) -> &Weights {
        // SOLUTION-BEGIN L10.1
        &self.weights
        // SOLUTION-END
    }

    /// The KV pool (a clone of the shared handle).
    pub fn pool(&self) -> SharedPool {
        // SOLUTION-BEGIN L10.1
        Arc::clone(&self.pool)
        // SOLUTION-END
    }

    /// Token positions per KV block.
    pub fn block_tokens(&self) -> usize {
        // SOLUTION-BEGIN L10.1
        lock(&self.pool).map(|p| p.cfg().block_tokens as usize).unwrap_or(1)
        // SOLUTION-END
    }

    /// One step: the next-token logits of every sequence in `batch`, after
    /// writing their new tokens' K and V into their blocks.
    pub fn forward(&mut self, batch: &ForwardBatch) -> Result<Logits> {
        // SOLUTION-BEGIN L10.1
        if batch.seqs.is_empty() {
            return Ok(Logits { vocab: self.cfg.vocab_size, data: Vec::new() });
        }
        match &self.weights {
            Weights::Bigram(w) => bigram_forward(w, batch, &self.device).map_err(|e| anyhow!(e)),
            Weights::Llama(w) => {
                let mut pool = lock(&self.pool)?;
                llama_forward(w, &self.cfg, &self.inv_freq, &self.device, &mut pool, batch, None).map_err(|e| anyhow!(e))
            }
        }
        // SOLUTION-END
    }

    /// Runs `tokens` as one fresh sequence in temporary blocks, freed
    /// afterwards; returns its final-normed hidden states [T, hidden] and
    /// the logits after its last token. A bigram's hidden state of a token
    /// is its row of W.
    pub fn run_once(&mut self, tokens: &[u32]) -> Result<(Vec<f32>, Vec<f32>)> {
        // SOLUTION-BEGIN L10.1
        if tokens.is_empty() {
            bail!("run_once: no tokens");
        }
        if let Weights::Bigram(w) = &self.weights {
            let mut hidden = Vec::with_capacity(tokens.len() * 256);
            for &t in tokens {
                let t = t as usize;
                if t >= 256 {
                    bail!("run_once: token id {t} outside the byte vocabulary");
                }
                hidden.extend_from_slice(&w[t * 256..(t + 1) * 256]);
            }
            let last = hidden[hidden.len() - 256..].to_vec();
            return Ok((hidden, last));
        }
        let bt = self.block_tokens();
        let blocks = lock(&self.pool)?.alloc(tokens.len().div_ceil(bt)).context("run_once: KV blocks")?;
        let batch = ForwardBatch { seqs: vec![ForwardSeq { tokens, start: 0, blocks: &blocks }] };
        let mut hidden = Vec::new();
        let out = match &self.weights {
            Weights::Llama(w) => {
                let mut pool = lock(&self.pool)?;
                llama_forward(w, &self.cfg, &self.inv_freq, &self.device, &mut pool, &batch, Some(&mut hidden))
            }
            Weights::Bigram(_) => unreachable!("handled above"),
        };
        let mut pool = lock(&self.pool)?;
        for &b in &blocks {
            pool.release(b)?;
        }
        let logits = out.map_err(|e| anyhow!(e))?;
        Ok((hidden, logits.data))
        // SOLUTION-END
    }
}
