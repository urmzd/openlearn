<!-- ol:module L10.1 -->
# Candle model runner (Llama + bigram, int4), Rust KV cache, sampler and PCG32

## Overview

| | |
|---|---|
| **Module** | `L10.1` · build · Rust · Pass 7 · 14 to 20 h |
| **You build** | `rust/crates/tl-engine/src/kv.rs`: the bounded Rust KV cache · `rust/crates/tl-engine/src/quant.rs` (f16, bf16, int4) · `model.rs` (config.json, the memory-mapped safetensors reader, the weights) · `forward.rs` (the Llama and bigram forwards built with Candle tensors) · `runner.rs` (`ModelRunner`, `EngineConfig`, the KV pool) · `sample.rs` (PCG32 and the sampler) · the module lines of the `tl-engine/src/lib.rs` crate root |
| **Contract** | files: [`formats/safetensors.md`](../../../course/contracts/formats/safetensors.md), [`formats/config.schema.json`](../../../course/contracts/formats/config.schema.json), [`formats/kv-block.md`](../../../course/contracts/formats/kv-block.md) · determinism: [`spec/sampling.md`](../../../course/contracts/spec/sampling.md), [`spec/pcg32.md`](../../../course/contracts/spec/pcg32.md) |
| **Tests** | `course/tests/rust/l10_1.rs`, 24 tests (what they check: section 4); parity suites `ol parity sampler rng` (your Rust against your Python, through one golden file) |
| **Needs** | `L10.0` the checkpoint contract ([chapter](00-your-first-endpoint.md)) · reading: `lang.04` Rust ([primer](../../../software-craftsmanship/12-language-and-tool-primers/04-rust.md)), `L8.1` the Python sampler you port, `M06.3` PCG32, `L7.9` the Llama model you port, `L8.5` the int4 scheme, `M09.4` f16 rounding, `L9.7` the Python twin of this runner, `L1.5` tl-tok · or `--ref-deps` |
| **Used by** | `L10.2` (the scheduler runs this runner), `L10.3` (chunked prefill feeds it), `L10.4` (the block manager hands it blocks), `L10.5` (the engine loop and the server) · later: `L10.6`, `L10.8`, `L10.9` |
| **Milestone** | `MS-L10` (the engine's greedy stream equals your Python's; `ol parity sampler`) |
| **Optional depth** | [Rust reference, `Drop`](https://doc.rust-lang.org/reference/destructors.html) (free); [safetensors format](https://github.com/huggingface/safetensors) (free); [Hugging Face `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) (free); [Kwon et al. 2023, PagedAttention](https://arxiv.org/abs/2309.06180) (free) |

## Key Takeaways

- The Rust engine owns the model graph and uses Candle for tensor operations. The learner writes the Llama layers and owns a bounded Rust KV cache.
- `KvPool` owns fixed-size f16 K and V slabs, block references, the prefix hash index, and LRU eviction (`kv_pool_bounds_and_cache_lifecycle`).
- One step runs a batch of sequences, each a run of new tokens at absolute positions; their K and V go into the pool as f16 and are read back for attention, so the logits match Hugging Face's float32 forward within the f16 bound (`tiny_llama_logits_match_hf`).
- The model forward is batch-invariant bit for bit and chunk-invariant within the test tolerance, so batching sequences together or splitting a prompt into steps preserves logits (`batched_forward_equals_single`, `incremental_decode_equals_full_prefill`).
- The sampler follows spec/sampling.md step by step in f64, so the same logits and seed give your Python's id and logprob exactly (`sampler_matches_l81_golden`, `hand_example_sampling`).

## How to work this chapter

```bash
ol start L10.1               # stubs tl-engine kv, quant, model, forward, runner, sample
ol tests L10.1               # read the test catalog first: rung R0, you write no graded tests here
ol check L10.1               # exit code is the verdict
ol check L10.1 --ref-deps    # only if a referenced module is not passing yet
ol parity sampler rng        # your Rust sampler and PCG32 against your Python, through the golden files
```

`ol start` never rewrites your files. Add one `pub mod` line per new file to `tl-engine/src/lib.rs` (yours since `L8.4`). Add Candle, `anyhow`, `memmap2`, and `serde_json` from `contracts/allowed-deps.toml` to `tl-engine/Cargo.toml`.

---

## 1. Why now

Your tracer engine (`L10.0`) serves one model, a byte bigram, with one request per thread and a sampler that is not the one your Python uses. Pass 7 turns it into a real inference engine, and everything after it in this part (scheduling, chunked prefill, prefix caching, the OpenAI server) needs one thing first: a component that loads a Llama checkpoint, runs a batch of sequences through it with Candle tensor operations, keeps each sequence's past in the Rust KV pool, and turns logits into tokens exactly as your Python does. That component is the model runner. Rust owns the graph and the production math path. Candle supplies tensor operations, while this module implements the layers and cache.

## 2. Principles

### 2.1 Three layers in one engine

```text
tl-engine runner.rs    ModelRunner::forward(&ForwardBatch) -> Logits
          forward.rs   llama_forward: embedding, per layer [norm, q k v, rope, write KV, attention, o, norm, swiglu], norm, LM head
          model.rs     config.json, model.safetensors (mmap), Weights
          quant.rs     f16 / bf16 / int4 numbers
          sample.rs    Pcg32, SamplingParams, sample()
candle-core + nn       tensor operations: matmul, embedding, RMSNorm, softmax
kv.rs                   Rust-owned f16 blocks, references, hash index, LRU
```

### 2.2 Candle tensors and Rust-owned state

Candle provides tensor storage, device placement, matrix multiplication, softmax, RMSNorm, and embedding lookup. The engine uses `candle-core` and `candle-nn` 0.10.2. The learner still writes the model structure: the layer order, RoPE positions, grouped-query attention, residual connections, SwiGLU, and final norm. CI uses the CPU device; macOS can select Metal.

The KV cache is a separate Rust data structure. `KvPool` allocates fixed-size K and V slabs as f16 values, tracks references, and keeps released full blocks in an LRU cache when prefix caching is enabled. Its methods return `Result` for invalid dimensions, exhausted capacity, invalid block ids, and format errors. No C library or foreign-function boundary is part of the model runner.

### 2.3 The model directory

`config.json` says what to build (`tl_arch`: `bigram` or `llama`; `tl_tokenizer`; the shape fields of Hugging Face's Llama config). Anything this engine cannot run correctly, a scaled RoPE (`rope_type` other than `default`), a sliding window, attention biases, is refused at load with a message, never served wrong.

`model.safetensors` is read through a **memory map** (`memmap2`): the operating system maps the file into the address space and pages bytes in on first touch, so a 270 MB checkpoint is not copied before the first request. The header rules are those of `L10.0`: a u64 little-endian header length $N$, $N$ bytes of JSON, then the data buffer, with every `data_offsets` range counted from the start of the **data buffer** and the ranges tiling it exactly.

Weights arrive as F32, F16, or BF16 and are widened to f32 at load:

$$\text{bf16}(h) = \text{f32 with bits } h \ll 16, \qquad \text{f16}(h) = (-1)^{s}\, 2^{e - 15}\,\big(1 + \tfrac{m}{1024}\big) \text{ for } 0 < e < 31,$$

with subnormals $(-1)^s\, 2^{-14}\, \tfrac{m}{1024}$ for $e = 0$. The reverse direction (f32 to f16, for the KV cache) **rounds to nearest, ties to even**: keep the top 10 mantissa bits, and add one when the dropped bits are above half, or exactly half with the kept value odd.

### 2.4 The Llama forward, one step

| Symbol | Meaning | Type / shape |
|---|---|---|
| $S$ | sequences in the step | integer |
| $n_s$, $p_s$ | new tokens of sequence $s$ and the absolute position of its first one | integers |
| $N = \sum_s n_s$ | tokens in the step | integer |
| $d$ | hidden size (`hidden_size`) | integer |
| $H$, $H_{kv}$, $D$ | query heads, key/value heads, head size | integers |
| $x \in \mathbb{R}^{N \times d}$ | hidden states of every new token, in batch order | `f32` |
| $W_q, W_k, W_v, W_o$ | attention projections, stored `[out, in]` | `f32` or int4 |
| $\mathrm{pos}(t)$ | absolute position of token $t$: $p_s + i$ for the $i$-th new token of $s$ | `i32` |
| $\epsilon$ | `rms_norm_eps` | `f32` |
| $B$ | token positions per KV block (`block_size`) | integer |

Each step computes, for every layer:

$$h = \mathrm{RMSNorm}(x) = \frac{x}{\sqrt{\tfrac{1}{d}\sum_i x_i^2 + \epsilon}} \odot w, \qquad q = h W_q^\top,\; k = h W_k^\top,\; v = h W_v^\top$$

then rotates $q$ and $k$ by RoPE at $\mathrm{pos}(t)$ (half-split layout), writes each token's $k$ and $v$ into its sequence's blocks (position $j$ lives in block $\lfloor j / B \rfloor$ of the block table, slot $j \bmod B$), and for each sequence gathers its keys and values for positions $0$ to $p_s + n_s - 1$ and uses Candle matrix multiplication and softmax with `q_offset` $= p_s$ and a causal mask:

$$o_t = \sum_{j \le \mathrm{pos}(t)} \mathrm{softmax}_j\!\Big(\tfrac{q_t \cdot k_j}{\sqrt{D}}\Big)\, v_j, \qquad x \mathrel{+}= o W_o^\top, \qquad x \mathrel{+}= \big(\mathrm{silu}(h_2 W_g^\top) \odot (h_2 W_u^\top)\big) W_d^\top$$

with $h_2 = \mathrm{RMSNorm}(x)$ before the MLP. After the last layer, only each sequence's **last** token goes through the final norm and the LM head (the embedding matrix itself when `tie_word_embeddings` is true), giving one row of logits per sequence.

Two properties carry the rest of this part. **Batch invariance**: a token's logits must not depend on which other tokens share its step. Candle picks its reduction path from the shape: a one-row product takes a matrix-vector kernel and a taller one a blocked kernel that adds in another order, so on x86 the same row comes out a few ulps apart depending on the batch. The runner therefore sends every row through the same $[1, K] \times [K, N]$ product (`matmul_rows`), and query $t$ attends over exactly its visible keys $j \le \mathrm{pos}(t)$ instead of a masked full row. Then a row is computed by the same operations in the same order whether it runs alone or in a batch, and `batched_forward_equals_single` compares bit for bit. **Chunk invariance**: positions are absolute and the cache holds the same f16 keys either way, so prefilling a prompt in one step or in pieces agrees within the tested tolerance. Without row-wise products this fails on Linux, not just in the last bits: a one-ulp change in a key can round to a different f16, which moves the logits by about $2 \times 10^{-3}$.

**The f16 bound.** K and V pass through f16, which rounds with relative error at most $2^{-11} \approx 4.9 \times 10^{-4}$. Through two layers this perturbs the logits of the tiny test model (magnitudes up to about 20) by about $7 \times 10^{-3}$, measured; the tests allow $10^{-3}$ of the largest logit, $2 \times 10^{-2}$. Your Python reference (`L7.9`) keeps f32 keys, so this bound, not 1e-4, is what an f16 cache can promise.

### 2.5 Int4 weights

A quantized linear weight $W \in \mathbb{R}^{\text{out} \times \text{in}}$ with groups of $g$ columns stores, per group, one f16 scale and $g$ four-bit integers (`L8.5`, `formats/safetensors.md`):

$$s = \mathrm{f16}\Big(\frac{\max |W_{\text{group}}|}{7}\Big), \qquad q = \mathrm{clamp}\big(\mathrm{rint}(W / s), -8, 7\big), \qquad \hat W = q\, s$$

with `rint` rounding half to even and $s$ the f16 value **actually stored**, so $|W - \hat W| \le s/2$. Two values share a byte: column $2b$ in the low nibble, $2b + 1$ in the high nibble, each in 4-bit two's complement. `QLinear::forward` unpacks the quantized values, widens the effective weight to f32, and uses Candle matrix multiplication. The runner loads int4 two ways: from an int4 file (`<name>.qweight`, `<name>.scales`, metadata `quant = "int4-g<g>-sym"`), or by quantizing f32 weights at load (`EngineConfig.quant = Some(Quant::Int4 { group })`).

### 2.6 The sampler and PCG32

`sample.rs` ports your `L8.1` sampler and `M06.3` generator, and **parity** means bit-identical: the same token id and the same f64 logprob from the same logits and seed. spec/sampling.md fixes the order: widen to f64; repetition penalty over the distinct ids of prompt and output ($\ell \mathrel{/}= r$ when positive, $\ell \mathrel{*}= r$ otherwise); presence and frequency over the output only ($\ell \mathrel{-}= a_f c + a_p$); the **logprob** is $\log\mathrm{softmax}(\ell)$ at this point; greedy ($T = 0$) takes the argmax with ties to the lowest id and **no draw**; otherwise divide by $T$, keep the top $k$ by $(\ell \text{ desc}, \text{id asc})$, keep the nucleus up to and including the token that reaches $p$, keep ids with $q_i \ge m \cdot \max q$, softmax over the kept ids in ascending id order, draw one $u$, and walk the cumulative sum in ascending id order to the first id with $u < c$. Every sum is a left-to-right loop, so Rust and Python add the same numbers in the same order.

Each request gets its own generator, `stream(seed, sample)`: $\mathrm{child\_seed}(s, 4) = \mathrm{mix64}(s + 4 \cdot \texttt{0x9E3779B97F4A7C15})$, then `pcg32_srandom_r(child_seed, 4)`. One uniform takes two outputs $a, b$: $u = \big((a \gg 5)\, 2^{26} + (b \gg 6)\big)\, 2^{-53}$.

## 3. Worked example by hand

**One sampled token** (spec/sampling.md, `hand_example_sampling`). Logits $x = [1, 3, 2, 3, -1]$, no history, $T = 1$, top-k 3, top-p 0.8, seed 0. Logprobs first: $M = 3$, $Z = e^{-2} + 1 + e^{-1} + 1 + e^{-4} = 2.52153$, so ids 1 and 3 have logprob $-\ln Z = -0.92487$. Top-k: order by (logit desc, id asc) is $1, 3, 2, 0, 4$; keep $\{1, 3, 2\}$. Top-p over the kept: $Z' = 1 + e^{-1} + 1 = 2.36788$, $q_1 = q_3 = 0.42232$, $q_2 = 0.15536$; walking $1, 3$: $0.42232$, then $0.84464 \ge 0.8$, so keep $\{1, 3\}$ and renormalize to $0.5, 0.5$. The generator: $\mathrm{child\_seed}(0, 4) = \texttt{0xF88BB8A8724C81EC}$, and the first uniform of `stream(0, sample)` is $u = 0.80209$. Walk ids in ascending order: after id 1, $c = 0.5$, not above $u$; after id 3, $c = 1.0 > u$: the token is **id 3**, logprob $-0.92487$.

**Int4** (`int4_hand_example`). Weights $[0.7, -1.4, 0, 0.29995]$ in one group of 4: $\max|W| = 1.4$, $1.4 / 7 = 0.2$, and $\mathrm{f16}(0.2) = 0.199951171875$ (0.2 is not a binary fraction; f16 keeps 10 mantissa bits). Dividing by the stored scale: $0.7 / 0.19995 = 3.5009 \to 4$, $-1.4 / 0.19995 = -7.0017 \to -7$, $0 \to 0$, $0.29995 / 0.19995 = 1.5001 \to 2$. With the unrounded $0.2$ the last would be $1.49975 \to 1$: the stored scale changes a value. Packing $q = [-8, 7, 1, -1]$: $-8$ is $1000_2$ and $7$ is $0111_2$, so byte 0 is $0111\,1000_2 = \texttt{0x78}$; $1 = 0001_2$ and $-1 = 1111_2$ give $1111\,0001_2 = \texttt{0xF1}$.

**f16 rounding** (`f16_rounds_to_nearest_even`). $1 + 2^{-11}$ lies exactly halfway between $1$ (`0x3C00`) and $1 + 2^{-10}$ (`0x3C01`); ties go to the even mantissa, so the result is `0x3C00`. $1 + 3 \cdot 2^{-11}$ lies halfway between `0x3C01` and `0x3C02`: the even one is `0x3C02`.

**Where a key lands.** Blocks of $B = 16$, a sequence with block table $[7, 2, 9]$: position 37 is block $\lfloor 37 / 16 \rfloor = 2$ of the table, pool block 9, slot $37 \bmod 16 = 5$. Its K for KV head $h$ starts at element $(h \cdot 16 + 5) \cdot D$ of layer $l$'s K slab of block 9.

## 4. The interface

```rust
// rust/crates/tl-engine/src/kv.rs
pub struct KvCfg { pub n_blocks: u32, pub block_tokens: u32, pub n_layers: u32,
                   pub n_kv_heads: u32, pub head_dim: u32, pub dtype: i32, pub format: u32 }
pub struct KvPool { /* Rust-owned K/V slabs, references, hash index, LRU */ }
impl KvPool {
    pub fn new(cfg: KvCfg) -> Result<KvPool, KvError>;
    pub fn alloc(&mut self, n: usize) -> Result<Vec<u32>, KvError>;
    pub fn retain(&mut self, id: u32) -> Result<(), KvError>;
    pub fn release(&mut self, id: u32) -> Result<(), KvError>;
    pub fn set_fill(&mut self, id: u32, n: u32) -> Result<(), KvError>;
    pub fn slab(&self, id: u32, layer: u32, is_v: bool) -> Result<&[u16], KvError>;
    pub fn slab_mut(&mut self, id: u32, layer: u32, is_v: bool) -> Result<&mut [u16], KvError>;
    pub fn export(&self, ids: &[u32]) -> Result<Vec<u8>, KvError>;
    pub fn import(&mut self, buf: &[u8]) -> Result<Vec<u32>, KvError>;
}
```

```rust
// rust/crates/tl-engine/src/{quant,model,forward,runner,sample}.rs
pub fn f16_to_f32(h: u16) -> f32;  pub fn f32_to_f16(x: f32) -> u16;  pub fn bf16_to_f32(h: u16) -> f32;
pub fn pack_int4(q: &[i8], rows: usize, cols: usize) -> Result<Vec<u8>, String>;  pub fn unpack_int4(packed: &[u8]) -> Vec<i8>;
pub struct QLinear { pub out: usize, pub inp: usize, pub group: usize, pub qweight: Vec<u8>, pub scales: Vec<u16> }
impl QLinear { pub fn quantize(w: &[f32], out: usize, inp: usize, group: usize) -> Result<QLinear, String>;
               pub fn dequantize(&self) -> Vec<f32>;  pub fn forward(&self, x: &[f32], m: usize, y: &mut [f32]) -> Result<(), TlError>; }

pub struct ModelConfig { pub arch: Arch, pub tokenizer: TokenizerKind, pub vocab_size: usize, pub hidden_size: usize, /* ... */ pub eos_token_ids: Vec<u32> }
impl ModelConfig { pub fn from_json(text: &str) -> Result<ModelConfig, String>;  pub fn load(dir: &Path) -> Result<ModelConfig, String>; }
pub fn parse_layout(bytes: &[u8]) -> Result<Layout, String>;
pub struct SafeTensors { pub layout: Layout /* + the map */ }
impl SafeTensors { pub fn open(path: &Path) -> Result<SafeTensors, String>;  pub fn f32(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>, String>;
                   pub fn q4(&self, name: &str, out: usize, inp: usize, group: usize) -> Result<QLinear, String>; }
pub enum Linear { F32 { w: Vec<f32>, out: usize, inp: usize }, Q4(QLinear) }   // forward: y = x @ W^T
pub enum Weights { Bigram(Vec<f32>), Llama(LlamaWeights) }

pub struct ForwardSeq<'a> { pub tokens: &'a [u32], pub start: usize, pub blocks: &'a [u32] }
pub struct ForwardBatch<'a> { pub seqs: Vec<ForwardSeq<'a>> }
pub struct Logits { pub vocab: usize, pub data: Vec<f32> }   // rows(), row(i): one per sequence

pub struct EngineConfig { pub max_batch_tokens: usize, pub max_seqs: usize, pub prefill_chunk: usize, pub prefix_cache: PrefixCache,
                          pub kv: KvConfig, pub policy: SchedPolicy, pub threads: usize, pub quant: Option<Quant> }
pub type SharedPool = Arc<Mutex<KvPool>>;
impl ModelRunner {
    pub fn load(dir: &Path, cfg: &EngineConfig) -> anyhow::Result<Self>;
    pub fn forward(&mut self, batch: &ForwardBatch) -> anyhow::Result<Logits>;
    pub fn run_once(&mut self, tokens: &[u32]) -> anyhow::Result<(Vec<f32>, Vec<f32>)>;   // hidden [T, d], last logits
    pub fn pool(&self) -> SharedPool;  pub fn block_tokens(&self) -> usize;  pub fn config(&self) -> &ModelConfig;
}

pub struct Pcg32 { /* state, inc */ }
impl Pcg32 { pub fn new(seed: u64, seq: u64) -> Pcg32;  pub fn next_u32(&mut self) -> u32;  pub fn uniform_f64(&mut self) -> f64; }
pub fn child_seed(seed: u64, purpose: u64) -> u64;  pub fn stream(seed: u64, purpose: u64) -> Pcg32;   // PURPOSE_SAMPLE = 4
pub struct SamplingParams { pub temperature: f64, pub top_k: usize, pub top_p: f64, pub min_p: f64,
                            pub repetition_penalty: f64, pub presence_penalty: f64, pub frequency_penalty: f64 }
pub fn sample(logits: &[f32], p: &SamplingParams, prompt: &[u32], output: &[u32], rng: &mut Pcg32) -> (u32, f64);
```

`forward` writes K and V for the new tokens of each sequence into the blocks its table names; the caller allocates the blocks (`L10.2` and `L10.4` do from here on). `run_once` takes temporary blocks and gives them back, for embeddings and tests.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `hand_example_sampling` | unit | section 3: top-k 3 and top-p 0.8 keep ids 1 and 3, $u = 0.80209$, token 3 with logprob $-0.92487$ | the spec's worked example, end to end |
| `pcg32_reference_vectors` | golden | `pcg32(42)` demo line, `uniform_f64`, `child_seed`, `stream` outputs from `pcg32.vectors.json` | one generator in every language (D10) |
| `sampler_matches_l81_golden` | differential, golden | 12 cases x 12 tokens: ids and f64 logprobs bit for bit against L8.1's golden | `ol parity sampler`; the engine's seeded streams equal your Python's |
| `greedy_takes_no_draw_and_sampling_takes_one` | property | greedy leaves the generator untouched; a sampled token takes one `uniform_f64` | draw accounting for the disaggregated hand-off (L10.6) |
| `top_p_keeps_the_crossing_token` | boundary | mass exactly at $p$ keeps the crossing id | nucleus edge, the most common sampler bug |
| `penalties_follow_hf_and_openai_semantics` | unit | repetition divides positives and multiplies negatives over prompt and output; presence and frequency over output only | the OpenAI request fields mean what clients expect |
| `seeded_sampling_matches_the_distribution` | statistical | 4000 draws fit 0.4, 0.3, 0.2, 0.1 (chi-square below 16.27); a seed repeats | the sampler draws from the right distribution |
| `kv_pool_bounds_and_cache_lifecycle` | boundary | capacity, references, registration, release, and lookup preserve pool invariants | block manager and prefix cache rely on these transitions |
| `kv_pool_evicts_the_oldest_cached_block` | boundary | exhausted allocation evicts the oldest cached prefix | cache policy preserves recent prefixes |
| `kv_pool_rejects_zero_capacity` | boundary | zero blocks is rejected during construction | invalid engine configuration fails early |
| `kv_pool_exports_and_imports_rust_owned_blocks` | differential | f16 cache data survives the transfer envelope round trip | disaggregated decode resumes the same state |
| `kv_pool_bounds_and_cache_lifecycle` | boundary | capacity is all-or-nothing, refs balance, full blocks register, release caches, lookup reacquires | scheduler and prefix cache share one pool |
| `kv_pool_exports_and_imports_rust_owned_blocks` | differential | a written f16 value survives an export/import round trip | disaggregated decode resumes with the same cache |
| `f16_rounds_to_nearest_even` | boundary | ties, overflow, subnormals, NaN; every f16 pattern round-trips | the KV cache stores f16 |
| `int4_hand_example` | unit | section 3: nibble order and the stored-scale rule | int4 files written by your Python load here |
| `q4_linear_matches_its_dequantized_weights` | differential | Candle output equals x times the dequantized weights within float tolerance | the packed layout means what the format says |
| `safetensors_reader_rules` | boundary | offsets from the data buffer, F16 and BF16 widening, gaps and trailing bytes refused | every checkpoint goes through this reader |
| `config_refuses_what_the_engine_cannot_run` | boundary | scaled RoPE, sliding window, unknown arch, missing tokenizer refused at load | wrong models fail loudly at start |
| `bigram_checkpoint_still_serves` | conformance | the tracer checkpoint loads and greedy continues `bcd` | the Pass 1 smoke stays green (D32) |
| `tiny_llama_logits_match_hf` | golden | last-token logits within $2 \times 10^{-2}$ of HF's float32 forward | the whole graph: GQA, tied embeddings, BF16 file |
| `tiny_llama_greedy_matches_hf` | golden | 32 greedy tokens per prompt equal HF's under the near-tie rule | what MS-L10 compares with your Python |
| `incremental_decode_equals_full_prefill` | differential | prefill then one token per step equals one prefill, within floating-point tolerance | chunked prefill and preemption by recompute |
| `batched_forward_equals_single` | differential | two sequences in one step agree with their alone logits within float tolerance | continuous batching changes nothing |
| `run_once_returns_its_blocks` | unit | five one-off forwards leave every block free | embeddings do not leak KV |
| `int4_runner_matches_its_dequantized_model` | differential | quantize-at-load equals an int4 file bit for bit and its dequantized f32 twin closely | both int4 paths agree |
| `forward_refuses_bad_batches` | boundary | a short block table or a position past the context is `Err` | the scheduler's mistakes surface, not corrupt KV |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| Swapping the shifts of the two draws in `uniform_f64` | uniforms that look fine and match nothing | `pcg32_reference_vectors` (mutant `s01`) |
| Seeding the sample stream on sequence 54 instead of the purpose id | every seeded stream differs from Python's | `hand_example_sampling` (mutant `s02`) |
| Taking the logprob after temperature | logprobs change with $T$; OpenAI clients get wrong values | `sampler_matches_l81_golden` (mutant `s03`) |
| `>` instead of `>=` in the nucleus walk | the token that reaches $p$ is dropped | `top_p_keeps_the_crossing_token` (mutant `s04`) |
| Walking the CDF in probability order | correct distribution, different ids for the same seed | `sampler_matches_l81_golden` (mutant `s05`) |
| Presence and frequency penalties over the prompt | prompt words become unlikely in the answer | `penalties_follow_hf_and_openai_semantics` (mutant `s06`) |
| Dividing negative logits by the repetition penalty | repeated unlikely tokens become MORE likely | `penalties_follow_hf_and_openai_semantics` (mutant `s07`) |
| Truncating instead of rounding to f16 | a drift of half an ulp per cached value | `f16_rounds_to_nearest_even` (mutant `s08`) |
| The odd column in the low nibble | int4 files from your Python decode to noise | `int4_hand_example` (mutant `s09`) |
| Rounding int4 with the f32 scale, storing the f16 one | some weights off by one level | `int4_hand_example` (mutant `s10`) |
| `data_offsets` from the start of the file | garbage weights | `safetensors_reader_rules` (mutant `s11`) |
| Reading BF16 as F16 | SmolLM2 weights come out wildly wrong | `safetensors_reader_rules` (mutant `s12`) |
| RoPE positions relative to the step | decode tokens all at position 0; output degrades after the prompt | `incremental_decode_equals_full_prefill` (mutant `s13`) |
| `q_offset` 0 for a decode step | the new token sees only key 0 | `incremental_decode_equals_full_prefill` (mutant `s14`) |
| K written to the V slab | logits far from HF | `tiny_llama_logits_match_hf` (mutant `s15`) |
| Skipping the final norm | logits scaled wrong | `tiny_llama_logits_match_hf` (mutant `s16`) |
| Not giving temporary blocks back | the pool drains a little per embedding call | `run_once_returns_its_blocks` (mutant `s17`) |

| Accepting a zero-block pool | the first allocation fails far from the invalid configuration | `kv_pool_rejects_zero_capacity` (mutant `s19`) |
| Evicting the newest prefix instead of the oldest | a useful recent prefix disappears early | `kv_pool_evicts_the_oldest_cached_block` (mutant `s18`) |
| Reversing RoPE frequency order | Llama logits differ from the reference | `tiny_llama_logits_match_hf` (mutant `s20`) |
| Not checking token ids against the vocabulary | an invalid id reaches the embedding lookup | `forward_refuses_bad_batches` (mutant `s21`) |
| `silu` applied to the up projection | close-looking, wrong logits | `tiny_llama_logits_match_hf` (mutant `s22`) |

## 6. Where it's used next
| Forward | `L10.6` | Registered call site uses this module. |
| Forward | `L10.8` | Registered call site uses this module. |
| Forward | `L10.9` | Registered call site uses this module. |

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L10.0` | checkpoint loading and the byte bigram tracer |
| Back | `L7.9`, `L8.5`, `M09.4` | the model graph, int4 layout, and f16 conversions reimplemented here |
| Back | `L8.1`, `M06.3` | the sampler and generator this one reproduces bit for bit |
| Back | `L7.9` | the Llama graph this one reproduces |
| Forward | `L10.2` | the scheduler forms each step's `ForwardBatch` |
| Forward | `L10.3` | chunked prefill relies on chunk invariance |
| Forward | `L10.4` | the block manager owns the pool's blocks; this runner writes into them |
| Forward | `L10.5` | the engine loop samples with `sample` and a per-request `stream(seed, sample)` |

If you skip this module, the engine has no model to run past the tracer bigram.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| a gather of K and V per step | [vLLM PagedAttention](https://github.com/vllm-project/vllm) | attention reads the blocks in place, no copy | `vllm/attention/` |
| f16 KV | fp8 KV with per-head scales | half the memory, a calibrated error | `craft.13`, `formats/kv-block.md` v2 |
| int4 groups of 32 | [GPTQ](https://arxiv.org/abs/2210.17323), [AWQ](https://arxiv.org/abs/2306.00978) | error-aware rounding, activation-aware scales | `L8.5` going further |
| CPU tensor operations | [candle](https://github.com/huggingface/candle) Metal backend | device execution with the same layer graph | Candle documentation |
