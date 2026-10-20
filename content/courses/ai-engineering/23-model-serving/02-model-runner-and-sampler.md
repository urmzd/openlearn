# Candle model runner (Llama + bigram, int4), Rust KV cache, sampler and PCG32

The tracer engine from [Your first endpoint](01-your-first-endpoint.md) serves one model, a byte bigram, with one request per thread and a sampler that is not the one the Python stack uses. This section turns it into a real inference engine. Everything after it in this chapter (scheduling, chunked prefill, prefix caching, the OpenAI server) needs one thing first: a component that loads a Llama checkpoint, runs a batch of sequences through it with Candle tensor operations, keeps each sequence's past in a Rust KV pool, and turns logits into tokens exactly as the Python stack does. That component is the **model runner**. Rust owns the graph and the production math path; Candle supplies tensor operations, and this section implements the layers and the cache. The goal it serves is simple to state: the engine's greedy stream must equal the Python stack's.

You will port several things you built earlier in the book: the Llama model from [The Llama-Family Model: Config, Weight Loading, and Download](../13-decoder-block/09-llama-family-model.md), the sampler from [Sampling and logit processors](../21-inference/01-sampling-and-logit-processors.md), and the int4 scheme from [Quantization](../21-inference/05-quantization.md). The generator comes from [Modular arithmetic, hashing, and PCG32 in Python](../../mathematics/12-discrete-math-2/03-modular-arithmetic-hashing-and-pcg32.md) and the f16 rounding rules from [FP8 and microscaling formats](../../mathematics/15-numerical-methods-and-floating-point/04-fp8-and-microscaling-formats.md), both in *Mathematics*. You should also be comfortable with the [Rust primer](../../programming-and-algorithms/01-language-primers/04-rust.md) in *Programming and Algorithms*, and the tokenizer comes from [Rust fast BPE](../07-tokenizers/05-rust-fast-bpe.md). There is also a Python twin of this runner that you can read alongside it.

The contracts are the file formats [`formats/safetensors.md`](../../../../course/contracts/formats/safetensors.md), [`formats/config.schema.json`](../../../../course/contracts/formats/config.schema.json), and [`formats/kv-block.md`](../../../../course/contracts/formats/kv-block.md), and the determinism specs [`spec/sampling.md`](../../../../course/contracts/spec/sampling.md) and [`spec/pcg32.md`](../../../../course/contracts/spec/pcg32.md). The solution lives in six files under `rust/crates/tl-engine/src/`: [`kv.rs`](../../../../course/solutions/rust/crates/tl-engine/src/kv.rs) (`KvPool`, the bounded Rust KV cache), [`quant.rs`](../../../../course/solutions/rust/crates/tl-engine/src/quant.rs) (`f16_to_f32`, `f32_to_f16`, `bf16_to_f32`, `pack_int4`, `QLinear`), [`model.rs`](../../../../course/solutions/rust/crates/tl-engine/src/model.rs) (`ModelConfig`, the memory-mapped `SafeTensors` reader, `Linear`, `matmul_rows`, the weights), [`forward.rs`](../../../../course/solutions/rust/crates/tl-engine/src/forward.rs) (`llama_forward`, `bigram_forward`, `write_kv`, `gather_kv`, built with Candle tensors), [`runner.rs`](../../../../course/solutions/rust/crates/tl-engine/src/runner.rs) (`ModelRunner`, `EngineConfig`), and [`sample.rs`](../../../../course/solutions/rust/crates/tl-engine/src/sample.rs) (`Pcg32`, `stream`, `sample`). The crate root `tl-engine/src/lib.rs` has one `pub mod` line per file, and `tl-engine/Cargo.toml` depends on Candle, `anyhow`, `memmap2`, and `serde_json` (all listed in `contracts/allowed-deps.toml`).

> **Learning objectives**
> - Describe the runner's layers and how a step's batch of sequences flows through a Llama forward with absolute positions and a block-addressed KV pool.
> - Explain why batch invariance requires one-row matrix products and per-query key slices, and state the f16 error bound an f16 KV cache can promise.
> - Quantize, pack, and dequantize int4 weights with the stored f16 scale, and round f32 to f16 with ties to even.
> - Port the sampler so that the same logits and seed give the Python sampler's id and f64 logprob bit for bit.
> - Implement a bounded KV pool with explicit free, used, and cached states and all-or-nothing allocation.

## Three layers in one engine

The engine is organized in layers, and it helps to keep the stack in view while reading the rest of the section:

```text
tl-engine runner.rs    ModelRunner::forward(&ForwardBatch) -> Logits
          forward.rs   llama_forward: embedding, per layer [norm, q k v, rope, write KV, attention, o, norm, swiglu], norm, LM head
          model.rs     config.json, model.safetensors (mmap), Weights
          quant.rs     f16 / bf16 / int4 numbers
          sample.rs    Pcg32, SamplingParams, sample()
candle-core + nn       tensor operations: matmul, embedding, RMSNorm, softmax
kv.rs                   Rust-owned f16 blocks, references, hash index, LRU
```

Candle provides tensor storage, device placement, matrix multiplication, softmax, RMSNorm, and embedding lookup; the engine uses `candle-core` and `candle-nn` 0.10.2. The solution still writes the model structure itself: the layer order, RoPE positions, grouped-query attention, residual connections, SwiGLU, and final norm. CI uses the CPU device, and macOS can select Metal. Enumerate devices with the matching `candle-metal-kernels` version before selecting GPU 0, because a macOS host without a Metal device must fall back to CPU.

The KV cache is a separate Rust data structure. `KvPool` allocates fixed-size K and V slabs as f16 values, tracks references, keeps the prefix hash index, and keeps released full blocks in an LRU cache when prefix caching is enabled. Its methods return `Result` for invalid dimensions, exhausted capacity, invalid block ids, and format errors. No C library or foreign-function boundary is part of the model runner.

## The model directory

Before anything runs, the runner has to decide what to build. `config.json` says so: `tl_arch` (`bigram` or `llama`), `tl_tokenizer`, and the shape fields of Hugging Face's Llama config. Anything this engine cannot run correctly (a scaled RoPE with `rope_type` other than `default`, a sliding window, attention biases) is refused at load with a message, never served wrong.

`model.safetensors` is read through a **memory map** (`memmap2`): the operating system maps the file into the address space and pages bytes in on first touch, so a 270 MB checkpoint is not copied before the first request. The header rules are those of the tracer engine: a u64 little-endian header length $N$, $N$ bytes of JSON, then the data buffer, with every `data_offsets` range counted from the start of the **data buffer** and the ranges tiling it exactly.

Weights arrive as F32, F16, or BF16 and are widened to f32 at load:

$$\text{bf16}(h) = \text{f32 with bits } h \ll 16, \qquad \text{f16}(h) = (-1)^{s}\, 2^{e - 15}\,\big(1 + \tfrac{m}{1024}\big) \text{ for } 0 < e < 31,$$

with subnormals $(-1)^s\, 2^{-14}\, \tfrac{m}{1024}$ for $e = 0$. The reverse direction, f32 to f16 for the KV cache, **rounds to nearest, ties to even**: keep the top 10 mantissa bits, and add one when the dropped bits are above half, or exactly half with the kept value odd.

## The Llama forward, one step

With weights loaded, the runner's job is to execute one step. The notation:

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

It then rotates $q$ and $k$ by RoPE at $\mathrm{pos}(t)$ (half-split layout) and writes each token's $k$ and $v$ into its sequence's blocks: position $j$ lives in block $\lfloor j / B \rfloor$ of the block table, slot $j \bmod B$. For each sequence it gathers keys and values for positions $0$ to $p_s + n_s - 1$ and uses Candle matrix multiplication and softmax with `q_offset` $= p_s$ and a causal mask:

$$o_t = \sum_{j \le \mathrm{pos}(t)} \mathrm{softmax}_j\!\Big(\tfrac{q_t \cdot k_j}{\sqrt{D}}\Big)\, v_j, \qquad x \mathrel{+}= o W_o^\top, \qquad x \mathrel{+}= \big(\mathrm{silu}(h_2 W_g^\top) \odot (h_2 W_u^\top)\big) W_d^\top$$

with $h_2 = \mathrm{RMSNorm}(x)$ before the MLP. After the last layer, only each sequence's **last** token goes through the final norm and the LM head (the embedding matrix itself when `tie_word_embeddings` is true), giving one row of logits per sequence.

## Batch invariance, chunk invariance, and the f16 bound

Two properties carry the rest of this chapter, and both are easy to lose by accident.

**Batch invariance** says a token's logits must not depend on which other tokens share its step. Candle picks its reduction path from the shape: a one-row product takes a matrix-vector kernel and a taller one a blocked kernel that adds in another order, so on x86 the same row comes out a few ulps apart depending on the batch. The runner therefore sends every row through the same $[1, K] \times [K, N]$ product (`matmul_rows`), and query $t$ attends over exactly its visible keys $j \le \mathrm{pos}(t)$ instead of a masked full row. A row is then computed by the same operations in the same order whether it runs alone or in a batch, and the batched and single logits agree bit for bit.

**Chunk invariance** says prefilling a prompt in one step or in pieces gives the same answer. Positions are absolute and the cache holds the same f16 keys either way, so the two agree within the tested tolerance. Without row-wise products this fails on Linux, and not just in the last bits: a one-ulp change in a key can round to a different f16, which moves the logits by about $2 \times 10^{-3}$.

**The f16 bound** is the price of the cache. K and V pass through f16, which rounds with relative error at most $2^{-11} \approx 4.9 \times 10^{-4}$. Through two layers this perturbs the logits of the tiny test model (magnitudes up to about 20) by about $7 \times 10^{-3}$, measured; the Hugging Face comparison allows $10^{-3}$ of the largest logit, $2 \times 10^{-2}$. The Python reference model keeps f32 keys, so this bound, not 1e-4, is what an f16 cache can promise.

## Int4 weights

The runner can also serve quantized weights. A quantized linear weight $W \in \mathbb{R}^{\text{out} \times \text{in}}$ with groups of $g$ columns stores, per group, one f16 scale and $g$ four-bit integers (as in [Quantization](../21-inference/05-quantization.md) and `formats/safetensors.md`):

$$s = \mathrm{f16}\Big(\frac{\max |W_{\text{group}}|}{7}\Big), \qquad q = \mathrm{clamp}\big(\mathrm{rint}(W / s), -8, 7\big), \qquad \hat W = q\, s$$

with `rint` rounding half to even and $s$ the f16 value **actually stored**, so $|W - \hat W| \le s/2$. Two values share a byte: column $2b$ in the low nibble and $2b + 1$ in the high nibble, each in 4-bit two's complement. `QLinear::forward` unpacks the quantized values, widens the effective weight to f32, and uses Candle matrix multiplication. The runner loads int4 two ways: from an int4 file (`<name>.qweight`, `<name>.scales`, metadata `quant = "int4-g<g>-sym"`), or by quantizing f32 weights at load (`EngineConfig.quant = Some(Quant::Int4 { group })`).

## The sampler and PCG32

Once logits exist, `sample.rs` turns them into tokens. It ports the Python sampler and generator, and **parity** here means bit-identical: the same token id and the same f64 logprob from the same logits and seed. `spec/sampling.md` fixes the order:

1. widen to f64;
2. apply the repetition penalty over the distinct ids of prompt and output ($\ell \mathrel{/}= r$ when positive, $\ell \mathrel{*}= r$ otherwise);
3. apply presence and frequency penalties over the output only ($\ell \mathrel{-}= a_f c + a_p$);
4. take the **logprob** as $\log\mathrm{softmax}(\ell)$ at this point;
5. for greedy ($T = 0$), take the argmax with ties to the lowest id and **no draw**;
6. otherwise divide by $T$, keep the top $k$ by $(\ell \text{ desc}, \text{id asc})$, keep the nucleus up to and including the token that reaches $p$, keep ids with $q_i \ge m \cdot \max q$, take the softmax over the kept ids in ascending id order, draw one $u$, and walk the cumulative sum in ascending id order to the first id with $u < c$.

Every sum is a left-to-right loop, so Rust and Python add the same numbers in the same order.

Each request gets its own generator, `stream(seed, sample)`: $\mathrm{child\_seed}(s, 4) = \mathrm{mix64}(s + 4 \cdot \texttt{0x9E3779B97F4A7C15})$, then `pcg32_srandom_r(child_seed, 4)`. One uniform takes two outputs $a, b$: $u = \big((a \gg 5)\, 2^{26} + (b \gg 6)\big)\, 2^{-53}$.

## Working it by hand

Each of these rules can be checked on paper, and the worked numbers below reappear as tests.

**One sampled token** (from `spec/sampling.md`). Take logits $x = [1, 3, 2, 3, -1]$, no history, $T = 1$, top-k 3, top-p 0.8, seed 0. Logprobs come first: $M = 3$, $Z = e^{-2} + 1 + e^{-1} + 1 + e^{-4} = 2.52153$, so ids 1 and 3 have logprob $-\ln Z = -0.92487$. For top-k, the order by (logit desc, id asc) is $1, 3, 2, 0, 4$; keep $\{1, 3, 2\}$. For top-p over the kept ids: $Z' = 1 + e^{-1} + 1 = 2.36788$, $q_1 = q_3 = 0.42232$, $q_2 = 0.15536$; walking $1, 3$ gives $0.42232$, then $0.84464 \ge 0.8$, so keep $\{1, 3\}$ and renormalize to $0.5, 0.5$. For the generator, $\mathrm{child\_seed}(0, 4) = \texttt{0xF88BB8A8724C81EC}$, and the first uniform of `stream(0, sample)` is $u = 0.80209$. Walk ids in ascending order: after id 1, $c = 0.5$, not above $u$; after id 3, $c = 1.0 > u$. The token is **id 3**, logprob $-0.92487$.

**Int4.** Take weights $[0.7, -1.4, 0, 0.29995]$ in one group of 4: $\max|W| = 1.4$, $1.4 / 7 = 0.2$, and $\mathrm{f16}(0.2) = 0.199951171875$ (0.2 is not a binary fraction; f16 keeps 10 mantissa bits). Dividing by the stored scale: $0.7 / 0.19995 = 3.5009 \to 4$, $-1.4 / 0.19995 = -7.0017 \to -7$, $0 \to 0$, $0.29995 / 0.19995 = 1.5001 \to 2$. With the unrounded $0.2$ the last would be $1.49975 \to 1$: the stored scale changes a value. Packing $q = [-8, 7, 1, -1]$: $-8$ is $1000_2$ and $7$ is $0111_2$, so byte 0 is $0111\,1000_2 = \texttt{0x78}$; $1 = 0001_2$ and $-1 = 1111_2$ give $1111\,0001_2 = \texttt{0xF1}$.

**f16 rounding.** $1 + 2^{-11}$ lies exactly halfway between $1$ (`0x3C00`) and $1 + 2^{-10}$ (`0x3C01`); ties go to the even mantissa, so the result is `0x3C00`. $1 + 3 \cdot 2^{-11}$ lies halfway between `0x3C01` and `0x3C02`, and the even one is `0x3C02`.

**Where a key lands.** With blocks of $B = 16$ and a sequence whose block table is $[7, 2, 9]$, position 37 is block $\lfloor 37 / 16 \rfloor = 2$ of the table, pool block 9, slot $37 \bmod 16 = 5$. Its K for KV head $h$ starts at element $(h \cdot 16 + 5) \cdot D$ of layer $l$'s K slab of block 9.

## Designing the interface

The pool's interface is in `kv.rs`:

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

The numbers, model, forward, runner, and sampler interfaces span the other five files:

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

`forward` writes K and V for the new tokens of each sequence into the blocks its table names. The caller allocates the blocks; from the next section on, [the continuous batching scheduler](03-continuous-batching.md) and [the block manager](05-block-manager-and-prefix-cache.md) do so. `run_once` takes temporary blocks and gives them back, for embeddings and one-off checks.

## Building it step by step

The solution is six files; this walkthrough covers the parts that carry the design and links the rest. Try to predict each step before reading it.

### The reasoning chain

1. **A step is a batch of sequences, each a run of new tokens at absolute positions.** So the input is `ForwardBatch { seqs: [ForwardSeq { tokens, start, blocks }] }`: the tokens, the position of the first one, and the block table that says where its past lives. Every new token's position is `start + i`.
2. **Every token-wise operation (embedding, norms, projections, RoPE, MLP) is independent across rows**, so all $N$ new tokens of all sequences can go through them together as one $[N, d]$ matrix. Only attention mixes rows, and only within one sequence.
3. **Attention needs each sequence's whole past**, which lives in the pool. So per layer: write the new K and V into the sequence's blocks (position $j$ at block $\lfloor j/B \rfloor$, slot $j \bmod B$), gather positions $0..\text{start} + n - 1$ back, and attend with query $i$ at position `start + i`.
4. **Batch invariance needs every row to go through the same kernel.** Candle picks a different matmul kernel for $[1, K]$ and $[M, K]$, so `matmul_rows` multiplies one row at a time, and each query attends over exactly its visible keys (no masked tail whose shape changes with the batch).
5. **Only the last token of each sequence needs logits**, so the final norm and LM head run on $S$ rows, not $N$.
6. **The sampler must be bit-identical to Python's**, so it is a literal port of `spec/sampling.md`: f64 throughout, every sum left-to-right in a fixed id order, one draw per sampled token and none for greedy.
7. **The pool must be bounded and its states explicit**: free (no refs, no hash), used (refs > 0), cached (refs 0, registered hash, in the LRU). Allocation takes free blocks first, then evicts from the front of the LRU.

### Flattening the batch

Every sequence is validated (non-empty, inside the context, ids inside the vocabulary) while its tokens and absolute positions are appended to one flat list; `offsets[s]` remembers where sequence $s$ starts in it.

```rust
    for s in &batch.seqs {
        if s.tokens.is_empty() { return Err("forward: a sequence with no new tokens".into()); }
        if s.start + s.tokens.len() > cfg.max_position_embeddings { return Err(/* past the context */); }
        offsets.push(ids.len());
        for (i, &t) in s.tokens.iter().enumerate() {
            if t as usize >= cfg.vocab_size { return Err(/* token id outside vocab */); } // before the lookup
            ids.push(t);
            positions.push(s.start + i);           // ABSOLUTE position, never relative to the step
        }
    }
```

### One layer

Each layer applies the norm, three projections, and RoPE on q and k at the absolute positions, then, per sequence, writes KV, gathers KV, and attends with `q_offset = s.start`. The residual adds and the SwiGLU follow the Llama order exactly.

```rust
    for (li, lw) in w.layers.iter().enumerate() {
        let layer = li as u32;
        let normed = rms_norm(&x, &lw.input_norm, n, d, cfg.rms_norm_eps, device)?;
        let mut q = project(&lw.q, &normed, n, device)?;
        let mut k = project(&lw.k, &normed, n, device)?;
        let v = project(&lw.v, &normed, n, device)?;
        rope(&mut q, &positions, h, hd, inv_freq, n)?;      // H query heads
        rope(&mut k, &positions, hkv, hd, inv_freq, n)?;    // Hkv key heads; V is not rotated
        let mut attn = vec![0.0f32; n * h * hd];
        for (s, &off) in batch.seqs.iter().zip(&offsets) {
            let ns = s.tokens.len();
            write_kv(pool, layer, s, &k[off * hkv * hd..(off + ns) * hkv * hd], &v[off * hkv * hd..(off + ns) * hkv * hd])?;
            let tk = s.start + ns;                            // keys visible to the last new token
            let kc = gather_kv(pool, layer, s.blocks, tk, false)?;   // K slab, f16 -> f32
            let vc = gather_kv(pool, layer, s.blocks, tk, true)?;    // V slab
            // ... copy this sequence's queries into qs ...
            let o = attention(&qs, &kc, &vc, ns, tk, h, hkv, hd, s.start, 1.0 / (hd as f32).sqrt(), device)?;
            // ... scatter o back into attn at rows off..off + ns ...
        }
        let ao = project(&lw.o, &attn, n, device)?; add_assign(&mut x, &ao)?;      // x += o W_o^T
        let normed = rms_norm(&x, &lw.post_norm, n, d, cfg.rms_norm_eps, device)?;
        let g = project(&lw.gate, &normed, n, device)?;
        let u = project(&lw.up, &normed, n, device)?;
        let m = silu_mul(&g, &u, device)?;                  // silu on the GATE, times up
        let dn = project(&lw.down, &m, n, device)?; add_assign(&mut x, &dn)?;
    }
```

Notice that the new keys are written *before* gathering, so the gathered keys already passed through f16: a new token attends to its own key exactly as a later decode step will see it. That is what makes incremental decode agree with a full prefill.

### Writing a key where the table says

Position `p` maps to `(seq.blocks[p / bt], p % bt)`, and inside the slab `[n_kv_heads][block_tokens][head_dim]`, head `h` at slot `slot` starts at `(h * bt + slot) * hd`. Values are rounded to f16 on the way in. On layer 0 the block fill counts are raised, so the block manager knows which blocks are full.

```rust
pub fn write_kv(pool: &mut KvPool, layer: u32, seq: &ForwardSeq, k: &[f32], v: &[f32]) -> Result<(), String> {
    // ...
    let end = seq.start + seq.tokens.len();
    if seq.blocks.len() * bt < end { return Err(/* blocks cover fewer positions than the step needs */); }
    for (src, is_v) in [(k, false), (v, true)] {           // K to the K slab, V to the V slab
        for r in 0..seq.tokens.len() {
            let p = seq.start + r;
            let (blk, slot) = (seq.blocks[p / bt], p % bt); // table lookup: logical -> physical
            let slab = pool.slab_mut(blk, layer, is_v).map_err(|e| e.to_string())?;
            for h in 0..hkv {
                let dst = &mut slab[(h * bt + slot) * hd..(h * bt + slot + 1) * hd];
                let row = &src[r * hkv * hd + h * hd..r * hkv * hd + (h + 1) * hd];
                for (d, x) in dst.iter_mut().zip(row) { *d = f32_to_f16(*x); } // round to nearest even
            }
        }
    }
    // layer 0: set_fill on every block the step touched
    // ...
}
```

### Attention, one query at a time

Query `i` sees keys `0..=q_offset + i`. The slice of the gathered keys is exactly that long, so no mask is needed and the shapes depend on the position only. Grouped-query attention maps head `h` to KV head `h / (heads / kv_heads)`.

```rust
    for h in 0..heads {
        let kh = h / (heads / kv_heads);                   // GQA: consecutive query heads share KV
        // ... khv, vhv: this KV head's keys and values, [tk, hd] ...
        for i in 0..ns {
            let seen = q_offset + i + 1;                   // causal: keys 0..=position
            let kt = Tensor::from_slice(&khv[..seen * hd], (seen, hd), device)
                .and_then(|t| t.transpose(0, 1)).map_err(|e| e.to_string())?;
            let qrow = &q[(i * heads + h) * hd..(i * heads + h + 1) * hd];
            let scores: Vec<f32> = matmul_rows(qrow, 1, hd, &kt, device)?.into_iter().map(|x| x * scale).collect();
            let scores = Tensor::from_vec(scores, (1, seen), device).map_err(|e| e.to_string())?;
            let probs = tensor_vec(ops::softmax(&scores, D::Minus1).map_err(|e| e.to_string())?)?;
            let vt = Tensor::from_slice(&vhv[..seen * hd], (seen, hd), device).map_err(|e| e.to_string())?;
            out[(i * heads + h) * hd..(i * heads + h + 1) * hd].copy_from_slice(&matmul_rows(&probs, 1, seen, &vt, device)?);
        }
    }
```

`matmul_rows` is the batch-invariance device: it always issues $[1, K] \times [K, N]$ products, so a row's arithmetic does not depend on how many rows came with it.

```rust
pub(crate) fn matmul_rows(x: &[f32], m: usize, k: usize, b: &Tensor, device: &Device) -> Result<Vec<f32>, String> {
    // ...
    for r in 0..m {   // one row at a time: Candle chooses the same kernel for every row
        let a = Tensor::from_slice(&x[r * k..(r + 1) * k], (1, k), device).map_err(|e| e.to_string())?;
        let y = a.matmul(b).and_then(|y| y.flatten_all()).and_then(|y| y.to_vec1::<f32>()).map_err(|e| e.to_string())?;
        out.extend_from_slice(&y);
    }
    Ok(out)
}
```

### RoPE, half-split

Pair `i` is `(x[i], x[i + half])`, and both are read before either is written. The angle is the absolute position times `inv_freq[i]`, where `inv_freq[i] = theta^(-2i/d)` in increasing `i`.

```rust
        for i in 0..half {
            let angle = positions[r] as f32 * inv[i];
            let (sin, cos) = angle.sin_cos();
            let a = data[base + i]; let b = data[base + i + half];
            data[base + i] = a * cos - b * sin;
            data[base + i + half] = a * sin + b * cos;
        }
```

### The head

Only each sequence's last row goes through the final norm and the LM head; with tied embeddings the head is the embedding matrix transposed.

```rust
    for (r, (s, &off)) in batch.seqs.iter().zip(&offsets).enumerate() {
        let i = off + s.tokens.len() - 1;                  // last new token of sequence r
        last[r * d..(r + 1) * d].copy_from_slice(&x[i * d..(i + 1) * d]);
    }
    let normed = rms_norm(&last, &w.norm, rows, d, cfg.rms_norm_eps, device)?;   // the final norm
    let data = match &w.lm_head {
        Some(head) => project(head, &normed, rows, device)?,
        None => { /* tie_word_embeddings: normed @ embed^T via matmul_rows */ }
    };
```

### f32 to f16, ties to even

For the normal range, keep the top 10 mantissa bits and look at the 13 dropped bits: above half (`0x1000`) rounds up, and exactly half rounds up only if the kept value is odd. A carry out of the mantissa correctly bumps the exponent (and overflows to infinity at the top).

```rust
    let half = ((e as u32) << 10) | (man >> 13);           // exponent and top 10 mantissa bits
    let rem = man & 0x1fff;                                 // the 13 dropped bits
    let r = if rem > 0x1000 || (rem == 0x1000 && half & 1 == 1) { half + 1 } else { half };
    sign | r as u16
```

In the worked example, $1 + 2^{-11}$ has `rem == 0x1000` and an even `half` (`0x3C00`), so it stays `0x3C00`.

### Int4 quantization with the stored scale

The scale is rounded to f16 first and the codes are computed against that rounded value, which is what makes $|W - \hat W| \le s/2$ true for the weights the file actually encodes.

```rust
                let amax = w[cols.clone()].iter().fold(0.0f32, |m, v| m.max(v.abs()));
                let s16 = f32_to_f16(amax / 7.0);
                scales[r * groups + g] = s16;
                let s = f16_to_f32(s16);                    // the scale ACTUALLY stored
                if s == 0.0 { continue; }                   // an all-zero group: q = 0
                for c in cols {
                    q[c] = (w[c] / s).round_ties_even().clamp(-8.0, 7.0) as i8;
                }
    // pack_int4: byte b = (q[2b] & 0x0f) | ((q[2b + 1] & 0x0f) << 4): even column in the LOW nibble
```

### The sampler

`sample` is the spec's order in four calls. The logprob is taken from the penalized logits, before temperature and filtering, and greedy returns without touching `rng`.

```rust
pub fn sample(logits: &[f32], p: &SamplingParams, prompt: &[u32], output: &[u32], rng: &mut Pcg32) -> (u32, f64) {
    let l = apply_penalties(logits, p, prompt, output);   // f64; repetition over prompt+output, presence/frequency over output
    if p.temperature == 0.0 {
        let tok = argmax(&l);                              // strict >: ties to the lowest id
        return (tok as u32, logprob_of(&l, tok));          // no draw
    }
    let q = distribution(&l, p);                           // /T, top-k, top-p, min-p, softmax in ascending id order
    let u = rng.uniform_f64();                             // exactly one draw
    let tok = inverse_cdf(&q, u);                          // first id with u < c, ascending ids
    (tok as u32, logprob_of(&l, tok))
}
```

The repetition penalty branches on the sign, so it always pushes a seen token's logit toward less likely:

```rust
                l[i] = if l[i] > 0.0 { l[i] / p.repetition_penalty } else { l[i] * p.repetition_penalty };
```

The nucleus walk uses `>=` so the token that crosses $p$ is kept; in the worked example the walk reaches $0.84464 \ge 0.8$ at id 3 and keeps both ids:

```rust
        for (n, &i) in order.iter().enumerate() {          // order: (t desc, id asc)
            s += q[i];
            if s >= p.top_p { cut = n + 1; break; }        // keep the crossing token
        }
```

The generator for a request is `stream(seed, PURPOSE_SAMPLE)`, which is `Pcg32::new(child_seed(seed, 4), 4)`: the purpose id is the PCG sequence, not the tracer's default 54.

### The pool

`alloc` is all-or-nothing. It checks that free plus cached blocks suffice before taking anything, takes never-registered free blocks first, and only then evicts the *oldest* cached block from the front of the LRU.

```rust
    pub fn alloc(&mut self, n: usize) -> Result<Vec<u32>, KvError> {
        let free = self.stats().free as usize;
        if free + self.lru.len() < n { return Err(error(TL_EFULL, "KV_EFULL", /* ... */)); } // nothing taken
        let mut ids = Vec::with_capacity(n);
        for i in 0..self.blocks.len() {
            if ids.len() == n { break; }
            if self.blocks[i].refs == 0 && self.blocks[i].hash.is_none() { ids.push(i as u32); } // truly free
        }
        while ids.len() < n {
            let id = self.lru.pop_front().expect("capacity checked");    // oldest cached prefix
            let block = &mut self.blocks[id as usize];
            if let Some(hash) = block.hash.take() { self.by_hash.remove(&hash); } // forget it
            block.fill = 0;
            self.evictions += 1;
            ids.push(id);
        }
        for &id in &ids { self.blocks[id as usize].refs = 1; self.blocks[id as usize].fill = 0; }
        Ok(ids)
    }
```

`release` decrements the reference count. At zero, a registered block goes to the back of the LRU (cached, still findable by hash) and an unregistered one becomes free. `KvPool::new` refuses any zero dimension, including zero blocks.

Finally, `run_once` allocates temporary blocks, runs one sequence, and releases the blocks *before* propagating any error from the forward, so a failed embedding call cannot leak KV.

The full solution is [`kv.rs`](../../../../course/solutions/rust/crates/tl-engine/src/kv.rs), [`quant.rs`](../../../../course/solutions/rust/crates/tl-engine/src/quant.rs), [`model.rs`](../../../../course/solutions/rust/crates/tl-engine/src/model.rs) (config, memory-mapped safetensors, `Linear`, `matmul_rows`), [`forward.rs`](../../../../course/solutions/rust/crates/tl-engine/src/forward.rs), [`runner.rs`](../../../../course/solutions/rust/crates/tl-engine/src/runner.rs), and [`sample.rs`](../../../../course/solutions/rust/crates/tl-engine/src/sample.rs).

### Why this design

Start with cost. Projections are $O(N d^2)$ per layer; attention is $O(\sum_s n_s (p_s + n_s) H D)$; and the gather copies each sequence's whole context per layer per step, $O(\text{context} \cdot H_{kv} D)$. Pool memory is fixed at `n_blocks` blocks of $2 L H_{kv} B D$ f16 values.

Why gather into a contiguous buffer instead of reading blocks in place, as PagedAttention does? Gathering lets attention use ordinary Candle tensors and keeps the code short; it costs a copy per step that production kernels avoid (see [Paged attention for decode in C](../22-kernels/06-paged-attention-decode.md)). Why one-row matmuls instead of one big matmul? A big matmul is faster, but Candle's kernel choice depends on the row count and the reduction order changes with it, which breaks bit-for-bit batch invariance. The engine trades throughput for the guarantee that batching never changes a token.

Why round K and V to f16 at all? It halves the memory per cached token, so twice the sequences fit per pool. The price is a logit error of about $7 \times 10^{-3}$ on the tiny model, which is why the Hugging Face comparison uses $2 \times 10^{-2}$ rather than $10^{-4}$. And why a literal port of the sampler instead of a faster one, such as a partial sort or a vectorized softmax? Any change in the order of additions changes the f64 bits, and the requirement is the same id and logprob as Python from the same seed.

## What can go wrong

A correct runner handles the following cases; each row names its kind of check and why it matters downstream.

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the sampling hand example | unit | top-k 3 and top-p 0.8 keep ids 1 and 3, $u = 0.80209$, token 3 with logprob $-0.92487$ | the spec's worked example, end to end |
| PCG32 reference vectors | golden | `pcg32(42)` demo line, `uniform_f64`, `child_seed`, `stream` outputs from `pcg32.vectors.json` | one generator in every language (D10) |
| the Python sampler's golden streams | differential, golden | 12 cases x 12 tokens: ids and f64 logprobs bit for bit against the Python sampler's golden | the engine's seeded streams equal the Python ones |
| draw accounting | property | greedy leaves the generator untouched; a sampled token takes one `uniform_f64` | the disaggregated hand-off |
| top-p keeps the crossing token | boundary | mass exactly at $p$ keeps the crossing id | nucleus edge, the most common sampler bug |
| penalty semantics | unit | repetition divides positives and multiplies negatives over prompt and output; presence and frequency over output only | the OpenAI request fields mean what clients expect |
| seeded sampling follows the distribution | statistical | 4000 draws fit 0.4, 0.3, 0.2, 0.1 (chi-square below 16.27); a seed repeats | the sampler draws from the right distribution |
| KV pool bounds and lifecycle | boundary | capacity is all-or-nothing, refs balance, full blocks register, release caches, lookup reacquires | scheduler and prefix cache share one pool |
| eviction order | boundary | exhausted allocation evicts the oldest cached prefix | cache policy preserves recent prefixes |
| zero capacity | boundary | zero blocks is rejected during construction | invalid engine configuration fails early |
| export and import | differential | a written f16 value survives an export/import round trip | disaggregated decode resumes with the same cache |
| f16 round to nearest even | boundary | ties, overflow, subnormals, NaN; every f16 pattern round-trips | the KV cache stores f16 |
| the int4 hand example | unit | nibble order and the stored-scale rule | int4 files written by the Python quantizer load here |
| `QLinear` against its dequantized weights | differential | Candle output equals x times the dequantized weights within float tolerance | the packed layout means what the format says |
| safetensors reader rules | boundary | offsets from the data buffer, F16 and BF16 widening, gaps and trailing bytes refused | every checkpoint goes through this reader |
| config refusals | boundary | scaled RoPE, sliding window, unknown arch, missing tokenizer refused at load | wrong models fail loudly at start |
| the bigram checkpoint still serves | conformance | the tracer checkpoint loads and greedy continues `bcd` | the tracer smoke test keeps working (D32) |
| tiny Llama logits match HF | golden | last-token logits within $2 \times 10^{-2}$ of HF's float32 forward | the whole graph: GQA, tied embeddings, BF16 file |
| tiny Llama greedy matches HF | golden | 32 greedy tokens per prompt equal HF's under the near-tie rule | the engine's greedy stream equals the Python one |
| incremental decode equals full prefill | differential | prefill then one token per step equals one prefill, within floating-point tolerance | chunked prefill and preemption by recompute |
| batched forward equals single | differential | two sequences in one step agree with their alone logits | continuous batching changes nothing |
| `run_once` returns its blocks | unit | five one-off forwards leave every block free | embeddings do not leak KV |
| int4 runner paths agree | differential | quantize-at-load equals an int4 file bit for bit and its dequantized f32 twin closely | both int4 paths agree |
| bad batches refused | boundary | a short block table or a position past the context is `Err` | the scheduler's mistakes surface, not corrupt KV |

The pitfalls cluster in three places: the sampler, the number formats, and the forward. In the sampler, the dangerous bugs produce output that looks statistically fine but matches nothing. Swapping the shifts of the two draws in `uniform_f64` gives uniforms that look fine and match nothing; the solution uses `a = next >> 5`, `b = next >> 6`, `(a << 26) + b`. Seeding the sample stream on sequence 54 instead of the purpose id makes every seeded stream differ from Python's; `stream` calls `Pcg32::new(child_seed(seed, p), p)`.

> **Warning:** Walking the CDF in probability order instead of ascending id order gives the correct distribution but different ids for the same seed. `softmax_over` sorts ids ascending.

The full list, with symptom and fix:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| Taking the logprob after temperature | logprobs change with $T$; OpenAI clients get wrong values | `logprob_of(&l, tok)` on the penalized logits |
| `>` instead of `>=` in the nucleus walk | the token that reaches $p$ is dropped | `if s >= p.top_p` |
| Presence and frequency penalties over the prompt | prompt words become unlikely in the answer | counts come from `output` only |
| Dividing negative logits by the repetition penalty | repeated unlikely tokens become MORE likely | divide if positive, multiply otherwise |
| Truncating instead of rounding to f16 | a drift of half an ulp per cached value | round to nearest, ties to even, on the 13 dropped bits |
| The odd column in the low nibble | int4 files from the Python quantizer decode to noise | `pack_int4`: column $2b$ low, $2b + 1$ high |
| Rounding int4 with the f32 scale, storing the f16 one | some weights off by one level | codes computed against `f16_to_f32(s16)` |
| `data_offsets` from the start of the file | garbage weights | the reader adds the data-buffer start (as in the tracer engine) |
| Reading BF16 as F16 | SmolLM2 weights come out wildly wrong | `bf16_to_f32` is `h << 16`, chosen by the header's dtype |
| RoPE positions relative to the step | decode tokens all at position 0; output degrades after the prompt | `positions.push(s.start + i)` |
| `q_offset` 0 for a decode step | the new token sees only key 0 | `attention(..., s.start, ...)` |
| K written to the V slab | logits far from HF | `write_kv` pairs `(k, false)` and `(v, true)` |
| Skipping the final norm | logits scaled wrong | `rms_norm(&last, &w.norm, ...)` before the head |
| Not giving temporary blocks back | the pool drains a little per embedding call | `run_once` releases before propagating errors |
| Accepting a zero-block pool | the first allocation fails far from the invalid configuration | `KvPool::new` rejects zero dimensions |
| Evicting the newest prefix instead of the oldest | a useful recent prefix disappears early | `lru.pop_front()`; release pushes to the back |
| Reversing RoPE frequency order | Llama logits differ from the reference | `inv_freq[i] = theta^(-2i/d)` for increasing `i` |
| Not checking token ids against the vocabulary | an invalid id reaches the embedding lookup | checked while flattening the batch |
| `silu` applied to the up projection | close-looking, wrong logits | `silu_mul(&g, &u)`: silu on the gate |

## Where this leads

This runner reimplements the model graph, int4 layout, and f16 conversions from the Llama, quantization, and floating-point sections, reproduces the Python sampler and generator bit for bit, and keeps the tracer's checkpoint loading and byte bigram working. Without it, the engine has no model to run past the tracer bigram.

The rest of this chapter runs on it. [The continuous batching scheduler](03-continuous-batching.md) forms each step's `ForwardBatch`. [Chunked prefill](04-chunked-prefill.md) relies on chunk invariance. [The block manager](05-block-manager-and-prefix-cache.md) owns the pool's blocks while this runner writes into them. [The tokio server](06-openai-server-on-tokio.md) samples with `sample` and a per-request `stream(seed, sample)`. [Disaggregated prefill and decode](07-disaggregated-prefill-decode.md) moves this pool's blocks (`export`, `import`) and the generator state, and [speculative decoding](09-speculative-decoding-in-the-engine.md) and [tool calls and constrained decoding](10-tool-calls-and-constrained-decoding.md) run on this runner too.

Each simplification has a production counterpart:

| This solution | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| a gather of K and V per step | [vLLM PagedAttention](https://github.com/vllm-project/vllm) | attention reads the blocks in place, no copy | `vllm/attention/` |
| f16 KV | fp8 KV with per-head scales | half the memory, a calibrated error | [Interface migration: KV format v1 to v2](../../software-engineering/08-maintenance/01-kv-format-v1-to-v2.md), `formats/kv-block.md` v2 |
| int4 groups of 32 | [GPTQ](https://arxiv.org/abs/2210.17323), [AWQ](https://arxiv.org/abs/2306.00978) | error-aware rounding, activation-aware scales | the further reading of [Quantization](../21-inference/05-quantization.md) |
| CPU tensor operations | [candle](https://github.com/huggingface/candle) Metal backend | device execution with the same layer graph | Candle documentation |

### Further reading

- [Rust reference, `Drop`](https://doc.rust-lang.org/reference/destructors.html) (free).
- [safetensors format](https://github.com/huggingface/safetensors) (free).
- [Hugging Face `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) (free).
- [Kwon et al. 2023, PagedAttention](https://arxiv.org/abs/2309.06180) (free).

## Key terms

- **model runner**: the component that loads a checkpoint and runs one batched forward step, writing K and V into the pool and returning last-token logits.
- **block table**: a sequence's list of pool block ids; position $j$ lives in entry $\lfloor j/B \rfloor$, slot $j \bmod B$.
- **batch invariance**: a token's logits do not depend on which other tokens share its step, bit for bit.
- **chunk invariance**: prefilling a prompt in one step or in pieces gives the same logits within tolerance.
- **stored scale**: the f16 scale actually written for an int4 group, against which the codes must be rounded.
- **ties to even**: rounding a value exactly halfway between two representable numbers to the one with an even last bit.
- **parity**: the same token id and the same f64 logprob as the Python sampler from the same logits and seed.

## Summary

- The Rust engine owns the model graph and uses Candle for tensor operations; the solution writes the Llama layers itself and owns a bounded Rust KV cache.
- `KvPool` owns fixed-size f16 K and V slabs, block references, the prefix hash index, and LRU eviction, with all-or-nothing allocation.
- One step runs a batch of sequences, each a run of new tokens at absolute positions; their K and V go into the pool as f16 and are read back for attention, so the logits match Hugging Face's float32 forward within the f16 bound.
- The forward is batch-invariant bit for bit (one-row products, per-query key slices) and chunk-invariant within a stated tolerance, so batching sequences together or splitting a prompt into steps preserves logits.
- The sampler follows `spec/sampling.md` step by step in f64, so the same logits and seed give the Python sampler's id and logprob exactly.

## Review questions

1. Why are the new K and V written to the pool before they are gathered for this step's attention?
   <details><summary>Answer</summary>So a new token attends to the f16-rounded version of its own key and the other new keys, exactly what a later step will read back. If the step used its f32 keys directly, prefill and incremental decode would see different key values and disagree.</details>
2. Why does `attention` slice the keys to `seen = q_offset + i + 1` instead of masking a full row?
   <details><summary>Answer</summary>A masked full row has a length that depends on the other tokens in the step, so the softmax and matmul shapes (and Candle's reduction path) would change with the batch. Slicing makes each query's computation depend only on its position.</details>
3. Why is the int4 code computed with the f16 scale rather than $\max|W|/7$ in f32?
   <details><summary>Answer</summary>The file stores the f16 scale, and dequantization multiplies by it. Rounding against the f32 value can choose a code one level off (in the example $0.29995$ becomes 1 instead of 2), breaking the $s/2$ error bound for the weights actually decoded.</details>
4. What would break if greedy decoding drew a uniform anyway?
   <details><summary>Answer</summary>The generator's position would advance on greedy tokens, so a request that mixes or switches modes, or is handed to another engine (as in disaggregated prefill and decode) with its generator state, would produce different later samples than Python, which takes no draw for greedy.</details>
5. Why does `alloc` check capacity before taking any block?
   <details><summary>Answer</summary>So a failed allocation changes nothing: no block is half-taken and no cached prefix is evicted for an allocation that cannot succeed. The scheduler can then preempt or wait with the pool in a known state.</details>
6. With $B = 16$ and block table $[7, 2, 9]$, where does position 37's key for KV head $h$ live?
   <details><summary>Answer</summary>Table entry $\lfloor 37/16 \rfloor = 2$, which is pool block 9, slot $37 \bmod 16 = 5$; within layer $l$'s K slab of block 9 it starts at element $(h \cdot 16 + 5) \cdot D$.</details>
