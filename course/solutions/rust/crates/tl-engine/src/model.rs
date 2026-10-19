//! A model directory on disk (L10.1): `config.json` and `model.safetensors`
//! (formats/safetensors.md, formats/config.schema.json), read through a
//! memory map, so the file's bytes are paged in by the OS instead of copied.
//!
//! Two architectures: `bigram` (the tracer checkpoint of L0.0, so the
//! Pass 1 model still serves) and `llama` (the Llama family of L7.9,
//! HF tensor names). Weights may be F32, F16, or BF16 (widened to f32 at
//! load), or int4 (`<name>.qweight` U8 + `<name>.scales` F16, metadata
//! `quant = "int4-g<group>-sym"`), which stays packed.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

use memmap2::Mmap;
use serde_json::Value;
use candle_core::{Device, Tensor};

use crate::quant::{bf16_to_f32, f16_to_f32, QLinear};

/// `tl_arch` of config.json.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arch {
    Bigram,
    Llama,
}

/// `tl_tokenizer` of config.json: the byte tokenizer, or `tokenizer.json`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenizerKind {
    Bytes,
    File,
}

/// The fields of config.json the engine reads.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelConfig {
    pub arch: Arch,
    pub tokenizer: TokenizerKind,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f64,
    pub tie_word_embeddings: bool,
    pub max_position_embeddings: usize,
    pub bos_token_id: Option<u32>,
    pub eos_token_ids: Vec<u32>,
}

fn get_usize(v: &Value, key: &str) -> Result<usize, String> {
    v.get(key)
        .and_then(Value::as_u64)
        .map(|x| x as usize)
        .ok_or_else(|| format!("config.json: `{key}` must be a non-negative integer"))
}

fn opt_ids(v: Option<&Value>) -> Vec<u32> {
    match v {
        Some(Value::Number(n)) => n.as_u64().map(|x| vec![x as u32]).unwrap_or_default(),
        Some(Value::Array(a)) => a.iter().filter_map(Value::as_u64).map(|x| x as u32).collect(),
        _ => Vec::new(),
    }
}

impl ModelConfig {
    /// Parses config.json. A bigram needs `vocab_size` 256 and the byte
    /// tokenizer; a llama needs the HF shape fields. Anything the engine
    /// cannot run (another arch, rope scaling, a sliding window, biases) is
    /// an error here, at load, never mid-request.
    pub fn from_json(text: &str) -> Result<ModelConfig, String> {
        let v: Value = serde_json::from_str(text).map_err(|e| format!("config.json: {e}"))?;
        let tokenizer = match v.get("tl_tokenizer").and_then(Value::as_str) {
            Some("bytes") => TokenizerKind::Bytes,
            Some("file") => TokenizerKind::File,
            Some(other) => return Err(format!("config.json: tl_tokenizer {other:?} is not bytes or file")),
            None => return Err("config.json: `tl_tokenizer` is missing".to_string()),
        };
        let arch = match v.get("tl_arch").and_then(Value::as_str) {
            Some("bigram") => Arch::Bigram,
            Some("llama") => Arch::Llama,
            Some(other) => return Err(format!("config.json: tl_arch {other:?} is not served by this engine (bigram, llama)")),
            None => return Err("config.json: `tl_arch` is missing".to_string()),
        };
        let vocab_size = get_usize(&v, "vocab_size")?;
        if arch == Arch::Bigram {
            if vocab_size != 256 || tokenizer != TokenizerKind::Bytes {
                return Err("config.json: a bigram needs vocab_size 256 and tl_tokenizer bytes".to_string());
            }
            return Ok(ModelConfig {
                arch,
                tokenizer,
                vocab_size,
                hidden_size: 256,
                intermediate_size: 0,
                n_layers: 0,
                n_heads: 0,
                n_kv_heads: 0,
                head_dim: 0,
                rms_norm_eps: 0.0,
                rope_theta: 0.0,
                tie_word_embeddings: false,
                max_position_embeddings: v.get("max_position_embeddings").and_then(Value::as_u64).unwrap_or(1 << 20) as usize,
                bos_token_id: None,
                eos_token_ids: Vec::new(),
            });
        }
        let hidden_size = get_usize(&v, "hidden_size")?;
        let n_heads = get_usize(&v, "num_attention_heads")?;
        let n_kv_heads = v.get("num_key_value_heads").and_then(Value::as_u64).map(|x| x as usize).unwrap_or(n_heads);
        let head_dim = match v.get("head_dim").and_then(Value::as_u64) {
            Some(d) => d as usize,
            None if n_heads > 0 => hidden_size / n_heads,
            None => 0,
        };
        if n_heads == 0 || n_kv_heads == 0 || head_dim == 0 || n_heads % n_kv_heads != 0 {
            return Err(format!("config.json: heads {n_heads}, kv heads {n_kv_heads}, head_dim {head_dim} do not form GQA"));
        }
        let rope = v.get("rope_parameters").or_else(|| v.get("rope_scaling"));
        let rope_type = rope.and_then(|r| r.get("rope_type").or_else(|| r.get("type"))).and_then(Value::as_str).unwrap_or("default");
        if rope_type != "default" {
            return Err(format!("config.json: rope_type {rope_type:?} is not supported by this engine (default only)"));
        }
        let rope_theta = rope
            .and_then(|r| r.get("rope_theta"))
            .or_else(|| v.get("rope_theta"))
            .and_then(Value::as_f64)
            .unwrap_or(10000.0);
        if v.get("sliding_window").is_some_and(|w| !w.is_null()) {
            return Err("config.json: sliding_window is not supported by this engine".to_string());
        }
        for key in ["attention_bias", "mlp_bias", "tl_qkv_bias"] {
            if v.get(key).and_then(Value::as_bool) == Some(true) {
                return Err(format!("config.json: {key} is not supported by this engine"));
            }
        }
        Ok(ModelConfig {
            arch,
            tokenizer,
            vocab_size,
            hidden_size,
            intermediate_size: get_usize(&v, "intermediate_size")?,
            n_layers: get_usize(&v, "num_hidden_layers")?,
            n_heads,
            n_kv_heads,
            head_dim,
            rms_norm_eps: v.get("rms_norm_eps").and_then(Value::as_f64).unwrap_or(1e-6) as f32,
            rope_theta,
            tie_word_embeddings: v.get("tie_word_embeddings").and_then(Value::as_bool).unwrap_or(false),
            max_position_embeddings: v.get("max_position_embeddings").and_then(Value::as_u64).unwrap_or(2048) as usize,
            bos_token_id: v.get("bos_token_id").and_then(Value::as_u64).map(|x| x as u32),
            eos_token_ids: opt_ids(v.get("eos_token_id")),
        })
    }

    /// `<dir>/config.json`.
    pub fn load(dir: &Path) -> Result<ModelConfig, String> {
        let p = dir.join("config.json");
        let text = std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
        ModelConfig::from_json(&text)
    }
}

/// One tensor of a safetensors header: name, dtype, shape, and its byte
/// range relative to the start of the data buffer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub begin: usize,
    pub end: usize,
}

/// Bytes per element of a safetensors dtype the engine reads.
pub fn dtype_size(dtype: &str) -> Option<usize> {
    match dtype {
        "F32" | "I32" | "U32" => Some(4),
        "F16" | "BF16" | "I16" | "U16" => Some(2),
        "U8" | "I8" | "BOOL" | "F8_E4M3" | "F8_E5M2" => Some(1),
        "F64" | "I64" | "U64" => Some(8),
        _ => None,
    }
}

/// A parsed safetensors header: where the data buffer starts, the tensors
/// (sorted by name), and `__metadata__`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    pub data_start: usize,
    pub tensors: Vec<TensorInfo>,
    pub metadata: BTreeMap<String, String>,
}

/// Reads and checks the header of a safetensors file (formats/safetensors.md):
/// the u64 LE header length fits in the file; the header is a JSON object;
/// every `data_offsets` range is inside the data buffer, matches dtype x
/// shape, and the ranges tile the buffer exactly (no gap, overlap, or
/// trailing byte).
pub fn parse_layout(bytes: &[u8]) -> Result<Layout, String> {
    if bytes.len() < 8 {
        return Err("safetensors: shorter than the 8-byte header length".to_string());
    }
    let n = u64::from_le_bytes(bytes[..8].try_into().expect("8 bytes")) as usize;
    if n > bytes.len() - 8 {
        return Err(format!("safetensors: header length {n} runs past the end of the file"));
    }
    let header = std::str::from_utf8(&bytes[8..8 + n]).map_err(|_| "safetensors: header is not UTF-8".to_string())?;
    let v: Value = serde_json::from_str(header.trim_end()).map_err(|e| format!("safetensors header: {e}"))?;
    let obj = v.as_object().ok_or("safetensors header: not a JSON object")?;
    let data_start = 8 + n;
    let data_len = bytes.len() - data_start;
    let mut metadata = BTreeMap::new();
    let mut tensors = Vec::new();
    for (name, t) in obj {
        if name == "__metadata__" {
            for (k, val) in t.as_object().ok_or("safetensors: __metadata__ is not an object")? {
                metadata.insert(k.clone(), val.as_str().unwrap_or_default().to_string());
            }
            continue;
        }
        let dtype = t.get("dtype").and_then(Value::as_str).ok_or(format!("safetensors: {name} has no dtype"))?;
        let size = dtype_size(dtype).ok_or(format!("safetensors: {name} has unknown dtype {dtype}"))?;
        let shape: Vec<usize> = t
            .get("shape")
            .and_then(Value::as_array)
            .ok_or(format!("safetensors: {name} has no shape"))?
            .iter()
            .map(|x| x.as_u64().map(|u| u as usize).ok_or(format!("safetensors: {name} shape")))
            .collect::<Result<_, _>>()?;
        let off = t.get("data_offsets").and_then(Value::as_array).ok_or(format!("safetensors: {name} has no data_offsets"))?;
        let (begin, end) = match (off.first().and_then(Value::as_u64), off.get(1).and_then(Value::as_u64), off.len()) {
            (Some(b), Some(e), 2) => (b as usize, e as usize),
            _ => return Err(format!("safetensors: {name} data_offsets must be [begin, end]")),
        };
        if begin > end || end > data_len {
            return Err(format!("safetensors: {name} [{begin}, {end}) is outside the {data_len}-byte data buffer"));
        }
        let want = shape.iter().product::<usize>() * size;
        if end - begin != want {
            return Err(format!("safetensors: {name} holds {} bytes; {dtype} {shape:?} needs {want}", end - begin));
        }
        tensors.push(TensorInfo { name: name.clone(), dtype: dtype.to_string(), shape, begin, end });
    }
    let mut spans: Vec<(usize, usize)> = tensors.iter().map(|t| (t.begin, t.end)).collect();
    spans.sort_unstable();
    let mut at = 0;
    for (b, e) in spans {
        if b != at {
            return Err(format!("safetensors: the data buffer has a gap or overlap at byte {at}"));
        }
        at = e;
    }
    if at != data_len {
        return Err(format!("safetensors: {} bytes after the last tensor", data_len - at));
    }
    tensors.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(Layout { data_start, tensors, metadata })
}

/// A memory-mapped safetensors file.
pub struct SafeTensors {
    map: Mmap,
    pub layout: Layout,
}

impl SafeTensors {
    /// Maps `path` read-only and checks its header.
    pub fn open(path: &Path) -> Result<SafeTensors, String> {
        let f = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        // SAFETY: the map is read-only; the engine never writes the file, and
        // a checkpoint is not rewritten while an engine serves it.
        let map = unsafe { Mmap::map(&f) }.map_err(|e| format!("{}: mmap: {e}", path.display()))?;
        let layout = parse_layout(&map)?;
        Ok(SafeTensors { map, layout })
    }

    /// The tensor called `name`, if present.
    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.layout.tensors.binary_search_by(|t| t.name.as_str().cmp(name)).ok().map(|i| &self.layout.tensors[i])
    }

    /// The raw bytes of a tensor.
    pub fn bytes(&self, t: &TensorInfo) -> &[u8] {
        &self.map[self.layout.data_start + t.begin..self.layout.data_start + t.end]
    }

    /// A float tensor (F32, F16, or BF16) widened to f32, with its shape
    /// checked against `shape`.
    pub fn f32(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
        let t = self.info(name).ok_or(format!("model.safetensors: tensor {name} is missing"))?;
        if t.shape != shape {
            return Err(format!("model.safetensors: {name} has shape {:?}, expected {shape:?}", t.shape));
        }
        let b = self.bytes(t);
        let le16 = |c: &[u8]| u16::from_le_bytes([c[0], c[1]]);
        match t.dtype.as_str() {
            "F32" => Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()),
            "F16" => Ok(b.chunks_exact(2).map(|c| f16_to_f32(le16(c))).collect()),
            "BF16" => Ok(b.chunks_exact(2).map(|c| bf16_to_f32(le16(c))).collect()),
            other => Err(format!("model.safetensors: {name} is {other}; F32, F16, or BF16 expected")),
        }
    }

    /// An int4 linear `<name>.qweight` + `<name>.scales` of shape [out, inp].
    pub fn q4(&self, name: &str, out: usize, inp: usize, group: usize) -> Result<QLinear, String> {
        let qw = self.info(&format!("{name}.qweight")).ok_or(format!("model.safetensors: {name}.qweight is missing"))?;
        let sc = self.info(&format!("{name}.scales")).ok_or(format!("model.safetensors: {name}.scales is missing"))?;
        if qw.dtype != "U8" || qw.shape != [out, inp / 2] {
            return Err(format!("model.safetensors: {name}.qweight must be U8 [{out}, {}]", inp / 2));
        }
        if sc.dtype != "F16" || group == 0 || sc.shape != [out, inp / group] {
            return Err(format!("model.safetensors: {name}.scales must be F16 [{out}, {}]", inp / group.max(1)));
        }
        Ok(QLinear {
            out,
            inp,
            group,
            qweight: self.bytes(qw).to_vec(),
            scales: self.bytes(sc).chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect(),
        })
    }
}

/// x [m, k] @ b [k, n], one row at a time. Candle picks its reduction path
/// from the shape (a one-row product takes a matrix-vector kernel, a taller
/// one a blocked kernel that sums in another order), so a row's result would
/// depend on how many rows share the call. Sending every row through the same
/// [1, k] x [k, n] product makes the result batch-invariant: a token's
/// projection is bit-identical whether it is decoded alone, prefilled with
/// its prompt, or batched with other sequences.
pub(crate) fn matmul_rows(x: &[f32], m: usize, k: usize, b: &Tensor, device: &Device) -> Result<Vec<f32>, String> {
    if x.len() != m * k { return Err(format!("matmul: input length {} is not [{m}, {k}]", x.len())); }
    let mut out = Vec::new();
    for r in 0..m {
        let a = Tensor::from_slice(&x[r * k..(r + 1) * k], (1, k), device).map_err(|e| e.to_string())?;
        let y = a.matmul(b).and_then(|y| y.flatten_all()).and_then(|y| y.to_vec1::<f32>()).map_err(|e| e.to_string())?;
        out.extend_from_slice(&y);
    }
    Ok(out)
}

/// A linear layer W [out, inp]: `forward` computes y = x @ W^T.
#[derive(Clone, Debug, PartialEq)]
pub enum Linear {
    F32 { w: Vec<f32>, out: usize, inp: usize },
    Q4(QLinear),
}

impl Linear {
    /// (out, inp).
    pub fn dims(&self) -> (usize, usize) {
        match self {
            Linear::F32 { out, inp, .. } => (*out, *inp),
            Linear::Q4(q) => (q.out, q.inp),
        }
    }

    /// y [m, out] = x [m, inp] @ W^T using Candle.
    pub fn forward(&self, x: &[f32], m: usize, y: &mut [f32]) -> Result<(), String> {
        self.forward_on(x, m, y, &Device::Cpu)
    }

    pub fn forward_on(&self, x: &[f32], m: usize, y: &mut [f32], device: &Device) -> Result<(), String> {
        let (out, inp) = self.dims();
        if x.len() != m.checked_mul(inp).ok_or_else(|| "linear input size overflow".to_string())? || y.len() != m.checked_mul(out).ok_or_else(|| "linear output size overflow".to_string())? {
            return Err(format!("linear: input/output lengths do not match [{m}, {inp}] -> [{m}, {out}]"));
        }
        let weight = match self { Linear::F32 { w, .. } => w.clone(), Linear::Q4(q) => q.dequantize() };
        let b = Tensor::from_vec(weight, (out, inp), device).map_err(|e| e.to_string())?;
        let transposed = b.transpose(0, 1).map_err(|e| e.to_string())?;
        y.copy_from_slice(&matmul_rows(x, m, inp, &transposed, device)?);
        Ok(())
    }
}

/// The weights of one decoder layer.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerWeights {
    pub input_norm: Vec<f32>,
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub o: Linear,
    pub post_norm: Vec<f32>,
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
}

/// A Llama model: token embedding, layers, final norm, and the LM head
/// (`None` when tied to the embedding).
#[derive(Clone, Debug, PartialEq)]
pub struct LlamaWeights {
    pub embed: Vec<f32>,
    pub layers: Vec<LayerWeights>,
    pub norm: Vec<f32>,
    pub lm_head: Option<Linear>,
}

/// What `ModelRunner` runs.
#[derive(Clone, Debug, PartialEq)]
pub enum Weights {
    /// `bigram.weight` [256, 256]: row i holds the logits after byte i.
    Bigram(Vec<f32>),
    Llama(LlamaWeights),
}

/// The group size of an int4 file's `quant` metadata (`int4-g32-sym`).
pub fn int4_group(meta: &BTreeMap<String, String>) -> Result<Option<usize>, String> {
    match meta.get("quant") {
        None => Ok(None),
        Some(q) => q
            .strip_prefix("int4-g")
            .and_then(|r| r.strip_suffix("-sym"))
            .and_then(|g| g.parse::<usize>().ok())
            .filter(|&g| g > 0 && g % 2 == 0)
            .map(Some)
            .ok_or(format!("model.safetensors: quant {q:?} is not int4-g<even group>-sym")),
    }
}

/// Loads every weight `cfg` needs from `st`. With `quantize = Some(group)`,
/// f32 projection weights are int4-quantized at load (embedding, norms, and
/// the LM head stay f32); an int4 file's linears load packed as they are.
pub fn load_weights(st: &SafeTensors, cfg: &ModelConfig, quantize: Option<usize>) -> Result<Weights, String> {
    if cfg.arch == Arch::Bigram {
        return Ok(Weights::Bigram(st.f32("bigram.weight", &[256, 256])?));
    }
    let file_group = int4_group(&st.layout.metadata)?;
    let d = cfg.hidden_size;
    let linear = |name: &str, out: usize, inp: usize| -> Result<Linear, String> {
        if let Some(g) = file_group {
            if st.info(&format!("{name}.qweight")).is_some() {
                return Ok(Linear::Q4(st.q4(name, out, inp, g)?));
            }
        }
        let w = st.f32(&format!("{name}.weight"), &[out, inp])?;
        match quantize {
            Some(g) => Ok(Linear::Q4(QLinear::quantize(&w, out, inp, g)?)),
            None => Ok(Linear::F32 { w, out, inp }),
        }
    };
    let (h, hkv, hd, ff) = (cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.intermediate_size);
    let mut layers = Vec::with_capacity(cfg.n_layers);
    for i in 0..cfg.n_layers {
        let p = format!("model.layers.{i}");
        layers.push(LayerWeights {
            input_norm: st.f32(&format!("{p}.input_layernorm.weight"), &[d])?,
            q: linear(&format!("{p}.self_attn.q_proj"), h * hd, d)?,
            k: linear(&format!("{p}.self_attn.k_proj"), hkv * hd, d)?,
            v: linear(&format!("{p}.self_attn.v_proj"), hkv * hd, d)?,
            o: linear(&format!("{p}.self_attn.o_proj"), d, h * hd)?,
            post_norm: st.f32(&format!("{p}.post_attention_layernorm.weight"), &[d])?,
            gate: linear(&format!("{p}.mlp.gate_proj"), ff, d)?,
            up: linear(&format!("{p}.mlp.up_proj"), ff, d)?,
            down: linear(&format!("{p}.mlp.down_proj"), d, ff)?,
        });
    }
    let lm_head = if cfg.tie_word_embeddings || st.info("lm_head.weight").is_none() {
        None
    } else {
        Some(Linear::F32 { w: st.f32("lm_head.weight", &[cfg.vocab_size, d])?, out: cfg.vocab_size, inp: d })
    };
    Ok(Weights::Llama(LlamaWeights {
        embed: st.f32("model.embed_tokens.weight", &[cfg.vocab_size, d])?,
        layers,
        norm: st.f32("model.norm.weight", &[d])?,
        lm_head,
    }))
}
