//! Llama and byte-bigram forward passes (L10.1), built from Candle tensors
//! and the layer operations the learner implements here. K and V live in the
//! engine's Rust-owned, f16 KV pool between steps.

use candle_core::{D, Device, Tensor};
use candle_nn::{ops, Module};

use crate::kv::KvPool;
use crate::model::{matmul_rows, Linear, LlamaWeights, ModelConfig};
use crate::quant::{f16_to_f32, f32_to_f16};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ForwardSeq<'a> { pub tokens: &'a [u32], pub start: usize, pub blocks: &'a [u32] }

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ForwardBatch<'a> { pub seqs: Vec<ForwardSeq<'a>> }

#[derive(Clone, Debug, PartialEq)]
pub struct Logits { pub vocab: usize, pub data: Vec<f32> }

impl Logits {
    pub fn rows(&self) -> usize {
        if self.vocab == 0 { 0 } else { self.data.len() / self.vocab }
    }
    pub fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.vocab..(i + 1) * self.vocab]
    }
}

pub fn rope_inv_freq(theta: f64, d: usize) -> Vec<f32> {
    let base = theta as f32;
    (0..d / 2).map(|i| 1.0 / base.powf((2 * i) as f32 / d as f32)).collect()
}

fn tensor_vec(t: Tensor) -> Result<Vec<f32>, String> {
    Ok(t.flatten_all().and_then(|x| x.to_vec1::<f32>()).map_err(|e| e.to_string())?)
}

fn rms_norm(x: &[f32], weight: &[f32], rows: usize, width: usize, eps: f32, device: &Device) -> Result<Vec<f32>, String> {
    let x = Tensor::from_vec(x.to_vec(), (rows, width), device).map_err(|e| e.to_string())?;
    let w = Tensor::from_vec(weight.to_vec(), width, device).map_err(|e| e.to_string())?;
    tensor_vec(ops::rms_norm(&x, &w, eps).map_err(|e| e.to_string())?)
}

fn project(l: &Linear, x: &[f32], rows: usize, device: &Device) -> Result<Vec<f32>, String> {
    let (out, _) = l.dims();
    let mut y = vec![0.0f32; rows * out];
    l.forward_on(x, rows, &mut y, device)?;
    Ok(y)
}

fn add_assign(x: &mut [f32], y: &[f32]) -> Result<(), String> {
    if x.len() != y.len() { return Err("residual add shape mismatch".into()); }
    for (a,b) in x.iter_mut().zip(y) { *a += *b; }
    Ok(())
}

fn silu_mul(gate: &[f32], up: &[f32], device: &Device) -> Result<Vec<f32>, String> {
    if gate.len() != up.len() { return Err("SwiGLU shape mismatch".into()); }
    let g = Tensor::from_vec(gate.to_vec(), gate.len(), device).map_err(|e| e.to_string())?;
    let u = Tensor::from_vec(up.to_vec(), up.len(), device).map_err(|e| e.to_string())?;
    tensor_vec(ops::silu(&g).and_then(|s| s.mul(&u)).map_err(|e| e.to_string())?)
}

fn rope(data: &mut [f32], positions: &[usize], heads: usize, head_dim: usize, inv: &[f32], rows: usize) -> Result<(), String> {
    if head_dim % 2 != 0 || inv.len() != head_dim / 2 || data.len() != rows * heads * head_dim || positions.len() != rows {
        return Err("RoPE shape mismatch".into());
    }
    let half = head_dim / 2;
    for r in 0..rows { for h in 0..heads {
        let base = (r * heads + h) * head_dim;
        for i in 0..half {
            let angle = positions[r] as f32 * inv[i];
            let (sin, cos) = angle.sin_cos();
            let a = data[base + i]; let b = data[base + i + half];
            data[base + i] = a * cos - b * sin;
            data[base + i + half] = a * sin + b * cos;
        }
    }}
    Ok(())
}

fn attention(q: &[f32], k: &[f32], v: &[f32], ns: usize, tk: usize, heads: usize, kv_heads: usize, hd: usize, q_offset: usize, scale: f32, device: &Device) -> Result<Vec<f32>, String> {
    if q_offset + ns > tk { return Err(format!("attention: {ns} queries from position {q_offset} see past the {tk} cached keys")); }
    let mut out = vec![0.0f32; ns * heads * hd];
    for h in 0..heads {
        let kh = h / (heads / kv_heads);
        let khv: Vec<f32> = (0..tk).flat_map(|i| k[(kh * tk + i) * hd..(kh * tk + i + 1) * hd].iter().copied()).collect();
        let vhv: Vec<f32> = (0..tk).flat_map(|i| v[(kh * tk + i) * hd..(kh * tk + i + 1) * hd].iter().copied()).collect();
        // Query i at absolute position q_offset + i sees keys 0..=q_offset + i
        // and nothing else: no masked tail, so its sums run over the same
        // keys in the same shapes whether it is decoded or prefilled.
        for i in 0..ns {
            let seen = q_offset + i + 1;
            let kt = Tensor::from_slice(&khv[..seen * hd], (seen, hd), device).and_then(|t| t.transpose(0, 1)).map_err(|e| e.to_string())?;
            let qrow = &q[(i * heads + h) * hd..(i * heads + h + 1) * hd];
            let scores: Vec<f32> = matmul_rows(qrow, 1, hd, &kt, device)?.into_iter().map(|x| x * scale).collect();
            let scores = Tensor::from_vec(scores, (1, seen), device).map_err(|e| e.to_string())?;
            let probs = tensor_vec(ops::softmax(&scores, D::Minus1).map_err(|e| e.to_string())?)?;
            let vt = Tensor::from_slice(&vhv[..seen * hd], (seen, hd), device).map_err(|e| e.to_string())?;
            out[(i * heads + h) * hd..(i * heads + h + 1) * hd].copy_from_slice(&matmul_rows(&probs, 1, seen, &vt, device)?);
        }
    }
    Ok(out)
}

pub fn write_kv(pool: &mut KvPool, layer: u32, seq: &ForwardSeq, k: &[f32], v: &[f32]) -> Result<(), String> {
    let cfg = pool.cfg();
    let (bt, hkv, hd) = (cfg.block_tokens as usize, cfg.n_kv_heads as usize, cfg.head_dim as usize);
    let end = seq.start + seq.tokens.len();
    if seq.blocks.len() * bt < end { return Err(format!("forward: {} blocks cover {} positions, the step needs {end}", seq.blocks.len(), seq.blocks.len() * bt)); }
    for (src, is_v) in [(k, false), (v, true)] {
        for r in 0..seq.tokens.len() { let p = seq.start + r; let (blk, slot) = (seq.blocks[p / bt], p % bt);
            let slab = pool.slab_mut(blk, layer, is_v).map_err(|e| e.to_string())?;
            for h in 0..hkv {
                let dst = &mut slab[(h * bt + slot) * hd..(h * bt + slot + 1) * hd];
                let row = &src[r * hkv * hd + h * hd..r * hkv * hd + (h + 1) * hd];
                for (d, x) in dst.iter_mut().zip(row) { *d = f32_to_f16(*x); }
            }
        }
    }
    if layer == 0 { for b in seq.start / bt..end.div_ceil(bt) { let fill = (end - b * bt).min(bt) as u32; if pool.fill(seq.blocks[b]) < fill { pool.set_fill(seq.blocks[b], fill).map_err(|e| e.to_string())?; } } }
    Ok(())
}

pub fn gather_kv(pool: &KvPool, layer: u32, blocks: &[u32], tk: usize, is_v: bool) -> Result<Vec<f32>, String> {
    let cfg = pool.cfg();
    let (bt, hkv, hd) = (cfg.block_tokens as usize, cfg.n_kv_heads as usize, cfg.head_dim as usize);
    let mut out = vec![0.0f32; hkv * tk * hd];
    for b in 0..tk.div_ceil(bt) { let slab = pool.slab(blocks[b], layer, is_v).map_err(|e| e.to_string())?;
        for slot in 0..bt.min(tk - b * bt) { let p = b * bt + slot; for h in 0..hkv {
            let src = &slab[(h * bt + slot) * hd..(h * bt + slot + 1) * hd];
            let dst = &mut out[(h * tk + p) * hd..(h * tk + p + 1) * hd];
            for (d,s) in dst.iter_mut().zip(src) { *d = f16_to_f32(*s); }
        }}
    }
    Ok(out)
}

pub fn llama_forward(w: &LlamaWeights, cfg: &ModelConfig, inv_freq: &[f32], device: &Device, pool: &mut KvPool, batch: &ForwardBatch, hidden: Option<&mut Vec<f32>>) -> Result<Logits, String> {
    let d = cfg.hidden_size; let (h, hkv, hd) = (cfg.n_heads, cfg.n_kv_heads, cfg.head_dim);
    if hkv == 0 || h % hkv != 0 { return Err("forward: invalid grouped-query attention shape".into()); }
    let mut ids = Vec::new(); let mut positions = Vec::new(); let mut offsets = Vec::with_capacity(batch.seqs.len());
    for s in &batch.seqs {
        if s.tokens.is_empty() { return Err("forward: a sequence with no new tokens".into()); }
        if s.start + s.tokens.len() > cfg.max_position_embeddings { return Err(format!("forward: position {} is past the context of {}", s.start + s.tokens.len(), cfg.max_position_embeddings)); }
        offsets.push(ids.len());
        for (i,&t) in s.tokens.iter().enumerate() { if t as usize >= cfg.vocab_size { return Err(format!("forward: token id {t} outside vocab {}", cfg.vocab_size)); } ids.push(t); positions.push(s.start + i); }
    }
    let n = ids.len();
    let embedding = Tensor::from_vec(w.embed.clone(), (cfg.vocab_size, d), device).map_err(|e| e.to_string())?;
    let index = Tensor::new(ids, device).map_err(|e| e.to_string())?;
    let embed = candle_nn::Embedding::new(embedding, d).forward(&index).map_err(|e| e.to_string())?;
    let mut x = tensor_vec(embed)?;
    for (li,lw) in w.layers.iter().enumerate() {
        let layer = li as u32;
        let normed = rms_norm(&x, &lw.input_norm, n, d, cfg.rms_norm_eps, device)?;
        let mut q = project(&lw.q, &normed, n, device)?; let mut k = project(&lw.k, &normed, n, device)?; let v = project(&lw.v, &normed, n, device)?;
        rope(&mut q, &positions, h, hd, inv_freq, n)?; rope(&mut k, &positions, hkv, hd, inv_freq, n)?;
        let mut attn = vec![0.0f32; n * h * hd];
        for (s,&off) in batch.seqs.iter().zip(&offsets) {
            let ns = s.tokens.len();
            write_kv(pool, layer, s, &k[off * hkv * hd..(off + ns) * hkv * hd], &v[off * hkv * hd..(off + ns) * hkv * hd])?;
            let tk = s.start + ns; let kc = gather_kv(pool, layer, s.blocks, tk, false)?; let vc = gather_kv(pool, layer, s.blocks, tk, true)?;
            let qs: Vec<f32> = (0..ns).flat_map(|i| q[(off + i) * h * hd..(off + i + 1) * h * hd].iter().copied()).collect();
            let o = attention(&qs, &kc, &vc, ns, tk, h, hkv, hd, s.start, 1.0 / (hd as f32).sqrt(), device)?;
            for i in 0..ns { attn[(off + i) * h * hd..(off + i + 1) * h * hd].copy_from_slice(&o[i * h * hd..(i + 1) * h * hd]); }
        }
        let ao = project(&lw.o, &attn, n, device)?; add_assign(&mut x, &ao)?;
        let normed = rms_norm(&x, &lw.post_norm, n, d, cfg.rms_norm_eps, device)?;
        let g = project(&lw.gate, &normed, n, device)?; let u = project(&lw.up, &normed, n, device)?;
        let m = silu_mul(&g, &u, device)?; let dn = project(&lw.down, &m, n, device)?; add_assign(&mut x, &dn)?;
    }
    if let Some(out) = hidden { *out = rms_norm(&x, &w.norm, n, d, cfg.rms_norm_eps, device)?; }
    let rows = batch.seqs.len(); let mut last = vec![0.0f32; rows * d];
    for (r,(s,&off)) in batch.seqs.iter().zip(&offsets).enumerate() { let i = off + s.tokens.len() - 1; last[r*d..(r+1)*d].copy_from_slice(&x[i*d..(i+1)*d]); }
    let normed = rms_norm(&last, &w.norm, rows, d, cfg.rms_norm_eps, device)?;
    let data = match &w.lm_head { Some(head) => project(head, &normed, rows, device)?, None => {
        let embed = Tensor::from_vec(w.embed.clone(), (cfg.vocab_size,d), device).map_err(|e| e.to_string())?;
        matmul_rows(&normed, rows, d, &embed.transpose(0,1).map_err(|e| e.to_string())?, device)?
    }};
    Ok(Logits { vocab: cfg.vocab_size, data })
}

pub fn bigram_forward(w: &[f32], batch: &ForwardBatch, device: &Device) -> Result<Logits, String> {
    let rows = batch.seqs.len(); let mut onehot = vec![0.0f32; rows * 256];
    for (r,s) in batch.seqs.iter().enumerate() { let id = *s.tokens.last().ok_or("forward: a sequence with no new tokens")? as usize; if id >= 256 { return Err(format!("forward: token id {id} outside the byte vocabulary")); } onehot[r*256+id] = 1.0; }
    let a = Tensor::from_vec(onehot, (rows,256), device).map_err(|e| e.to_string())?;
    let b = Tensor::from_vec(w.to_vec(), (256,256), device).map_err(|e| e.to_string())?;
    Ok(Logits { vocab: 256, data: tensor_vec(a.matmul(&b).map_err(|e| e.to_string())?)? })
}
