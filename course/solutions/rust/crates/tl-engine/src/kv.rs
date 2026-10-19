//! Rust-owned, bounded KV cache for the inference engine (L10.1).
//!
//! The cache stores K and V slabs as f16 bit patterns, but owns all memory
//! itself. A free block has no references, a used block has a positive
//! reference count, and a registered zero-reference block remains cached
//! until allocation evicts it. The crate never crosses a C ABI.

use std::collections::{HashMap, VecDeque};
use std::fmt;

pub const TL_KV_FORMAT_V1: u32 = 1;
pub const TL_F16: i32 = 1;
pub const TL_EINVAL: i32 = 1;
pub const TL_ENOMEM: i32 = 2;
pub const TL_EFULL: i32 = 5;
pub const TL_EFORMAT: i32 = 7;
pub const TL_EBUSY: i32 = 8;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KvCfg {
    pub n_blocks: u32,
    pub block_tokens: u32,
    pub n_layers: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub dtype: i32,
    pub format: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KvStats {
    pub free: u32,
    pub used: u32,
    pub cached: u32,
    pub evictions: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KvError {
    pub status: i32,
    pub name: &'static str,
    pub message: String,
}

impl fmt::Display for KvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{}: {}", self.name, self.message) }
}
impl std::error::Error for KvError {}

fn error(status: i32, name: &'static str, message: impl Into<String>) -> KvError {
    KvError { status, name, message: message.into() }
}

#[derive(Clone)]
struct Block {
    refs: u32,
    fill: u32,
    hash: Option<u64>,
    slabs: Vec<Vec<u16>>,
}

/// A bounded Rust KV pool shared under a mutex by the model runner and
/// block manager.
pub struct KvPool {
    cfg: KvCfg,
    blocks: Vec<Block>,
    lru: VecDeque<u32>,
    by_hash: HashMap<u64, u32>,
    evictions: u32,
}

impl KvPool {
    pub fn new(cfg: KvCfg) -> Result<Self, KvError> {
        if cfg.n_blocks == 0 || cfg.block_tokens == 0 || cfg.n_layers == 0 || cfg.n_kv_heads == 0 || cfg.head_dim == 0 {
            return Err(error(TL_EINVAL, "KV_EINVAL", "all KV dimensions must be positive"));
        }
        if cfg.dtype != TL_F16 || cfg.format != TL_KV_FORMAT_V1 {
            return Err(error(TL_EINVAL, "KV_EINVAL", "the engine KV cache uses f16 format v1"));
        }
        let slab_len = cfg.n_kv_heads as usize * cfg.block_tokens as usize * cfg.head_dim as usize;
        let block = Block { refs: 0, fill: 0, hash: None, slabs: (0..cfg.n_layers as usize * 2).map(|_| vec![0; slab_len]).collect() };
        Ok(Self { cfg, blocks: vec![block; cfg.n_blocks as usize], lru: VecDeque::new(), by_hash: HashMap::new(), evictions: 0 })
    }

    pub fn cfg(&self) -> KvCfg { self.cfg }

    pub fn alloc(&mut self, n: usize) -> Result<Vec<u32>, KvError> {
        let free = self.stats().free as usize;
        if free + self.lru.len() < n { return Err(error(TL_EFULL, "KV_EFULL", format!("need {n} blocks, {} available", free + self.lru.len()))); }
        let mut ids = Vec::with_capacity(n);
        for i in 0..self.blocks.len() {
            if ids.len() == n { break; }
            if self.blocks[i].refs == 0 && self.blocks[i].hash.is_none() { ids.push(i as u32); }
        }
        while ids.len() < n {
            let id = self.lru.pop_front().expect("capacity checked");
            let block = &mut self.blocks[id as usize];
            if let Some(hash) = block.hash.take() { self.by_hash.remove(&hash); }
            block.fill = 0;
            self.evictions += 1;
            ids.push(id);
        }
        for &id in &ids { self.blocks[id as usize].refs = 1; self.blocks[id as usize].fill = 0; }
        Ok(ids)
    }

    pub fn retain(&mut self, id: u32) -> Result<(), KvError> {
        let Some(b) = self.blocks.get_mut(id as usize) else { return Err(error(TL_EINVAL,"KV_EINVAL",format!("block {id} outside pool"))); };
        if b.refs == 0 { self.lru.retain(|&x| x != id); }
        b.refs = b.refs.checked_add(1).ok_or_else(|| error(TL_EINVAL,"KV_EINVAL","reference count overflow"))?;
        Ok(())
    }

    pub fn release(&mut self, id: u32) -> Result<(), KvError> {
        let Some(b) = self.blocks.get_mut(id as usize) else { return Err(error(TL_EINVAL,"KV_EINVAL",format!("block {id} outside pool"))); };
        if b.refs == 0 { return Err(error(TL_EINVAL,"KV_EINVAL",format!("block {id} is not in use"))); }
        b.refs -= 1;
        if b.refs == 0 {
            if b.hash.is_some() { self.lru.push_back(id); } else { b.fill = 0; }
        }
        Ok(())
    }

    pub fn cow(&mut self, id: u32) -> Result<u32, KvError> {
        let Some(block) = self.blocks.get(id as usize) else { return Err(error(TL_EINVAL,"KV_EINVAL",format!("block {id} outside pool"))); };
        if block.refs == 1 { return Ok(id); }
        let src = block.clone();
        let new_id = self.alloc(1)?[0];
        self.blocks[new_id as usize] = Block { refs: 1, fill: src.fill, hash: None, slabs: src.slabs };
        self.release(id)?;
        Ok(new_id)
    }

    pub fn set_fill(&mut self, id: u32, n_tokens: u32) -> Result<(), KvError> {
        let Some(b) = self.blocks.get_mut(id as usize) else { return Err(error(TL_EINVAL,"KV_EINVAL",format!("block {id} outside pool"))); };
        if b.refs == 0 || n_tokens > self.cfg.block_tokens { return Err(error(TL_EINVAL,"KV_EINVAL",format!("invalid fill {n_tokens} for block {id}"))); }
        b.fill = n_tokens;
        Ok(())
    }

    pub fn fill(&self, id: u32) -> u32 { self.blocks.get(id as usize).map_or(0, |b| b.fill) }

    pub fn register(&mut self, id: u32, hash: u64) -> Result<bool, KvError> {
        let Some(b) = self.blocks.get_mut(id as usize) else { return Err(error(TL_EINVAL,"KV_EINVAL",format!("block {id} outside pool"))); };
        if b.refs == 0 || b.fill != self.cfg.block_tokens { return Err(error(TL_EINVAL,"KV_EINVAL","only a used full block can be cached")); }
        if self.by_hash.contains_key(&hash) { return Ok(false); }
        b.hash = Some(hash);
        self.by_hash.insert(hash, id);
        Ok(true)
    }

    pub fn lookup(&mut self, hash: u64) -> Option<u32> {
        let id = *self.by_hash.get(&hash)?;
        self.lru.retain(|&x| x != id);
        self.blocks[id as usize].refs += 1;
        Some(id)
    }

    pub fn slab_len(&self) -> usize { self.cfg.n_kv_heads as usize * self.cfg.block_tokens as usize * self.cfg.head_dim as usize }
    pub fn block_bytes(&self) -> usize { self.slab_len() * self.cfg.n_layers as usize * 2 * 2 }

    pub fn slab(&self, id: u32, layer: u32, is_v: bool) -> Result<&[u16], KvError> {
        let b = self.blocks.get(id as usize).ok_or_else(|| error(TL_EINVAL,"KV_EINVAL",format!("block {id} outside pool")))?;
        let i = layer as usize * 2 + usize::from(is_v);
        b.slabs.get(i).map(Vec::as_slice).ok_or_else(|| error(TL_EINVAL,"KV_EINVAL",format!("layer {layer} outside pool")))
    }

    pub fn slab_mut(&mut self, id: u32, layer: u32, is_v: bool) -> Result<&mut [u16], KvError> {
        let i = layer as usize * 2 + usize::from(is_v);
        let b = self.blocks.get_mut(id as usize).ok_or_else(|| error(TL_EINVAL,"KV_EINVAL",format!("block {id} outside pool")))?;
        b.slabs.get_mut(i).map(Vec::as_mut_slice).ok_or_else(|| error(TL_EINVAL,"KV_EINVAL",format!("layer {layer} outside pool")))
    }

    pub fn stats(&self) -> KvStats {
        let cached = self.lru.len() as u32;
        let used = self.blocks.iter().filter(|b| b.refs > 0).count() as u32;
        KvStats { free: self.cfg.n_blocks - cached - used, used, cached, evictions: self.evictions }
    }

    pub fn export(&self, ids: &[u32]) -> Result<Vec<u8>, KvError> {
        let mut out = Vec::with_capacity(32 + ids.len() * (12 + self.block_bytes()));
        out.extend_from_slice(b"TLKV");
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&(TL_F16 as u16).to_le_bytes());
        out.extend_from_slice(&(ids.len() as u32).to_le_bytes());
        for n in [self.cfg.block_tokens, self.cfg.n_layers, self.cfg.n_kv_heads, self.cfg.head_dim] { out.extend_from_slice(&n.to_le_bytes()); }
        for &id in ids {
            let b = self.blocks.get(id as usize).ok_or_else(|| error(TL_EINVAL,"KV_EINVAL",format!("block {id} outside pool")))?;
            out.extend_from_slice(&if b.fill == self.cfg.block_tokens { b.hash.unwrap_or(0) } else { 0 }.to_le_bytes());
            out.extend_from_slice(&b.fill.to_le_bytes());
            for slab in &b.slabs { for h in slab { out.extend_from_slice(&h.to_le_bytes()); } }
        }
        let crc = crc32c(&out, 0);
        out.extend_from_slice(&crc.to_le_bytes());
        Ok(out)
    }

    pub fn import(&mut self, buf: &[u8]) -> Result<Vec<u32>, KvError> {
        if buf.len() < 32 || &buf[..4] != b"TLKV" { return Err(error(TL_EFORMAT,"KV_EFORMAT","bad or short KV envelope")); }
        let u16at = |i| u16::from_le_bytes([buf[i], buf[i + 1]]);
        let u32at = |i| u32::from_le_bytes([buf[i], buf[i + 1], buf[i + 2], buf[i + 3]]);
        let n = u32at(8) as usize;
        if u16at(4) != 1 || u16at(6) != TL_F16 as u16 || u32at(12) != self.cfg.block_tokens || u32at(16) != self.cfg.n_layers || u32at(20) != self.cfg.n_kv_heads || u32at(24) != self.cfg.head_dim {
            return Err(error(TL_EFORMAT,"KV_EFORMAT","unsupported version, dtype, or dimensions"));
        }
        let record = 12 + self.block_bytes();
        if buf.len() != 28 + n * record + 4 { return Err(error(TL_EFORMAT,"KV_EFORMAT","envelope length does not match its block count")); }
        let got_crc = u32at(buf.len() - 4);
        if crc32c(&buf[..buf.len() - 4], 0) != got_crc { return Err(error(TL_EFORMAT,"KV_EFORMAT","CRC-32C mismatch")); }
        let ids = self.alloc(n)?;
        let slab_len = self.slab_len();
        for (i, &id) in ids.iter().enumerate() {
            let off = 28 + i * record;
            let hash = u64::from_le_bytes(buf[off..off + 8].try_into().expect("checked envelope"));
            let fill = u32::from_le_bytes(buf[off + 8..off + 12].try_into().expect("checked envelope"));
            if fill > self.cfg.block_tokens || (hash != 0 && fill != self.cfg.block_tokens) {
                for &allocated in &ids { let _ = self.release(allocated); }
                return Err(error(TL_EFORMAT,"KV_EFORMAT","invalid fill or hash on a partial block"));
            }
            self.blocks[id as usize].fill = fill;
            let mut cursor = off + 12;
            for layer in 0..self.cfg.n_layers {
                for is_v in [false, true] {
                    let slab = self.slab_mut(id, layer, is_v)?;
                    for h in slab.iter_mut().take(slab_len) {
                        *h = u16::from_le_bytes([buf[cursor], buf[cursor + 1]]);
                        cursor += 2;
                    }
                }
            }
            if hash != 0 { self.register(id, hash)?; }
        }
        Ok(ids)
    }

    /// FNV-1a chained token-block hash from formats/kv-block.md.
    pub fn block_hash(parent: u64, tokens: &[u32]) -> u64 {
        let mut h = 0xcbf29ce484222325u64;
        for b in parent.to_le_bytes().into_iter().chain(tokens.iter().flat_map(|t| t.to_le_bytes())) { h = (h ^ u64::from(b)).wrapping_mul(0x100000001b3); }
        if h == 0 { 1 } else { h }
    }
}

pub fn crc32c(data: &[u8], crc: u32) -> u32 {
    let mut value = !crc;
    for &byte in data { value ^= byte as u32; for _ in 0..8 { value = if value & 1 == 1 { (value >> 1) ^ 0x82f63b78 } else { value >> 1 }; } }
    !value
}

/// Chained block hash used by the block manager.
pub fn kv_block_hash(parent: u64, tokens: &[u32]) -> u64 { KvPool::block_hash(parent, tokens) }
