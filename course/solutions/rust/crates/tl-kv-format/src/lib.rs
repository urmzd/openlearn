//! File-boundary KV envelope migration helper (craft.13).
//! This crate has no C ABI or language binding. Engines exchange envelopes
//! over the documented transfer protocol; this tool upgrades stored/exported
//! v1 envelopes to v2 for staged deployments and parity checks.

use std::fmt;

pub const V1: u16 = 1;
pub const V2: u16 = 2;
const HEADER: usize = 28;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape { pub block_tokens: u32, pub layers: u32, pub kv_heads: u32, pub head_dim: u32 }
#[derive(Clone, Debug, PartialEq)]
pub struct Block { pub hash: u64, pub n_tokens: u32, pub values: Vec<f32> }
#[derive(Clone, Debug, PartialEq)]
pub struct Envelope { pub version: u16, pub shape: Shape, pub blocks: Vec<Block> }
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error { Truncated, Magic, Version, Dtype, Shape, Length, Checksum, Tokens, Overflow, NonFinite }
impl fmt::Display for Error { fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "KV envelope: {self:?}") } }
impl std::error::Error for Error {}

fn product(s: Shape) -> Result<usize, Error> {
    [s.layers as usize, 2, s.kv_heads as usize, s.block_tokens as usize, s.head_dim as usize]
        .into_iter().try_fold(1usize, |a,b| a.checked_mul(b).ok_or(Error::Overflow))

}
fn slab_count(s: Shape) -> Result<usize, Error> {
        (s.layers as usize).checked_mul(2).and_then(|x| x.checked_mul(s.kv_heads as usize)).ok_or(Error::Overflow) 
}
fn u16at(b:&[u8],i:usize)->u16 {
        u16::from_le_bytes([b[i],b[i+1]]) 
}
fn u32at(b:&[u8],i:usize)->u32 {
        u32::from_le_bytes(b[i..i+4].try_into().unwrap()) 
}
fn u64at(b:&[u8],i:usize)->u64 {
        u64::from_le_bytes(b[i..i+8].try_into().unwrap()) 
}
fn crc32c(bytes:&[u8])->u32 {
        let mut c=!0u32; for &b in bytes { c^=b as u32; for _ in 0..8 { c=if c&1!=0 {(c>>1)^0x82f63b78} else {c>>1}; } } !c 
}
fn read_f16(h:u16)->f32 {
        let sign=if h&0x8000==0 {1.0} else {-1.0}; let e=(h>>10)&31; let m=h&1023; match e { 0 if m==0=>sign*0.0, 0=>sign*(m as f32)*2f32.powi(-24), 31 if m==0=>sign*f32::INFINITY, 31=>f32::NAN, _=>sign*(1.0+m as f32/1024.0)*2f32.powi(e as i32-15) } 
}
fn write_f16(x:f32)->Result<u16,Error> {
    if !x.is_finite(){return Err(Error::NonFinite)}
    let bits=x.to_bits(); let sign=((bits>>16)&0x8000) as u16; let e=((bits>>23)&255) as i32-127+15; let m=bits&0x7fffff;
    if e>=31 { return Err(Error::NonFinite) }
    if e<=0 { if e < -10 { return Ok(sign) } let v=(m|0x800000)>>(1-e); let rounded=v+0xfff+((v>>13)&1); return Ok(sign|((rounded>>13) as u16)); }
    let rounded=m+0xfff+((m>>13)&1); let mut he=e; let mut hm=rounded>>13; if hm&0x400!=0 {he+=1;hm=0;} if he>=31{return Err(Error::NonFinite)} Ok(sign|((he as u16)<<10)|(hm as u16))

}
fn e4_decode(b:u8)->f32 {
        let sign=if b&0x80==0 {1.0} else {-1.0}; let e=(b>>3)&15; let m=b&7; if e==15&&m==7 {return f32::NAN} if e==0 {sign*(m as f32/8.0)*2f32.powi(-6)} else {sign*(1.0+m as f32/8.0)*2f32.powi(e as i32-7)} 
}
fn e4_encode(x:f32)->Result<u8,Error> {
        if !x.is_finite(){return Err(Error::NonFinite)} let sign=if x.is_sign_negative(){0x80}else{0}; let a=x.abs().min(448.0); let mut best=0u8; let mut err=f32::INFINITY; for b in 0..=0x7e {let v=e4_decode(b); let d=(v-a).abs(); if d<err || (d==err && b&1==0) {best=b;err=d}} Ok(sign|best) 
}

/// Decode and fully validate an envelope before returning its blocks.
pub fn decode(bytes:&[u8], expected:Shape)->Result<Envelope,Error> {
    if bytes.len()<HEADER+4{return Err(Error::Truncated)} if &bytes[..4]!=b"TLKV"{return Err(Error::Magic)}
    let version=u16at(bytes,4); let dtype=u16at(bytes,6); if version!=V1&&version!=V2{return Err(Error::Version)} if dtype!=if version==V1{1}else{3}{return Err(Error::Dtype)}
    let n=u32at(bytes,8) as usize; let shape=Shape{block_tokens:u32at(bytes,12),layers:u32at(bytes,16),kv_heads:u32at(bytes,20),head_dim:u32at(bytes,24)}; if shape!=expected{return Err(Error::Shape)}
    let elems=product(shape)?; let slabs=slab_count(shape)?; let payload=if version==V1 { elems.checked_mul(2).ok_or(Error::Overflow)? } else { elems.checked_add(slabs.checked_mul(4).ok_or(Error::Overflow)?).ok_or(Error::Overflow)? };
    let rec=12usize.checked_add(payload).ok_or(Error::Overflow)?; let len=HEADER.checked_add(n.checked_mul(rec).ok_or(Error::Overflow)?).and_then(|x|x.checked_add(4)).ok_or(Error::Overflow)?; if len!=bytes.len(){return Err(Error::Length)} if crc32c(&bytes[..len-4])!=u32at(bytes,len-4){return Err(Error::Checksum)}
    let slab_elems=elems/slabs; let mut blocks=Vec::with_capacity(n); let mut at=HEADER;
    for _ in 0..n { let hash=u64at(bytes,at);let fill=u32at(bytes,at+8);if fill==0||fill>shape.block_tokens||(hash!=0&&fill!=shape.block_tokens){return Err(Error::Tokens)} let p=&bytes[at+12..at+12+payload];let mut values=Vec::with_capacity(elems);
        if version==V1 { for x in p.chunks_exact(2){values.push(read_f16(u16::from_le_bytes([x[0],x[1]])))} }
        else {for (i,&b) in p[..elems].iter().enumerate(){let scale=f32::from_le_bytes(p[elems+(i/slab_elems)*4..elems+(i/slab_elems+1)*4].try_into().unwrap()); if !scale.is_finite()||scale<=0.0{return Err(Error::Length)} values.push(e4_decode(b)*scale)} }
        blocks.push(Block{hash,n_tokens:fill,values});at+=rec;
    } Ok(Envelope{version,shape,blocks})

}
/// Encode decoded values in the requested format. v2 stores one amax/448 scale per (layer, K/V, head) slab.
pub fn encode(env:&Envelope, version:u16)->Result<Vec<u8>,Error> {
    if version!=V1&&version!=V2{return Err(Error::Version)} let s=env.shape; let elems=product(s)?;let slabs=slab_count(s)?;let se=elems/slabs; let payload=if version==V1{elems.checked_mul(2).ok_or(Error::Overflow)?}else{elems.checked_add(slabs*4).ok_or(Error::Overflow)?};
    let cap=HEADER.checked_add(env.blocks.len().checked_mul(12+payload).ok_or(Error::Overflow)?).and_then(|x|x.checked_add(4)).ok_or(Error::Overflow)?;let mut out=Vec::with_capacity(cap);out.extend_from_slice(b"TLKV");out.extend_from_slice(&version.to_le_bytes());out.extend_from_slice(&(if version==V1{1u16}else{3u16}).to_le_bytes());out.extend_from_slice(&(env.blocks.len() as u32).to_le_bytes());for x in [s.block_tokens,s.layers,s.kv_heads,s.head_dim]{out.extend_from_slice(&x.to_le_bytes())}
    for block in &env.blocks {if block.values.len()!=elems||block.n_tokens==0||block.n_tokens>s.block_tokens||(block.hash!=0&&block.n_tokens!=s.block_tokens){return Err(Error::Tokens)}out.extend_from_slice(&block.hash.to_le_bytes());out.extend_from_slice(&block.n_tokens.to_le_bytes());
        if version==V1 {for (i,&x) in block.values.iter().enumerate(){let pos=(i%se)/(s.head_dim as usize);let v=if pos>=block.n_tokens as usize{0.0}else{x};out.extend_from_slice(&write_f16(v)?.to_le_bytes())}}
        else {let mut scales=Vec::with_capacity(slabs);for slab in 0..slabs {let x=&block.values[slab*se..(slab+1)*se];let max=x.iter().map(|v|v.abs()).fold(0.0f32,f32::max);scales.push(if max==0.0{1.0}else{max/448.0});}for (i,&x) in block.values.iter().enumerate(){let pos=(i%se)/(s.head_dim as usize);let v=if pos>=block.n_tokens as usize{0.0}else{x};out.push(e4_encode(v/scales[i/se])?)}for scale in scales{out.extend_from_slice(&scale.to_le_bytes())}
        }
    }let crc=crc32c(&out);out.extend_from_slice(&crc.to_le_bytes());Ok(out)

}
/// Converts a complete, validated v1 envelope to v2 without losing block identity or fill counts.
pub fn upgrade_v1(bytes:&[u8],shape:Shape)->Result<Vec<u8>,Error>{
        let env=decode(bytes,shape)?;if env.version!=V1{return Err(Error::Version)}encode(&env,V2)
}
