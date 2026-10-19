//! Low-precision numbers the engine reads and writes (L10.1).
//!
//! - f16 and bf16 bit patterns to f32 and back: checkpoints store weights as
//!   BF16 (SmolLM2) or F16, and KV format v1 stores K and V as f16
//!   (formats/kv-block.md). Conversions to 16 bits round to nearest, ties to
//!   even, as M09.4 teaches and numpy's `astype(np.float16)` does.
//! - int4 group quantization (contracts/py/tinyllm/infer/quant.pyi, L8.5):
//!   W[r, c] ~ q[r, c] * s[r, c / group], q in [-8, 7], s = f16(amax / 7),
//!   q = rint(W / s) with the f16 scale actually stored; packed two per byte,
//!   the even column in the low nibble (formats/safetensors.md).
//! - [`QLinear`]: an int4 linear layer that unpacks and widens its weights
//!   before applying them with Candle tensor operations.

use candle_core::{Device, Tensor};

/// f16 bits to f32: exact (every f16 is an f32).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = match (exp, man) {
        (0, 0) => sign,
        (0, m) => {
            // subnormal: m * 2^-24, renormalized
            let mut e: i32 = -14;
            let mut m = m;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (((e + 127) as u32) << 23) | ((m & 0x3ff) << 13)
        }
        (0x1f, 0) => sign | 0x7f80_0000,
        (0x1f, m) => sign | 0x7fc0_0000 | (m << 13),
        (e, m) => sign | ((e + 127 - 15) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

/// f32 to f16 bits, round to nearest even; overflow gives +-inf, NaN 0x7E00.
pub fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    if x.is_nan() {
        return 0x7e00;
    }
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00;
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        // subnormal or zero in f16: value = full_man * 2^(exp - 150)
        if e < -10 {
            return sign;
        }
        let full = man | 0x80_0000;
        let shift = (14 - e) as u32; // 1 - e + 13
        let half = full >> shift;
        let rem = full & ((1u32 << shift) - 1);
        let mid = 1u32 << (shift - 1);
        let r = if rem > mid || (rem == mid && half & 1 == 1) { half + 1 } else { half };
        return sign | r as u16;
    }
    let half = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    let r = if rem > 0x1000 || (rem == 0x1000 && half & 1 == 1) { half + 1 } else { half };
    // a carry out of the mantissa bumps the exponent, up to inf: still right
    sign | r as u16
}

/// bf16 bits to f32: the upper half of the f32 bits.
pub fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// Packs int values in [-8, 7] of a [rows, cols] matrix (cols even) into
/// [rows, cols / 2] bytes: byte b holds column 2b in its low nibble and
/// 2b + 1 in its high nibble, each in 4-bit two's complement.
pub fn pack_int4(q: &[i8], rows: usize, cols: usize) -> Result<Vec<u8>, String> {
    if cols % 2 != 0 || q.len() != rows * cols {
        return Err(format!("pack_int4: {} values for [{rows}, {cols}] (cols must be even)", q.len()));
    }
    if let Some(&bad) = q.iter().find(|&&v| !(-8..=7).contains(&v)) {
        return Err(format!("pack_int4: {bad} outside [-8, 7]"));
    }
    Ok(q.chunks_exact(2).map(|p| ((p[0] as u8) & 0x0f) | (((p[1] as u8) & 0x0f) << 4)).collect())
}

/// The inverse of [`pack_int4`]: [rows, 2 * bytes per row] signed values.
pub fn unpack_int4(packed: &[u8]) -> Vec<i8> {
    let nib = |v: u8| -> i8 {
        let v = (v & 0x0f) as i8;
        if v >= 8 {
            v - 16
        } else {
            v
        }
    };
    packed.iter().flat_map(|&b| [nib(b), nib(b >> 4)]).collect()
}

/// An int4 group-quantized linear weight W [out, inp] (W @ x computes
/// `inp` -> `out`): `qweight` [out, inp / 2], `scales` [out, inp / group] f16.
#[derive(Clone, Debug, PartialEq)]
pub struct QLinear {
    pub out: usize,
    pub inp: usize,
    pub group: usize,
    pub qweight: Vec<u8>,
    pub scales: Vec<u16>,
}

impl QLinear {
    /// Quantizes a row-major f32 weight [out, inp] as L8.5 does: per group,
    /// s = f16(amax / 7); q = rint(w / s) (ties to even) using the f16 scale
    /// actually stored, clamped to [-8, 7]; a zero scale gives q = 0.
    pub fn quantize(w: &[f32], out: usize, inp: usize, group: usize) -> Result<QLinear, String> {
        if group == 0 || group % 2 != 0 || inp % group != 0 || w.len() != out * inp {
            return Err(format!("QLinear::quantize: [{out}, {inp}] with group {group} (need even group dividing inp)"));
        }
        if w.iter().any(|v| !v.is_finite()) {
            return Err("QLinear::quantize: weights must be finite".to_string());
        }
        let groups = inp / group;
        let mut q = vec![0i8; out * inp];
        let mut scales = vec![0u16; out * groups];
        for r in 0..out {
            for g in 0..groups {
                let cols = r * inp + g * group..r * inp + (g + 1) * group;
                let amax = w[cols.clone()].iter().fold(0.0f32, |m, v| m.max(v.abs()));
                let s16 = f32_to_f16(amax / 7.0);
                scales[r * groups + g] = s16;
                let s = f16_to_f32(s16);
                if s == 0.0 {
                    continue;
                }
                for c in cols {
                    q[c] = (w[c] / s).round_ties_even().clamp(-8.0, 7.0) as i8;
                }
            }
        }
        Ok(QLinear { out, inp, group, qweight: pack_int4(&q, out, inp)?, scales })
    }

    /// W in f32: q * s for every element (for tests and the f32 fallback).
    pub fn dequantize(&self) -> Vec<f32> {
        let q = unpack_int4(&self.qweight);
        let groups = self.inp / self.group;
        (0..self.out * self.inp)
            .map(|i| {
                let (r, c) = (i / self.inp, i % self.inp);
                q[i] as f32 * f16_to_f32(self.scales[r * groups + c / self.group])
            })
            .collect()
    }

    /// y [m, out] = x [m, inp] @ W^T using Candle after groupwise
    /// dequantization at the tensor boundary.
    pub fn forward(&self, x: &[f32], m: usize, y: &mut [f32]) -> Result<(), String> {
        if x.len() != m * self.inp || y.len() != m * self.out { return Err("qlinear: input/output length mismatch".to_string()); }
        let input = Tensor::from_vec(x.to_vec(), (m, self.inp), &Device::Cpu).map_err(|e| e.to_string())?;
        let weight = Tensor::from_vec(self.dequantize(), (self.out, self.inp), &Device::Cpu).map_err(|e| e.to_string())?;
        let result = input.matmul(&weight.transpose(0, 1).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        for (dst, row) in y.chunks_exact_mut(self.out).zip(result.to_vec2::<f32>().map_err(|e| e.to_string())?) { dst.copy_from_slice(&row); }
        Ok(())
    }
}
