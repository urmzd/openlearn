// Parity driver for `kv.wire.v1`, implementation `rust` (L10.6): one JSON case
// per stdin line ({B, L, H, D, tokens, values}), one {"hex": envelope} per
// stdout line, written by the Rust envelope writer of tl-engine's
// kv_transfer (no pool, no C): the tokens cut into blocks of B, each full
// block hashed with the chained FNV-1a of formats/kv-block.md, values (f16
// bit patterns, row-major [position][layer][K, V][head][dim]) laid out per
// block as [L][2][H][B][D] with zeros past the fill. The bytes compare with
// the golden blobs, the C exporter (drivers/kv_wire_v1.c), and the Python
// transcription.
include!("../parity/ol_parity.rs");

use tl_engine::kv_transfer::{prompt_hashes, write_envelope, BlockRecord, EnvHeader};

fn main() {
    for line in ssp_lines() {
        if line.trim().is_empty() {
            continue;
        }
        let b = ssp_u64(&line, "B").expect("B") as usize;
        let l = ssp_u64(&line, "L").expect("L") as usize;
        let h = ssp_u64(&line, "H").expect("H") as usize;
        let d = ssp_u64(&line, "D").expect("D") as usize;
        let toks: Vec<u32> = ssp_f64s(&line, "tokens").expect("tokens").iter().map(|&x| x as u32).collect();
        let vals: Vec<u16> = ssp_f64s(&line, "values").expect("values").iter().map(|&x| x as u16).collect();
        let n = toks.len();
        assert_eq!(vals.len(), n * l * 2 * h * d, "values must be [n][L][2][H][D]");
        let nb = n.div_ceil(b);
        let hashes = prompt_hashes(&toks, b);
        let mut recs = Vec::with_capacity(nb);
        for blk in 0..nb {
            let fill = (n - blk * b).min(b);
            let mut payload = Vec::with_capacity(l * 2 * h * b * d * 2);
            for li in 0..l {
                for kv in 0..2 {
                    for hi in 0..h {
                        for slot in 0..b {
                            for di in 0..d {
                                let pos = blk * b + slot;
                                let x = if slot < fill { vals[(((pos * l + li) * 2 + kv) * h + hi) * d + di] } else { 0 };
                                payload.extend_from_slice(&x.to_le_bytes());
                            }
                        }
                    }
                }
            }
            recs.push(BlockRecord { hash: hashes.get(blk).copied().unwrap_or(0), n_tokens: fill as u32, payload });
        }
        let hdr = EnvHeader { version: 1, dtype: 1, n_blocks: nb as u32, block_tokens: b as u32, n_layers: l as u32, n_kv_heads: h as u32, head_dim: d as u32 };
        let hex: String = write_envelope(&hdr, &recs).iter().map(|x| format!("{x:02x}")).collect();
        println!("{{\"hex\":\"{hex}\"}}");
    }
}
