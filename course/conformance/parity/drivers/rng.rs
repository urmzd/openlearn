// Parity driver for `rng`, implementation `rust` (L10.1): an example of the
// harness crate ol-tests. Reads one case per stdin line
// {"seed", "seq", "n_u32", "n_uniform", "n_normal"} and prints
// {"u32": [...], "uniform": [...], "normal": [...]}, each list drawn from a
// fresh tl_engine::sample::Pcg32::new(seed, seq).
//
// Normals follow spec/pcg32.md: Box-Muller over two consecutive uniforms,
// r = sqrt(-2 ln(1 - u1)), cosine first, the sine kept as the spare. L10.1
// owns the generator; the transform is written here, as in drivers/rng.py.
include!("../parity/ol_parity.rs");

use tl_engine::sample::Pcg32;

fn normals(g: &mut Pcg32, n: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(n);
    let mut spare: Option<f64> = None;
    while out.len() < n {
        if let Some(s) = spare.take() {
            out.push(s);
            continue;
        }
        let (u1, u2) = (g.uniform_f64(), g.uniform_f64());
        let r = (-2.0 * (1.0 - u1).ln()).sqrt();
        out.push(r * (2.0 * std::f64::consts::PI * u2).cos());
        spare = Some(r * (2.0 * std::f64::consts::PI * u2).sin());
    }
    out
}

fn join<T: std::fmt::Display>(xs: &[T]) -> String {
    xs.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",")
}

fn main() {
    for line in ssp_lines() {
        if line.trim().is_empty() {
            continue;
        }
        let seed = ssp_u64(&line, "seed").expect("seed");
        let seq = ssp_u64(&line, "seq").unwrap_or(54);
        let n_u32 = ssp_u64(&line, "n_u32").unwrap_or(0) as usize;
        let n_uni = ssp_u64(&line, "n_uniform").unwrap_or(0) as usize;
        let n_nor = ssp_u64(&line, "n_normal").unwrap_or(0) as usize;
        let mut g = Pcg32::new(seed, seq);
        let u32s: Vec<u32> = (0..n_u32).map(|_| g.next_u32()).collect();
        let mut g = Pcg32::new(seed, seq);
        let uni: Vec<f64> = (0..n_uni).map(|_| g.uniform_f64()).collect();
        let nor = normals(&mut Pcg32::new(seed, seq), n_nor);
        println!("{{\"u32\":[{}],\"uniform\":[{}],\"normal\":[{}]}}", join(&u32s), join(&uni), join(&nor));
    }
}
