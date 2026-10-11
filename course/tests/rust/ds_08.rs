//! ds.08 course tests, Rust half: the Bloom filter (tl-ds `bloom`).
//!
//! Annotated exemplars (DESIGN 5.12). The worked example and the golden
//! cases come from contracts/formats/bloom.md, transcribed independently in
//! course/oracle/ds.08/bloom_golden.py (course/fixtures/parity/bloom.json).
//! JSON is read with the small parser in `mod j`. Random items come from the
//! PCG32 transcribed from contracts/spec/pcg32.md (the frozen generator, D35).
//! The corpus pipeline has a separate Python Bloom implementation; these
//! tests exercise the Rust module and its serialized format.

use tl_ds::bloom::{fnv1a64, mix64, optimal_k, optimal_m, predicted_fp_rate, Bloom, BloomError, HEADER_LEN};

// ---------------------------------------------------------------------------
// helpers

struct Pcg32 {
    state: u64,
    inc: u64,
}

impl Pcg32 {
    fn new(seed: u64, seq: u64) -> Pcg32 {
        let mut g = Pcg32 { state: 0, inc: (seq << 1) | 1 };
        g.next_u32();
        g.state = g.state.wrapping_add(seed);
        g.next_u32();
        g
    }
    fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(6364136223846793005).wrapping_add(self.inc);
        let xs = (((old >> 18) ^ old) >> 27) as u32;
        xs.rotate_right((old >> 59) as u32)
    }
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next_u32() as u8).collect()
    }
}

fn ol_seed() -> u64 {
    std::env::var("OL_SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(0)
}

fn fixtures() -> std::path::PathBuf {
    std::path::PathBuf::from(std::env::var("TINYLLM_FIXTURES").expect("TINYLLM_FIXTURES is set by ol"))
}

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
}

/// A minimal JSON reader for the fixtures (never your parser).
mod j {
    #[derive(Debug, Clone, PartialEq)]
    pub enum V {
        Null,
        Bool(bool),
        Num(f64),
        Str(String),
        Arr(Vec<V>),
        Obj(Vec<(String, V)>),
    }

    impl V {
        pub fn get(&self, k: &str) -> &V {
            match self {
                V::Obj(o) => o.iter().find(|(a, _)| a == k).map(|(_, v)| v).unwrap_or_else(|| panic!("no key {k}")),
                _ => panic!("not an object"),
            }
        }
        pub fn str(&self) -> &str {
            match self {
                V::Str(s) => s,
                _ => panic!("not a string"),
            }
        }
        pub fn num(&self) -> f64 {
            match self {
                V::Num(x) => *x,
                _ => panic!("not a number"),
            }
        }
        pub fn arr(&self) -> &[V] {
            match self {
                V::Arr(a) => a,
                _ => panic!("not an array"),
            }
        }
        pub fn bool(&self) -> bool {
            match self {
                V::Bool(b) => *b,
                _ => panic!("not a bool"),
            }
        }
    }

    pub fn parse(s: &str) -> V {
        let b = s.as_bytes();
        let mut i = 0;
        let v = val(b, &mut i);
        ws(b, &mut i);
        assert_eq!(i, b.len(), "trailing bytes after JSON");
        v
    }

    fn ws(b: &[u8], i: &mut usize) {
        while *i < b.len() && matches!(b[*i], b' ' | b'\n' | b'\r' | b'\t') {
            *i += 1;
        }
    }

    fn hex4(b: &[u8], i: &mut usize) -> u32 {
        let s = std::str::from_utf8(&b[*i..*i + 4]).unwrap();
        *i += 4;
        u32::from_str_radix(s, 16).unwrap()
    }

    fn val(b: &[u8], i: &mut usize) -> V {
        ws(b, i);
        match b[*i] {
            b'{' => {
                *i += 1;
                let mut o = Vec::new();
                ws(b, i);
                if b[*i] == b'}' {
                    *i += 1;
                    return V::Obj(o);
                }
                loop {
                    let k = match val(b, i) {
                        V::Str(s) => s,
                        other => panic!("object key {other:?}"),
                    };
                    ws(b, i);
                    assert_eq!(b[*i], b':');
                    *i += 1;
                    o.push((k, val(b, i)));
                    ws(b, i);
                    *i += 1;
                    match b[*i - 1] {
                        b',' => continue,
                        b'}' => return V::Obj(o),
                        c => panic!("unexpected {:?} in object", c as char),
                    }
                }
            }
            b'[' => {
                *i += 1;
                let mut a = Vec::new();
                ws(b, i);
                if b[*i] == b']' {
                    *i += 1;
                    return V::Arr(a);
                }
                loop {
                    a.push(val(b, i));
                    ws(b, i);
                    *i += 1;
                    match b[*i - 1] {
                        b',' => continue,
                        b']' => return V::Arr(a),
                        c => panic!("unexpected {:?} in array", c as char),
                    }
                }
            }
            b'"' => {
                *i += 1;
                let mut out = String::new();
                loop {
                    let start = *i;
                    while b[*i] != b'"' && b[*i] != b'\\' {
                        *i += 1;
                    }
                    out.push_str(std::str::from_utf8(&b[start..*i]).unwrap());
                    if b[*i] == b'"' {
                        *i += 1;
                        return V::Str(out);
                    }
                    *i += 1;
                    let e = b[*i];
                    *i += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let hi = hex4(b, i);
                            let cp = if (0xD800..0xDC00).contains(&hi) {
                                assert_eq!(&b[*i..*i + 2], b"\\u");
                                *i += 2;
                                0x10000 + ((hi - 0xD800) << 10) + (hex4(b, i) - 0xDC00)
                            } else {
                                hi
                            };
                            out.push(char::from_u32(cp).unwrap());
                        }
                        c => panic!("bad escape \\{}", c as char),
                    }
                }
            }
            b't' => {
                *i += 4;
                V::Bool(true)
            }
            b'f' => {
                *i += 5;
                V::Bool(false)
            }
            b'n' => {
                *i += 4;
                V::Null
            }
            _ => {
                let start = *i;
                while *i < b.len() && matches!(b[*i], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9') {
                    *i += 1;
                }
                V::Num(std::str::from_utf8(&b[start..*i]).unwrap().parse().unwrap())
            }
        }
    }
}

// ---------------------------------------------------------------------------
// the worked example

#[test]
fn hand_example_sizing_and_bits() {
    // WHY: section 3 by hand (formats/bloom.md, Worked example): n = 4,
    //      p = 0.1 gives m = ceil(19.17) = 20 bits and k = round(3.47) = 3;
    //      cat sets bits 11, 14, 17 and dog 13, 0, 3, so the bit array is
    //      09 68 02 (least significant bit first). bird probes 14, 7, 16:
    //      bit 7 is clear, so bird is (correctly) absent.
    // KIND: unit
    // CATCHES: s01, s02, s03
    // CHAPTER: ds.08 section 3, Worked example by hand
    let mut b = Bloom::with_rate(4, 0.1).unwrap();
    assert_eq!((b.m(), b.k()), (20, 3));
    assert_eq!(b.bit_positions(b"cat"), vec![11, 14, 17]);
    assert_eq!(b.bit_positions(b"dog"), vec![13, 0, 3]);
    assert_eq!(b.bit_positions(b"bird"), vec![14, 7, 16]);
    b.insert(b"cat");
    b.insert(b"dog");
    let want = unhex("544c424601000000140000000000000003000000000000000200000000000000096802");
    assert_eq!(b.to_bytes(), want);
    assert!(b.contains(b"cat") && b.contains(b"dog") && !b.contains(b"bird"));
    assert_eq!((b.n_inserted(), b.count_ones()), (2, 6));
}

#[test]
fn hash_functions_hand_values() {
    // WHY: the two hashes are fixed by the format, so every implementation
    //      sets the same bits: FNV-1a 64 of "" is the offset basis and of "a"
    //      is af63dc4c8601ec8c; the SplitMix64 finalizer maps 0 to 0 and 1 to
    //      5692161d100b05e5.
    // KIND: golden
    // CATCHES: s10
    // CHAPTER: ds.08 section 2.3
    assert_eq!(fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
    assert_eq!(fnv1a64(b"a"), 0xaf63_dc4c_8601_ec8c);
    assert_eq!(fnv1a64(b"cat"), 0xf5e3_0719_0ce4_a327);
    assert_eq!(mix64(0), 0);
    assert_eq!(mix64(1), 0x5692_161d_100b_05e5);
}

#[test]
fn sizing_formulas() {
    // WHY: m = ceil(-n ln p / (ln 2)^2) and k = max(1, round(m / n ln 2)).
    //      Truncating k (6.64 to 6, 1.80 to 1) costs false positives;
    //      flooring m gives a filter one bit short of the target rate.
    // KIND: unit
    // CATCHES: s01, s04
    // CHAPTER: ds.08 section 2.2
    for (n, p, m, k) in [(1000u64, 0.01, 9586u64, 7u32), (1, 0.5, 2, 1), (4, 0.1, 20, 3), (100_000, 0.001, 1_437_759, 10), (10, 0.3, 26, 2)] {
        assert_eq!(optimal_m(n, p), m, "m for n={n} p={p}");
        assert_eq!(optimal_k(m, n), k, "k for n={n} p={p}");
        let b = Bloom::with_rate(n, p).unwrap();
        assert_eq!((b.m(), b.k()), (m, k));
        assert_eq!(b.to_bytes().len(), HEADER_LEN + m.div_ceil(8) as usize);
    }
    assert_eq!(optimal_k(1, 1000), 1, "k is at least 1");
    let q = predicted_fp_rate(9586, 7, 1000);
    assert!((q - 0.01).abs() < 0.0005, "predicted rate {q} at the design point");
}

#[test]
fn bad_parameters_are_errors() {
    // WHY: n = 0, p outside (0, 1), and NaN have no filter; they are
    //      BadParams, never a panic, an empty filter, or an enormous one.
    // KIND: boundary
    // CATCHES: m04
    // CHAPTER: ds.08 section 4
    for (n, p) in [(0u64, 0.1), (10, 0.0), (10, 1.0), (10, -0.5), (10, 1.5), (10, f64::NAN)] {
        match Bloom::with_rate(n, p) {
            Err(BloomError::BadParams(_)) => {}
            other => panic!("with_rate({n}, {p}) = {other:?}, want BadParams"),
        }
    }
    assert!(matches!(Bloom::new(0, 3), Err(BloomError::BadParams(_))));
    assert!(matches!(Bloom::new(8, 0), Err(BloomError::BadParams(_))));
    assert!(matches!(Bloom::new((1 << 40) + 1, 1), Err(BloomError::BadParams(_))));
}

// ---------------------------------------------------------------------------
// properties

#[test]
fn no_false_negatives() {
    // WHY: the guarantee data.03 is built on: every inserted item is
    //      reported present, so a paragraph the screen calls new really is
    //      new. 5,000 random byte strings of length 0 to 40, through a
    //      to_bytes / from_bytes round trip and a union.
    // KIND: property
    // CATCHES: s06, s08
    // CHAPTER: ds.08 section 2.1
    let mut g = Pcg32::new(ol_seed(), 5);
    let mut a = Bloom::with_rate(5000, 0.02).unwrap();
    let mut b = Bloom::with_rate(5000, 0.02).unwrap();
    let mut items = Vec::new();
    for i in 0..5000 {
        let n = (g.next_u32() % 41) as usize;
        let it = g.bytes(n);
        if i % 2 == 0 { a.insert(&it) } else { b.insert(&it) }
        items.push(it);
    }
    for (i, it) in items.iter().enumerate() {
        assert!((if i % 2 == 0 { &a } else { &b }).contains(it), "false negative for item {i}");
    }
    let c = Bloom::from_bytes(&a.to_bytes()).unwrap();
    assert_eq!(c, a);
    let mut u = a.clone();
    u.union(&b).unwrap();
    for it in &items {
        assert!(u.contains(it));
    }
}

#[test]
fn fp_rate_within_three_sigma() {
    // WHY: a non-member is reported present with probability q = f^k, where
    //      f is the fraction of set bits (each of the k probes lands on a set
    //      bit). Over N = 100,000 fresh items the count of false positives is
    //      binomial(N, q): it must sit within 3 standard deviations of N q.
    //      A filter whose k probes collapse to one position has q near f,
    //      not f^k.
    // KIND: statistical
    // CATCHES: s07
    // CHAPTER: ds.08 section 2.4
    let mut g = Pcg32::new(ol_seed(), 5);
    let mut b = Bloom::with_rate(2000, 0.01).unwrap();
    for _ in 0..2000 {
        let it = g.bytes(16);
        b.insert(&it);
    }
    let f = b.count_ones() as f64 / b.m() as f64;
    let q = f.powi(b.k() as i32);
    let n = 100_000f64;
    let mut fp = 0u32;
    for _ in 0..100_000 {
        let it = g.bytes(17); // a different length: never an inserted item
        fp += b.contains(&it) as u32;
    }
    let (mean, sd) = (n * q, (n * q * (1.0 - q)).sqrt());
    assert!((fp as f64 - mean).abs() <= 3.0 * sd, "{fp} false positives, predicted {mean:.1} +- {sd:.1} (f = {f:.4}, k = {})", b.k());
    let rate = fp as f64 / n;
    assert!(rate < 0.02, "false-positive rate {rate} at a 0.01 design");
}

#[test]
fn union_is_the_filter_of_both_sets() {
    // WHY: union ORs the bit arrays, so it answers exactly like one filter
    //      that saw both sets (data.03 merges per-shard screens this way);
    //      the counts add. Filters of different m or k cannot be merged.
    // KIND: unit
    // CATCHES: s08, m02
    // CHAPTER: ds.08 section 4
    let mut a = Bloom::with_rate(100, 0.05).unwrap();
    let mut b = Bloom::with_rate(100, 0.05).unwrap();
    let mut both = Bloom::with_rate(100, 0.05).unwrap();
    for i in 0..60u32 {
        let it = format!("item-{i}");
        if i < 30 { a.insert(it.as_bytes()) } else { b.insert(it.as_bytes()) }
        both.insert(it.as_bytes());
    }
    a.union(&b).unwrap();
    assert_eq!(a.to_bytes(), both.to_bytes());
    assert_eq!(a.n_inserted(), 60);
    let other = Bloom::with_rate(1000, 0.05).unwrap();
    assert!(matches!(a.union(&other), Err(BloomError::Mismatch(_))));
    let same_m_other_k = Bloom::new(a.m(), a.k() + 1).unwrap();
    assert!(matches!(a.union(&same_m_other_k), Err(BloomError::Mismatch(_))));
    assert_eq!(a.n_inserted(), 60, "a failed union changes nothing");
}

// ---------------------------------------------------------------------------
// bytes

#[test]
fn golden_bytes_match_the_format() {
    // WHY: every case of the independent oracle (course/fixtures/parity/
    //      bloom.json): the filter built from the case's items serializes to
    //      the oracle's bytes, and answers its probes the same way. Equal
    //      bytes are what let a filter built in Rust be read in Python.
    // KIND: golden
    // CATCHES: s02, s03, s10
    // CHAPTER: ds.08 section 4
    let doc = j::parse(&std::fs::read_to_string(fixtures().join("parity/bloom.json")).unwrap());
    let cases = doc.get("cases").arr();
    assert!(cases.len() >= 10);
    for c in cases {
        let (inp, out) = (c.get("input"), c.get("output"));
        let mut b = Bloom::with_rate(inp.get("n").num() as u64, inp.get("p").num()).unwrap();
        for it in inp.get("insert").arr() {
            b.insert(&unhex(it.str()));
        }
        let name = c.get("name").str();
        assert_eq!((b.m(), b.k() as u64), (out.get("m").num() as u64, out.get("k").num() as u64), "{name}");
        assert_eq!(b.to_bytes(), unhex(out.get("bytes").str()), "{name}");
        for (p, want) in inp.get("probe").arr().iter().zip(out.get("contains").arr()) {
            assert_eq!(b.contains(&unhex(p.str())), want.bool(), "{name}: probe {}", p.str());
        }
    }
}

#[test]
fn from_bytes_rejects_malformed_input() {
    // WHY: a filter file is read across processes (data.03 writes one per
    //      run); a corrupt or foreign file must be refused with a Format
    //      error naming the problem, never read as a filter that answers
    //      wrongly.
    // KIND: boundary
    // CATCHES: s09, m03
    // CHAPTER: ds.08 section 5, Pitfalls
    let mut b = Bloom::with_rate(4, 0.1).unwrap();
    b.insert(b"cat");
    let good = b.to_bytes();
    assert_eq!(Bloom::from_bytes(&good).unwrap(), b);
    let bad = |f: &dyn Fn(&mut Vec<u8>)| {
        let mut x = good.clone();
        f(&mut x);
        Bloom::from_bytes(&x)
    };
    let cases: Vec<(&str, Result<Bloom, BloomError>)> = vec![
        ("magic", bad(&|x| x[0] = b'X')),
        ("version", bad(&|x| x[4] = 2)),
        ("m = 0", bad(&|x| x[8..16].copy_from_slice(&0u64.to_le_bytes()))),
        ("k = 0", bad(&|x| x[16..20].copy_from_slice(&0u32.to_le_bytes()))),
        ("reserved", bad(&|x| x[20] = 1)),
        ("short", bad(&|x| {
            x.pop();
        })),
        ("long", bad(&|x| x.push(0))),
        ("m disagrees with length", bad(&|x| x[8..16].copy_from_slice(&40u64.to_le_bytes()))),
        ("header only", Bloom::from_bytes(&good[..20])),
    ];
    for (what, r) in cases {
        assert!(matches!(r, Err(BloomError::Format(_))), "{what}: {r:?}");
    }
}
