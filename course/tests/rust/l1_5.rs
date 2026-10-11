//! L1.5 course tests, Rust half: tl-tok (byte-level BPE, pre-tokenizer,
//! loader, streaming decoder, byte tokenizer).
//!
//! Annotated exemplars (DESIGN 5.12). Oracles: 10,000 strings with the ids
//! Hugging Face `tokenizers` gives under GPT-2's and SmolLM2's tokenizer.json
//! (GPT-2's cross-checked by tiktoken), course/fixtures/L1.5/golden.jsonl,
//! from course/oracle/L1.5/golden.py. Hand-written expectations come from the
//! Python specification (L1.2). JSON is read with `mod j` below, never with
//! yours. The independent Python implementation also checks its ids against
//! the shared golden vectors in course/tests/L1.5/.

#![allow(dead_code)] // the fixture helpers are shared with ds_08.rs

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use tl_tok::bpe::{bytes_to_unicode, AddedToken};
use tl_tok::pretok::{pretokenize_gpt2, split_digits, PreTokenizer};
use tl_tok::{ByteBpe, ByteTokenizer, DecodeError, LoadError, SpecialSet, StreamDecoder, Tokenizer};

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

fn fixture(rel: &str) -> PathBuf {
    fixtures().join(rel)
}

fn gpt2() -> &'static ByteBpe {
    static T: OnceLock<ByteBpe> = OnceLock::new();
    T.get_or_init(|| ByteBpe::from_hf_json(&fixture("tok-gpt2/tokenizer.json")).expect("GPT-2 tokenizer.json loads"))
}

fn smollm2() -> &'static ByteBpe {
    static T: OnceLock<ByteBpe> = OnceLock::new();
    T.get_or_init(|| ByteBpe::from_hf_json(&fixture("tok-smollm2/tokenizer.json")).expect("SmolLM2 tokenizer.json loads"))
}

struct Golden {
    text: String,
    gpt2: Vec<u32>,
    smollm2: Vec<u32>,
}

fn golden() -> &'static [Golden] {
    static G: OnceLock<Vec<Golden>> = OnceLock::new();
    G.get_or_init(|| {
        let text = std::fs::read_to_string(fixture("L1.5/golden.jsonl")).unwrap();
        let ids = |v: &j::V| v.arr().iter().map(|x| x.num() as u32).collect::<Vec<u32>>();
        text.lines()
            .map(|line| {
                let v = j::parse(line);
                Golden { text: v.get("text").str().to_string(), gpt2: ids(v.get("gpt2")), smollm2: ids(v.get("smollm2")) }
            })
            .collect()
    })
}

/// The 256 byte tokens in the order L1.2's trainer gives them ids: by the
/// code point of their byte-map character. Byte b gets id `order[b]`.
fn byte_ids() -> [u32; 256] {
    let map = bytes_to_unicode();
    let mut bytes: Vec<u8> = (0..=255u8).collect();
    bytes.sort_by_key(|&b| map[b as usize]);
    let mut out = [0u32; 256];
    for (id, b) in bytes.into_iter().enumerate() {
        out[b as usize] = id as u32;
    }
    out
}

/// A tokenizer with the 256 byte tokens and the given merges (pairs of byte
/// strings), ids in L1.2's order; merge r gets id 256 + r.
fn tiny(merges: &[(&str, &str)]) -> ByteBpe {
    let ids = byte_ids();
    let mut vocab: Vec<(Vec<u8>, u32)> = (0..=255u8).map(|b| (vec![b], ids[b as usize])).collect();
    for (r, (l, rr)) in merges.iter().enumerate() {
        vocab.push(([l.as_bytes(), rr.as_bytes()].concat(), 256 + r as u32));
    }
    let ms = merges.iter().map(|(l, r)| (l.as_bytes().to_vec(), r.as_bytes().to_vec())).collect();
    ByteBpe::new(vocab, ms, vec![], PreTokenizer::gpt2()).unwrap()
}

/// O(n^2) BPE of one piece, written from the definition: repeatedly merge the
/// adjacent pair with the lowest rank, leftmost on ties. Uses only
/// `ByteBpe::merge` (the table lookup) and the byte ids.
fn naive_piece(t: &ByteBpe, piece: &[u8]) -> Vec<u32> {
    let mut ids: Vec<u32> = piece.iter().filter_map(|&b| t.token_to_id(&[b])).collect();
    loop {
        let mut best: Option<(u32, usize, u32)> = None;
        for i in 0..ids.len().saturating_sub(1) {
            if let Some(m) = t.merge(ids[i], ids[i + 1]) {
                if best.is_none_or(|(r, _, _)| m.rank < r) {
                    best = Some((m.rank, i, m.id));
                }
            }
        }
        match best {
            None => return ids,
            Some((_, i, id)) => {
                ids[i] = id;
                ids.remove(i + 1);
            }
        }
    }
}

fn json_str(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn tmpdir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ol-l1_5-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

// ---------------------------------------------------------------------------
// the worked example

#[test]
fn hand_example_merges_from_the_heap() {
    // WHY: section 3 by hand, the L1.2 example run through the heap. With
    //      merges (u g), (h ug), (p ug), "hugs pug" pre-tokenizes to "hugs"
    //      and " pug"; the heap pops (rank 0, pos 1) then (rank 1, pos 0) in
    //      "hugs" and (0, 2) then (2, 1) in " pug": ids [257, 82, 220, 258],
    //      what your Python gives. In "aaaa" with (a a), (aa aa), the merge
    //      at position 0 destroys the queued pair at position 1: it must be
    //      removed from the heap, or the next pop merges a pair that no
    //      longer exists.
    // KIND: unit
    // CATCHES: s01, m04
    // CHAPTER: L1.5 section 3, Worked example by hand
    let t = tiny(&[("u", "g"), ("h", "ug"), ("p", "ug")]);
    assert_eq!(t.encode("hugs pug"), vec![257, 82, 220, 258]);
    let a = tiny(&[("a", "a"), ("aa", "aa")]);
    assert_eq!(a.encode("aaaa"), vec![257]);
    assert_eq!(a.encode("aaaaa"), vec![257, a.token_to_id(b"a").unwrap()]);
    assert_eq!(a.encode("aaaaaa"), vec![257, 256], "leftmost pairs merge first");
}

#[test]
fn heap_bpe_equals_naive_bpe() {
    // WHY: the heap is only a faster way to find the pair the definition
    //      names. On 3,000 random pieces (GPT-2's merges, long runs of a few
    //      letters so that many merges compete) the heap's ids equal an
    //      O(n^2) rescan written from the definition.
    // KIND: differential
    // CATCHES: s01, s02
    // CHAPTER: L1.5 section 2.3
    let t = gpt2();
    let mut g = Pcg32::new(ol_seed(), 5);
    let alphabet: Vec<&str> = vec!["a", "b", "e", "r", "s", "t", "th", "in", " ", "é", "0"];
    for _ in 0..3000 {
        let n = (g.next_u32() % 40) as usize;
        let s: String = (0..n).map(|_| alphabet[(g.next_u32() % alphabet.len() as u32) as usize]).collect();
        let mut fast = Vec::new();
        t.encode_piece(s.as_bytes(), &mut fast);
        assert_eq!(fast, naive_piece(t, s.as_bytes()), "piece {s:?}");
    }
}

// ---------------------------------------------------------------------------
// bytes

#[test]
fn byte_tokenizer_is_the_identity() {
    // WHY: the tracer's tokenizer (D32): id = byte value, 256 ids, no
    //      specials. Every engine keeps serving it, so tl-tok has it too.
    // KIND: unit
    // CATCHES: s03
    // CHAPTER: L1.5 section 2.1
    let b = ByteTokenizer;
    assert_eq!(b.vocab_size(), 256);
    assert_eq!(b.encode("hé"), vec![0x68, 0xC3, 0xA9]);
    assert_eq!(b.encode_with_special("<|endoftext|>", &SpecialSet::all()).len(), 13);
    assert_eq!(b.token_bytes(255), Some(&[255u8][..]));
    assert_eq!(b.token_bytes(256), None);
    assert_eq!(b.decode(&[0x68, 0xC3, 0xA9]).unwrap(), "hé");
    assert_eq!(b.decode(&[0xFF]).unwrap(), "\u{FFFD}");
    assert_eq!(b.decode(&[256]), Err(DecodeError::UnknownId { id: 256, vocab_size: 256 }));
}

#[test]
fn byte_map_matches_gpt2() {
    // WHY: token strings in tokenizer.json are written in GPT-2's byte
    //      alphabet: the 188 printable bytes stand for themselves, the other
    //      68 map to U+0100 onward in byte order. Space is U+0120 (G with a
    //      dot), 0x00 is U+0100, 0xAD (the soft hyphen, the last non-printable)
    //      is U+0143.
    // KIND: unit
    // CATCHES: s04
    // CHAPTER: L1.5 section 2.1
    let m = bytes_to_unicode();
    assert_eq!((m[0x20], m[0x21], m[0x7E], m[0x00], m[0x7F], m[0xAD], m[0xAE], m[0xFF]), ('Ġ', '!', '~', 'Ā', 'ġ', 'Ń', '®', 'ÿ'));
    let mut seen = std::collections::HashSet::new();
    assert!(m.iter().all(|c| seen.insert(*c)), "the map is a bijection");
    assert_eq!(gpt2().token_to_id("Ġthe".as_bytes()), None, "vocab keys are bytes, not map characters");
    assert_eq!(gpt2().token_to_id(b" the"), Some(262));
}

// ---------------------------------------------------------------------------
// pre-tokenizer

#[test]
fn gpt2_split_hand_cases() {
    // WHY: the GPT-2 expression, one alternative at a time, against your
    //      Python pre-tokenizer: lowercase contractions only; one optional
    //      space joins the run after it; a white-space run gives back its
    //      last character when a non-space follows; marks (Mn, Mc) are not
    //      letters, so Devanagari vowel signs split.
    // KIND: unit
    // CATCHES: s05, s06, s08
    // CHAPTER: L1.5 section 2.2
    let cases: &[(&str, &[&str])] = &[
        ("don't STOP'S we'll", &["don", "'t", " STOP", "'", "S", " we", "'ll"]),
        ("a  b", &["a", " ", " b"]),
        ("a \tb", &["a", " ", "\t", "b"]),
        ("x\n\ny", &["x", "\n", "\n", "y"]),
        ("end  ", &["end", "  "]),
        ("  lead", &[" ", " lead"]),
        ("hi!!! ok?", &["hi", "!!!", " ok", "?"]),
        ("abc123 4½", &["abc", "123", " 4½"]),
        ("é and e\u{301}", &["é", " and", " e", "\u{301}"]),
        ("हिन्दी", &["ह", "ि", "न", "्", "द", "ी"]),
        ("", &[]),
    ];
    for (text, want) in cases {
        assert_eq!(pretokenize_gpt2(text), *want, "{text:?}");
    }
}

#[test]
fn digits_split_for_smollm2() {
    // WHY: SmolLM2 runs Digits(individual_digits = true) before ByteLevel:
    //      every \p{N} code point (½ and Ⅻ included) is its own piece, so
    //      "12345" is five tokens. GPT-2 keeps digit runs together.
    // KIND: unit
    // CATCHES: s07
    // CHAPTER: L1.5 section 2.2
    assert_eq!(split_digits("abc123 4½", true), vec!["abc", "1", "2", "3", " ", "4", "½"]);
    assert_eq!(split_digits("abc123 4½", false), vec!["abc", "123", " ", "4½"]);
    assert_eq!(smollm2().encode("12345"), vec![33, 34, 35, 36, 37]);
    assert_eq!(smollm2().pre_tokenizer().split("x12"), vec!["x", "1", "2"]);
}

// ---------------------------------------------------------------------------
// the oracles

#[test]
fn gpt2_golden_10k() {
    // WHY: GPT-2 ids of 10,000 strings (emoji sequences, CJK and other
    //      scripts, every kind of white space, contractions in every case,
    //      Unicode digits, controls, added-token texts, long words) equal
    //      Hugging Face's and tiktoken's exactly.
    // KIND: golden
    // CATCHES: s01, s04, s05, s06, s08, m01
    // CHAPTER: L1.5 section 4
    let t = gpt2();
    let g = golden();
    assert_eq!(g.len(), 10_000);
    for (i, c) in g.iter().enumerate() {
        assert_eq!(t.encode(&c.text), c.gpt2, "string {i}: {:?}", c.text);
    }
}

#[test]
fn smollm2_golden_10k() {
    // WHY: the same 10,000 strings under SmolLM2's tokenizer.json: its
    //      Digits step, its 17 added tokens, and its own merges.
    // KIND: golden
    // CATCHES: s08, m01, m04
    // CHAPTER: L1.5 section 4
    let t = smollm2();
    for (i, c) in golden().iter().enumerate() {
        assert_eq!(t.encode(&c.text), c.smollm2, "string {i}: {:?}", c.text);
    }
}

#[test]
fn decode_inverts_encode() {
    // WHY: byte-level BPE is lossless: decode_bytes(encode(x)) is x's UTF-8
    //      exactly, for every golden string and both tokenizers.
    // KIND: property
    // CATCHES: s04, m02
    // CHAPTER: L1.5 section 2.1
    for c in golden() {
        for t in [gpt2(), smollm2()] {
            let ids = t.encode(&c.text);
            assert_eq!(t.decode_bytes(&ids).unwrap(), c.text.as_bytes(), "{:?}", c.text);
            assert_eq!(t.decode(&ids).unwrap(), c.text);
        }
    }
    assert_eq!(gpt2().decode(&[50257]), Err(DecodeError::UnknownId { id: 50257, vocab_size: 50257 }));
}

// ---------------------------------------------------------------------------
// added tokens

#[test]
fn added_tokens_are_matched_first() {
    // WHY: added tokens that appear literally are split out before
    //      pre-tokenization and never merged, as Hugging Face does; with
    //      SpecialSet::none() the same text is ordinary text, which is how a
    //      server keeps user input from forging control tokens.
    // KIND: unit
    // CATCHES: s09
    // CHAPTER: L1.5 section 2.4
    let s = smollm2();
    assert_eq!(s.encode("<|endoftext|>"), vec![0]);
    assert_eq!(s.encode("x<|endoftext|>y"), vec![104, 0, 105]);
    assert_eq!(s.encode("<|im_start|>user\nhi<|im_end|>"), vec![1, 4093, 198, 6004, 2]);
    let plain = s.encode_with_special("<|endoftext|>", &SpecialSet::none());
    assert!(plain.len() > 1 && !plain.contains(&0), "{plain:?}");
    assert_eq!(s.decode(&plain).unwrap(), "<|endoftext|>");
    let only = s.encode_with_special("<|im_start|><|endoftext|>", &SpecialSet::only(&["<|endoftext|>"]));
    assert_eq!(only.last(), Some(&0));
    assert!(!only.contains(&1));
    assert_eq!(gpt2().encode("<|endoftext|>"), vec![50256]);
    assert_eq!(s.added_tokens().len(), 17);
    let a = &s.added_tokens()[1];
    assert_eq!(a, &AddedToken { content: "<|im_start|>".into(), id: 1, special: true });
}

// ---------------------------------------------------------------------------
// streaming

#[test]
fn stream_decoder_holds_back_partial_characters() {
    // WHY: GPT-2 splits the emoji U+1F600 (f0 9f 98 80) into two tokens,
    //      47249 (f0 9f 98) and 222 (80): the first push completes nothing
    //      and must emit nothing, the second emits the whole emoji. A byte
    //      that can never start a character (ff) becomes one U+FFFD at once;
    //      an incomplete tail at the end becomes one U+FFFD in finish.
    // KIND: boundary
    // CATCHES: s10, s11
    // CHAPTER: L1.5 section 2.5
    let t = gpt2();
    let mut d = StreamDecoder::new(t);
    assert_eq!(d.push(47249).unwrap(), None);
    assert_eq!(d.pending(), &[0xF0, 0x9F, 0x98]);
    assert_eq!(d.push(222).unwrap(), Some("\u{1F600}".to_string()));
    assert!(d.pending().is_empty());
    assert_eq!(d.finish(), "");
    let b = ByteTokenizer;
    let mut d = StreamDecoder::new(&b);
    let out: Vec<Option<String>> = [0xC3, 0xA9, 0xFF, 0xE2, 0x82].iter().map(|&i| d.push(i).unwrap()).collect();
    assert_eq!(out, vec![None, Some("é".into()), Some("\u{FFFD}".into()), None, None]);
    assert_eq!(d.finish(), "\u{FFFD}");
    assert!(d.pending().is_empty());
    assert!(d.push(256).is_err());
}

#[test]
fn stream_equals_decode_on_golden() {
    // WHY: concatenating every chunk and finish gives exactly decode(ids):
    //      the rule the engine's SSE stream relies on (the conformance case
    //      chat.stream.equals_nonstream), on 2,000 golden strings.
    // KIND: property
    // CATCHES: s10, s11
    // CHAPTER: L1.5 section 2.5
    let t = gpt2();
    for c in &golden()[..2000] {
        let mut d = StreamDecoder::new(t);
        let mut s = String::new();
        for &id in &c.gpt2 {
            if let Some(x) = d.push(id).unwrap() {
                s.push_str(&x);
            }
            assert!(d.pending().len() <= 3, "at most 3 bytes wait for a character");
        }
        s.push_str(&d.finish());
        assert_eq!(s, c.text);
    }
}

// ---------------------------------------------------------------------------
// batch

#[test]
fn encode_batch_equals_serial_encode() {
    // WHY: encode_batch splits the texts over threads and must give back
    //      exactly [encode(t) for t in texts], in order, for any thread
    //      count (0 means all cores). data.07 tokenizes shards this way.
    // KIND: property
    // CATCHES: s12
    // CHAPTER: L1.5 section 2.6
    let t = gpt2();
    let texts: Vec<&str> = golden()[..997].iter().map(|c| c.text.as_str()).collect();
    let serial: Vec<Vec<u32>> = texts.iter().map(|x| t.encode(x)).collect();
    for threads in [0, 1, 2, 3, 4, 8] {
        assert_eq!(t.encode_batch(&texts, threads), serial, "threads = {threads}");
    }
    assert!(t.encode_batch(&[], 4).is_empty());
}

// ---------------------------------------------------------------------------
// loading

#[test]
fn loader_rejects_files_outside_the_subset() {
    // WHY: a tokenizer.json that loads is one this tokenizer encodes
    //      exactly; anything outside the formats/tokenizer.md subset (a
    //      normalizer, another model type, byte fallback, add_prefix_space)
    //      is refused with an error naming the field, not silently ignored.
    // KIND: boundary
    // CATCHES: s13
    // CHAPTER: L1.5 section 5, Pitfalls
    let smol = std::fs::read_to_string(fixture("tok-smollm2/tokenizer.json")).unwrap();
    assert!(ByteBpe::from_hf_json_str(&smol).is_ok());
    let edits: &[(&str, &str, &str)] = &[
        ("\"normalizer\": null", "\"normalizer\": {\"type\": \"NFC\"}", "normalizer"),
        ("\"type\": \"BPE\"", "\"type\": \"WordPiece\"", "model.type"),
        ("\"add_prefix_space\": false", "\"add_prefix_space\": true", "add_prefix_space"),
        ("\"byte_fallback\": false", "\"byte_fallback\": true", "byte_fallback"),
    ];
    for (from, to, field) in edits {
        assert!(smol.contains(from), "SmolLM2's tokenizer.json spells {from}");
        match ByteBpe::from_hf_json_str(&smol.replacen(from, to, 1)) {
            Err(LoadError::Unsupported { field: f, .. }) => assert!(f.contains(field), "{f} should name {field}"),
            other => panic!("{field}: want Unsupported, got {:?}", other.map(|_| ())),
        }
    }
    let gpt = std::fs::read_to_string(fixture("tok-gpt2/tokenizer.json")).unwrap();
    let bad = gpt.replacen("\"normalizer\":null", "\"normalizer\":{\"type\":\"NFC\"}", 1);
    assert_ne!(bad, gpt, "GPT-2's tokenizer.json has a null normalizer");
    assert!(matches!(ByteBpe::from_hf_json_str(&bad), Err(LoadError::Unsupported { .. })));
    assert!(matches!(ByteBpe::from_hf_json_str("{"), Err(LoadError::Json(_))));
    assert!(matches!(ByteBpe::from_hf_json(Path::new("/nonexistent/tokenizer.json")), Err(LoadError::Io { .. })));
}

#[test]
fn gpt2_classic_files_load_identically() {
    // WHY: GPT-2 also ships as vocab.json + merges.txt (with a "#version"
    //      first line); both forms describe one tokenizer and encode
    //      identically. The test writes the two files from tokenizer.json.
    // KIND: differential
    // CATCHES: m03
    // CHAPTER: L1.5 section 4
    let doc = j::parse(&std::fs::read_to_string(fixture("tok-gpt2/tokenizer.json")).unwrap());
    let model = doc.get("model");
    let d = tmpdir("classic");
    let vocab = match model.get("vocab") {
        j::V::Obj(o) => o.iter().map(|(k, v)| format!("{}: {}", json_str(k), v.num() as u64)).collect::<Vec<_>>().join(", "),
        _ => panic!("model.vocab is an object"),
    };
    std::fs::write(d.join("vocab.json"), format!("{{{vocab}}}")).unwrap();
    let mut merges = String::from("#version: 0.2\n");
    for m in model.get("merges").arr() {
        match m {
            j::V::Str(s) => merges.push_str(s),
            j::V::Arr(a) => merges.push_str(&format!("{} {}", a[0].str(), a[1].str())),
            _ => panic!("merge entry"),
        }
        merges.push('\n');
    }
    std::fs::write(d.join("merges.txt"), merges).unwrap();
    let classic = ByteBpe::from_gpt2_files(&d.join("vocab.json"), &d.join("merges.txt")).unwrap();
    assert_eq!(classic.vocab_size(), 50257);
    assert_eq!(classic.merge_count(), gpt2().merge_count());
    for c in &golden()[..1000] {
        assert_eq!(classic.encode(&c.text), c.gpt2, "{:?}", c.text);
    }
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn vocab_and_merge_tables() {
    // WHY: the tables live in RobinHoodMap (ds.05): bytes to id, and
    //      (left id, right id) to (rank, merged id). GPT-2's first merge is
    //      (Ġ, t) with rank 0 and id 256.
    // KIND: unit
    // CATCHES: m04
    // CHAPTER: L1.5 section 2.3
    let t = gpt2();
    let (sp, tt) = (t.token_to_id(b" ").unwrap(), t.token_to_id(b"t").unwrap());
    let m = t.merge(sp, tt).unwrap();
    assert_eq!((m.rank, m.id), (0, 256));
    assert_eq!(t.token_bytes(256), Some(&b" t"[..]));
    assert_eq!(t.merge(tt, sp), None);
    assert_eq!((t.vocab_size(), t.merge_count()), (50257, 50000));
    assert_eq!(smollm2().vocab_size(), 49152);
}
