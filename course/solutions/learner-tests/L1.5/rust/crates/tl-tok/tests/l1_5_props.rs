//! Reference learner tests for L1.5 (rung R4: properties with proptest,
//! craft.04). Each property is stated in the chapter (section 4); they run
//! against GPT-2's and SmolLM2's tokenizer.json from $TINYLLM_FIXTURES, which
//! `ol check` sets. Deterministic: a fixed proptest seed, no failure files.

use std::path::PathBuf;
use std::sync::OnceLock;

use proptest::prelude::*;
use proptest::test_runner::{Config, RngSeed};

use tl_tok::pretok::{pretokenize_gpt2, split_digits};
use tl_tok::{ByteBpe, ByteTokenizer, SpecialSet, StreamDecoder, Tokenizer};

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(std::env::var("TINYLLM_FIXTURES").expect("TINYLLM_FIXTURES")).join(rel)
}

fn gpt2() -> &'static ByteBpe {
    static T: OnceLock<ByteBpe> = OnceLock::new();
    T.get_or_init(|| ByteBpe::from_hf_json(&fixture("tok-gpt2/tokenizer.json")).unwrap())
}

fn smollm2() -> &'static ByteBpe {
    static T: OnceLock<ByteBpe> = OnceLock::new();
    T.get_or_init(|| ByteBpe::from_hf_json(&fixture("tok-smollm2/tokenizer.json")).unwrap())
}

fn cfg() -> Config {
    Config { cases: 128, rng_seed: RngSeed::Fixed(5), failure_persistence: None, ..Config::default() }
}

/// Text that stresses the pre-tokenizer: contractions, white-space runs,
/// digits, marks, CJK, emoji.
fn text() -> impl Strategy<Value = String> {
    let atoms = prop::sample::select(vec![
        "a", "the", " ", "  ", "\t", "\n", "\n\n", "'s", "'S", "'ll", "12", "7", "½", "é", "e\u{301}", "ि", "日本",
        "\u{1F600}", "\u{1F468}\u{200D}\u{1F469}", "!", "?!", "\u{a0}", "\u{3000}", "x", "ab",
    ]);
    prop::collection::vec(atoms, 0..16).prop_map(|v| v.concat())
}

/// O(n^2) BPE of one piece straight from the definition.
fn naive(t: &ByteBpe, piece: &[u8]) -> Vec<u32> {
    let mut ids: Vec<u32> = piece.iter().filter_map(|&b| t.token_to_id(&[b])).collect();
    loop {
        let mut best: Option<(u32, usize, u32)> = None;
        for i in 0..ids.len().saturating_sub(1) {
            if let Some(m) = t.merge(ids[i], ids[i + 1]) {
                if best.map_or(true, |(r, _, _)| m.rank < r) {
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

proptest! {
    #![proptest_config(cfg())]

    #[test]
    fn decode_inverts_encode(s in any::<String>()) {
        for t in [gpt2(), smollm2()] {
            prop_assert_eq!(t.decode(&t.encode(&s)).unwrap(), s.clone());
        }
    }

    #[test]
    fn heap_equals_naive(s in text()) {
        let t = gpt2();
        for piece in pretokenize_gpt2(&s) {
            let mut fast = Vec::new();
            t.encode_piece(piece.as_bytes(), &mut fast);
            prop_assert_eq!(fast, naive(t, piece.as_bytes()));
        }
    }

    #[test]
    fn pieces_concatenate_to_the_input(s in any::<String>()) {
        prop_assert_eq!(pretokenize_gpt2(&s).concat(), s.clone());
        prop_assert_eq!(split_digits(&s, true).concat(), s.clone());
    }

    #[test]
    fn stream_equals_decode(s in text()) {
        let t = gpt2();
        let ids = t.encode(&s);
        let mut d = StreamDecoder::new(t);
        let mut out = String::new();
        for &id in &ids {
            if let Some(x) = d.push(id).unwrap() {
                out.push_str(&x);
            }
        }
        out.push_str(&d.finish());
        prop_assert_eq!(out, s);
    }

    #[test]
    fn batch_equals_serial(v in prop::collection::vec(text(), 0..12), threads in 0usize..5) {
        let t = gpt2();
        let refs: Vec<&str> = v.iter().map(String::as_str).collect();
        let serial: Vec<Vec<u32>> = refs.iter().map(|x| t.encode(x)).collect();
        prop_assert_eq!(t.encode_batch(&refs, threads), serial);
    }

    #[test]
    fn byte_tokenizer_roundtrips(bytes in prop::collection::vec(any::<u8>(), 0..64)) {
        let b = ByteTokenizer;
        let ids: Vec<u32> = bytes.iter().map(|&x| x as u32).collect();
        prop_assert_eq!(b.decode_bytes(&ids).unwrap(), bytes);
    }
}

#[test]
fn known_ids() {
    assert_eq!(gpt2().encode("Hello world"), vec![15496, 995]);
    assert_eq!(gpt2().encode("don't"), vec![9099, 470]);
    assert_eq!(gpt2().encode("a  b"), vec![64, 220, 275]);
    assert_eq!(smollm2().encode("12345"), vec![33, 34, 35, 36, 37]);
    assert_eq!(smollm2().encode("x<|endoftext|>y"), vec![104, 0, 105]);
}

#[test]
fn split_examples() {
    assert_eq!(pretokenize_gpt2("x\n\ny"), vec!["x", "\n", "\n", "y"]);
    assert_eq!(pretokenize_gpt2("IT'S"), vec!["IT", "'", "S"]);
    assert_eq!(pretokenize_gpt2("हिन्दी").len(), 6);
    assert_eq!(split_digits("a12", true), vec!["a", "1", "2"]);
}

#[test]
fn special_set_none_is_plain_text() {
    let ids = smollm2().encode_with_special("<|endoftext|>", &SpecialSet::none());
    assert!(!ids.contains(&0));
}

#[test]
fn stream_holds_back_partial_characters() {
    let b = ByteTokenizer;
    let mut d = StreamDecoder::new(&b);
    assert_eq!(d.push(0xC3).unwrap(), None);
    assert_eq!(d.push(0xA9).unwrap().as_deref(), Some("é"));
    assert_eq!(d.push(0xE2).unwrap(), None);
    assert_eq!(d.finish(), "\u{FFFD}");
    assert!(b.token_bytes(256).is_none());
    assert_eq!(b.decode(&[0xFF]).unwrap(), "\u{FFFD}");
}

#[test]
fn loader_refuses_a_normalizer() {
    let src = std::fs::read_to_string(fixture("tok-smollm2/tokenizer.json")).unwrap();
    let bad = src.replacen("\"normalizer\": null", "\"normalizer\": {\"type\": \"NFC\"}", 1);
    assert!(ByteBpe::from_hf_json_str(&bad).is_err());
}
