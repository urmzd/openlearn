// Parity driver for `tokenizer.bpe`, implementation `rust` (L1.5): reads one
// JSON input per stdin line (`{"text": ...}`), loads GPT-2 tokenizer.json from
// TINYLLM_FIXTURES, and prints the token ids as one JSON array per line.
include!("../parity/ol_parity.rs");

use std::path::PathBuf;
use tl_tok::{ByteBpe, Tokenizer};

fn hex4(bytes: &[u8], i: &mut usize) -> u32 {
    let end = *i + 4;
    let value = u32::from_str_radix(std::str::from_utf8(&bytes[*i..end]).expect("unicode escape"), 16)
        .expect("unicode escape");
    *i = end;
    value
}

/// Decode one JSON string field, including escaped controls and surrogate pairs.
fn json_string(line: &str, key: &str) -> String {
    let raw = ssp_find(line, key).expect("text field").as_bytes();
    assert_eq!(raw.first(), Some(&b'"'));
    let mut i = 1;
    let mut out = String::new();
    while i < raw.len() {
        match raw[i] {
            b'"' => return out,
            b'\\' => {
                i += 1;
                let escaped = raw[i];
                i += 1;
                match escaped {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'b' => out.push('\u{8}'),
                    b'f' => out.push('\u{c}'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    b'u' => {
                        let hi = hex4(raw, &mut i);
                        let codepoint = if (0xD800..0xDC00).contains(&hi) {
                            assert_eq!(&raw[i..i + 2], b"\\u");
                            i += 2;
                            0x10000 + ((hi - 0xD800) << 10) + (hex4(raw, &mut i) - 0xDC00)
                        } else {
                            hi
                        };
                        out.push(char::from_u32(codepoint).expect("valid JSON code point"));
                    }
                    _ => panic!("invalid JSON escape"),
                }
            }
            _ => {
                let tail = std::str::from_utf8(&raw[i..]).expect("UTF-8 JSON input");
                let ch = tail.chars().next().expect("string character");
                out.push(ch);
                i += ch.len_utf8();
            }
        }
    }
    panic!("unterminated JSON string")
}

fn main() {
    let mut path = PathBuf::from(std::env::var("TINYLLM_FIXTURES").expect("TINYLLM_FIXTURES"));
    path.push("tok-gpt2/tokenizer.json");
    let tok = ByteBpe::from_hf_json(&path).expect("GPT-2 tokenizer.json loads");
    for line in ssp_lines() {
        if line.trim().is_empty() {
            continue;
        }
        let ids = tok.encode(&json_string(&line, "text"));
        println!("[{}]", ids.iter().map(u32::to_string).collect::<Vec<_>>().join(","));
    }
}
