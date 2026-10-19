//! Loading byte-level BPE files (L1.5): Hugging Face `tokenizer.json` in the
//! formats/tokenizer.md BPE subset, and GPT-2's `vocab.json` + `merges.txt`.
//!
//! The loader accepts exactly what GPT-2's and SmolLM2's files use and
//! rejects everything else with an error naming the field, so a file that
//! loads is a file this tokenizer encodes exactly:
//!
//! | Field | Accepted |
//! |---|---|
//! | `model.type` | `"BPE"`, or absent when `model.merges` is present (GPT-2's file) |
//! | `model.dropout`, `model.unk_token` | `null` |
//! | `model.continuing_subword_prefix`, `model.end_of_word_suffix` | `null` or `""` |
//! | `model.fuse_unk`, `model.byte_fallback`, `model.ignore_merges` | `false` or absent |
//! | `model.merges` | `"left right"` strings or `["left", "right"]` pairs, by rank |
//! | `normalizer` | `null` |
//! | `pre_tokenizer` | `ByteLevel` (`use_regex` true or absent, `add_prefix_space` false), or a `Sequence` of `Digits` and that `ByteLevel`, `ByteLevel` last |
//! | `post_processor` | `null` or `ByteLevel` |
//! | `decoder` | `null` or `ByteLevel` |
//! | `added_tokens[*]` | `single_word`, `lstrip`, `rstrip` false |

use std::path::Path;

use serde_json::Value;

use crate::bpe::{token_str_to_bytes, unicode_to_bytes, AddedToken, ByteBpe};
use crate::pretok::{PreTokenizer, Step};
use crate::LoadError;

fn unsupported(field: &str, v: &Value) -> LoadError {
    LoadError::Unsupported { field: field.to_string(), value: v.to_string() }
}

fn read(path: &Path) -> Result<String, LoadError> {
    std::fs::read_to_string(path).map_err(|e| LoadError::Io { path: path.to_path_buf(), source: e })
}

/// `field` must be absent, `null`, or one of `ok`.
fn expect_in(obj: &Value, key: &str, path: &str, ok: &[Value]) -> Result<(), LoadError> {
    match obj.get(key) {
        None | Some(Value::Null) => Ok(()),
        Some(v) if ok.contains(v) => Ok(()),
        Some(v) => Err(unsupported(&format!("{path}.{key}"), v)),
    }
}

/// One `ByteLevel` pre-tokenizer object: the GPT-2 split, nothing else.
fn byte_level(v: &Value, path: &str) -> Result<Step, LoadError> {
    expect_in(v, "use_regex", path, &[Value::Bool(true)])?;
    expect_in(v, "add_prefix_space", path, &[Value::Bool(false)])?;
    Ok(Step::ByteLevel)
}

/// The `pre_tokenizer` object as a [`PreTokenizer`].
pub fn parse_pre_tokenizer(v: &Value) -> Result<PreTokenizer, LoadError> {
    let ty = v.get("type").and_then(Value::as_str).unwrap_or("");
    let steps = match ty {
        "ByteLevel" => vec![byte_level(v, "pre_tokenizer")?],
        "Sequence" => {
            let list = v
                .get("pretokenizers")
                .and_then(Value::as_array)
                .ok_or_else(|| unsupported("pre_tokenizer.pretokenizers", v))?;
            let mut steps = Vec::new();
            for (i, p) in list.iter().enumerate() {
                let path = format!("pre_tokenizer.pretokenizers[{i}]");
                match p.get("type").and_then(Value::as_str) {
                    Some("Digits") => {
                        let individual = p.get("individual_digits").and_then(Value::as_bool).unwrap_or(false);
                        steps.push(Step::Digits { individual });
                    }
                    Some("ByteLevel") => steps.push(byte_level(p, &path)?),
                    _ => return Err(unsupported(&format!("{path}.type"), p.get("type").unwrap_or(&Value::Null))),
                }
            }
            if steps.last() != Some(&Step::ByteLevel) || steps.iter().filter(|s| **s == Step::ByteLevel).count() != 1 {
                return Err(unsupported("pre_tokenizer.pretokenizers", v));
            }
            steps
        }
        _ => return Err(unsupported("pre_tokenizer.type", v.get("type").unwrap_or(v))),
    };
    Ok(PreTokenizer { steps })
}

/// One merge entry: `"left right"` or `["left", "right"]`.
fn merge_pair(v: &Value, rank: usize) -> Result<(String, String), LoadError> {
    let bad = || LoadError::Format(format!("model.merges[{rank}] = {v}: want \"left right\" or [\"left\", \"right\"]"));
    match v {
        Value::String(s) => {
            let (l, r) = s.split_once(' ').ok_or_else(bad)?;
            if r.contains(' ') {
                return Err(bad());
            }
            Ok((l.to_string(), r.to_string()))
        }
        Value::Array(a) if a.len() == 2 => match (&a[0], &a[1]) {
            (Value::String(l), Value::String(r)) => Ok((l.clone(), r.clone())),
            _ => Err(bad()),
        },
        _ => Err(bad()),
    }
}

/// Token strings to bytes through the GPT-2 byte map.
fn to_bytes(s: &str, inv: &[Option<u8>], what: &str) -> Result<Vec<u8>, LoadError> {
    token_str_to_bytes(s, inv)
        .ok_or_else(|| LoadError::Format(format!("{what} {s:?} has a character outside the byte-level alphabet")))
}

/// The vocabulary and merges of a BPE model object (`model` in
/// tokenizer.json, or vocab.json plus merges.txt), as token bytes. A token
/// in the vocabulary that is also an added token keeps its literal content.
fn build(
    vocab: &serde_json::Map<String, Value>,
    merges: Vec<(String, String)>,
    added: Vec<AddedToken>,
    pretok: PreTokenizer,
) -> Result<ByteBpe, LoadError> {
    let inv = unicode_to_bytes();
    let mut pairs = Vec::with_capacity(vocab.len());
    for (s, id) in vocab {
        let id = id
            .as_u64()
            .filter(|&x| x <= u32::MAX as u64)
            .ok_or_else(|| LoadError::Format(format!("model.vocab[{s:?}] = {id} is not an id")))? as u32;
        let bytes = match added.iter().find(|a| a.id == id && a.content == *s) {
            Some(a) => a.content.as_bytes().to_vec(),
            None => to_bytes(s, &inv, "vocab token")?,
        };
        pairs.push((bytes, id));
    }
    let mut ms = Vec::with_capacity(merges.len());
    for (l, r) in merges {
        ms.push((to_bytes(&l, &inv, "merge part")?, to_bytes(&r, &inv, "merge part")?));
    }
    ByteBpe::new(pairs, ms, added, pretok)
}

impl ByteBpe {
    /// Loads a Hugging Face `tokenizer.json` (SmolLM2's, GPT-2's).
    pub fn from_hf_json(path: &Path) -> Result<ByteBpe, LoadError> {
        ByteBpe::from_hf_json_str(&read(path)?)
    }

    /// The same, from the file's text.
    pub fn from_hf_json_str(text: &str) -> Result<ByteBpe, LoadError> {
        let root: Value = serde_json::from_str(text).map_err(|e| LoadError::Json(e.to_string()))?;
        if !root.is_object() {
            return Err(LoadError::Json("the top level is not an object".into()));
        }
        let model = root.get("model").filter(|m| m.is_object()).ok_or_else(|| LoadError::Json("no model object".into()))?;
        match model.get("type") {
            None if model.get("merges").is_some() => {}
            Some(Value::String(t)) if t == "BPE" => {}
            Some(v) => return Err(unsupported("model.type", v)),
            None => return Err(unsupported("model.type", &Value::Null)),
        }
        expect_in(model, "dropout", "model", &[])?;
        expect_in(model, "unk_token", "model", &[])?;
        expect_in(model, "continuing_subword_prefix", "model", &[Value::String(String::new())])?;
        expect_in(model, "end_of_word_suffix", "model", &[Value::String(String::new())])?;
        for key in ["fuse_unk", "byte_fallback", "ignore_merges"] {
            expect_in(model, key, "model", &[Value::Bool(false)])?;
        }
        expect_in(&root, "normalizer", "tokenizer", &[])?;
        for key in ["post_processor", "decoder"] {
            match root.get(key) {
                None | Some(Value::Null) => {}
                Some(v) if v.get("type").and_then(Value::as_str) == Some("ByteLevel") => {}
                Some(v) => return Err(unsupported(&format!("{key}.type"), v.get("type").unwrap_or(v))),
            }
        }
        let pretok = match root.get("pre_tokenizer") {
            None | Some(Value::Null) => return Err(unsupported("pre_tokenizer", &Value::Null)),
            Some(v) => parse_pre_tokenizer(v)?,
        };
        let mut added = Vec::new();
        if let Some(list) = root.get("added_tokens") {
            let list = list.as_array().ok_or_else(|| unsupported("added_tokens", list))?;
            for (i, a) in list.iter().enumerate() {
                for key in ["single_word", "lstrip", "rstrip"] {
                    expect_in(a, key, &format!("added_tokens[{i}]"), &[Value::Bool(false)])?;
                }
                let content = a.get("content").and_then(Value::as_str);
                let id = a.get("id").and_then(Value::as_u64).filter(|&x| x <= u32::MAX as u64);
                match (content, id) {
                    (Some(c), Some(id)) if !c.is_empty() => added.push(AddedToken {
                        content: c.to_string(),
                        id: id as u32,
                        special: a.get("special").and_then(Value::as_bool).unwrap_or(false),
                    }),
                    _ => return Err(LoadError::Format(format!("added_tokens[{i}] needs a non-empty content and an id"))),
                }
            }
        }
        let vocab = model
            .get("vocab")
            .and_then(Value::as_object)
            .ok_or_else(|| LoadError::Json("model.vocab is not an object".into()))?;
        let merges = model
            .get("merges")
            .and_then(Value::as_array)
            .ok_or_else(|| LoadError::Json("model.merges is not a list".into()))?
            .iter()
            .enumerate()
            .map(|(r, v)| merge_pair(v, r))
            .collect::<Result<Vec<_>, _>>()?;
        build(vocab, merges, added, pretok)
    }

    /// Loads GPT-2's `vocab.json` and `merges.txt` (lines "left right";
    /// `#version` and empty lines skipped). `<|endoftext|>`, when present in
    /// the vocabulary, is an added special token.
    pub fn from_gpt2_files(vocab_json: &Path, merges_txt: &Path) -> Result<ByteBpe, LoadError> {
        let vocab: Value = serde_json::from_str(&read(vocab_json)?).map_err(|e| LoadError::Json(e.to_string()))?;
        let vocab = vocab.as_object().ok_or_else(|| LoadError::Json("vocab.json is not an object".into()))?;
        let mut merges = Vec::new();
        for (i, line) in read(merges_txt)?.lines().enumerate() {
            if line.is_empty() || line.starts_with("#version") {
                continue;
            }
            merges.push(merge_pair(&Value::String(line.to_string()), i)?);
        }
        let mut added = Vec::new();
        if let Some(id) = vocab.get("<|endoftext|>").and_then(Value::as_u64) {
            added.push(AddedToken { content: "<|endoftext|>".into(), id: id as u32, special: true });
        }
        build(vocab, merges, added, PreTokenizer::gpt2())
    }
}
