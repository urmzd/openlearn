//! tl-tok: the Rust tokenizer of the system (L1.5).
//!
//! * [`Tokenizer`]: the trait the engine (L10.1, L10.5) and the binding
//!   program against.
//! * [`ByteTokenizer`]: the tracer's byte tokenizer (formats/tokenizer.md):
//!   id = byte value, 256 ids, no specials.
//! * [`bpe::ByteBpe`]: byte-level BPE, exact against GPT-2 and SmolLM2, with
//!   merges run from a lazy heap (ds.06) over Robin Hood maps (ds.05).
//! * [`pretok`]: the GPT-2 pre-tokenizer and the `Digits` splitter, written
//!   by hand over general-category tables.
//! * [`stream::StreamDecoder`]: ids in, complete UTF-8 text out, one token at
//!   a time (the engine's SSE path).
//! * [`hfjson`]: `tokenizer.json` (the formats/tokenizer.md BPE subset) and
//!   GPT-2's `vocab.json` + `merges.txt`.
//!
//! The Python BPE (L1.2) is the specification: every id here equals the id
//! the Python computes. Chapter: content/courses/build-an-llm/02-tokenizers/05-rust-fast-bpe.md

use std::fmt;
use std::path::PathBuf;

pub mod bpe;
pub mod hfjson;
pub mod pretok;
pub mod stream;

pub use bpe::ByteBpe;
pub use stream::StreamDecoder;

/// Which added tokens `encode_with_special` matches in the text. Added tokens
/// that are not allowed are encoded as ordinary text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpecialSet {
    only: Option<Vec<String>>, // None: every added token
}

impl SpecialSet {
    /// Every added token is matched (what `encode` does, as Hugging Face does).
    pub fn all() -> Self {
        SpecialSet { only: None }
    }

    /// No added token is matched: `<|endoftext|>` in user text is 7 ordinary pieces.
    pub fn none() -> Self {
        SpecialSet { only: Some(Vec::new()) }
    }

    /// Only these added tokens (by their text) are matched.
    pub fn only(tokens: &[&str]) -> Self {
        SpecialSet { only: Some(tokens.iter().map(|s| s.to_string()).collect()) }
    }

    pub fn allows(&self, token: &str) -> bool {
        match &self.only {
            None => true,
            Some(v) => v.iter().any(|s| s == token),
        }
    }
}

/// An id outside the vocabulary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DecodeError {
    UnknownId { id: u32, vocab_size: u32 },
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecodeError::UnknownId { id, vocab_size } => {
                write!(f, "id {id} is outside the vocabulary (size {vocab_size})")
            }
        }
    }
}

impl std::error::Error for DecodeError {}

/// Why a tokenizer file did not load. `Io` maps to Python `OSError`, the
/// rest to `ValueError`; every message names the file or the field.
#[derive(Debug)]
pub enum LoadError {
    Io { path: PathBuf, source: std::io::Error },
    /// Not valid JSON, or not the expected JSON shape.
    Json(String),
    /// A field outside the formats/tokenizer.md subset: its JSON path and value.
    Unsupported { field: String, value: String },
    /// Inconsistent content: a merge of unknown tokens, a gap in the ids.
    Format(String),
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadError::Io { path, source } => write!(f, "{}: {source}", path.display()),
            LoadError::Json(s) => write!(f, "tokenizer file: {s}"),
            LoadError::Unsupported { field, value } => {
                write!(f, "unsupported tokenizer field {field} = {value}")
            }
            LoadError::Format(s) => write!(f, "tokenizer file: {s}"),
        }
    }
}

impl std::error::Error for LoadError {}

/// A tokenizer: text to ids and back. `Send + Sync` so one instance serves
/// every engine thread and `encode_batch` workers.
pub trait Tokenizer: Send + Sync {
    /// Ids of `text`; added tokens that appear literally are matched first.
    fn encode(&self, text: &str) -> Vec<u32>;

    /// Ids of `text`, matching only the added tokens `allowed` names.
    fn encode_with_special(&self, text: &str, allowed: &SpecialSet) -> Vec<u32>;

    /// The bytes of one id, or `None` outside the vocabulary.
    fn token_bytes(&self, id: u32) -> Option<&[u8]>;

    /// Number of ids: every id in `0..vocab_size()` decodes.
    fn vocab_size(&self) -> u32;

    /// The bytes of `ids`, concatenated.
    fn decode_bytes(&self, ids: &[u32]) -> Result<Vec<u8>, DecodeError> {
        let mut out = Vec::new();
        for &id in ids {
            match self.token_bytes(id) {
                Some(b) => out.extend_from_slice(b),
                None => return Err(DecodeError::UnknownId { id, vocab_size: self.vocab_size() }),
            }
        }
        Ok(out)
    }

    /// The text of `ids`: their bytes decoded as UTF-8 with replacement (one
    /// U+FFFD per maximal ill-formed subsequence, formats/tokenizer.md).
    fn decode(&self, ids: &[u32]) -> Result<String, DecodeError> {
        Ok(String::from_utf8_lossy(&self.decode_bytes(ids)?).into_owned())
    }
}

/// Every byte value, so `token_bytes` of the byte tokenizer can lend a slice.
static ALL_BYTES: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        t[i] = i as u8;
        i += 1;
    }
    t
};

/// The tracer's tokenizer: the id of a byte is its value.
#[derive(Clone, Copy, Debug, Default)]
pub struct ByteTokenizer;

impl Tokenizer for ByteTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        text.bytes().map(u32::from).collect()
    }

    fn encode_with_special(&self, text: &str, _allowed: &SpecialSet) -> Vec<u32> {
        self.encode(text)
    }

    fn token_bytes(&self, id: u32) -> Option<&[u8]> {
        let i = id as usize;
        if i < 256 {
            Some(&ALL_BYTES[i..i + 1])
        } else {
            None
        }
    }

    fn vocab_size(&self) -> u32 {
        256
    }
}
