//! The streaming UTF-8 decoder (L1.5): one id in, the complete characters
//! it finishes out.
//!
//! A byte-level token is a run of bytes, not of characters: `é` (`c3 a9`)
//! can be split across two tokens, and an emoji across four. A server that
//! streams one chunk per generated token must not emit half a character, and
//! must still make progress on bytes that can never form one. The rules are
//! those of formats/tokenizer.md ("Incremental decoding"):
//!
//! 1. Append the token's bytes to the pending buffer.
//! 2. Emit the longest valid UTF-8 prefix. If the next bytes are an invalid
//!    sequence, emit one U+FFFD for that maximal subpart, drop it, and
//!    continue. If they are an incomplete sequence that could still
//!    complete, stop and keep them pending.
//! 3. At the end, `finish` decodes whatever is pending with replacement.
//!
//! Concatenating every chunk and `finish` gives exactly `decode(ids)`.
//! `std::str::from_utf8` reports the split: `valid_up_to()`, and
//! `error_len()`, which is `None` exactly when the tail is incomplete.

use crate::{DecodeError, Tokenizer};

pub struct StreamDecoder<'a, T: Tokenizer + ?Sized> {
    tok: &'a T,
    pending: Vec<u8>,
}

impl<'a, T: Tokenizer + ?Sized> StreamDecoder<'a, T> {
    pub fn new(tok: &'a T) -> Self {
        StreamDecoder { tok, pending: Vec::new() }
    }

    /// Feeds one id. `Ok(Some(text))` when it completes at least one
    /// character (or an invalid byte becomes U+FFFD); `Ok(None)` when every
    /// byte so far is still pending; `Err` for an id outside the vocabulary
    /// (the decoder is unchanged).
    pub fn push(&mut self, id: u32) -> Result<Option<String>, DecodeError> {
        let bytes = self
            .tok
            .token_bytes(id)
            .ok_or(DecodeError::UnknownId { id, vocab_size: self.tok.vocab_size() })?;
        self.pending.extend_from_slice(bytes);
        let mut out = String::new();
        let mut start = 0;
        loop {
            match std::str::from_utf8(&self.pending[start..]) {
                Ok(s) => {
                    out.push_str(s);
                    start = self.pending.len();
                    break;
                }
                Err(e) => {
                    let good = e.valid_up_to();
                    out.push_str(std::str::from_utf8(&self.pending[start..start + good]).unwrap());
                    start += good;
                    match e.error_len() {
                        Some(bad) => {
                            out.push('\u{FFFD}');
                            start += bad;
                        }
                        None => break, // incomplete: keep it pending
                    }
                }
            }
        }
        self.pending.drain(..start);
        Ok(if out.is_empty() { None } else { Some(out) })
    }

    /// Bytes held back, waiting for the rest of a character (at most 3).
    pub fn pending(&self) -> &[u8] {
        &self.pending
    }

    /// Ends the stream: the pending bytes decoded with replacement (one
    /// U+FFFD for an incomplete tail), or `""`. The decoder is empty after.
    pub fn finish(&mut self) -> String {
        let out = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        out
    }
}
