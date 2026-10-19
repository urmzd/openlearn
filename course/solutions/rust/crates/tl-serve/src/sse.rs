//! Server-Sent Events framing and the text side of streaming (L10.5).
//!
//! Each token arrives as bytes (a byte-level BPE token can be half a
//! character). [`TextStream`] turns that byte stream into text that is safe
//! to send now:
//!
//! 1. **Incremental UTF-8** (formats/tokenizer.md): bytes that may still
//!    complete a character wait; an invalid sequence becomes one U+FFFD.
//! 2. **Stop strings** (openai-subset.v1.yaml): a stop string ends the
//!    generation and is never sent, even when it spans several tokens. So
//!    the tail of the text that could be the beginning of a stop string is
//!    held back until the next token decides.
//!
//! Concatenating every piece `push` returns plus `finish` gives exactly the
//! non-streamed text, which conformance case `chat.stream.equals_nonstream`
//! checks.

/// One SSE event carrying `payload` (a JSON document): `data: <payload>\n\n`.
pub fn event(payload: &str) -> String {
    format!("data: {payload}\n\n")
}

/// The terminator of an OpenAI stream.
pub const DONE: &str = "data: [DONE]\n\n";

/// An SSE comment, sent when no token is ready for a while, so proxies
/// keep the connection open.
pub const PING: &str = ": ping\n\n";

/// Bytes in, displayable text out, with stop strings removed.
#[derive(Clone, Debug, Default)]
pub struct TextStream {
    pending: Vec<u8>,
    held: String,
    stops: Vec<String>,
    stopped: bool,
}

impl TextStream {
    /// A stream that ends at the first occurrence of any of `stops`.
    pub fn new(stops: Vec<String>) -> TextStream {
        TextStream { pending: Vec::new(), held: String::new(), stops: stops.into_iter().filter(|s| !s.is_empty()).collect(), stopped: false }
    }

    /// A stop string was found; nothing more is emitted.
    pub fn stopped(&self) -> bool {
        self.stopped
    }

    /// Decodes as much of `pending` as is complete into `held`.
    fn decode(&mut self, final_: bool) {
        let mut start = 0;
        loop {
            match std::str::from_utf8(&self.pending[start..]) {
                Ok(s) => {
                    self.held.push_str(s);
                    start = self.pending.len();
                    break;
                }
                Err(e) => {
                    let good = e.valid_up_to();
                    self.held.push_str(std::str::from_utf8(&self.pending[start..start + good]).expect("valid prefix"));
                    start += good;
                    match e.error_len() {
                        Some(bad) => {
                            self.held.push('\u{FFFD}');
                            start += bad;
                        }
                        None if final_ => {
                            self.held.push('\u{FFFD}');
                            start = self.pending.len();
                            break;
                        }
                        None => break,
                    }
                }
            }
        }
        self.pending.drain(..start);
    }

    /// Adds one token's bytes; returns the text that can be sent now. After
    /// a stop string is found it returns the text before it and then only
    /// empty strings.
    pub fn push(&mut self, bytes: &[u8]) -> String {
        if self.stopped {
            return String::new();
        }
        self.pending.extend_from_slice(bytes);
        self.decode(false);
        if let Some(at) = self.stops.iter().filter_map(|s| self.held.find(s.as_str())).min() {
            self.stopped = true;
            let out = self.held[..at].to_string();
            self.held.clear();
            self.pending.clear();
            return out;
        }
        // hold back the longest tail that is a proper prefix of a stop string
        let mut keep = 0;
        for s in &self.stops {
            for (i, _) in s.char_indices().skip(1) {
                if self.held.ends_with(&s[..i]) {
                    keep = keep.max(i);
                }
            }
        }
        let cut = self.held.len() - keep;
        let out = self.held[..cut].to_string();
        self.held.drain(..cut);
        out
    }

    /// The end of generation (length or EOS): everything still held, with
    /// an incomplete UTF-8 tail as U+FFFD.
    pub fn finish(&mut self) -> String {
        if self.stopped {
            return String::new();
        }
        self.decode(true);
        std::mem::take(&mut self.held)
    }
}
