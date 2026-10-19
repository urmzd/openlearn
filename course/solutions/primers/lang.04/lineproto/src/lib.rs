//! lineproto: the request and reply rules of the lang.04 echo server.
//!
//! One request is one line ending in `\n` (a `\r` just before it is
//! dropped, so `nc` and `telnet` work). The first word is the command; the
//! text is everything after the first space.
//!
//! | request      | reply                                                  |
//! |--------------|--------------------------------------------------------|
//! | `PING`       | `PONG`                                                 |
//! | `ECHO <text>`| `<text>`, byte for byte                                |
//! | `LEN <text>` | the UTF-8 byte length of `<text>`, from Rust's `.len()` |
//! | `QUIT`       | `BYE`, then the server closes the connection           |
//! | anything else| `ERR <reason>`; the connection stays open              |
//!
//! Nothing here does I/O. The binary in `echo/` reads lines from a socket
//! and writes `reply(line)` back, so every rule is testable without a socket.

use std::fmt;

/// A parsed request. `Echo` and `Len` own their text (a `String`), so a
/// `Command` outlives the buffer the line was read into.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    Ping,
    Echo(String),
    Len(String),
    Quit,
}

/// Every way a line can fail. The `Display` impl is the text after `ERR `.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProtoError {
    /// The line held no bytes before its `\n`.
    Empty,
    /// The line's bytes are not valid UTF-8.
    InvalidUtf8,
    /// The first word is not a command; carries that word.
    Unknown(String),
}

impl fmt::Display for ProtoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProtoError::Empty => write!(f, "empty line"),
            ProtoError::InvalidUtf8 => write!(f, "invalid utf-8"),
            ProtoError::Unknown(word) => write!(f, "unknown command {word}"),
        }
    }
}

impl std::error::Error for ProtoError {}

/// Parses one line (with or without its `\n` / `\r\n`) into a command.
pub fn parse(line: &str) -> Result<Command, ProtoError> {
    let line = line.strip_suffix('\n').unwrap_or(line);
    let line = line.strip_suffix('\r').unwrap_or(line);
    if line.is_empty() {
        return Err(ProtoError::Empty);
    }
    let (word, text) = line.split_once(' ').unwrap_or((line, ""));
    match word {
        "PING" => Ok(Command::Ping),
        "QUIT" => Ok(Command::Quit),
        "ECHO" => Ok(Command::Echo(text.to_string())),
        "LEN" => Ok(Command::Len(text.to_string())),
        _ => Err(ProtoError::Unknown(word.to_string())),
    }
}

/// The number of UTF-8 bytes in `text`, including any NUL bytes.
///
/// Rust strings store valid UTF-8, and `.len()` reports the byte length while
/// `.chars().count()` reports Unicode scalar values.
pub fn byte_len(text: &str) -> usize {
    text.len()
}

/// The reply text for a command (without the trailing `\n`).
pub fn respond(cmd: &Command) -> String {
    match cmd {
        Command::Ping => "PONG".to_string(),
        Command::Echo(text) => text.clone(),
        Command::Len(text) => byte_len(text).to_string(),
        Command::Quit => "BYE".to_string(),
    }
}

/// The whole protocol for one raw line from the socket: the reply text
/// (no `\n`) and whether the server must close the connection after it.
/// Errors become `ERR <reason>` and never close the connection.
pub fn reply(line: &[u8]) -> (String, bool) {
    let answer = std::str::from_utf8(line)
        .map_err(|_| ProtoError::InvalidUtf8)
        .and_then(parse)
        .map(|cmd| (respond(&cmd), cmd == Command::Quit));
    match answer {
        Ok(pair) => pair,
        Err(e) => (format!("ERR {e}"), false),
    }
}

// The unit tests live next to the code they test. `cargo test` in the
// workspace runs them.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ping_is_pong() {
        assert_eq!(reply(b"PING\n"), ("PONG".to_string(), false));
    }

    #[test]
    fn echo_keeps_everything_after_the_first_space() {
        assert_eq!(parse("ECHO  two  spaces\r\n"), Ok(Command::Echo(" two  spaces".to_string())));
    }

    #[test]
    fn len_counts_bytes_not_characters() {
        assert_eq!(byte_len("h\u{e9}llo"), 6);
        assert_eq!(byte_len("a\0b"), 3);
    }

    #[test]
    fn errors_keep_the_connection_open() {
        assert_eq!(reply(b"JUMP high\n"), ("ERR unknown command JUMP".to_string(), false));
    }
}
