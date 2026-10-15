//! Constrained decoding in the engine (L10.9): a Rust port of L8.7.
//!
//! A regex (or a JSON schema, translated to a regex) becomes a DFA over
//! BYTES; every token is a byte string; a token is allowed in state s
//! exactly when walking its bytes from s never leaves the live states. The
//! sampler (L10.1) then draws from the logits with every disallowed token at
//! -inf. The regex subset, the JSON-schema subset, and the canonical DFA
//! (dead states removed, Moore-minimized, numbered breadth-first from the
//! start taking bytes 0..255 in order) are those of
//! contracts/py/tinyllm/infer/constrain.pyi, so two equal languages give
//! equal transition tables and the masks equal your Python L8.7's.
//!
//! JSON documents (tool parameter schemas, enum values) are read with
//! [`Json`], which keeps object keys in the order written: "the properties
//! in the order given" is part of the language.
//!
//! Chapter: content/courses/build-an-llm/p10-serving/09-tool-calls-and-constrained-decoding.md.
//! tl-serve's `tools` (L10.9) builds the grammars of `tools`,
//! `tool_choice`, and `response_format` on top of this module.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::sync::{Arc, Mutex};

use crate::sample::{sample, Pcg32, SamplingParams};

/// The "state" after a byte with no live continuation.
pub const DEAD: i32 = -1;

/// Largest m or n in a {m,n} repetition.
pub const MAX_REPEAT: usize = 256;

/// A pattern or schema outside the subset, or a pattern with an empty
/// language; or a constraint misuse (a token the mask forbids).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConstrainError(pub String);

impl fmt::Display for ConstrainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // SOLUTION-BEGIN L10.9
        f.write_str(&self.0)
        // SOLUTION-END
    }
}

impl std::error::Error for ConstrainError {}

fn err<T>(msg: impl Into<String>) -> Result<T, ConstrainError> {
    // SOLUTION-BEGIN L10.9
    Err(ConstrainError(msg.into()))
    // SOLUTION-END
}

// -- byte sets ----------------------------------------------------------------

/// A set of byte values: bit b of the 256-bit mask.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct ByteSet(pub [u64; 4]);

impl ByteSet {
    pub const EMPTY: ByteSet = ByteSet([0; 4]);
    pub const ALL: ByteSet = ByteSet([u64::MAX; 4]);

    pub fn single(b: u8) -> ByteSet {
        // SOLUTION-BEGIN L10.9
        let mut s = ByteSet::EMPTY;
        s.0[(b >> 6) as usize] |= 1u64 << (b & 63);
        s
        // SOLUTION-END
    }

    /// Bytes lo..=hi.
    pub fn range(lo: u8, hi: u8) -> ByteSet {
        // SOLUTION-BEGIN L10.9
        let mut s = ByteSet::EMPTY;
        for b in lo..=hi {
            s = s.union(ByteSet::single(b));
        }
        s
        // SOLUTION-END
    }

    pub fn of(bytes: &[u8]) -> ByteSet {
        // SOLUTION-BEGIN L10.9
        bytes.iter().fold(ByteSet::EMPTY, |s, &b| s.union(ByteSet::single(b)))
        // SOLUTION-END
    }

    pub fn contains(&self, b: u8) -> bool {
        // SOLUTION-BEGIN L10.9
        self.0[(b >> 6) as usize] >> (b & 63) & 1 == 1
        // SOLUTION-END
    }

    pub fn union(self, o: ByteSet) -> ByteSet {
        // SOLUTION-BEGIN L10.9
        ByteSet([self.0[0] | o.0[0], self.0[1] | o.0[1], self.0[2] | o.0[2], self.0[3] | o.0[3]])
        // SOLUTION-END
    }

    pub fn inter(self, o: ByteSet) -> ByteSet {
        // SOLUTION-BEGIN L10.9
        ByteSet([self.0[0] & o.0[0], self.0[1] & o.0[1], self.0[2] & o.0[2], self.0[3] & o.0[3]])
        // SOLUTION-END
    }

    pub fn complement(self) -> ByteSet {
        // SOLUTION-BEGIN L10.9
        ByteSet([!self.0[0], !self.0[1], !self.0[2], !self.0[3]])
        // SOLUTION-END
    }

    pub fn is_empty(&self) -> bool {
        // SOLUTION-BEGIN L10.9
        self.0 == [0; 4]
        // SOLUTION-END
    }

    /// The single byte of a one-byte set (the parser's literals).
    fn only(&self) -> u8 {
        // SOLUTION-BEGIN L10.9
        (0..=255u8).find(|&b| self.contains(b)).unwrap_or(0)
        // SOLUTION-END
    }
}

fn digit() -> ByteSet {
    // SOLUTION-BEGIN L10.9
    ByteSet::range(b'0', b'9')
    // SOLUTION-END
}

fn word() -> ByteSet {
    // SOLUTION-BEGIN L10.9
    digit().union(ByteSet::range(b'A', b'Z')).union(ByteSet::range(b'a', b'z')).union(ByteSet::single(b'_'))
    // SOLUTION-END
}

fn space() -> ByteSet {
    // SOLUTION-BEGIN L10.9
    ByteSet::of(b" \t\n\r\x0c\x0b")
    // SOLUTION-END
}

/// Characters a backslash makes literal (L8.7's LITERAL_ESC).
const LITERAL_ESC: &[u8] = b"\\.*+?()[]{}|^$/\"-,:#&~ '";

// -- 1. parse -------------------------------------------------------------------

/// A parsed pattern: byte sets, concatenations, alternations, repetitions
/// (n = None: unbounded). `Cat(vec![])` is the empty string.
#[derive(Clone, Debug, PartialEq)]
pub enum Node {
    Set(ByteSet),
    Cat(Vec<Node>),
    Alt(Vec<Node>),
    Rep(Box<Node>, usize, Option<usize>),
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

impl<'a> Parser<'a> {
    fn error<T>(&self, msg: &str) -> Result<T, ConstrainError> {
        // SOLUTION-BEGIN L10.9
        err(format!("regex: {msg} at position {} in {:?}", self.i, String::from_utf8_lossy(self.s)))
        // SOLUTION-END
    }

    fn peek(&self) -> Option<u8> {
        // SOLUTION-BEGIN L10.9
        self.s.get(self.i).copied()
        // SOLUTION-END
    }

    fn alt(&mut self) -> Result<Node, ConstrainError> {
        // SOLUTION-BEGIN L10.9
        let mut branches = vec![self.cat()?];
        while self.peek() == Some(b'|') {
            self.i += 1;
            branches.push(self.cat()?);
        }
        Ok(if branches.len() == 1 { branches.pop().unwrap() } else { Node::Alt(branches) })
        // SOLUTION-END
    }

    fn cat(&mut self) -> Result<Node, ConstrainError> {
        // SOLUTION-BEGIN L10.9
        let mut items = Vec::new();
        while !matches!(self.peek(), None | Some(b'|') | Some(b')')) {
            items.push(self.repeat()?);
        }
        Ok(if items.len() == 1 { items.pop().unwrap() } else { Node::Cat(items) })
        // SOLUTION-END
    }

    fn repeat(&mut self) -> Result<Node, ConstrainError> {
        // SOLUTION-BEGIN L10.9
        let mut node = self.atom()?;
        while let Some(c @ (b'*' | b'+' | b'?' | b'{')) = self.peek() {
            let (m, n) = if c == b'{' {
                self.braces()?
            } else {
                self.i += 1;
                match c {
                    b'*' => (0, None),
                    b'+' => (1, None),
                    _ => (0, Some(1)),
                }
            };
            if self.peek() == Some(b'?') {
                return self.error("lazy quantifiers are not in the subset");
            }
            node = Node::Rep(Box::new(node), m, n);
        }
        Ok(node)
        // SOLUTION-END
    }

    fn braces(&mut self) -> Result<(usize, Option<usize>), ConstrainError> {
        // SOLUTION-BEGIN L10.9
        let j = match self.s[self.i..].iter().position(|&c| c == b'}') {
            Some(k) => self.i + k,
            None => return self.error("unclosed {"),
        };
        let body = String::from_utf8_lossy(&self.s[self.i + 1..j]).into_owned();
        let int = |t: &str| -> Option<i64> {
            let t = t.trim();
            let t = t.strip_prefix('+').unwrap_or(t);
            if t.is_empty() || !t.trim_start_matches('-').bytes().all(|b| b.is_ascii_digit()) || t.trim_start_matches('-').is_empty() {
                return None;
            }
            t.parse().ok()
        };
        let parts: Vec<&str> = body.split(',').collect();
        let (m, n) = match parts.as_slice() {
            [a] => match int(a) {
                Some(v) => (v, Some(v)),
                None => return self.error(&format!("bad repetition {{{body}}}")),
            },
            [a, b] => match (int(a), b.is_empty()) {
                (Some(v), true) => (v, None),
                (Some(v), false) => match int(b) {
                    Some(w) => (v, Some(w)),
                    None => return self.error(&format!("bad repetition {{{body}}}")),
                },
                (None, _) => return self.error(&format!("bad repetition {{{body}}}")),
            },
            _ => return self.error(&format!("bad repetition {{{body}}}")),
        };
        if m < 0 || n.is_some_and(|n| n < m) || m.max(n.unwrap_or(0)) > MAX_REPEAT as i64 {
            return self.error(&format!("repetition {{{body}}} out of range (0 <= m <= n <= {MAX_REPEAT})"));
        }
        self.i = j + 1;
        Ok((m as usize, n.map(|n| n as usize)))
        // SOLUTION-END
    }

    /// After a backslash: (byte set, is_class).
    fn escape(&mut self) -> Result<(ByteSet, bool), ConstrainError> {
        // SOLUTION-BEGIN L10.9
        let c = match self.peek() {
            Some(c) => c,
            None => return self.error("trailing backslash"),
        };
        self.i += 1;
        let class = match c {
            b'd' => Some(digit()),
            b'w' => Some(word()),
            b's' => Some(space()),
            b'D' => Some(digit().complement()),
            b'W' => Some(word().complement()),
            b'S' => Some(space().complement()),
            _ => None,
        };
        if let Some(set) = class {
            return Ok((set, true));
        }
        let single = match c {
            b'n' => Some(0x0A),
            b't' => Some(0x09),
            b'r' => Some(0x0D),
            b'f' => Some(0x0C),
            b'v' => Some(0x0B),
            _ => None,
        };
        if let Some(b) = single {
            return Ok((ByteSet::single(b), false));
        }
        if c == b'x' {
            let h = &self.s[self.i.min(self.s.len())..(self.i + 2).min(self.s.len())];
            if h.len() != 2 || !h.iter().all(|b| b.is_ascii_hexdigit()) {
                return self.error("\\x needs two hex digits");
            }
            self.i += 2;
            let v = u8::from_str_radix(std::str::from_utf8(h).unwrap(), 16).unwrap();
            return Ok((ByteSet::single(v), false));
        }
        if LITERAL_ESC.contains(&c) {
            return Ok((ByteSet::single(c), false));
        }
        self.error(&format!("unsupported escape \\{}", c as char))
        // SOLUTION-END
    }

    fn atom(&mut self) -> Result<Node, ConstrainError> {
        // SOLUTION-BEGIN L10.9
        let c = self.peek().unwrap_or(0);
        match c {
            b'(' => {
                self.i += 1;
                if self.s[self.i..].starts_with(b"?:") {
                    self.i += 2;
                } else if self.peek() == Some(b'?') {
                    return self.error("only (?:...) groups are in the subset");
                }
                let node = self.alt()?;
                if self.peek() != Some(b')') {
                    return self.error("unclosed (");
                }
                self.i += 1;
                Ok(node)
            }
            b'[' => Ok(Node::Set(self.bracket()?)),
            b'.' => {
                self.i += 1;
                Ok(Node::Set(ByteSet::single(b'\n').complement()))
            }
            b'\\' => {
                self.i += 1;
                Ok(Node::Set(self.escape()?.0))
            }
            b'*' | b'+' | b'?' | b'{' | b')' | b'^' | b'$' | b']' | b'}' => self.error(&format!("unexpected {:?}", c as char)),
            _ => {
                self.i += 1;
                Ok(Node::Set(ByteSet::single(c)))
            }
        }
        // SOLUTION-END
    }

    fn bracket(&mut self) -> Result<ByteSet, ConstrainError> {
        // SOLUTION-BEGIN L10.9
        self.i += 1; // [
        let neg = self.peek() == Some(b'^');
        if neg {
            self.i += 1;
        }
        let mut bits = ByteSet::EMPTY;
        let mut first = true;
        loop {
            let c = match self.peek() {
                Some(c) => c,
                None => return self.error("unclosed ["),
            };
            if c == b']' && !first {
                self.i += 1;
                break;
            }
            first = false;
            let (lo_set, is_class) = if c == b'\\' {
                self.i += 1;
                self.escape()?
            } else {
                self.i += 1;
                (ByteSet::single(c), false)
            };
            let next = self.s.get(self.i + 1).copied();
            if !is_class && self.peek() == Some(b'-') && next.is_some() && next != Some(b']') {
                self.i += 1;
                let d = self.peek().unwrap();
                let hi_set = if d == b'\\' {
                    self.i += 1;
                    let (h, hc) = self.escape()?;
                    if hc {
                        return self.error("a class cannot end a range");
                    }
                    h
                } else {
                    self.i += 1;
                    ByteSet::single(d)
                };
                let (lo, hi) = (lo_set.only(), hi_set.only());
                if hi < lo {
                    return self.error("range out of order");
                }
                bits = bits.union(ByteSet::range(lo, hi));
            } else {
                bits = bits.union(lo_set);
            }
        }
        Ok(if neg { bits.complement() } else { bits })
        // SOLUTION-END
    }
}

/// Parses the regex subset of constrain.pyi into a tree.
pub fn parse(pattern: &str) -> Result<Node, ConstrainError> {
    // SOLUTION-BEGIN L10.9
    if let Some(ch) = pattern.chars().find(|c| !c.is_ascii()) {
        return err(format!("non-ASCII character {ch:?} in pattern: write it as \\xHH bytes"));
    }
    let mut p = Parser { s: pattern.as_bytes(), i: 0 };
    let node = p.alt()?;
    if p.i != p.s.len() {
        return p.error(&format!("unexpected {:?}", p.s[p.i] as char));
    }
    Ok(node)
    // SOLUTION-END
}

// -- 2. NFA (Thompson) ------------------------------------------------------------

#[derive(Default)]
struct Nfa {
    eps: Vec<Vec<usize>>,
    edges: Vec<Vec<(ByteSet, usize)>>,
}

impl Nfa {
    fn state(&mut self) -> usize {
        // SOLUTION-BEGIN L10.9
        self.eps.push(Vec::new());
        self.edges.push(Vec::new());
        self.eps.len() - 1
        // SOLUTION-END
    }

    /// A fragment (start, end) for node; end has no outgoing edges yet.
    fn build(&mut self, node: &Node) -> (usize, usize) {
        // SOLUTION-BEGIN L10.9
        match node {
            Node::Set(set) => {
                let (s, e) = (self.state(), self.state());
                self.edges[s].push((*set, e));
                (s, e)
            }
            Node::Cat(items) => {
                let s = self.state();
                let mut e = s;
                for child in items {
                    let (cs, ce) = self.build(child);
                    self.eps[e].push(cs);
                    e = ce;
                }
                (s, e)
            }
            Node::Alt(items) => {
                let (s, e) = (self.state(), self.state());
                for child in items {
                    let (cs, ce) = self.build(child);
                    self.eps[s].push(cs);
                    self.eps[ce].push(e);
                }
                (s, e)
            }
            Node::Rep(child, m, n) => {
                let s = self.state();
                let mut e = s;
                for _ in 0..*m {
                    let (cs, ce) = self.build(child);
                    self.eps[e].push(cs);
                    e = ce;
                }
                match n {
                    None => {
                        let (cs, ce) = self.build(child);
                        let lp = self.state();
                        self.eps[e].push(lp);
                        self.eps[lp].push(cs);
                        self.eps[ce].push(lp);
                        (s, lp)
                    }
                    Some(n) => {
                        let end = self.state();
                        self.eps[e].push(end);
                        for _ in 0..(n - m) {
                            let (cs, ce) = self.build(child);
                            self.eps[e].push(cs);
                            self.eps[ce].push(end);
                            e = ce;
                        }
                        (s, end)
                    }
                }
            }
        }
        // SOLUTION-END
    }

    fn closure(&self, states: &[usize]) -> Vec<usize> {
        // SOLUTION-BEGIN L10.9
        let mut seen: HashSet<usize> = states.iter().copied().collect();
        let mut stack: Vec<usize> = states.to_vec();
        while let Some(x) = stack.pop() {
            for &t in &self.eps[x] {
                if seen.insert(t) {
                    stack.push(t);
                }
            }
        }
        let mut v: Vec<usize> = seen.into_iter().collect();
        v.sort_unstable();
        v
        // SOLUTION-END
    }
}

// -- 3. DFA ---------------------------------------------------------------------

/// A canonical DFA over bytes. `trans[s * 256 + b]` is the state after byte
/// b from s, or DEAD.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dfa {
    pub n_states: usize,
    pub accept: Vec<bool>,
    pub trans: Vec<i32>,
}

impl Dfa {
    /// The state after `data` from `state`; DEAD once a byte has no live
    /// continuation (and DEAD stays DEAD). Error for a state outside
    /// [0, n_states) other than DEAD.
    pub fn step(&self, state: i32, data: &[u8]) -> Result<i32, ConstrainError> {
        // SOLUTION-BEGIN L10.9
        if state != DEAD && !(0..self.n_states as i32).contains(&state) {
            return err(format!("state {state} outside [0, {})", self.n_states));
        }
        let mut s = state;
        for &b in data {
            if s == DEAD {
                return Ok(DEAD);
            }
            s = self.trans[s as usize * 256 + b as usize];
        }
        Ok(s)
        // SOLUTION-END
    }

    /// Whether `data` is a whole string of the language.
    pub fn matches(&self, data: &[u8]) -> bool {
        // SOLUTION-BEGIN L10.9
        match self.step(0, data) {
            Ok(s) if s != DEAD => self.accept[s as usize],
            _ => false,
        }
        // SOLUTION-END
    }
}

/// Partitions the 256 bytes so every NFA edge set is a union of classes.
fn byte_classes(nfa: &Nfa) -> Vec<ByteSet> {
    // SOLUTION-BEGIN L10.9
    let mut classes = vec![ByteSet::ALL];
    for out in &nfa.edges {
        for (bits, _) in out {
            let mut next = Vec::new();
            for c in classes {
                for x in [c.inter(*bits), c.inter(bits.complement())] {
                    if !x.is_empty() {
                        next.push(x);
                    }
                }
            }
            classes = next;
        }
    }
    classes
    // SOLUTION-END
}

/// Pattern to canonical DFA: parse, Thompson NFA, subset construction over
/// byte classes, dead states removed, Moore minimization, breadth-first
/// numbering from the start over bytes 0..255. Error for syntax outside the
/// subset or an empty language.
pub fn regex_to_dfa(pattern: &str) -> Result<Dfa, ConstrainError> {
    // SOLUTION-BEGIN L10.9
    let tree = parse(pattern)?;
    let mut nfa = Nfa::default();
    let (start, fin) = nfa.build(&tree);
    let classes = byte_classes(&nfa);
    let covers: Vec<Vec<(Vec<usize>, usize)>> = nfa
        .edges
        .iter()
        .map(|out| out.iter().map(|(bits, t)| ((0..classes.len()).filter(|&ci| !classes[ci].inter(*bits).is_empty()).collect(), *t)).collect())
        .collect();

    // Subset construction: a DFA state is a closed set of NFA states.
    let s0 = nfa.closure(&[start]);
    let mut ids: HashMap<Vec<usize>, usize> = HashMap::new();
    ids.insert(s0.clone(), 0);
    let mut sets = vec![s0.clone()];
    let mut moves: Vec<Vec<i32>> = Vec::new();
    let mut q = VecDeque::from([s0]);
    while let Some(cur) = q.pop_front() {
        let mut targets: Vec<Vec<usize>> = vec![Vec::new(); classes.len()];
        for &s in &cur {
            for (cis, t) in &covers[s] {
                for &ci in cis {
                    targets[ci].push(*t);
                }
            }
        }
        let mut row = Vec::with_capacity(classes.len());
        for tgt in targets {
            if tgt.is_empty() {
                row.push(DEAD);
                continue;
            }
            let nxt = nfa.closure(&tgt);
            let id = match ids.get(&nxt) {
                Some(&id) => id,
                None => {
                    let id = sets.len();
                    ids.insert(nxt.clone(), id);
                    sets.push(nxt.clone());
                    q.push_back(nxt);
                    id
                }
            };
            row.push(id as i32);
        }
        moves.push(row);
    }
    let n = sets.len();
    let acc: Vec<bool> = sets.iter().map(|s| s.binary_search(&fin).is_ok()).collect();

    // Live states: those from which an accepting state is reachable.
    let mut rev: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (s, row) in moves.iter().enumerate() {
        for &t in row {
            if t != DEAD {
                rev[t as usize].push(s);
            }
        }
    }
    let mut live = vec![false; n];
    let mut stack: Vec<usize> = (0..n).filter(|&s| acc[s]).collect();
    for &s in &stack {
        live[s] = true;
    }
    while let Some(x) = stack.pop() {
        for &p in &rev[x] {
            if !live[p] {
                live[p] = true;
                stack.push(p);
            }
        }
    }
    if !live[0] {
        return err(format!("regex: {pattern:?} matches nothing"));
    }
    for row in moves.iter_mut() {
        for t in row.iter_mut() {
            if *t != DEAD && !live[*t as usize] {
                *t = DEAD;
            }
        }
    }

    // Moore minimization over the live states.
    let alive: Vec<usize> = (0..n).filter(|&s| live[s]).collect();
    let mut block: HashMap<usize, usize> = alive.iter().map(|&s| (s, acc[s] as usize)).collect();
    loop {
        let sig: HashMap<usize, Vec<i64>> = alive
            .iter()
            .map(|&s| {
                let mut v = vec![block[&s] as i64];
                v.extend(moves[s].iter().map(|&t| if t == DEAD { -1 } else { block[&(t as usize)] as i64 }));
                (s, v)
            })
            .collect();
        let mut keys: Vec<&Vec<i64>> = sig.values().collect();
        keys.sort();
        keys.dedup();
        let index: HashMap<&Vec<i64>, usize> = keys.iter().enumerate().map(|(i, k)| (*k, i)).collect();
        let nb: HashMap<usize, usize> = alive.iter().map(|&s| (s, index[&sig[&s]])).collect();
        let before: HashSet<usize> = block.values().copied().collect();
        let stable = keys.len() == before.len();
        block = nb;
        if stable {
            break;
        }
    }

    // Canonical numbering: breadth-first from the start block, bytes ascending.
    let mut byte_class = [0usize; 256];
    for (ci, c) in classes.iter().enumerate() {
        for b in 0..=255u8 {
            if c.contains(b) {
                byte_class[b as usize] = ci;
            }
        }
    }
    let mut rep_of: HashMap<usize, usize> = HashMap::new();
    for &s in &alive {
        rep_of.entry(block[&s]).or_insert(s);
    }
    let mut order: HashMap<usize, usize> = HashMap::from([(block[&0], 0)]);
    let mut q = VecDeque::from([block[&0]]);
    let mut trans = Vec::new();
    let mut accept_of_block = Vec::new();
    while let Some(b0) = q.pop_front() {
        let s = rep_of[&b0];
        accept_of_block.push(acc[s]);
        for byte in 0..256 {
            let t = moves[s][byte_class[byte]];
            if t == DEAD {
                trans.push(DEAD);
                continue;
            }
            let bt = block[&(t as usize)];
            let next = order.len();
            let id = *order.entry(bt).or_insert_with(|| {
                q.push_back(bt);
                next
            });
            trans.push(id as i32);
        }
    }
    Ok(Dfa { n_states: order.len(), accept: accept_of_block, trans })
    // SOLUTION-END
}

// -- JSON values with key order kept ------------------------------------------

/// A JSON value. Objects keep their keys in the order written; numbers keep
/// whether they were integers (and their digits).
#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    /// An integer literal: an optional '-' and its digits, as written.
    Int(String),
    Float(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    /// Parses one JSON text (RFC 8259); error for anything else.
    pub fn parse(text: &str) -> Result<Json, ConstrainError> {
        // SOLUTION-BEGIN L10.9
        let b = text.as_bytes();
        let mut i = 0;
        let v = json_value(b, &mut i, 0)?;
        json_ws(b, &mut i);
        if i != b.len() {
            return err(format!("JSON: trailing characters at {i}"));
        }
        Ok(v)
        // SOLUTION-END
    }

    /// The value under `key` of an object (None for a missing key or a
    /// non-object).
    pub fn get(&self, key: &str) -> Option<&Json> {
        // SOLUTION-BEGIN L10.9
        match self {
            Json::Obj(kv) => kv.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
        // SOLUTION-END
    }

    /// Compact JSON exactly as Python's json.dumps(v, separators=(",",
    /// ":"), ensure_ascii=False) writes it: integers as their digits
    /// (-0 is 0), floats in Python's repr, strings with \" \\ \n \r \t \b
    /// \f escaped and other control characters as \u00xx, non-ASCII raw.
    pub fn dumps(&self) -> String {
        // SOLUTION-BEGIN L10.9
        let mut out = String::new();
        dump_into(self, &mut out);
        out
        // SOLUTION-END
    }
}

/// A JSON string literal (Python's escaping, ensure_ascii=False).
pub fn json_quote(s: &str) -> String {
    // SOLUTION-BEGIN L10.9
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
    // SOLUTION-END
}

/// Python's repr of a finite float: the shortest round-trip digits, in
/// scientific notation when the exponent is below -4 or at least 16
/// ("1e+16", "1.5e-05"), else positional with at least one fractional
/// digit ("100000.0").
pub fn python_float(x: f64) -> String {
    // SOLUTION-BEGIN L10.9
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    let sci = format!("{:e}", x.abs()); // shortest digits: d[.ddd]e<exp>
    let (mant, exp) = sci.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    let sign = if x < 0.0 { "-" } else { "" };
    if !(-4..16).contains(&exp) {
        let m = if digits.len() > 1 { format!("{}.{}", &digits[..1], &digits[1..]) } else { digits.clone() };
        return format!("{sign}{m}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs());
    }
    if exp < 0 {
        return format!("{sign}0.{}{digits}", "0".repeat((-exp - 1) as usize));
    }
    let point = exp as usize + 1;
    if digits.len() <= point {
        format!("{sign}{digits}{}.0", "0".repeat(point - digits.len()))
    } else {
        format!("{sign}{}.{}", &digits[..point], &digits[point..])
    }
    // SOLUTION-END
}

fn dump_into(v: &Json, out: &mut String) {
    // SOLUTION-BEGIN L10.9
    match v {
        Json::Null => out.push_str("null"),
        Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Json::Int(s) => {
            let neg = s.starts_with('-');
            let d = s.trim_start_matches('-').trim_start_matches('0');
            if d.is_empty() {
                out.push('0');
            } else {
                if neg {
                    out.push('-');
                }
                out.push_str(d);
            }
        }
        Json::Float(f) => out.push_str(&python_float(*f)),
        Json::Str(s) => out.push_str(&json_quote(s)),
        Json::Arr(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                dump_into(x, out);
            }
            out.push(']');
        }
        Json::Obj(kv) => {
            out.push('{');
            for (i, (k, x)) in kv.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&json_quote(k));
                out.push(':');
                dump_into(x, out);
            }
            out.push('}');
        }
    }
    // SOLUTION-END
}

fn json_ws(b: &[u8], i: &mut usize) {
    // SOLUTION-BEGIN L10.9
    while *i < b.len() && matches!(b[*i], b' ' | b'\t' | b'\n' | b'\r') {
        *i += 1;
    }
    // SOLUTION-END
}

fn json_value(b: &[u8], i: &mut usize, depth: usize) -> Result<Json, ConstrainError> {
    // SOLUTION-BEGIN L10.9
    if depth > 128 {
        return err("JSON: nested too deeply");
    }
    json_ws(b, i);
    let Some(&c) = b.get(*i) else { return err("JSON: unexpected end") };
    match c {
        b'{' => {
            *i += 1;
            let mut kv: Vec<(String, Json)> = Vec::new();
            json_ws(b, i);
            if b.get(*i) == Some(&b'}') {
                *i += 1;
                return Ok(Json::Obj(kv));
            }
            loop {
                json_ws(b, i);
                let Json::Str(k) = json_value(b, i, depth + 1)? else { return err("JSON: object key is not a string") };
                json_ws(b, i);
                if b.get(*i) != Some(&b':') {
                    return err(format!("JSON: ':' expected at {i}"));
                }
                *i += 1;
                let v = json_value(b, i, depth + 1)?;
                // A repeated key keeps the last value at the first key's place (Python's dict).
                match kv.iter_mut().find(|(a, _)| *a == k) {
                    Some(slot) => slot.1 = v,
                    None => kv.push((k, v)),
                }
                json_ws(b, i);
                match b.get(*i) {
                    Some(b',') => *i += 1,
                    Some(b'}') => {
                        *i += 1;
                        return Ok(Json::Obj(kv));
                    }
                    _ => return err(format!("JSON: ',' or '}}' expected at {i}")),
                }
            }
        }
        b'[' => {
            *i += 1;
            let mut a = Vec::new();
            json_ws(b, i);
            if b.get(*i) == Some(&b']') {
                *i += 1;
                return Ok(Json::Arr(a));
            }
            loop {
                a.push(json_value(b, i, depth + 1)?);
                json_ws(b, i);
                match b.get(*i) {
                    Some(b',') => *i += 1,
                    Some(b']') => {
                        *i += 1;
                        return Ok(Json::Arr(a));
                    }
                    _ => return err(format!("JSON: ',' or ']' expected at {i}")),
                }
            }
        }
        b'"' => {
            *i += 1;
            let mut s: Vec<u8> = Vec::new();
            loop {
                let Some(&c) = b.get(*i) else { return err("JSON: unterminated string") };
                *i += 1;
                match c {
                    b'"' => break,
                    b'\\' => {
                        let Some(&e) = b.get(*i) else { return err("JSON: unterminated escape") };
                        *i += 1;
                        match e {
                            b'"' | b'\\' | b'/' => s.push(e),
                            b'b' => s.push(8),
                            b'f' => s.push(12),
                            b'n' => s.push(b'\n'),
                            b'r' => s.push(b'\r'),
                            b't' => s.push(b'\t'),
                            b'u' => {
                                let hex = |at: usize| -> Option<u32> {
                                    let h = b.get(at..at + 4)?;
                                    u32::from_str_radix(std::str::from_utf8(h).ok()?, 16).ok()
                                };
                                let Some(mut cp) = hex(*i) else { return err("JSON: bad \\u escape") };
                                *i += 4;
                                if (0xD800..0xDC00).contains(&cp) && b.get(*i..*i + 2) == Some(b"\\u") {
                                    if let Some(lo) = hex(*i + 2).filter(|lo| (0xDC00..0xE000).contains(lo)) {
                                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                                        *i += 6;
                                    }
                                }
                                let ch = char::from_u32(cp).unwrap_or('\u{FFFD}');
                                let mut buf = [0u8; 4];
                                s.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                            }
                            _ => return err("JSON: bad escape"),
                        }
                    }
                    c if c < 0x20 => return err("JSON: control character in a string"),
                    c => s.push(c),
                }
            }
            String::from_utf8(s).map(Json::Str).map_err(|_| ConstrainError("JSON: string is not UTF-8".into()))
        }
        b't' if b[*i..].starts_with(b"true") => {
            *i += 4;
            Ok(Json::Bool(true))
        }
        b'f' if b[*i..].starts_with(b"false") => {
            *i += 5;
            Ok(Json::Bool(false))
        }
        b'n' if b[*i..].starts_with(b"null") => {
            *i += 4;
            Ok(Json::Null)
        }
        b'-' | b'0'..=b'9' => {
            let st = *i;
            if b[*i] == b'-' {
                *i += 1;
            }
            let int_start = *i;
            while *i < b.len() && b[*i].is_ascii_digit() {
                *i += 1;
            }
            if *i == int_start || (b[int_start] == b'0' && *i - int_start > 1) {
                return err(format!("JSON: bad number at {st}"));
            }
            let mut float = false;
            if b.get(*i) == Some(&b'.') {
                float = true;
                *i += 1;
                let f0 = *i;
                while *i < b.len() && b[*i].is_ascii_digit() {
                    *i += 1;
                }
                if *i == f0 {
                    return err(format!("JSON: bad number at {st}"));
                }
            }
            if matches!(b.get(*i), Some(b'e' | b'E')) {
                float = true;
                *i += 1;
                if matches!(b.get(*i), Some(b'+' | b'-')) {
                    *i += 1;
                }
                let e0 = *i;
                while *i < b.len() && b[*i].is_ascii_digit() {
                    *i += 1;
                }
                if *i == e0 {
                    return err(format!("JSON: bad number at {st}"));
                }
            }
            let text = std::str::from_utf8(&b[st..*i]).unwrap();
            if float {
                let f: f64 = text.parse().map_err(|_| ConstrainError(format!("JSON: bad number {text}")))?;
                if !f.is_finite() {
                    return err(format!("JSON: number {text} is out of range"));
                }
                Ok(Json::Float(f))
            } else {
                Ok(Json::Int(text.to_string()))
            }
        }
        _ => err(format!("JSON: unexpected {:?} at {i}", c as char)),
    }
    // SOLUTION-END
}

// -- JSON schema subset -------------------------------------------------------------

/// One JSON-string character: printable ASCII except '"' and '\', or a
/// well-formed UTF-8 sequence of 2 to 4 bytes (RFC 3629).
pub const UTF8_CHAR: &str = concat!(
    r"[\x20\x21\x23-\x5b\x5d-\x7f]|[\xc2-\xdf][\x80-\xbf]|\xe0[\xa0-\xbf][\x80-\xbf]",
    r"|[\xe1-\xec\xee\xef][\x80-\xbf]{2}|\xed[\x80-\x9f][\x80-\xbf]|\xf0[\x90-\xbf][\x80-\xbf]{2}",
    r"|[\xf1-\xf3][\x80-\xbf]{3}|\xf4[\x80-\x8f][\x80-\xbf]{2}"
);
pub const JSON_INTEGER: &str = r"-?(?:0|[1-9][0-9]*)";
pub const JSON_NUMBER: &str = r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?";

/// The regex of a JSON string (any characters, every escape).
pub fn json_string_regex() -> String {
    // SOLUTION-BEGIN L10.9
    format!(r#""(?:{UTF8_CHAR}|\\["\\/bfnrt]|\\u[0-9a-fA-F]{{4}})*""#)
    // SOLUTION-END
}

/// A regex matching exactly the UTF-8 bytes of `text`: ASCII letters and
/// digits as themselves, the escapable punctuation with a backslash, every
/// other byte as \xHH.
pub fn escape(text: &str) -> String {
    // SOLUTION-BEGIN L10.9
    let mut out = String::new();
    for &b in text.as_bytes() {
        if b.is_ascii_alphanumeric() {
            out.push(b as char);
        } else if b < 0x80 && LITERAL_ESC.contains(&b) {
            out.push('\\');
            out.push(b as char);
        } else {
            out.push_str(&format!("\\x{b:02x}"));
        }
    }
    out
    // SOLUTION-END
}

const IGNORED: [&str; 3] = ["title", "description", "default"];

/// L8.7's json_schema_to_regex: the compact JSON texts valid under the
/// schema subset of constrain.pyi. Error naming any other keyword.
pub fn json_schema_to_regex(schema: &Json) -> Result<String, ConstrainError> {
    // SOLUTION-BEGIN L10.9
    let Json::Obj(kv) = schema else {
        let t = match schema {
            Json::Arr(_) => "list",
            Json::Str(_) => "str",
            Json::Int(_) => "int",
            Json::Float(_) => "float",
            Json::Bool(_) => "bool",
            _ => "NoneType",
        };
        return err(format!("schema must be a dict, got {t}"));
    };
    let keys: Vec<&str> = kv.iter().map(|(k, _)| k.as_str()).filter(|k| !IGNORED.contains(k)).collect();
    let only = |allowed: &[&str]| -> Result<(), ConstrainError> {
        let mut extra: Vec<&str> = keys.iter().copied().filter(|k| !allowed.contains(k)).collect();
        extra.sort_unstable();
        match extra.first() {
            Some(k) => err(format!("JSON schema keyword {k:?} is not in the subset")),
            None => Ok(()),
        }
    };
    if schema.get("enum").is_some() || schema.get("const").is_some() {
        only(&["enum", "const", "type"])?;
        let values: Vec<Json> = match (schema.get("enum"), schema.get("const")) {
            (Some(Json::Arr(a)), _) => a.clone(),
            (Some(_), _) => return err("enum must be a list"),
            (None, Some(c)) => vec![c.clone()],
            (None, None) => unreachable!(),
        };
        if values.is_empty() {
            return err("enum must not be empty");
        }
        let alts: Vec<String> = values.iter().map(|v| escape(&v.dumps())).collect();
        return Ok(format!("(?:{})", alts.join("|")));
    }
    if let Some(subs) = schema.get("anyOf") {
        only(&["anyOf"])?;
        let Json::Arr(subs) = subs else { return err("anyOf must be a list") };
        if subs.is_empty() {
            return err("anyOf must not be empty");
        }
        let parts = subs.iter().map(json_schema_to_regex).collect::<Result<Vec<_>, _>>()?;
        return Ok(format!("(?:{})", parts.join("|")));
    }
    let t = match schema.get("type") {
        Some(Json::Str(s)) => s.as_str(),
        Some(other) => return err(format!("JSON schema type {} is not in the subset", other.dumps())),
        None => return err("JSON schema type None is not in the subset"),
    };
    let int_of = |v: Option<&Json>, name: &str| -> Result<Option<i64>, ConstrainError> {
        match v {
            None | Some(Json::Null) => Ok(None),
            Some(Json::Int(s)) => s.parse().map(Some).map_err(|_| ConstrainError(format!("{name} is too large"))),
            Some(other) => err(format!("{name} must be an integer, got {}", other.dumps())),
        }
    };
    match t {
        "string" => {
            only(&["type"])?;
            Ok(json_string_regex())
        }
        "integer" => {
            only(&["type"])?;
            Ok(JSON_INTEGER.to_string())
        }
        "number" => {
            only(&["type"])?;
            Ok(JSON_NUMBER.to_string())
        }
        "boolean" => {
            only(&["type"])?;
            Ok("(?:true|false)".to_string())
        }
        "null" => {
            only(&["type"])?;
            Ok("null".to_string())
        }
        "array" => {
            only(&["type", "items", "minItems", "maxItems"])?;
            let empty = Json::Obj(Vec::new());
            let item = json_schema_to_regex(schema.get("items").unwrap_or(&empty))?;
            let lo = int_of(schema.get("minItems"), "minItems")?.unwrap_or(0);
            let hi = int_of(schema.get("maxItems"), "maxItems")?;
            if lo < 0 || hi.is_some_and(|h| h < lo) {
                return err(format!("bad array bounds minItems={lo} maxItems={hi:?}"));
            }
            if hi == Some(0) {
                return Ok(r"\[\]".to_string());
            }
            let tail_hi = hi.map(|h| (h - 1).to_string()).unwrap_or_default();
            let tail_lo = (lo - 1).max(0);
            let body = format!("{item}(?:,{item}){{{tail_lo},{tail_hi}}}");
            Ok(if lo >= 1 { format!(r"\[{body}\]") } else { format!(r"\[(?:{body})?\]") })
        }
        "object" => {
            only(&["type", "properties", "required", "additionalProperties"])?;
            match schema.get("additionalProperties") {
                None | Some(Json::Bool(false)) => {}
                Some(_) => return err("additionalProperties must be false (or absent) in the subset"),
            }
            let props: &[(String, Json)] = match schema.get("properties") {
                None => &[],
                Some(Json::Obj(p)) => p,
                Some(_) => return err("properties must be an object"),
            };
            let required: Vec<&str> = match schema.get("required") {
                None => Vec::new(),
                Some(Json::Arr(a)) => a.iter().map(|x| if let Json::Str(s) = x { Ok(s.as_str()) } else { err("required names must be strings") }).collect::<Result<_, _>>()?,
                Some(_) => return err("required must be a list"),
            };
            let mut unknown: Vec<&str> = required.iter().copied().filter(|r| !props.iter().any(|(k, _)| k == r)).collect();
            unknown.sort_unstable();
            unknown.dedup();
            if !unknown.is_empty() {
                return err(format!("required names unknown properties {unknown:?}"));
            }
            let mut members = Vec::new();
            for (k, v) in props {
                members.push((format!("{}:{}", escape(&json_quote(k)), json_schema_to_regex(v)?), required.contains(&k.as_str())));
            }
            // Built from the end: `after` is members i.. once something has
            // been written (each needs a leading comma); `first` is members
            // i.. when nothing has been written yet.
            let (mut after, mut first) = (String::new(), String::new());
            for (m, req) in members.iter().rev() {
                first = if *req { format!("{m}{after}") } else { format!("(?:{m}{after}|{first})") };
                after = if *req { format!(",{m}{after}") } else { format!("(?:,{m})?{after}") };
            }
            Ok(format!(r"\{{{first}\}}"))
        }
        other => err(format!("JSON schema type {other:?} is not in the subset")),
    }
    // SOLUTION-END
}

/// A regex for any compact JSON object nested at most `depth` levels
/// (response_format json_object): a regular language needs a bound.
pub fn json_object_regex(depth: usize) -> String {
    // SOLUTION-BEGIN L10.9
    let s = json_string_regex();
    let scalar = format!("(?:{s}|{JSON_NUMBER}|true|false|null)");
    let mut value = scalar.clone();
    let mut object = String::new();
    for _ in 0..depth.max(1) {
        object = format!(r"\{{(?:{s}:{value}(?:,{s}:{value})*)?\}}");
        let array = format!(r"\[(?:{value}(?:,{value})*)?\]");
        value = format!("(?:{scalar}|{object}|{array})");
    }
    object
    // SOLUTION-END
}

// -- 4. token masks ----------------------------------------------------------------

/// A state's mask and next states, cached per state.
type MaskEntry = (Vec<bool>, Vec<i32>);

#[derive(Default)]
struct TrieNode {
    children: Vec<(u8, usize)>,
    ids: Vec<u32>,
}

/// Which tokens each DFA state allows, and where each leads. `vocab[i]` is
/// token i's bytes; None marks a special token (never allowed), as is an
/// empty token. `eos_id` is allowed exactly in accepting states. Masks are
/// computed on first use per state and cached (shared across requests).
pub struct TokenIndex {
    pub dfa: Dfa,
    pub vocab_size: usize,
    pub eos_id: Option<u32>,
    trie: Vec<TrieNode>,
    cache: Mutex<HashMap<usize, MaskEntry>>,
}

impl TokenIndex {
    /// Error for an eos_id outside [0, vocab.len()).
    pub fn new(dfa: Dfa, vocab: &[Option<Vec<u8>>], eos_id: Option<u32>) -> Result<TokenIndex, ConstrainError> {
        // SOLUTION-BEGIN L10.9
        if let Some(e) = eos_id {
            if e as usize >= vocab.len() {
                return err(format!("eos_id {e} outside [0, {})", vocab.len()));
            }
        }
        let mut trie = vec![TrieNode::default()];
        for (i, tb) in vocab.iter().enumerate() {
            let Some(tb) = tb else { continue };
            if tb.is_empty() || Some(i as u32) == eos_id {
                continue;
            }
            let mut node = 0;
            for &b in tb {
                node = match trie[node].children.iter().find(|(c, _)| *c == b) {
                    Some(&(_, n)) => n,
                    None => {
                        trie.push(TrieNode::default());
                        let n = trie.len() - 1;
                        trie[node].children.push((b, n));
                        n
                    }
                };
            }
            trie[node].ids.push(i as u32);
        }
        Ok(TokenIndex { dfa, vocab_size: vocab.len(), eos_id, trie, cache: Mutex::new(HashMap::new()) })
        // SOLUTION-END
    }

    fn build(&self, state: usize) -> MaskEntry {
        // SOLUTION-BEGIN L10.9
        let mut mask = vec![false; self.vocab_size];
        let mut next = vec![DEAD; self.vocab_size];
        let mut stack = vec![(0usize, state as i32)];
        while let Some((node, s)) = stack.pop() {
            for &(b, child) in &self.trie[node].children {
                let t = self.dfa.trans[s as usize * 256 + b as usize];
                if t == DEAD {
                    continue; // every token below this byte is dead too
                }
                for &id in &self.trie[child].ids {
                    mask[id as usize] = true;
                    next[id as usize] = t;
                }
                stack.push((child, t));
            }
        }
        if let Some(e) = self.eos_id {
            if self.dfa.accept[state] {
                mask[e as usize] = true;
                next[e as usize] = state as i32;
            }
        }
        (mask, next)
        // SOLUTION-END
    }

    fn entry<R>(&self, state: i32, f: impl FnOnce(&MaskEntry) -> R) -> Result<R, ConstrainError> {
        // SOLUTION-BEGIN L10.9
        if !(0..self.dfa.n_states as i32).contains(&state) {
            return err(format!("state {state} outside [0, {})", self.dfa.n_states));
        }
        let mut cache = self.cache.lock().unwrap_or_else(|p| p.into_inner());
        let e = cache.entry(state as usize).or_insert_with(|| self.build(state as usize));
        Ok(f(e))
        // SOLUTION-END
    }

    /// Token t is allowed iff dfa.step(state, vocab[t]) != DEAD (or t is
    /// eos_id and state accepts).
    pub fn mask(&self, state: i32) -> Result<Vec<bool>, ConstrainError> {
        // SOLUTION-BEGIN L10.9
        self.entry(state, |e| e.0.clone())
        // SOLUTION-END
    }

    /// The state after token_id: dfa.step(state, vocab[token_id]); for
    /// eos_id, the state itself when accepting, else DEAD; DEAD for a special.
    pub fn next_state(&self, state: i32, token_id: u32) -> Result<i32, ConstrainError> {
        // SOLUTION-BEGIN L10.9
        if token_id as usize >= self.vocab_size {
            return err(format!("token {token_id} outside [0, {})", self.vocab_size));
        }
        self.entry(state, |e| e.1[token_id as usize])
        // SOLUTION-END
    }
}

/// One request's place in the language. Starts in state 0. The index is
/// shared: every request with the same grammar and vocabulary uses one.
pub struct Constraint {
    pub index: Arc<TokenIndex>,
    pub state: i32,
    /// eos_id was emitted.
    pub done: bool,
}

impl Constraint {
    pub fn new(index: Arc<TokenIndex>) -> Constraint {
        // SOLUTION-BEGIN L10.9
        Constraint { index, state: 0, done: false }
        // SOLUTION-END
    }

    /// index.mask(state); all false once done.
    pub fn mask(&self) -> Vec<bool> {
        // SOLUTION-BEGIN L10.9
        if self.done {
            return vec![false; self.index.vocab_size];
        }
        self.index.mask(self.state).expect("a constraint's state is always live")
        // SOLUTION-END
    }

    /// Moves past an allowed token (eos_id sets done). Error for a token the
    /// mask forbids, or after done.
    pub fn advance(&mut self, token_id: u32) -> Result<(), ConstrainError> {
        // SOLUTION-BEGIN L10.9
        if self.done {
            return err("the constraint is done (EOS was emitted)");
        }
        let next = self.index.next_state(self.state, token_id)?;
        if next == DEAD {
            return err(format!("token {token_id} is not allowed in state {}", self.state));
        }
        if Some(token_id) == self.index.eos_id {
            self.done = true;
        }
        self.state = next;
        Ok(())
        // SOLUTION-END
    }

    /// The output so far is a whole string of the language.
    pub fn is_complete(&self) -> bool {
        // SOLUTION-BEGIN L10.9
        self.index.dfa.accept[self.state as usize]
        // SOLUTION-END
    }
}

/// logits with -inf wherever mask is false. Error for mismatched lengths or
/// a mask that allows nothing.
pub fn apply_mask(logits: &[f32], mask: &[bool]) -> Result<Vec<f32>, ConstrainError> {
    // SOLUTION-BEGIN L10.9
    if logits.len() != mask.len() {
        return err(format!("logits ({}) and mask ({}) differ in length", logits.len(), mask.len()));
    }
    if !mask.iter().any(|&m| m) {
        return err("the mask allows no token");
    }
    Ok(logits.iter().zip(mask).map(|(&l, &m)| if m { l } else { f32::NEG_INFINITY }).collect())
    // SOLUTION-END
}

/// L10.1's sample on the masked logits, then advance: the token and its
/// logprob (renormalized over the allowed tokens, before temperature).
/// Greedy picks the largest allowed logit, ties to the lowest id.
pub fn constrained_sample(
    logits: &[f32],
    c: &mut Constraint,
    p: &SamplingParams,
    prompt: &[u32],
    output: &[u32],
    rng: &mut Pcg32,
) -> Result<(u32, f64), ConstrainError> {
    // SOLUTION-BEGIN L10.9
    let masked = apply_mask(logits, &c.mask())?;
    let (tok, lp) = sample(&masked, p, prompt, output, rng);
    c.advance(tok)?;
    Ok((tok, lp))
    // SOLUTION-END
}
