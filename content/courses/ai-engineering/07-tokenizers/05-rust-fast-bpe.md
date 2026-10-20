# Fast BPE in Rust

The Python BPE of [Byte-level BPE](02-byte-level-bpe.md) matches GPT-2 and SmolLM2 exactly. But the serving engine of this book is written in Rust: [the model runner](../23-model-serving/02-model-runner-and-sampler.md) and [the OpenAI-compatible server](../23-model-serving/06-openai-server-on-tokio.md) need a tokenizer on every request, and the server's SSE path needs incremental decoding for the moment a token ends in the middle of a UTF-8 character. This section builds that tokenizer independently in Rust as the crate `tl-tok`. It adds the byte tokenizer the tracer has served from the start, a standalone CLI, and a streaming decoder.

The Rust code has no freedom in what it outputs; the ids are fixed by the real files. Both implementations are held to the same oracle: a shared golden JSON fixture pins Hugging Face ids for both vocabularies, the Rust crate is checked against those frozen vectors, and the Python is checked against the same golden JSON. Parity crosses the language boundary only through tokenizer files and frozen JSON vectors; there is no extension build step and no function call from one language into the other. The Rust version's freedom is all in speed, which comes from three things: Robin Hood maps for the tables, a lazy heap for the merges, and threads.

You will lean on several earlier sections. The Python BPE is the specification every id is checked against. From *Algorithms* you need the [Robin Hood hash map](../../programming-and-algorithms/15-systems-data-structures/05-robin-hood-hash-map.md) and the [binary heap with lazy deletion](../../programming-and-algorithms/15-systems-data-structures/06-binary-heap-lazy-deletion.md). For background, read the [Rust primer](../../programming-and-algorithms/01-language-primers/04-rust.md) in *Programming and Algorithms* and the [GPT-2 byte map](../../mathematics/11-discrete-math-1/02-injective-surjective-bijective-gpt2-byte-map.md) in *Mathematics*.

The crate is a library of five modules plus a small binary: [`lib.rs`](../../../../course/solutions/rust/crates/tl-tok/src/lib.rs) (the `Tokenizer` trait, `ByteTokenizer`, `SpecialSet`, errors), [`pretok.rs`](../../../../course/solutions/rust/crates/tl-tok/src/pretok.rs) (`Gpt2Split`, `DigitSplit`, `PreTokenizer` over general-category tables), [`bpe.rs`](../../../../course/solutions/rust/crates/tl-tok/src/bpe.rs) (`ByteBpe`, `encode_piece_with`, `encode_batch`), [`hfjson.rs`](../../../../course/solutions/rust/crates/tl-tok/src/hfjson.rs) (`from_hf_json`, `from_gpt2_files`), [`stream.rs`](../../../../course/solutions/rust/crates/tl-tok/src/stream.rs) (`StreamDecoder`), and the binary [`main.rs`](../../../../course/solutions/entry/rust/crates/tl-tok/src/main.rs) (`tl-tok encode`, `tl-tok bench`). The file formats and parity rules are in [`tokenizer.md`](../../../../course/contracts/formats/tokenizer.md) (the bytes tokenizer, the BPE subset, incremental decoding), and the process role is in [`cli-roles.md`](../../../../course/contracts/spec/cli-roles.md). The binary's two verbs are part of the system: `tl-tok encode --tokenizer <file> --in <texts>` prints the ids and bytes per token as one JSON line, and `tl-tok bench` times `encode_batch` on one thread and on `--threads T` (run it with `cargo run --release -p tl-tok --`). With the Python, `tl-tok encode` and `tinyllm tok encode` both give the GPT-2 and SmolLM2 oracle ids.

> **Learning objectives**
> - Explain why a Rust tokenizer must work on bytes and use character-class tables generated from Python's `unicodedata`.
> - Encode a pre-token with a lazy heap keyed by (rank, position) and explain why each merge must remove the two pairs it destroys.
> - Build a streaming UTF-8 decoder whose concatenated chunks equal one-shot decoding.
> - Parallelize batch encoding so the output does not depend on the thread count.

## Bytes, the byte map, and the trait

The notation for this section:

| Symbol | Meaning | Type |
|---|---|---|
| $V$ | vocabulary size: ids $0 .. V-1$ | `u32` |
| $b$ | a byte, $0 \le b \le 255$ | `u8` |
| $\beta(b)$ | GPT-2's byte-to-character map | `[char; 256]` |
| $x$ | a pre-token: a run of bytes merged in isolation | `&[u8]` |
| $s_0 \dots s_{n-1}$ | the symbols of $x$ while merging: token ids | `u32` |
| $r(a, b)$ | the rank of merging ids $a$ and $b$ (lower merges first), if any | `Option<u32>` |
| $T$ | threads of `encode_batch` | `usize` |

Every tokenizer implements one trait: `encode`, `encode_with_special` (which says which added tokens may match), `token_bytes(id)`, `vocab_size`, and, defined once from `token_bytes`, `decode_bytes` and `decode`. Decoding is UTF-8 with replacement: one U+FFFD per maximal ill-formed subsequence, as `tokenizer.md` specifies. The tracer's **byte tokenizer** is the simplest implementation: the id of a byte is its value, 256 ids, nothing to load.

A byte-level BPE file writes its tokens as strings in GPT-2's byte alphabet $\beta$. The 188 printable bytes stand for themselves and the other 68 map to U+0100, U+0101, ... in byte order, so the space 0x20 is `Ġ` (U+0120) and 0xAD, the last non-printable, is U+0143. `tl-tok` turns every token string back into bytes once, at load, with $\beta^{-1}$ (a bijection), and works on bytes from then on. `ByteBpe` keeps the vocabulary as a `RobinHoodMap<Vec<u8>, u32>` (looked up by `&[u8]`), the id-to-bytes table as a `Vec<Vec<u8>>`, and the merge table as `RobinHoodMap<(u32, u32), Merge>` with `Merge { rank, id }`.

## The pre-tokenizer, by hand

Merges never cross a pre-token boundary, so the pre-tokenizer decides what the merge loop ever sees. GPT-2's boundaries come from one regular expression, tried left to right at each position:

```text
's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

`Gpt2Split` is that expression as a loop. At the current position: (1) an apostrophe followed by one of the lowercase suffixes `s t re ve m ll d` is a contraction (`'S` is not: it splits into `'` and `S`); (2 to 4) otherwise an optional single U+0020 followed by a maximal run of one class, letters (`\p{L}`, the general categories Lu Ll Lt Lm Lo), numbers (`\p{N}`: Nd Nl No), or other non-space characters; (5) a white-space run gives back its last character when a non-space follows, because that character belongs to the next pre-token (`\s+(?!\S)`); (6) else the white space itself. `\s` is the Unicode White_Space property, 25 code points. Rust's `char::is_whitespace` is the same set; Python's `str.isspace` is not.

The character classes must be the **general categories the Python uses**, or the ids drift apart on rare characters. Rust's `char::is_alphabetic` is the Alphabetic property, which also includes marks such as the Devanagari vowel signs (Mc), so `हिन्दी` would stay one run in Rust and split into six pieces in Python. `tl-tok` therefore embeds two tables of code-point ranges, `LETTER_RANGES` and `NUMBER_RANGES`, generated from Python's `unicodedata` by a short script that walks every code point and records the runs whose `unicodedata.category` starts with `L` or `N`; membership is a binary search. The Unicode version matters too. Each Python version ships one (3.11 has 14.0, 3.12 has 15.0, 3.14 has 16.0), and a character added later is unassigned, so not a letter, to an older one. The tables must come from the Python the Python BPE runs under; the solution's header records Unicode 15.0.0 from Python 3.12.

SmolLM2 adds one step before ByteLevel: `Digits` with `individual_digits = true` cuts every `\p{N}` code point into its own piece, so `12345` is five pre-tokens and five tokens. A `PreTokenizer` is a list of steps, each applied to every piece the previous step produced.

## Merging with a lazy heap

Inside one pre-token, BPE starts from one symbol per byte and repeatedly merges the adjacent pair $(s_i, s_{i+1})$ with the lowest $r$, the **leftmost** among equal ranks, until no adjacent pair has a rank. The rescan written straight from the definition costs $O(n)$ per merge and $O(n^2)$ per pre-token.

`encode_piece` does better. It keeps the symbols in a doubly linked list over an array (`prev`, `next`) and every mergeable pair in a `LazyHeap` keyed by the tuple (rank, position of the left symbol), so the tuple order is exactly "lowest rank, then leftmost". Popping $(r, i)$ merges $s_i$ and its successor $s_j$:

1. the pairs $(s_{prev}, s_i)$ and $(s_j, s_{next})$ no longer exist, so remove them **by handle** from the heap (this is what lazy deletion is for);
2. $s_i$ becomes the merged id and absorbs $s_j$'s link;
3. the new pairs $(s_{prev}, s_i)$ and $(s_i, s_{next})$ are pushed if they have a rank.

Each merge is $O(\log n)$, so a pre-token costs $O(n \log n)$ instead of $O(n^2)$, and the ids equal the rescan's, because both always apply the pair the definition names. A differential check on 3,000 random pieces confirms it. One `LazyHeap`, cleared between pieces, serves a whole text.

**Added tokens** (`<|endoftext|>`, SmolLM2's `<|im_start|>` and 15 more) are split out first, leftmost-longest, as the Hugging Face library does, and never pre-tokenized or merged. `encode` allows all of them; `encode_with_special(text, &SpecialSet::none())` allows none, so a server can keep user text from forging control tokens.

## Loading `tokenizer.json`

`ByteBpe::from_hf_json` reads the BPE subset of `tokenizer.md` that GPT-2's and SmolLM2's files use: `model.type` `BPE` (or absent, when `model.merges` is present), `model.vocab`, `model.merges` as `"left right"` strings or `[left, right]` pairs, a `ByteLevel` pre-tokenizer or a `Sequence` of `Digits` and one final `ByteLevel`, and `added_tokens`. Anything else (a normalizer, another model, `byte_fallback`, `add_prefix_space: true`) is refused with an error that names the field, so a file that loads is a file this code encodes exactly. A merge whose parts or result are not in the vocabulary is a `Format` error, and a repeated merge keeps its first rank. `from_gpt2_files` reads the classic `vocab.json` and `merges.txt`, skipping the `#version` line. The tokenizer the Python trainer saves loads here unchanged, which is how [the TinyStories capstone](../capstones/01-tinystories.md) serves the vocabulary it trains.

## Streaming decode

A token's bytes need not end on a character boundary: GPT-2 splits the emoji U+1F600 (`f0 9f 98 80`) into tokens 47249 (`f0 9f 98`) and 222 (`80`). An engine that streams one SSE chunk per token must neither emit half a character nor stall on bytes that can never form one.

`StreamDecoder` keeps the pending bytes and, after each token, emits the longest valid UTF-8 prefix. An invalid sequence becomes one U+FFFD for its maximal subpart, and an incomplete tail that could still complete stays pending, at most 3 bytes. `std::str::from_utf8` gives exactly the two numbers needed: `valid_up_to()`, and `error_len()`, which is `None` for "incomplete, wait". `finish` decodes what is left with replacement. Concatenating every chunk and `finish` gives `decode(ids)`, the property the server's conformance case `chat.stream.equals_nonstream` relies on.

## Threads and a bounded piece cache

The last speed source is parallelism. `encode_batch(texts, T)` cuts the texts into $T$ contiguous chunks, encodes each on its own thread with `std::thread::scope` (which lets the threads borrow the tokenizer and the texts), and joins the results **in chunk order**, so the output does not depend on $T$. $T = 0$ means the hardware concurrency. Each worker keeps its own piece cache, another `RobinHoodMap` from pre-token bytes to ids, cleared at 65,536 entries. Real text repeats its words, so most pieces are hits, and no lock is shared between threads.

## Working it by hand

Take the hand example from the Python section: the 256 byte tokens in the Python's order (ids by the code point of their byte-map character, so `g` is 70, `s` is 82, `Ġ` is 220) and three merges: rank 0 `u g` makes id 256 (`ug`), rank 1 `h ug` makes 257 (`hug`), rank 2 `p ug` makes 258 (`pug`). Encode `hugs pug`.

The pre-tokens are `hugs`, a letter run, and ` pug`, since the space joins the next run.

For `hugs`, the symbols are `h u g s` at positions 0 to 3. The only mergeable pair is `(u, g)` at position 1, rank 0, so the heap starts as $\{(0, 1)\}$. Each pop:

| Pop | Merge | Removed pairs | New pairs pushed | Symbols |
|---|---|---|---|---|
| (0, 1) | `u g` to 256 at position 1 | none queued at 0 or 2 | `(h, ug)` at 0, rank 1; `(ug, s)` has no rank | `h ug s` |
| (1, 0) | `h ug` to 257 at position 0 | none | `(hug, s)` has no rank | `hug s` |

The ids are 257, 82.

For ` pug`, the symbols are `Ġ p u g`, and the only ranked pair is `(u, g)` at position 2. Pop (0, 2): `ug` at 2, and push `(p, ug)` at 1 with rank 2. Pop (2, 1): `pug` at 1; `(Ġ, pug)` has no rank. The ids are 220, 258. The result is `[257, 82, 220, 258]`, exactly the Python's.

Lazy removal shows up on `aaaa` with merges rank 0 `a a` (256) and rank 1 `aa aa` (257). The heap holds $(0,0), (0,1), (0,2)$. Pop $(0,0)$: positions 0 and 1 merge, and the queued pair at position 1 (`a` at 1 with `a` at 2) no longer exists, so it is removed by its handle. Pop $(0, 2)$: positions 2 and 3 merge, and the new pair `(aa, aa)` at 0 is pushed with rank 1. Pop $(1, 0)$: one token, 257. Without the removal, the stale $(0, 1)$ would pop second and merge a symbol that is already gone.

Streaming the emoji: push 47249, and the pending bytes are `f0 9f 98`, a 4-byte lead and two continuations, incomplete, so the chunk is empty. Push 222: `80` completes the character and the chunk is `😀`.

## Designing the Rust interface

The crate's public surface:

```rust
// rust/crates/tl-tok/src/lib.rs
pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str) -> Vec<u32>;                          // every added token matches
    fn encode_with_special(&self, text: &str, allowed: &SpecialSet) -> Vec<u32>;
    fn token_bytes(&self, id: u32) -> Option<&[u8]>;
    fn vocab_size(&self) -> u32;
    fn decode_bytes(&self, ids: &[u32]) -> Result<Vec<u8>, DecodeError>;   // provided
    fn decode(&self, ids: &[u32]) -> Result<String, DecodeError>;          // provided: UTF-8 with replacement
}
pub struct SpecialSet;  impl SpecialSet { pub fn all() -> Self; pub fn none() -> Self; pub fn only(t: &[&str]) -> Self; pub fn allows(&self, t: &str) -> bool; }
pub enum DecodeError { UnknownId { id: u32, vocab_size: u32 } }
pub enum LoadError { Io { path: PathBuf, source: io::Error }, Json(String), Unsupported { field: String, value: String }, Format(String) }
pub struct ByteTokenizer;                                              // impl Tokenizer: id = byte

// pretok.rs
pub const LETTER_RANGES: &[(u32, u32)];  pub const NUMBER_RANGES: &[(u32, u32)];  pub const WHITE_SPACE: &[char];
pub fn is_letter(c: char) -> bool;  pub fn is_number(c: char) -> bool;  pub fn is_space(c: char) -> bool;
pub fn pretokenize_gpt2(text: &str) -> Vec<&str>;  pub fn split_digits(text: &str, individual: bool) -> Vec<&str>;
pub enum Step { Digits { individual: bool }, ByteLevel }
pub struct PreTokenizer { pub steps: Vec<Step> }  impl PreTokenizer { pub fn gpt2() -> Self; pub fn split<'a>(&self, text: &'a str) -> Vec<&'a str>; }

// bpe.rs
pub fn bytes_to_unicode() -> [char; 256];  pub fn unicode_to_bytes() -> Vec<Option<u8>>;
pub struct AddedToken { pub content: String, pub id: u32, pub special: bool }
pub struct Merge { pub rank: u32, pub id: u32 }
pub fn by_rank(a: &(u32, usize), b: &(u32, usize)) -> Ordering;     // the merge queue's order
pub struct ByteBpe;                                                    // impl Tokenizer
impl ByteBpe {
    pub fn new(vocab: Vec<(Vec<u8>, u32)>, merges: Vec<(Vec<u8>, Vec<u8>)>, added: Vec<AddedToken>, pretok: PreTokenizer) -> Result<Self, LoadError>;
    pub fn from_hf_json(path: &Path) -> Result<Self, LoadError>;  pub fn from_hf_json_str(text: &str) -> Result<Self, LoadError>;
    pub fn from_gpt2_files(vocab_json: &Path, merges_txt: &Path) -> Result<Self, LoadError>;
    pub fn token_to_id(&self, bytes: &[u8]) -> Option<u32>;  pub fn merge(&self, left: u32, right: u32) -> Option<Merge>;
    pub fn merge_count(&self) -> usize;  pub fn added_tokens(&self) -> &[AddedToken];  pub fn pre_tokenizer(&self) -> &PreTokenizer;
    pub fn encode_piece(&self, piece: &[u8], out: &mut Vec<u32>);
    pub fn encode_piece_with(&self, piece: &[u8], out: &mut Vec<u32>, scratch: &mut MergeScratch);
    pub fn encode_ordinary(&self, text: &str) -> Vec<u32>;
    pub fn encode_cached(&self, text: &str, cache: &mut RobinHoodMap<Vec<u8>, Vec<u32>>) -> Vec<u32>;
    pub fn encode_batch(&self, texts: &[&str], threads: usize) -> Vec<Vec<u32>>;
}

// stream.rs
pub struct StreamDecoder<'a, T: Tokenizer + ?Sized>;
impl<'a, T: Tokenizer + ?Sized> StreamDecoder<'a, T> {
    pub fn new(tok: &'a T) -> Self;
    pub fn push(&mut self, id: u32) -> Result<Option<String>, DecodeError>;   // None: nothing completed yet
    pub fn pending(&self) -> &[u8];  pub fn finish(&mut self) -> String;
}
```

The Python API remains `BPETokenizer` from the Python section, and its parity checks read the same frozen golden vectors as the Rust crate.

The binary is entry-point glue over the library. `tl-tok encode --tokenizer <file> --in <texts>` loads with `ByteBpe::from_hf_json`, encodes every text with `encode_batch(texts, 0)`, and prints one JSON line (`ids`, `texts`, `tokens`, `bytes_per_token`). `tl-tok bench` times `encode_batch` on one thread and on `--threads T` (best of `--repeat R`) and prints `tokens_per_s`, `batch_tokens_per_s`, and `batch_speedup`. It is the `tl-tok` role of `cli-roles.md`, a separate Rust process from the Python CLI's `tok train` and `tok encode`.

The model property tests, [`l1_5_props.rs`](../../../../course/solutions/learner-tests/L1.5/rust/crates/tl-tok/tests/l1_5_props.rs), state the contract as proptest properties ([Property-based tests](../../software-engineering/03-testing/02-property-based-tests.md) in *Software Engineering* teaches the tool): `decode(encode(s)) == s` for any `String` under both files; the heap equals a naive rescan written inside the test; pre-token pieces concatenate to the input; streamed chunks equal `decode`; `encode_batch` equals serial `encode` for any thread count; and the byte tokenizer round-trips any bytes. A few pinned examples (known GPT-2 and SmolLM2 ids, `IT'S`, a normalizer that must be refused, `SpecialSet::none`) sit beside them, and the runner uses a fixed seed with no failure persistence so a run is reproducible.

## Building it step by step

Seven observations drive the design.

1. **The ids are fixed by someone else.** GPT-2's and SmolLM2's ids come from their files and the Hugging Face algorithm; the Rust code has no freedom in what it outputs, only in how fast it gets there. So every design choice below must preserve "the same ids as the definition", and any speed-up must be a different route to the same answer.
2. **Work on bytes, not strings.** The file's token strings are written in the byte alphabet $\beta$, but the algorithm is defined on bytes. Since $\beta$ is a bijection, converting once at load with $\beta^{-1}$ loses nothing and removes the alphabet from every hot loop. The vocabulary then maps `Vec<u8> -> u32`, and the 256 single-byte ids can live in a plain array.
3. **Merges never cross pre-token boundaries,** so the pre-tokenizer must reproduce the regex exactly, including its character classes. The classes are general categories as Python's `unicodedata` sees them, so the only faithful source is a table generated from that Python; Rust's `is_alphabetic` is a different property and gives different runs.
4. **The merge rule is a priority order.** "Lowest rank, then leftmost" is the lexicographic order on the tuple (rank, left position). Any structure that pops the minimum tuple applies exactly the definition's merge, so a min-heap of (rank, position) replaces the rescan.
5. **A merge changes three pairs.** Merging $s_i$ with its successor $s_j$ destroys $(s_{prev}, s_i)$ and $(s_j, s_{next})$ and creates $(s_{prev}, s_i')$ and $(s_i', s_{next})$. The destroyed pairs are somewhere inside the heap, not at the top, so the heap must support removal by handle; otherwise a stale pair pops later and merges a symbol that no longer exists. Neighbors change too, so the symbols form a doubly linked list over an array.
6. **Independent texts parallelize trivially.** Encoding one text never reads another, so contiguous chunks on separate threads, joined in order, give the serial answer for any thread count. Words repeat, so a per-thread cache of piece to ids pays for itself; per thread, because a shared one would need a lock on the hot path.
7. **A token is a run of bytes, not of characters.** A streaming server must emit only complete UTF-8, hold back an incomplete tail (at most 3 bytes, since a character is at most 4), and turn truly invalid bytes into U+FFFD so the stream always makes progress. If it follows the same replacement rule as `decode`, the concatenated stream equals the one-shot decode.

The code follows the order the data flows: bytes, pre-tokens, merges, threads, streaming.

### The byte map

The 188 printable bytes keep their value; the other 68 are numbered upward from U+0100 in byte order. The counter `n` is the whole trick: the order of the non-printable bytes is the order they are met.

```rust
pub fn bytes_to_unicode() -> [char; 256] {
    // 0xAD (soft hyphen) is excluded: it is the gap between the two upper ranges.
    let printable = |b: u32| (0x21..=0x7E).contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b);
    let mut out = ['\0'; 256];
    let mut n = 0u32;
    for b in 0u32..256 {
        out[b as usize] = if printable(b) {
            char::from_u32(b).unwrap()
        } else {
            n += 1;                       // the n-th non-printable byte
            char::from_u32(255 + n).unwrap() // U+0100, U+0101, ... ; 0x20 lands on U+0120 'Ġ'
        };
    }
    out
}
```

`unicode_to_bytes` inverts it into a 324-entry table (the alphabet ends at U+0143), and `token_str_to_bytes` maps each character of a token string through it, returning `None` if any character is outside the alphabet; the loader turns that `None` into a `Format` error.

### Character classes by bisection

`LETTER_RANGES` and `NUMBER_RANGES` are sorted, disjoint, inclusive ranges, so membership is a binary search: $O(\log R)$ with $R = 659$ letter ranges, no allocation, and no dependency on Rust's own Unicode tables.

```rust
pub fn in_ranges(ranges: &[(u32, u32)], c: char) -> bool {
    let x = c as u32;
    let (mut lo, mut hi) = (0usize, ranges.len());
    while lo < hi {                 // invariant: the answer, if any, is in ranges[lo..hi]
        let mid = (lo + hi) / 2;
        let (a, b) = ranges[mid];
        if x < a { hi = mid; } else if x > b { lo = mid + 1; } else { return true; }
    }
    false
}

// White space wins (no code point is both), then letter, then number.
pub fn class(c: char) -> Class {
    if is_space(c) { Class::Space } else if is_letter(c) { Class::Letter }
    else if is_number(c) { Class::Number } else { Class::Other }
}
```

### The GPT-2 split as a loop

`next_len` tries the regex alternatives in the order the regex engine does and returns the byte length of the next pre-token. Pieces are slices of the input, so splitting allocates nothing per pre-token.

```rust
fn next_len(s: &str) -> usize {
    let c0 = s.chars().next().unwrap();
    // 1. contractions, lowercase only: "'S" falls through and splits into "'" and "S".
    if c0 == '\'' {
        if let Some(suf) = CONTRACTIONS.iter().find(|suf| s[1..].starts_with(**suf)) {
            return 1 + suf.len();
        }
    }
    // 2 to 4. one optional U+0020 (only the space, not any \s), then a run of one class.
    let (j, cj) = match s[c0.len_utf8()..].chars().next() {
        Some(c1) if c0 == ' ' => (1, c1),
        _ => (0, c0),
    };
    let k = class(cj);
    if k != Class::Space {
        let mut end = j + cj.len_utf8();
        for c in s[end..].chars() {
            if class(c) != k { break; }
            end += c.len_utf8();
        }
        return end;
    }
    // 5. \s+(?!\S): a run of 2+ spaces followed by a non-space gives back its
    //    last character, which becomes the optional space of the next piece.
    // 6. \s+: otherwise the whole white-space run.
    let (mut end, mut last, mut count) = (0, 0, 0);
    for (i, c) in s.char_indices() {
        if !is_space(c) { break; }
        last = i;
        end = i + c.len_utf8();
        count += 1;
    }
    if end < s.len() && count >= 2 {
        end = last;
    }
    end
}
```

The `count >= 2` guard is the regex's behavior when a single space precedes a non-space: alternative 5 cannot match (it would be empty), so alternative 6 takes the one space. In practice a single U+0020 before a letter never reaches this branch (step 2 absorbs it), but a single tab or newline before a letter does, and stays its own piece. This is why `a  b` becomes `a`, ` `, ` b`.

### Composing pre-tokenizer steps

SmolLM2 runs `Digits { individual: true }` and then `ByteLevel`; each step runs over every piece the previous one produced, and empty pieces are dropped at the end.

```rust
pub fn split<'a>(&self, text: &'a str) -> Vec<&'a str> {
    let mut pieces = vec![text];
    for step in &self.steps {
        let mut next = Vec::with_capacity(pieces.len() * 2);
        for p in pieces {
            match *step {
                Step::Digits { individual } => next.extend(DigitSplit::new(p, individual)),
                Step::ByteLevel => next.extend(Gpt2Split::new(p)),
            }
        }
        pieces = next;
    }
    pieces.retain(|p| !p.is_empty());
    pieces
}
```

### Building the tables

`ByteBpe::new` turns the loaded vocabulary and merge list into the three lookups the hot loop needs. The merge table is keyed by the pair of ids, not of strings, so the loop never touches bytes; `or_insert` keeps the first (lowest) rank of a repeated merge, as Hugging Face does.

```rust
for (rank, (l, r)) in merges.into_iter().enumerate() {
    let lid = map.get(&l).copied();
    let rid = map.get(&r).copied();
    let mut joined = l.clone();
    joined.extend_from_slice(&r);
    let id = map.get(&joined).copied();      // the merged token must exist too
    match (lid, rid, id) {
        (Some(a), Some(b), Some(id)) => {
            table.entry((a, b)).or_insert(Merge { rank: rank as u32, id });
        }
        _ => return Err(LoadError::Format(/* names the merge and its rank */)),
    }
}
// The 256 single-byte ids, so encode_piece maps a byte with one array index.
let mut byte_ids = [None; 256];
for (b, slot) in byte_ids.iter_mut().enumerate() {
    *slot = map.get(&[b as u8][..]).copied();
}
```

The constructor also refuses two tokens with one id and a gap in the id range, so `token_bytes(id)` can be a plain `Vec` index and `vocab_size` is `tokens.len()`.

### The merge loop

This is the heart of the crate. `pair[i]` holds the heap handle of the pair whose left symbol is at position `i`, which is how a merge finds the two pairs it destroys.

```rust
pub fn encode_piece_with(&self, piece: &[u8], out: &mut Vec<u32>, scratch: &mut MergeScratch) {
    let MergeScratch { syms, pair, heap } = scratch;   // buffers reused across pieces
    syms.clear();
    for &b in piece {
        if let Some(id) = self.byte_ids[b as usize] {  // a byte with no token is dropped, as HF does
            let i = syms.len();
            syms.push(Sym { id, prev: if i == 0 { NONE } else { i - 1 }, next: i + 1 });
        }
    }
    let n = syms.len();
    if n == 0 { return; }
    syms[n - 1].next = NONE;
    if n > 1 {
        heap.clear();
        pair.clear();
        pair.resize(n, None);
        for i in 0..n - 1 {
            if let Some(m) = self.merge(syms[i].id, syms[i + 1].id) {
                pair[i] = Some(heap.push((m.rank, i)));   // key (rank, left position)
            }
        }
        while let Some((_, i)) = heap.pop() {             // lowest rank, then leftmost
            pair[i] = None;
            let j = syms[i].next;
            let m = self.merge(syms[i].id, syms[j].id).expect("a live pair always merges");
            // The pairs (prev, i) and (j, next) disappear with this merge.
            let p = syms[i].prev;
            if p != NONE {
                if let Some(h) = pair[p].take() { heap.remove(h); }
            }
            if let Some(h) = pair[j].take() { heap.remove(h); }
            // i absorbs j: j drops out of the linked list.
            let nx = syms[j].next;
            syms[i].id = m.id;
            syms[i].next = nx;
            if nx != NONE { syms[nx].prev = i; }
            // The pairs (prev, i) and (i, next) are new.
            if p != NONE {
                if let Some(mm) = self.merge(syms[p].id, syms[i].id) {
                    pair[p] = Some(heap.push((mm.rank, p)));
                }
            }
            if nx != NONE {
                if let Some(mm) = self.merge(syms[i].id, syms[nx].id) {
                    pair[i] = Some(heap.push((mm.rank, i)));
                }
            }
        }
    }
    let mut i = 0;                                         // position 0 is never absorbed
    while i != NONE { out.push(syms[i].id); i = syms[i].next; }
}
```

Three invariants make it correct. First, every pair in the heap is live: its left symbol is at `i` and its right symbol is `syms[i].next`, because the only operation that changes adjacency (a merge) removes the old pairs around it and pushes the new ones. That is why the `expect` can never fire. Second, a symbol keeps the position of its leftmost byte, so "position" in the key still means "leftmost" after merges, and the tie-break stays the definition's. Third, position 0 is always a left symbol and never absorbed, so walking `next` from 0 visits the final tokens in order. Trace `aaaa` from the hand example through it: popping $(0,0)$ removes `pair[1]` (the pair whose left symbol was absorbed), which is the lazy removal the definition needs.

### Added tokens first

`next_added` finds the allowed added token that occurs leftmost, preferring the longer one on a tie, and `encode_with_special` alternates "ordinary text up to it" and "its id".

```rust
fn encode_with_special(&self, text: &str, allowed: &SpecialSet) -> Vec<u32> {
    let mut out = Vec::with_capacity(text.len() / 3 + 1);   // ~3 bytes per token, a guess
    let mut scratch = MergeScratch::default();               // one set of buffers per text
    let mut rest = text;
    while let Some((start, len, id)) = self.next_added(rest, allowed) {
        for piece in self.pretok.split(&rest[..start]) {
            self.encode_piece_with(piece.as_bytes(), &mut out, &mut scratch);
        }
        out.push(id);                                        // never pre-tokenized or merged
        rest = &rest[start + len..];
    }
    for piece in self.pretok.split(rest) {
        self.encode_piece_with(piece.as_bytes(), &mut out, &mut scratch);
    }
    out
}
```

`encode` is `encode_with_special(text, &SpecialSet::all())`. With `SpecialSet::none()` the `allows` check skips every added token, so `<|endoftext|>` in user text becomes ordinary pieces and can never end a document.

### Threads in order

`encode_batch` resolves `0` to the hardware concurrency, clamps the thread count to the number of texts, and splits into contiguous chunks. `std::thread::scope` lets each worker borrow `self` and its slice without `Arc` or copies, and `flat_map` over the workers in spawn order rebuilds the input order.

```rust
pub fn encode_batch(&self, texts: &[&str], threads: usize) -> Vec<Vec<u32>> {
    let threads = if threads == 0 {
        std::thread::available_parallelism().map_or(1, |n| n.get())
    } else { threads };
    let threads = threads.min(texts.len()).max(1);
    let run = |part: &[&str]| -> Vec<Vec<u32>> {
        let mut cache = RobinHoodMap::new();               // one cache per worker: no lock
        part.iter().map(|t| self.encode_cached(t, &mut cache)).collect()
    };
    if threads == 1 { return run(texts); }
    let chunk = texts.len().div_ceil(threads);
    std::thread::scope(|s| {
        let workers: Vec<_> = texts.chunks(chunk).map(|part| s.spawn(move || run(part))).collect();
        // Join in spawn order: the output order is the input order for any T.
        workers.into_iter().flat_map(|w| w.join().expect("an encode worker panicked")).collect()
    })
}
```

`encode_cached` is `encode` with a lookup before each `encode_piece_with`; when the cache reaches `CACHE_CAP` ($2^{16}$) pieces it is cleared outright, which bounds memory with no eviction bookkeeping.

### The streaming decoder

`std::str::from_utf8` already classifies a failure: `valid_up_to()` is the length of the good prefix, and `error_len()` is `Some(k)` for an invalid subpart of `k` bytes or `None` for an incomplete tail. The loop just acts on those three outcomes.

```rust
pub fn push(&mut self, id: u32) -> Result<Option<String>, DecodeError> {
    let bytes = self.tok.token_bytes(id)
        .ok_or(DecodeError::UnknownId { id, vocab_size: self.tok.vocab_size() })?; // decoder unchanged on error
    self.pending.extend_from_slice(bytes);
    let mut out = String::new();
    let mut start = 0;
    loop {
        match std::str::from_utf8(&self.pending[start..]) {
            Ok(s) => { out.push_str(s); start = self.pending.len(); break; }
            Err(e) => {
                let good = e.valid_up_to();
                out.push_str(std::str::from_utf8(&self.pending[start..start + good]).unwrap());
                start += good;
                match e.error_len() {
                    Some(bad) => { out.push('\u{FFFD}'); start += bad; } // invalid: replace, move on
                    None => break,                                       // incomplete: keep it pending
                }
            }
        }
    }
    self.pending.drain(..start);            // drop what was emitted, or the next chunk repeats it
    Ok(if out.is_empty() { None } else { Some(out) })
}

pub fn finish(&mut self) -> String {
    let out = String::from_utf8_lossy(&self.pending).into_owned(); // same rule as decode
    self.pending.clear();
    out
}
```

Because the replacement rule (one U+FFFD per maximal ill-formed subpart) is the one `String::from_utf8_lossy` uses in `decode`, chunks plus `finish` equal `decode(ids)` exactly. On the emoji, pushing 47249 leaves `f0 9f 98` (incomplete, `None`), and pushing 222 makes `from_utf8` succeed and emits `😀`.

### The byte tokenizer

`token_bytes` must return a borrowed slice, so the solution keeps a `static ALL_BYTES: [u8; 256]` and lends `&ALL_BYTES[i..i + 1]` for `i < 256`, `None` otherwise. `decode` and `decode_bytes` are trait default methods, so the byte tokenizer and `ByteBpe` share one decoding rule.

The full solution is in [`bpe.rs`](../../../../course/solutions/rust/crates/tl-tok/src/bpe.rs), [`pretok.rs`](../../../../course/solutions/rust/crates/tl-tok/src/pretok.rs), [`stream.rs`](../../../../course/solutions/rust/crates/tl-tok/src/stream.rs), [`lib.rs`](../../../../course/solutions/rust/crates/tl-tok/src/lib.rs), and [`hfjson.rs`](../../../../course/solutions/rust/crates/tl-tok/src/hfjson.rs). In the loader, each accepted field is checked with `expect_in`, and anything else becomes `LoadError::Unsupported` naming its JSON path.

### Why this design

Start with complexity. Per pre-token of $n$ bytes, building the heap is $O(n \log n)$ pushes, each merge does at most two removals and two pushes at $O(\log n)$, and there are at most $n - 1$ merges, so the total is $O(n \log n)$ against the rescan's $O(n^2)$. Space is $O(n)$ for `syms`, `pair`, and the heap, reused across pieces through `MergeScratch`. Class lookups are $O(\log R)$. The piece cache makes repeated words $O(1)$ amortized after the first hit.

Why not the rescan? It is correct and simpler, and for GPT-2's typical 3 to 6 byte pieces it is competitive. But it degrades quadratically on long runs (a line of 10,000 `=` signs is one pre-token), which a server must survive. The solution keeps the rescan as the reference inside the differential tests instead.

Why not a regex crate? GPT-2's pattern uses a lookahead (`(?!\S)`), which Rust's `regex` crate does not support. `fancy-regex` does, but its classes are Rust's Unicode tables, not the Python's, so the ids would drift on characters assigned between versions. The hand loop has no backtracking and allocates nothing. And why tables generated from Python? Because parity with the Python BPE is the specification. Generating from the same `unicodedata` makes Rust and Python classify every code point identically by construction; any other source is a second opinion.

Per-thread caches and contiguous chunks are also deliberate. A shared cache needs a lock on every piece; per-thread caches trade some duplicated entries for zero contention. Contiguous chunks keep the join trivial and deterministic; a work-stealing pool would balance better on skewed inputs but would need an index to restore order. Finally, the loader is strict because accepting a normalizer or `add_prefix_space: true` and ignoring it would load and then encode differently from Hugging Face, silently. Refusing with the field's JSON path turns a silent wrong answer into a loud error.

Three tradeoffs are accepted: the cache is cleared, not evicted, so a pathological input can thrash it; the subset excludes many real tokenizer files; and the tables pin one Unicode version.

## What can go wrong

The checks a correct solution must pass fall into a few groups. The hand example from the heap: `hugs pug` is `[257, 82, 220, 258]`, and `aaaa`, `aaaaa`, `aaaaaa` behave correctly with lazy removals. A differential check: on 3,000 random pieces, heap ids equal the $O(n^2)$ rescan. The byte tokenizer is the identity: 256 ids, id = byte, `token_bytes(256)` is `None`, and `decode([0xFF])` is U+FFFD. The byte map matches GPT-2 at 0x20, 0x21, 0x7E, 0x00, 0x7F, 0xAD, 0xAE, and 0xFF, is a bijection, and vocabulary keys are bytes. The GPT-2 split handles contractions in both cases, space runs, `\s+(?!\S)`, digits, marks, and Devanagari, and `Digits` works individually and not, with `12345` as five SmolLM2 tokens.

The golden checks run 10,000 strings against Hugging Face's and tiktoken's GPT-2 ids, and the same strings under SmolLM2's file with Digits and 17 added tokens. `decode_bytes(encode(x))` must be $x$'s UTF-8 for every golden string under both files. SmolLM2's chat markers must match first, with `SpecialSet::none` and `only` respected. The stream decoder must hold back the emoji split over two GPT-2 tokens, handle `C3 A9 FF E2 82` byte by byte, and flush on `finish`; on 2,000 golden strings, chunks plus `finish` must equal `decode`, with at most 3 pending bytes. `encode_batch` on 997 texts with 0, 1, 2, 3, 4, and 8 threads must equal serial `encode`, so the output of [Tokenize and pack](../../data-engineering/05-corpus-pipeline/07-tokenize-and-pack.md) in *Data Engineering* does not depend on core count. The loader must reject a normalizer, `WordPiece`, `add_prefix_space`, `byte_fallback`, bad JSON, and a missing file; `vocab.json` + `merges.txt` written from `tokenizer.json` must encode 1,000 strings identically; and GPT-2's first merge `(Ġ, t)` must be rank 0, id 256, with sizes 50,257 and 50,000.

> **Warning:** Not removing the pairs a merge destroys is the bug the lazy heap exists to prevent. A stale pair pops later and merges a symbol that is gone, giving wrong ids or a panic. `pair[p].take()` and `pair[j].take()` remove both by handle before the new pairs are pushed.

The full list of pitfalls:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| Breaking rank ties toward the rightmost pair | `aaa` encodes as `a aa`, not `aa a`; long runs of one letter disagree with Python | the heap key is the tuple (rank, left position), compared by `by_rank`; see the merge loop |
| An off-by-one in the byte tokenizer's id range | id 256 is accepted and indexes past the table | `token_bytes` checks `i < 256` and returns `None` otherwise |
| Treating 0xAD as printable in the byte map | two bytes share a character; a whole class of tokens loads to the wrong bytes | `printable` uses `0xA1..=0xAC` and `0xAE..=0xFF`, skipping 0xAD |
| Forgetting the `\s+(?!\S)` back-off | `a  b` encodes as `a`, `  `, `b`, not `a`, ` `, ` b` | `next_len` sets `end = last` when a non-space follows a run of 2 or more |
| Matching contractions in any case | `IT'S` splits `'S` off; GPT-2 is case-sensitive here | `CONTRACTIONS` holds lowercase suffixes and `starts_with` is case-sensitive |
| Ignoring `individual_digits` | the pre-tokens differ from SmolLM2's; the ids agree only by luck of its vocabulary | `DigitSplit` stops after one digit when `individual` is set |
| Using `char::is_alphabetic` for `\p{L}` | marks join letter runs: Devanagari, Thai, and combining accents tokenize differently | `is_letter` bisects `LETTER_RANGES`, generated from Python's categories |
| Matching added tokens whatever the `SpecialSet` says | user text containing `<\|endoftext\|>` ends a document | `next_added` skips any token `allowed.allows` rejects |
| Flushing an incomplete character as U+FFFD | every emoji split over two tokens streams as two replacement characters | `push` breaks and keeps the tail pending when `error_len()` is `None` |
| Not draining the emitted bytes | every chunk repeats the text before it | `self.pending.drain(..start)` after every push |
| Joining thread results out of order | `encode_batch` reorders documents when $T > 1$ | contiguous `chunks`, joined in spawn order |
| Accepting a normalizer and ignoring it | a file with NFC or lowercasing loads and encodes differently from Hugging Face | the loader returns `LoadError::Unsupported` for any non-null `normalizer` |
| Decoding strictly instead of with replacement | a partial character decodes to an empty string, and text is lost | `decode` uses `String::from_utf8_lossy` |
| Generating the category tables with a different Python than the Python BPE | a handful of characters assigned in a newer Unicode version split differently | the tables record their source (Unicode 15.0.0, Python 3.12); regenerate them whenever the Python changes |

## Where this leads

The serving chapters build directly on this crate. [The model runner](../23-model-serving/02-model-runner-and-sampler.md) encodes prompts with `ByteBpe` on the engine's request path and keeps serving `ByteTokenizer` for the tracer model. [The OpenAI-compatible server](../23-model-serving/06-openai-server-on-tokio.md) streams each token through `StreamDecoder`. [Tool calls and constrained decoding](../23-model-serving/10-tool-calls-and-constrained-decoding.md) reads token bytes through the `Tokenizer` trait. And [the TinyStories capstone](../capstones/01-tinystories.md) serves the tokenizer it trains through this loader.

Each piece has a production counterpart:

- `ByteBpe` merges correspond to [Hugging Face tokenizers](https://github.com/huggingface/tokenizers), which use the same lazy merge queue and add dropout, `byte_fallback`, normalizers, and offsets for every token (`tokenizers/src/models/bpe/word.rs`, `model.rs`).
- `Gpt2Split` by hand corresponds to [tiktoken](https://github.com/openai/tiktoken), which compiles the regex with `fancy-regex` and does a rank-based merge over byte pairs with no merges table (`src/lib.rs`, `byte_pair_merge`).
- The piece cache corresponds to tokenizers' `Cache`, one bounded cache shared by every thread behind a read-write lock (`tokenizers/src/utils/cache.rs`).
- `StreamDecoder` corresponds to [vLLM's detokenizer](https://github.com/vllm-project/vllm), incremental detokenization with prefix offsets for any tokenizer (`vllm/transformers_utils/detokenizer_utils.py`).

### Further reading

- [Unicode Standard Annex #44](https://www.unicode.org/reports/tr44/), on general categories (free).
- The Hugging Face [tokenizers](https://github.com/huggingface/tokenizers) source.

## Key terms

- **`tl-tok`**: the Rust tokenizer crate: a library of five modules plus a binary with `encode` and `bench` verbs.
- **Golden fixture**: frozen JSON vectors of Hugging Face ids that both the Python and Rust implementations are checked against.
- **Lazy heap**: a priority queue supporting removal by handle, used here to drop pairs a merge destroys.
- **General category**: the Unicode classification (Lu, Ll, Nd, Mc, ...) that defines `\p{L}` and `\p{N}`; distinct from Rust's Alphabetic property.
- **Streaming decoder**: a decoder that emits only complete UTF-8 after each token and holds back up to 3 bytes of an unfinished character.
- **`SpecialSet`**: the set of added tokens allowed to match in a given text, so servers can refuse forged control tokens.

## Summary

- The shared golden JSON fixture pins Hugging Face ids for both vocabularies, so the Python specification and the Rust implementation are held to the same oracle.
- BPE encoding with a lazy heap applies exactly the pair the definition names (lowest rank, then leftmost) in $O(n \log n)$ instead of $O(n^2)$; a merge must remove the two pairs it destroys.
- The GPT-2 pre-tokenizer is a regular expression you can write as a loop over character classes, if the classes are Unicode general categories generated from Python, not Rust's `is_alphabetic`.
- A streaming decoder holds back the bytes of an unfinished character and emits them when it completes; the concatenated chunks equal `decode(ids)`.
- Contiguous chunks joined in order and per-thread caches make `encode_batch` fast and independent of the thread count.

## Review questions

1. Why must the heap key be (rank, position) and not just rank?
   <details><summary>Answer</summary>The definition breaks rank ties toward the leftmost pair. With rank alone the heap would pop equal-rank pairs in arbitrary order, so `aaa` with merge `a a` could become `a aa` instead of `aa a`. Tuple order makes "lowest rank, then leftmost" the heap's own order.</details>
2. What goes wrong if a merge does not remove `pair[p]` and `pair[j]` from the heap?
   <details><summary>Answer</summary>Those pairs describe adjacencies that no longer exist. Later one pops: `pair[j]`'s left symbol `j` has been absorbed, so `syms[j].next` points into a stale chain and the code merges a symbol that is gone (wrong ids), or `merge` returns `None` and the `expect` panics. Lazy deletion by handle is what lets the loop remove them without searching the heap.</details>
3. Why does `Gpt2Split` back off one space only when the run has at least two characters and a non-space follows?
   <details><summary>Answer</summary>That is `\s+(?!\S)`: the run may not be followed by a non-space, so the regex engine shortens it by one character, which then joins the next pre-token through the ` ?` prefix. A single white-space character cannot be shortened to an empty match, so alternative 6 (`\s+`) takes it whole. At the end of the text no non-space follows, so nothing is given back.</details>
4. Why is it safe for `StreamDecoder` to hold back at most 3 bytes, and why does the concatenation equal `decode`?
   <details><summary>Answer</summary>A UTF-8 character is at most 4 bytes, so an incomplete tail that could still complete is at most 3 bytes; anything longer is either complete or invalid, and invalid bytes are replaced and dropped immediately. The decoder uses the same replacement rule as `from_utf8_lossy` (one U+FFFD per maximal ill-formed subpart), and an incomplete subpart only becomes U+FFFD once it is known to be invalid or at `finish`, which is exactly what one-shot decoding does at the same bytes.</details>
5. Why does `encode_batch` give identical output for 1 and 8 threads?
   <details><summary>Answer</summary>Encoding one text reads only that text and the immutable tokenizer; the per-thread cache only stores what `encode_piece_with` would compute anyway. Chunks are contiguous and joined in spawn order, so concatenating the workers' outputs reproduces the input order.</details>
