# Tokenizers
<!-- chapter: content/courses/build-an-llm/p01-tokenizers/05-rust-fast-bpe.md -->

A tokenizer turns text into a list of integer ids and back. A model directory names its tokenizer in [`config.json`](config.schema.json) with `tl_tokenizer`:

| `tl_tokenizer` | File in the model directory | Introduced by |
|---|---|---|
| `bytes` | none | the tracer (L0.0, L10.0): the identity byte tokenizer below |
| `file` | `tokenizer.json` (HF subset), or `tinyllm_char.json` for the char tokenizer | L1.1 to L1.6 |

## The byte tokenizer (`bytes`)

The vocabulary is the 256 byte values. The id of a byte is its value, so `vocab_size` is exactly 256 and there are no special tokens: no BOS, no EOS, no padding.

- **Encode.** `encode(text)` is the list of the UTF-8 bytes of `text`, each as an integer in `0..=255`.
- **Decode.** `decode(ids)` is those bytes decoded as UTF-8 **with replacement**: each maximal ill-formed subsequence becomes one U+FFFD (the Unicode "maximal subpart" practice). Python `bytes(ids).decode("utf-8", "replace")` and Rust `String::from_utf8_lossy(&bytes)` both do exactly this.
- **Decode rejects** an id outside `0..=255` (Python `ValueError`, Rust an error, HTTP 400 with `param` naming the field).

Worked example by hand:

| Text | Bytes (hex) | Ids |
|---|---|---|
| `hi` | `68 69` | `[104, 105]` |
| `héllo` | `68 c3 a9 6c 6c 6f` | `[104, 195, 169, 108, 108, 111]` |

`é` is U+00E9, which UTF-8 encodes in two bytes, `c3 a9`. So one character can be two tokens, and a model that samples bytes can emit sequences that are not valid UTF-8:

| Ids | Decoded | Why |
|---|---|---|
| `[255]` | `"�"` | `ff` never appears in UTF-8 |
| `[226, 130]` | `"�"` | `e2 82` is the start of a three-byte sequence that never finishes: one maximal subpart, one U+FFFD |
| `[195, 169]` | `"é"` | complete |

### Incremental decoding (streaming)

A server that streams one chunk per token cannot decode each byte on its own. It keeps a buffer of pending bytes and, for each new token:

1. Append the token's byte to the buffer.
2. Repeat: decode the longest valid UTF-8 prefix of the buffer and emit it. If the next bytes are an invalid sequence, emit one U+FFFD, drop that maximal subpart, and continue. If they are the start of a sequence that could still complete, stop and keep them pending.
3. The chunk's text is everything emitted in step 2 (possibly `""`).

At the end of generation, decode whatever is still pending with replacement and append it to the last chunk. The concatenated chunk texts then equal `decode(all generated ids)`, which is what conformance case `v0.stream.equals_nonstream` checks. Python's `codecs.getincrementaldecoder("utf-8")("replace")` implements exactly these steps; in Rust, `std::str::from_utf8` reports `valid_up_to()` and `error_len()` (`None` means "incomplete, keep pending").

Streaming the ids `[195, 169, 255]`: chunk texts `""`, `"é"`, `"�"`.

## The char tokenizer (L1.1)

Saved as `tinyllm_char.json` (schema [`tokenizer-char.schema.json`](tokenizer-char.schema.json)):

```json
{"type": "char", "vocab": ["<unk>", "\n", " ", "a", "b"], "specials": {"unk": 0}}
```

- `vocab[i]` is the string of id `i`: one Unicode code point, or a special token's text. Entries are unique.
- `specials` maps the names `unk`, `bos`, `eos`, `pad` to ids; only `unk` is required.
- **Encode** maps each code point of the text to the id whose `vocab` entry equals it, and an unknown code point to `specials.unk`. Special-token texts in the input are not matched (they are ordinary characters).
- **Decode** concatenates `vocab[id]`; with `skip_special` the ids in `specials` are dropped. An id outside `0..len(vocab)` is an error.

## `tokenizer.json`, the HF subset (L1.2 to L1.6)

A subset of the Hugging Face `tokenizers` JSON format, schema [`tokenizer-json.schema.json`](tokenizer-json.schema.json). GPT-2's, SmolLM2's, and BERT's files load **unchanged**; a loader rejects anything outside the subset with an error naming the field (Python `ValueError`, Rust an error). Encoding runs the stages in this order: added tokens are split out first (they are never normalized or merged), then `normalizer`, `pre_tokenizer`, `model`, and `post_processor` when the caller asks for special tokens (`add_special`).

| Key | Accepted values |
|---|---|
| `version` | `"1.0"` |
| `truncation`, `padding` | anything; ignored (the engine and trainers never truncate or pad through the tokenizer) |
| `added_tokens` | list of `{id, content, single_word, lstrip, rstrip, normalized, special}`; `single_word`, `lstrip`, `rstrip` must be `false`, and `normalized` may be `true` only when `normalizer` is `null` (where it has no effect) |
| `normalizer` | `null`, `{"type": "NFC"}`, `{"type": "NFKC"}`, `{"type": "Lowercase"}`, `BertNormalizer` (`clean_text`, `handle_chinese_chars`, `strip_accents`, `lowercase`), or a `Sequence` of these. The course loaders implement the per-family subset: BPE and Unigram files need `null`, WordPiece needs `BertNormalizer` |
| `pre_tokenizer` | `ByteLevel` (`add_prefix_space`, `use_regex: true`: the GPT-2 split regex), `Digits` (`individual_digits`), `BertPreTokenizer`, `Metaspace` (`replacement`, `prepend_scheme`), or a `Sequence` of these |
| `model` | `BPE`, `WordPiece`, or `Unigram`, below |
| `post_processor` | `null`, `ByteLevel` (no effect on ids), or `TemplateProcessing` (`single`, `pair`, `special_tokens`) |
| `decoder` | `null`, `ByteLevel`, `WordPiece` (`prefix`, `cleanup`), or `Metaspace` |

**BPE** (`model.type = "BPE"` or absent with `merges` present, GPT-2 and SmolLM2 byte-level BPE, L1.2 and L1.5): `vocab` maps token strings (in the ByteLevel alphabet, where each byte is one printable code point) to ids; `merges` lists merges by rank, each `"left right"` or `["left", "right"]`; `dropout` and `unk_token` are `null`; `continuing_subword_prefix` and `end_of_word_suffix` are `null` or `""`; `fuse_unk`, `byte_fallback`, `ignore_merges` are `false`. Encoding a pre-token repeatedly applies the lowest-ranked merge present among adjacent pairs (leftmost on ties) until none applies.

**WordPiece** (`model.type = "WordPiece"` or absent with `max_input_chars_per_word` present, BERT, L1.3): `vocab`, `unk_token`, `continuing_subword_prefix` (`"##"`), `max_input_chars_per_word`. Encoding is greedy longest-match-first within each pre-token; a pre-token with no full segmentation, or longer than `max_input_chars_per_word`, becomes `unk_token`.

**Unigram** (`model.type = "Unigram"`, L1.4): `vocab` is a list of `[piece, log_prob]`, `unk_id` an index into it, `byte_fallback` `false`. Encoding is the Viterbi segmentation that maximizes the sum of piece log-probabilities (the M06.2 trie lattice); ties go to the path whose last piece starts earliest (sentencepiece and Hugging Face keep the first candidate that reaches the best score, scanning start positions in order). For `▁aaa` with `▁` = `a` = -1 and `aa` = -2, the result is `▁ a aa`.

**Special tokens and chat.** BOS, EOS, and the chat template are not in `tokenizer.json`: they come from the model directory's `generation_config.json` and `tokenizer_config.json` ([checkpoint.md](checkpoint.md)).
