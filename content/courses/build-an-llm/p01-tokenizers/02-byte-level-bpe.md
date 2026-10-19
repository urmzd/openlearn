# Byte-level BPE (GPT-2 compatible): pre-tokenizer, trainer, HF loader

## Overview

| | |
|---|---|
| **Module** | `L1.2` · build · Python · Pass 3 · 6 to 8 h |
| **Solution** | [`pretok.py`](../../../../course/solutions/python/tinyllm/tok/pretok.py): `pretokenize_gpt2`, `split_digits`, `is_space`, `is_letter`, `is_number` · [`bpe.py`](../../../../course/solutions/python/tinyllm/tok/bpe.py): `BPETokenizer` (`train`, `from_gpt2`, `from_hf_json`, `encode`, `encode_batch`, `decode`, `save`, `load`) · the `tok train` and `tok encode` verbs in [`cli_tok.py`](../../../../course/solutions/entry/python/tinyllm/cli_tok.py) · model tests: [`test_bpe.py`](../../../../course/solutions/learner-tests/L1.2/python/tests/l1-2-bpe/test_bpe.py) |
| **Contract** | [`course/contracts/py/tinyllm/tok/bpe.pyi`](../../../../course/contracts/py/tinyllm/tok/bpe.pyi) · [`course/contracts/py/tinyllm/tok/pretok.pyi`](../../../../course/contracts/py/tinyllm/tok/pretok.pyi) · formats: [`tokenizer.md`](../../../../course/contracts/formats/tokenizer.md) (the `tokenizer.json` subset), [`tokenizer-json.schema.json`](../../../../course/contracts/formats/tokenizer-json.schema.json) |
| **Needs** | `L1.1` the protocol and `check_ids` · `M05.2` the byte map · `M06.2` the trie (`insert`, `longest_prefix`) that splits out added tokens |
| **Used by** | `L1.5` ports it to Rust (`tl-tok`) and is proven id for id against this Python · `L1.6` measures GPT-2 and SmolLM2 through `from_hf_json` · `data.07` tokenizes corpus files with the Rust tokenizer process · later: `L7.9` loads SmolLM2, `L8.2` detokenizes, `C1` trains a 4096-token vocabulary; `L14.4`. The finished system trains a vocabulary with `tok train --algo bpe` and encodes with SmolLM2's real file through `tok encode`, matching Hugging Face's ids |
| **Optional depth** | Sennrich, Haddow, and Birch, *Neural Machine Translation of Rare Words with Subword Units* (2016); Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2, 2019), section 2.2 |

## Key Takeaways

- BPE training is a loop: count adjacent pairs inside pre-tokens, weighted by how often each pre-token occurs, merge the most frequent pair, and update only the counts the merge touched; ties go to the pair with the smaller ids.
- Encoding replays the learned merges by rank, lowest rank first and leftmost on a tie, which is not the same as merging left to right.
- Byte-level means the alphabet is the 256 bytes, so every text encodes with no unknown token, and decoding joins token bytes before UTF-8 decoding.
- The GPT-2 pre-tokenizer regex can be written by hand over `unicodedata`, and only an exact transcription matches the real ids: Unicode White_Space is not `str.isspace`.
- The loader reads GPT-2's and SmolLM2's real `tokenizer.json` unchanged and reproduces Hugging Face's ids on 300 hard strings, and refuses fields it does not implement.

## How to read this chapter

Sections 1 to 3 build the pre-tokenizer, the trainer, and the rank-ordered encoder from first principles and work each by hand. Section 4 fixes the interface and lists what a correct solution must handle. Section 5 walks through `pretok.py` and `bpe.py` and explains why each choice is the right one. Try to predict each step of the solution before reading it.

The pre-tokenizer comes first in the walkthrough because every id downstream depends on it; then training on the hand example, then `encode` with merges by rank, then the two loaders. The solution's entry point, [`cli_tok.py`](../../../../course/solutions/entry/python/tinyllm/cli_tok.py), exposes the module as two verbs: `tok train --algo bpe --vocab V --in FILE --out DIR [--special S] [--min-freq M]` trains on the lines of `FILE` and saves `DIR/tokenizer.json`, and `tok encode --tokenizer tokenizer.json --in FILE` loads any file in the subset through `from_hf_json` and reports the ids and bytes per token.

---

## 1. Why now

`L1.1` gave every tokenizer one shape, and the only trained one so far is the char tokenizer: one id per code point, so a 10M-parameter model on TinyStories spends most of its context window on spelling, and any character it never saw is `<unk>`. The byte tokenizer of the tracer never fails but costs one token per byte. Byte-level BPE sits between them: it starts from the 256 bytes (so nothing is ever unknown) and learns merges until frequent words are one token. It is also what the real models the system serves use: SmolLM2 (`L7.9`, the agents' model) ships a GPT-2-style byte-level BPE in `tokenizer.json`, and the engine cannot answer a single request until the tokenizer turns text into exactly its ids. This module builds the trainer for the capstone's own vocabulary and the loader for the real ones.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $\beta(b)$ | the byte map of `M05.2`: byte $b$ as one printable code point | `str` of length 1 |
| $w$ | a pre-token: a piece of text no merge may cross | `str` |
| $f(w)$ | how often pre-token $w$ occurs in the training texts | `int` |
| $s = (s_1, \dots, s_n)$ | the current symbols of one pre-token (each a token string) | `list[str]` |
| $(a, b)$ | an adjacent pair of symbols | `tuple[str, str]` |
| $c(a, b)$ | the pair count, $\sum_w f(w) \cdot \#\{i : s_i = a, s_{i+1} = b\}$ | `int` |
| $r(a, b)$ | the rank of merge $(a, b)$: its position in `merges` | `int` |
| $V$ | the target vocabulary size | `int` |
| $m$ | `min_freq`: the smallest count still worth a merge | `int` |

**Pre-tokenize first.** GPT-2 splits text into pre-tokens with one regular expression, and merges never cross a pre-token boundary. Without it, "dog." and "dog!" would learn different merges and a frequent pair could fuse the end of one word with the start of the next. The expression is:

```text
's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

At each position the engine tries the alternatives in order and takes the first that matches:

1. one of the seven lowercase contractions, `'s 't 're 've 'm 'll 'd`;
2. an optional single space U+0020, then a run of letters (`\p{L}`: categories Lu, Ll, Lt, Lm, Lo);
3. an optional single space, then a run of numbers (`\p{N}`: Nd, Nl, No);
4. an optional single space, then a run of anything that is not white space, a letter, or a number;
5. `\s+(?!\S)`: a run of white space not followed by a non-space. When the run is followed by a word, the engine backtracks one character, so the run gives up its last space, which then starts the next pre-token via the optional space of 2 to 4;
6. `\s+`: the white space that 5 rejected (a single white-space character right before a word, such as a tab).

`\s` is the Unicode **White_Space** property: tab, newline, U+000B to U+000D, the space, U+0085, U+00A0, U+1680, U+2000 to U+200A, U+2028, U+2029, U+202F, U+205F, U+3000. Python's `str.isspace()` also accepts U+001C to U+001F, which are not White_Space, so it gives wrong pre-tokens on text with those control characters. The solution writes `pretokenize_gpt2` over `unicodedata.category` and this set, with no regex module; the result always joins back to the text and has no empty piece.

SmolLM2 declares a `Sequence` of two pre-tokenizers: first `Digits` with `individual_digits`, which makes every number code point its own piece, then the GPT-2 split on each piece. That is `split_digits`, and it is why SmolLM2 spells `2024` as four tokens.

**Bytes, not characters.** Each pre-token is turned into its UTF-8 bytes and each byte into $\beta(b)$, so a pre-token becomes an ordinary string over 256 printable symbols: the space is `Ġ`, the newline `Ċ`, and `é` (bytes `c3 a9`) is `Ã©`. The vocabulary starts with these 256 symbols, so every text is encodable, there is no `<unk>`, and a token can be part of a character.

**Training.** Start every pre-token as its sequence of byte symbols. Count each adjacent pair over all pre-tokens, weighting by $f(w)$. Then repeat until the vocabulary has $V$ entries:

1. pick the pair $(a, b)$ with the largest count $c(a, b)$; on a tie, the pair whose ids are smallest, compared as (left id, right id);
2. stop if that count is below $m$;
3. add the token $ab$ with the next id and record the merge $(a, b)$ with the next rank;
4. replace every non-overlapping occurrence of $a, b$ (left to right) in every pre-token that contains it, and update the counts of the neighboring pairs that changed.

Ids are assigned in a fixed order: the specials first (ids 0, 1, ...), then the 256 byte symbols in code point order (so `!` is the first one, and `Ġ` is 220 when there are no specials), then one token per merge. The tie rule and this order are what the Hugging Face `BpeTrainer` does, which is why the solution's merges can be compared with its merges one for one. Recounting every pair after every merge costs a pass over the corpus per merge; the trainer keeps a max-heap of `(count, pair)` with lazy updates (pop a stale entry, push it back with its current count) and a map from each pair to the pre-tokens that contain it, so a merge costs only the pre-tokens it touches. Specials are cut out of the training texts before counting (`<|endoftext|>` between documents must never become the merges `<|`, `end`, ...).

**Encoding.** Split out the added tokens first, by leftmost-longest match through the `M06.2` trie: SmolLM2's `<|im_start|>` is one id, and the regex would otherwise cut it into `<|`, `im`, `_`, `start`, `|>`. Then pre-tokenize the rest, map bytes, and inside each pre-token repeatedly merge the adjacent pair with the **lowest rank** among those present; when the lowest rank occurs twice, merge the leftmost occurrence. Stop when no adjacent pair has a rank, and look each symbol up in the vocabulary. Caching the result per pre-token makes encoding real text fast (words repeat).

**Decoding.** Concatenate the bytes of every token ($\beta^{-1}$ of each symbol, or the UTF-8 of an added token's text) and decode the whole byte string as UTF-8 with replacement, once. Decoding token by token breaks every character whose bytes landed in two tokens.

**Loading real files.** `from_gpt2(vocab.json, merges.txt)` reads the original GPT-2 release (merges.txt has a `#version` first line) and makes `<|endoftext|>` an added special token. `from_hf_json(tokenizer.json)` reads the subset in [`tokenizer.md`](../../../../course/contracts/formats/tokenizer.md): it honors the declared pre-tokenizer sequence and added tokens, and it raises `ValueError` naming the field for anything it does not implement (a normalizer, dropout, an unknown token, byte fallback, a template post-processor). Silently ignoring such a field would encode different ids than the model was trained on, with no error anywhere. One quirk to reproduce: SmolLM2's vocabulary has no symbol for six control bytes (0x04 among them) and no unknown token, and Hugging Face drops such a byte, so the solution drops it too.

## 3. Worked example by hand

**Pre-tokenizing.** `"I'm   here!!\n"` (three spaces before `here`):

| Position | Matches | Pre-token |
|---|---|---|
| `I` | alternative 2, a letter run | `I` |
| `'m` | alternative 1, a contraction | `'m` |
| three spaces | alternative 5: the run minus its last space, because `h` follows | two spaces |
| ` here` | alternative 2 with the leading space | ` here` |
| `!!` | alternative 4 | `!!` |
| `\n` | alternative 5 at the end of the text | `\n` |

So `["I", "'m", "  ", " here", "!!", "\n"]`. Likewise `"don't 2024 café"` is `["don", "'t", " 2024", " café"]`.

**Training.** The texts `"hug"`, `"hug"`, `"pug"`, `"hugs"`, with $m = 1$ and $V = 259$ (three merges after the 256 byte symbols). Each text is one pre-token, so $f(\text{hug}) = 2$, $f(\text{pug}) = 1$, $f(\text{hugs}) = 1$. The byte symbols involved have ids `g` 70, `h` 71, `p` 79, `s` 82, `u` 84.

| Step | Symbols | Pair counts | Merge (new id) |
|---|---|---|---|
| 1 | h u g (x2), p u g, h u g s | (u, g) 4, (h, u) 3, (p, u) 1, (g, s) 1 | (u, g) = `ug` (256) |
| 2 | h ug (x2), p ug, h ug s | (h, ug) 3, (p, ug) 1, (ug, s) 1 | (h, ug) = `hug` (257) |
| 3 | hug (x2), p ug, hug s | (p, ug) 1, (hug, s) 1 | (p, ug) = `pug` (258) |

At step 3 two pairs tie at 1. Their ids are (79, 256) and (257, 82); compared as tuples, (79, 256) is smaller, so `pug` wins. With the default $m = 2$ training stops after step 2, because the best count is then 1.

**Encoding.** `"hugs pug"` pre-tokenizes to `"hugs"` and `" pug"`; the space becomes `Ġ` (id 220). In `h u g s`, the ranked pairs present are (u, g) rank 0: merge, giving `h ug s`; then (h, ug) rank 1, giving `hug s`; (hug, s) has no rank, so stop: `[257, 82]`. In `Ġ p u g`: (u, g) first, then (p, ug) rank 2: `[220, 258]`. The ids are `[257, 82, 220, 258]`, and decoding them gives back `"hugs pug"`.

**Rank, not position.** With merges ranked (b, c) before (a, b), `"abc"` is `a bc`, although (a, b) is the leftmost pair. With one merge (a, a), `"aaa"` is `aa a`: both occurrences have rank 0, and the leftmost merges first.

These are the first cases a correct solution must reproduce (section 4).

## 4. The interface

```python
# python/tinyllm/tok/pretok.py
def is_space(ch: str) -> bool; def is_letter(ch: str) -> bool; def is_number(ch: str) -> bool
def pretokenize_gpt2(text: str) -> list[str]
def split_digits(text: str, individual: bool = True) -> list[str]

# python/tinyllm/tok/bpe.py
class BPETokenizer(Tokenizer):
    vocab: dict[str, int]; merges: list[tuple[str, str]]; added: dict[str, int]; pre_tokenizer: dict
    def __init__(self, vocab, merges, added_tokens=(), pre_tokenizer=None) -> None
    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int, specials: Sequence[str] = (), min_freq: int = 2) -> "BPETokenizer"
    @classmethod
    def from_gpt2(cls, vocab_json: str, merges_txt: str) -> "BPETokenizer"
    @classmethod
    def from_hf_json(cls, tokenizer_json: str) -> "BPETokenizer"
    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]
```

`added_tokens` are `(content, id, special)`; `special_ids` holds the special ones, `unk_id` is `None`, and `vocab_size` is the largest id plus one, with no gap allowed. `save` writes `tokenizer.json` in the subset (Hugging Face loads it), and `load` reads it back through `from_hf_json`. The fixtures hold only `tokenizer.json` for GPT-2; the tests derive `vocab.json` and `merges.txt` from it to exercise `from_gpt2`.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_training` | unit | the section 3 merges `ug`, `hug`, `pug` with ids 256 to 258, and the tie rule | you and the test agree on the definition |
| `test_hand_example_encode` | unit | `"hugs pug"` is `[257, 82, 220, 258]` and decodes back | encoding replays training's merges |
| `test_hand_example_pretokenize` | unit | the two section 3 strings | every id test depends on the pre-tokens |
| `test_min_freq_stops_training` | boundary | `min_freq` 2, 3, 4 give 2, 2, 1 merges | a count equal to `min_freq` still merges |
| `test_trainer_matches_hf_merges` | golden | five configs on a 3 KB story give Hugging Face's merges in order | your `C1` vocabulary is the standard algorithm |
| `test_specials_are_not_trained_on` | boundary | `<\|endoftext\|>` is id 0 and never part of a merge | document separators stay one token |
| `test_train_rejects_bad_args` | boundary | `vocab_size` below 256 plus specials, `min_freq` 0 | no silent empty training |
| `test_pretokenize_white_space_rules` | boundary | U+001C is not white space, contractions are lowercase only, a tab never leads a word | ids match on real text |
| `test_pretokenize_property` | property | 300 random texts: pieces join back, none empty, the `(?!\S)` rule | `decode(encode(x)) == x` |
| `test_split_digits` | unit | individual and run modes of `Digits` | SmolLM2 spells numbers digit by digit |
| `test_gpt2_ids_match_oracle` | golden | GPT-2's `tokenizer.json`, 300 strings, ids equal Hugging Face's and tiktoken's | you can serve GPT-2-tokenized models |
| `test_gpt2_files_load_like_tokenizer_json` | golden | `from_gpt2` on `vocab.json` and `merges.txt` gives the same ids | the original release format |
| `test_smollm2_ids_match_oracle` | golden | SmolLM2's file, the Digits sequence, 17 added tokens, 300 strings | `L7.9` and the agents' model |
| `test_smollm2_drops_unrepresentable_bytes` | boundary | `"a\x04b"` encodes to `a`, `b` | ids match Hugging Face on control bytes |
| `test_decode_matches_oracle` | golden | decode of every oracle case, and `skip_special` | streamed text equals Hugging Face's |
| `test_decode_joins_bytes_before_utf8` | boundary | the three bytes of `日` decode together, one alone is U+FFFD | `L8.2` detokenizes streams |
| `test_added_tokens_split_first_leftmost_longest` | unit | `<\|im_start\|>` is id 1, `<\|im` is text, GPT-2's end-of-text is 50256 | chat templates |
| `test_merges_apply_by_rank` | unit | swapping two ranks changes `"abc"` | the definition of BPE encoding |
| `test_rank_ties_go_leftmost` | boundary | `"aaa"` is `aa a`, `"aaaa"` is `aa aa` | identical ids in Rust (`L1.5`) |
| `test_roundtrip_property` | property | a trained tokenizer round-trips 200 random texts | the 256 symbols are a fallback |
| `test_save_load_roundtrip` | property | `load(save(t))` has the same merges, specials, and ids | the engine loads your trained vocabulary |
| `test_from_hf_json_rejects_outside_subset` | boundary | a normalizer, dropout, an unknown token, byte fallback, a template: each `ValueError` | never wrong ids without an error |
| `test_encode_batch_equals_encode` | property | `encode_batch` equals `encode` per text, empty texts included | the corpus pipeline's call |
| `test_ids_must_be_contiguous` | boundary | a vocabulary with a gap is `ValueError` | every id below `vocab_size` decodes |

### Your tests (rung R2)

Write these in `python/tests/l1-2-bpe/test_bpe.py`, importing only names from the two contracts:

```python
def test_hand_example_merges_and_tie():
    """hug hug pug hugs, min_freq 1, 259 ids: merges (u,g), (h,ug), (p,ug)."""
def test_count_equal_to_min_freq_still_merges():
    """min_freq 3 keeps the (h, ug) merge, whose count is exactly 3."""
def test_lowest_rank_merges_first():
    """With (b,c) ranked before (a,b), "abc" encodes as [a, bc]."""
def test_equal_ranks_merge_leftmost():
    """With the one merge (a,a), "aaa" encodes as [aa, a]."""
def test_pretokenizer_examples():
    """Contractions, white-space runs, U+001C, U+00A0, a leading tab, and ½."""
def test_split_digits_both_modes():
    """individual: one piece per digit; otherwise one piece per run."""
def test_character_split_across_tokens_decodes_together():
    """The three bytes of a CJK character decode to it only together."""
def test_special_token_is_one_id_and_never_trained_on():
    """A special is id 0, matched in text, skipped on request, never merged."""
def test_save_load_keeps_specials():
    """load(save(t)) has the same merges, specials, and ids."""
def test_bad_arguments_raise():
    """vocab_size below 256 + specials, or min_freq 0, is ValueError."""
def test_round_trip_any_text():
    """decode(encode(x)) == x for text with emoji, accents, tabs, and runs of spaces."""
```

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. merging left to right, or the rightmost of equal ranks | ids differ from GPT-2's on most words longer than three bytes | `test_merges_apply_by_rank` (mutant `s03`), `test_rank_ties_go_leftmost` (mutant `s04`) |
| 2. `str.isspace` or `\s` of another regex flavor; a white-space run taken whole before a word | pre-tokens differ on control characters and runs of spaces; ` here` loses its space | `test_pretokenize_white_space_rules` (mutants `s07`, `s09`, `s10`), `test_hand_example_pretokenize` (mutant `s08`) |
| 3. decoding token by token | every character split across tokens becomes U+FFFD | `test_decode_joins_bytes_before_utf8` (mutant `s16`) |
| 4. ignoring a field the loader does not implement | a normalizer or the declared Digits step is skipped and ids are silently wrong | `test_from_hf_json_rejects_outside_subset` (mutant `s19`), `test_smollm2_ids_match_oracle` (mutant `s11`) |
| 5. training on special-token text | merges such as `<\|` and `endoftext` waste the vocabulary | `test_specials_are_not_trained_on` (mutant `s06`) |
| 6. treating a byte the vocabulary lacks as an error | SmolLM2 raises on a control byte where Hugging Face drops it | `test_smollm2_drops_unrepresentable_bytes` (mutant `s15`) |
| 7. ties broken by the largest ids, or counting each pre-token once | merges diverge from Hugging Face's after a few steps | `test_trainer_matches_hf_merges` (mutants `s01`, `s02`) |
| 8. Latin-1 instead of the byte map | `é` and every non-ASCII text get the wrong symbols | `test_gpt2_ids_match_oracle` (mutant `s12`) |

## 6. Where it's used next
| Forward | `L8.2` | Registered call site uses this module. |

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L1.1` | the protocol, `check_ids` in `decode` and `id_to_token` |
| Back | `M05.2` | `bytes_to_unicode` turns pre-token bytes into symbols; `unicode_to_bytes` decodes |
| Back | `M06.2` | `Trie.longest_prefix` splits out added tokens, leftmost longest |
| Forward | `L1.5` | the Rust `tl-tok` reimplements encode and decode; its differential tests compare its ids with yours on the fixtures and a TinyStories sample |
| Forward | `L1.6` | measures GPT-2 and SmolLM2, loaded by `from_hf_json`, against Hugging Face's numbers |
| Forward | `data.07` | tokenizes corpus files through the Rust tokenizer process and shared file fixtures |

Later batches add call sites: `L7.9` and the engine load SmolLM2 with it, `L8.2` detokenizes streams, and `C1` trains its 4096-token vocabulary with your trainer. If you skip this module, `ol check L1.6` stops with `needs L1.2`: build it, or rerun with `--ref-deps`.

| Forward | `L14.4` | Whisper decoding consumes this implementation. |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `train` with a lazy heap | Hugging Face `BpeTrainer` | parallel word counting, the same heap with a `Word::merge` delta, progress bars | `tokenizers/src/models/bpe/trainer.rs` |
| `encode` by rank | tiktoken | a rank-only encoder over a byte-pair map, written in Rust, with a regex pre-split | `tiktoken/src/lib.rs` (`byte_pair_merge`) |
| `pretokenize_gpt2` | the `fancy-regex` / `onig` pre-tokenizer | the real regex engine, plus `cl100k`-style splits (numbers of 1 to 3 digits) | `tokenizers/src/pre_tokenizers/byte_level.rs` |
| `from_hf_json` | `tokenizers` `Tokenizer::from_file` | the full pipeline (normalizers, offsets, truncation, padding) | `tokenizers/src/tokenizer/serialization.rs` |
| merge cache per pre-token | `tokenizers` BPE cache | a bounded cache shared across threads; `L1.5` builds the Rust version | `tokenizers/src/utils/cache.rs` |

