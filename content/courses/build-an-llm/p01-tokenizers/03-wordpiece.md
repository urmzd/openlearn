<!-- ol:module L1.3 -->
# WordPiece (BERT basic tokenizer + greedy longest match)

## Overview

| | |
|---|---|
| **Module** | `L1.3` · build · Python · Pass 3 · 3 to 4 h |
| **You build** | `python/tinyllm/tok/wordpiece.py`: `is_bert_whitespace`, `is_bert_control`, `is_bert_punctuation`, `is_cjk`, and `WordPieceTokenizer` (`from_vocab`, `from_hf_json`, `normalize`, `basic_tokenize`, `wordpiece`, `encode`, `decode`, `save`, `load`) |
| **Contract** | [`course/contracts/py/tinyllm/tok/wordpiece.pyi`](../../../../course/contracts/py/tinyllm/tok/wordpiece.pyi) · format: [`tokenizer.md`](../../../../course/contracts/formats/tokenizer.md) (WordPiece in the `tokenizer.json` subset) |
| **Tests** | `course/tests/L1.3/` (what they check: section 4) · your own tests in `python/tests/l1-3-wordpiece/`, rung R2, graded by mutation (threshold 0.60) |
| **Needs** | `L1.1` the protocol and `check_ids` · `M06.2` the trie behind longest match and special tokens (or `--ref-deps`) |
| **Used by** | `L1.6` measures bert-base-uncased through `from_hf_json` · later: `L6.2` masked language modeling and `L6.3` ELECTRA train on its ids |
| **Milestone** | `MS-L1` |
| **Optional depth** | Devlin et al., *BERT* (2019), section 3; Wu et al., *Google's Neural Machine Translation System* (2016), section 4.1 (wordpieces); Schuster and Nakajima, *Japanese and Korean Voice Search* (2012) |

## Key Takeaways

- BERT's tokenizer is two stages: a basic tokenizer that cleans, lowercases, strips accents, and splits on white space and punctuation, then greedy longest-match WordPiece inside each word (`test_hand_example_basic_tokenize`, `test_hand_example_encode_decode`).
- Greedy longest match is not optimal segmentation: it commits to the longest prefix, and a word with any unmatched position becomes one whole `[UNK]` (`test_greedy_is_longest_first`, `test_whole_word_becomes_unk`).
- A `##` piece continues a word and can never start one; the decoder glues it back without a space (`test_continuation_needs_prefix`).
- Every normalization detail changes ids on real text: controls vanish before white space is mapped, only nonspacing marks are stripped, and lowercasing is per code point (`test_clean_text_rules`, `test_only_nonspacing_marks_are_stripped`, `test_lowercase_per_code_point`).
- With those rules exact, your tokenizer reproduces bert-base-uncased's ids, template, and decodes on 300 hard strings (`test_bert_ids_match_oracle`, `test_bert_template_matches_oracle`, `test_bert_decode_matches_oracle`).

## How to work this chapter

```bash
ol start L1.3              # stubs wordpiece.py into your repo
ol tests L1.3              # read the test catalog first
ol check L1.3              # exit code is the verdict; then grades your tests by mutation
ol check L1.3 --ref-deps   # only if your L1.1 or M06.2 is not passing yet
ol diff  L1.3              # after passing: your code against the reference
```

---

## 1. Why now

Byte-level BPE (`L1.2`) is the tokenizer of decoder-only models. Pass 5 trains encoders too: the masked-language-model objective of `L6.2` and ELECTRA in `L6.3` follow BERT, and the evaluation in `L6.2` compares against bert-base-uncased's vocabulary of 30522 WordPieces. Its ids come from a different pipeline (lowercasing, accent stripping, punctuation splitting, `##` continuations, a whole-word `[UNK]`), and a `[MASK]` written in the input must be one id, because that is how the masked-LM objective marks its targets. Without this module your encoder models have no tokenizer, and `L1.6` has no third family to compare when you choose the capstone's vocabulary.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $W$ | the vocabulary: word-start pieces and `##` continuation pieces | `dict[str, int]` |
| $x$ | one word after the basic tokenizer | `str` |
| $p$ | the continuation prefix, `##` | `str` |
| $L$ | `max_input_chars_per_word` (100 for BERT) | `int` |
| $\mathrm{Mn}$, $\mathrm{C*}$, $\mathrm{P*}$ | Unicode general categories: nonspacing mark, any control or format or unassigned, any punctuation | category |

**The basic tokenizer** (Hugging Face `BertNormalizer` then `BertPreTokenizer`). In this order:

1. **Clean.** Drop U+0000, U+FFFD, and every control character: category `C*`, except tab, newline, and carriage return, which count as white space. Then map every white-space character (the Unicode White_Space set) to U+0020. The order matters for U+000B: it is white space and a control, and the control rule wins, so it vanishes. The zero-width space U+200B is category Cf and vanishes too.
2. **CJK.** Put a space on both sides of every CJK ideograph (the eight blocks in the contract, starting at U+4E00), so each becomes its own word: Chinese has no spaces, and BERT's vocabulary holds single ideographs.
3. **Strip accents** when `strip_accents` is true, or when it is `None` and `lowercase` is true (uncased BERT strips, cased BERT keeps): decompose with NFD, then drop category Mn. `é` becomes `e` + U+0301, and U+0301 is Mn. A spacing mark such as the Devanagari vowel sign U+093E (category Mc) is a vowel, not an accent, and stays.
4. **Lowercase** one code point at a time. Python's `str.lower()` on a whole string applies the Greek final-sigma rule (`"ΣΑΣ".lower()` ends in `ς`); per code point it is `σ`, which is what BERT's ids assume.
5. **Split** on white space, and make every punctuation character its own word. BERT's punctuation is category `P*` plus every ASCII character in 33 to 47, 58 to 64, 91 to 96, 123 to 126, so `$`, `+`, `^` (symbols, category S) split too.

**Special tokens** are split out of the raw text before any of this, by leftmost-longest match through the `M06.2` trie, as written: `[MASK]` is the mask id, while `[mask]` is ordinary text (it would be lowercased and split into `[`, `mask`, `]`).

**WordPiece, greedy longest match.** For one word $x$:

1. if $x$ has more than $L$ code points, return `[UNK]` without searching;
2. at position 0, take the longest entry of $W$ that is a prefix of $x$;
3. at every later position $i$, take the longest entry $p\,y$ of $W$ such that $y$ is a prefix of $x[i:]$;
4. if some position has no match, the **whole word** is one `[UNK]`; the pieces found so far are discarded.

Two tries make each step one walk (`M06.2`): one over all entries for the word start, one over the `##` entries stored without the prefix. Greedy longest match is not the best segmentation under any score. It is a fixed, cheap rule, and BERT's vocabulary was built for it, so it is the rule you must reproduce.

**Decoding** is the Hugging Face WordPiece decoder with cleanup: every token after the first gets a leading space, unless it starts with `##`, which is removed instead; then, inside each token, the replacements ` .` to `.`, ` ?` to `?`, ` !` to `!`, ` ,` to `,`, ` ' ` to `'`, ` n't` to `n't`, ` 'm` to `'m`, ` do not` to ` don't`, ` 's` to `'s`, ` 've` to `'ve`, ` 're` to `'re`. Decoding cannot restore case, accents, or white space: the basic tokenizer removed them.

**Templates.** `add_special` is BERT's `TemplateProcessing`: `[CLS]` + ids + `[SEP]`. BERT reads `[CLS]`'s final vector as the sequence summary, and `[SEP]` ends a segment.

## 3. Worked example by hand

A 15-entry vocabulary:

| id | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| token | `[PAD]` | `[UNK]` | `[CLS]` | `[SEP]` | `[MASK]` | the | un | ##aff | ##able | ##a | ! | u | ##n | cafe | ##s |

**Basic tokenizer.** `"The UNAFFABLE!"` lowercases to `"the unaffable!"`, splits on the space, and `!` is punctuation: `["the", "unaffable", "!"]`. `"日本x y"` gives `["日", "本", "x", "y"]` (each ideograph spaced). `"Café"` gives `["cafe"]` (uncased: the accent is stripped).

**WordPiece on "unaffable".** Position 0: the entries that are prefixes are `u` and `un`; the longest is `un` (6). Position 2, `"affable"`: the `##` entries whose rest is a prefix are `##a` and `##aff`; the longest is `##aff` (7). Position 5, `"able"`: `##able` (8). So `[6, 7, 8]`. Had you taken the shortest match, `##a` at position 2 would leave `"ffable"`, which no `##` entry starts, and the whole word would be `[UNK]`.

**Encode.** `"The UNAFFABLE!"` is `[5, 6, 7, 8, 10]`; with `add_special=True`, `[2, 5, 6, 7, 8, 10, 3]`. `"cafés"` strips to `"cafes"`: `cafe` (13) then `##s` (14).

**Whole-word unknown.** `"unaffablex"` finds `un`, `##aff`, `##able`, then nothing for `x`: the result is `[1]`, one `[UNK]`, not `[6, 7, 8, 1]`.

**Decode.** `[5, 6, 7, 8, 10]` is the tokens `the un ##aff ##able !`: `"the"`, then `" un"`, then `"aff"` and `"able"` glued, then `" !"`, which cleanup turns into `"!"`. Result: `"the unaffable!"`.

These are the first cases in section 4: `test_hand_example_basic_tokenize` and `test_hand_example_encode_decode`.

## 4. The interface

```python
# python/tinyllm/tok/wordpiece.py
def is_bert_whitespace(ch: str) -> bool; def is_bert_control(ch: str) -> bool
def is_bert_punctuation(ch: str) -> bool; def is_cjk(ch: str) -> bool

class WordPieceTokenizer(Tokenizer):
    vocab: dict[str, int]
    def __init__(self, vocab: Mapping[str, int], unk_token: str = "[UNK]", prefix: str = "##",
                 max_input_chars_per_word: int = 100, lowercase: bool = True,
                 strip_accents: Optional[bool] = None) -> None
    @classmethod
    def from_vocab(cls, vocab_txt: str, lowercase: bool = True, strip_accents: Optional[bool] = None) -> "WordPieceTokenizer"
    @classmethod
    def from_hf_json(cls, tokenizer_json: str) -> "WordPieceTokenizer"
    def normalize(self, text: str) -> str
    def basic_tokenize(self, text: str) -> list[str]
    def wordpiece(self, word: str) -> list[int]
```

`special_ids` are whichever of `[PAD] [UNK] [CLS] [SEP] [MASK]` the vocabulary has (from `added_tokens` when loading a `tokenizer.json`), and `unk_id` is the id of `unk_token`. `vocab.txt` (the original BERT release) has one token per line, line $n$ being id $n$ from 0. `save` writes a `tokenizer.json` with `BertNormalizer`, `BertPreTokenizer`, `WordPiece`, the `[CLS] $A [SEP]` template, and the WordPiece decoder.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_basic_tokenize` | unit | the section 3 words, CJK spacing, accent stripping, an all-space text | you and the test agree on the definition |
| `test_hand_example_encode_decode` | unit | `[5, 6, 7, 8, 10]`, the template, the decode, `cafés` | the masked-LM data of `L6.2` |
| `test_whole_word_becomes_unk` | boundary | `"unaffablex"` is `[1]` | unknown words cost one id, as in BERT |
| `test_greedy_is_longest_first` | boundary | `un ##aff ##able`, and `"una"` is `un ##a` | the rule BERT's vocabulary was built for |
| `test_continuation_needs_prefix` | boundary | `"unn"` is `un ##n`; `"aff"` alone is `[UNK]` | `##` pieces never start a word |
| `test_max_input_chars_per_word` | boundary | a limit of 5 segments `"unaff"` and refuses `"unaffa"` | long junk tokens cannot blow up the search |
| `test_clean_text_rules` | boundary | U+0000, U+FFFD, U+000B, U+200B vanish; tab, U+00A0, U+3000 become spaces | ids match on scraped text |
| `test_strip_accents_follows_lowercase` | unit | the four combinations of `lowercase` and `strip_accents` | cased and uncased models from one class |
| `test_only_nonspacing_marks_are_stripped` | boundary | `é` loses its accent, the Devanagari vowel sign stays | non-Latin scripts keep their letters |
| `test_punctuation_is_split` | boundary | `$`, `+`, `^`, `«`, `»` are words of their own | ASCII symbols are punctuation to BERT |
| `test_lowercase_per_code_point` | boundary | `"ΣΑΣ"` is `"σασ"` | no final sigma, as in BERT's vocabulary |
| `test_specials_match_as_written` | boundary | `[MASK]` is id 4; `[mask]` is text | `L6.2` writes `[MASK]` into its inputs |
| `test_bert_ids_match_oracle` | golden | bert-base-uncased, 300 strings, ids equal Hugging Face's | your encoders use the real vocabulary |
| `test_bert_template_matches_oracle` | golden | `add_special` ids on the same strings | encoder inputs carry the template BERT was trained with |
| `test_bert_decode_matches_oracle` | golden | decode and `skip_special` on the same strings | printed predictions in `L6.2` |
| `test_from_vocab_equals_tokenizer_json` | golden | `vocab.txt` and `tokenizer.json` give the same ids | the original release format |
| `test_save_load_roundtrip` | property | `load(save(t))` keeps ids, casing, and accent rules | a trained encoder ships its tokenizer |
| `test_vocab_needs_unk` | boundary | no `[UNK]`, or an id gap, is `ValueError` | every word has somewhere to go |

### Your tests (rung R2)

Write these in `python/tests/l1-3-wordpiece/test_wordpiece.py`, importing only names from the contract:

```python
def test_unaffable_is_three_pieces():
    """"The UNAFFABLE!" encodes to the, un, ##aff, ##able, !."""
def test_unsegmentable_word_is_one_unk():
    """"unaffablex" is a single [UNK], not three pieces and an [UNK]."""
def test_continuation_pieces_only_inside_words():
    """"unn" is un ##n; "aff" alone is [UNK]."""
def test_word_length_limit_is_inclusive():
    """With a limit of 5, "unaff" segments and "unaffa" is [UNK]."""
def test_cleaning_drops_controls_and_spaces_white_space():
    """U+000B and U+200B vanish; tab and U+00A0 become spaces."""
def test_accent_rules():
    """Uncased strips accents; cased keeps them; strip_accents=False keeps them."""
def test_spacing_marks_survive():
    """The Devanagari vowel sign U+093E (category Mc) is not stripped."""
def test_ascii_symbols_and_cjk_split():
    """$ + ^ are punctuation; each CJK ideograph is its own word."""
def test_sigma_lowercases_per_code_point():
    """"ΣΑΣ" lowercases to "σασ" (no final sigma)."""
def test_mask_in_text_and_template():
    """"[MASK]" is one id, "[mask]" is not; add_special wraps [CLS] ... [SEP]."""
def test_decode_glues_and_cleans():
    """[the, un, ##aff, ##able, !] decodes to "the unaffable!"."""
def test_vocab_file_and_save_load():
    """vocab.txt line n is id n; save then load keeps ids and casing."""
def test_vocab_without_unk_is_rejected():
    """A vocab with no [UNK], or with an id gap, is ValueError."""
```

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. keeping the pieces found before an unmatched position | `"unaffablex"` is four ids where BERT has one; sequence lengths drift | `test_whole_word_becomes_unk` (mutant `s01`) |
| 2. shortest match, or trying `##` pieces at the word start | words BERT segments become `[UNK]` | `test_greedy_is_longest_first` (mutant `s02`), `test_continuation_needs_prefix` (mutant `s03`) |
| 3. stripping accents only when `strip_accents` is true | uncased BERT keeps `é`, and every accented word gets different ids | `test_strip_accents_follows_lowercase` (mutant `s06`) |
| 4. `str.lower()` on the whole string | Greek words ending in sigma get different ids | `test_lowercase_per_code_point` (mutant `s09`) |
| 5. testing white space before control characters | U+000B becomes a space and splits a word that BERT keeps whole | `test_clean_text_rules` (mutant `s05`) |
| 6. stripping every mark category | Hindi and other scripts lose vowel signs | `test_only_nonspacing_marks_are_stripped` (mutant `s07`) |
| 7. `unicodedata` punctuation only | `$5` stays one word | `test_punctuation_is_split` (mutant `s08`) |
| 8. matching specials after normalization, or case-insensitively | `[mask]` typed by a user becomes the mask id | `test_specials_match_as_written` (mutant `s10`) |

## 6. Where it's used next
| Forward | `L6.2` | Registered call site uses this module. |

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L1.1` | the protocol and `check_ids`; `add_special` is the `[CLS] ... [SEP]` template |
| Back | `M06.2` | two tries make each longest-match step one walk; a third splits out special tokens |
| Forward | `L1.6` | measures bert-base-uncased against Hugging Face's numbers, `[UNK]` counted as a fallback |

`L6.2` (masked language modeling with `[MASK]`) and `L6.3` (ELECTRA) train on these ids when Pass 5 arrives. If you skip this module, `ol check L1.6` stops with `needs L1.3`: build it, or rerun with `--ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `basic_tokenize` | Hugging Face `BertNormalizer` and `BertPreTokenizer` | offsets for every piece, so predictions map back to the original text | `tokenizers/src/normalizers/bert.rs` |
| `wordpiece` | `tokenizers` `WordPiece` model | the same greedy loop in Rust; a trainer that derives WordPieces from BPE merges | `tokenizers/src/models/wordpiece/mod.rs` |
| `[CLS] ... [SEP]` | `TemplateProcessing` | pair templates with token type ids for two-segment inputs | `tokenizers/src/processors/template.rs` |
| greedy longest match | Fast WordPiece (Song et al., 2021) | linear-time WordPiece with an Aho-Corasick-style trie | Song et al., *Fast WordPiece Tokenization* |
