# WordPiece (BERT basic tokenizer + greedy longest match)

## Overview

| | |
|---|---|
| **Module** | `L1.3` · build · Python · Pass 3 · 3 to 4 h |
| **Solution** | [`wordpiece.py`](../../../../course/solutions/python/tinyllm/tok/wordpiece.py): `is_bert_whitespace`, `is_bert_control`, `is_bert_punctuation`, `is_cjk`, and `WordPieceTokenizer` (`from_vocab`, `from_hf_json`, `normalize`, `basic_tokenize`, `wordpiece`, `encode`, `decode`, `save`, `load`) · model tests: [`test_wordpiece.py`](../../../../course/solutions/learner-tests/L1.3/python/tests/l1-3-wordpiece/test_wordpiece.py) |
| **Contract** | [`course/contracts/py/tinyllm/tok/wordpiece.pyi`](../../../../course/contracts/py/tinyllm/tok/wordpiece.pyi) · format: [`tokenizer.md`](../../../../course/contracts/formats/tokenizer.md) (WordPiece in the `tokenizer.json` subset) |
| **Needs** | `L1.1` the protocol and `check_ids` · `M06.2` the trie behind longest match and special tokens |
| **Used by** | `L1.6` measures bert-base-uncased through `from_hf_json` · later: `L6.2` masked language modeling and `L6.3` ELECTRA train on its ids · part of the Pass 3 tokenizer layer, with `L1.1`, `L1.2`, `L1.4`, and `L1.6` |
| **Optional depth** | Devlin et al., *BERT* (2019), section 3; Wu et al., *Google's Neural Machine Translation System* (2016), section 4.1 (wordpieces); Schuster and Nakajima, *Japanese and Korean Voice Search* (2012) |

## Key Takeaways

- BERT's tokenizer is two stages: a basic tokenizer that cleans, lowercases, strips accents, and splits on white space and punctuation, then greedy longest-match WordPiece inside each word.
- Greedy longest match is not optimal segmentation: it commits to the longest prefix, and a word with any unmatched position becomes one whole `[UNK]`.
- A `##` piece continues a word and can never start one; the decoder glues it back without a space.
- Every normalization detail changes ids on real text: controls vanish before white space is mapped, only nonspacing marks are stripped, and lowercasing is per code point.
- With those rules exact, the tokenizer reproduces bert-base-uncased's ids, template, and decodes on 300 hard strings.

## How to read this chapter

Sections 1 to 3 build BERT's basic tokenizer and greedy longest match from first principles and work them by hand. Section 4 fixes the interface and lists what a correct solution must handle. Section 5 walks through `wordpiece.py` and explains why each choice is the right one. Try to predict each step of the solution before reading it.

---

## 1. Why now

Byte-level BPE (`L1.2`) is the tokenizer of decoder-only models. Pass 5 trains encoders too: the masked-language-model objective of `L6.2` and ELECTRA in `L6.3` follow BERT, and the evaluation in `L6.2` compares against bert-base-uncased's vocabulary of 30522 WordPieces. Its ids come from a different pipeline (lowercasing, accent stripping, punctuation splitting, `##` continuations, a whole-word `[UNK]`), and a `[MASK]` written in the input must be one id, because that is how the masked-LM objective marks its targets. Without this module the encoder models have no tokenizer, and `L1.6` has no third family to compare when the capstone's vocabulary is chosen.

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

Two tries make each step one walk (`M06.2`): one over all entries for the word start, one over the `##` entries stored without the prefix. Greedy longest match is not the best segmentation under any score. It is a fixed, cheap rule, and BERT's vocabulary was built for it, so it is the rule to reproduce.

**Decoding** is the Hugging Face WordPiece decoder with cleanup: every token after the first gets a leading space, unless it starts with `##`, which is removed instead; then, inside each token, the replacements ` .` to `.`, ` ?` to `?`, ` !` to `!`, ` ,` to `,`, ` ' ` to `'`, ` n't` to `n't`, ` 'm` to `'m`, ` do not` to ` don't`, ` 's` to `'s`, ` 've` to `'ve`, ` 're` to `'re`. Decoding cannot restore case, accents, or white space: the basic tokenizer removed them.

**Templates.** `add_special` is BERT's `TemplateProcessing`: `[CLS]` + ids + `[SEP]`. BERT reads `[CLS]`'s final vector as the sequence summary, and `[SEP]` ends a segment.

## 3. Worked example by hand

A 15-entry vocabulary:

| id | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| token | `[PAD]` | `[UNK]` | `[CLS]` | `[SEP]` | `[MASK]` | the | un | ##aff | ##able | ##a | ! | u | ##n | cafe | ##s |

**Basic tokenizer.** `"The UNAFFABLE!"` lowercases to `"the unaffable!"`, splits on the space, and `!` is punctuation: `["the", "unaffable", "!"]`. `"日本x y"` gives `["日", "本", "x", "y"]` (each ideograph spaced). `"Café"` gives `["cafe"]` (uncased: the accent is stripped).

**WordPiece on "unaffable".** Position 0: the entries that are prefixes are `u` and `un`; the longest is `un` (6). Position 2, `"affable"`: the `##` entries whose rest is a prefix are `##a` and `##aff`; the longest is `##aff` (7). Position 5, `"able"`: `##able` (8). So `[6, 7, 8]`. Taking the shortest match instead, `##a` at position 2 would leave `"ffable"`, which no `##` entry starts, and the whole word would be `[UNK]`.

**Encode.** `"The UNAFFABLE!"` is `[5, 6, 7, 8, 10]`; with `add_special=True`, `[2, 5, 6, 7, 8, 10, 3]`. `"cafés"` strips to `"cafes"`: `cafe` (13) then `##s` (14).

**Whole-word unknown.** `"unaffablex"` finds `un`, `##aff`, `##able`, then nothing for `x`: the result is `[1]`, one `[UNK]`, not `[6, 7, 8, 1]`.

**Decode.** `[5, 6, 7, 8, 10]` is the tokens `the un ##aff ##able !`: `"the"`, then `" un"`, then `"aff"` and `"able"` glued, then `" !"`, which cleanup turns into `"!"`. Result: `"the unaffable!"`.

These are the first cases a correct solution must reproduce (section 4).

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

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the hand-worked basic tokenizer | unit | the section 3 words, CJK spacing, accent stripping, an all-space text | everyone agrees on the definition |
| the hand-worked encode and decode | unit | `[5, 6, 7, 8, 10]`, the template, the decode, `cafés` | the masked-LM data of `L6.2` |
| a whole word becomes `[UNK]` | boundary | `"unaffablex"` is `[1]` | unknown words cost one id, as in BERT |
| greedy is longest first | boundary | `un ##aff ##able`, and `"una"` is `un ##a` | the rule BERT's vocabulary was built for |
| continuations need the prefix | boundary | `"unn"` is `un ##n`; `"aff"` alone is `[UNK]` | `##` pieces never start a word |
| `max_input_chars_per_word` | boundary | a limit of 5 segments `"unaff"` and refuses `"unaffa"` | long junk tokens cannot blow up the search |
| clean-text rules | boundary | U+0000, U+FFFD, U+000B, U+200B vanish; tab, U+00A0, U+3000 become spaces | ids match on scraped text |
| accent stripping follows lowercase | unit | the four combinations of `lowercase` and `strip_accents` | cased and uncased models from one class |
| only nonspacing marks are stripped | boundary | `é` loses its accent, the Devanagari vowel sign stays | non-Latin scripts keep their letters |
| punctuation is split | boundary | `$`, `+`, `^`, `«`, `»` are words of their own | ASCII symbols are punctuation to BERT |
| lowercase per code point | boundary | `"ΣΑΣ"` is `"σασ"` | no final sigma, as in BERT's vocabulary |
| specials match as written | boundary | `[MASK]` is id 4; `[mask]` is text | `L6.2` writes `[MASK]` into its inputs |
| BERT ids match the oracle | golden | bert-base-uncased, 300 strings, ids equal Hugging Face's | the encoders use the real vocabulary |
| BERT template matches the oracle | golden | `add_special` ids on the same strings | encoder inputs carry the template BERT was trained with |
| BERT decode matches the oracle | golden | decode and `skip_special` on the same strings | printed predictions in `L6.2` |
| `vocab.txt` equals `tokenizer.json` | golden | `vocab.txt` and `tokenizer.json` give the same ids | the original release format |
| save and load round trip | property | `load(save(t))` keeps ids, casing, and accent rules | a trained encoder ships its tokenizer |
| the vocabulary needs `[UNK]` | boundary | no `[UNK]`, or an id gap, is `ValueError` | every word has somewhere to go |

### The model tests

The model suite, [`test_wordpiece.py`](../../../../course/solutions/learner-tests/L1.3/python/tests/l1-3-wordpiece/test_wordpiece.py), imports only names from the contract. Its names and docstrings list the rules most worth pinning:

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

## 5. The solution, step by step

### From first principles

1. **The ids are defined by BERT's pipeline**, so the solution is a transcription of it, stage by stage, in Hugging Face's order: specials, clean, CJK, accents, lowercase, split, WordPiece. Each stage is a pure function of its input, which makes each one testable on its own (`normalize`, `basic_tokenize`, `wordpiece`).
2. **Specials must be found before normalization**, because normalization would lowercase `[MASK]` and split its brackets. So `encode` first cuts the raw text at leftmost-longest special matches.
3. **Character rules are per code point and order-sensitive.** Cleaning tests "control" before "white space" (U+000B is both), CJK spacing is applied during the same pass, accent stripping needs NFD first so that accents become separate marks, and lowercasing must be per code point to avoid the final-sigma rule.
4. **Splitting is a single scan.** White space ends a word; punctuation ends a word and is a word; everything else extends the current word.
5. **Greedy longest match is a loop of longest-prefix queries.** At position 0 query all entries; at later positions query the `##` entries with the prefix removed, so one trie walk finds the longest continuation. The first failed query makes the whole word `[UNK]`, so no backtracking is needed.
6. **Decoding inverts only what can be inverted**: glue `##` pieces, add spaces between words, then apply the fixed cleanup replacements.

### The code, annotated

**Character classes.** Each predicate encodes one BERT rule exactly.

```python
def is_bert_control(ch: str) -> bool:
    if ch in "\t\n\r":
        return False                               # these count as white space instead
    return unicodedata.category(ch).startswith("C")   # Cc, Cf (U+200B), Cn, Co, Cs

def is_bert_punctuation(ch: str) -> bool:
    o = ord(ch)
    if 33 <= o <= 47 or 58 <= o <= 64 or 91 <= o <= 96 or 123 <= o <= 126:
        return True                                # all ASCII non-alphanumerics: $ + ^ too
    return unicodedata.category(ch).startswith("P")

def is_cjk(ch: str) -> bool:
    o = ord(ch)
    return any(lo <= o <= hi for lo, hi in CJK)    # the eight ideograph blocks
```

**Normalization.** One pass for cleaning and CJK, then the two whole-string transforms.

```python
    def normalize(self, text: str) -> str:
        """Steps 2 to 5: clean, space CJK, strip accents, lowercase."""
        out = []
        for ch in text:
            o = ord(ch)
            # control first: U+000B is white space *and* a control, and it must vanish
            if o == 0 or o == 0xFFFD or is_bert_control(ch):
                continue
            if is_bert_whitespace(ch):
                out.append(" ")
            elif is_cjk(ch):
                out.append(" " + ch + " ")           # each ideograph becomes its own word
            else:
                out.append(ch)
        s = "".join(out)
        # None means "follow lowercase": uncased BERT strips, cased BERT keeps
        strip = self.strip_accents if self.strip_accents is not None else self.lowercase
        if strip:
            # NFD splits é into e + U+0301; only Mn is dropped, so Mc vowel signs stay
            s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
        if self.lowercase:
            s = "".join(c.lower() for c in s)        # per code point: no final-sigma rule
        return s
```

**Splitting.**

```python
    def basic_tokenize(self, text: str) -> list[str]:
        words: list[str] = []
        cur: list[str] = []
        for ch in self.normalize(text):
            if is_bert_whitespace(ch):               # after normalize, only U+0020 remains
                if cur:
                    words.append("".join(cur))
                    cur = []
            elif is_bert_punctuation(ch):
                if cur:
                    words.append("".join(cur))
                    cur = []
                words.append(ch)                     # punctuation is a word of its own
            else:
                cur.append(ch)
        if cur:
            words.append("".join(cur))
        return words
```

On section 3, `"The UNAFFABLE!"` normalizes to `"the unaffable!"`; the scan emits `the` at the space, `unaffable` at `!`, and `!` itself.

**Greedy longest match.** The two tries are built in the constructor: `_word` holds every entry, `_cont` holds each `##` entry with the prefix stripped.

```python
        self._word = Trie()  # pieces that may start a word
        self._cont = Trie()  # `##` pieces, stored without the prefix
        for t, i in self.vocab.items():
            if prefix and t.startswith(prefix) and len(t) > len(prefix):
                self._cont.insert(t[len(prefix) :], i)    # "##aff" is found by walking "aff"
            self._word.insert(t, i)

    def wordpiece(self, word: str) -> list[int]:
        """Step 7 for one word: greedy longest match, or [UNK]."""
        if len(word) > self.max_input_chars_per_word:  # > , so a word of exactly L is searched
            return [self.unk_id]
        ids: list[int] = []
        start = 0
        while start < len(word):
            trie = self._word if start == 0 else self._cont   # ## pieces only after the start
            n, v = trie.longest_prefix(word, start)
            if n == 0 or v is None:
                return [self.unk_id]                 # whole word: discard the pieces found
            ids.append(v)
            start += n
        return ids
```

Trace `"unaffable"`: `_word.longest_prefix` at 0 sees `u` and `un` and returns length 2, id 6. `_cont` at 2 sees `a` (`##a`) and `aff` (`##aff`) and returns length 3, id 7. At 5 it finds `able`, id 8. For `"unaffablex"`, the query at 9 finds nothing, and the result is `[1]`.

**Encode.** Specials first, on the raw text.

```python
    def encode(self, text: str, add_special: bool = False) -> list[int]:
        ids: list[int] = []
        start = i = 0
        segs: list[tuple[str, Optional[int]]] = []
        while i < len(text):
            # as written: "[MASK]" matches, "[mask]" does not
            n, v = self._special_trie.longest_prefix(text, i)
            if n > 0 and v is not None:
                segs.append((text[start:i], None))
                segs.append(("", v))
                i += n
                start = i
            else:
                i += 1
        segs.append((text[start:], None))
        for seg, sid in segs:
            if sid is not None:
                ids.append(sid)
            else:
                for w in self.basic_tokenize(seg):
                    ids += self.wordpiece(w)
        if add_special:
            ids = [self.vocab["[CLS]"]] + ids + [self.vocab["[SEP]"]]   # BERT's single template
        return ids
```

**Decode.** Glue, space, clean up.

```python
    def decode(self, ids: Sequence[int], skip_special: bool = False) -> str:
        special = set(self.special_ids.values())
        toks = [
            self._id_to_tok[i]
            for i in check_ids(ids, self.vocab_size)
            if not (skip_special and i in special)
        ]
        out = []
        for k, t in enumerate(toks):
            if k > 0:
                # a continuation loses its ##; anything else gets a leading space
                t = t[len(self.prefix) :] if t.startswith(self.prefix) else " " + t
            for dirty, clean in (
                (" .", "."), (" ?", "?"), (" !", "!"), (" ,", ","), (" ' ", "'"),
                (" n't", "n't"), (" 'm", "'m"), (" do not", " don't"), (" 's", "'s"),
                (" 've", "'ve"), (" 're", "'re"),
            ):
                t = t.replace(dirty, clean)          # per token, as Hugging Face's cleanup
            out.append(t)
        return "".join(out)
```

On `[5, 6, 7, 8, 10]`: `the`, ` un`, `aff`, `able`, and ` !` cleaned to `!`, joined as `"the unaffable!"`.

Full solution: [`wordpiece.py`](../../../../course/solutions/python/tinyllm/tok/wordpiece.py) (also `from_vocab`, `from_hf_json`, `save`).

### Why this solution

- **Cost.** Normalization and splitting are $O(n)$ per text (plus NFD). WordPiece on a word of length $m$ is a sequence of trie walks; each walk can read ahead up to the longest entry, so the worst case is $O(m \cdot \ell_{\max})$, and in practice close to $O(m)$. The length limit caps pathological words at $L = 100$.
- **Alternatives.** A max-score segmentation (dynamic programming over a unigram score) would often find a better split, and different ids from BERT's. Hash-set lookups of every prefix length work but cost $O(m^2)$ string slices per word; the trie avoids building candidates at all. Python's `str.lower()` or `casefold()` on the whole string would be simpler and wrong for Greek. Linear-time Fast WordPiece (see Going further) is the production optimization; it is not needed at this scale.
- **Accepted tradeoffs.** No offsets: the tokenizer cannot map a piece back to its span in the original text, which `L6.2` does not need. Decoding is lossy by design, since case, accents, and white space were removed on the way in. `token_to_id` reads the vocabulary directly, so it does not apply normalization.

### Explain it back

1. Why does `normalize` check `is_bert_control` before `is_bert_whitespace`?
   <details><summary>Answer</summary>
   U+000B (vertical tab) is in the White_Space set and in category Cc. BERT's cleaner drops controls first, so U+000B vanishes and the word around it stays whole. Checking white space first would map it to a space and split the word, giving different ids from BERT.
   </details>
2. Why are `##` pieces stored in a second trie without the prefix, instead of querying `"##" + word[start:]` in one trie?
   <details><summary>Answer</summary>
   Building `"##" + word[start:]` creates a new string at every position, and it makes one trie answer two different questions. With a dedicated trie keyed by the rest of the piece, the continuation query is a walk starting directly at `word[start]`, and a `##` piece can never be returned at the word start because `_cont` is only queried for `start > 0` (and `#` is punctuation, so no word out of `basic_tokenize` begins with `##`).
   </details>
3. Why does a failed match make the whole word `[UNK]` instead of emitting the pieces found so far plus `[UNK]`?
   <details><summary>Answer</summary>
   That is BERT's rule and its vocabulary and models were trained with it: an unknown word costs exactly one id. Keeping partial pieces would change sequence lengths and produce id sequences BERT never saw in training.
   </details>
4. When `strip_accents` is `None`, why does the solution strip accents only if `lowercase` is true?
   <details><summary>Answer</summary>
   That mirrors Hugging Face's `BertNormalizer`: uncased BERT models were trained on lowercased, accent-stripped text, while cased models keep accents. Making `None` follow `lowercase` lets one class load both families from their files.
   </details>
5. Why must special tokens be split out before `normalize` runs?
   <details><summary>Answer</summary>
   Normalization lowercases and the splitter treats `[` and `]` as punctuation, so `[MASK]` would become `[`, `mask`, `]`. Matching on the raw text, as written, keeps `[MASK]` one id and leaves a user's `[mask]` as ordinary text.
   </details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. keeping the pieces found before an unmatched position | `"unaffablex"` is four ids where BERT has one; sequence lengths drift | `wordpiece` returns `[self.unk_id]` at the first failed query |
| 2. shortest match, or trying `##` pieces at the word start | words BERT segments become `[UNK]` | `longest_prefix` on `_word` at position 0 and on `_cont` only after it |
| 3. stripping accents only when `strip_accents` is true | uncased BERT keeps `é`, and every accented word gets different ids | `strip` falls back to `self.lowercase` when `strip_accents` is `None` |
| 4. `str.lower()` on the whole string | Greek words ending in sigma get different ids | lowercases one code point at a time |
| 5. testing white space before control characters | U+000B becomes a space and splits a word that BERT keeps whole | the control test comes first in `normalize` |
| 6. stripping every mark category | Hindi and other scripts lose vowel signs | drops only category `Mn` after NFD |
| 7. `unicodedata` punctuation only | `$5` stays one word | `is_bert_punctuation` includes the four ASCII ranges |
| 8. matching specials after normalization, or case-insensitively | `[mask]` typed by a user becomes the mask id | `encode` matches specials on the raw text through `_special_trie` |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L1.1` | the protocol and `check_ids`; `add_special` is the `[CLS] ... [SEP]` template |
| Back | `M06.2` | two tries make each longest-match step one walk; a third splits out special tokens |
| Forward | `L1.6` | measures bert-base-uncased against Hugging Face's numbers, `[UNK]` counted as a fallback |
| Forward | `L6.2` | BERT tokenizes with `WordPieceTokenizer` |

`L6.2` (masked language modeling with `[MASK]`) and `L6.3` (ELECTRA) train on these ids when Pass 5 arrives.

## Going further

| Piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `basic_tokenize` | Hugging Face `BertNormalizer` and `BertPreTokenizer` | offsets for every piece, so predictions map back to the original text | `tokenizers/src/normalizers/bert.rs` |
| `wordpiece` | `tokenizers` `WordPiece` model | the same greedy loop in Rust; a trainer that derives WordPieces from BPE merges | `tokenizers/src/models/wordpiece/mod.rs` |
| `[CLS] ... [SEP]` | `TemplateProcessing` | pair templates with token type ids for two-segment inputs | `tokenizers/src/processors/template.rs` |
| greedy longest match | Fast WordPiece (Song et al., 2021) | linear-time WordPiece with an Aho-Corasick-style trie | Song et al., *Fast WordPiece Tokenization* |
