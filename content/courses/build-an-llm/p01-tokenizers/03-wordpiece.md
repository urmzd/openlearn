# WordPiece

[Byte-level BPE](02-byte-level-bpe.md) is the tokenizer of decoder-only models. Later in this book you will train encoders too: [BERT's masked-language-model objective](../p06-objectives/02-bert-masked-lm.md) and [ELECTRA's replaced-token detection](../p06-objectives/03-electra-replaced-token-detection.md) both follow BERT, and the BERT evaluation compares against bert-base-uncased's vocabulary of 30522 WordPieces. Those ids come from a quite different pipeline: lowercasing, accent stripping, punctuation splitting, `##` continuations, and a whole-word `[UNK]`. And a `[MASK]` written in the input must be one id, because that is how the masked-LM objective marks its targets.

This section builds that pipeline exactly. BERT's tokenizer has two stages: a basic tokenizer that cleans, lowercases, strips accents, and splits on white space and punctuation, and then greedy longest-match WordPiece inside each word. Every normalization detail changes ids on real text, so the work here is mostly about getting small rules exactly right. With those rules exact, the tokenizer reproduces bert-base-uncased's ids, template, and decodes on 300 hard strings, and [Tokenizer metrics](06-tokenizer-metrics.md) gains its third tokenizer family to compare when the capstone's vocabulary is chosen.

You need the protocol and `check_ids` from [the tokenizer protocol](01-tokenizer-protocol-and-char.md), and the trie behind longest match and special tokens from [Trees and tries, longest-prefix match](../../mathematics-for-computing/06-discrete-math-2/02-trees-and-tries-longest-prefix-match.md) in *Mathematics for Computing*. The solution is [`wordpiece.py`](../../../../course/solutions/python/tinyllm/tok/wordpiece.py), which defines `is_bert_whitespace`, `is_bert_control`, `is_bert_punctuation`, `is_cjk`, and `WordPieceTokenizer` (`from_vocab`, `from_hf_json`, `normalize`, `basic_tokenize`, `wordpiece`, `encode`, `decode`, `save`, `load`). The contract is [`wordpiece.pyi`](../../../../course/contracts/py/tinyllm/tok/wordpiece.pyi), and the WordPiece part of the `tokenizer.json` subset is described in [`tokenizer.md`](../../../../course/contracts/formats/tokenizer.md).

> **Learning objectives**
> - Apply BERT's basic tokenizer by hand, in the right order: clean, space CJK, strip accents, lowercase, split.
> - Segment a word with greedy longest-match WordPiece, and explain why one unmatched position makes the whole word `[UNK]`.
> - Explain why special tokens must be matched on the raw text before normalization.
> - Decode WordPiece ids, gluing `##` pieces and applying the cleanup rules, and say what decoding cannot restore.

## The basic tokenizer

The notation for this section:

| Symbol | Meaning | Type / shape |
|---|---|---|
| $W$ | the vocabulary: word-start pieces and `##` continuation pieces | `dict[str, int]` |
| $x$ | one word after the basic tokenizer | `str` |
| $p$ | the continuation prefix, `##` | `str` |
| $L$ | `max_input_chars_per_word` (100 for BERT) | `int` |
| $\mathrm{Mn}$, $\mathrm{C*}$, $\mathrm{P*}$ | Unicode general categories: nonspacing mark, any control or format or unassigned, any punctuation | category |

The basic tokenizer is what Hugging Face calls `BertNormalizer` followed by `BertPreTokenizer`. It runs five steps, in this order, and the order is part of the definition.

1. **Clean.** Drop U+0000, U+FFFD, and every control character: category `C*`, except tab, newline, and carriage return, which count as white space. Then map every white-space character (the Unicode White_Space set) to U+0020. The order matters for U+000B: it is both white space and a control, and the control rule wins, so it vanishes. The zero-width space U+200B is category Cf and vanishes too.
2. **CJK.** Put a space on both sides of every CJK ideograph (the eight blocks in the contract, starting at U+4E00), so each becomes its own word. Chinese has no spaces, and BERT's vocabulary holds single ideographs.
3. **Strip accents** when `strip_accents` is true, or when it is `None` and `lowercase` is true (uncased BERT strips, cased BERT keeps): decompose with NFD, then drop category Mn. `é` becomes `e` + U+0301, and U+0301 is Mn. A spacing mark such as the Devanagari vowel sign U+093E (category Mc) is a vowel, not an accent, and stays.
4. **Lowercase** one code point at a time. Python's `str.lower()` on a whole string applies the Greek final-sigma rule (`"ΣΑΣ".lower()` ends in `ς`); per code point it is `σ`, which is what BERT's ids assume.
5. **Split** on white space, and make every punctuation character its own word. BERT's punctuation is category `P*` plus every ASCII character in 33 to 47, 58 to 64, 91 to 96, and 123 to 126, so `$`, `+`, `^` (symbols, category S) split too.

Special tokens are handled before any of this. They are split out of the raw text by leftmost-longest match through a trie, as written: `[MASK]` is the mask id, while `[mask]` is ordinary text (it would be lowercased and split into `[`, `mask`, `]`).

## Greedy longest match

Once text is split into words, WordPiece segments each word $x$:

1. if $x$ has more than $L$ code points, return `[UNK]` without searching;
2. at position 0, take the longest entry of $W$ that is a prefix of $x$;
3. at every later position $i$, take the longest entry $p\,y$ of $W$ such that $y$ is a prefix of $x[i:]$;
4. if some position has no match, the **whole word** is one `[UNK]`; the pieces found so far are discarded.

A `##` piece continues a word and can never start one. Two tries make each step a single walk: one over all entries for the word start, and one over the `##` entries stored without the prefix.

Be clear about what this rule is. Greedy longest match is not the best segmentation under any score; it commits to the longest prefix and never reconsiders. It is a fixed, cheap rule, and BERT's vocabulary was built for it, so it is the rule to reproduce.

## Decoding and templates

Decoding is the Hugging Face WordPiece decoder with cleanup. Every token after the first gets a leading space, unless it starts with `##`, in which case the prefix is removed instead. Then, inside each token, a fixed list of replacements runs: ` .` to `.`, ` ?` to `?`, ` !` to `!`, ` ,` to `,`, ` ' ` to `'`, ` n't` to `n't`, ` 'm` to `'m`, ` do not` to ` don't`, ` 's` to `'s`, ` 've` to `'ve`, ` 're` to `'re`. Decoding cannot restore case, accents, or white space, because the basic tokenizer removed them.

`add_special` is BERT's `TemplateProcessing`: `[CLS]` + ids + `[SEP]`. BERT reads `[CLS]`'s final vector as the sequence summary, and `[SEP]` ends a segment.

## Working it by hand

Use this 15-entry vocabulary:

| id | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| token | `[PAD]` | `[UNK]` | `[CLS]` | `[SEP]` | `[MASK]` | the | un | ##aff | ##able | ##a | ! | u | ##n | cafe | ##s |

First the basic tokenizer. `"The UNAFFABLE!"` lowercases to `"the unaffable!"`, splits on the space, and `!` is punctuation, giving `["the", "unaffable", "!"]`. `"日本x y"` gives `["日", "本", "x", "y"]`, each ideograph spaced. `"Café"` gives `["cafe"]`, because uncased BERT strips the accent.

Now WordPiece on `"unaffable"`. At position 0 the entries that are prefixes are `u` and `un`; the longest is `un` (6). At position 2, `"affable"`, the `##` entries whose rest is a prefix are `##a` and `##aff`; the longest is `##aff` (7). At position 5, `"able"`, it is `##able` (8). So the word is `[6, 7, 8]`. Had we taken the shortest match instead, `##a` at position 2 would leave `"ffable"`, which no `##` entry starts, and the whole word would be `[UNK]`.

Putting it together, `"The UNAFFABLE!"` encodes to `[5, 6, 7, 8, 10]`, and with `add_special=True` to `[2, 5, 6, 7, 8, 10, 3]`. `"cafés"` strips to `"cafes"`, which is `cafe` (13) then `##s` (14).

The whole-word rule shows up on `"unaffablex"`: it finds `un`, `##aff`, `##able`, then nothing for `x`, so the result is `[1]`, one `[UNK]`, not `[6, 7, 8, 1]`.

Finally decode `[5, 6, 7, 8, 10]`. The tokens are `the un ##aff ##able !`: `"the"`, then `" un"`, then `"aff"` and `"able"` glued on, then `" !"`, which cleanup turns into `"!"`. The result is `"the unaffable!"`.

## Designing the API

The interface exposes each stage separately so that each can be tested on its own:

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

`special_ids` are whichever of `[PAD] [UNK] [CLS] [SEP] [MASK]` the vocabulary has (from `added_tokens` when loading a `tokenizer.json`), and `unk_id` is the id of `unk_token`. `vocab.txt`, the original BERT release format, has one token per line, line $n$ being id $n$ from 0. `save` writes a `tokenizer.json` with `BertNormalizer`, `BertPreTokenizer`, `WordPiece`, the `[CLS] $A [SEP]` template, and the WordPiece decoder.

The model test suite, [`test_wordpiece.py`](../../../../course/solutions/learner-tests/L1.3/python/tests/l1-3-wordpiece/test_wordpiece.py), imports only names from the contract. Its names and docstrings list the rules most worth pinning:

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

## Building it step by step

Six observations shape the solution.

1. **The ids are defined by BERT's pipeline**, so the solution is a transcription of it, stage by stage, in Hugging Face's order: specials, clean, CJK, accents, lowercase, split, WordPiece. Each stage is a pure function of its input, which makes each one testable on its own (`normalize`, `basic_tokenize`, `wordpiece`).
2. **Specials must be found before normalization**, because normalization would lowercase `[MASK]` and split its brackets. So `encode` first cuts the raw text at leftmost-longest special matches.
3. **Character rules are per code point and order-sensitive.** Cleaning tests "control" before "white space" (U+000B is both), CJK spacing is applied during the same pass, accent stripping needs NFD first so that accents become separate marks, and lowercasing must be per code point to avoid the final-sigma rule.
4. **Splitting is a single scan.** White space ends a word; punctuation ends a word and is a word; everything else extends the current word.
5. **Greedy longest match is a loop of longest-prefix queries.** At position 0, query all entries; at later positions, query the `##` entries with the prefix removed, so one trie walk finds the longest continuation. The first failed query makes the whole word `[UNK]`, so no backtracking is needed.
6. **Decoding inverts only what can be inverted**: glue `##` pieces, add spaces between words, then apply the fixed cleanup replacements.

### Character classes

Each predicate encodes one BERT rule exactly.

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

### Normalization

One pass handles cleaning and CJK; then come the two whole-string transforms.

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

### Splitting

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

On the hand example, `"The UNAFFABLE!"` normalizes to `"the unaffable!"`; the scan emits `the` at the space, `unaffable` at `!`, and then `!` itself.

### Greedy longest match

The two tries are built in the constructor: `_word` holds every entry, and `_cont` holds each `##` entry with the prefix stripped.

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

### Encoding

Specials come first, on the raw text.

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

### Decoding

Glue, space, clean up.

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

On `[5, 6, 7, 8, 10]` this produces `the`, ` un`, `aff`, `able`, and ` !` cleaned to `!`, joined as `"the unaffable!"`.

The full solution is [`wordpiece.py`](../../../../course/solutions/python/tinyllm/tok/wordpiece.py), which also has `from_vocab`, `from_hf_json`, and `save`.

### Why this design

Normalization and splitting are $O(n)$ per text (plus NFD). WordPiece on a word of length $m$ is a sequence of trie walks; each walk can read ahead up to the longest entry, so the worst case is $O(m \cdot \ell_{\max})$, and in practice close to $O(m)$. The length limit caps pathological words at $L = 100$.

The alternatives are tempting and wrong for this purpose. A max-score segmentation (dynamic programming over a unigram score, as in [the Unigram tokenizer](04-unigram-lm.md)) would often find a better split, and different ids from BERT's. Hash-set lookups of every prefix length work but cost $O(m^2)$ string slices per word; the trie avoids building candidates at all. Python's `str.lower()` or `casefold()` on the whole string would be simpler and wrong for Greek. Linear-time Fast WordPiece is the production optimization, and it is not needed at this scale.

Three tradeoffs are accepted. There are no offsets: the tokenizer cannot map a piece back to its span in the original text, which BERT's masked-LM training does not need. Decoding is lossy by design, since case, accents, and white space were removed on the way in. And `token_to_id` reads the vocabulary directly, so it does not apply normalization.

## What can go wrong

A correct solution must reproduce the hand-worked words, encodings, template, and decode, and pass a set of boundary checks: `"unaffablex"` is `[1]`; `"una"` is `un ##a`; `"unn"` is `un ##n`, while `"aff"` alone is `[UNK]`; with a limit of 5, `"unaff"` segments and `"unaffa"` is refused (the limit is inclusive). U+0000, U+FFFD, U+000B, and U+200B vanish, while tab, U+00A0, and U+3000 become spaces. All four combinations of `lowercase` and `strip_accents` behave as described, `é` loses its accent while the Devanagari vowel sign stays, `$`, `+`, `^`, `«`, `»` are words of their own, `"ΣΑΣ"` is `"σασ"`, and `[MASK]` is id 4 while `[mask]` is text. The golden checks use bert-base-uncased on 300 strings: ids, `add_special` ids, decodes, and `skip_special` must all equal Hugging Face's, and `vocab.txt` must give the same ids as `tokenizer.json`. `load(save(t))` must keep ids, casing, and accent rules, and a vocabulary with no `[UNK]`, or with an id gap, is `ValueError`.

> **Warning:** Keeping the pieces found before an unmatched position makes `"unaffablex"` four ids where BERT has one, and sequence lengths drift away from what BERT saw in training. `wordpiece` returns `[self.unk_id]` at the first failed query.

The other pitfalls each break ids on real text:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| shortest match, or trying `##` pieces at the word start | words BERT segments become `[UNK]` | `longest_prefix` on `_word` at position 0 and on `_cont` only after it |
| stripping accents only when `strip_accents` is true | uncased BERT keeps `é`, and every accented word gets different ids | `strip` falls back to `self.lowercase` when `strip_accents` is `None` |
| `str.lower()` on the whole string | Greek words ending in sigma get different ids | lowercases one code point at a time |
| testing white space before control characters | U+000B becomes a space and splits a word that BERT keeps whole | the control test comes first in `normalize` |
| stripping every mark category | Hindi and other scripts lose vowel signs | drops only category `Mn` after NFD |
| `unicodedata` punctuation only | `$5` stays one word | `is_bert_punctuation` includes the four ASCII ranges |
| matching specials after normalization, or case-insensitively | `[mask]` typed by a user becomes the mask id | `encode` matches specials on the raw text through `_special_trie` |

## Where this leads

[Tokenizer metrics](06-tokenizer-metrics.md) measures bert-base-uncased through `from_hf_json` against Hugging Face's numbers, counting `[UNK]` as a fallback. Later, [BERT and masked-LM masking](../p06-objectives/02-bert-masked-lm.md) tokenizes with `WordPieceTokenizer` and writes `[MASK]` into its inputs, and [ELECTRA](../p06-objectives/03-electra-replaced-token-detection.md) trains on the same ids.

Each piece has a production counterpart:

- `basic_tokenize` corresponds to Hugging Face's `BertNormalizer` and `BertPreTokenizer`, which add offsets for every piece so predictions map back to the original text (`tokenizers/src/normalizers/bert.rs`).
- `wordpiece` corresponds to the `tokenizers` `WordPiece` model: the same greedy loop in Rust, plus a trainer that derives WordPieces from BPE merges (`tokenizers/src/models/wordpiece/mod.rs`).
- The `[CLS] ... [SEP]` template corresponds to `TemplateProcessing`, which adds pair templates with token type ids for two-segment inputs (`tokenizers/src/processors/template.rs`).
- Greedy longest match has a linear-time version in Fast WordPiece, which uses an Aho-Corasick-style trie (Song et al., *Fast WordPiece Tokenization*, 2021).

### Further reading

- Devlin et al., *BERT* (2019), section 3.
- Wu et al., *Google's Neural Machine Translation System* (2016), section 4.1 (wordpieces).
- Schuster and Nakajima, *Japanese and Korean Voice Search* (2012).

## Key terms

- **Basic tokenizer**: BERT's normalizer and pre-tokenizer: clean, space CJK, strip accents, lowercase, split on white space and punctuation.
- **Continuation piece**: a vocabulary entry prefixed with `##` that may only follow another piece within a word.
- **Greedy longest match**: segmenting a word by repeatedly taking the longest vocabulary entry that matches at the current position.
- **Whole-word `[UNK]`**: the rule that a word with any unmatched position becomes a single unknown id.
- **Nonspacing mark (Mn)**: the Unicode category that accent stripping removes after NFD; spacing marks (Mc) stay.
- **Template**: the `[CLS] ... [SEP]` wrapping `add_special` applies.

## Summary

- BERT's tokenizer is two stages: a basic tokenizer that cleans, lowercases, strips accents, and splits on white space and punctuation, then greedy longest-match WordPiece inside each word.
- Greedy longest match is not optimal segmentation: it commits to the longest prefix, and a word with any unmatched position becomes one whole `[UNK]`.
- A `##` piece continues a word and can never start one; the decoder glues it back without a space.
- Every normalization detail changes ids on real text: controls vanish before white space is mapped, only nonspacing marks are stripped, and lowercasing is per code point.
- With those rules exact, the tokenizer reproduces bert-base-uncased's ids, template, and decodes on 300 hard strings.

## Review questions

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
