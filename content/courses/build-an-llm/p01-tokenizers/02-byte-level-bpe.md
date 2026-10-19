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

`added_tokens` are `(content, id, special)`; `special_ids` holds the special ones, `unk_id` is `None`, and `vocab_size` is the largest id plus one, with no gap allowed. `save` writes `tokenizer.json` in the subset (Hugging Face loads it), and `load` reads it back through `from_hf_json`.

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the hand-worked training | unit | the section 3 merges `ug`, `hug`, `pug` with ids 256 to 258, and the tie rule | everyone agrees on the definition |
| the hand-worked encode | unit | `"hugs pug"` is `[257, 82, 220, 258]` and decodes back | encoding replays training's merges |
| the hand-worked pre-tokens | unit | the two section 3 strings | every id depends on the pre-tokens |
| `min_freq` stops training | boundary | `min_freq` 2, 3, 4 give 2, 2, 1 merges | a count equal to `min_freq` still merges |
| merges match Hugging Face's trainer | golden | five configs on a 3 KB story give Hugging Face's merges in order | the `C1` vocabulary is the standard algorithm |
| specials are not trained on | boundary | `<\|endoftext\|>` is id 0 and never part of a merge | document separators stay one token |
| bad training arguments | boundary | `vocab_size` below 256 plus specials, `min_freq` 0 | no silent empty training |
| white-space rules of the pre-tokenizer | boundary | U+001C is not white space, contractions are lowercase only, a tab never leads a word | ids match on real text |
| pre-tokenizer properties | property | 300 random texts: pieces join back, none empty, the `(?!\S)` rule | `decode(encode(x)) == x` |
| `split_digits` | unit | individual and run modes of `Digits` | SmolLM2 spells numbers digit by digit |
| GPT-2 ids match the oracle | golden | GPT-2's `tokenizer.json`, 300 strings, ids equal Hugging Face's and tiktoken's | the system can serve GPT-2-tokenized models |
| GPT-2 release files load the same | golden | `from_gpt2` on `vocab.json` and `merges.txt` (derived from the `tokenizer.json`) gives the same ids | the original release format |
| SmolLM2 ids match the oracle | golden | SmolLM2's file, the Digits sequence, 17 added tokens, 300 strings | `L7.9` and the agents' model |
| SmolLM2 drops unrepresentable bytes | boundary | `"a\x04b"` encodes to `a`, `b` | ids match Hugging Face on control bytes |
| decode matches the oracle | golden | decode of every oracle case, and `skip_special` | streamed text equals Hugging Face's |
| decode joins bytes before UTF-8 | boundary | the three bytes of `日` decode together, one alone is U+FFFD | `L8.2` detokenizes streams |
| added tokens split first, leftmost longest | unit | `<\|im_start\|>` is id 1, `<\|im` is text, GPT-2's end-of-text is 50256 | chat templates |
| merges apply by rank | unit | swapping two ranks changes `"abc"` | the definition of BPE encoding |
| rank ties go leftmost | boundary | `"aaa"` is `aa a`, `"aaaa"` is `aa aa` | identical ids in Rust (`L1.5`) |
| round trip | property | a trained tokenizer round-trips 200 random texts | the 256 symbols are a fallback |
| save and load round trip | property | `load(save(t))` has the same merges, specials, and ids | the engine loads a trained vocabulary |
| fields outside the subset | boundary | a normalizer, dropout, an unknown token, byte fallback, a template: each `ValueError` | never wrong ids without an error |
| `encode_batch` equals `encode` | property | `encode_batch` equals `encode` per text, empty texts included | the corpus pipeline's call |
| ids must be contiguous | boundary | a vocabulary with a gap is `ValueError` | every id below `vocab_size` decodes |

### The model tests

The model suite, [`test_bpe.py`](../../../../course/solutions/learner-tests/L1.2/python/tests/l1-2-bpe/test_bpe.py), imports only names from the two contracts. Its names and docstrings are a compact summary of the definition:

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

## 5. The solution, step by step

### From first principles

1. **Ids must match the real models exactly**, so every stage must be a faithful transcription: the regex alternatives in order, White_Space rather than `isspace`, the byte map, and merges applied by rank. Any approximation produces plausible but different ids.
2. **A regex with backtracking can be written as a scanner.** Each alternative of the GPT-2 regex matches a run of one character class, optionally preceded by one space; the only lookahead is `(?!\S)`, which amounts to "a white-space run gives back its last character when a non-space follows". A hand-written loop over `unicodedata` reproduces this without a regex engine that supports `\p{L}`.
3. **Training counts pairs inside pre-tokens, weighted by frequency.** Merging is defined per pre-token, so the natural data structure is a counter of distinct pre-tokens, each a list of ids, with its count. Real corpora have far fewer distinct pre-tokens than occurrences.
4. **Each merge changes only a few counts.** Replacing $a, b$ by $ab$ in a word with neighbours $x$ and $y$ removes $(x, a)$ and $(b, y)$ and adds $(x, ab)$ and $(ab, y)$. So maintain counts incrementally and keep a map from pair to the words containing it; then a merge costs only the words it touches.
5. **The most frequent pair needs a priority queue, and counts change under it.** A max-heap keyed by `(count, left id, right id)` gives the tie rule for free. Instead of updating entries in place, pop, compare with the current count, and requeue if stale: lazy deletion.
6. **Encoding must replay training's priorities, not its procedure.** Training merged the earliest-learned pair first everywhere, so at encode time, inside one pre-token, the pair with the lowest rank must merge first, leftmost on a tie. Merging left to right gives different ids.
7. **Added tokens are matched before anything else**, because the pre-tokenizer would split them. Leftmost-longest through a trie handles overlapping specials (`<|im` versus `<|im_start|>`).
8. **Decoding must work on bytes**, because a token can hold part of a UTF-8 character. Join all bytes, then decode once with replacement.

### The code, annotated

**The pre-tokenizer.** The loop tries alternatives in the regex's order at each position.

```python
WHITE_SPACE = frozenset(
    "\t\n\x0b\x0c\r \x85\xa0 "
    + "".join(chr(c) for c in range(0x2000, 0x200B))
    + "    　"
)   # Unicode White_Space: not str.isspace(), which also accepts U+001C to U+001F

def pretokenize_gpt2(text: str) -> list[str]:
    out: list[str] = []
    n = len(text)
    i = 0
    while i < n:
        ch = text[i]
        # 1. contractions, lowercase only: 's 't 're 've 'm 'll 'd
        if ch == "'":
            suf = next((s for s in CONTRACTIONS if text.startswith(s, i + 1)), None)
            if suf is not None:
                out.append(text[i : i + 1 + len(suf)])
                i += 1 + len(suf)
                continue
        # 2 to 4. an optional single U+0020, then a run of one class (L, N, or O)
        j = i + 1 if ch == " " and i + 1 < n else i
        k = _class(text[j])                 # "L", "N", "S" (white space), or "O"
        if k != "S":
            end = j + 1
            while end < n and _class(text[end]) == k:
                end += 1
            out.append(text[i:end])
            i = end
            continue
        # 5. \s+(?!\S): a white-space run, minus its last character when a
        #    non-space follows (that character starts the next pre-token).
        end = i
        while end < n and is_space(text[end]):
            end += 1
        if end < n and end - i >= 2:
            end -= 1
        # 6. \s+: a single white-space character before a non-space.
        out.append(text[i:end])
        i = end
    return out
```

Trace section 3's three spaces before `here`: at the first space, `text[j]` is the second space, class `S`, so alternatives 2 to 4 fail. The run is three long and `h` follows, so it gives back one: the piece is two spaces. The third space then starts ` here` through alternative 2. Only the U+0020 space is optional in 2 to 4 (`ch == " "`), which is why a tab never leads a word.

**The merge update.** `_merge_word` rewrites one word and reports exactly the count changes around each occurrence.

```python
def _merge_word(w: list[int], a: int, b: int, new: int) -> list[tuple[tuple[int, int], int]]:
    """Merge every non-overlapping (a, b) in w, left to right, in place.
    Returns the pair count changes the merge causes around each occurrence
    (Hugging Face `Word::merge`): the merged pair itself is not decremented."""
    changes: list[tuple[tuple[int, int], int]] = []
    i = 0
    while i < len(w):
        if w[i] == a and i + 1 < len(w) and w[i + 1] == b:
            if i > 0:
                changes.append(((w[i - 1], a), -1))     # left neighbour loses (x, a)
                changes.append(((w[i - 1], new), 1))    # and gains (x, ab)
            w[i : i + 2] = [new]
            if i < len(w) - 1:
                changes.append(((b, w[i + 1]), -1))     # right neighbour loses (b, y)
                changes.append(((new, w[i + 1]), 1))    # and gains (ab, y)
        i += 1          # after a merge, w[i] is `new`, so overlapping (a, b) cannot re-match
    return changes
```

The merged pair is not decremented because the trainer never looks at it again: it was just popped, and its count is irrelevant once its token exists.

**The trainer.** Word counts, initial ids, pair counts, then the lazy heap loop.

```python
    @classmethod
    def train(cls, texts, vocab_size, specials=(), min_freq=2) -> "BPETokenizer":
        specials = list(dict.fromkeys(specials))          # dedupe, keep order
        # ... ValueError for vocab_size < 256 + len(specials) or min_freq < 1
        b2u = bytes_to_unicode()
        trie = Trie()
        for k, s in enumerate(specials):
            trie.insert(s, k)
        # 1. word counts: pre-tokens in the byte alphabet, specials removed
        wc: Counter[str] = Counter()
        for text in texts:
            for seg, sid in _split_added(trie, text) if specials else [(text, None)]:
                if sid is None:                            # special text is cut out
                    for p in pretokenize_gpt2(seg):
                        wc["".join(b2u[b] for b in p.encode("utf-8"))] += 1
        # 2. the initial vocabulary: specials, then the 256 symbols by code point
        id_to_tok = list(specials) + sorted(b2u.values(), key=ord)
        tok_to_id = {t: i for i, t in enumerate(id_to_tok)}
        words = [[tok_to_id[c] for c in w] for w in wc]   # one entry per distinct pre-token
        counts = list(wc.values())                         # f(w)
        # 3. pair counts and where each pair occurs
        pair_counts: dict[tuple[int, int], int] = {}
        where: dict[tuple[int, int], set[int]] = {}
        for wi, w in enumerate(words):
            for pair in zip(w, w[1:]):
                pair_counts[pair] = pair_counts.get(pair, 0) + counts[wi]
                where.setdefault(pair, set()).add(wi)
        # heap entries: [-count, left id, right id, seq, words]: a min-heap on this
        # key pops the largest count, then the smallest (left, right) ids
        # ... build and heapify
        merges: list[tuple[str, str]] = []
        # 4. merge the most frequent pair until the vocabulary is full
        while len(id_to_tok) < vocab_size and heap:
            top = heapq.heappop(heap)
            pair = (top[1], top[2])
            cur = pair_counts.get(pair, 0)
            if -top[0] != cur:  # stale: requeue with the current count
                top[0] = -cur
                heapq.heappush(heap, top)
                continue
            if cur < 1 or cur < min_freq:                  # equal to min_freq still merges
                break
            a, b = id_to_tok[pair[0]], id_to_tok[pair[1]]
            new_tok = a + b
            # ... new_id: the existing id if new_tok is already a token, else the next id
            merges.append((a, b))
            update: dict[tuple[int, int], set[int]] = {}
            for wi in sorted(top[4]):                      # only the words that contain the pair
                for p, d in _merge_word(words[wi], pair[0], pair[1], new_id):
                    pair_counts[p] = pair_counts.get(p, 0) + d * counts[wi]
                    if d > 0:
                        update.setdefault(p, set()).add(wi)
            for p, pos in update.items():                  # new pairs enter the heap
                if pair_counts[p] > 0:
                    heapq.heappush(heap, [-pair_counts[p], p[0], p[1], seq, pos])
                    seq += 1
        # ... vocab from id_to_tok; specials become added tokens with ids 0, 1, ...
```

On section 3: `wc` is `{hug: 2, pug: 1, hugs: 1}`. The heap's top is `(u, g)` with count 4. Merging it touches all three words and pushes `(h, ug)` with 3, `(p, ug)` with 1, `(ug, s)` with 1. At step 3, `(p, ug)` is `[-1, 79, 256, ...]` and `(hug, s)` is `[-1, 257, 82, ...]`; the heap pops the smaller list, `(p, ug)`.

A popped entry whose count went down (a neighbour merged away part of it) is pushed back with its real count. An entry whose count is current is the true maximum: a pair whose count fell still has its old, higher entry, which pops first and gets corrected, and a pair whose count rose was pushed fresh in `update`.

**Encoding one pre-token.** The rank loop, with the cache and the dropped-byte rule.

```python
    def _bpe(self, word: str) -> list[int]:
        hit = self._cache.get(word)
        if hit is not None:
            return hit
        # A byte symbol the vocab lacks is dropped, as Hugging Face does when
        # unk_token is null (SmolLM2 has no symbol for six control bytes).
        syms = [c for c in word if c in self.vocab]
        while len(syms) > 1:
            best_rank, best_k = None, -1
            for k in range(len(syms) - 1):
                r = self.ranks.get((syms[k], syms[k + 1]))
                # strict <: on equal ranks the first (leftmost) k is kept
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_k = r, k
            if best_rank is None:
                break                          # no ranked pair left
            syms[best_k : best_k + 2] = [syms[best_k] + syms[best_k + 1]]
        ids = [self.vocab[s] for s in syms]    # (a KeyError here means a broken file: ValueError)
        self._cache[word] = ids
        return ids

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        ids: list[int] = []
        segs = _split_added(self._trie, text) if self.added else [(text, None)]
        for seg, aid in segs:
            if aid is not None:
                ids.append(aid)                # an added token: one id, no BPE
                continue
            for p in self._pieces(seg):        # the declared steps: Digits, then ByteLevel
                ids += self._bpe("".join(self._b2u[b] for b in p.encode("utf-8")))
        return ids
```

`self.ranks` is built with `reversed(list(enumerate(merges)))` so that if a file lists the same merge twice, the first (lowest) rank wins.

**Decoding.** Bytes first, UTF-8 once.

```python
    def decode(self, ids: Sequence[int], skip_special: bool = False) -> str:
        buf = bytearray()
        for i in check_ids(ids, self.vocab_size):
            if i in self._added_by_id:
                content = self._added_by_id[i]
                if not (skip_special and content in self.special_ids):
                    buf += content.encode("utf-8")
            else:
                buf += bytes(self._u2b[c] for c in self._id_to_tok[i])   # symbol -> byte (M05.2)
        return buf.decode("utf-8", "replace")
```

**Loading `tokenizer.json`.** The loader is a list of refusals followed by one constructor call.

```python
    @classmethod
    def from_hf_json(cls, tokenizer_json: str) -> "BPETokenizer":
        # ... json.load; version must be "1.0"
        if doc.get("normalizer") is not None:
            raise ValueError("normalizer: a byte-level BPE file has none (null)")
        model = doc.get("model") or {}
        # Older files (GPT-2's) omit model.type; a BPE model is the one with merges.
        mtype = model.get("type", "BPE" if "merges" in model else None)
        # ... must be "BPE"
        for key, ok in (
            ("dropout", (None,)),
            ("unk_token", (None,)),
            ("continuing_subword_prefix", (None, "")),
            ("end_of_word_suffix", (None, "")),
            ("fuse_unk", (False, None)),
            ("byte_fallback", (False, None)),
            ("ignore_merges", (False, None)),
        ):
            if model.get(key) not in ok:
                raise ValueError(f"model.{key}: {model.get(key)!r} is outside the subset")
        # ... merges as "a b" strings or [a, b] lists; added tokens with single_word,
        #     lstrip, rstrip all false; post_processor and decoder ByteLevel or null
        return cls(model["vocab"], merges, added, doc.get("pre_tokenizer"))
```

The constructor then validates the pre-tokenizer (`_check_pre_tokenizer`: `Digits` and `ByteLevel` only, `ByteLevel` last with `use_regex` true) and that ids are contiguous. `save` writes the same subset, so `load` is `from_hf_json` on the saved file.

Full solution: [`pretok.py`](../../../../course/solutions/python/tinyllm/tok/pretok.py) (also `split_digits`) · [`bpe.py`](../../../../course/solutions/python/tinyllm/tok/bpe.py) (also `from_gpt2`, `_pieces`, `_split_added`, `save`).

### Why this solution

- **Cost.** Training is $O(N)$ to count pre-tokens over $N$ characters, then per merge $O(\text{words touched} \cdot \text{word length} + \log H)$ for a heap of $H$ entries, instead of the naive $O(\text{all words})$ recount per merge. Encoding a pre-token of $L$ symbols is $O(L^2)$ in the worst case (each merge rescans), but pre-tokens are short and the cache makes repeated words $O(1)$.
- **Alternatives.** The `regex` module would handle `\p{L}` directly, but it adds a dependency and hides the `(?!\S)` backtracking the scanner makes explicit; the Rust port (`L1.5`) has the same problem, and a scanner ports line for line. An eager heap with decrease-key needs an indexed heap; the lazy version is simpler and amortizes to the same cost. A rank-free encoder that merges greedily left to right is faster to write and wrong. Ignoring unknown fields in `tokenizer.json` would load more files and silently produce wrong ids for them.
- **Accepted tradeoffs.** The cache is unbounded (fine for a training or evaluation process, not for a long-lived server; `L1.5` bounds it in Rust). `add_special` is accepted for protocol compatibility but adds nothing, since the subset has no template post-processor.

### Explain it back

1. Why does the scanner treat only U+0020 as the optional leading space, and not any white space?
   <details><summary>Answer</summary>
   The regex alternatives 2 to 4 begin with ` ?`, a literal space, not `\s?`. A tab or newline before a word is therefore its own pre-token (alternative 5 or 6), and the word starts without it. Accepting any white space would merge `\tword` into one pre-token and change the ids.
   </details>
2. In `_merge_word`, why is the change for the merged pair $(a, b)$ itself not recorded?
   <details><summary>Answer</summary>
   That pair was just popped from the heap and turned into a token; the trainer never needs its count again. The changes that matter are the neighbouring pairs whose counts move, $(x, a) \to (x, ab)$ and $(b, y) \to (ab, y)$. This matches Hugging Face's `Word::merge`, which is necessary for the merges to agree one for one.
   </details>
3. Why is the heap key `(-count, left id, right id)`, and what would break with `(-count, right id, left id)`?
   <details><summary>Answer</summary>
   Python's heap is a min-heap, so `-count` pops the largest count first, and the next two fields break ties by the smallest ids compared as (left, right), which is the Hugging Face rule. Swapping them would choose a different pair on ties; in section 3, `(hug, s)` with right id 82 would beat `(p, ug)` with right id 256, and every later merge would diverge.
   </details>
4. Why must encoding merge by lowest rank and not left to right?
   <details><summary>Answer</summary>
   Training applied merges in rank order across the whole corpus: by the time a later merge was learned, every earlier one had already been applied everywhere. Replaying by rank reproduces the segmentation training produced. Left to right gives `ab c` for `"abc"` when `(b, c)` was learned first, which training would never produce.
   </details>
5. What goes wrong if `decode` converted each token to text separately and joined the strings?
   <details><summary>Answer</summary>
   A byte-level token can end in the middle of a multi-byte character. Decoding its bytes alone yields U+FFFD, and so does the next token's half, so every character split across tokens is replaced. Joining all bytes first and decoding once gives the original character.
   </details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. merging left to right, or the rightmost of equal ranks | ids differ from GPT-2's on most words longer than three bytes | `_bpe` picks the minimum rank with strict `<`, so the leftmost wins ties |
| 2. `str.isspace` or `\s` of another regex flavor; a white-space run taken whole before a word | pre-tokens differ on control characters and runs of spaces; ` here` loses its space | `WHITE_SPACE` is the Unicode property; the run gives back its last character before a non-space |
| 3. decoding token by token | every character split across tokens becomes U+FFFD | `decode` builds one `bytearray` and decodes it once |
| 4. ignoring a field the loader does not implement | a normalizer or the declared Digits step is skipped and ids are silently wrong | `from_hf_json` raises for every field outside the subset; `_pieces` runs the declared steps |
| 5. training on special-token text | merges such as `<\|` and `endoftext` waste the vocabulary | `_split_added` cuts specials out before pre-tokenizing |
| 6. treating a byte the vocabulary lacks as an error | SmolLM2 raises on a control byte where Hugging Face drops it | `_bpe` filters symbols not in the vocabulary |
| 7. ties broken by the largest ids, or counting each pre-token once | merges diverge from Hugging Face's after a few steps | heap key `(-count, left, right)`; every count is weighted by `counts[wi]` |
| 8. Latin-1 instead of the byte map | `é` and every non-ASCII text get the wrong symbols | bytes go through `bytes_to_unicode()` in training and encoding |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L1.1` | the protocol, `check_ids` in `decode` and `id_to_token` |
| Back | `M05.2` | `bytes_to_unicode` turns pre-token bytes into symbols; `unicode_to_bytes` decodes |
| Back | `M06.2` | `Trie.longest_prefix` splits out added tokens, leftmost longest |
| Forward | `L1.5` | the Rust `tl-tok` reimplements encode and decode; its differential tests compare its ids with this solution's on the fixtures and a TinyStories sample |
| Forward | `L1.6` | measures GPT-2 and SmolLM2, loaded by `from_hf_json`, against Hugging Face's numbers |
| Forward | `data.07` | tokenizes corpus files through the Rust tokenizer process and shared file fixtures |
| Forward | `L8.2` | `IncrementalDecoder` streams byte-level ids through `Tokenizer.decode`, holding back text that ends in a partial UTF-8 character |
| Forward | `L14.4` | Whisper decoding consumes this implementation. |

Later batches add call sites: `L7.9` and the engine load SmolLM2 with it, `L8.2` detokenizes streams, and `C1` trains its 4096-token vocabulary with this trainer.

## Going further

| Piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `train` with a lazy heap | Hugging Face `BpeTrainer` | parallel word counting, the same heap with a `Word::merge` delta, progress bars | `tokenizers/src/models/bpe/trainer.rs` |
| `encode` by rank | tiktoken | a rank-only encoder over a byte-pair map, written in Rust, with a regex pre-split | `tiktoken/src/lib.rs` (`byte_pair_merge`) |
| `pretokenize_gpt2` | the `fancy-regex` / `onig` pre-tokenizer | the real regex engine, plus `cl100k`-style splits (numbers of 1 to 3 digits) | `tokenizers/src/pre_tokenizers/byte_level.rs` |
| `from_hf_json` | `tokenizers` `Tokenizer::from_file` | the full pipeline (normalizers, offsets, truncation, padding) | `tokenizers/src/tokenizer/serialization.rs` |
| merge cache per pre-token | `tokenizers` BPE cache | a bounded cache shared across threads; `L1.5` builds the Rust version | `tokenizers/src/utils/cache.rs` |

