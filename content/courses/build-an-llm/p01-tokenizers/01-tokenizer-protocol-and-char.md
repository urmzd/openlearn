# Tokenizer protocol, char tokenizer

## Overview

| | |
|---|---|
| **Module** | `L1.1` · build · Python · Pass 3 · 2 to 3 h |
| **Solution** | [`base.py`](../../../../course/solutions/python/tinyllm/tok/base.py): the `Tokenizer` protocol, `check_ids`, `ByteTokenizer` · [`char.py`](../../../../course/solutions/python/tinyllm/tok/char.py): `CharTokenizer` (`train`, `encode`, `decode`, `save`, `load`) · model tests: [`test_tokenizers.py`](../../../../course/solutions/learner-tests/L1.1/python/tests/l1-1-tokenizers/test_tokenizers.py) |
| **Contract** | [`course/contracts/py/tinyllm/tok/base.pyi`](../../../../course/contracts/py/tinyllm/tok/base.pyi) · [`course/contracts/py/tinyllm/tok/char.pyi`](../../../../course/contracts/py/tinyllm/tok/char.pyi) · formats: [`tokenizer.md`](../../../../course/contracts/formats/tokenizer.md), [`tokenizer-char.schema.json`](../../../../course/contracts/formats/tokenizer-char.schema.json) |
| **Needs** | `M05.2` the GPT-2 byte map (`bytes_to_unicode`, `unicode_to_bytes`) · reading: `L0.0` the tracer's byte ids |
| **Used by** | `L1.2` byte-level BPE, `L1.3` WordPiece, and `L1.4` Unigram implement the protocol and call `check_ids` · `L1.6` measures any `Tokenizer` · later: `L1.5` the Rust tokenizer behind the same protocol, `L3.6` char-level language models. With `L1.2` to `L1.6`, this completes the tokenizer layer of Pass 3 |
| **Optional depth** | The Unicode Standard, ch. 2 "General Structure" and ch. 3 section 3.9 (UTF-8); Jurafsky and Martin, *Speech and Language Processing* (3rd ed. draft), ch. 2 "Regular Expressions, Tokenization, Edit Distance" |

## Key Takeaways

- A tokenizer is a contract, not an algorithm: every tokenizer in the system (bytes, char, BPE, WordPiece, Unigram, and later Rust) has the same six members, so everything that consumes tokens is written once.
- The byte tokenizer of the tracer is already a tokenizer: 256 ids, no training, every text round-trips, and its token strings are the GPT-2 byte map, the alphabet BPE starts from.
- A char tokenizer's ids must not depend on the order of the training texts: `<unk>` is 0, the specials follow, then code points sorted by code point.
- Unknown input is data, not an exception: an unseen character becomes `<unk>`, and special-token texts typed by a user stay ordinary characters.
- Decoding never trusts ids and loading never trusts files: `-1`, `vocab_size`, `True`, and a malformed `tinyllm_char.json` are all `ValueError`.

## How to read this chapter

Sections 1 to 3 build the protocol and the char tokenizer's vocabulary rule from first principles and work them by hand. Section 4 fixes the interface and lists what a correct solution must handle. Section 5 walks through `base.py` and `char.py` and explains why each choice is the right one. Try to predict each step of the solution before reading it.

---

## 1. Why now

The tracer speaks bytes. `L0.0` trained on raw UTF-8 bytes and `L10.0` serves `tl_tokenizer = "bytes"`, so `é` costs two tokens and the model spends capacity learning how to spell UTF-8. Pass 3 replaces that with trained vocabularies: byte-level BPE (`L1.2`), WordPiece (`L1.3`), and Unigram (`L1.4`), compared by `L1.6` and chosen for the capstone. Four tokenizers with four call conventions would mean four versions of the corpus pipeline, the metrics, and the engine's tokenizer factory. This module fixes the shape first: one `Tokenizer` protocol, the existing byte tokenizer written against it, and the simplest trained tokenizer, one id per character, which also becomes the vocabulary of the char-level models in `L3.6`.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $\Sigma$ | the alphabet of Unicode code points, `U+0000` to `U+10FFFF` | set |
| $t \in \Sigma^*$ | a text: a finite sequence of code points | `str` |
| $V$ | vocabulary size: ids are $0, \dots, V - 1$ | `int` |
| $\mathrm{enc}: \Sigma^* \to \{0..V-1\}^*$ | encode: text to ids | `str -> list[int]` |
| $\mathrm{dec}: \{0..V-1\}^* \to \Sigma^*$ | decode: ids to text | `list[int] -> str` |
| $u$ | the unknown id (`unk_id`), or none when every text encodes | `int` or `None` |
| $S$ | the special ids: `<unk>`, `<bos>`, `<eos>`, `<pad>` | `dict[str, int]` |
| $\beta: \{0..255\} \to \Sigma$ | the GPT-2 byte map of `M05.2` (byte 0x20 is `Ġ`) | bijection |

**Text is code points, storage is bytes.** A Python `str` is a sequence of code points; `len("é") == 1`. On disk and on the wire it is UTF-8: one to four bytes per code point, `é` (U+00E9) is `c3 a9`, an emoji is four bytes. Two texts that render the same may be different sequences: `é` can be the single code point U+00E9 or `e` followed by the combining acute accent U+0301. A tokenizer that silently normalizes one into the other can never give the original text back.

**The protocol.** A tokenizer is anything with these members (a `typing.Protocol`, so no inheritance is needed):

- `vocab_size`: ids live in $[0, V)$, and every id in that range decodes (no gaps).
- `special_ids`: the special tokens' texts mapped to their ids.
- `unk_id`: where unknown input goes, or `None` when every text encodes (bytes, byte-level BPE).
- `encode(text, add_special=False)` and `decode(ids, skip_special=False)`.
- `token_to_id`, `id_to_token`: the vocabulary as a two-way table.
- `save(dir)` and the class method `load(dir)`: a tokenizer lives in a model directory.

The two laws are $\mathrm{dec}(\mathrm{enc}(t)) = t$ for every text the vocabulary covers, and $\mathrm{enc}$ never raises on a text: unknown input becomes $u$. `add_special` adds a template around the ids (BOS and EOS here, `[CLS]` and `[SEP]` for WordPiece) and never changes the ids of the text itself. `skip_special` drops the ids in $S$ when decoding.

**Ids are checked once, in one place.** Every `decode` starts with `check_ids(ids, V)`: each id must be an integer (`hasattr(i, "__index__")`, so numpy integers pass) and not a `bool` (in Python `True == 1`, and a mask passed by mistake would decode as text), and $0 \le i < V$. The range check matters more than it looks: `vocab[-1]` is valid Python and returns the last entry, so without it a model that emits `-1` decodes to a plausible character.

**Bytes as a tokenizer.** `ByteTokenizer` is the tracer's tokenizer (D32) under the protocol: $V = 256$, $\mathrm{enc}(t)$ is the UTF-8 bytes of $t$, $\mathrm{dec}$ is UTF-8 decoding **with replacement**, so a byte sequence that is not valid UTF-8 becomes U+FFFD instead of an exception. A sampled model can emit any bytes, and the server must still answer. Its token strings are $\beta(i)$, the M05.2 byte map: printable code points for every byte, so `id_to_token(32)` is `Ġ` and `id_to_token(10)` is `Ċ`. That is exactly the alphabet byte-level BPE (`L1.2`) starts from, so "bytes" is BPE with zero merges.

**The char tokenizer.** One id per code point. Training only collects the code points that occur, and the vocabulary is fixed by a rule that does not depend on the order of the texts:

1. id 0 is `<unk>`, always;
2. then the requested specials, each one of `<bos>`, `<eos>`, `<pad>`, in the order given, a repeat counted once;
3. then every code point of the training texts, sorted by code point.

Encoding maps each code point to its id, or to $u = 0$ when it was never seen. The texts of the specials are not matched in the input: if a user could type `<eos>` and get the end-of-sequence id, any prompt could stop generation. `add_special` puts `<bos>` first and `<eos>` last when those roles exist.

**The file.** `save(dir)` writes `dir/tinyllm_char.json` ([`tokenizer.md`](../../../../course/contracts/formats/tokenizer.md), the char tokenizer): `{"type": "char", "vocab": [...], "specials": {"unk": 0, ...}}`. `specials` maps **roles** to ids, not texts to ids, so a reader in any language finds `<unk>` without guessing its spelling. `load` checks everything the constructor checks: unique entries, `unk` present, every role one of the four, every special id inside the vocabulary, every other entry exactly one code point.

## 3. Worked example by hand

**Train.** `CharTokenizer.train(["hello", "hi"], specials=("<bos>", "<eos>"))`. The code points seen are `h e l o i`. Sorted by code point: `e` (101), `h` (104), `i` (105), `l` (108), `o` (111). With `<unk>` first and the two specials next:

| id | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| text | `<unk>` | `<bos>` | `<eos>` | e | h | i | l | o |

Training on `["hi", "hello"]` gives the same table: the order of the texts does not matter. First-appearance order would have given `h` id 3 and `e` id 4, and a tokenizer retrained on shuffled data would silently disagree with every saved model.

**Encode.** `"hello"` is h e l l o, so `[4, 3, 6, 6, 7]`. `"hi!"` is `[4, 5, 0]`: `!` was never seen, so it is `<unk>`. With `add_special=True`, `"hi"` is `[1, 4, 5, 2]`.

**Decode.** `[4, 5, 0]` decodes to `"hi<unk>"` (the special's text), and with `skip_special=True` to `"hi"`. `[-1]` and `[8]` are `ValueError`.

**Bytes.** `ByteTokenizer().encode("hé")`: `h` is 0x68 = 104, `é` is U+00E9, two UTF-8 bytes `c3 a9` = 195, 169. So `[104, 195, 169]`. Decoding `[195]` alone is an incomplete sequence: one U+FFFD, `"�"`. Decoding `[226, 130]` (the first two bytes of a three-byte character) is also one `"�"`: one maximal ill-formed subpart, one replacement character.

These are the first cases in section 4: `test_hand_example_vocab`, `test_hand_example_encode_decode`, and `test_hand_example_bytes`.

## 4. The interface

```python
# python/tinyllm/tok/base.py
@runtime_checkable
class Tokenizer(Protocol):
    vocab_size: int
    special_ids: dict[str, int]
    unk_id: Optional[int]
    def encode(self, text: str, add_special: bool = False) -> list[int]: ...
    def decode(self, ids: Sequence[int], skip_special: bool = False) -> str: ...
    def token_to_id(self, s: str) -> Optional[int]: ...
    def id_to_token(self, i: int) -> str: ...
    def save(self, dir: str) -> None: ...
    @classmethod
    def load(cls, dir: str) -> "Tokenizer": ...

def check_ids(ids: Sequence[int], vocab_size: int) -> list[int]
class ByteTokenizer: ...            # vocab_size 256, special_ids {}, unk_id None

# python/tinyllm/tok/char.py
class CharTokenizer(Tokenizer):
    vocab: list[str]; roles: dict[str, int]
    def __init__(self, vocab: Sequence[str], specials: Mapping[str, int]) -> None
    @classmethod
    def train(cls, texts: Iterable[str], specials: Sequence[str] = ()) -> "CharTokenizer"
```

The contracts carry the exact rules: which inputs are `ValueError`, that `ByteTokenizer.save` creates the directory and writes nothing (its `config.json` says `tl_tokenizer = "bytes"`), and that `token_to_id` of the byte tokenizer is `unicode_to_bytes()[s]` from `M05.2`.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_vocab` | unit | the section 3 table, id for id, in either text order | saved models and retrained tokenizers agree |
| `test_hand_example_encode_decode` | unit | `"hello"` is `[4, 3, 6, 6, 7]`, `"hi!"` ends in `<unk>`, decode with and without specials | you and the test agree on the definition |
| `test_hand_example_bytes` | unit | `"hé"` is `[104, 195, 169]`; a partial character decodes to one U+FFFD; id 256 is rejected | the tracer engine and `L10.0` decode the same way |
| `test_add_special_wraps_bos_eos` | unit | `<bos>` first, `<eos>` last, only when the role exists | `L3.6` trains on wrapped sequences |
| `test_no_normalization` | boundary | U+00E9 and e + U+0301 get different ids and both round-trip | your corpus pipeline owns normalization, not the tokenizer |
| `test_roundtrip_unicode_property` | property | 200 random texts across scripts and planes decode back exactly | any training text survives |
| `test_special_text_is_not_matched` | boundary | typing `<eos>` gives five character ids | a user cannot end generation from the prompt |
| `test_decode_rejects_bad_ids` | boundary | `-1`, `V`, `1.0`, `True`, `"1"` are `ValueError` | a corrupt id never decodes to a plausible character |
| `test_save_writes_the_char_format` | unit | the exact `tinyllm_char.json` document, roles to ids, and load gives the same ids | any reader of the format finds `<unk>` |
| `test_load_rejects_invalid_files` | boundary | no `unk`, a duplicate, a two-code-point entry, a fifth role, an id past the end, another `type` | a broken file fails at load, not mid-training |
| `test_train_rejects_unknown_special` | boundary | `<mask>` is refused; a repeated `<pad>` and an explicit `<unk>` are counted once | the format has exactly four roles |
| `test_both_satisfy_the_protocol` | unit | both classes are `Tokenizer` instances with every attribute | `L1.6` and the corpus pipeline take any tokenizer |
| `test_byte_tokens_are_the_gpt2_byte_map` | unit | `id_to_token(32)` is `Ġ`, and the map is a bijection on 0 to 255 | bytes are BPE with no merges (`L1.2`) |
| `test_byte_roundtrip_property` | property | 200 random texts survive bytes encode and decode | the tracer's tokenizer loses nothing |

### Your tests (rung R2)

Write these in `python/tests/l1-1-tokenizers/test_tokenizers.py`, importing only names from the two contracts. Each docstring says what the test must establish; the bodies are yours.

```python
def test_vocab_is_unk_specials_then_sorted_code_points():
    """train(["hello", "hi"], ("<bos>", "<eos>")) has vocab <unk> <bos> <eos> e h i l o."""
def test_unseen_character_is_unk():
    """An unseen character encodes to unk_id, it does not raise."""
def test_skip_special_drops_unk_bos_eos():
    """decode(..., skip_special=True) drops every special id, <unk> included."""
def test_decomposed_accent_round_trips():
    """"e" + U+0301 survives encode and decode unchanged (no normalization)."""
def test_special_text_in_input_is_plain_text():
    """Typing "<eos>" gives five character ids, never the eos id."""
def test_negative_and_out_of_range_ids_raise():
    """decode([-1]) and decode([vocab_size]) are ValueError."""
def test_save_load_round_trip():
    """load(save(t)) encodes like t, and the file's specials map roles to ids."""
def test_load_rejects_two_character_entry():
    """A vocab entry of two code points, or a role outside unk/bos/eos/pad, is ValueError."""
def test_byte_tokenizer_utf8_and_replacement():
    """ByteTokenizer encodes UTF-8 bytes and decodes a lone continuation byte to U+FFFD."""
```

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. normalizing in `encode` (NFC) | `"café"` typed with a combining accent comes back as a different string; byte counts in `L1.6` change | `test_no_normalization` (mutant `s06`) |
| 2. vocabulary in first-appearance order | ids depend on the order of the training files; a retrained tokenizer disagrees with the saved model | `test_hand_example_vocab` (mutant `s01`) |
| 3. a dictionary lookup without a default | an emoji in a prompt raises `KeyError` inside the server | `test_hand_example_encode_decode` (mutant `s03`) |
| 4. trusting `vocab[i]` | `-1` decodes to the last character; `True` decodes as id 1 | `test_decode_rejects_bad_ids` (mutants `s09`, `m01`, `m02`) |
| 5. matching special texts in the input | a prompt containing `<eos>` stops generation | `test_special_text_is_not_matched` (mutant `s08`) |
| 6. a loader that trusts the file | a two-character entry or a made-up role loads and encodes wrong ids without a word | `test_load_rejects_invalid_files` (mutants `s11`, `s12`) |
| 7. decoding bytes with `errors="ignore"` or as Latin-1 | a partial character vanishes, or `é` decodes as `Ã©` | `test_hand_example_bytes` (mutants `s15`, `s17`) |
| 8. `chr(i)` as the byte token string | byte 32 is `" "` instead of `Ġ`, and BPE's alphabet no longer matches | `test_byte_tokens_are_the_gpt2_byte_map` (mutant `s16`) |

## 6. Where it's used next
| Forward | `L3.6` | Registered call site uses this module. |

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M05.2` | `bytes_to_unicode` and `unicode_to_bytes` give the byte tokenizer its token strings |
| Back | `L0.0` | the tracer's byte ids, now behind the protocol (reading) |
| Forward | `L1.2` | byte-level BPE implements the protocol, calls `check_ids`, and starts from the same 256 byte symbols |
| Forward | `L1.3` | WordPiece implements the protocol; `add_special` becomes `[CLS] ... [SEP]` |
| Forward | `L1.4` | Unigram implements the protocol; `<unk>` is id 0 there too |
| Forward | `L1.6` | fertility, bytes per token, and the fallback rate take any `Tokenizer` |

`L1.5` implements a Rust tokenizer independently; Python and Rust compare tokenization through shared `tokenizer.json` files and fixtures. `L3.6` (char-level language models) arrives in a later batch and uses this protocol. If you skip this module, `ol check L1.2` stops with `needs L1.1`: build it, or rerun with `--ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `Tokenizer` protocol | Hugging Face `tokenizers` `Tokenizer` | a pipeline object (normalizer, pre-tokenizer, model, post-processor, decoder) with offsets and padding | `tokenizers/src/tokenizer/mod.rs` |
| `ByteTokenizer` | ByT5, and llama.cpp's byte fallback tokens `<0x41>` | byte models at scale; byte fallback inside a subword vocabulary | Xue et al., *ByT5* (2021); `llama-vocab.cpp` |
| `CharTokenizer` | Karpathy's char-level nanoGPT on Shakespeare | the same vocabulary rule, used to train a small GPT | `nanoGPT/data/shakespeare_char/prepare.py` |
| `check_ids` | `tokenizers` `decode` | rejects ids past the vocabulary with an error, never wraps | `tokenizers/src/tokenizer/mod.rs` `decode` |
