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

These are the first cases a correct solution must reproduce (section 4).

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

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the hand-worked vocabulary | unit | the section 3 table, id for id, in either text order | saved models and retrained tokenizers agree |
| the hand-worked encode and decode | unit | `"hello"` is `[4, 3, 6, 6, 7]`, `"hi!"` ends in `<unk>`, decode with and without specials | everyone agrees on the definition |
| the hand-worked bytes | unit | `"hé"` is `[104, 195, 169]`; a partial character decodes to one U+FFFD; id 256 is rejected | the tracer engine and `L10.0` decode the same way |
| `add_special` wraps BOS and EOS | unit | `<bos>` first, `<eos>` last, only when the role exists | `L3.6` trains on wrapped sequences |
| no normalization | boundary | U+00E9 and e + U+0301 get different ids and both round-trip | the corpus pipeline owns normalization, not the tokenizer |
| Unicode round trip | property | 200 random texts across scripts and planes decode back exactly | any training text survives |
| special text is not matched | boundary | typing `<eos>` gives five character ids | a user cannot end generation from the prompt |
| bad ids on decode | boundary | `-1`, `V`, `1.0`, `True`, `"1"` are `ValueError` | a corrupt id never decodes to a plausible character |
| save writes the char format | unit | the exact `tinyllm_char.json` document, roles to ids, and load gives the same ids | any reader of the format finds `<unk>` |
| invalid files on load | boundary | no `unk`, a duplicate, a two-code-point entry, a fifth role, an id past the end, another `type` | a broken file fails at load, not mid-training |
| unknown specials in train | boundary | `<mask>` is refused; a repeated `<pad>` and an explicit `<unk>` are counted once | the format has exactly four roles |
| both satisfy the protocol | unit | both classes are `Tokenizer` instances with every attribute | `L1.6` and the corpus pipeline take any tokenizer |
| byte tokens are the GPT-2 byte map | unit | `id_to_token(32)` is `Ġ`, and the map is a bijection on 0 to 255 | bytes are BPE with no merges (`L1.2`) |
| byte round trip | property | 200 random texts survive bytes encode and decode | the tracer's tokenizer loses nothing |

### The model tests

The model suite, [`test_tokenizers.py`](../../../../course/solutions/learner-tests/L1.1/python/tests/l1-1-tokenizers/test_tokenizers.py), imports only names from the two contracts and pins the behaviours a buggy tokenizer most often gets wrong. Each name says what the test establishes:

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

Note the first test trains on `["hi", "hello"]`, the reverse of section 3's order, and still expects section 3's table: one test covers both the rule and its order independence.

## 5. The solution, step by step

### From first principles

1. **Consumers need one shape.** The corpus pipeline, the metrics of `L1.6`, and the engine's factory should not know which tokenizer they hold. A structural `typing.Protocol` with `@runtime_checkable` gives that shape without forcing inheritance, so `ByteTokenizer` and later tokenizers satisfy it just by having the members.
2. **Id validation is shared, so it lives in one function.** Every `decode` and `id_to_token` must reject non-integers, `bool`, and out-of-range ids, for the reasons in section 2. Writing that once as `check_ids` means each tokenizer is one call away from being safe.
3. **The byte tokenizer needs no state but the byte map.** Encoding is `text.encode("utf-8")`; decoding must survive any byte sequence a model can emit, which forces `errors="replace"`. Token strings come from `M05.2` so that bytes are BPE's starting alphabet.
4. **The char vocabulary must be a function of the set of characters, not their order.** Sorting by code point is the simplest total order with that property. Putting `<unk>` at 0 and specials next gives every char tokenizer the same fixed prefix.
5. **Specials are ids, not texts the encoder searches for.** So the encoder's lookup table excludes special texts entirely: a typed `<eos>` is five characters, and only `add_special` inserts the special ids.
6. **A loaded file must satisfy every rule the constructor enforces.** The cheapest way is for `load` to parse the JSON, check types, and then call the constructor, so validation exists once.

### The code, annotated

**`check_ids`.** Two rejections that look pedantic and are not.

```python
def check_ids(ids: Sequence[int], vocab_size: int) -> list[int]:
    out: list[int] = []
    for i in ids:
        # bool first: True has __index__ and equals 1, but a mask is not an id.
        # hasattr(__index__) accepts Python and numpy integers, refuses 1.0 and "1".
        if isinstance(i, bool) or not hasattr(i, "__index__"):
            raise ValueError(f"token ids must be integers, got {i!r}")
        v = int(i)
        if not 0 <= v < vocab_size:      # -1 would otherwise index vocab[-1]
            raise ValueError(f"token id {v} is outside [0, {vocab_size})")
        out.append(v)
    return out
```

**`ByteTokenizer`.** Each method is one line because UTF-8 and `M05.2` do the work.

```python
class ByteTokenizer:
    vocab_size = 256

    def __init__(self) -> None:
        self.special_ids: dict[str, int] = {}
        self.unk_id: Optional[int] = None      # every text encodes: no unknown id
        self._b2u = bytes_to_unicode()          # byte -> printable code point (M05.2)
        self._u2b = unicode_to_bytes()

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: Sequence[int], skip_special: bool = False) -> str:
        # "replace": one U+FFFD per maximal ill-formed subpart, never an exception
        return bytes(check_ids(ids, 256)).decode("utf-8", "replace")

    def token_to_id(self, s: str) -> Optional[int]:
        return self._u2b.get(s)

    def id_to_token(self, i: int) -> str:
        (v,) = check_ids([i], 256)
        return self._b2u[v]                     # 32 -> "Ġ", 10 -> "Ċ"

    def save(self, dir: str) -> None:
        # Nothing to write: config.json declares tl_tokenizer = "bytes".
        os.makedirs(dir, exist_ok=True)
```

Python's UTF-8 decoder follows the Unicode recommendation of one replacement per maximal ill-formed subpart, which is why section 3's `[226, 130]` is one `"�"` and not two.

**`CharTokenizer.__init__`.** All validation, and the two lookup tables.

```python
    def __init__(self, vocab: Sequence[str], specials: Mapping[str, int]) -> None:
        vocab = list(vocab)
        if len(set(vocab)) != len(vocab):
            raise ValueError("vocab entries must be unique")
        if "unk" not in specials:
            raise ValueError("specials must name the `unk` id")
        for role, i in specials.items():
            if role not in ROLES:                       # unk, bos, eos, pad only
                raise ValueError(f"special role {role!r} is not one of {ROLES}")
            if not 0 <= i < len(vocab):
                raise ValueError(f"special {role} = {i} is outside the vocab")
        special_texts = {vocab[i] for i in specials.values()}
        for i, t in enumerate(vocab):
            if t not in special_texts and len(t) != 1:  # len counts code points
                raise ValueError(f"vocab[{i}] = {t!r} is not one code point")
        self.vocab = vocab
        self.roles = dict(specials)                     # role -> id, what the file stores
        self.vocab_size = len(vocab)
        self.special_ids = {vocab[i]: i for i in specials.values()}   # text -> id, the protocol's view
        self.unk_id: Optional[int] = specials["unk"]
        # The encoder's table excludes special texts: "<eos>" in input is never matched.
        self._ids = {t: i for i, t in enumerate(vocab) if t not in special_texts}
```

**`train`.** The vocabulary rule of section 2, line for line.

```python
    @classmethod
    def train(cls, texts: Iterable[str], specials: Sequence[str] = ()) -> "CharTokenizer":
        vocab = ["<unk>"]                      # rule 1: id 0
        roles = {"unk": 0}
        for s in specials:                     # rule 2: in the order given
            role = s[1:-1] if s.startswith("<") and s.endswith(">") else ""
            if role not in ROLES:
                raise ValueError(f"special {s!r} must be one of <unk>, <bos>, <eos>, <pad>")
            if role not in roles:              # a repeat, or an explicit <unk>, counted once
                roles[role] = len(vocab)
                vocab.append(s)
        seen: set[str] = set()
        for t in texts:
            seen.update(t)                     # a set: the order of texts cannot matter
        vocab += sorted(seen, key=ord)         # rule 3: by code point
        return cls(vocab, roles)
```

On section 3: specials give `<unk> <bos> <eos>` at 0 to 2; `seen` is `{h, e, l, o, i}`; sorting by `ord` gives `e h i l o` at 3 to 7.

**Encode and decode.**

```python
    def encode(self, text: str, add_special: bool = False) -> list[int]:
        ids = [self._ids.get(ch, self.unk_id) for ch in text]   # unseen -> <unk>, never KeyError
        if add_special:
            if "bos" in self.roles:
                ids.insert(0, self.roles["bos"])
            if "eos" in self.roles:
                ids.append(self.roles["eos"])
        return ids

    def decode(self, ids: Sequence[int], skip_special: bool = False) -> str:
        special = set(self.roles.values())
        out = []
        for i in check_ids(ids, self.vocab_size):
            if skip_special and i in special:   # <unk> is a special too
                continue
            out.append(self.vocab[i])
        return "".join(out)
```

Iterating over a `str` yields code points, so there is no normalization step to forget to remove: `"é"` is two entries, `e` and U+0301, and decodes back as the same two.

**Save and load.** `load` checks the JSON's types, then lets the constructor check the rules.

```python
    def save(self, dir: str) -> None:
        os.makedirs(dir, exist_ok=True)
        doc = {"type": "char", "vocab": self.vocab, "specials": self.roles}   # roles, not texts
        with open(os.path.join(dir, FILE), "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=1)
            f.write("\n")

    @classmethod
    def load(cls, dir: str) -> "CharTokenizer":
        with open(os.path.join(dir, FILE), encoding="utf-8") as f:
            doc = json.load(f)
        if not isinstance(doc, dict) or doc.get("type") != "char":
            raise ValueError(f"{FILE}: `type` must be \"char\"")
        vocab, specials = doc.get("vocab"), doc.get("specials")
        # ... vocab must be a list of non-empty strings; specials must map roles to
        #     ints that are not bools
        return cls(vocab, specials)            # the constructor enforces every other rule
```

Full solution: [`base.py`](../../../../course/solutions/python/tinyllm/tok/base.py) · [`char.py`](../../../../course/solutions/python/tinyllm/tok/char.py).

### Why this solution

- **Cost.** Training is $O(\sum |t|)$ to collect the set plus $O(K \log K)$ to sort $K$ distinct code points. Encoding and decoding are $O(n)$ dictionary and list lookups. The byte tokenizer is a single C-level UTF-8 conversion each way.
- **Alternatives.** An abstract base class would work but forces every tokenizer, including later Rust-backed ones wrapped in Python, to inherit from it; a protocol states the contract without coupling. Frequency-ordered vocabularies (common characters get small ids) are popular and depend on corpus statistics, so a retrain on more data reshuffles ids. Normalizing to NFC in `encode` makes some texts impossible to reproduce; normalization belongs to the corpus pipeline, where it is a visible, versioned decision.
- **Accepted tradeoffs.** A char vocabulary grows with every script in the corpus and has no way to represent an unseen character except `<unk>`; that is the motivation for byte-level BPE in `L1.2`. `ByteTokenizer` ignores `add_special` and `skip_special` because it has no specials.

### Explain it back

1. Why does `check_ids` test `isinstance(i, bool)` before `hasattr(i, "__index__")`?
   <details><summary>Answer</summary>
   `bool` is a subclass of `int` in Python, so `True` has `__index__` and would pass as id 1. A boolean mask passed to `decode` by mistake would then decode to text. Rejecting `bool` explicitly first closes that hole while still accepting numpy integers.
   </details>
2. Why is `_ids` built without the special texts, while `special_ids` contains them?
   <details><summary>Answer</summary>
   `_ids` is what `encode` searches, and input text must never produce a special id, or a user typing `<eos>` could stop generation. `special_ids` is the protocol's text-to-id map for code that intentionally wants a special (for example `token_to_id("<eos>")`). Keeping them separate makes the safe behaviour the default.
   </details>
3. Why `sorted(seen, key=ord)` and not `sorted(seen)`?
   <details><summary>Answer</summary>
   For single-character strings both give the same order, since Python compares strings by code point. `key=ord` states the rule (by code point) explicitly and would raise if a multi-character string ever slipped into `seen`, rather than sorting it silently.
   </details>
4. Why does the byte tokenizer decode with `"replace"` rather than raising on invalid UTF-8?
   <details><summary>Answer</summary>
   A sampled model can emit any byte sequence, including half of a multi-byte character at the end of a response. The server must still return text. Replacement gives a visible U+FFFD per ill-formed subpart and loses nothing valid; `"ignore"` would silently drop bytes and raising would fail the request.
   </details>
5. Why does `load` call the constructor instead of assigning fields directly?
   <details><summary>Answer</summary>
   The constructor already enforces uniqueness, the `unk` role, the four roles, ids inside the vocabulary, and one code point per ordinary entry. Reusing it means a file is held to exactly the rules a trained tokenizer is, and there is only one place to change if a rule changes.
   </details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. normalizing in `encode` (NFC) | `"café"` typed with a combining accent comes back as a different string; byte counts in `L1.6` change | iterates code points as they are; no normalization anywhere |
| 2. vocabulary in first-appearance order | ids depend on the order of the training files; a retrained tokenizer disagrees with the saved model | collects a set, then `sorted(seen, key=ord)` |
| 3. a dictionary lookup without a default | an emoji in a prompt raises `KeyError` inside the server | `self._ids.get(ch, self.unk_id)` |
| 4. trusting `vocab[i]` | `-1` decodes to the last character; `True` decodes as id 1 | every `decode` and `id_to_token` goes through `check_ids` |
| 5. matching special texts in the input | a prompt containing `<eos>` stops generation | `_ids` excludes special texts; only `add_special` inserts special ids |
| 6. a loader that trusts the file | a two-character entry or a made-up role loads and encodes wrong ids without a word | `load` type-checks the JSON, then the constructor validates every rule |
| 7. decoding bytes with `errors="ignore"` or as Latin-1 | a partial character vanishes, or `é` decodes as `Ã©` | `decode("utf-8", "replace")` |
| 8. `chr(i)` as the byte token string | byte 32 is `" "` instead of `Ġ`, and BPE's alphabet no longer matches | token strings come from `M05.2`'s `bytes_to_unicode()` |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M05.2` | `bytes_to_unicode` and `unicode_to_bytes` give the byte tokenizer its token strings |
| Back | `L0.0` | the tracer's byte ids, now behind the protocol (reading) |
| Forward | `L1.2` | byte-level BPE implements the protocol, calls `check_ids`, and starts from the same 256 byte symbols |
| Forward | `L1.3` | WordPiece implements the protocol; `add_special` becomes `[CLS] ... [SEP]` |
| Forward | `L1.4` | Unigram implements the protocol; `<unk>` is id 0 there too |
| Forward | `L1.6` | fertility, bytes per token, and the fallback rate take any `Tokenizer` |
| Forward | `L3.6` | Registered call site uses this module. |

`L1.5` implements a Rust tokenizer independently; Python and Rust compare tokenization through shared `tokenizer.json` files and fixtures. `L3.6` (char-level language models) uses this protocol.

## Going further

| Piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `Tokenizer` protocol | Hugging Face `tokenizers` `Tokenizer` | a pipeline object (normalizer, pre-tokenizer, model, post-processor, decoder) with offsets and padding | `tokenizers/src/tokenizer/mod.rs` |
| `ByteTokenizer` | ByT5, and llama.cpp's byte fallback tokens `<0x41>` | byte models at scale; byte fallback inside a subword vocabulary | Xue et al., *ByT5* (2021); `llama-vocab.cpp` |
| `CharTokenizer` | Karpathy's char-level nanoGPT on Shakespeare | the same vocabulary rule, used to train a small GPT | `nanoGPT/data/shakespeare_char/prepare.py` |
| `check_ids` | `tokenizers` `decode` | rejects ids past the vocabulary with an error, never wraps | `tokenizers/src/tokenizer/mod.rs` `decode` |
