# The tokenizer protocol and the char tokenizer

So far the tracer has spoken bytes. The [byte bigram](../p00-foundations/00-byte-bigram.md) trained on raw UTF-8 bytes, and [your first endpoint](../p10-serving/00-your-first-endpoint.md) serves `tl_tokenizer = "bytes"`. That works, but it is wasteful: `é` costs two tokens, and the model spends capacity learning how to spell UTF-8. This chapter replaces bytes with trained vocabularies: [byte-level BPE](02-byte-level-bpe.md), [WordPiece](03-wordpiece.md), and the [Unigram language model](04-unigram-lm.md), compared in [Tokenizer metrics](06-tokenizer-metrics.md) and chosen for the capstone.

Four tokenizers with four calling conventions would mean four versions of the corpus pipeline, of the metrics, and of the engine's tokenizer factory. So before writing any trained tokenizer, this section fixes the shape: one `Tokenizer` protocol, the existing byte tokenizer rewritten against it, and the simplest trained tokenizer there is, one id per character. That char tokenizer is not just a warm-up; it becomes the vocabulary of the char-level models in [the RNN language model](../p03-recurrent/06-rnn-language-model-stateful-tbptt.md).

You need one piece of mathematics before starting: the GPT-2 byte map (`bytes_to_unicode`, `unicode_to_bytes`) from [Injective, surjective, bijective; the GPT-2 byte map](../../mathematics-for-computing/05-discrete-math-1/02-injective-surjective-bijective-gpt2-byte-map.md) in *Mathematics for Computing*. The solution code lives in [`base.py`](../../../../course/solutions/python/tinyllm/tok/base.py) (the `Tokenizer` protocol, `check_ids`, `ByteTokenizer`) and [`char.py`](../../../../course/solutions/python/tinyllm/tok/char.py) (`CharTokenizer` with `train`, `encode`, `decode`, `save`, `load`). The contracts are [`base.pyi`](../../../../course/contracts/py/tinyllm/tok/base.pyi) and [`char.pyi`](../../../../course/contracts/py/tinyllm/tok/char.pyi), and the file formats are described in [`tokenizer.md`](../../../../course/contracts/formats/tokenizer.md) and [`tokenizer-char.schema.json`](../../../../course/contracts/formats/tokenizer-char.schema.json).

> **Learning objectives**
> - State the six members of the `Tokenizer` protocol and its two laws.
> - Explain why every `decode` must validate ids, including rejecting `bool` and negative ids.
> - Build a char vocabulary whose ids do not depend on the order of the training texts, and reproduce one by hand.
> - Explain why special-token texts typed by a user must stay ordinary characters.
> - Describe why the byte tokenizer decodes with replacement and why its token strings are the GPT-2 byte map.

## Text is code points, storage is bytes

Start with what a text actually is. The notation used through this chapter is summarized here:

| Symbol | Meaning | Type / shape |
|---|---|---|
| $\Sigma$ | the alphabet of Unicode code points, `U+0000` to `U+10FFFF` | set |
| $t \in \Sigma^*$ | a text: a finite sequence of code points | `str` |
| $V$ | vocabulary size: ids are $0, \dots, V - 1$ | `int` |
| $\mathrm{enc}: \Sigma^* \to \{0..V-1\}^*$ | encode: text to ids | `str -> list[int]` |
| $\mathrm{dec}: \{0..V-1\}^* \to \Sigma^*$ | decode: ids to text | `list[int] -> str` |
| $u$ | the unknown id (`unk_id`), or none when every text encodes | `int` or `None` |
| $S$ | the special ids: `<unk>`, `<bos>`, `<eos>`, `<pad>` | `dict[str, int]` |
| $\beta: \{0..255\} \to \Sigma$ | the GPT-2 byte map (byte 0x20 is `Ġ`) | bijection |

A Python `str` is a sequence of code points, so `len("é") == 1`. On disk and on the wire the same text is UTF-8: one to four bytes per code point. `é` (U+00E9) is the two bytes `c3 a9`, and an emoji is four bytes. Worse, two texts that render identically may be different sequences: `é` can be the single code point U+00E9, or `e` followed by the combining acute accent U+0301. A tokenizer that silently normalizes one into the other can never give the original text back, which is why the tokenizers in this book never normalize. Normalization belongs to the corpus pipeline, where it is a visible, versioned decision.

## A tokenizer is a contract

With that settled, what is a tokenizer? Not an algorithm, but a contract. Every tokenizer in the system (bytes, char, BPE, WordPiece, Unigram, and later the Rust one) has the same members, so everything that consumes tokens is written once. In Python the contract is a `typing.Protocol`, which means a class satisfies it simply by having the members; no inheritance is needed. The members are:

- `vocab_size`: ids live in $[0, V)$, and every id in that range decodes (there are no gaps).
- `special_ids`: the special tokens' texts mapped to their ids.
- `unk_id`: where unknown input goes, or `None` when every text encodes (bytes, byte-level BPE).
- `encode(text, add_special=False)` and `decode(ids, skip_special=False)`.
- `token_to_id` and `id_to_token`: the vocabulary as a two-way table.
- `save(dir)` and the class method `load(dir)`: a tokenizer lives in a model directory.

The protocol has two laws. First, $\mathrm{dec}(\mathrm{enc}(t)) = t$ for every text the vocabulary covers. Second, $\mathrm{enc}$ never raises on a text: unknown input becomes $u$. Unknown input is data, not an exception. The flag `add_special` wraps a template around the ids (BOS and EOS here, `[CLS]` and `[SEP]` for WordPiece) and never changes the ids of the text itself; `skip_special` drops the ids in $S$ when decoding.

### Ids are checked once, in one place

Decoding never trusts its input. Every `decode` starts with `check_ids(ids, V)`, which requires each id to be an integer (`hasattr(i, "__index__")`, so numpy integers pass), not a `bool`, and in the range $0 \le i < V$. The `bool` rule exists because in Python `True == 1`: a mask passed by mistake would decode as text. The range check matters more than it looks: `vocab[-1]` is valid Python and returns the last entry, so without the check a model that emits `-1` decodes to a perfectly plausible character.

### Bytes are already a tokenizer

The tracer's byte tokenizer fits the protocol as it stands. `ByteTokenizer` has $V = 256$; $\mathrm{enc}(t)$ is the UTF-8 bytes of $t$, and $\mathrm{dec}$ is UTF-8 decoding **with replacement**, so a byte sequence that is not valid UTF-8 becomes U+FFFD instead of an exception. That choice is forced: a sampled model can emit any bytes, and the server must still answer.

Its token strings are $\beta(i)$, the GPT-2 byte map, which assigns a printable code point to every byte. So `id_to_token(32)` is `Ġ` and `id_to_token(10)` is `Ċ`. That is exactly the alphabet [byte-level BPE](02-byte-level-bpe.md) starts from, so "bytes" is simply BPE with zero merges.

## The char tokenizer

The simplest trained tokenizer gives one id per code point. Training only collects the code points that occur, and the vocabulary is fixed by a rule that does not depend on the order of the texts:

1. id 0 is `<unk>`, always;
2. then the requested specials, each one of `<bos>`, `<eos>`, `<pad>`, in the order given, with a repeat counted once;
3. then every code point of the training texts, sorted by code point.

Encoding maps each code point to its id, or to $u = 0$ when it was never seen. Crucially, the texts of the specials are not matched in the input. If a user could type `<eos>` and get the end-of-sequence id, any prompt could stop generation. Special ids enter a sequence only through `add_special`, which puts `<bos>` first and `<eos>` last when those roles exist.

The tokenizer saves itself as `dir/tinyllm_char.json`, in the char format of [`tokenizer.md`](../../../../course/contracts/formats/tokenizer.md): `{"type": "char", "vocab": [...], "specials": {"unk": 0, ...}}`. Notice that `specials` maps **roles** to ids, not texts to ids, so a reader in any language finds `<unk>` without guessing its spelling. Loading never trusts files: `load` checks everything the constructor checks, namely unique entries, `unk` present, every role one of the four, every special id inside the vocabulary, and every other entry exactly one code point.

## Working it by hand

Train with `CharTokenizer.train(["hello", "hi"], specials=("<bos>", "<eos>"))`. The code points seen are `h e l o i`. Sorted by code point they are `e` (101), `h` (104), `i` (105), `l` (108), `o` (111). With `<unk>` first and the two specials next, the vocabulary is:

| id | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| text | `<unk>` | `<bos>` | `<eos>` | e | h | i | l | o |

Training on `["hi", "hello"]` gives the same table, because the order of the texts does not matter. First-appearance order would have given `h` id 3 and `e` id 4, and a tokenizer retrained on shuffled data would silently disagree with every saved model.

Now encode. `"hello"` is h e l l o, so `[4, 3, 6, 6, 7]`. `"hi!"` is `[4, 5, 0]`: `!` was never seen, so it becomes `<unk>`. With `add_special=True`, `"hi"` is `[1, 4, 5, 2]`.

Decoding reverses this. `[4, 5, 0]` decodes to `"hi<unk>"` (the special's text), and with `skip_special=True` to `"hi"`. Both `[-1]` and `[8]` are `ValueError`.

Finally, the byte tokenizer. `ByteTokenizer().encode("hé")`: `h` is 0x68 = 104, and `é` is U+00E9, the two UTF-8 bytes `c3 a9` = 195, 169. So the ids are `[104, 195, 169]`. Decoding `[195]` alone is an incomplete sequence and gives one U+FFFD, `"�"`. Decoding `[226, 130]` (the first two bytes of a three-byte character) also gives one `"�"`: one maximal ill-formed subpart, one replacement character.

## Designing the interface

Those hand-worked numbers pin down behavior. The interface that carries them is short:

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

The contracts carry the exact rules: which inputs are `ValueError`; that `ByteTokenizer.save` creates the directory and writes nothing, because its `config.json` already says `tl_tokenizer = "bytes"`; and that `token_to_id` of the byte tokenizer is `unicode_to_bytes()[s]` from the byte map.

The model test suite, [`test_tokenizers.py`](../../../../course/solutions/learner-tests/L1.1/python/tests/l1-1-tokenizers/test_tokenizers.py), imports only names from the two contracts and pins the behaviors a buggy tokenizer most often gets wrong. Each test's name says what it establishes:

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

Note that the first test trains on `["hi", "hello"]`, the reverse of the hand-worked order, and still expects the same table: one test covers both the rule and its order independence.

## Building it step by step

Before reading the code, follow the chain of reasoning that produces it.

1. **Consumers need one shape.** The corpus pipeline, the metrics, and the engine's factory should not know which tokenizer they hold. A structural `typing.Protocol` with `@runtime_checkable` gives that shape without forcing inheritance, so `ByteTokenizer` and later tokenizers satisfy it just by having the members.
2. **Id validation is shared, so it lives in one function.** Every `decode` and `id_to_token` must reject non-integers, `bool`, and out-of-range ids. Writing that once as `check_ids` means each tokenizer is one call away from being safe.
3. **The byte tokenizer needs no state but the byte map.** Encoding is `text.encode("utf-8")`; decoding must survive any byte sequence a model can emit, which forces `errors="replace"`. Token strings come from the byte map so that bytes are BPE's starting alphabet.
4. **The char vocabulary must be a function of the set of characters, not their order.** Sorting by code point is the simplest total order with that property. Putting `<unk>` at 0 and specials next gives every char tokenizer the same fixed prefix.
5. **Specials are ids, not texts the encoder searches for.** So the encoder's lookup table excludes special texts entirely: a typed `<eos>` is five characters, and only `add_special` inserts the special ids.
6. **A loaded file must satisfy every rule the constructor enforces.** The cheapest way is for `load` to parse the JSON, check types, and then call the constructor, so validation exists once.

### Validating ids

`check_ids` makes two rejections that look pedantic and are not.

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

### The byte tokenizer

Each method of `ByteTokenizer` is one line, because UTF-8 and the byte map do the work.

```python
class ByteTokenizer:
    vocab_size = 256

    def __init__(self) -> None:
        self.special_ids: dict[str, int] = {}
        self.unk_id: Optional[int] = None      # every text encodes: no unknown id
        self._b2u = bytes_to_unicode()          # byte -> printable code point (GPT-2 byte map)
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

Python's UTF-8 decoder follows the Unicode recommendation of one replacement per maximal ill-formed subpart, which is why the hand-worked `[226, 130]` decodes to one `"�"` and not two.

### The char tokenizer's constructor

`CharTokenizer.__init__` holds all the validation and builds the two lookup tables.

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

### Training

`train` is the vocabulary rule, line for line.

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

On the hand-worked example, the specials give `<unk> <bos> <eos>` at 0 to 2; `seen` is `{h, e, l, o, i}`; sorting by `ord` gives `e h i l o` at 3 to 7.

### Encoding and decoding

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

Iterating over a `str` yields code points, so there is no normalization step to forget to remove: a decomposed `é` is two entries, `e` and U+0301, and decodes back as the same two.

### Saving and loading

`load` checks the JSON's types, then lets the constructor check the rules.

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

The full solution is in [`base.py`](../../../../course/solutions/python/tinyllm/tok/base.py) and [`char.py`](../../../../course/solutions/python/tinyllm/tok/char.py).

### Why this design

The costs are small. Training is $O(\sum |t|)$ to collect the set plus $O(K \log K)$ to sort $K$ distinct code points; encoding and decoding are $O(n)$ dictionary and list lookups; the byte tokenizer is a single C-level UTF-8 conversion each way.

The alternatives each lose something. An abstract base class would work, but it forces every tokenizer, including later Rust-backed ones wrapped in Python, to inherit from it; a protocol states the contract without coupling. Frequency-ordered vocabularies (common characters get small ids) are popular, but they depend on corpus statistics, so a retrain on more data reshuffles ids. Normalizing to NFC in `encode` makes some texts impossible to reproduce.

Two tradeoffs are accepted knowingly. A char vocabulary grows with every script in the corpus and has no way to represent an unseen character except `<unk>`; that limitation is the motivation for [byte-level BPE](02-byte-level-bpe.md). And `ByteTokenizer` ignores `add_special` and `skip_special`, because it has no specials.

## What can go wrong

Every pitfall below has been seen in real tokenizers, and each one is closed by a specific line of the solution. A correct solution must reproduce the hand-worked vocabulary (in either text order), encodings, decodings, and byte results; wrap BOS and EOS only when the role exists; round-trip U+00E9 and `e` + U+0301 to different ids; survive 200 random texts across scripts and planes for both char and byte tokenizers; refuse `<mask>` in `train` while counting a repeated `<pad>` and an explicit `<unk>` once; and make both classes `Tokenizer` instances with every attribute.

> **Warning:** Normalizing in `encode` (to NFC, say) means `"café"` typed with a combining accent comes back as a different string, and byte counts in the metrics change. The solution iterates code points as they are and normalizes nowhere.

The remaining failures, with their symptoms and the fix, fit in one table:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| vocabulary in first-appearance order | ids depend on the order of the training files; a retrained tokenizer disagrees with the saved model | collects a set, then `sorted(seen, key=ord)` |
| a dictionary lookup without a default | an emoji in a prompt raises `KeyError` inside the server | `self._ids.get(ch, self.unk_id)` |
| trusting `vocab[i]` | `-1` decodes to the last character; `True` decodes as id 1; `1.0` and `"1"` slip through | every `decode` and `id_to_token` goes through `check_ids` |
| matching special texts in the input | a prompt containing `<eos>` stops generation | `_ids` excludes special texts; only `add_special` inserts special ids |
| a loader that trusts the file | a two-character entry, a duplicate, a fifth role, an id past the end, or another `type` loads and encodes wrong ids without a word | `load` type-checks the JSON, then the constructor validates every rule |
| decoding bytes with `errors="ignore"` or as Latin-1 | a partial character vanishes, or `é` decodes as `Ã©` | `decode("utf-8", "replace")` |
| `chr(i)` as the byte token string | byte 32 is `" "` instead of `Ġ`, and BPE's alphabet no longer matches | token strings come from `bytes_to_unicode()` |

## Where this leads

Everything after this section stands on the protocol. [Byte-level BPE](02-byte-level-bpe.md) implements it, calls `check_ids`, and starts from the same 256 byte symbols. [WordPiece](03-wordpiece.md) implements it, and there `add_special` becomes `[CLS] ... [SEP]`. The [Unigram tokenizer](04-unigram-lm.md) implements it, with `<unk>` at id 0 there too. [Tokenizer metrics](06-tokenizer-metrics.md) computes fertility, bytes per token, and the fallback rate for any `Tokenizer`. The [Rust fast BPE](05-rust-fast-bpe.md) is an independent implementation; Python and Rust compare tokenization through shared `tokenizer.json` files and fixtures. Later, the [RNN language model](../p03-recurrent/06-rnn-language-model-stateful-tbptt.md) trains char-level models on wrapped sequences, and its checkpoint names its tokenizer in `tl_tokenizer` (`bytes` or a saved tokenizer file).

Each piece here has a production counterpart worth reading:

- The `Tokenizer` protocol corresponds to the Hugging Face `tokenizers` `Tokenizer`, a pipeline object (normalizer, pre-tokenizer, model, post-processor, decoder) with offsets and padding; see `tokenizers/src/tokenizer/mod.rs`. Its `decode` likewise rejects ids past the vocabulary with an error and never wraps.
- `ByteTokenizer` corresponds to ByT5 (Xue et al., *ByT5*, 2021), byte models at scale, and to llama.cpp's byte fallback tokens such as `<0x41>`, byte fallback inside a subword vocabulary (`llama-vocab.cpp`).
- `CharTokenizer` uses the same vocabulary rule as Karpathy's char-level nanoGPT on Shakespeare (`nanoGPT/data/shakespeare_char/prepare.py`).

### Further reading

- The Unicode Standard, chapter 2, "General Structure", and chapter 3, section 3.9 (UTF-8).
- Jurafsky and Martin, *Speech and Language Processing* (3rd ed. draft), chapter 2, "Regular Expressions, Tokenization, Edit Distance".

## Key terms

- **Tokenizer protocol**: the six-member structural interface (`vocab_size`, `special_ids`, `unk_id`, `encode`/`decode`, `token_to_id`/`id_to_token`, `save`/`load`) every tokenizer satisfies.
- **Code point**: one element of a Python `str`; a Unicode scalar between U+0000 and U+10FFFF.
- **Unknown id (`unk_id`)**: the id an unseen input maps to, or `None` when every text encodes.
- **Special token**: an id with a role (`unk`, `bos`, `eos`, `pad`) that enters a sequence only through `add_special`, never by matching input text.
- **GPT-2 byte map**: the bijection from the 256 bytes to printable code points that gives byte tokens their strings.
- **Maximal ill-formed subpart**: the unit of invalid UTF-8 that decoding with replacement turns into one U+FFFD.

## Summary

- A tokenizer is a contract, not an algorithm: every tokenizer in the system has the same members, so everything that consumes tokens is written once.
- The tracer's byte tokenizer is already a tokenizer: 256 ids, no training, every text round-trips, and its token strings are the GPT-2 byte map, the alphabet BPE starts from.
- A char tokenizer's ids must not depend on the order of the training texts: `<unk>` is 0, the specials follow, then code points sorted by code point.
- Unknown input is data, not an exception: an unseen character becomes `<unk>`, and special-token texts typed by a user stay ordinary characters.
- Decoding never trusts ids and loading never trusts files: `-1`, `vocab_size`, `True`, and a malformed `tinyllm_char.json` are all `ValueError`.

## Review questions

1. Why does `check_ids` test `isinstance(i, bool)` before `hasattr(i, "__index__")`?
   <details><summary>Answer</summary>
   `bool` is a subclass of `int` in Python, so `True` has `__index__` and would pass as id 1. A boolean mask passed to `decode` by mistake would then decode to text. Rejecting `bool` explicitly first closes that hole while still accepting numpy integers.
   </details>
2. Why is `_ids` built without the special texts, while `special_ids` contains them?
   <details><summary>Answer</summary>
   `_ids` is what `encode` searches, and input text must never produce a special id, or a user typing `<eos>` could stop generation. `special_ids` is the protocol's text-to-id map for code that intentionally wants a special (for example `token_to_id("<eos>")`). Keeping them separate makes the safe behavior the default.
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
6. Train on `["hello", "hi"]` with specials `("<bos>", "<eos>")`. What does `"hi!"` encode to with `add_special=True`?
   <details><summary>Answer</summary>
   The vocabulary is `<unk> <bos> <eos> e h i l o` at ids 0 to 7, so `"hi!"` is `[4, 5, 0]` and wrapping gives `[1, 4, 5, 0, 2]`.
   </details>
