# Injective, surjective, bijective: the GPT-2 byte map

Until the tokenizers chapter of *Build Your Own LLM System*, the system's tokenizer is the simplest possible one: one token per byte, vocabulary 256. [Byte-level BPE](../../build-an-llm/p01-tokenizers/02-byte-level-bpe.md) replaces it, and the first thing that section does is load a real vocabulary, SmolLM2's `tokenizer.json`. Open it and the tokens look wrong: `"Ġthe"`, `"Ċ"`, `"ĠĠĠ"`. There is no `" the"` anywhere. A loader that reads those keys as text finds no token that starts with a space, so every word after the first gets split into single bytes and the ids stop matching the model's.

The fix is a fixed bijection between the 256 byte values and 256 visible characters, the one GPT-2 introduced. This section derives that bijection, together with the three definitions (injective, surjective, bijective) that prove it can be inverted. It is a small piece of discrete math with an unusually concrete payoff: get one range boundary wrong and no real vocabulary loads.

No code from earlier sections is needed; the functions and bijections problems in [this chapter's problem set](90-problem-set.md) are good preparation. The solution is [`bytes_unicode.py`](../../../../course/solutions/python/tinyllm/tok/bytes_unicode.py), with `is_injective`, `is_surjective`, `inverse`, `bytes_to_unicode`, `unicode_to_bytes`, `encode_bytes`, and `decode_chars`; the typed interface is [`course/contracts/py/tinyllm/tok/bytes_unicode.pyi`](../../../../course/contracts/py/tinyllm/tok/bytes_unicode.pyi).

> **Learning objectives**
> - Decide whether a finite function is injective, surjective onto a given codomain, or bijective, and say exactly what each check inspects.
> - Use the counting argument to show that between finite sets of equal size the three properties coincide.
> - Derive GPT-2's byte-to-character table from its rule and encode a sentence by hand.
> - Implement an inverse that refuses to hide collisions, and a decoder that rejects characters outside the image.

## Notation

The section uses these symbols and their Python counterparts:

| Symbol | Meaning | Type / shape |
|---|---|---|
| $A$, $B$ | finite sets: the domain (inputs) and the codomain (allowed outputs) | `set` |
| $f: A \to B$ | a function: exactly one output $f(a) \in B$ for every $a \in A$ | `Mapping` (keys $A$) |
| $f(A) = \{f(a) : a \in A\}$ | the image: the outputs that actually occur | `set` |
| $\lvert A \rvert$ | the number of elements of $A$ | `int` |
| $f^{-1}: B \to A$ | the inverse of a bijection: $f^{-1}(f(a)) = a$ and $f(f^{-1}(b)) = b$ | `dict` |
| $b \in \{0, \dots, 255\}$ | a byte value | `int` |
| $n$ | how many bytes before $b$ were remapped (the running count) | `int` |
| $\mathrm{chr}(c)$, $\mathrm{ord}(s)$ | code point $c$ to its one-character string, and back | `str`, `int` |

## Functions as tables

A function $f: A \to B$ assigns one output to every input. When $A$ is finite it can be written down as a table, and a Python `dict` is exactly that: its keys are $A$ and its values are the outputs. A dict cannot give one key two values, so every dict is a function from its key set. The codomain $B$ is not stored anywhere; it is part of the question being asked.

## Injective, surjective, bijective

With functions as tables, each of the three properties becomes a check on the table.

$f$ is **injective** (one to one) when different inputs give different outputs: $a \ne a' \Rightarrow f(a) \ne f(a')$. Equivalently, no value appears twice. Notice what is checked: the values. The keys of a dict are always distinct, so checking them proves nothing.

$f$ is **surjective** onto $B$ when every $b \in B$ is some $f(a)$, that is, $f(A) = B$. This is a statement about $B$, so it needs $B$. If some $f(a)$ lies outside $B$, then $f$ is not a function into $B$ at all, and `is_surjective` raises instead of answering.

$f$ is **bijective** when it is both. Then every $b \in B$ has exactly one preimage, and $f^{-1}(b)$, defined as that preimage, is a function $B \to A$. If $f$ is not injective, two keys compete for $f^{-1}(b)$. A dict comprehension `{v: k for k, v in f.items()}` silently keeps the last one, which is why `inverse` must check and raise.

## Counting settles it for equal sizes

For finite sets there is a shortcut. If $f$ is injective, its image has exactly $\lvert A \rvert$ elements, one per input. So when $\lvert A \rvert = \lvert B \rvert$, an injective $f$ hits all of $B$ and is also surjective. Conversely, a surjective $f$ needs at least $\lvert B \rvert$ distinct outputs, which uses up all $\lvert A \rvert$ inputs with no repeats; that is the pigeonhole principle, which the [problem set](90-problem-set.md) practices. For finite sets of equal size the three properties coincide: 256 bytes with 256 distinct outputs is a bijection onto those 256 outputs.

## Why GPT-2 needs a byte map

Byte-level BPE works on bytes, so its tokens are byte strings: `b" the"`, `b"\n"`, and halves of UTF-8 characters such as `b"\xc3"`. Three places need them as text, and each rules out a different set of characters:

1. `vocab.json` and `tokenizer.json` are JSON, whose keys must be valid Unicode text. A lone byte `0xC3` is not valid UTF-8, so a raw byte string cannot be a key.
2. `merges.txt` stores one merge per line as two tokens separated by a space (`Ġ t`). A token that contains a space would split the line in the wrong place.
3. Control characters (`0x00` to `0x1F`) and the soft hyphen `0xAD` are invisible, so a vocabulary full of them cannot be read or diffed.

So GPT-2 maps each byte to a character that is printable and not whitespace, by a fixed rule.

## The rule

Latin-1 gives every byte $b$ the character $\mathrm{chr}(b)$, and 188 of those are already visible: `!` to `~` (0x21 to 0x7E, 94 characters), `¡` to `¬` (0xA1 to 0xAC, 12), and `®` to `ÿ` (0xAE to 0xFF, 82). Those keep their own code point. The other $256 - 188 = 68$ bytes (0x00 to 0x20, 0x7F to 0xA0, and 0xAD) are walked in increasing order, and the $n$-th of them ($n = 0, 1, \dots, 67$) gets $\mathrm{chr}(256 + n)$, the code points U+0100 to U+0143. Those are Latin Extended-A letters such as `Ā`, `Ġ`, `Ń`, all visible.

The map is injective because the kept bytes land in $\{0x21, \dots, 0xFF\}$ (each on itself, so distinct) and the remapped bytes land in $\{256, \dots, 323\}$ (each on a new $n$, so distinct), and the two ranges do not overlap. By the counting argument it is a bijection onto its 256-character image, so `unicode_to_bytes` exists. The image is not "all characters", though: a real space `" "` or `"ń"` (U+0144) is not the image of any byte, and `decode_chars` must reject them.

Encoding and decoding then follow. `encode_bytes(data)` replaces each byte by its character, so the output has exactly `len(data)` characters. `decode_chars(text)` applies the inverse character by character. Composing them in either order is the identity, for any bytes, including invalid UTF-8.

## Working it by hand

Start with three small maps. Take $f = \{1 \mapsto a, 2 \mapsto b, 3 \mapsto a\}$ and $g = \{1 \mapsto a, 2 \mapsto b\}$. The four questions below exercise each definition:

| Question | Work | Answer |
|---|---|---|
| is $f$ injective? | $f(1) = f(3) = a$ with $1 \ne 3$ | no; `inverse(f)` raises, naming `a` |
| is $g$ injective? | values $a, b$ are distinct | yes |
| is $g$ onto $\{a, b\}$? | image $\{a, b\}$ equals it | yes, so $g^{-1} = \{a \mapsto 1, b \mapsto 2\}$ |
| is $g$ onto $\{a, b, c\}$? | $c$ has no preimage | no |

Now the byte map. **Where the space goes**: bytes 0x00 to 0x20 are all remapped (the first kept byte is 0x21 `!`), and they come first, so byte $b \le 0x20$ has $n = b$. The space 0x20 = 32 has $n = 32$ and maps to $\mathrm{chr}(256 + 32) = $ U+0120 `Ġ`. The newline 0x0A has $n = 10$ and maps to U+010A `Ċ`. Byte 0x00 maps to U+0100 `Ā`.

**Past the gap**: DEL, 0x7F, is the first remapped byte after `~`. Before it come 33 remapped bytes (0x00 to 0x20), so $n = 33$ and it maps to U+0121 `ġ`. The soft hyphen 0xAD comes after 0x7F to 0xA0 (34 more), so $n = 33 + 34 = 67$, the last one: U+0143 `Ń`.

**A sentence**: `"Hi there\n"` is the bytes `48 69 20 74 68 65 72 65 0A`, which map as follows:

| byte | 48 | 69 | 20 | 74 | 68 | 65 | 72 | 65 | 0A |
|---|---|---|---|---|---|---|---|---|---|
| kept? | `H` | `i` | no, $n = 32$ | `t` | `h` | `e` | `r` | `e` | no, $n = 10$ |
| char | `H` | `i` | `Ġ` | `t` | `h` | `e` | `r` | `e` | `Ċ` |

So `encode_bytes(b"Hi there\n") == "HiĠthereĊ"`, nine bytes to nine characters, and in a GPT-2 vocabulary the word " there" (with its leading space) is the token `"Ġthere"`.

**A non-ASCII character**: `"é"` is the two UTF-8 bytes `C3 A9`. Both are kept Latin-1 characters, `Ã` and `©`, so it encodes to `"Ã©"`: two characters, the familiar look of mis-decoded UTF-8.

These examples are the first two test cases below.

## Designing the interface

Seven functions, three for the definitions and four for the byte map:

```python
# python/tinyllm/tok/bytes_unicode.py
def is_injective(f: Mapping[K, V]) -> bool
def is_surjective(f: Mapping[K, V], codomain: Iterable[V]) -> bool   # ValueError if a value is outside
def inverse(f: Mapping[K, V]) -> dict[V, K]                          # ValueError if not injective
def bytes_to_unicode() -> dict[int, str]                             # a new dict each call, keys 0..255 in order
def unicode_to_bytes() -> dict[str, int]
def encode_bytes(data: bytes) -> str                                 # TypeError for str
def decode_chars(text: str) -> bytes                                 # ValueError outside the image
```

The solution builds `bytes_to_unicode` from the rule, not by pasting a table. A correct table equals GPT-2's entry for entry, and the rule is what [the Rust fast BPE](../../build-an-llm/p01-tokenizers/05-rust-fast-bpe.md) reimplements.

A correct solution must handle these cases:

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| Hand example, encode | unit | `"Hi there\n"` encodes to `"HiĠthereĊ"`; byte 0 is `Ā`; decode inverts it | pins the rule |
| Hand example, small maps | unit | $f$ and $g$ of the worked example | the three definitions, on paper and in code |
| GPT-2's table | golden | all 256 entries equal the transformers table | byte-level BPE loads GPT-2 and SmolLM2 vocabularies |
| Bijection onto 256 characters | property | 256 keys, 256 distinct one-character values | lossless bytes to text and back |
| Printable, not whitespace | property | no output is a space, control, or invisible character | merges files split on spaces |
| Kept bytes are fixed points | unit | the 188 kept bytes are fixed points; 0xAD is not | ASCII tokens read as themselves |
| Remapped bytes consecutive from 256 | unit | the 68 others take U+0100 to U+0143 in order | the running count, not the byte value |
| Byte order, fresh dict | boundary | a caller's edit does not leak into the next call | tokenizers build their own tables from it |
| The inverse | property | both compositions are the identity | decoding generated tokens |
| Random bytes round trip | property | decode(encode(x)) == x for random bytes | tokens can end inside a UTF-8 character |
| UTF-8 text round trip | unit | `é` is two characters, the emoji four | the map works on bytes, not characters |
| Characters outside the image | boundary | `" "`, `"\n"`, U+0144, `€` raise | a corrupt vocabulary fails loudly |
| A `str` passed to encode | boundary | `encode_bytes("abc")` is a `TypeError` | forgetting `.encode("utf-8")` is a bug |
| Inverse of a collision | boundary | a repeated value raises and names it | two ids never share a string |
| Injective cases | unit | empty, identity, permutation, collisions | injectivity is about values |
| Surjective cases | boundary | empty codomain, missing targets, values outside | surjectivity needs a codomain |

## Building it step by step

Six steps take you from the definitions to the code.

1. A finite function is a dict, and its keys are distinct by construction. So injectivity is a question about the *values* only: scan them once and stop at the first repeat.
2. Surjectivity needs the codomain as an argument, and it only makes sense for a function into that codomain. So the solution first checks that the image lies inside the codomain (raising otherwise), and then compares the image with the codomain as sets.
3. An inverse exists only for an injective map, and a dict comprehension would hide a collision. So `inverse` builds the dict one pair at a time and raises the moment a value is seen twice, naming both keys.
4. The byte map must be GPT-2's exactly, so it is built from GPT-2's rule: three kept ranges, then the 68 other bytes in increasing order, each taking the next code point from 256. The counter $n$ is the running count of remapped bytes, not the byte value.
5. Injectivity of that map follows from the two disjoint output ranges, and equal sizes make it a bijection. So `unicode_to_bytes` is simply `inverse` of the table, and `inverse` re-checks the claim at run time.
6. Encoding and decoding are the table and its inverse applied character by character. Decoding must reject anything outside the 256-character image instead of computing a byte by arithmetic.

The three definitions come first. `is_injective` looks only at values; `is_surjective` checks "into" before "onto":

```python
def is_injective(f: Mapping[Hashable, Hashable]) -> bool:
    seen: set = set()
    for v in f.values():          # the values, never the keys: keys are always distinct
        if v in seen:
            return False          # two inputs share an output
        seen.add(v)
    return True


def is_surjective(f: Mapping[Hashable, Hashable], codomain: Iterable[Hashable]) -> bool:
    target = set(codomain)
    image = set(f.values())
    outside = image - target
    if outside:                   # f is not a function into B: the question has no answer
        raise ValueError(
            f"f is not a function into the codomain: value {next(iter(outside))!r} is outside it"
        )
    return image == target        # equality, not image <= target: every b must be hit
```

`inverse` is the only safe way to flip a dict, because it refuses to overwrite:

```python
def inverse(f: Mapping[Hashable, Hashable]) -> dict:
    inv: dict = {}
    for k, v in f.items():
        if v in inv:              # a second preimage for v: f is not injective
            raise ValueError(
                f"f is not injective: keys {inv[v]!r} and {k!r} both map to {v!r}"
            )
        inv[v] = k
    return inv
```

The byte map is the rule, line for line. The kept set is written as three inclusive ranges; the gap at 0xAD is what splits the last two:

```python
def bytes_to_unicode() -> dict[int, str]:
    # The printable Latin-1 ranges keep their own code point. 0xAD (soft
    # hyphen) sits between the last two and is invisible, so it is remapped.
    ranges = ((0x21, 0x7E), (0xA1, 0xAC), (0xAE, 0xFF))
    keep = {b for lo, hi in ranges for b in range(lo, hi + 1)}   # 94 + 12 + 82 = 188
    out: dict[int, str] = {}
    n = 0                         # how many bytes have been remapped so far
    for b in range(256):          # increasing byte order: keys come out 0..255
        if b in keep:
            out[b] = chr(b)       # visible already: a fixed point
        else:
            out[b] = chr(256 + n) # the n-th invisible byte -> U+0100 + n
            n += 1
    return out                    # a new dict on every call: callers may edit theirs
```

Tracing it reproduces the worked example: bytes 0 to 32 are the first 33 remapped (space is $n = 32$, `Ġ`), DEL gets $n = 33$, and the soft hyphen is last with $n = 67$, `Ń`.

`unicode_to_bytes` reuses `inverse`, so the bijection is checked, not assumed. Encoding refuses `str`, because a `str` would have to be encoded to bytes first, and guessing Latin-1 is wrong past U+00FF. Decoding refuses characters outside the image:

```python
def unicode_to_bytes() -> dict[str, int]:
    return inverse(bytes_to_unicode())


def encode_bytes(data: bytes) -> str:
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError(f"encode_bytes takes bytes, got {type(data).__name__}")
    table = bytes_to_unicode()
    return "".join(table[b] for b in data)     # one character per byte


def decode_chars(text: str) -> bytes:
    table = unicode_to_bytes()
    out = bytearray()
    for i, ch in enumerate(text):
        b = table.get(ch)                       # a lookup, never ord(ch) - 256
        if b is None:                           # " ", "\n", U+0144, "€": not an image
            raise ValueError(
                f"character {ch!r} (U+{ord(ch):04X}) at index {i} is not in the byte map's image"
            )
        out.append(b)
    return bytes(out)
```

The full solution is in [`bytes_unicode.py`](../../../../course/solutions/python/tinyllm/tok/bytes_unicode.py).

### Why this design

On cost: `is_injective`, `is_surjective`, and `inverse` are $O(\lvert A \rvert + \lvert B \rvert)$ with set and dict lookups. `bytes_to_unicode` is 256 steps. `encode_bytes` and `decode_chars` are linear in their input plus a rebuild of the 256-entry table on every call. That constant is negligible next to tokenization, and it buys a table no caller can corrupt.

The table is built from the rule rather than pasted because a pasted table of 256 entries is easy to get subtly wrong and impossible to read. The rule is three ranges and a counter, matches GPT-2's original `bytes_to_unicode`, and is what the Rust tokenizer reimplements.

Decoding uses a lookup, not arithmetic, because `ord(c) - 256` is right only for the first 33 remapped bytes; after DEL the remapped code points no longer equal $256 + b$. Arithmetic also turns characters outside the image into some byte instead of an error, so a corrupt vocabulary would decode to garbage silently.

The table is not cached in a module global because a shared dict lets one caller's mutation change every later tokenizer, and returning a fresh dict costs microseconds. (Production code builds it once and freezes it; see the further reading.)

One tradeoff is accepted: `is_surjective` raises instead of returning `False` for a map that is not into $B$. That is stricter than a boolean, but "onto $B$" is not a meaningful question for such a map, and a silent `False` would hide a caller's bug.

## What can go wrong

Several of these mistakes still produce a valid bijection, just not GPT-2's, which is why the golden comparison against the real table matters:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| mapping every byte to itself, `chr(b)` | a bijection, but the space and control bytes stay invisible; merges lines split inside tokens | remaps every byte outside the three kept ranges |
| one range 0xA1 to 0xFF | the soft hyphen 0xAD keeps its invisible code point and every later remapped byte is off | three ranges, with 0xAD left out between them |
| remapping to `chr(256 + b)` | still injective and visible, but not GPT-2's table: no real vocabulary loads | `chr(256 + n)` with the running count $n$ (step 4) |
| decoding by arithmetic (`ord(c) - 256`, `& 0xFF`) | happens to work for bytes 0 to 32, decodes everything after DEL to the wrong byte, and accepts characters outside the image | decodes by looking up `unicode_to_bytes()` and raises on a miss |
| inverting with a dict comprehension | a collision keeps the last key silently; two ids share one string | `inverse` raises on the first repeated value |
| checking keys for injectivity | always true, since dict keys are distinct | scans `f.values()` |
| "onto" as $f(A) \subseteq B$ | a map that misses targets reports onto | returns `image == target` |
| ignoring values outside the codomain | the question "onto $B$?" answered for a map that is not into $B$ | raises when `image - target` is not empty |
| caching the table in a module global | one caller's edit changes every later tokenizer | builds and returns a new dict on every call |
| encoding a `str` by code points | `"é"` becomes one byte, wrong for anything past Latin-1 | `encode_bytes` raises `TypeError` unless given `bytes` or `bytearray` |
| an off-by-one range end (`0x7E` vs `0x7D`) | `~` is remapped | the first range is `(0x21, 0x7E)`, inclusive |

> **Warning:** `ord(c) - 256` passes every test on ASCII text with spaces and newlines, because those bytes are all below DEL. It fails only on the bytes after the gap, which is exactly where UTF-8 continuation bytes live.

## Where this leads

The functions, inverses, and pigeonhole problems in [this chapter's problem set](90-problem-set.md) are the pen-and-paper side of this section. In *Build Your Own LLM System*, the byte map is used throughout the tokenizer and decoding code:

- [Tokenizer protocol, char tokenizer](../../build-an-llm/p01-tokenizers/01-tokenizer-protocol-and-char.md): every byte-level tokenizer decodes ids through `decode_chars`.
- [Byte-level BPE](../../build-an-llm/p01-tokenizers/02-byte-level-bpe.md): pre-tokenized text is encoded to UTF-8, mapped with `encode_bytes`, merged, and looked up in `vocab.json`; the Hugging Face loader reads `Ġ`-keys with `unicode_to_bytes`.
- [Rust fast BPE](../../build-an-llm/p01-tokenizers/05-rust-fast-bpe.md): `tl-tok` rebuilds the same table from the same rule (a reimplementation, not a call) and is tested id for id against byte-level BPE.
- [Constrained decoding](../../build-an-llm/p08-inference/07-constrained-decoding.md) uses this module.
- [Whisper decoding and long-form transcription](../../build-an-llm/p14-speech/04-whisper-decoding.md) consumes this implementation.

### Further reading

- Hugging Face `tokenizers` ByteLevel (`tokenizers/src/pre_tokenizers/byte_level.rs`, `bytes_char`): the same table in Rust, built once in a `lazy_static`, plus `add_prefix_space` and offset tracking.
- OpenAI `tiktoken` uses no character map at all: ranks are keyed by raw bytes and stored base64 encoded in `.tiktoken` files. `tiktoken/load.py` (`data_gym_to_mergeable_bpe_ranks`) rebuilds this map to read GPT-2's original files.
- openai/gpt-2, `src/encoder.py`: the original 2019 function, with the comment explaining why whitespace and control characters are avoided.
- Rosen, *Discrete Mathematics and Its Applications*, section 2.3, "Functions".
- Radford et al., "Language Models are Unsupervised Multitask Learners" (2019), section 2.2.

## Key terms

- **Injective**: no two inputs share an output; for a dict, no value appears twice.
- **Surjective onto $B$**: every element of $B$ is the output of some input; the image equals $B$.
- **Bijective**: both injective and surjective; exactly the functions that have an inverse.
- **Image**: the set of outputs that actually occur, $f(A)$.
- **Pigeonhole principle**: $\lvert A \rvert$ inputs cannot produce more than $\lvert A \rvert$ distinct outputs, so for equal sizes injective and surjective coincide.
- **Byte map**: GPT-2's fixed bijection from the 256 byte values to 256 visible, non-whitespace characters.

## Summary

- A finite function is injective when no two inputs share an output, surjective onto a set when every element of that set is hit, and bijective when both; only a bijection has an inverse.
- Between two finite sets of the same size, injective, surjective, and bijective are the same thing, so 256 distinct outputs from 256 bytes is enough to prove the byte map is invertible.
- GPT-2 maps every byte to a visible, non-whitespace character so a vocabulary of byte strings can live in JSON and a merges file can split on spaces.
- The table is built from a rule (three kept ranges, then $\mathrm{chr}(256 + n)$ for the $n$-th remapped byte), and it equals the one every GPT-2 style vocabulary uses, byte for byte, so byte-level BPE can load GPT-2 and SmolLM2 tokenizers.

## Review questions

1. Why does `is_injective` iterate over `f.values()` and never over the keys?

   <details><summary>Answer</summary>

   Injective means no two inputs share an output, so the repeats to look for are among the outputs. The keys of a dict are distinct by construction, so checking them would always return `True` and prove nothing.

   </details>

2. Why does `inverse` raise on a repeated value instead of building `{v: k for k, v in f.items()}`?

   <details><summary>Answer</summary>

   The comprehension silently keeps the last key for a repeated value, so it returns a "function" that is not the inverse of anything: one preimage is lost. In a tokenizer that means two ids share one string and decoding one of them returns the other. Raising, with both keys named, turns that into an immediate error.

   </details>

3. In `bytes_to_unicode`, why is the remapped character `chr(256 + n)` and not `chr(256 + b)`?

   <details><summary>Answer</summary>

   GPT-2 assigns the code points U+0100 to U+0143 consecutively, in byte order, to the 68 remapped bytes. $n$ counts those bytes, so it runs from 0 to 67. `chr(256 + b)` would still be injective and visible, but for DEL (0x7F) it gives U+017F instead of U+0121 `ġ`, so no real vocabulary would load.

   </details>

4. Why is it enough to show the map is injective to conclude it is a bijection onto its image, and why does `decode_chars` still need to check its input?

   <details><summary>Answer</summary>

   Every function is surjective onto its own image, so injective plus "onto the image" is a bijection between the 256 bytes and the 256 image characters. But the image is not all of Unicode: a real space or `ń` has no preimage. `decode_chars` is defined only on the image, so it must reject anything else.

   </details>

5. `encode_bytes("é")` raises `TypeError`. Why not encode the string's code points?

   <details><summary>Answer</summary>

   The map is defined on bytes. `"é"` is one code point, U+00E9, but two UTF-8 bytes, `C3 A9`, and the vocabulary is keyed by the encoding of those two bytes, `"Ã©"`. Treating the code point as a byte gives `"é"`, which is wrong, and for any character above U+00FF there is no single byte at all. The caller must choose the encoding (`.encode("utf-8")`).

   </details>
