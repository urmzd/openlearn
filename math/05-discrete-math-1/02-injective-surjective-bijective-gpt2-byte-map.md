<!-- ol:module M05.2 -->
# Injective, surjective, bijective; the GPT-2 byte map

## Overview

| | |
|---|---|
| **Module** | `M05.2` · build · Python · Pass 3 · 2 to 3 h |
| **You build** | `python/tinyllm/tok/bytes_unicode.py`: `is_injective`, `is_surjective`, `inverse`, `bytes_to_unicode`, `unicode_to_bytes`, `encode_bytes`, `decode_chars` |
| **Contract** | [`course/contracts/py/tinyllm/tok/bytes_unicode.pyi`](../../course/contracts/py/tinyllm/tok/bytes_unicode.pyi) |
| **Tests** | `course/tests/M05.2/` (what they check: section 4) |
| **Needs** | no code from earlier modules · reading: `S-M05` (functions and bijections problems) |
| **Used by** | `L1.1` the tokenizer protocol · `L1.2` byte-level BPE stores its vocabulary and merges in this alphabet · later: `L1.5` re-implements the table in Rust · later: `L8.7`; `L14.4` |
| **Milestone** | `MS-P3` (tokens and data) |
| **Optional depth** | Rosen, *Discrete Mathematics and Its Applications*, section 2.3 "Functions"; Radford et al., "Language Models are Unsupervised Multitask Learners" (2019), section 2.2 |

## Key Takeaways

- A finite function is injective when no two inputs share an output, surjective onto a set when every element of that set is hit, and bijective when both; only a bijection has an inverse (`test_hand_example_small_maps`, `test_inverse_rejects_a_collision`).
- Between two finite sets of the same size, injective, surjective, and bijective are the same thing, so 256 distinct outputs from 256 bytes is enough to prove the byte map is invertible (`test_is_a_bijection_onto_256_characters`).
- GPT-2 maps every byte to a visible, non-whitespace character so a vocabulary of byte strings can live in JSON and a merges file can split on spaces (`test_every_character_is_printable_and_not_whitespace`).
- Your table equals the one every GPT-2 style vocabulary uses, byte for byte (`test_matches_gpt2_table`), so `L1.2` can load GPT-2 and SmolLM2 tokenizers.

## How to work this chapter

```bash
ol start M05.2              # stubs bytes_unicode.py into your repo
ol tests M05.2              # read the test catalog first
ol check M05.2              # exit code is the verdict
ol diff  M05.2              # after passing: your code against the reference
```

---

## 1. Why now

Your system's tokenizer is still the tracer's: one token per byte, vocabulary 256 (D32). Pass 3 replaces it with byte-level BPE (`L1.2`), and the first thing `L1.2` does is load a real vocabulary, SmolLM2's `tokenizer.json`. Open it and the tokens look wrong: `"Ġthe"`, `"Ċ"`, `"ĠĠĠ"`. There is no `" the"` anywhere. A loader that reads those keys as text finds no token that starts with a space, so every word after the first gets split into single bytes and your ids stop matching the model's. The fix is a fixed bijection between the 256 byte values and 256 visible characters, the one GPT-2 introduced. This module builds that bijection, and the three definitions (injective, surjective, bijective) that prove it can be inverted.

## 2. Principles

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

**Functions as tables.** A function $f: A \to B$ assigns one output to every input. When $A$ is finite you can write it down as a table, and a Python `dict` is exactly that: its keys are $A$ and its values are the outputs. A dict cannot give one key two values, so every dict is a function from its key set. The codomain $B$ is not stored anywhere; it is part of the question you ask.

**Injective (one to one).** $f$ is injective when different inputs give different outputs: $a \ne a' \Rightarrow f(a) \ne f(a')$. Equivalently, no value appears twice. Note what is checked: the values. The keys of a dict are always distinct, so checking them proves nothing.

**Surjective (onto $B$).** $f$ is surjective onto $B$ when every $b \in B$ is some $f(a)$, that is $f(A) = B$. This is a statement about $B$, so it needs $B$. If some $f(a)$ lies outside $B$, then $f$ is not a function into $B$ at all, and `is_surjective` raises instead of answering.

**Bijective.** Both at once. Then every $b \in B$ has exactly one preimage, and $f^{-1}(b)$ = that preimage is a function $B \to A$. If $f$ is not injective, two keys compete for $f^{-1}(b)$; a dict comprehension `{v: k for k, v in f.items()}` silently keeps the last one, which is why `inverse` must check and raise.

**Counting settles it for equal sizes.** If $f$ is injective, its image has exactly $\lvert A \rvert$ elements, one per input. So when $\lvert A \rvert = \lvert B \rvert$, an injective $f$ hits all of $B$ and is also surjective. Conversely a surjective $f$ needs at least $\lvert B \rvert$ distinct outputs, which uses up all $\lvert A \rvert$ inputs with no repeats (the pigeonhole principle, `S-M05`). For finite sets of equal size the three properties coincide: 256 bytes with 256 distinct outputs is a bijection onto those 256 outputs.

**Why GPT-2 needs a byte map.** Byte-level BPE works on bytes, so its tokens are byte strings: `b" the"`, `b"\n"`, and halves of UTF-8 characters such as `b"\xc3"`. Three places need them as text:

1. `vocab.json` and `tokenizer.json` are JSON, whose keys must be valid Unicode text. A lone byte `0xC3` is not valid UTF-8, so a raw byte string cannot be a key.
2. `merges.txt` stores one merge per line as two tokens separated by a space (`Ġ t`). A token that contains a space would split the line in the wrong place.
3. Control characters (`0x00` to `0x1F`) and the soft hyphen `0xAD` are invisible, so a vocabulary full of them cannot be read or diffed.

So GPT-2 maps each byte to a character that is printable and not whitespace, by a fixed rule.

**The rule.** Latin-1 gives every byte $b$ the character $\mathrm{chr}(b)$, and 188 of those are already visible: `!` to `~` (0x21 to 0x7E, 94 characters), `¡` to `¬` (0xA1 to 0xAC, 12), and `®` to `ÿ` (0xAE to 0xFF, 82). Those keep their own code point. The other $256 - 188 = 68$ bytes (0x00 to 0x20, 0x7F to 0xA0, and 0xAD) are walked in increasing order, and the $n$-th of them ($n = 0, 1, \dots, 67$) gets $\mathrm{chr}(256 + n)$, the code points U+0100 to U+0143. Those are Latin Extended-A letters such as `Ā`, `Ġ`, `Ń`, all visible.

The map is injective because the kept bytes land in $\{0x21, \dots, 0xFF\}$ (each on itself, so distinct) and the remapped bytes land in $\{256, \dots, 323\}$ (each on a new $n$, so distinct), and the two ranges do not overlap. By the counting argument it is a bijection onto its 256-character image, so `unicode_to_bytes` exists. The image is not "all characters": a real space `" "` or `"ń"` (U+0144) is not the image of any byte, and `decode_chars` must reject them.

**Encode and decode.** `encode_bytes(data)` replaces each byte by its character, so the output has exactly `len(data)` characters. `decode_chars(text)` applies the inverse character by character. Composing them in either order is the identity, for any bytes, including invalid UTF-8.

## 3. Worked example by hand

**Three small maps.** Take $f = \{1 \mapsto a, 2 \mapsto b, 3 \mapsto a\}$ and $g = \{1 \mapsto a, 2 \mapsto b\}$.

| Question | Work | Answer |
|---|---|---|
| is $f$ injective? | $f(1) = f(3) = a$ with $1 \ne 3$ | no; `inverse(f)` raises, naming `a` |
| is $g$ injective? | values $a, b$ are distinct | yes |
| is $g$ onto $\{a, b\}$? | image $\{a, b\}$ equals it | yes, so $g^{-1} = \{a \mapsto 1, b \mapsto 2\}$ |
| is $g$ onto $\{a, b, c\}$? | $c$ has no preimage | no |

**Where the space goes.** Bytes 0x00 to 0x20 are all remapped (the first kept byte is 0x21 `!`), and they come first, so byte $b \le 0x20$ has $n = b$. The space 0x20 = 32 has $n = 32$ and maps to $\mathrm{chr}(256 + 32) = $ U+0120 `Ġ`. The newline 0x0A has $n = 10$ and maps to U+010A `Ċ`. Byte 0x00 maps to U+0100 `Ā`.

**Past the gap.** DEL, 0x7F, is the first remapped byte after `~`. Before it come 33 remapped bytes (0x00 to 0x20), so $n = 33$ and it maps to U+0121 `ġ`. The soft hyphen 0xAD comes after 0x7F to 0xA0 (34 more), so $n = 33 + 34 = 67$, the last one: U+0143 `Ń`.

**A sentence.** `"Hi there\n"` is the bytes `48 69 20 74 68 65 72 65 0A`:

| byte | 48 | 69 | 20 | 74 | 68 | 65 | 72 | 65 | 0A |
|---|---|---|---|---|---|---|---|---|---|
| kept? | `H` | `i` | no, $n = 32$ | `t` | `h` | `e` | `r` | `e` | no, $n = 10$ |
| char | `H` | `i` | `Ġ` | `t` | `h` | `e` | `r` | `e` | `Ċ` |

So `encode_bytes(b"Hi there\n") == "HiĠthereĊ"`, nine bytes to nine characters, and in a GPT-2 vocabulary the word " there" (with its leading space) is the token `"Ġthere"`.

**A non-ASCII character.** `"é"` is the two UTF-8 bytes `C3 A9`. Both are kept Latin-1 characters, `Ã` and `©`, so it encodes to `"Ã©"`: two characters, the familiar look of mis-decoded UTF-8.

These are the first cases in section 4: `test_hand_example_encode` and `test_hand_example_small_maps`.

## 4. The interface

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

Write `bytes_to_unicode` from the rule in section 2, not by pasting a table: the tests compare it with GPT-2's, and the rule is what `L1.5` reimplements in Rust.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_encode` | unit | `"Hi there\n"` encodes to `"HiĠthereĊ"`; byte 0 is `Ā`; decode inverts it | you and the test agree on the rule |
| `test_hand_example_small_maps` | unit | $f$ and $g$ of section 3 | the three definitions, on paper and in code |
| `test_matches_gpt2_table` | golden | all 256 entries equal the transformers table | `L1.2` loads GPT-2 and SmolLM2 vocabularies |
| `test_is_a_bijection_onto_256_characters` | property | 256 keys, 256 distinct one-character values | lossless bytes to text and back |
| `test_every_character_is_printable_and_not_whitespace` | property | no output is a space, control, or invisible character | merges files split on spaces |
| `test_printable_bytes_map_to_themselves` | unit | the 188 kept bytes are fixed points; 0xAD is not | ASCII tokens read as themselves |
| `test_remapped_bytes_are_consecutive_from_256` | unit | the 68 others take U+0100 to U+0143 in order | the running count, not the byte value |
| `test_keys_in_byte_order_and_a_fresh_dict_each_call` | boundary | a caller's edit does not leak into the next call | tokenizers build their own tables from it |
| `test_unicode_to_bytes_is_the_inverse` | property | both compositions are the identity | decoding generated tokens |
| `test_roundtrip_random_bytes` | property | decode(encode(x)) == x for random bytes | tokens can end inside a UTF-8 character |
| `test_roundtrip_utf8_text` | unit | `é` is two characters, the emoji four | the map works on bytes, not characters |
| `test_decode_rejects_characters_outside_the_image` | boundary | `" "`, `"\n"`, U+0144, `€` raise | a corrupt vocabulary fails loudly |
| `test_encode_rejects_str` | boundary | `encode_bytes("abc")` is a `TypeError` | forgetting `.encode("utf-8")` is a bug |
| `test_inverse_rejects_a_collision` | boundary | a repeated value raises and names it | two ids never share a string |
| `test_is_injective_cases` | unit | empty, identity, permutation, collisions | injectivity is about values |
| `test_is_surjective_cases` | boundary | empty codomain, missing targets, values outside | surjectivity needs a codomain |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. mapping every byte to itself, `chr(b)` | a bijection, but the space and control bytes stay invisible; merges lines split inside tokens | `test_every_character_is_printable_and_not_whitespace` (mutant `s01`) |
| 2. one range 0xA1 to 0xFF | the soft hyphen 0xAD keeps its invisible code point and every later remapped byte is off | `test_printable_bytes_map_to_themselves` (mutant `s02`) |
| 3. remapping to `chr(256 + b)` | still injective and visible, but not GPT-2's table: no real vocabulary loads | `test_remapped_bytes_are_consecutive_from_256` (mutant `s03`) |
| 4. decoding by arithmetic (`ord(c) - 256`, `& 0xFF`) | happens to work for bytes 0 to 32, decodes everything after DEL to the wrong byte, and accepts characters outside the image | `test_roundtrip_random_bytes`, `test_decode_rejects_characters_outside_the_image` (mutant `s05`) |
| 5. inverting with a dict comprehension | a collision keeps the last key silently; two ids share one string | `test_inverse_rejects_a_collision` (mutant `s04`) |
| 6. checking keys for injectivity | always true, since dict keys are distinct | `test_is_injective_cases` (mutant `s08`) |
| 7. "onto" as $f(A) \subseteq B$ | a map that misses targets reports onto | `test_is_surjective_cases` (mutant `s07`) |
| 8. ignoring values outside the codomain | the question "onto $B$?" answered for a map that is not into $B$ | `test_is_surjective_cases` (mutant `s06`) |
| 9. caching the table in a module global | one caller's edit changes every later tokenizer | `test_keys_in_byte_order_and_a_fresh_dict_each_call` (mutant `s10`) |
| 10. encoding a `str` by code points | `"é"` becomes one byte, wrong for anything past Latin-1 | `test_encode_rejects_str` (mutant `s09`) |
| 11. an off-by-one range end (`0x7E` vs `0x7D`) | `~` is remapped | `test_matches_gpt2_table` (mutant `m01`) |

## 6. Where it's used next
| Forward | `L8.7` | Registered call site uses this module. |

| Direction | Module | How it uses this |
|---|---|---|
| Back | `S-M05` | functions, inverses, and the pigeonhole principle as pen-and-paper problems |
| Forward | `L1.1` | the tokenizer protocol: every byte-level tokenizer decodes ids through `decode_chars` |
| Forward | `L1.2` | byte-level BPE: pre-tokenized text is encoded to UTF-8, mapped with `encode_bytes`, merged, and looked up in `vocab.json`; the HF loader reads `Ġ`-keys with `unicode_to_bytes` |
| Forward | `L1.5` | your Rust `tl-tok` rebuilds the same table from the same rule (a re-implementation, not a call) and is tested id for id against `L1.2` |

If you skip this module, `ol check L1.2` stops with `L1.2 needs M05.2`: build it, or rerun with `--ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `bytes_to_unicode` | Hugging Face `tokenizers` ByteLevel | the same table in Rust, built once in a `lazy_static`, plus `add_prefix_space` and offset tracking | `tokenizers/src/pre_tokenizers/byte_level.rs` (`bytes_char`) |
| `encode_bytes` / `decode_chars` | OpenAI `tiktoken` | no character map at all: ranks are keyed by raw bytes and stored base64 encoded in `.tiktoken` files | `tiktoken/load.py` (`data_gym_to_mergeable_bpe_ranks` rebuilds this map to read GPT-2's original files) |
| the rule | openai/gpt-2 | the original 2019 function, with the comment explaining why whitespace and control characters are avoided | `src/encoder.py` |

| Forward | `L14.4` | Whisper decoding consumes this implementation. |
