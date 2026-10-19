"""Pre-tokenizers for byte-level BPE (L1.2), written over unicodedata.

GPT-2 splits text into pre-tokens with one regular expression before any
merge runs, so a merge can never cross a pre-token boundary:

    's|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+

`pretokenize_gpt2` is that expression written by hand, one alternative at a
time, in the order the regex engine tries them: \\p{L} is a code point in a
letter category (Lu Ll Lt Lm Lo), \\p{N} a number category (Nd Nl No), and \\s
the Unicode White_Space property. `split_digits` is the Hugging Face `Digits`
pre-tokenizer that SmolLM2 runs first.

Contract: contracts/py/tinyllm/tok/pretok.pyi.
"""

from __future__ import annotations

import unicodedata

# Unicode White_Space (PropList.txt): what \s matches in the GPT-2 regex.
# str.isspace() differs: it also accepts U+001C to U+001F.
WHITE_SPACE = frozenset(
    "\t\n\x0b\x0c\r \x85\xa0\u1680"
    + "".join(chr(c) for c in range(0x2000, 0x200B))
    + "\u2028\u2029\u202f\u205f\u3000"
)
CONTRACTIONS = ("s", "t", "re", "ve", "m", "ll", "d")


def is_space(ch: str) -> bool:
    return ch in WHITE_SPACE


def is_letter(ch: str) -> bool:
    return unicodedata.category(ch) in ("Lu", "Ll", "Lt", "Lm", "Lo")


def is_number(ch: str) -> bool:
    return unicodedata.category(ch) in ("Nd", "Nl", "No")


def _class(ch: str) -> str:
    """'L' letter, 'N' number, 'S' white space, 'O' anything else."""
    if is_space(ch):
        return "S"
    if is_letter(ch):
        return "L"
    if is_number(ch):
        return "N"
    return "O"


def pretokenize_gpt2(text: str) -> list[str]:
    """Split text exactly as the GPT-2 regex does; "".join(result) == text."""
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
        k = _class(text[j])
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


def split_digits(text: str, individual: bool = True) -> list[str]:
    """The `Digits` pre-tokenizer: every \\p{N} code point is its own piece
    (individual) or each run of them is one piece; the text between is kept
    whole. "".join(result) == text and no piece is empty."""
    out: list[str] = []
    start = 0
    i = 0
    n = len(text)
    while i < n:
        if is_number(text[i]):
            if start < i:
                out.append(text[start:i])
            end = i + 1
            if not individual:
                while end < n and is_number(text[end]):
                    end += 1
            out.append(text[i:end])
            start = i = end
        else:
            i += 1
    if start < n:
        out.append(text[start:])
    return out
