"""craft.04 kata: two text operations whose laws you test with properties.

1. `normalize(text)`, the shape of the corpus normalizer (data.02 builds the
   real one): Unicode NFC, one line-ending convention, one space between
   words, at most one blank line between paragraphs, nothing at either end.
   Its law is IDEMPOTENCE: normalize(normalize(x)) == normalize(x). A
   pipeline that reruns a stage (a retried activity, a resumed run) must get
   the same bytes the second time.

2. `MiniBPE`, a byte-level BPE with a fixed merge list (the shape of L1.2
   and L1.5 without a pre-tokenizer). Its law is the ROUNDTRIP:
   decode(encode(x)) == x for every string x. Roundtrip alone is weak: a BPE
   that merges nothing also round-trips. The chapter's other properties
   (fixpoint, agreement with a model) catch the rest.

The rules, exactly (the planted faults break one each):

normalize(text):
    a. NFC (unicodedata.normalize("NFC", text)).
    b. "\\r\\n" and lone "\\r" become "\\n".
    c. Within each line, every run of horizontal white space (any character
       with str.isspace() other than "\\n") becomes one " ", and each line is
       stripped of spaces at both ends.
    d. Runs of 3 or more "\\n" become exactly "\\n\\n".
    e. The result is stripped of white space at both ends.

MiniBPE(merges): merges[r] = (a, b) creates id 256 + r from ids a and b;
ids 0 to 255 are the bytes. encode(text) takes the UTF-8 bytes and merges
repeatedly: the adjacent pair with the lowest rank, the leftmost on ties,
until no adjacent pair has a merge. decode_bytes(ids) expands every id to
its bytes; decode(ids) is decode_bytes(ids) decoded as UTF-8 with
replacement (errors="replace"). An id outside 0 .. 255 + len(merges) is a
ValueError.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

_HSPACE = re.compile(r"[^\S\n]+")  # white space other than "\n"
_BLANKS = re.compile(r"\n{3,}")


def normalize(text: str) -> str:
    """Rules a to e of the module docstring."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_HSPACE.sub(" ", line).strip(" ") for line in text.split("\n")]
    text = "\n".join(lines)
    text = _BLANKS.sub("\n\n", text)
    return text.strip()


class MiniBPE:
    """Byte-level BPE over a fixed merge list."""

    def __init__(self, merges: Sequence[tuple[int, int]]) -> None:
        """merges[r] = (a, b): ids a and b merge into 256 + r. ValueError when
        a part names an id that does not exist yet."""
        self.merges = [(int(a), int(b)) for a, b in merges]
        self.rank: dict[tuple[int, int], int] = {}
        self.parts: list[tuple[int, int]] = []
        for r, (a, b) in enumerate(self.merges):
            if not (0 <= a < 256 + r and 0 <= b < 256 + r):
                raise ValueError(
                    f"merge {r} = ({a}, {b}) uses an id that does not exist yet"
                )
            self.rank.setdefault((a, b), r)
            self.parts.append((a, b))

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges)

    def encode(self, text: str) -> list[int]:
        """Lowest rank first, leftmost on ties, until no pair merges."""
        ids = list(text.encode("utf-8"))
        while len(ids) > 1:
            best = None
            for i in range(len(ids) - 1):
                r = self.rank.get((ids[i], ids[i + 1]))
                if r is not None and (best is None or r < best[0]):
                    best = (r, i)
            if best is None:
                break
            r, i = best
            ids[i : i + 2] = [256 + r]
        return ids

    def decode_bytes(self, ids: Sequence[int]) -> bytes:
        """Every id expanded to its bytes, left part before right part."""
        out = bytearray()
        for i in ids:
            if not 0 <= i < self.vocab_size:
                raise ValueError(f"id {i} is outside 0 .. {self.vocab_size - 1}")
            stack = [i]
            while stack:
                x = stack.pop()
                if x < 256:
                    out.append(x)
                else:
                    a, b = self.parts[x - 256]
                    stack.append(b)
                    stack.append(a)
        return bytes(out)

    def decode(self, ids: Sequence[int]) -> str:
        """decode_bytes, then UTF-8 with replacement."""
        return self.decode_bytes(ids).decode("utf-8", errors="replace")
