# contracts/py/tinyllm/tok/base.pyi (L1.1)
# chapter: content/courses/ai-engineering/07-tokenizers/01-tokenizer-protocol-and-char.md
#
# The Tokenizer protocol every tokenizer implements (bytes, char, BPE,
# WordPiece, Unigram, and the Python BPE from L1.5), and the tracer's
# byte tokenizer (D32, formats/tokenizer.md), whose token strings come from
# M05.2 (tinyllm.tok.bytes_unicode). Ids are ints in [0, vocab_size).
from typing import Optional, Protocol, Sequence, runtime_checkable

@runtime_checkable
class Tokenizer(Protocol):
    vocab_size: int              # every id in [0, vocab_size) decodes; no gaps
    special_ids: dict[str, int]  # special token text -> id ({} for bytes)
    unk_id: Optional[int]        # the id unknown input maps to, or None when every input encodes

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        """Ids of text. add_special adds the tokenizer's template (BOS/EOS for
        char, [CLS] ... [SEP] for WordPiece); it never changes the ids of text."""
    def decode(self, ids: Sequence[int], skip_special: bool = False) -> str:
        """Text of ids; skip_special drops the ids in special_ids.
        ValueError for an id outside [0, vocab_size) or a non-integer."""
    def token_to_id(self, s: str) -> Optional[int]:
        """The id whose token string is s, or None."""
    def id_to_token(self, i: int) -> str:
        """The token string of id i. ValueError outside [0, vocab_size)."""
    def save(self, dir: str) -> None:
        """Write the tokenizer's file into dir (created if missing)."""
    @classmethod
    def load(cls, dir: str) -> "Tokenizer":
        """Read what save wrote: load(d) encodes exactly like the saved one."""

def check_ids(ids: Sequence[int], vocab_size: int) -> list[int]:
    """ids as a list of ints. ValueError for a bool, a non-integer, or an id
    outside [0, vocab_size). Every decode starts with it."""

class ByteTokenizer:
    """The identity byte tokenizer: id = byte value, vocab_size 256, no
    specials, unk_id None. encode(t) = list(t.encode("utf-8")); decode =
    bytes(ids).decode("utf-8", "replace"). Token strings are the M05.2 GPT-2
    byte map (id 32 is "Ġ"), the alphabet byte-level BPE starts from."""
    vocab_size: int
    special_ids: dict[str, int]
    unk_id: Optional[int]
    def __init__(self) -> None: ...
    def encode(self, text: str, add_special: bool = False) -> list[int]: ...
    def decode(self, ids: Sequence[int], skip_special: bool = False) -> str: ...
    def token_to_id(self, s: str) -> Optional[int]:
        """unicode_to_bytes()[s] (M05.2), or None for a string outside the byte map."""
    def id_to_token(self, i: int) -> str: ...
    def save(self, dir: str) -> None:
        """Creates dir and writes nothing: config.json says tl_tokenizer = "bytes"."""
    @classmethod
    def load(cls, dir: str) -> "ByteTokenizer": ...
