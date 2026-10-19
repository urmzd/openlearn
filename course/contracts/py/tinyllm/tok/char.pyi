# contracts/py/tinyllm/tok/char.pyi (L1.1)
# chapter: content/courses/build-an-llm/02-tokenizers/01-tokenizer-protocol-and-char.md
#
# One id per Unicode code point; saved as tinyllm_char.json
# (formats/tokenizer.md, formats/tokenizer-char.schema.json).
from typing import Iterable, Mapping, Optional, Sequence

from tinyllm.tok.base import Tokenizer

class CharTokenizer(Tokenizer):
    vocab: list[str]             # vocab[i] = the text of id i
    roles: dict[str, int]        # the file's `specials`: "unk" (always), "bos", "eos", "pad" -> id
    def __init__(self, vocab: Sequence[str], specials: Mapping[str, int]) -> None:
        """ValueError unless vocab entries are unique, specials names `unk`,
        every role is one of unk/bos/eos/pad with an id inside vocab, and every
        entry that is not a special is exactly one code point."""
    @classmethod
    def train(cls, texts: Iterable[str], specials: Sequence[str] = ()) -> "CharTokenizer":
        """vocab = ["<unk>"] + specials (each "<bos>", "<eos>", or "<pad>", in
        the order given, duplicates once) + every code point of texts sorted by
        code point. No normalization: "é" and "e" + U+0301 are different.
        ValueError for any other special text."""
    def encode(self, text: str, add_special: bool = False) -> list[int]:
        """One id per code point; an unseen code point is unk_id. Special
        texts in the input are ordinary characters. add_special puts bos
        first and eos last when those roles exist."""
    def decode(self, ids: Sequence[int], skip_special: bool = False) -> str:
        """"".join(vocab[i]); skip_special drops every role id (unk included)."""
    def token_to_id(self, s: str) -> Optional[int]: ...
    def id_to_token(self, i: int) -> str: ...
    def save(self, dir: str) -> None:
        """dir/tinyllm_char.json: {"type": "char", "vocab": [...], "specials": {...}}, UTF-8."""
    @classmethod
    def load(cls, dir: str) -> "CharTokenizer":
        """ValueError for a file that is not a valid tinyllm_char.json."""
