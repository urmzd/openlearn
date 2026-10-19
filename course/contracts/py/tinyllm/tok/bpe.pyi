# contracts/py/tinyllm/tok/bpe.pyi (L1.2)
# chapter: content/courses/build-an-llm/02-tokenizers/02-byte-level-bpe.md
#
# Byte-level BPE, GPT-2 compatible: ids equal tiktoken's and Hugging Face
# tokenizers' on GPT-2 and SmolLM2 files (formats/tokenizer.md, BPE).
# Uses M05.2 (tinyllm.tok.bytes_unicode) for the byte map and M06.2
# (tinyllm.tok.trie) to split out added tokens.
from typing import Iterable, Mapping, Optional, Sequence

from tinyllm.tok.base import Tokenizer

class BPETokenizer(Tokenizer):
    vocab: dict[str, int]                 # token string (byte-map alphabet) -> id
    merges: list[tuple[str, str]]         # merges[r] has rank r: lower merges first
    added: dict[str, int]                 # added tokens (special or not) -> id
    pre_tokenizer: dict                   # the tokenizer.json pre_tokenizer object
    def __init__(
        self,
        vocab: Mapping[str, int],
        merges: Sequence[tuple[str, str]],
        added_tokens: Sequence[tuple[str, int, bool]] = (),
        pre_tokenizer: Optional[Mapping] = None,
    ) -> None:
        """added_tokens are (content, id, special). pre_tokenizer defaults to
        GPT-2's ByteLevel; Digits, ByteLevel (use_regex true), or a Sequence of
        them with ByteLevel last are accepted, anything else is ValueError.
        ValueError when the ids of vocab and added tokens leave a gap."""
    @classmethod
    def from_gpt2(cls, vocab_json: str, merges_txt: str) -> "BPETokenizer":
        """GPT-2's vocab.json and merges.txt. merges.txt lines are "left right";
        "#version" lines and empty lines are skipped. <|endoftext|>, when in
        vocab.json, is an added special token."""
    @classmethod
    def from_hf_json(cls, tokenizer_json: str) -> "BPETokenizer":
        """A tokenizer.json in the BPE subset of formats/tokenizer.md: honors
        the declared pre_tokenizer sequence and added_tokens. model.type may be
        absent (GPT-2's file) when model.merges is present. ValueError, naming
        the field, for anything outside the subset (a normalizer, dropout,
        unk_token, byte_fallback, a TemplateProcessing post_processor, ...)."""
    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int,
        specials: Sequence[str] = (),
        min_freq: int = 2,
    ) -> "BPETokenizer":
        """Hugging Face BpeTrainer with the ByteLevel alphabet: ids 0.. are the
        specials (removed from the training text), then the 256 byte symbols
        in code point order, then one token per merge. Each step merges the
        pair with the highest count over the GPT-2 pre-tokens; ties go to the
        smaller (left id, right id). Stops at vocab_size or when the best count
        is below min_freq. ValueError when vocab_size < 256 + len(specials) or
        min_freq < 1."""
    def encode(self, text: str, add_special: bool = False) -> list[int]:
        """Added tokens first (leftmost longest), then the pre_tokenizer, the
        byte map, and merges by lowest rank (leftmost on ties). A byte symbol
        missing from vocab is dropped, as Hugging Face does with no unk_token.
        add_special has no effect (no post-processor template)."""
    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        """[encode(t) for t in texts]."""
    def decode(self, ids: Sequence[int], skip_special: bool = False) -> str:
        """Token bytes concatenated (added tokens as their UTF-8), decoded as
        UTF-8 with replacement; skip_special drops special added tokens."""
    def token_to_id(self, s: str) -> Optional[int]: ...
    def id_to_token(self, i: int) -> str: ...
    def save(self, dir: str) -> None:
        """dir/tokenizer.json in the HF subset; Hugging Face tokenizers loads it."""
    @classmethod
    def load(cls, dir: str) -> "BPETokenizer": ...
