# contracts/py/tinyllm/tok/metrics.pyi (L1.6)
# chapter: content/courses/build-an-llm/p01-tokenizers/06-tokenizer-metrics.md
#
# Compression metrics over any Tokenizer (L1.1). Inputs are encoded without
# special tokens. bits per byte (M11.2) = nll_per_token / (ln 2 * bytes_per_token).
from typing import Sequence

from tinyllm.tok.base import Tokenizer

def fertility(tok: Tokenizer, words: Sequence[str]) -> float:
    """Mean tokens per word, each word encoded on its own. ValueError for no words."""
def bytes_per_token(tok: Tokenizer, texts: Sequence[str]) -> float:
    """Total UTF-8 bytes / total tokens. ValueError when there are no tokens."""
def is_fallback(tok: Tokenizer, i: int) -> bool:
    """i == tok.unk_id, or i is not special and tok.decode([i]) contains U+FFFD."""
def byte_fallback_rate(tok: Tokenizer, texts: Sequence[str]) -> float:
    """Fallback tokens / all tokens. ValueError when there are no tokens."""
