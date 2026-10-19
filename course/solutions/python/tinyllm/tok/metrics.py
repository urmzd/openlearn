"""Tokenizer metrics (L1.6): how well a vocabulary compresses text.

Every function takes any `Tokenizer` (the protocol of L1.1), so one table
compares bytes, char, BPE, WordPiece, and Unigram on the same text:

  fertility           tokens per word: how many pieces an average word costs
  bytes_per_token     UTF-8 bytes per token: compression (higher is better)
  byte_fallback_rate  share of tokens that are not a whole character: a
                      fragment of a multi-byte character, or the unknown id

`bytes_per_token` is also the exchange rate between a model's loss per
token and bits per byte (M11.2): bpb = nll_per_token / (ln 2 * bytes_per_token).

Contract: contracts/py/tinyllm/tok/metrics.pyi.
"""

from __future__ import annotations

from typing import Sequence

from tinyllm.tok.base import Tokenizer


def fertility(tok: Tokenizer, words: Sequence[str]) -> float:
    """sum over words of len(tok.encode(word)) / len(words); each word is
    encoded on its own, without special tokens. ValueError for no words."""
    if len(words) == 0:
        raise ValueError("fertility needs at least one word")
    return sum(len(tok.encode(w)) for w in words) / len(words)


def bytes_per_token(tok: Tokenizer, texts: Sequence[str]) -> float:
    """total UTF-8 bytes of texts / total tokens of their encodings (no
    special tokens). ValueError when the texts encode to no tokens."""
    n_bytes = sum(len(t.encode("utf-8")) for t in texts)
    n_tokens = sum(len(tok.encode(t)) for t in texts)
    if n_tokens == 0:
        raise ValueError("bytes_per_token needs texts that encode to at least one token")
    return n_bytes / n_tokens


def is_fallback(tok: Tokenizer, i: int) -> bool:
    """True when id i is the unknown id, or a token that is not whole
    characters: decoding it alone yields U+FFFD (a piece of a multi-byte
    UTF-8 sequence). Other special tokens are never fallbacks. A text that
    already holds U+FFFD counts too: that character marks text lost upstream."""
    if tok.unk_id is not None and i == tok.unk_id:
        return True
    if i in tok.special_ids.values():
        return False
    return "\ufffd" in tok.decode([i])


def byte_fallback_rate(tok: Tokenizer, texts: Sequence[str]) -> float:
    """fallback tokens / all tokens over the encodings of texts.
    ValueError when the texts encode to no tokens."""
    total = fallback = 0
    cache: dict[int, bool] = {}
    for t in texts:
        for i in tok.encode(t):
            total += 1
            if i not in cache:
                cache[i] = is_fallback(tok, i)
            fallback += cache[i]
    if total == 0:
        raise ValueError("byte_fallback_rate needs texts that encode to at least one token")
    return fallback / total
