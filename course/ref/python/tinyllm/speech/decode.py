"""Whisper decoding primitives (L14.4).

The logits seam accepts complete prefixes, including when beam search changes
branches. The model adapter reuses cross-attention for a window and only appends
self-attention state when the requested prefix extends the cached one.
"""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from tinyllm.speech.decoder import CrossKV, Whisper


class LogitsSource(Protocol):
    def step(self, prefix: Sequence[int]) -> NDArray:
        """Next-token logits [vocab] for the entire nonempty prefix."""
        ...


class WhisperLogitsSource:
    """One audio window, one bounded decoder cache, arbitrary prefix queries."""

    def __init__(self, model: Whisper, xkv: CrossKV) -> None:
        self.model = model
        self.xkv = xkv
        self.cache = model.new_cache()
        self.prefix: tuple[int, ...] = ()
        self.last: NDArray | None = None

    def step(self, prefix: Sequence[int]) -> NDArray:
        ids = np.asarray(prefix)
        if (
            ids.ndim != 1
            or not np.issubdtype(ids.dtype, np.integer)
            or not 0 < ids.size <= self.model.config.n_text_ctx
            or np.any(ids < 0)
            or np.any(ids >= self.model.config.vocab)
        ):
            raise ValueError("prefix must contain valid token ids within the context")
        key = tuple(int(x) for x in ids)
        if key == self.prefix and self.last is not None:
            return self.last.copy()
        extends = key[: len(self.prefix)] == self.prefix
        if not extends:
            self.cache = self.model.new_cache()
        offset = len(self.prefix) if extends else 0
        try:
            row = self.model.decoder(ids[offset:], self.xkv, self.cache).data[-1]
        except Exception:
            # A failing layer may already have appended earlier layers. Never
            # reuse that partial state on the next request.
            self.cache = self.model.new_cache()
            self.prefix = ()
            self.last = None
            raise
        self.prefix = key
        self.last = np.asarray(row, dtype=np.float32).copy()
        return self.last.copy()


def sot_sequence(tok, language: str | None, task: str, timestamps: bool) -> list[int]:
    """Construct explicit language/task controls without changing the tokenizer."""
    if task not in ("transcribe", "translate"):
        raise ValueError("task must be transcribe or translate")
    languages = dict(zip(tok.all_language_codes, tok.all_language_tokens, strict=True))
    result = [tok.sot]
    if languages:
        language = language or getattr(tok, "language", None) or "en"
        if language not in languages:
            raise ValueError(f"unsupported language: {language}")
        result.extend([languages[language], getattr(tok, task)])
    elif language not in (None, "en") or task != "transcribe":
        raise ValueError("English-only tokenizers cannot translate or select languages")
    if not timestamps:
        result.append(tok.no_timestamps)
    return result


def language_probs(src: LogitsSource, tok) -> tuple[str, dict[str, float]]:
    """Normalize only language-token logits after SOT, excluding all text ids."""
    codes = tuple(tok.all_language_codes)
    ids = np.asarray(tok.all_language_tokens, dtype=np.int64)
    if (
        not codes
        or len(codes) != len(ids)
        or len(set(codes)) != len(codes)
        or len(set(ids.tolist())) != len(ids)
    ):
        raise ValueError("language detection requires unique language tokens and codes")
    logits = np.asarray(src.step([tok.sot]), dtype=np.float64)
    if logits.ndim != 1 or np.any(ids < 0) or np.any(ids >= len(logits)):
        raise ValueError("language ids must index the logits vocabulary")
    values = logits[ids]
    if (
        np.any(np.isnan(values))
        or np.any(np.isposinf(values))
        or not np.isfinite(values).any()
    ):
        raise ValueError("language logits have no finite probability distribution")
    weights = np.exp(values - np.max(values))
    probs = dict(zip(codes, (weights / weights.sum()).tolist(), strict=True))
    return codes[int(np.argmax(weights))], probs


def detect_language(
    model: Whisper, mel_window: NDArray, tok
) -> tuple[str, dict[str, float]]:
    """Encode one window, then detect language using the unconditioned SOT row."""
    mel = np.asarray(mel_window, dtype=np.float32)
    if mel.ndim == 2:
        mel = mel[None]
    if mel.ndim != 3 or mel.shape[0] != 1:
        raise ValueError("language detection accepts exactly one audio window")
    return language_probs(WhisperLogitsSource(model, model.encode(mel)), tok)


def apply_timestamp_rules(
    logits: NDArray,
    prefix: Sequence[int],
    sample_begin: int,
    tok,
    max_initial_index: int,
) -> NDArray:
    """Return masked logits without changing a recorded source's row.

    Timestamp ids are contiguous from timestamp_begin to the vocabulary end.
    Segment ends advance strictly; a following start may equal the prior end.
    Timestamp probability mass, not the best single timestamp, competes with
    the best text token. See contracts/spec/asr-decoding.md.
    """
    row = np.asarray(logits, dtype=np.float64).copy()
    begin = tok.timestamp_begin
    if (
        row.ndim != 1
        or not 0 <= tok.eot < begin < row.size
        or not 0 <= tok.no_timestamps < begin
        or not 0 <= sample_begin <= len(prefix)
        or max_initial_index < 0
        or np.any(np.isnan(row))
        or np.any(np.isposinf(row))
    ):
        raise ValueError("invalid timestamp geometry, prefix, or logits")
    generated = list(prefix[sample_begin:])
    if any(
        not isinstance(t, (int, np.integer)) or not 0 <= t < row.size for t in generated
    ):
        raise ValueError("generated token outside the vocabulary")
    row[tok.no_timestamps] = -np.inf
    last_is_time = bool(generated) and generated[-1] >= begin
    follows_time = len(generated) < 2 or generated[-2] >= begin
    if last_is_time:
        if follows_time:
            row[begin:] = -np.inf
        else:
            row[: tok.eot] = -np.inf
    times = [t for t in generated if t >= begin]
    if times:
        minimum = times[-1] + (0 if last_is_time and not follows_time else 1)
        row[begin:minimum] = -np.inf
    if not generated:
        row[:begin] = -np.inf
        row[begin + max_initial_index + 1 :] = -np.inf
    peak = np.max(row)
    if not np.isfinite(peak):
        raise ValueError("timestamp rules leave no legal token")
    # Subtract a common peak before exponentiation; the normalizing constant
    # cancels from the comparison, including when all text is masked.
    time_mass = np.exp(row[begin:] - peak).sum()
    text_mass = np.exp(np.max(row[:begin]) - peak)
    if time_mass > text_mass:
        row[:begin] = -np.inf
    return row
