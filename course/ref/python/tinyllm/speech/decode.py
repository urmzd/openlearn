"""Whisper decoding primitives (L14.4).

The logits seam accepts complete prefixes, including when beam search changes
branches. The model adapter reuses cross-attention for a window and only appends
self-attention state when the requested prefix extends the cached one.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import zlib
from typing import Protocol, Sequence

from tinyllm.num.rng import PCG32

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


@dataclass(frozen=True)
class DecodeOptions:
    task: str = "transcribe"
    language: str | None = None
    timestamps: bool = True
    beam_size: int = 0
    best_of: int = 5
    patience: float = 1.0
    temperatures: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    compression_ratio_threshold: float = 2.4
    logprob_threshold: float = -1.0
    no_speech_threshold: float = 0.6
    condition_on_previous_text: bool = True
    max_initial_timestamp: float = 1.0
    seed: int = 0
    max_tokens: int = 224
    suppress_tokens: tuple[int, ...] = (-1,)
    suppress_blank: bool = True

    def validate(self) -> None:
        if self.task not in ("transcribe", "translate"):
            raise ValueError("invalid decoding task")
        if any(
            type(x) is not int for x in (self.beam_size, self.best_of, self.max_tokens)
        ):
            raise ValueError("search sizes must be integers")
        if self.beam_size < 0 or self.best_of < 1 or self.max_tokens < 1:
            raise ValueError("invalid search sizes")
        if not math.isfinite(self.patience) or self.patience <= 0:
            raise ValueError("patience must be positive and finite")
        if self.beam_size and round(self.beam_size * self.patience) < 1:
            raise ValueError("patience leaves no finished candidates")
        if not self.temperatures or any(
            not math.isfinite(t) or t < 0 for t in self.temperatures
        ):
            raise ValueError("temperatures must be finite and nonnegative")
        if (
            not math.isfinite(self.max_initial_timestamp)
            or self.max_initial_timestamp < 0
        ):
            raise ValueError("max_initial_timestamp must be finite and nonnegative")
        if (
            not math.isfinite(self.compression_ratio_threshold)
            or self.compression_ratio_threshold <= 0
        ):
            raise ValueError("compression threshold must be positive and finite")
        if (
            not math.isfinite(self.logprob_threshold)
            or not 0 <= self.no_speech_threshold <= 1
        ):
            raise ValueError("invalid probability thresholds")


@dataclass
class Segment:
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: list[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float


def _logprobs(row: NDArray) -> NDArray:
    values = np.asarray(row, dtype=np.float64)
    if (
        values.ndim != 1
        or not values.size
        or np.any(np.isnan(values))
        or np.any(np.isposinf(values))
    ):
        raise ValueError("logits must be a vector without NaN or positive infinity")
    peak = np.max(values)
    if not np.isfinite(peak):
        raise ValueError("no legal token remains")
    shifted = values - peak
    return shifted - np.log(np.exp(shifted).sum())


def _filtered_logits(src, tok, initial, tokens, opts):
    row = np.array(src.step(initial + list(tokens)), dtype=np.float64, copy=True)
    # Validate before indexing, and mask control tokens independently of the
    # user suppress list. EOT remains available after the first position.
    _logprobs(row)
    blocked = set(opts.suppress_tokens)
    if -1 in blocked:
        blocked.remove(-1)
        blocked.update(tok.non_speech_tokens)
    for name in ("sot", "sot_prev", "sot_lm", "transcribe", "translate", "no_speech"):
        value = getattr(tok, name, None)
        if value is not None:
            blocked.add(value)
    if opts.suppress_blank and not tokens:
        blocked.update(tok.encode(" "))
        blocked.add(tok.eot)
    if any(type(t) is not int or not 0 <= t < row.size for t in blocked):
        raise ValueError("suppressed token outside the vocabulary")
    if (
        not 0 <= tok.eot < tok.timestamp_begin < row.size
        or not 0 <= tok.no_timestamps < tok.timestamp_begin
    ):
        raise ValueError("invalid tokenizer vocabulary geometry")
    row[list(blocked)] = -np.inf
    if opts.timestamps:
        return apply_timestamp_rules(
            row,
            initial + list(tokens),
            len(initial),
            tok,
            round(opts.max_initial_timestamp / 0.02),
        )
    row[tok.timestamp_begin :] = -np.inf
    row[tok.no_timestamps] = -np.inf
    return row


def decode_window(
    src: LogitsSource,
    tok,
    prompt: Sequence[int],
    opts: DecodeOptions,
    temperature: float,
    rng: PCG32,
) -> Segment:
    """Decode one window with greedy, beam search, or best-of sampling.

    `prompt` is preceding transcript context, not the SOT control sequence.
    The result retains timestamp ids but excludes EOT. Absolute segment times
    and long-form seeking belong to the transcribe loop.
    """
    opts.validate()
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and nonnegative")
    context = getattr(getattr(src, "model", None), "config", None)
    context_size = (
        context.n_text_ctx if context is not None else getattr(src, "context_size", 448)
    )
    initial = sot_sequence(tok, opts.language, opts.task, opts.timestamps)
    if len(prompt):
        if any(not isinstance(t, (int, np.integer)) or t < 0 for t in prompt):
            raise ValueError("prompt must contain nonnegative integer token ids")
        initial = [tok.sot_prev] + list(prompt[-(context_size // 2 - 1) :]) + initial
    limit = min(opts.max_tokens, context_size - len(initial))
    if limit < 1:
        raise ValueError("initial prompt leaves no decoding context")
    # Silence is evaluated at SOT before masking or conditioning on a language.
    silence = getattr(tok, "no_speech", None)
    no_speech = 0.0
    if silence is not None:
        first = _logprobs(src.step([tok.sot]))
        if not 0 <= silence < len(first):
            raise ValueError("no_speech token outside the vocabulary")
        no_speech = float(np.exp(first[silence]))

    # Each candidate stores generated ids including a terminating EOT and the
    # sum of untempered log probabilities. Stable token-id ordering breaks ties.
    finished: list[tuple[tuple[int, ...], float]] = []
    beam = opts.beam_size if temperature == 0 else 0
    if beam > 1:
        active = [((), 0.0)]
        wanted = round(beam * opts.patience)
        for _ in range(limit):
            candidates = {}
            for tokens, score in active:
                lp = _logprobs(_filtered_logits(src, tok, initial, tokens, opts))
                order = np.argsort(-lp, kind="stable")[: beam + 1]
                for token in order:
                    if np.isfinite(lp[token]):
                        ids = tokens + (int(token),)
                        candidates[ids] = score + float(lp[token])
            active = []
            ended = []
            for ids, score in sorted(
                candidates.items(), key=lambda pair: (-pair[1], pair[0])
            ):
                if ids[-1] == tok.eot:
                    ended.append((ids, score))
                else:
                    active.append((ids, score))
                    if len(active) == beam:
                        break
            finished.extend(ended[: max(0, wanted - len(finished))])
            if len(finished) >= wanted or not active:
                break
        if len(finished) < beam:
            for tokens, score in sorted(active, key=lambda pair: (-pair[1], pair[0])):
                finished.append((tokens + (tok.eot,), score))
                if len(finished) >= beam:
                    break
    else:
        for _ in range(opts.best_of if temperature > 0 else 1):
            tokens: tuple[int, ...] = ()
            score = 0.0
            for _ in range(limit):
                row = _filtered_logits(src, tok, initial, tokens, opts)
                lp = _logprobs(row)
                if temperature == 0:
                    token = int(np.argmax(lp))
                else:
                    probabilities = np.exp(_logprobs(row / temperature))
                    # PCG32 is shared with the Rust parity implementation.
                    cdf = np.cumsum(probabilities)
                    cdf[-1] = 1.0
                    token = int(np.searchsorted(cdf, rng.uniform(), side="right"))
                score += float(lp[token])
                tokens += (token,)
                if token == tok.eot:
                    break
            if tokens[-1] != tok.eot:
                tokens += (tok.eot,)
            finished.append((tokens, score))
    if not finished:
        raise ValueError("search produced no candidate")
    # Rank by generated length excluding EOT, as Whisper's default ranker does.
    tokens, score = max(finished, key=lambda pair: pair[1] / max(1, len(pair[0]) - 1))
    ids = list(tokens[:-1])
    text = tok.decode([t for t in ids if t < tok.eot]).strip()
    encoded = text.encode("utf-8")
    ratio = len(encoded) / len(zlib.compress(encoded))
    times = [0.02 * (t - tok.timestamp_begin) for t in ids if t >= tok.timestamp_begin]
    return Segment(
        0,
        0,
        times[0] if times else 0.0,
        times[-1] if len(times) > 1 else 0.0,
        text,
        ids,
        temperature,
        score / (len(ids) + 1),
        ratio,
        no_speech,
    )


def decode_with_fallback(src, tok, prompt, opts: DecodeOptions, rng: PCG32) -> Segment:
    """Retry only failed quality gates; likely silence does not need hotter draws."""
    opts.validate()
    for temperature in opts.temperatures:
        result = decode_window(src, tok, prompt, opts, temperature, rng)
        repetitive = result.compression_ratio > opts.compression_ratio_threshold
        uncertain = result.avg_logprob < opts.logprob_threshold
        silent = result.no_speech_prob > opts.no_speech_threshold and uncertain
        if silent or not (repetitive or uncertain):
            return result
    return result


def _window_segments(result: Segment, tok, offset: float, duration: float):
    """Return completed pieces and the next seek, in seconds relative to a window."""
    ids = result.tokens
    is_time = [t >= tok.timestamp_begin for t in ids]
    boundaries = [i for i in range(1, len(ids)) if is_time[i - 1] and is_time[i]]
    terminal_time = len(ids) >= 2 and not is_time[-2] and is_time[-1]
    pieces = []
    advance = duration
    if boundaries:
        cuts = boundaries + ([len(ids)] if terminal_time else [])
        start = 0
        for end in cuts:
            piece = ids[start:end]
            if (
                len(piece) >= 2
                and piece[0] >= tok.timestamp_begin
                and piece[-1] >= tok.timestamp_begin
            ):
                pieces.append(
                    (
                        piece,
                        0.02 * (piece[0] - tok.timestamp_begin),
                        0.02 * (piece[-1] - tok.timestamp_begin),
                    )
                )
            start = end
        if not terminal_time:
            advance = 0.02 * (ids[boundaries[-1] - 1] - tok.timestamp_begin)
    elif ids:
        end = duration
        times = [t for t in ids if t >= tok.timestamp_begin]
        if times and times[-1] != tok.timestamp_begin:
            end = min(duration, 0.02 * (times[-1] - tok.timestamp_begin))
        pieces.append((ids, 0.0, end))
    segments = []
    for piece, start, end in pieces:
        start, end = min(duration, max(0.0, start)), min(duration, max(0.0, end))
        text = tok.decode([t for t in piece if t < tok.eot]).strip()
        if end > start and text:
            segments.append(
                Segment(
                    0,
                    0,
                    offset + start,
                    offset + end,
                    text,
                    list(piece),
                    result.temperature,
                    result.avg_logprob,
                    result.compression_ratio,
                    result.no_speech_prob,
                )
            )
    # A truncated or malformed timestamp trace must never spin on the same window.
    return segments, min(duration, advance) if advance > 0 else duration


def _transcribe_windows(
    source_for_seek, tok, duration: float, window_frames: int, opts: DecodeOptions
):
    """Shared long-form policy for model windows and recorded-logit sources.

    Seek units are 10 ms mel frames. A factory call encodes exactly one window;
    fallback reuses that source and therefore its cross-attention state.
    """
    from dataclasses import replace

    opts.validate()
    if not math.isfinite(duration) or duration < 0 or window_frames < 1:
        raise ValueError("invalid audio duration or window size")
    if duration == 0:
        return []
    seek = 0
    history: list[int] = []
    results = []
    rng = PCG32(opts.seed).substream("sample")
    while seek * 0.01 < duration:
        source = source_for_seek(seek)
        if opts.language is None and tok.all_language_codes:
            language, _ = language_probs(source, tok)
            opts = replace(opts, language=language)
        remaining = min(window_frames * 0.01, duration - seek * 0.01)
        result = decode_with_fallback(source, tok, history, opts, rng)
        silent = (
            result.no_speech_prob > opts.no_speech_threshold
            and result.avg_logprob <= opts.logprob_threshold
        )
        if silent:
            seek += window_frames
            continue
        segments, advance = _window_segments(result, tok, seek * 0.01, remaining)
        for segment in segments:
            segment.id = len(results)
            segment.seek = seek
            results.append(segment)
            history.extend(segment.tokens)
        if not opts.condition_on_previous_text or result.temperature > 0.5:
            history.clear()
        seek += max(1, round(advance / 0.01))
    return results


def transcribe(
    model: Whisper, tok, audio_16k: NDArray, opts: DecodeOptions
) -> list[Segment]:
    """Transcribe mono 16 kHz samples with one encoder call per sought window."""
    from tinyllm.speech.frontend import FeatureConfig, whisper_log_mel

    opts.validate()
    audio = np.asarray(audio_16k, dtype=np.float32)
    if audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError("audio must be a finite mono 16 kHz array")
    if not audio.size:
        return []
    if audio.size > 30 * 60 * 16000:
        raise ValueError("audio exceeds the 30 minute input limit")
    window_frames = 2 * model.config.n_audio_ctx
    # Analyze the full recording once: normalizing independently per window
    # would change quiet-window features and the reference timestamp decisions.
    padded = np.pad(audio, (0, max(400, window_frames * 160)))
    mel = whisper_log_mel(padded, FeatureConfig(feature_size=model.config.n_mels))

    def source_for_seek(seek):
        features = mel[:, seek : seek + window_frames]
        if features.shape[1] != window_frames:
            raise ValueError("feature padding failed to supply a complete window")
        return WhisperLogitsSource(model, model.encode(features[None]))

    return _transcribe_windows(
        source_for_seek, tok, audio.size / 16000, window_frames, opts
    )
