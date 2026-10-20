# chapter: content/courses/ai-engineering/15-speech/04-whisper-decoding.md
"""L14.4: the course's BPE with explicit Whisper controls, no automatic template."""
from typing import Sequence
from tinyllm.tok.bpe import BPETokenizer

class WhisperTokenizer:
    bpe: BPETokenizer
    language: str | None
    sot: int
    eot: int
    sot_prev: int
    sot_lm: int
    transcribe: int
    translate: int
    no_speech: int
    no_timestamps: int
    timestamp_begin: int
    all_language_tokens: tuple[int, ...]
    all_language_codes: tuple[str, ...]
    non_speech_tokens: tuple[int, ...]
    def __init__(self, bpe: BPETokenizer, *, multilingual: bool = True, language: str | None = None, suppress_tokens: Sequence[int] | None = None) -> None:
        """Require all bytes, complete timestamp suffix, and required controls.
        Accept HF nocaptions as the no_speech alias. None derives suppression."""
    def encode(self, text: str) -> list[int]:
        """Ordinary byte BPE, rejecting special-token syntax in transcript text."""
    def decode(self, ids: Sequence[int]) -> str:
        """Byte BPE decode, omitting timestamp ids."""
    @classmethod
    def from_pretrained(cls, directory: str, *, language: str | None = None) -> WhisperTokenizer:
        """Local tokenizer.json plus optional generation_config.json; no downloads.
        Validate supported byte-BPE settings, control geometry and language."""
