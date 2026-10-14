# chapter: ml/08-tinyllm/p14-speech/04-whisper-decoding.md
"""L14.4: contracts/spec/asr-decoding.md is normative for token policy."""
from dataclasses import dataclass
from typing import Protocol, Sequence
from numpy.typing import NDArray
from tinyllm.num.rng import PCG32
from tinyllm.speech.decoder import CrossKV, Whisper

class LogitsSource(Protocol):
    def step(self, prefix: Sequence[int]) -> NDArray:
        """Next logits [V] for the complete prefix; returned rows are caller-owned."""

class WhisperLogitsSource:
    def __init__(self, model: Whisper, xkv: CrossKV) -> None: ...
    def step(self, prefix: Sequence[int]) -> NDArray:
        """Reuse self-KV only for prefix extensions; reset on branch/rewind/error.
        Cross-KV is shared for this audio window; duplicate queries return copies."""

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
    def validate(self) -> None: ...

@dataclass
class Segment:
    id: int
    seek: int  # 10 ms mel frames
    start: float  # absolute seconds for transcribe; relative for decode_window
    end: float
    text: str
    tokens: list[int]  # excludes EOT, retains timestamps
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float

def sot_sequence(tok, language: str | None, task: str, timestamps: bool) -> list[int]: ...
def language_probs(src: LogitsSource, tok) -> tuple[str, dict[str, float]]:
    """Language-only softmax of the unconditioned SOT row."""
def detect_language(model: Whisper, mel_window: NDArray, tok) -> tuple[str, dict[str, float]]: ...
def apply_timestamp_rules(logits: NDArray, prefix: Sequence[int], sample_begin: int, tok, max_initial_index: int) -> NDArray:
    """Copied row: initial bound, timestamp pairing/progress and probability-mass rule.
    Reject malformed geometry/nonfinite choices; never modify the source row."""
def decode_window(src: LogitsSource, tok, prompt: Sequence[int], opts: DecodeOptions, temperature: float, rng: PCG32) -> Segment:
    """Greedy, beam with patience, or best-of sampling over one encoded window.
    Prompt is previous context, not control tokens. Scores use untempered logits."""
def decode_with_fallback(src, tok, prompt, opts: DecodeOptions, rng: PCG32) -> Segment:
    """Retry failed compression/logprob gates, stopping for probable silence."""
def transcribe(model: Whisper, tok, audio_16k: NDArray, opts: DecodeOptions) -> list[Segment]:
    """Finite mono 16 kHz input, <=30 minutes. Encode once per sought window,
    reuse cross-KV for fallback; duration-bound absolute segments with stable ids.
    Empty audio yields []; invalid shape/nonfinite audio raises ValueError."""
