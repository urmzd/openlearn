# chapter: ml/08-tinyllm/p14-speech/04-whisper-decoding.md
"""L14.4: dense/sparse NPZ v1 in spec/asr-decoding.md."""
from tinyllm.speech.decode import LogitsSource

class RecordedLogits:
    context_size: int
    vocab_size: int
    def __init__(self, path: str, *, context_size: int = 448) -> None:
        """Load without pickle. Reject malformed prefixes, duplicate keys,
        invalid logits and sparse ids; retain sparse storage until queried."""
    def at_seek(self, seek: int) -> LogitsSource:
        """Nonnegative integer frame offset. Each query returns an independent
        row for its exact full prefix; missing rows raise ValueError."""
