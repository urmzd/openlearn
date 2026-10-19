# Speech recognition

Build WAV and log-mel processing, Whisper's encoder and decoder, then token
search, timestamps and long-form transcription. Evaluation keeps generated
transcripts separate from teacher-forced model diagnostics.

Follow [Pass 12](../paths/course-p12-multimodal/) for the prerequisites.

## Chapters

| # | Module | Chapter | Kind | Pass |
|---|---|---|---|---|
| 1 | `L14.1` | [Audio frontend: WAV parsing and Whisper's log-mel](01-audio-frontend.md) | build | 12 |
| 2 | `L14.2` | [Whisper encoder: conv stem, sinusoids, pre-LN blocks](02-whisper-encoder.md) | build | 12 |
| 3 | `L14.3` | [Whisper decoder and checkpoint weights](03-whisper-decoder-and-weights.md) | build | 12 |
| 4 | `L14.4` | [Whisper decoding and long-form transcription](04-whisper-decoding.md) | build | 12 |
| 5 | `L14.5` | [Multimodal eval metrics: WER, CER, VQA, captions, zero-shot](05-multimodal-eval-metrics.md) | build | 12 |
