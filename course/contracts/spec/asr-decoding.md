# Speech decoding policy

L14.4 owns token selection and long-form seeking; L14.3 owns model execution.
A `LogitsSource.step(prefix)` returns the next-token row for a complete,
nonempty prefix. Prefixes may repeat, shrink, or switch branches. A source
must not mutate previously returned rows. Whisper's adapter shares cross-KV
within an audio window, appends self-KV only for extensions, and resets self-KV
on branch changes or failed decoder calls. Its storage remains bounded by one
model context, rather than retaining every explored beam.

## Initial controls and language

Multilingual prompts start with SOT, the selected language token, then the
transcribe or translate token. Text-only generation appends `no_timestamps`.
An English-only tokenizer emits SOT without language/task controls and rejects
translation or non-English language selection. Unknown tasks and languages
are errors. Language detection queries the SOT row without a task or language
prefix and normalizes over language tokens only. Ties choose the first language
in tokenizer order.

## Timestamp state machine

Timestamp ids form the vocabulary suffix beginning at `timestamp_begin`.
Each increment represents 20 ms for standard Whisper. The caller identifies
`sample_begin`, excluding prompt and control tokens from all rules:

1. Suppress `no_timestamps` during timestamp decoding.
2. The initial generated token must be a timestamp, at most
   `timestamp_begin + max_initial_index` (inclusive).
3. After a start timestamp, suppress timestamps so text or EOT follows.
4. After text followed by an end timestamp, suppress ordinary text. Another
   timestamp may begin the next segment; EOT may finish the window.
5. A segment end must exceed its start. A new segment start may equal the
   preceding end. Earlier timestamp ids are masked.
6. After structural masking, compare total timestamp probability with the
   largest individual non-timestamp probability. If timestamp mass is greater,
   suppress all non-timestamp ids. Equality does not force a timestamp.

The filter returns a separate float64 row, permits negative infinity for
suppressed tokens, rejects NaNs and positive infinity, and fails if no legal
token remains. It does not suppress other control tokens or enforce the actual
recording duration; those are responsibilities of the decode-window policy.

The state machine follows the reference
[OpenAI Whisper decoding policy](https://github.com/openai/whisper/blob/main/whisper/decoding.py).
Recorded-logit parity must check it independently of the model adapter.

## Completion status

This contract currently covers the logits adapter, initial controls, language
selection and timestamp filtering. L14.4 is not yet registered or complete.
The tokenizer loader, decode-window search, suppress lists, temperature fallback,
no-speech handling, prompt conditioning, long-form segments, CLI, independent
oracles, mutants and six-beat chapter remain required by DESIGN.md.
