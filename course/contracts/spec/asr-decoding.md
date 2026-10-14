<!-- chapter: ml/08-tinyllm/p14-speech/04-whisper-decoding.md -->
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

## Window search

`DecodeOptions` defaults match DESIGN.md. Additional explicit bounds are
`max_tokens=224`, `suppress_tokens=(-1,)` and `suppress_blank=True`. Minus one
expands to the tokenizer's non-speech list. SOT, previous-context, task and
no-speech control tokens are always suppressed during generation. The initial
blank and EOT are suppressed when `suppress_blank` is enabled.

Previous transcript tokens are prefixed by `sot_prev`, truncated to at most
half the model context minus one, before the current SOT sequence. Generated
length is bounded by both `max_tokens` and remaining model context. A result
excludes EOT and preceding context. Timestamp ids remain in `tokens`; `text`
decodes only ordinary text ids and strips boundary whitespace.

Temperature zero uses greedy selection when beam size is zero or one. Larger
beams retain the best cumulative-log-probability live paths and collect up to
`round(beam_size * patience)` completed paths. At the length limit, unfinished
paths can fill missing candidates with an unscored EOT. Positive temperatures
use `best_of` independent trajectories with the supplied course PCG32 stream.
Sampling uses the temperature-scaled distribution; scores always use the
untempered filtered distribution. Candidate selection divides cumulative log
probability by generated length excluding EOT. `avg_logprob` divides by that
length plus one. Stable ascending token ids break equal-logit choices.

No-speech probability comes from the unfiltered SOT row. Compression ratio is
UTF-8 text byte length divided by `len(zlib.compress(text_bytes))`. Window
results have relative timestamp endpoints; the long-form loop must still turn
them into duration-bounded absolute segments and decide fallback/seek behavior.

## Fallback and long-form seeking

Fallback tries `temperatures` in order until compression ratio and average
log probability pass their thresholds. A window with high no-speech probability
and low average log probability stops retries. The long-form loop skips that
window; confident speech overrides a high no-speech probability.

`transcribe` accepts finite mono 16 kHz samples, at most 30 minutes. It computes
log-mel features once for the full recording with a padded final window. Each
seek encodes its feature window once; fallback attempts reuse the same cross-KV.
Language detection runs only for the first window when no language was supplied.
Seek is measured in 10 ms feature frames; output times are absolute seconds.

Consecutive timestamp tokens delimit completed segments. An incomplete trailing
segment resumes at the last completed endpoint. A terminal timestamp consumes
the full current window. Empty or zero-duration pieces are omitted, endpoints
are clamped to the recording, and every iteration advances at least one frame.
Only accepted segment tokens enter prompt history. Disabled conditioning or a
selected temperature above 0.5 clears that history before the next window.

## Tokenizer and command line

`WhisperTokenizer.from_pretrained` reads local HF `tokenizer.json` and optional
`generation_config.json`, reusing the learner's byte BPE. It validates every
byte symbol and all 1501 contiguous timestamp ids. Both no-speech spellings,
`<|nospeech|>` and HF's `<|nocaptions|>`, are supported. Template post-processing
is not executed: the decode policy constructs control sequences explicitly.
The generation config supplies multilingual mode and suppression ids; without
it the standard 51864-token English vocabulary is recognized. Other vocabularies
must contain language controls. Ordinary transcript input rejects special-token
syntax instead of allowing it to alter the decoder prompt.

`tinyllm transcribe --model DIR --audio FILE.wav` loads model and tokenizer from
the same directory, checks vocabulary agreement, mixes/resamples WAV input to
16 kHz and emits JSON text plus segments. `--out` also writes that same JSON.
`--greedy` disables fallback, while the default temperature schedule retains it.
`--beam`, `--best-of`, `--patience`, `--language`, `--task`, `--max-tokens`,
`--seed`, `--no-timestamps` and `--no-condition-on-previous-text` expose the
policy settings. `--detect-language` returns language probabilities for the
first window. Invalid inputs return exit code 2 with an error on stderr.

## Recorded-logit playback

`tinyllm transcribe --logits-trace traces.npz --opts opts.json` uses the same
fallback and long-form loop without loading weights. Options contain a local
`tokenizer` directory (relative to the options file), `duration` in seconds,
`window_frames` in 10 ms frames, optional `context_size` (448), and a `decode`
object matching `DecodeOptions`. Unknown option keys fail.

NPZ arrays use this v1 layout, loaded with pickle disabled:

| Array | Shape | Meaning |
|---|---|---|
| `prefix_offsets` | `[R+1]` integer | Strictly increasing boundaries, from 0 to N |
| `prefix_ids` | `[N]` integer | Concatenated complete, nonempty prefixes |
| `seeks` | `[R]` integer, optional | Nonnegative mel-frame offsets; default zero |
| `logits` | `[R,V]` float | Dense next-token rows |
| `token_ids`, `values` | `[R,K]` integer/float | Alternative sparse top-k rows |
| `vocab_size` | scalar integer | Required only for sparse rows |

Dense and sparse encodings cannot be mixed. Sparse ids must be unique within
a row; unrecorded vocabulary entries are negative infinity. Each row must have
a finite choice. Duplicate seek/prefix keys, missing queried prefixes, NaN,
positive infinity, invalid offsets, and vocabulary/context violations fail.
Playback returns copies so timestamp masks cannot alter the recording. Sparse
rows remain sparse in storage and expand only when queried.

A recording must include every branch visited by its selected search settings,
SOT language/silence queries, and all fallback attempts. Missing branches are
errors, never extrapolated logits. Python and Rust can therefore consume the
same recordings without re-running an encoder or downloading model weights.

## Completion status

L14.4 is registered with public contracts, a six-beat chapter, semantic mutants
and generated-transcript evidence consumers. Full course verification remains
the completion gate. Independent timestamp and greedy/beam fixtures pass;
real-checkpoint tokenizer and long-form transcript parity require separate
oracle evidence. Registration does not establish production ASR accuracy.
