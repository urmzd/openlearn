<!-- ol:module L14.4 -->
# Whisper decoding and long-form transcription

## Overview

| | |
|---|---|
| **Module** | `L14.4` · build · Python · Pass 12 · 6 to 8 h |
| **You build** | `python/tinyllm/speech/decode.py`, `tokenizer.py`, `trace.py` |
| **Contract** | `course/contracts/py/tinyllm/speech/decode.pyi`, `course/contracts/py/tinyllm/speech/tokenizer.pyi`, `course/contracts/py/tinyllm/speech/trace.pyi`; `course/contracts/spec/asr-decoding.md` |
| **Tests** | `course/tests/L14.4/` |
| **Needs** | `L14.3`, `L14.1`, `L1.2`, `M05.2`, `M06.3`. Reading: `L8.1`, `L4.4` |
| **Used by** | `ethics.08` generated transcript evidence |
| **Milestone** | `MS-L14` |
| **Optional depth** | Compare beam cache reuse with branch recomputation without changing token decisions. |

## Key Takeaways

- Model logits do not define transcription policy: control tokens, timestamp constraints and search do.
- Keep probability scores before temperature scaling for fallback decisions.
- A completed timestamp boundary controls long-form seeking; an incomplete tail must be revisited.

## How to work this chapter

```bash
ol start L14.4
ol tests L14.4
ol check L14.4 --ref-deps
ol diff L14.4
```

## 1. Why now

The encoder and decoder can score any prefix. Turning those scores into a transcript requires decisions about language, silence, timestamps, search and where to continue in a recording. Your implementation exposes the same policy to a real Whisper window and a recording of logits. This separation makes Python/Rust parity possible without repeatedly downloading weights or running an encoder.

## 2. Principles

Load byte BPE from the checkpoint and identify controls by spelling. HF calls the no-speech token `nocaptions`; the OpenAI tokenizer also supports the spelling `nospeech`. Require all 256 bytes and a contiguous suffix of 1501 timestamp tokens, spaced 20 ms apart. Ignore automatic post-processing templates: construct SOT, language, task and optional no-timestamps controls yourself. Detect language by restricting the SOT row to language ids and renormalizing their probabilities.

Treat a logits source as a function of the complete prefix. Repeated queries return independent rows. Extending the same prefix appends only new self-KV; rewinding or switching a beam branch resets that cache. Cross-KV is constant for an audio window. If a decoder layer fails after another appended state, reset before accepting another request.

Timestamp filtering is a state machine. The first generated token is a bounded timestamp. A start timestamp must be followed by text or EOT; an end timestamp must be followed by another start or EOT. End times advance strictly, while a new start may equal the preceding end. After these masks, force a timestamp when the sum of timestamp probabilities exceeds the best individual text probability. Compare probabilities after subtracting a common logit maximum to avoid overflow.

For greedy search choose the largest filtered logit, with smallest-id tie breaking. For beam search retain the best cumulative-log-probability live candidates and collect completed paths until the patience budget is met. Rank completed paths by score divided by generated length excluding EOT. Sampling divides logits by temperature before drawing from the course PCG32 stream, but accumulates untempered log probabilities. Report average log probability with the EOT position included in the denominator. This distinction matters when deciding whether a hot sample actually improved confidence.

Fallback retries the configured temperatures for excessive compression ratio or low average log probability. High no-speech probability combined with low confidence instead stops retries and skips the window. High-confidence text overrides a no-speech score. Compute log-mel features once for the padded recording, encode once per sought window, and reuse cross-KV for every retry. Split completed segments at adjacent timestamps. Revisit an incomplete tail from the last completed endpoint. Clamp endpoints to the recording, omit empty segments, and always advance seek. Clear prior-text conditioning after temperatures above 0.5 or when conditioning is disabled.

## 3. Worked example

`test_start_controls` checks this hand calculation: with SOT 6, French 4, translate 8 and no-timestamps 9, a French translation prompt is `[6, 4, 8, 9]`. An English transcription with timestamps uses `[6, 3, 7]`. The tokenizer is not mutated by either request.

Let the timestamp vocabulary begin at 100. A generated sequence `[100, 7, 102, 102, 8]` has a completed segment ending at 40 ms and an incomplete next segment. Emit the text represented by token 7 over `[0, 0.04]`, and seek forward four 10 ms mel frames. The next window receives the completed segment's tokens as previous context. It must decode token 8 again from new audio context rather than count the unfinished tail as a completed transcript.

For a scored two-token transcript followed by EOT, cumulative log probability `-1.5` gives average log probability `-0.5`. Candidate ranking divides by two, giving `-0.75`. A temperature of 0.6 changes sampling probabilities, not either denominator. A high-temperature result clears context before the next window so repetition does not propagate.

## 4. The interface and tests

The public contracts specify source ownership, tokenizer geometry, decoding options and segment fields. The normative policy document defines every mask and the NPZ playback layout. Build the tokenizer and source adapter first, then timestamp filtering and single-window search, then fallback and long-form seeking. Recorded sources must fail on missing prefixes instead of inventing logits.

| Test | Kind | Contract |
|---|---|---|
| `test_start_controls` | boundary | Language selection must not silently reuse a tokenizer's old task. |
| `test_initial_timestamp_bound_and_source_ownership` | boundary | Text cannot precede the initial timestamp, even with a dominant logit. |
| `test_timestamp_pairs_and_progress` | boundary | Endpoints advance, but adjacent segments may share a boundary. |
| `test_timestamp_mass_forces_time_even_when_each_token_is_weaker` | property | Comparing argmaxes misses a diffuse but dominant timestamp mass. |
| `test_no_legal_timestamp_is_an_error` | boundary | An all-masked row must not turn into NaNs and an arbitrary token. |
| `test_language_distribution_excludes_dominant_text` | property | Detection renormalizes among language tokens, not the vocabulary. |
| `test_cached_source_extends_repeats_rewinds_and_switches_branches` | property | Beam branches cannot inherit another branch's self-attention state. |
| `test_greedy_and_beam_one_match_with_control_suppression` | property | Beam size one must not explore a different path than greedy. |
| `test_beam_explores_lower_first_token_and_ranks_complete_paths` | golden | Choosing only the locally largest logit is not beam search. |
| `test_sampling_is_reproducible_and_scores_untempered_probability` | property | Temperature changes draws, not the average-logprob fallback statistic. |
| `test_prompt_context_and_generation_limit` | boundary | Previous text conditions the model but never appears as new output. |
| `test_invalid_search_options_fail_before_model_calls` | boundary | Malformed requests must not mutate decoder state. |
| `test_fallback_retries_quality_but_stops_for_probable_silence` | boundary | Silence must not trigger six expensive retries and hallucinated text. |
| `test_longform_seeks_last_complete_boundary_and_reuses_context` | golden | Incomplete trailing speech must be retried from its prior boundary. |
| `test_longform_clears_hot_context_and_bounds_final_segment` | boundary | Fallback hallucinations must not condition subsequent windows. |
| `test_longform_skips_silence_but_preserves_confident_speech` | boundary | High no-speech probability alone cannot veto confident transcription. |
| `test_zero_timestamp_does_not_stall_seek` | boundary | A zero-duration or truncated decode must always advance the window. |
| `test_transcribe_real_model_encodes_each_window_once` | property | The long-form path must actually connect audio features and Whisper. |
| `test_independent_openai_longform_policy_fixture` | golden | Seek, fallback, silence and context decisions match the pinned OpenAI transcription function. |
| `test_independent_openai_timestamp_and_search_fixture` | golden | Internal consistency cannot detect a shared search-policy mistake. |
| `test_whisper_loader_preserves_bytes_and_controls` | property | Template controls must not be appended to ordinary transcript tokens. |
| `test_missing_timestamp_or_byte_is_rejected` | boundary | Timestamp arithmetic and byte preservation cannot silently degrade. |
| `test_english_only_and_suppression_defaults` | boundary | English checkpoints still contain language tokens but must not use them. |
| `test_cli_whisper_logits_match_independent_hf` | golden | The speech milestone CLI preserves full teacher-forced HF logits and rejects malformed prompts. |
| `test_cli_transcribes_real_wav_with_real_model` | property | Tokenizer, WAV frontend, model loading and search must work together. |
| `test_sparse_seek_and_prefix_identity` | boundary | Same-prefix logits from a different audio window are not reusable. |
| `test_bad_sparse_recordings_rejected` | boundary | Broken traces must not be repaired into plausible model outputs. |
| `test_dense_rows_and_duplicate_prefix_validation` | property | Dense and sparse storage share the same unambiguous prefix contract. |
| `test_cli_replays_sparse_logits_without_model_weights` | golden | The parity CLI must invoke the real decoding policy without a model. |

| `test_pinned_real_whisper_tokenizer_matches_hf` | golden | Exact token ids and decoded text match pinned HF tokenizers on real checkpoint merges. |

## 5. Pitfalls

Start with a tiny scripted source. Log prefixes, selected ids, cumulative scores and seek frames, not raw user audio. A mismatch in beam-one output points to control masking or termination. A mismatch only after a branch switch points to stale self-KV. Repetition after a hot retry points to a prompt reset error. A stalled seek often means an endpoint was interpreted as an absolute token id rather than a timestamp offset.

The committed OpenAI oracle covers timestamp masks, greedy/beam search, and long-form window policy. It is independent of the learner implementation. Real-model tests connect WAV decoding, feature extraction, weights, tokenizer and CLI. These tests do not establish production speech accuracy. Real-checkpoint tokenizer parity is checked against pinned HF tokenizers. Real-checkpoint speech accuracy remains integration validation work.

```bash
python -m tinyllm transcribe --model models/whisper-tiny --audio sample.wav --language en --greedy --out segments.json
python -m tinyllm transcribe --logits-trace traces.npz --opts opts.json
```

## 6. Where it's used next

| Direction | Module | Connection |
|---|---|---|
| Back | `L14.3` | Whisper logits and cross/self attention caches. |
| Back | `L14.1` | WAV conversion and full-recording log-mel features. |
| Back | `L1.2` | Byte BPE and checkpoint tokenizer loading. |
| Back | `M05.2` | Complete reversible byte alphabet. |
| Back | `M06.3` | Repeatable PCG32 sampling. |
| Back | `L8.1` | Temperature and categorical sampling. |
| Back | `L4.4` | Beam-search reasoning. |
| Forward | `ethics.08` | Generate consent-bound transcripts and compute WER/CER evidence. |

## Going further

| Resource | Exercise |
|---|---|
| [Decoding policy](../../../course/contracts/spec/asr-decoding.md) | Construct a sparse trace that fails if an incomplete tail is skipped. |
| [Whisper implementation](https://github.com/openai/whisper) | Compare a recorded trace with the pinned independent oracle. |
