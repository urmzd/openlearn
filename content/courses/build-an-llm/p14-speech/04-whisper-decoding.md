# Whisper decoding and long-form transcription

After [the previous section](03-whisper-decoder-and-weights.md), the encoder and decoder can score any prefix. That is not yet a transcript. Turning scores into text requires a series of decisions: which language is spoken, whether a window is silence, where timestamps go, which search to run, and where to continue in a recording longer than 30 seconds. Model logits do not define any of this. Control tokens, timestamp constraints and search do, and together they make up the transcription policy this section builds. When you finish, the speech path is complete: a WAV file goes in, timestamped segments come out.

The design choice that shapes everything is a seam between model and policy. The policy needs only one operation, "next-token logits for this complete prefix", and the solution exposes that same operation for a real Whisper window and for a recording of logits. That separation makes Python/Rust parity possible without repeatedly downloading weights or running an encoder: both languages can replay the same recorded trace through the same policy.

This section uses byte BPE and checkpoint tokenizer loading from [Byte-level BPE](../p01-tokenizers/02-byte-level-bpe.md), the complete reversible byte alphabet from [Injective, surjective, bijective: the GPT-2 byte map](../../mathematics-for-computing/05-discrete-math-1/02-injective-surjective-bijective-gpt2-byte-map.md) and repeatable PCG32 sampling from [Modular arithmetic, hashing, and PCG32 in Python](../../mathematics-for-computing/06-discrete-math-2/03-modular-arithmetic-hashing-and-pcg32.md) in *Mathematics for Computing*, and WAV conversion and full-recording log-mel features from [the audio frontend](01-audio-frontend.md). It helps to have read [Sampling and logit processors](../p08-inference/01-sampling-and-logit-processors.md) for temperature and categorical sampling and [Beam search, generic over a step function](../p04-attention-origins/04-beam-search.md) for the beam-search reasoning. The solution is [`decode.py`](../../../../course/solutions/python/tinyllm/speech/decode.py) (`WhisperLogitsSource`, `sot_sequence`, `language_probs`, `apply_timestamp_rules`, `DecodeOptions`, `decode_window`, `decode_with_fallback`, `transcribe`), [`tokenizer.py`](../../../../course/solutions/python/tinyllm/speech/tokenizer.py) (`WhisperTokenizer`), [`trace.py`](../../../../course/solutions/python/tinyllm/speech/trace.py) (`RecordedLogits`), and the CLI entry point [`cli_speech.py`](../../../../course/solutions/entry/python/tinyllm/cli_speech.py). The contracts are [`decode.pyi`](../../../../course/contracts/py/tinyllm/speech/decode.pyi), [`tokenizer.pyi`](../../../../course/contracts/py/tinyllm/speech/tokenizer.pyi) and [`trace.pyi`](../../../../course/contracts/py/tinyllm/speech/trace.pyi), and the normative [ASR decoding policy](../../../../course/contracts/spec/asr-decoding.md) defines every mask and the NPZ playback layout.

> **Learning objectives**
> - Build Whisper's control prompt explicitly and detect language by renormalizing over language tokens.
> - Implement a logits source that behaves as a pure function of the prefix while reusing KV state across beam branches.
> - Apply the timestamp state machine, including the rule that compares total timestamp mass with the best text token.
> - Run greedy, beam and sampled search, scoring with untempered log probabilities, and decide when to fall back or skip silence.
> - Seek through a long recording by completed segment boundaries and explain why an incomplete tail is revisited.

## The tokenizer and the control prompt

The tokenizer is loaded as byte BPE from the checkpoint, and control tokens are identified by spelling. One spelling difference matters: Hugging Face calls the no-speech token `nocaptions`, and the OpenAI tokenizer also supports `nospeech`. The loader requires all 256 bytes and a contiguous suffix of 1501 timestamp tokens, spaced 20 ms apart, so timestamp arithmetic and byte preservation cannot silently degrade.

Automatic post-processing templates in the tokenizer files are ignored. Instead the policy constructs the SOT, language, task and optional no-timestamps controls explicitly for each request. Language detection restricts the row after SOT to the language ids and renormalizes their probabilities, so a dominant text logit cannot dilute the language distribution.

## A logits source as a function of the prefix

The policy treats a logits source as a function of the complete prefix. Repeated queries return independent rows. Extending the same prefix appends only new self-KV; rewinding, or switching to another beam branch, truncates self-KV to the shared prefix. On an exact rewind, the final token is recomputed to recover its logits. Cross-KV is constant for an audio window. And if a decoder layer fails after another layer has appended state, the source resets before accepting another request.

## The timestamp state machine

Timestamp filtering is a state machine over the generated tokens. The first generated token is a timestamp, bounded by a maximum initial time. A start timestamp must be followed by text or EOT; an end timestamp must be followed by another start or EOT. End times advance strictly, while a new start may equal the preceding end, which is how adjacent segments share a boundary.

After these masks comes one more rule: force a timestamp when the sum of timestamp probabilities exceeds the best individual text probability. When the model is confident a segment has ended but unsure exactly when, its mass spreads over many neighbouring timestamps, each below the best text token but together dominant. Comparing argmaxes would miss that. The probabilities are compared after subtracting a common logit maximum, which avoids overflow without changing the comparison.

## Search and scoring

For greedy search, choose the largest filtered logit, breaking ties toward the smallest id. For beam search, retain the best live candidates by cumulative log probability and collect completed paths until the patience budget is met; completed paths are ranked by score divided by generated length excluding EOT. Sampling divides logits by temperature before drawing from the book's PCG32 stream, but accumulates untempered log probabilities. The reported average log probability includes the EOT position in its denominator.

The temperature distinction matters when deciding whether a hot sample actually improved confidence. A tempered distribution is flatter, so its log probabilities are not comparable across temperatures; untempered scores measure how plausible the model finds the text, whichever temperature drew it.

## Fallback, silence, and long-form seeking

Fallback retries the configured temperatures when the compression ratio is too high (repetition) or the average log probability too low. High no-speech probability combined with low confidence is a different failure: it stops retries and skips the window, because retrying silence only hallucinates. High-confidence text overrides a no-speech score.

For long recordings, compute log-mel features once for the padded recording, encode once per sought window, and reuse cross-KV for every retry. Split completed segments at adjacent timestamps. Revisit an incomplete tail from the last completed endpoint. Clamp endpoints to the recording, omit empty segments, and always advance the seek. Clear prior-text conditioning after temperatures above 0.5, or whenever conditioning is disabled.

## Working it by hand

Take SOT 6, French 4, translate 8 and no-timestamps 9. A French translation prompt without timestamps is `[6, 4, 8, 9]`; an English transcription with timestamps uses `[6, 3, 7]` (English 3, transcribe 7). Neither request mutates the tokenizer.

Let the timestamp vocabulary begin at 100. The generated sequence `[100, 7, 102, 102, 8]` has a completed segment ending at 40 ms and an incomplete next segment. Emit the text represented by token 7 over `[0, 0.04]`, and seek forward four 10 ms mel frames. The next window receives the completed segment's tokens as previous context. It must decode token 8 again from new audio context, rather than count the unfinished tail as part of a completed transcript.

For a scored two-token transcript followed by EOT, cumulative log probability `-1.5` gives average log probability `-0.5`, since the denominator counts EOT. Candidate ranking divides by two, the generated length excluding EOT, giving `-0.75`. A temperature of 0.6 changes the sampling probabilities, not either denominator. A high-temperature result clears context before the next window so repetition does not propagate.

## The shape of the solution

The public contracts specify source ownership, tokenizer geometry, decoding options and segment fields. The solution is layered in the order the data flows: tokenizer and source adapter first, then timestamp filtering and single-window search, then fallback and long-form seeking. Recorded sources must fail on missing prefixes instead of inventing logits.

The reasoning behind each layer:

1. **Separate the model from the policy.** Everything here needs only "next-token logits for this complete prefix". Making that a protocol (`LogitsSource.step(prefix)`) lets the same policy run against a real Whisper window or a recorded trace (`RecordedLogits`), which is what makes Python/Rust parity testable without weights.
2. **A source must behave as if it were stateless.** Beam search asks for prefixes in any order. Causal attention means the KV state for a prefix depends only on that prefix, so the adapter can keep one cache, truncate it to the longest common prefix with the previous request, and append the rest. On an exact rewind it keeps one token fewer, because the logits row is produced by the last token's forward pass.
3. **Controls are built, not inferred.** The prompt is `[sot, language, task]` plus `no_timestamps` when timestamps are off, with optional previous text in front after `sot_prev`. Building it per request avoids hidden state on the tokenizer.
4. **Timestamps are a grammar.** Generated tokens must form `t text... t t text... t` with nondecreasing times, and the mask at each step depends only on the last two generated tokens and the last timestamp. The mass rule needs a numerically safe comparison: subtracting the row maximum leaves the ratio unchanged.
5. **Score with untempered probabilities.** Temperature is a sampling device. Fallback compares `avg_logprob` against a threshold that must mean the same thing at every temperature, so the score accumulates $\log p$ from the untempered row.
6. **Fallback and silence are different failures.** Repetition (a high gzip compression ratio) and low confidence call for a hotter retry. Probable silence (high no-speech probability *and* low confidence) calls for skipping, since retries only hallucinate.
7. **Seeking follows completed segments.** A window's output is trusted up to the last pair of adjacent timestamps. If the output ends mid-segment, the next window starts at that last completed endpoint; otherwise it advances a full window. Every step advances at least one frame, so the loop terminates.

## Building it step by step

### The logits source

One audio window, one bounded cache, any prefix.

```python
def step(self, prefix):
    ids = np.asarray(prefix)
    # ... require a nonempty 1-D integer prefix inside the context and vocabulary
    key = tuple(int(x) for x in ids)
    if key == self.prefix and self.last is not None:
        return self.last.copy()                 # repeated query: an independent copy
    common = 0
    for previous, requested in zip(self.prefix, key):
        if previous != requested:
            break
        common += 1
    # A rewind needs its last token recomputed to recover the logits row.
    # Causal attention makes all earlier shared-prefix KV reusable.
    offset = min(common, len(key) - 1)
    self.cache.truncate(offset)                 # drop the other branch's suffix
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
```

The cross keys and values (`self.xkv`) are computed once by `model.encode` for the window and reused by every step, every beam branch and every fallback temperature.

### The control prompt

Built per request, never stored on the tokenizer.

```python
def sot_sequence(tok, language, task, timestamps):
    # ... task must be transcribe or translate
    languages = dict(zip(tok.all_language_codes, tok.all_language_tokens, strict=True))
    result = [tok.sot]
    if languages:                                # multilingual checkpoint
        language = language or getattr(tok, "language", None) or "en"
        # ... unsupported language raises
        result.extend([languages[language], getattr(tok, task)])
    elif language not in (None, "en") or task != "transcribe":
        raise ValueError("English-only tokenizers cannot translate or select languages")
    if not timestamps:
        result.append(tok.no_timestamps)
    return result
```

With the hand example's ids this gives `[6, 4, 8, 9]` for French translation without timestamps. Language detection (`language_probs`) asks for the row after `[sot]` alone, keeps only the language-token entries, and normalizes those with a max-shifted softmax.

### Timestamp rules

The state machine reads the last two generated tokens.

```python
row[tok.no_timestamps] = -np.inf
last_is_time = bool(generated) and generated[-1] >= begin
follows_time = len(generated) < 2 or generated[-2] >= begin
if last_is_time:
    if follows_time:
        row[begin:] = -np.inf        # two times in a row, or the first time: text or EOT next
    else:
        row[: tok.eot] = -np.inf     # text then an end time: a start time or EOT next
times = [t for t in generated if t >= begin]
if times:
    # After text, the next time is an end: strictly later than its start (+1).
    # Right after an end, the next start may equal that end (+0).
    minimum = times[-1] + (0 if last_is_time and not follows_time else 1)
    row[begin:minimum] = -np.inf
if not generated:
    row[:begin] = -np.inf                              # first token is a timestamp
    row[begin + max_initial_index + 1 :] = -np.inf     # no later than max_initial_timestamp
peak = np.max(row)
if not np.isfinite(peak):
    raise ValueError("timestamp rules leave no legal token")
# Subtract a common peak before exponentiation; the normalizing constant
# cancels from the comparison, including when all text is masked.
time_mass = np.exp(row[begin:] - peak).sum()
text_mass = np.exp(np.max(row[:begin]) - peak)
if time_mass > text_mass:
    row[:begin] = -np.inf
```

Before these rules, `_filtered_logits` masks the control tokens (`sot`, `sot_prev`, `sot_lm`, `transcribe`, `translate`, `no_speech`), the configured suppress list (`-1` expands to the tokenizer's non-speech symbols), and, at the first position only, blank and EOT.

### Search

Greedy and sampling share one loop; beam search keeps the best `beam` live candidates per step and collects finished ones until `round(beam * patience)` are done.

```python
for token in order:                     # top beam+1 of each live candidate
    if np.isfinite(lp[token]):
        ids = tokens + (int(token),)
        candidates[ids] = score + float(lp[token])
active = []
ended = []
for ids, score in sorted(candidates.items(), key=lambda pair: (-pair[1], pair[0])):
    if ids[-1] == tok.eot:
        ended.append((ids, score))
    else:
        active.append((ids, score))
        if len(active) == beam:
            break
finished.extend(ended[: max(0, wanted - len(finished))])
```

Taking `beam + 1` per candidate guarantees `beam` live continuations even if one of them is EOT. Ties sort by token ids, which makes greedy and beam-one agree and keeps Python and Rust identical. For sampling:

```python
lp = _logprobs(row)                                    # untempered: used for the score
if temperature == 0:
    token = int(np.argmax(lp))                         # argmax picks the smallest id on ties
else:
    probabilities = np.exp(_logprobs(row / temperature))
    # PCG32 is shared with the Rust parity implementation.
    cdf = np.cumsum(probabilities)
    cdf[-1] = 1.0                                      # no draw can fall past the end
    token = int(np.searchsorted(cdf, rng.uniform(), side="right"))
score += float(lp[token])
```

The final choice and statistics:

```python
# Rank by generated length excluding EOT, as Whisper's default ranker does.
tokens, score = max(finished, key=lambda pair: pair[1] / max(1, len(pair[0]) - 1))
ids = list(tokens[:-1])
# ...
ratio = len(encoded) / len(zlib.compress(encoded))     # repetition detector
# ... avg_logprob = score / (len(ids) + 1): EOT counted in the denominator
```

In the hand example, `-1.5` over two tokens plus EOT gives an average of `-0.5`, and the ranking key is `-1.5 / 2 = -0.75`.

### Fallback

Retry only for quality failures, and stop immediately on probable silence.

```python
for temperature in opts.temperatures:
    result = decode_window(src, tok, prompt, opts, temperature, rng)
    repetitive = result.compression_ratio > opts.compression_ratio_threshold
    uncertain = result.avg_logprob < opts.logprob_threshold
    silent = result.no_speech_prob > opts.no_speech_threshold and uncertain
    if silent or not (repetitive or uncertain):
        return result
return result
```

### Long form

Seek is measured in 10 ms mel frames. Each iteration builds one source (one encoder call), detects the language if needed, decodes with fallback, then either skips silence or emits completed segments and advances.

```python
while seek * 0.01 < duration:
    source = source_for_seek(seek)                  # encodes exactly one window
    # ... language detection on the first window when none is given
    remaining = min(window_frames * 0.01, duration - seek * 0.01)
    result = decode_with_fallback(source, tok, history, opts, rng)
    silent = (result.no_speech_prob > opts.no_speech_threshold
              and result.avg_logprob <= opts.logprob_threshold)
    if silent:
        seek += window_frames
        continue
    segments, advance = _window_segments(result, tok, seek * 0.01, remaining)
    for segment in segments:
        # ... number the segment, record its seek
        history.extend(segment.tokens)              # condition the next window
    if not opts.condition_on_previous_text or result.temperature > 0.5:
        history.clear()                             # do not propagate hot output
    seek += max(1, round(advance / 0.01))           # always make progress
```

`_window_segments` cuts at each pair of adjacent timestamps. If the output does not end with a closing timestamp, `advance` is the last completed endpoint, so in the hand example the tail token 8 is decoded again from the next window, which starts 4 frames later. Segment times are clamped to the remaining audio, and empty segments are dropped. `transcribe` computes log-mel features once for the whole padded recording, because Whisper's clip at `max - 8` would differ if each window were normalized on its own, and then slices one 3000-frame window per seek.

The full solutions are [`decode.py`](../../../../course/solutions/python/tinyllm/speech/decode.py), [`tokenizer.py`](../../../../course/solutions/python/tinyllm/speech/tokenizer.py) (control ids by spelling, 1501 contiguous timestamps at the vocabulary end, all 256 bytes, no executed post-processing templates) and [`trace.py`](../../../../course/solutions/python/tinyllm/speech/trace.py) (strict NPZ playback keyed by seek and full prefix).

### Why this design

Per window, the cost is one encoder pass, then per generated token $O(o\,d)$ self-attention against the cache and $O(A\,d)$ cross-attention per layer. Beam search multiplies the token steps by the beam size; prefix truncation means that switching between branches recomputes only the tokens after their common prefix, not the history. Fallback can multiply the cost by up to six temperatures, which is why probable silence short-circuits it.

Running the policy inside the model, as Hugging Face's `generate` does, would tie Python/Rust parity to identical floating-point model outputs; the seam lets both languages replay the same recorded logits. Comparing the best timestamp logit to the best text logit is simpler but misses a diffuse timestamp distribution. Normalizing features per window is simpler too, but it changes quiet windows and therefore the reference timestamp decisions.

Inputs over 30 minutes are refused. The policy follows OpenAI's reference transcription function, including its heuristics (gzip ratio, temperature ladder); these are not guarantees of speech accuracy.

## What a correct implementation must handle

The test suite is large because each rule has a failure that looks plausible. The cases, and the failure each one guards against:

| Case | Why it matters |
|---|---|
| Start controls for different languages and tasks | Language selection must not silently reuse a tokenizer's old task. |
| Initial timestamp bound with a dominant text logit | Text cannot precede the initial timestamp, even with a dominant logit. |
| Timestamp pairs and progress | Endpoints advance, but adjacent segments may share a boundary. |
| Diffuse timestamp mass vs one strong text token | Comparing argmaxes misses a diffuse but dominant timestamp mass. |
| No legal timestamp left | An all-masked row must not turn into NaNs and an arbitrary token. |
| Language detection with a dominant text logit | Detection renormalizes among language tokens, not the vocabulary. |
| Shared-prefix reuse in beam search | Beam branches retain causal prefix KV instead of rescoring history. |
| Cached source: extend, repeat, rewind, switch branch | Beam branches cannot inherit another branch's self-attention state. |
| Greedy vs beam size one, with control suppression | Beam size one must not explore a different path than greedy. |
| Beam explores a lower first token | Choosing only the locally largest logit is not beam search. |
| Reproducible sampling and untempered scores | Temperature changes draws, not the average-logprob fallback statistic. |
| Prompt context and generation limit | Previous text conditions the model but never appears as new output. |
| Invalid search options | Malformed requests must not mutate decoder state. |
| Fallback on quality vs probable silence | Silence must not trigger six expensive retries and hallucinated text. |
| Long-form seek to the last complete boundary | Incomplete trailing speech must be retried from its prior boundary. |
| Hot context reset and final-segment bounds | Fallback hallucinations must not condition subsequent windows. |
| Silence skipping vs confident speech | High no-speech probability alone cannot veto confident transcription. |
| Zero timestamp or truncated decode | A zero-duration or truncated decode must always advance the window. |
| Real model, one encoder call per window | The long-form path must actually connect audio features and Whisper. |
| OpenAI long-form policy reference | Seek, fallback, silence and context decisions match the pinned OpenAI transcription function. |
| OpenAI timestamp and search reference | Internal consistency cannot detect a shared search-policy mistake. |
| Tokenizer loader bytes and controls | Template controls must not be appended to ordinary transcript tokens. |
| Missing timestamp or byte token | Timestamp arithmetic and byte preservation cannot silently degrade. |
| English-only checkpoints and suppression defaults | English checkpoints still contain language tokens but must not use them. |
| CLI features vs HF | Both mel-band configurations match independent HF features for all six WAV clips. |
| CLI teacher-forced logits vs HF | The speech CLI preserves full teacher-forced HF logits and rejects malformed prompts. |
| CLI on a real WAV and a real model | Tokenizer, WAV frontend, model loading and search must work together. |
| Sparse traces: seek and prefix identity | Same-prefix logits from a different audio window are not reusable. |
| Bad sparse recordings | Broken traces must not be repaired into plausible model outputs. |
| Dense rows and duplicate prefixes | Dense and sparse storage share the same unambiguous prefix contract. |
| Recorded search traces via the CLI | Greedy, beam-5, compression fallback, hot context reset, partial-tail seeks and prompt conditioning must match independent recorded traces. |
| Replaying sparse logits without weights | The parity CLI must invoke the real decoding policy without a model. |
| Real tokenizers vs pinned HF tokenizers | Exact token ids and decoded text match pinned HF tokenizers on real checkpoint merges. |

## Debugging a transcript

When a transcript differs from the reference, start with a tiny scripted source. Log prefixes, selected ids, cumulative scores and seek frames, not raw user audio. The symptom usually points at the layer. A mismatch in beam-one output points to control masking or termination. A mismatch only after a branch switch points to stale self-KV. Repetition after a hot retry points to a prompt reset error. A stalled seek often means an endpoint was interpreted as an absolute token id rather than a timestamp offset.

> **Note:** The solution's policy matches a pinned copy of OpenAI's transcription function on timestamp masks, greedy and beam search, and long-form window policy, and real-checkpoint tokenizer parity is checked against pinned Hugging Face tokenizers. Real-model runs connect WAV decoding, feature extraction, weights, tokenizer and CLI. None of this establishes production speech accuracy; real-checkpoint speech accuracy remains integration validation work.

The entry point, [`cli_speech.py`](../../../../course/solutions/entry/python/tinyllm/cli_speech.py), exposes both paths: a real model on a WAV file, and replay of recorded logits through the same policy without weights.

```bash
python -m tinyllm transcribe --model models/whisper-tiny --audio sample.wav --language en --greedy --out segments.json
python -m tinyllm transcribe --logits-trace traces.npz --opts opts.json
```

## Where this leads

This section completes the speech path. The next section, [Multimodal eval metrics](05-multimodal-eval-metrics.md), scores the transcripts it produces. In *Responsible AI*, [Consent and biometric evidence in model cards](../../responsible-ai/02-privacy-and-pii/02-consent-and-biometric-evidence.md) uses it to generate consent-bound transcripts and compute WER and CER evidence.

### Further reading

- The [ASR decoding policy](../../../../course/contracts/spec/asr-decoding.md). As an exercise, construct a sparse trace that fails if an incomplete tail is skipped.
- [OpenAI's Whisper implementation](https://github.com/openai/whisper). Compare a recorded trace with its `transcribe` function and explain any difference in seek decisions.
- Compare beam cache reuse with branch recomputation without changing token decisions.

## Key terms

- **control prompt**: the explicit `[sot, language, task]` sequence, plus `no_timestamps` when timestamps are off, that tells Whisper what to do.
- **logits source**: anything that returns next-token logits for a complete prefix; a real model window or a recorded trace.
- **timestamp mass rule**: forcing a timestamp when the summed probability of legal timestamps exceeds the best single text token.
- **temperature fallback**: retrying a window at hotter temperatures when output is repetitive or low-confidence.
- **seek**: the position in the recording, in 10 ms mel frames, where the next window starts.

## Summary

- Model logits do not define transcription policy; control tokens, timestamp constraints and search do.
- The logits source behaves as a pure function of the prefix, truncating its cache to the shared prefix and recomputing the last token on an exact rewind.
- Scores use untempered probabilities, so fallback thresholds mean the same thing at every temperature.
- Repetition and low confidence trigger a hotter retry; probable silence skips the window instead.
- A completed timestamp boundary controls long-form seeking, and an incomplete tail is revisited from the last completed endpoint.

## Review questions

1. Why does an exact rewind truncate the cache to `len(prefix) - 1` rather than to `len(prefix)`?
   <details><summary>Answer</summary>

   The cache stores keys and values, not logits. The logits for a prefix come from the forward pass of its last token, so that token must be run again; all earlier positions are reusable because causal attention makes them independent of what follows.

   </details>

2. Why compare total timestamp probability with the single best text probability?
   <details><summary>Answer</summary>

   When the model is confident a segment has ended but unsure exactly when, its mass spreads over many neighbouring timestamps. Each one may be below the best text token while together they dominate; forcing a timestamp then follows the model's actual belief.

   </details>

3. Why accumulate untempered log probabilities when sampling at temperature 0.8?
   <details><summary>Answer</summary>

   Fallback compares `avg_logprob` with a fixed threshold. A tempered distribution is flatter, so its log probabilities are not comparable across temperatures. Untempered scores measure how plausible the model finds the text, whichever temperature drew it.

   </details>

4. What goes wrong if long-form seeking always advances by a full window?
   <details><summary>Answer</summary>

   A segment cut off by the window edge would be emitted half-finished or lost, and the next window would start after it. Seeking to the last completed boundary lets the next window see the whole utterance.

   </details>

5. Why clear the conditioning history after a temperature above 0.5?
   <details><summary>Answer</summary>

   A hot retry is more likely to contain hallucinated or repetitive text. Feeding it as previous context would push the next window to continue the hallucination, so the history is reset.

   </details>

6. For the sequence `[100, 7, 102, 102, 8]` with timestamps starting at 100, what is emitted and how far does the seek advance?
   <details><summary>Answer</summary>

   Token 7 is emitted over `[0, 0.04]`, the completed segment. Token 8 begins an unfinished segment, so the seek advances only four 10 ms frames, to the last completed endpoint, and token 8 is decoded again from the next window.

   </details>
