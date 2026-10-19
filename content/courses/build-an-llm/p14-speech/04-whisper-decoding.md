# Whisper decoding and long-form transcription

## Overview

| | |
|---|---|
| **Module** | `L14.4` · build · Python · Pass 12 · 3 to 4 h |
| **Solution** | [`decode.py`](../../../../course/solutions/python/tinyllm/speech/decode.py): `WhisperLogitsSource`, `sot_sequence`, `language_probs`, `apply_timestamp_rules`, `DecodeOptions`, `decode_window`, `decode_with_fallback`, `transcribe`; [`tokenizer.py`](../../../../course/solutions/python/tinyllm/speech/tokenizer.py): `WhisperTokenizer`; [`trace.py`](../../../../course/solutions/python/tinyllm/speech/trace.py): `RecordedLogits`; CLI entry point [`cli_speech.py`](../../../../course/solutions/entry/python/tinyllm/cli_speech.py) |
| **Contract** | [`decode.pyi`](../../../../course/contracts/py/tinyllm/speech/decode.pyi), [`tokenizer.pyi`](../../../../course/contracts/py/tinyllm/speech/tokenizer.pyi), [`trace.pyi`](../../../../course/contracts/py/tinyllm/speech/trace.pyi); [ASR decoding policy](../../../../course/contracts/spec/asr-decoding.md) |
| **Needs** | `L14.3`, `L14.1`, `L1.2`, `M05.2`, `M06.3`. Reading: `L8.1`, `L4.4` |
| **Used by** | `ethics.08` generated transcript evidence. This module completes the speech path: a WAV file in, timestamped segments out |
| **Optional depth** | Compare beam cache reuse with branch recomputation without changing token decisions. |

## Key Takeaways

- Model logits do not define transcription policy: control tokens, timestamp constraints and search do.
- Keep probability scores before temperature scaling for fallback decisions.
- A completed timestamp boundary controls long-form seeking; an incomplete tail must be revisited.

## How to read this chapter

Sections 1 to 3 state the decoding policy and work a prompt, a timestamp sequence and a score by hand. Section 4 lists what a correct implementation must handle. Section 5 walks through the solution in the order the data flows: the logits source, the control prompt, timestamp masking, search, fallback and long-form seeking. Try to predict each rule's effect before reading the code.

## 1. Why now

The encoder and decoder can score any prefix. Turning those scores into a transcript requires decisions about language, silence, timestamps, search and where to continue in a recording. The solution exposes the same policy to a real Whisper window and a recording of logits. This separation makes Python/Rust parity possible without repeatedly downloading weights or running an encoder.

## 2. Principles

Load byte BPE from the checkpoint and identify controls by spelling. HF calls the no-speech token `nocaptions`; the OpenAI tokenizer also supports the spelling `nospeech`. Require all 256 bytes and a contiguous suffix of 1501 timestamp tokens, spaced 20 ms apart. Ignore automatic post-processing templates: construct SOT, language, task and optional no-timestamps controls explicitly. Detect language by restricting the SOT row to language ids and renormalizing their probabilities.

Treat a logits source as a function of the complete prefix. Repeated queries return independent rows. Extending the same prefix appends only new self-KV; rewinding or switching a beam branch truncates self-KV to the shared prefix. Recompute the final token on an exact rewind to recover its logits. Cross-KV is constant for an audio window. If a decoder layer fails after another appended state, reset before accepting another request.

Timestamp filtering is a state machine. The first generated token is a bounded timestamp. A start timestamp must be followed by text or EOT; an end timestamp must be followed by another start or EOT. End times advance strictly, while a new start may equal the preceding end. After these masks, force a timestamp when the sum of timestamp probabilities exceeds the best individual text probability. Compare probabilities after subtracting a common logit maximum to avoid overflow.

For greedy search choose the largest filtered logit, with smallest-id tie breaking. For beam search retain the best cumulative-log-probability live candidates and collect completed paths until the patience budget is met. Rank completed paths by score divided by generated length excluding EOT. Sampling divides logits by temperature before drawing from the course PCG32 stream, but accumulates untempered log probabilities. Report average log probability with the EOT position included in the denominator. This distinction matters when deciding whether a hot sample actually improved confidence.

Fallback retries the configured temperatures for excessive compression ratio or low average log probability. High no-speech probability combined with low confidence instead stops retries and skips the window. High-confidence text overrides a no-speech score. Compute log-mel features once for the padded recording, encode once per sought window, and reuse cross-KV for every retry. Split completed segments at adjacent timestamps. Revisit an incomplete tail from the last completed endpoint. Clamp endpoints to the recording, omit empty segments, and always advance seek. Clear prior-text conditioning after temperatures above 0.5 or when conditioning is disabled.

## 3. Worked example

With SOT 6, French 4, translate 8 and no-timestamps 9, a French translation prompt is `[6, 4, 8, 9]`. An English transcription with timestamps uses `[6, 3, 7]`. The tokenizer is not mutated by either request.

Let the timestamp vocabulary begin at 100. A generated sequence `[100, 7, 102, 102, 8]` has a completed segment ending at 40 ms and an incomplete next segment. Emit the text represented by token 7 over `[0, 0.04]`, and seek forward four 10 ms mel frames. The next window receives the completed segment's tokens as previous context. It must decode token 8 again from new audio context rather than count the unfinished tail as a completed transcript.

For a scored two-token transcript followed by EOT, cumulative log probability `-1.5` gives average log probability `-0.5`. Candidate ranking divides by two, giving `-0.75`. A temperature of 0.6 changes sampling probabilities, not either denominator. A high-temperature result clears context before the next window so repetition does not propagate.

## 4. The interface

The public contracts specify source ownership, tokenizer geometry, decoding options and segment fields. The normative policy document defines every mask and the NPZ playback layout. The solution is layered: tokenizer and source adapter first, then timestamp filtering and single-window search, then fallback and long-form seeking. Recorded sources must fail on missing prefixes instead of inventing logits.

### What a correct solution must handle

| Case | Kind of check | Why it matters |
|---|---|---|
| Start controls for different languages and tasks | boundary | Language selection must not silently reuse a tokenizer's old task. |
| Initial timestamp bound with a dominant text logit | boundary | Text cannot precede the initial timestamp, even with a dominant logit. |
| Timestamp pairs and progress | boundary | Endpoints advance, but adjacent segments may share a boundary. |
| Diffuse timestamp mass vs one strong text token | property | Comparing argmaxes misses a diffuse but dominant timestamp mass. |
| No legal timestamp left | boundary | An all-masked row must not turn into NaNs and an arbitrary token. |
| Language detection with a dominant text logit | property | Detection renormalizes among language tokens, not the vocabulary. |
| Shared-prefix reuse in beam search | property | Beam branches retain causal prefix KV instead of rescoring history. |
| Cached source: extend, repeat, rewind, switch branch | property | Beam branches cannot inherit another branch's self-attention state. |
| Greedy vs beam size one, with control suppression | property | Beam size one must not explore a different path than greedy. |
| Beam explores a lower first token | golden | Choosing only the locally largest logit is not beam search. |
| Reproducible sampling and untempered scores | property | Temperature changes draws, not the average-logprob fallback statistic. |
| Prompt context and generation limit | boundary | Previous text conditions the model but never appears as new output. |
| Invalid search options | boundary | Malformed requests must not mutate decoder state. |
| Fallback on quality vs probable silence | boundary | Silence must not trigger six expensive retries and hallucinated text. |
| Long-form seek to the last complete boundary | golden | Incomplete trailing speech must be retried from its prior boundary. |
| Hot context reset and final-segment bounds | boundary | Fallback hallucinations must not condition subsequent windows. |
| Silence skipping vs confident speech | boundary | High no-speech probability alone cannot veto confident transcription. |
| Zero timestamp or truncated decode | boundary | A zero-duration or truncated decode must always advance the window. |
| Real model, one encoder call per window | property | The long-form path must actually connect audio features and Whisper. |
| OpenAI long-form policy reference | golden | Seek, fallback, silence and context decisions match the pinned OpenAI transcription function. |
| OpenAI timestamp and search reference | golden | Internal consistency cannot detect a shared search-policy mistake. |
| Tokenizer loader bytes and controls | property | Template controls must not be appended to ordinary transcript tokens. |
| Missing timestamp or byte token | boundary | Timestamp arithmetic and byte preservation cannot silently degrade. |
| English-only checkpoints and suppression defaults | boundary | English checkpoints still contain language tokens but must not use them. |
| CLI features vs HF | golden | Both mel-band configurations match independent HF features for all six WAV clips. |
| CLI teacher-forced logits vs HF | golden | The speech CLI preserves full teacher-forced HF logits and rejects malformed prompts. |
| CLI on a real WAV and a real model | property | Tokenizer, WAV frontend, model loading and search must work together. |
| Sparse traces: seek and prefix identity | boundary | Same-prefix logits from a different audio window are not reusable. |
| Bad sparse recordings | boundary | Broken traces must not be repaired into plausible model outputs. |
| Dense rows and duplicate prefixes | property | Dense and sparse storage share the same unambiguous prefix contract. |
| Recorded search traces via the CLI | golden | Greedy, beam-5, compression fallback, hot context reset, partial-tail seeks and prompt conditioning must match independent recorded traces. |
| Replaying sparse logits without weights | golden | The parity CLI must invoke the real decoding policy without a model. |
| Real tokenizers vs pinned HF tokenizers | golden | Exact token ids and decoded text match pinned HF tokenizers on real checkpoint merges. |

## 5. The solution, step by step

### From first principles

1. **Separate the model from the policy.** Everything in this module needs only one operation: "next-token logits for this complete prefix". Making that a protocol (`LogitsSource.step(prefix)`) lets the same policy run against a real Whisper window or a recorded trace (`RecordedLogits`), which is what makes Python/Rust parity testable without weights.
2. **A source must behave as if it were stateless.** Beam search asks for prefixes in any order. Causal attention means the KV state for a prefix depends only on that prefix, so the adapter can keep one cache, truncate it to the longest common prefix with the previous request, and append the rest. On an exact rewind it keeps one token fewer, because the logits row is produced by the last token's forward pass.
3. **Controls are built, not inferred.** The prompt is `[sot, language, task]` plus `no_timestamps` when timestamps are off, with optional previous text in front after `sot_prev`. Building it per request avoids hidden state on the tokenizer.
4. **Timestamps are a grammar.** Generated tokens must form `t text... t t text... t` with nondecreasing times; the mask at each step depends only on the last two generated tokens and the last timestamp. Section 2 also requires that the total probability of all legal timestamps beats the single best text token before text is allowed, which needs a numerically safe comparison: subtracting the row maximum leaves the ratio unchanged.
5. **Score with untempered probabilities.** Temperature is a sampling device. Fallback compares `avg_logprob` against a threshold that must mean the same thing at every temperature, so the score accumulates $\log p$ from the untempered row.
6. **Fallback and silence are different failures.** Repetition (high gzip compression ratio) and low confidence call for a hotter retry. Probable silence (high no-speech probability *and* low confidence) calls for skipping, since retries only hallucinate.
7. **Seeking follows completed segments.** A window's output is trusted up to the last pair of adjacent timestamps. If the output ends mid-segment, the next window starts at that last completed endpoint; otherwise it advances a full window. Every step advances at least one frame so the loop terminates.

### The code, annotated

**The logits source.** One audio window, one bounded cache, any prefix.

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

**The control prompt.** Built per request, never stored on the tokenizer.

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

With section 3's ids this gives `[6, 4, 8, 9]` for French translation without timestamps. Language detection (`language_probs`) asks for the row after `[sot]` alone, keeps only the language-token entries, and normalizes those with a max-shifted softmax, so a dominant text logit cannot dilute the language distribution.

**Timestamp rules.** The state machine reads the last two generated tokens.

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

Before these rules, `_filtered_logits` masks control tokens (`sot`, `sot_prev`, `sot_lm`, `transcribe`, `translate`, `no_speech`), the configured suppress list (`-1` expands to the tokenizer's non-speech symbols), and, at the first position only, blank and EOT.

**Search.** Greedy and sampling share one loop; beam search keeps the best `beam` live candidates per step and collects finished ones until `round(beam * patience)` are done.

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

In section 3's example, `-1.5` over two tokens plus EOT gives an average of `-0.5`, and the ranking key is `-1.5 / 2 = -0.75`.

**Fallback.** Retry only for quality failures; stop immediately on probable silence.

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

**Long form.** Seek is in 10 ms mel frames. Each iteration builds one source (one encoder call), detects the language if needed, decodes with fallback, then either skips silence or emits completed segments and advances.

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

`_window_segments` cuts at each pair of adjacent timestamps. If the output does not end with a closing timestamp, `advance` is the last completed endpoint, so in section 3 the tail token 8 is decoded again from the next window, which starts 4 frames later. Segment times are clamped to the remaining audio, and empty segments are dropped. `transcribe` computes log-mel features once for the whole padded recording, because Whisper's clip at `max - 8` would differ if each window were normalized on its own, and slices one `3000`-frame window per seek.

Full solutions: [`decode.py`](../../../../course/solutions/python/tinyllm/speech/decode.py), [`tokenizer.py`](../../../../course/solutions/python/tinyllm/speech/tokenizer.py) (control ids by spelling, 1501 contiguous timestamps at the vocabulary end, all 256 bytes, no executed post-processing templates), [`trace.py`](../../../../course/solutions/python/tinyllm/speech/trace.py) (strict NPZ playback keyed by seek and full prefix).

### Why this solution

- **Cost.** Per window: one encoder pass, then per generated token $O(o\,d)$ self-attention against the cache and $O(A\,d)$ cross-attention per layer. Beam search multiplies the token steps by the beam size; prefix truncation means switching between branches recomputes only the tokens after their common prefix, not the history. Fallback can multiply the cost by up to six temperatures, which is why probable silence short-circuits it.
- **Alternatives.** Running the policy inside the model (as HF's `generate` does) would tie Python/Rust parity to identical floating-point model outputs; the seam lets both languages replay the same recorded logits. Comparing the best timestamp logit to the best text logit is simpler but misses a diffuse timestamp distribution. Normalizing features per window is simpler but changes quiet windows and therefore the reference timestamp decisions.
- **Tradeoffs accepted.** Inputs over 30 minutes are refused. The policy follows OpenAI's reference transcription function, including its heuristics (gzip ratio, temperature ladder); these are not guarantees of speech accuracy.

### Explain it back

<details><summary>Why does an exact rewind truncate the cache to <code>len(prefix) - 1</code> rather than to <code>len(prefix)</code>?</summary>

The cache stores keys and values, not logits. The logits for a prefix come from the forward pass of its last token, so that token must be run again; all earlier positions are reusable because causal attention makes them independent of what follows.

</details>

<details><summary>Why compare total timestamp probability with the single best text probability?</summary>

When the model is confident a segment has ended but unsure exactly when, its mass spreads over many neighbouring timestamps. Each one may be below the best text token while together they dominate; forcing a timestamp then follows the model's actual belief.

</details>

<details><summary>Why accumulate untempered log probabilities when sampling at temperature 0.8?</summary>

Fallback compares `avg_logprob` with a fixed threshold. A tempered distribution is flatter, so its log probabilities are not comparable across temperatures. Untempered scores measure how plausible the model finds the text, whichever temperature drew it.

</details>

<details><summary>What goes wrong if long-form seeking always advances by a full window?</summary>

A segment cut off by the window edge would be emitted half-finished or lost, and the next window would start after it. Seeking to the last completed boundary lets the next window see the whole utterance.

</details>

<details><summary>Why clear the conditioning history after a temperature above 0.5?</summary>

A hot retry is more likely to contain hallucinated or repetitive text. Feeding it as previous context would push the next window to continue the hallucination, so the history is reset.

</details>

## 6. Pitfalls

When a transcript differs from the reference, start with a tiny scripted source. Log prefixes, selected ids, cumulative scores and seek frames, not raw user audio. A mismatch in beam-one output points to control masking or termination. A mismatch only after a branch switch points to stale self-KV. Repetition after a hot retry points to a prompt reset error. A stalled seek often means an endpoint was interpreted as an absolute token id rather than a timestamp offset.

The solution's policy matches a pinned copy of OpenAI's transcription function on timestamp masks, greedy/beam search and long-form window policy. Real-model runs connect WAV decoding, feature extraction, weights, tokenizer and CLI. These do not establish production speech accuracy. Real-checkpoint tokenizer parity is checked against pinned HF tokenizers. Real-checkpoint speech accuracy remains integration validation work.

The solution's entry point ([`cli_speech.py`](../../../../course/solutions/entry/python/tinyllm/cli_speech.py)) exposes both paths: a real model on a WAV file, and replay of recorded logits through the same policy without weights.

```bash
python -m tinyllm transcribe --model models/whisper-tiny --audio sample.wav --language en --greedy --out segments.json
python -m tinyllm transcribe --logits-trace traces.npz --opts opts.json
```

## 7. Where it's used next

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
| [Decoding policy](../../../../course/contracts/spec/asr-decoding.md) | Construct a sparse trace that fails if an incomplete tail is skipped. |
| [Whisper implementation](https://github.com/openai/whisper) | Compare a recorded trace with OpenAI's `transcribe` function and explain any difference in seek decisions. |
