# Audio frontend: WAV parsing and Whisper's log-mel

The book has so far taught the system to read text and, in the previous chapter, to see images. This section is where it starts to hear. *Mathematics for Computing* already built the mathematics of a spectrogram and a mel filterbank in [Sampling, Nyquist, windows, and the STFT](../../mathematics-for-computing/13-signals-and-images/05-sampling-windows-and-stft.md) and [Mel scale and filterbanks](../../mathematics-for-computing/13-signals-and-images/06-mel-scale-and-filterbanks.md). What is missing is the path from a real file on disk to the exact picture Whisper was trained on, and that path deserves a section of its own for two reasons.

First, audio arrives as bytes from untrusted clients. The gateway's `/v1/audio/transcriptions` endpoint accepts WAV and nothing else (a deliberate design decision for the system), so the WAV parser is a security boundary as much as a decoder. Second, the Whisper encoder you build in the next section was trained on the output of Hugging Face's `WhisperFeatureExtractor`, and a frontend that differs from it by more than about $10^{-4}$ quietly degrades every transcript. A symmetric window, one frame too many, or a clip taken over the wrong frames are all enough. The engine's later Rust port of this frontend is held to the same Hugging Face reference outputs.

You will need the band-limited `resample_audio` from [Resampling and interpolation](../../mathematics-for-computing/13-signals-and-images/03-resampling-and-interpolation.md), the `hann`, `stft` and `power` functions from the STFT section, and `mel_filterbank` and `log_mel_whisper` from the mel section, all in *Mathematics for Computing*. Nothing else is assumed about the RIFF container: the first section below covers it from the bytes up. By the end you will have log-mel features within $10^{-4}$ of Hugging Face's on real clips, the first step of the speech path. The solution lives in [`wav.py`](../../../../course/solutions/python/tinyllm/io/wav.py) (`read_wav`, covering RIFF chunks, PCM s16 and s24, IEEE f32, `WAVE_FORMAT_EXTENSIBLE`, mono or stereo, and `write_wav`) and [`frontend.py`](../../../../course/solutions/python/tinyllm/speech/frontend.py) (`FeatureConfig`, `to_mono_16k`, `pad_or_trim`, `whisper_log_mel`); the contracts are [`wav.pyi`](../../../../course/contracts/py/tinyllm/io/wav.pyi) and [`frontend.pyi`](../../../../course/contracts/py/tinyllm/speech/frontend.pyi).

> **Learning objectives**
> - Walk a RIFF/WAVE file chunk by chunk, skipping unknown chunks and their pad bytes, and decode s16, s24 and f32 samples.
> - Explain why every size field must be checked against the bytes present, and which checks make a lying size harmless.
> - State Whisper's feature recipe exactly: mono, 16 kHz, 30 s, periodic Hann(400) at hop 160, drop the last of 3001 frames, Slaney mel, $\log_{10}$, an 8-unit clip, $(L+4)/4$.
> - Explain why the clip must come after dropping the last frame, and predict the features of silence by hand.

## The notation

The symbols used through the rest of the section:

| Symbol | Meaning | Type / shape |
|---|---|---|
| $x$ | mono signal at 16 kHz | `float64[n]` |
| $N = 480000$ | samples in one 30 s window | int |
| $n_\text{fft} = 400$, $h = 160$ | frame length and hop (25 ms, 10 ms) | int |
| $w$ | periodic Hann window | `float64[400]` |
| $P$ | power spectrogram | `float64[201, T]` |
| $M$ | Slaney mel filterbank | `float64[n_mels, 201]` |
| $L$ | $\log_{10}$ mel energies | `float64[n_mels, T]` |

## RIFF and WAVE

A WAV file is `RIFF <u32 size> WAVE` followed by chunks of the form `<4-byte id> <u32 size> <payload>`, with one zero pad byte after a payload of odd size so that the next chunk starts on an even offset. All integers are little-endian. That is the only structure there is, and you need to understand just two chunks; you must, however, skip every other one correctly, pad byte included.

The `fmt ` chunk gives the encoding: `format` (1 for PCM integers, 3 for IEEE float, `0xFFFE` for extensible), `channels`, `sample_rate`, `byte_rate`, `block_align` (bytes per frame, $\text{channels} \cdot \text{bits}/8$), and `bits`. The extensible form appends 24 bytes whose last 16 are a GUID, and the GUID's first two bytes are the real format code. The `data` chunk holds frames; a stereo frame is left then right, which is what "interleaved" means. Samples are scaled to $[-1, 1)$: s16 by $2^{-15}$, s24 by $2^{-23}$ after sign-extending three bytes into an int, and f32 as stored.

Everything else (`LIST` metadata, `fact`, `cue `, `junk`) is skipped by its size. And here is the part that makes this a security boundary: the parser trusts no size. The RIFF size and every chunk size must fit in the bytes present, `block_align` and `byte_rate` must agree with the format, and the data must be a whole number of frames. A size field that lies is exactly how a truncated upload, or a malicious one, would make a naive parser read past the buffer or allocate gigabytes. Every check below exists so that such a file fails with a `ValueError` instead of returning garbage.

## Mono, 16 kHz, 30 seconds

Once the samples are decoded, they must be brought to the one shape Whisper accepts. Stereo becomes mono by the mean of the channels, not by taking the left channel: a source panned right would vanish. Any other sample rate goes through the band-limited `resample_audio`, which removes energy above the new Nyquist frequency before decimating. And because Whisper's encoder has a fixed input length, each window is exactly $N = 30 \cdot 16000 = 480000$ samples. Shorter audio is padded with zeros at the end; longer audio is cut, and [Whisper decoding and long-form transcription](04-whisper-decoding.md) walks a long file window by window.

## The log-mel recipe

With $x$ of length $n$, the centered STFT from *Mathematics for Computing* pads 200 reflected samples on each side and produces $1 + \lfloor n/h \rfloor$ frames. Whisper keeps the first $\lfloor n/h \rfloor$, which is 3000 for a full window. Then

$$P = |\text{STFT}_w(x)|^2_{[:,\,:-1]}, \qquad L = \log_{10}\max(M P,\ 10^{-10}), \qquad L \leftarrow \max(L,\ \max L - 8), \qquad \text{features} = \frac{L + 4}{4},$$

with $w$ the periodic Hann window ($0.5 - 0.5\cos(2\pi k/400)$, divided by 400 rather than 399) and $M$ librosa's Slaney filterbank. Results are float32.

One detail in that recipe deserves emphasis because it is easy to get wrong. The maximum in the clip is taken over the kept frames of the whole window, so the loudest moment anywhere sets the floor everywhere, and the order "drop the last frame, then clip" matters. If the clip saw the dropped frame, a burst at the very end of the window could set the floor using a frame the encoder never sees.

## Working it by hand

Here is a two-frame stereo s16 file, byte by byte, 52 bytes in all:

| Offset | Bytes | Meaning |
|---|---|---|
| 0 | `RIFF` `2C 00 00 00` `WAVE` | RIFF size 44 = 52 - 8 |
| 12 | `fmt ` `10 00 00 00` | a 16-byte fmt chunk |
| 20 | `01 00` `02 00` `80 3E 00 00` `00 FA 00 00` `04 00` `10 00` | PCM, 2 channels, 16000 Hz, 64000 B/s, 4 B per frame, 16 bits |
| 36 | `data` `08 00 00 00` | 8 bytes of frames |
| 44 | `00 40` `FF FF` `00 80` `FF 7F` | frame 0: L = 16384, R = -1; frame 1: L = -32768, R = 32767 |

Divide by $2^{15}$ and de-interleave into `[channels, frames]`, and you get left $= (0.5, -1)$ and right $= (-1/32768,\ 32767/32768)$.

Now the frontend on silence. A 30 s window of zeros has $P = 0$, so $M P = 0$, every $L$ is $\log_{10} 10^{-10} = -10$, the maximum is $-10$, the clip at $-18$ changes nothing, and every feature is $(-10 + 4)/4 = -1.5$. The window has $1 + 480000/160 = 3001$ STFT frames, of which 3000 are kept. Both of these hand results become tests.

## Designing the API

The interface is small. A `Wav` holds float32 samples shaped `[channels, frames]` along with the rate, channel count and format name; `FeatureConfig` carries Hugging Face's defaults and can be read from a checkpoint's preprocessor config.

```python
@dataclass
class Wav: samples: NDArray; sr: int; channels: int; fmt: str   # float32 [channels, frames]
def read_wav(data: bytes) -> Wav: ...
def write_wav(samples: NDArray, sr: int) -> bytes: ...           # s16le mono

@dataclass
class FeatureConfig:                                               # HF defaults; from_hf(path)
    sampling_rate = 16000; n_fft = 400; hop_length = 160; feature_size = 80; chunk_length = 30
def to_mono_16k(w: Wav, cfg: FeatureConfig) -> NDArray: ...
def pad_or_trim(x: NDArray, n_samples: int = 480000) -> NDArray: ...
def whisper_log_mel(x: NDArray, cfg: FeatureConfig) -> NDArray: ... # float32 [n_mels, len(x) // hop]
```

A correct implementation is tested against the hand example, against real files, and against hostile ones. The 52-byte file above checks fields, scaling and de-interleaving. Files that `scipy.io.wavfile` reads (s24, f32 with a `fact` chunk, odd `LIST` and `junk` chunks, extensible with trailing bytes) must decode to exactly what scipy returns, covering every encoding the system accepts. Broken and hostile files must all fail: truncated data, a 4 GiB RIFF size, a RIFF size 2 bytes too long, a missing or late `fmt `, unsupported encodings, 3 channels, bad rates, bad `block_align` or `byte_rate`, and partial frames. `write_wav` must round-trip with a 44-byte header, round half to even, and clip, so fixtures and tests can write audio. The log-mel features must land within $10^{-4}$ of Hugging Face's `WhisperFeatureExtractor` on six LibriSpeech clips at both 80 and 128 mels, since that is the encoder's input. Silence must give 3000 frames of $-1.5$, a loud burst in the dropped last frame must not affect the clip, stereo and other rates must go through the channel mean and the resampler (with a copy at 16 kHz), and `pad_or_trim` must pad zeros at the end, trim, and never alias its input. `FeatureConfig.from_hf` must read Hugging Face config keys, ignore unknown ones and refuse another processor, which is what lets 128-mel checkpoints load. Finally, bad frontend arguments (2-D input, fewer than 201 samples) must fail early, because they are caller bugs.

## Building it step by step

The reasoning comes first, then the code.

1. **The file is a list of sized chunks.** The only structure is `id, size, payload, pad`. So the parser is a loop with one cursor `pos`, advanced by `8 + size + (size & 1)`; it never needs to understand chunks other than `fmt ` and `data`.
2. **Every size is a claim to verify.** The RIFF size bounds the walk (`end = 8 + riff_size`, which must not exceed the buffer), and each chunk must fit inside `end`. With those two checks no slice can run past the buffer and no size can cause a large allocation, since every array is a view of bytes already present.
3. **`fmt ` must be self-consistent.** `block_align` and `byte_rate` are redundant with `channels`, `bits` and `sr`; a mismatch means a broken writer, and the data length must be a whole number of frames. For the extensible form the real code is the first two bytes of the GUID at offset 24 of the payload (16 bytes of base fields, 2 of `cbSize`, 2 of valid bits, 4 of channel mask).
4. **Scaling is a power of two.** Signed $b$-bit integers lie in $[-2^{b-1}, 2^{b-1})$, so dividing by $2^{b-1}$ maps them to $[-1, 1)$ exactly; in the hand example, `00 80` is $-32768 \to -1$.
5. **Interleaved means frame-major.** The flat sample array reshapes to `[frames, channels]`, and a transpose gives `[channels, frames]`.
6. **The frontend is a fixed pipeline whose pieces already exist.** The resampler, STFT and Whisper log-mel come from *Mathematics for Computing*. What this section adds is the exact order: mean of channels, resample, pad or trim to $N$, periodic Hann, drop the last frame, mel, then the clip over the kept frames only.

### The chunk walk

Two bounds make the loop safe, and the pad byte keeps it aligned.

```python
(riff_size,) = struct.unpack_from("<I", b, 4)
end = 8 + riff_size
if end > len(b):                       # the RIFF size itself may lie
    raise ValueError(...)
pos = 12
fmt = None
pcm = None
while pos + 8 <= end:                  # room for another chunk header
    cid = b[pos : pos + 4]
    (size,) = struct.unpack_from("<I", b, pos + 4)
    body = pos + 8
    if body + size > end:              # a chunk may not run past the RIFF end
        raise ValueError(...)
    if cid == b"fmt ":
        fmt = _parse_fmt(b[body : body + size])
    elif cid == b"data":
        if fmt is None:                # cannot decode frames without a format
            raise ValueError("read_wav: data chunk before the fmt chunk")
        pcm = b[body : body + size]
        break                          # trailing chunks are irrelevant
    # Chunks are word-aligned: an odd size is followed by one pad byte.
    pos = body + size + (size & 1)
```

### The format chunk

Every combination the solution cannot decode is refused by name, and the redundant fields are cross-checked.

```python
tag, channels, sr, byte_rate, block, bits = struct.unpack_from("<HHIIHH", f, 0)
if tag == _EXTENSIBLE:
    if len(f) < 40:
        raise ValueError("read_wav: WAVE_FORMAT_EXTENSIBLE needs a 40-byte fmt chunk")
    (tag,) = struct.unpack_from("<H", f, 24)  # first two bytes of the GUID
if tag == _PCM and bits == 16:
    kind = "s16"
elif tag == _PCM and bits == 24:
    kind = "s24"
elif tag == _FLOAT and bits == 32:
    kind = "f32"
else:
    raise ValueError(...)
if channels not in (1, 2):                    # 3+ channels have no agreed downmix here
    raise ValueError(...)
if not 1 <= sr <= _MAX_SR:                    # 384000 Hz is the inclusive maximum
    raise ValueError(...)
if block != channels * bits // 8 or byte_rate != sr * block:
    raise ValueError(...)
```

### Decoding samples

s24 has no NumPy dtype, so the three little-endian bytes are assembled into an int32 and sign-extended by hand.

```python
if len(pcm) % block:                          # partial frame: truncated or corrupt
    raise ValueError(...)
frames = len(pcm) // block
if kind == "s16":
    x = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / np.float32(32768.0)
elif kind == "s24":
    u = np.frombuffer(pcm, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
    v = u[:, 0] | (u[:, 1] << 8) | (u[:, 2] << 16)
    v = np.where(v >= 1 << 23, v - (1 << 24), v)  # sign-extend 24 bits
    x = v.astype(np.float32) / np.float32(8388608.0)
else:
    x = np.frombuffer(pcm, dtype="<f4").astype(np.float32)
# Interleaved L R L R ... -> [channels, frames].
samples = np.ascontiguousarray(x.reshape(frames, channels).T)
```

### Writing

`write_wav` is the inverse for mono s16, used by fixtures and tests. It multiplies by 32768 (the same scale the reader divides by), rounds half to even with `np.round`, and clips to the int16 range so that $+1.0$ saturates at 32767 instead of wrapping to $-32768$.

```python
q = np.clip(np.round(x * 32768.0), -32768, 32767).astype("<i2")
pcm = q.tobytes()
header = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE"   # 4 + 24 + 8 + data
fmt = b"fmt " + struct.pack("<IHHIIHH", 16, _PCM, 1, int(sr), int(sr) * 2, 2, 16)
return header + fmt + b"data" + struct.pack("<I", len(pcm)) + pcm
```

The full parser and writer are in [`wav.py`](../../../../course/solutions/python/tinyllm/io/wav.py).

### The frontend

Each function is a few lines, because the signal processing lives in *Mathematics for Computing*. What matters here is the order and the edge behaviour.

```python
def to_mono_16k(w: Wav, cfg: FeatureConfig) -> NDArray:
    x = np.asarray(w.samples, dtype=np.float64)
    mono = x.mean(axis=0) if x.ndim == 2 else x.copy()   # mean, not the left channel
    if w.sr == cfg.sampling_rate:
        return mono
    return resample_audio(mono, w.sr, cfg.sampling_rate)  # band-limited

def pad_or_trim(x: NDArray, n_samples: int = 480000) -> NDArray:
    a = np.asarray(x, dtype=np.float64)
    # ... reject non-1-D input and n_samples < 1
    if a.size >= n_samples:
        return a[:n_samples].copy()                       # a copy: never alias the caller
    return np.concatenate([a, np.zeros(n_samples - a.size)])  # zeros at the end only

def whisper_log_mel(x: NDArray, cfg: FeatureConfig) -> NDArray:
    a = np.asarray(x, dtype=np.float64)
    # ... reject non-1-D input and fewer than n_fft // 2 + 1 samples
    spec = stft(a, cfg.n_fft, cfg.hop_length, hann(cfg.n_fft, periodic=True))
    # 1 + n // hop frames; Whisper keeps n // hop of them (3001 -> 3000).
    p = power(spec)[:, :-1]
    fb = mel_filterbank(cfg.sampling_rate, cfg.n_fft, cfg.feature_size)
    # The clip at max - 8 sees only the kept frames.
    return log_mel_whisper(p, fb).astype(np.float32)
```

The minimum length $n_\text{fft}/2 + 1 = 201$ comes from reflect padding: reflecting 200 samples needs at least 201 to reflect from. `log_mel_whisper` computes `log10(max(fb @ p, 1e-10))`, clips at `max - 8` and maps through `(L + 4) / 4`; because `p` has already lost its last column, the maximum is taken over the kept frames. `FeatureConfig.from_hf` reads only the fields it knows, ignores the rest, and refuses any `feature_extractor_type` other than `WhisperFeatureExtractor`. The full frontend is in [`frontend.py`](../../../../course/solutions/python/tinyllm/speech/frontend.py).

### Why this design

Parsing is $O(\text{bytes})$ with no copy beyond the float conversion. The log-mel costs one STFT, $O(T \cdot n_\text{fft} \log n_\text{fft})$ for $T = 3000$ frames, plus an $n_\text{mels} \times 201 \times T$ matrix product.

You could reach for `scipy.io.wavfile` or `soundfile`, which parse more formats. But the gateway accepts uploads from untrusted clients, and a parser of under a hundred lines with every bound visible is easier to audit, and easier to port to Rust with identical rules. Calling Hugging Face's `WhisperFeatureExtractor` directly would add a large dependency at inference time and hide the recipe the Rust port must match.

The tradeoffs are deliberate. Only s16, s24 and f32, mono or stereo, are accepted; 8-bit, 32-bit integer, RF64 and multichannel files fail with a clear error. Trailing chunks after `data` are ignored.

## What can go wrong

Most bugs in this code produce output that looks like audio, which is why each one is worth naming. The parser's classic failures come from the byte layout. If 24-bit samples are not sign-extended, negative samples become large positive ones; the solution applies `np.where(v >= 1 << 23, v - (1 << 24), v)` after assembling the bytes. If the pad byte after an odd chunk is ignored, the next chunk id is read one byte early and the parser reports "no data chunk"; the cursor advances by `size + (size & 1)`. If the extensible subformat is read from the wrong offset, valid 24-bit files are rejected; the GUID's first two bytes are at payload offset 24. If interleaved frames are reshaped as planar, left and right are mixed sample by sample; `reshape(frames, channels).T` is the fix. Dividing s16 by 32767 puts full scale slightly above 1 and moves every sample; the solution divides by $2^{15}$ (and s24 by $2^{23}$).

> **Warning:** A chunk size that is not checked against the file lets a truncated upload return half a clip. Both `body + size > end` and `end > len(b)` raise, and a RIFF size even a few bytes too long is refused because `8 + riff_size` must not exceed `len(b)`.

The boundary checks have their own traps. The top supported rate, 384000 Hz, must be accepted, so the bound is inclusive: `1 <= sr <= _MAX_SR`. A 3-channel file must be refused (`channels not in (1, 2)` raises) rather than averaged as if it were stereo. In `write_wav`, clipping to 32768 instead of 32767 makes $+1.0$ wrap to $-1.0$ in int16, and scaling by 32767 makes round trips drift by one step; the solution scales by 32768 and clips to `[-32768, 32767]` before the cast.

The frontend's failures are subtler, because they shift numbers by small amounts:

| Mistake | Symptom | Fix in the solution |
|---|---|---|
| keeping the 3001st frame | features are 3001 wide; the encoder's positions overflow | `power(spec)[:, :-1]` |
| dropping the first frame instead of the last | every feature shifted by 10 ms | slices off the last column |
| a symmetric Hann window | features off by about $10^{-3}$ | `hann(cfg.n_fft, periodic=True)` |
| clipping before dropping the last frame | quiet frames clipped to the wrong floor | drop first, then `log_mel_whisper` |
| the left channel instead of the mean | a right-panned speaker disappears | `x.mean(axis=0)` |
| padding at the front | speech starts late; timestamps shift | `pad_or_trim` appends zeros |
| accepting `n_samples = 0` | empty windows | requires `n_samples >= 1` |
| refusing Whisper's own config | every HF checkpoint rejected | `from_hf` accepts a missing type or `WhisperFeatureExtractor` |
| returning float64 features | twice the memory; the engine's f32 parity drifts | `.astype(np.float32)` |

## Where this leads

The features you just built are the encoder's entire input. In [Whisper encoder: conv stem, sinusoids, pre-LN blocks](02-whisper-encoder.md), the conv stem reads `[n_mels, 3000]` straight from `whisper_log_mel`. [Whisper decoder and checkpoint weights](03-whisper-decoder-and-weights.md) wraps the WAV parser, the resampler and the log-mel frontend into `Whisper.encode_wav`, and [Whisper decoding and long-form transcription](04-whisper-decoding.md) computes features for a whole recording and slices it window by window. In *Responsible AI*, [Consent and biometric evidence in model cards](../../responsible-ai/02-privacy-and-pii/02-consent-and-biometric-evidence.md) decodes consent-bound WAV evidence with this parser before computing generated WER. Later, the audio corpus pipeline in *Data Engineering* parses and normalizes clips with `read_wav` and `to_mono_16k`, and the engine re-implements this frontend in Rust, checked against the same reference outputs within $10^{-4}$.

Looking back, three pieces of *Mathematics for Computing* did the heavy lifting: `resample_audio` brings 22.05 or 48 kHz audio to 16 kHz, `stft(x, 400, 160, hann(400))` and `power` give the spectrogram, and `mel_filterbank(16000, 400, n_mels)` with `log_mel_whisper` give the features.

### Further reading

- Microsoft, "Multimedia Programming Interface and Data Specifications 1.0" (1991), the RIFF and WAVE chapters, and the `WAVEFORMATEXTENSIBLE` documentation.
- Radford et al., "Robust speech recognition via large-scale weak supervision" (2022), section 2.2.
- `soundfile` (libsndfile) and `scipy.io.wavfile` are the production equivalents of `read_wav`: dozens of containers and codecs, and RF64 for files over 4 GiB. See `scipy/io/wavfile.py`.
- Hugging Face's `WhisperFeatureExtractor` adds batching, attention masks and a torch path on GPU; see `transformers/models/whisper/feature_extraction_whisper.py`.
- `openai-whisper`'s `log_mel_spectrogram` is the same recipe in torch, with the filters loaded from `mel_filters.npz`; see `whisper/audio.py`.
- `ffmpeg -ac 1 -ar 16000` is the production `to_mono_16k`: every codec and container, and Whisper's CLI shells out to it (`libswresample`).

## Key terms

- **RIFF chunk**: a `<4-byte id> <u32 size> <payload>` record, followed by a pad byte when the size is odd.
- **block align**: bytes per frame, $\text{channels} \cdot \text{bits}/8$; redundant with the other `fmt ` fields and therefore checkable.
- **`WAVE_FORMAT_EXTENSIBLE`**: the `0xFFFE` format whose real format code is the first two bytes of a GUID at payload offset 24.
- **periodic Hann window**: $0.5 - 0.5\cos(2\pi k/n)$ with denominator $n$ rather than $n-1$, the window Whisper's STFT uses.
- **log-mel clip**: the floor $\max L - 8$ applied to $\log_{10}$ mel energies, computed over the kept frames of the whole window.

## Summary

- A WAV file is a RIFF list of chunks; only `fmt ` and `data` matter, but every other chunk must be skipped by its size, including the pad byte after an odd size.
- Every size field is checked against the bytes actually present, so a truncated or hostile file fails with a `ValueError` instead of returning garbage.
- Whisper's input is a fixed recipe: mono by channel mean, 16 kHz, exactly 30 s, a power STFT with a periodic Hann(400) at hop 160, the last of 3001 frames dropped, Slaney mel, $\log_{10}$, an 8-unit clip, $(L+4)/4$.
- The clip is measured from the loudest value of the kept frames only, so "drop, then clip" is part of the recipe.
- Silence maps to 3000 frames of $-1.5$, a hand result worth testing.

## Review questions

1. Why is the next chunk at `body + size + (size & 1)`?
   <details><summary>Answer</summary>

   RIFF chunks are word-aligned. After an odd-sized payload the writer inserts one pad byte that is not counted in `size`. Without `(size & 1)` the cursor lands one byte early and reads garbage as the next chunk id.

   </details>

2. Why are both `end > len(b)` and `body + size > end` needed?
   <details><summary>Answer</summary>

   The first rejects a RIFF header that claims more bytes than the file holds (a truncated upload). The second rejects a chunk that claims more bytes than its container. Together they guarantee every slice is inside the buffer, so a lying size can neither read out of bounds nor silently return half a clip.

   </details>

3. Why divide s16 by 32768 rather than 32767?
   <details><summary>Answer</summary>

   The int16 range is $[-32768, 32767]$. Dividing by $2^{15}$ maps it to $[-1, 1)$ exactly and matches scipy and Hugging Face; dividing by 32767 makes $-32768$ map below $-1$ and shifts every sample by a tiny factor, which breaks bit-exact round trips with `write_wav`.

   </details>

4. What changes if the clip is computed before dropping the last frame?
   <details><summary>Answer</summary>

   The floor is `max - 8` over the frames used. If the dropped 3001st frame holds the loudest value (a burst at the very end), the floor is set by a frame the encoder never sees, so quiet kept frames are clipped higher than Hugging Face clips them.

   </details>

5. Why average channels instead of taking the left one?
   <details><summary>Answer</summary>

   Mono is meant to keep all the content. A speaker panned to the right channel would be silent in the left; the mean keeps both, at the cost of halving the amplitude of a one-sided source, which the log scale and normalization absorb.

   </details>

6. What features does a 30 s window of zeros produce, and how many frames?
   <details><summary>Answer</summary>

   $P = 0$, so every $L$ is $\log_{10} 10^{-10} = -10$; the clip at $-18$ changes nothing, and every feature is $(-10+4)/4 = -1.5$. The STFT gives $1 + 480000/160 = 3001$ frames, of which 3000 are kept.

   </details>
