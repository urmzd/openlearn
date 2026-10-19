# Audio frontend: WAV parsing and Whisper's log-mel

## Overview

| | |
|---|---|
| **Module** | `L14.1` · build · Python · Pass 12 · 2 to 3 h |
| **Solution** | [`wav.py`](../../../../course/solutions/python/tinyllm/io/wav.py): `read_wav` (RIFF chunks, PCM s16 and s24, IEEE f32, `WAVE_FORMAT_EXTENSIBLE`, mono or stereo) and `write_wav`; [`frontend.py`](../../../../course/solutions/python/tinyllm/speech/frontend.py): `FeatureConfig`, `to_mono_16k`, `pad_or_trim`, `whisper_log_mel` |
| **Contract** | [`course/contracts/py/tinyllm/io/wav.pyi`](../../../../course/contracts/py/tinyllm/io/wav.pyi), [`course/contracts/py/tinyllm/speech/frontend.pyi`](../../../../course/contracts/py/tinyllm/speech/frontend.pyi) |
| **Needs** | `M12.3` `resample_audio` · `M12.5` `hann`, `stft`, `power` · `M12.6` `mel_filterbank`, `log_mel_whisper`. Catalog prereq `lang.12` (RIFF from the bytes up) is not registered yet; section 2.1 covers what you need |
| **Used by** | `ethics.08`; `L14.3`; `L14.2` (the encoder eats these features) · later `data.11` (audio corpus) and `L15.1` (the Rust port in the engine); `L14.4`. This is the first step of the speech path: log-mel features within $10^{-4}$ of HF on real clips |
| **Optional depth** | Microsoft, "Multimedia Programming Interface and Data Specifications 1.0" (1991), the RIFF and WAVE chapters; Microsoft, `WAVEFORMATEXTENSIBLE` documentation; Radford et al., "Robust speech recognition via large-scale weak supervision" (2022), section 2.2 |

## Key Takeaways

- A WAV file is a RIFF list of chunks; you need only `fmt ` and `data`, but you must skip every other chunk correctly, including its pad byte when the size is odd.
- Every size field is checked against the bytes actually present, so a truncated or hostile file fails with a `ValueError` instead of returning garbage.
- Whisper's input is a fixed recipe: mono, 16 kHz, exactly 30 s, a power STFT with a periodic Hann(400) at hop 160, the last of 3001 frames dropped, Slaney mel, $\log_{10}$, an 8-unit clip, $(L+4)/4$.
- The clip is measured from the loudest value of the kept frames only, so the order "drop, then clip" matters.

## How to read this chapter

Sections 1 to 3 build the WAV format and Whisper's feature recipe from first principles and work both by hand. Section 4 fixes the interface and the cases a correct implementation must handle. Section 5 walks through the solution, the parser first and then the frontend, and explains why each line is there. Try to predict each check before reading it.

---

## 1. Why now

Pass 12 teaches the system to hear. `M12.5` and `M12.6` built the math of a spectrogram and a mel filterbank; this module turns real files into the exact picture Whisper was trained on. Two things make it a module of its own. First, audio arrives as bytes from untrusted clients (the gateway's `/v1/audio/transcriptions`, D47: WAV only), so the parser is a security boundary as much as a decoder. Second, the encoder (`L14.2`) was trained on Hugging Face's `WhisperFeatureExtractor` output, and a frontend that differs by more than about $10^{-4}$ (a symmetric window, one frame too many, the clip taken over the wrong frames) quietly degrades every transcript. The engine's Rust port (`L15.1`) is later proven against the same Hugging Face reference outputs.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $x$ | mono signal at 16 kHz | `float64[n]` |
| $N = 480000$ | samples in one 30 s window | int |
| $n_\text{fft} = 400$, $h = 160$ | frame length and hop (25 ms, 10 ms) | int |
| $w$ | periodic Hann window | `float64[400]` |
| $P$ | power spectrogram | `float64[201, T]` |
| $M$ | Slaney mel filterbank | `float64[n_mels, 201]` |
| $L$ | $\log_{10}$ mel energies | `float64[n_mels, T]` |

### 2.1 RIFF and WAVE

A WAV file is `RIFF <u32 size> WAVE` followed by chunks `<4-byte id> <u32 size> <payload>`, with one zero pad byte after a payload of odd size so the next chunk starts on an even offset. All integers are little-endian. The `fmt ` chunk gives the encoding: `format` (1 PCM integers, 3 IEEE float, `0xFFFE` extensible), `channels`, `sample_rate`, `byte_rate`, `block_align` (bytes per frame, $\text{channels} \cdot \text{bits}/8$), and `bits`. The extensible form appends 24 bytes whose last 16 are a GUID; the GUID's first two bytes are the real format code. The `data` chunk holds frames; a stereo frame is left then right (interleaved). Samples are scaled to $[-1, 1)$: s16 by $2^{-15}$, s24 by $2^{-23}$ after sign-extending three bytes into an int, f32 as stored.

Everything else (`LIST` metadata, `fact`, `cue `, `junk`) is skipped by its size. The parser trusts no size: the RIFF size and every chunk size must fit in the bytes present, `block_align` and `byte_rate` must agree with the format, and the data must be a whole number of frames. A size field that lies is how a truncated upload, or a malicious one, would make a naive parser read past the buffer or allocate gigabytes.

### 2.2 Mono, 16 kHz, 30 s

Stereo becomes mono by the mean of the channels (not the left channel: a source panned right would vanish). Any other rate goes through `M12.3`'s band-limited `resample_audio`, which removes energy above the new Nyquist frequency before decimating. Whisper's encoder has a fixed input length, so each window is exactly $N = 30 \cdot 16000 = 480000$ samples: shorter audio is padded with zeros at the end, longer audio is cut (`L14.4` walks a long file window by window).

### 2.3 The log-mel recipe

With $x$ of length $n$, `M12.5`'s centered STFT pads 200 reflected samples on each side and produces $1 + \lfloor n/h \rfloor$ frames; Whisper keeps the first $\lfloor n/h \rfloor$ (3000 for a full window). Then
$$P = |\text{STFT}_w(x)|^2_{[:,\,:-1]}, \qquad L = \log_{10}\max(M P,\ 10^{-10}), \qquad L \leftarrow \max(L,\ \max L - 8), \qquad \text{features} = \frac{L + 4}{4},$$
with $w$ the periodic Hann window ($0.5 - 0.5\cos(2\pi k/400)$, not divided by 399) and $M$ librosa's Slaney filterbank. The maximum in the clip is over the kept frames of the whole window, so the loudest moment anywhere sets the floor everywhere. Results are float32.

## 3. Worked example by hand

A two-frame stereo s16 file, byte by byte (52 bytes):

| Offset | Bytes | Meaning |
|---|---|---|
| 0 | `RIFF` `2C 00 00 00` `WAVE` | RIFF size 44 = 52 - 8 |
| 12 | `fmt ` `10 00 00 00` | a 16-byte fmt chunk |
| 20 | `01 00` `02 00` `80 3E 00 00` `00 FA 00 00` `04 00` `10 00` | PCM, 2 channels, 16000 Hz, 64000 B/s, 4 B per frame, 16 bits |
| 36 | `data` `08 00 00 00` | 8 bytes of frames |
| 44 | `00 40` `FF FF` `00 80` `FF 7F` | frame 0: L = 16384, R = -1; frame 1: L = -32768, R = 32767 |

Divided by $2^{15}$ and de-interleaved into `[channels, frames]`: left $= (0.5, -1)$, right $= (-1/32768,\ 32767/32768)$.

Silence: a 30 s window of zeros has $P = 0$, so $M P = 0$, every $L$ is $\log_{10} 10^{-10} = -10$, the maximum is $-10$, the clip at $-18$ changes nothing, and every feature is $(-10 + 4)/4 = -1.5$. The window has $1 + 480000/160 = 3001$ STFT frames, 3000 kept.

## 4. The interface

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

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| Section 3's 52-byte file | unit, smoke | fields, scale, de-interleaving | the byte layout |
| Files that `scipy.io.wavfile` reads | golden | s24, f32 with fact, odd LIST and junk chunks, extensible with trailing bytes: equal to `scipy.io.wavfile` | every encoding D47 accepts |
| Broken and hostile files | fault | truncated data, a 4 GiB RIFF size, a RIFF size 2 bytes too long, missing or late fmt, unsupported encodings, 3 channels, bad rates, bad block_align or byte_rate, partial frames | the parser as a security boundary |
| `write_wav` round trip | property | 44-byte header, round half to even, clipping, exact read back | fixtures and tests that write audio |
| Log-mel against HF `WhisperFeatureExtractor` | golden | six LibriSpeech clips, 80 and 128 mels, within $10^{-4}$ of HF | the encoder's input (`L14.2`) |
| Silence | property | section 3: 3000 frames of $-1.5$ | the floor and the frame count |
| A loud burst in the dropped last frame | unit | the clip after dropping the last frame | the order of the recipe |
| Stereo and other rates | property | channel mean, `M12.3`'s resampler, a copy at 16 kHz | stereo and 22.05 kHz uploads |
| `pad_or_trim` | unit | zeros at the end, trimming, no aliasing of the input | fixed-length windows |
| `FeatureConfig.from_hf` | unit | HF config keys, unknown keys ignored, another processor refused | 128-mel checkpoints |
| Bad frontend arguments | boundary | 2-D input, fewer than 201 samples | caller bugs fail early |

## 5. The solution, step by step

### From first principles

1. **The file is a list of sized chunks.** Section 2.1 says the only structure is `id, size, payload, pad`. So the parser is a loop with one cursor `pos`, advanced by `8 + size + (size & 1)`; it never needs to understand chunks other than `fmt ` and `data`.
2. **Every size is a claim to verify.** The RIFF size bounds the walk (`end = 8 + riff_size`, which must not exceed the buffer), and each chunk must fit inside `end`. With those two checks, no slice can run past the buffer and no size can cause a large allocation, since every array is a view of bytes already present.
3. **`fmt ` must be self-consistent.** `block_align` and `byte_rate` are redundant with `channels`, `bits` and `sr`; a mismatch means a broken writer, and the data length must be a whole number of frames. For the extensible form the real code is the first two bytes of the GUID at offset 24 of the payload (16 bytes of base fields, 2 of `cbSize`, 2 of valid bits, 4 of channel mask).
4. **Scaling is a power of two.** Signed $b$-bit integers lie in $[-2^{b-1}, 2^{b-1})$, so dividing by $2^{b-1}$ maps them to $[-1, 1)$ exactly; in section 3, `00 80` is $-32768 \to -1$.
5. **Interleaved means frame-major.** The flat sample array reshapes to `[frames, channels]`, and a transpose gives `[channels, frames]`.
6. **The frontend is a fixed pipeline whose pieces already exist.** M12.3, M12.5 and M12.6 provide the resampler, STFT and Whisper log-mel. What this module adds is the exact order: mean of channels, resample, pad or trim to $N$, periodic Hann, drop the last frame, mel, then the clip over the kept frames only.

### The code, annotated

**The chunk walk.** Two bounds make the loop safe; the pad byte keeps it aligned.

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

**The format chunk.** Every combination the solution cannot decode is refused by name, and the redundant fields are cross-checked.

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

**Decoding samples.** s24 has no NumPy dtype, so the three little-endian bytes are assembled into an int32 and sign-extended by hand.

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

**Writing.** `write_wav` is the inverse for mono s16, used by fixtures and tests. It multiplies by 32768 (the same scale the reader divides by), rounds half to even with `np.round`, and clips to the int16 range so $+1.0$ saturates at 32767 instead of wrapping to $-32768$:

```python
q = np.clip(np.round(x * 32768.0), -32768, 32767).astype("<i2")
pcm = q.tobytes()
header = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE"   # 4 + 24 + 8 + data
fmt = b"fmt " + struct.pack("<IHHIIHH", 16, _PCM, 1, int(sr), int(sr) * 2, 2, 16)
return header + fmt + b"data" + struct.pack("<I", len(pcm)) + pcm
```

Full solution: [`wav.py`](../../../../course/solutions/python/tinyllm/io/wav.py).

**The frontend.** Each function is a few lines because the signal processing lives in M12.x; what matters is the order and the edge behaviour.

```python
def to_mono_16k(w: Wav, cfg: FeatureConfig) -> NDArray:
    x = np.asarray(w.samples, dtype=np.float64)
    mono = x.mean(axis=0) if x.ndim == 2 else x.copy()   # mean, not the left channel
    if w.sr == cfg.sampling_rate:
        return mono
    return resample_audio(mono, w.sr, cfg.sampling_rate)  # band-limited (M12.3)

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

The minimum length $n_\text{fft}/2 + 1 = 201$ comes from reflect padding: reflecting 200 samples needs at least 201 to reflect from. `log_mel_whisper` (M12.6) computes `log10(max(fb @ p, 1e-10))`, clips at `max - 8` and maps through `(L + 4) / 4`; because `p` has already lost its last column, the maximum is taken over the kept frames. `FeatureConfig.from_hf` reads only the fields it knows, ignores the rest, and refuses any `feature_extractor_type` other than `WhisperFeatureExtractor`.

Full solution: [`frontend.py`](../../../../course/solutions/python/tinyllm/speech/frontend.py).

### Why this solution

- **Cost.** Parsing is $O(\text{bytes})$ with no copy beyond the float conversion. The log-mel costs one STFT, $O(T \cdot n_\text{fft} \log n_\text{fft})$ for $T = 3000$ frames, plus an $n_\text{mels} \times 201 \times T$ matrix product.
- **Alternatives.** `scipy.io.wavfile` or `soundfile` would parse more formats, but the gateway accepts uploads from untrusted clients, and a parser of under a hundred lines with every bound visible is easier to audit and to port to Rust (`L15.1`) with identical rules. Calling HF's `WhisperFeatureExtractor` directly would add a large dependency at inference time and hide the recipe the Rust port must match.
- **Tradeoffs accepted.** Only s16, s24 and f32, mono or stereo, are accepted (D47); 8-bit, 32-bit integer, RF64 and multichannel files fail with a clear error. Trailing chunks after `data` are ignored.

### Explain it back

<details><summary>Why is the next chunk at <code>body + size + (size &amp; 1)</code>?</summary>

RIFF chunks are word-aligned. After an odd-sized payload the writer inserts one pad byte that is not counted in `size`. Without `(size & 1)` the cursor lands one byte early and reads garbage as the next chunk id.

</details>

<details><summary>Why are both <code>end &gt; len(b)</code> and <code>body + size &gt; end</code> needed?</summary>

The first rejects a RIFF header that claims more bytes than the file holds (a truncated upload). The second rejects a chunk that claims more bytes than its container. Together they guarantee every slice is inside the buffer, so a lying size can neither read out of bounds nor silently return half a clip.

</details>

<details><summary>Why divide s16 by 32768 rather than 32767?</summary>

The int16 range is $[-32768, 32767]$. Dividing by $2^{15}$ maps it to $[-1, 1)$ exactly and matches scipy and HF; dividing by 32767 makes $-32768$ map below $-1$ and shifts every sample by a tiny factor, which breaks bit-exact round trips with `write_wav`.

</details>

<details><summary>What changes if the clip is computed before dropping the last frame?</summary>

The floor is `max - 8` over the frames used. If the dropped 3001st frame holds the loudest value (a burst at the very end), the floor is set by a frame the encoder never sees, so quiet kept frames are clipped higher than HF clips them.

</details>

<details><summary>Why average channels instead of taking the left one?</summary>

Mono is meant to keep all the content. A speaker panned to the right channel would be silent in the left; the mean keeps both, at the cost of halving the amplitude of a one-sided source, which the log scale and normalization absorb.

</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. 24-bit samples not sign-extended | negative samples become large positive ones | `np.where(v >= 1 << 23, v - (1 << 24), v)` after assembling the bytes |
| 2. the pad byte after an odd chunk ignored | the next chunk id is read one byte early: "no data chunk" | the cursor advances by `size + (size & 1)` |
| 3. the extensible subformat read from the wrong offset | valid 24-bit files rejected | reads the GUID's first two bytes at payload offset 24 |
| 4. interleaved frames reshaped as planar | left and right mixed sample by sample | `reshape(frames, channels).T` |
| 5. dividing s16 by 32767 | full scale slightly above 1, every sample off | divides by $2^{15}$ and s24 by $2^{23}$ |
| 6. keeping the 3001st frame | features are 3001 wide; the encoder's positions overflow | `power(spec)[:, :-1]` |
| 7. dropping the first frame instead of the last | every feature shifted by 10 ms | slices off the last column, not the first |
| 8. a symmetric Hann window | features off by about $10^{-3}$ | `hann(cfg.n_fft, periodic=True)` |
| 9. clipping before dropping the last frame | quiet frames clipped to the wrong floor | drops the frame first, then calls `log_mel_whisper` |
| 10. the left channel instead of the mean | a right-panned speaker disappears | `x.mean(axis=0)` |
| 11. padding at the front | speech starts late in the window; timestamps shift | `pad_or_trim` appends zeros after the signal |
| 12. not checking a chunk size against the file | a truncated upload returns half a clip | `body + size > end` and `end > len(b)` both raise |
| 13. `write_wav` scaling by 32767 | round trips drift by one step | scales by 32768, the reader's inverse |
| rejecting 384000 Hz as out of range | the top supported rate rejected | the bound is inclusive: `1 <= sr <= _MAX_SR` |
| accepting 3 channels | a surround file averaged as if stereo | `channels not in (1, 2)` raises |
| a RIFF size a few bytes too long accepted | trailing chunk read past the buffer | `8 + riff_size` must not exceed `len(b)` |
| clipping to 32768 | +1.0 wraps to -1.0 in int16 | clips to `[-32768, 32767]` before the cast |
| accepting `n_samples = 0` | empty windows | `pad_or_trim` requires `n_samples >= 1` |
| refusing Whisper's own config | every HF checkpoint rejected | `from_hf` accepts a missing type or `WhisperFeatureExtractor` |
| returning float64 features | twice the memory per window; the engine's f32 parity drifts | casts the result with `.astype(np.float32)` |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M12.3` | `resample_audio` brings 22.05 or 48 kHz audio to 16 kHz |
| Back | `M12.5` | `stft(x, 400, 160, hann(400))` and `power` give the spectrogram |
| Back | `M12.6` | `mel_filterbank(16000, 400, n_mels)` and `log_mel_whisper` give the features |
| Forward | `L14.2` | the encoder's conv stem reads `[n_mels, 3000]` from here |
| Forward | `data.11` | the audio corpus pipeline parses and normalizes clips with `read_wav` and `to_mono_16k` (with B14's data group) |
| Forward | `L15.1` | the engine re-implements this frontend in Rust and is checked against the same reference outputs within $10^{-4}$ |
| Forward | `L14.3` | `Whisper.encode_wav` consumes the WAV parser, resampler and log-mel frontend. |
| Forward | `L14.4` | Whisper decoding consumes this implementation. |
| Forward | `ethics.08` | Decode consent-bound WAV evidence before computing generated WER. |

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `read_wav` | `soundfile` (libsndfile), `scipy.io.wavfile` | dozens of containers and codecs, RF64 for files over 4 GiB | `scipy/io/wavfile.py` |
| `whisper_log_mel` | HF `WhisperFeatureExtractor` | batching, attention masks, a torch path on GPU | `transformers/models/whisper/feature_extraction_whisper.py` |
| `whisper_log_mel` | `openai-whisper` `log_mel_spectrogram` | the same recipe in torch with the filters loaded from `mel_filters.npz` | `whisper/audio.py` |
| `to_mono_16k` | `ffmpeg -ac 1 -ar 16000` | every codec and container; Whisper's CLI shells out to it | ffmpeg `libswresample` |


