<!-- ol:module L14.1 -->
# Audio frontend: WAV parsing and Whisper's log-mel

## Overview

| | |
|---|---|
| **Module** | `L14.1` · build · Python · Pass 12 · 4 to 6 h |
| **You build** | `python/tinyllm/io/wav.py`: `read_wav` (RIFF chunks, PCM s16 and s24, IEEE f32, `WAVE_FORMAT_EXTENSIBLE`, mono or stereo) and `write_wav`; `python/tinyllm/speech/frontend.py`: `FeatureConfig`, `to_mono_16k`, `pad_or_trim`, `whisper_log_mel` |
| **Contract** | [`course/contracts/py/tinyllm/io/wav.pyi`](../../../course/contracts/py/tinyllm/io/wav.pyi), [`course/contracts/py/tinyllm/speech/frontend.pyi`](../../../course/contracts/py/tinyllm/speech/frontend.pyi) |
| **Tests** | `course/tests/L14.1/test_frontend.py` (what they check: section 4), fixtures from `course/oracle/L14.1/frontend_hf.py` (scipy 1.17, transformers 5.19) in `course/fixtures/L14.1/` |
| **Needs** | `M12.3` `resample_audio` · `M12.5` `hann`, `stft`, `power` · `M12.6` `mel_filterbank`, `log_mel_whisper` (or `--ref-deps`). Catalog prereq `lang.12` (RIFF from the bytes up) is not registered yet; section 2.1 covers what you need |
| **Used by** | `L14.2` (the encoder eats these features) · later `data.11` (audio corpus) and `L15.1` (the Rust port in the engine) |
| **Milestone** | `MS-L14` step 1 (log-mel within $10^{-4}$ of HF on the fixture clips) |
| **Optional depth** | Microsoft, "Multimedia Programming Interface and Data Specifications 1.0" (1991), the RIFF and WAVE chapters; Microsoft, `WAVEFORMATEXTENSIBLE` documentation; Radford et al., "Robust speech recognition via large-scale weak supervision" (2022), section 2.2 |

## Key Takeaways

- A WAV file is a RIFF list of chunks; you need only `fmt ` and `data`, but you must skip every other chunk correctly, including its pad byte when the size is odd (`test_reads_fixture_wavs_like_scipy`).
- Every size field is checked against the bytes actually present, so a truncated or hostile file fails with a `ValueError` instead of returning garbage (`test_rejects_broken_files`).
- Whisper's input is a fixed recipe: mono, 16 kHz, exactly 30 s, a power STFT with a periodic Hann(400) at hop 160, the last of 3001 frames dropped, Slaney mel, $\log_{10}$, an 8-unit clip, $(L+4)/4$ (`test_log_mel_matches_hf_feature_extractor`).
- The clip is measured from the loudest value of the kept frames only, so the order "drop, then clip" matters (`test_clip_uses_the_kept_frames_only`).

## How to work this chapter

```bash
ol start L14.1              # stubs wav.py and frontend.py into your repo
ol tests L14.1              # read the test catalog first: rung R0, you write no tests here
ol check L14.1              # exit code is the verdict
ol check L14.1 --ref-deps   # only if your M12.3, M12.5, or M12.6 is not passing yet
ol diff  L14.1              # after passing: your code against the reference
```

---

## 1. Why now

Pass 12 teaches the system to hear. `M12.5` and `M12.6` built the math of a spectrogram and a mel filterbank; this module turns real files into the exact picture Whisper was trained on. Two things make it a module of its own. First, audio arrives as bytes from untrusted clients (the gateway's `/v1/audio/transcriptions`, D47: WAV only), so the parser is a security boundary as much as a decoder. Second, the encoder (`L14.2`) was trained on Hugging Face's `WhisperFeatureExtractor` output, and a frontend that differs by more than about $10^{-4}$ (a symmetric window, one frame too many, the clip taken over the wrong frames) quietly degrades every transcript. The engine's Rust port (`L15.1`) is later proven against the fixtures this module's oracle writes.

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

Divided by $2^{15}$ and de-interleaved into `[channels, frames]`: left $= (0.5, -1)$, right $= (-1/32768,\ 32767/32768)$. This is `test_hand_example_wav_header`.

Silence: a 30 s window of zeros has $P = 0$, so $M P = 0$, every $L$ is $\log_{10} 10^{-10} = -10$, the maximum is $-10$, the clip at $-18$ changes nothing, and every feature is $(-10 + 4)/4 = -1.5$. The window has $1 + 480000/160 = 3001$ STFT frames, 3000 kept (`test_silence_gives_the_floor_everywhere`).

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

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_wav_header` | unit, smoke | section 3's 52-byte file: fields, scale, de-interleaving | the byte layout |
| `test_reads_fixture_wavs_like_scipy` | golden | s24, f32 with fact, odd LIST and junk chunks, extensible with trailing bytes: equal to `scipy.io.wavfile` | every encoding D47 accepts |
| `test_rejects_broken_files` | fault | truncated data, a 4 GiB RIFF size, a RIFF size 2 bytes too long, missing or late fmt, unsupported encodings, 3 channels, bad rates, bad block_align or byte_rate, partial frames | the parser as a security boundary |
| `test_write_wav_round_trip` | property | 44-byte header, round half to even, clipping, exact read back | fixtures and tests that write audio |
| `test_log_mel_matches_hf_feature_extractor` | golden | six LibriSpeech clips, 80 and 128 mels, within $10^{-4}$ of HF | the encoder's input (`L14.2`), MS-L14 step 1 |
| `test_silence_gives_the_floor_everywhere` | property | section 3: 3000 frames of $-1.5$ | the floor and the frame count |
| `test_clip_uses_the_kept_frames_only` | unit | the clip after dropping the last frame, on a burst at the end | the order of the recipe |
| `test_to_mono_averages_channels_then_resamples` | property | channel mean, `M12.3`'s resampler, a copy at 16 kHz | stereo and 22.05 kHz uploads |
| `test_pad_or_trim` | unit | zeros at the end, trimming, no aliasing of the input | fixed-length windows |
| `test_feature_config_from_hf` | unit | HF config keys, unknown keys ignored, another processor refused | 128-mel checkpoints |
| `test_rejects_bad_frontend_arguments` | boundary | 2-D input, fewer than 201 samples | caller bugs fail early |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. 24-bit samples not sign-extended | negative samples become large positive ones | `test_reads_fixture_wavs_like_scipy` (mutant `s01`) |
| 2. the pad byte after an odd chunk ignored | the next chunk id is read one byte early: "no data chunk" | `test_reads_fixture_wavs_like_scipy` (mutant `s02`) |
| 3. the extensible subformat read from the wrong offset | valid 24-bit files rejected | `test_reads_fixture_wavs_like_scipy` (mutant `s03`) |
| 4. interleaved frames reshaped as planar | left and right mixed sample by sample | `test_hand_example_wav_header` (mutant `s04`) |
| 5. dividing s16 by 32767 | full scale slightly above 1, every sample off | `test_hand_example_wav_header` (mutant `s05`) |
| 6. keeping the 3001st frame | features are 3001 wide; the encoder's positions overflow | `test_log_mel_matches_hf_feature_extractor`, `test_silence_gives_the_floor_everywhere` (mutant `s06`) |
| 7. dropping the first frame instead of the last | every feature shifted by 10 ms | `test_log_mel_matches_hf_feature_extractor` (mutant `s07`) |
| 8. a symmetric Hann window | features off by about $10^{-3}$ | `test_log_mel_matches_hf_feature_extractor` (mutant `s08`) |
| 9. clipping before dropping the last frame | quiet frames clipped to the wrong floor | `test_clip_uses_the_kept_frames_only` (mutant `s09`) |
| 10. the left channel instead of the mean | a right-panned speaker disappears | `test_to_mono_averages_channels_then_resamples` (mutant `s10`) |
| 11. padding at the front | speech starts late in the window; timestamps shift | `test_pad_or_trim` (mutant `s11`) |
| 12. not checking a chunk size against the file | a truncated upload returns half a clip | `test_rejects_broken_files` (mutant `s12`) |
| 13. `write_wav` scaling by 32767 | round trips drift by one step | `test_write_wav_round_trip` (mutant `s13`) |
| accepting 384000 Hz as out of range | the top supported rate rejected | `test_rejects_broken_files` (mutant `m01`) |
| accepting 3 channels | a surround file averaged as if stereo | `test_rejects_broken_files` (mutant `m02`) |
| a RIFF size a few bytes long accepted | trailing chunk read past the buffer | `test_rejects_broken_files` (mutant `m03`) |
| clipping to 32768 | +1.0 wraps to -1.0 in int16 | `test_write_wav_round_trip` (mutant `m04`) |
| accepting `n_samples = 0` | empty windows | `test_pad_or_trim` (mutant `m05`) |
| refusing Whisper's own config | every HF checkpoint rejected | `test_feature_config_from_hf` (mutant `m06`) |
| returning float64 features | twice the memory per window; the engine's f32 parity drifts | `test_log_mel_matches_hf_feature_extractor` (mutant `m07`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M12.3` | `resample_audio` brings 22.05 or 48 kHz audio to 16 kHz |
| Back | `M12.5` | `stft(x, 400, 160, hann(400))` and `power` give the spectrogram |
| Back | `M12.6` | `mel_filterbank(16000, 400, n_mels)` and `log_mel_whisper` give the features |
| Forward | `L14.2` | the encoder's conv stem reads `[n_mels, 3000]` from here |
| Forward | `data.11` | the audio corpus pipeline parses and normalizes clips with `read_wav` and `to_mono_16k` (with B14's data group) |
| Forward | `L15.1` | the engine re-implements this frontend in Rust and is checked against these fixtures within $10^{-4}$ |

| Forward | `L14.3` | `Whisper.encode_wav` consumes the WAV parser, resampler and log-mel frontend. |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `read_wav` | `soundfile` (libsndfile), `scipy.io.wavfile` | dozens of containers and codecs, RF64 for files over 4 GiB | `scipy/io/wavfile.py` |
| `whisper_log_mel` | HF `WhisperFeatureExtractor` | batching, attention masks, a torch path on GPU | `transformers/models/whisper/feature_extraction_whisper.py` |
| `whisper_log_mel` | `openai-whisper` `log_mel_spectrogram` | the same recipe in torch with the filters loaded from `mel_filters.npz` | `whisper/audio.py` |
| `to_mono_16k` | `ffmpeg -ac 1 -ar 16000` | every codec and container; Whisper's CLI shells out to it | ffmpeg `libswresample` |
