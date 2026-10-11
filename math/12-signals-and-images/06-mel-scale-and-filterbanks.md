<!-- ss:module M12.6 -->
# Mel scale and filterbanks

## Overview

| | |
|---|---|
| **Module** | `M12.6` · build · Python · Pass 12 · 2 to 3 h |
| **You build** | `python/tinyllm/sig/mel.py`: `hz_to_mel` and `mel_to_hz` (Slaney and HTK), `mel_filterbank` (librosa's triangles with Slaney area normalization), `power_to_db`, and `log_mel_whisper` (log10, floor, 8-unit range, rescale) |
| **Contract** | [`course/contracts/py/tinyllm/sig/mel.pyi`](../../course/contracts/py/tinyllm/sig/mel.pyi) |
| **Tests** | `course/tests/M12.6/test_mel.py` (what they check: section 4), golden values from `course/oracle/M12.6/mel_librosa.py` (librosa 1.0) in `course/fixtures/M12.6/mel_librosa.npz` |
| **Needs** | `M00.1` `log_base` (or `--ref-deps`). Reading: `M12.5` (the power spectrogram the filters pool) |
| **Used by** | later `L14.1` (Whisper's log-mel) · `data.11` (mel dHash for audio near-duplicates); they join the registry with B14's L14 and data groups |
| **Milestone** | `MS-P12` (the multimodal gate) |
| **Optional depth** | Stevens, Volkmann, Newman, "A scale for the measurement of the psychological magnitude pitch" (1937); Slaney, *Auditory Toolbox* (1998), `MakeERBFilters` and `mfcc`; Radford et al., "Robust speech recognition via large-scale weak supervision" (2022), section 2.2 |

## Key Takeaways

- The Slaney mel scale is linear below 1000 Hz (15 mel, $200/3$ Hz per mel) and logarithmic above (27 mel per factor 6.4); HTK's is $2595 \log_{10}(1 + f/700)$ (`test_hand_example_mel_scale`).
- A mel filterbank is $n_\text{mels}$ triangles with edges equally spaced in mel, sampled at the rfft bin frequencies (`test_triangles_have_the_declared_edges`, `test_whisper_filterbanks_match_librosa`).
- Slaney normalization gives every triangle unit area in Hz, so wide high bands do not collect more energy for being wide (`test_slaney_rows_have_unit_area`).
- Power becomes decibels as $10 \log_{10}$ with a floor; Whisper uses $\log_{10}$, clips 8 units (80 dB) below the maximum, and maps by $(x + 4)/4$ (`test_hand_example_decibels`, `test_whisper_log_mel_scaling`).

## How to work this chapter

```bash
ss start M12.6              # stubs python/tinyllm/sig/mel.py into your repo
ss tests M12.6              # read the test catalog first: rung R0, you write no tests here
ss check M12.6              # exit code is the verdict
ss check M12.6 --ref-deps   # only if your M00.1 is not passing yet
ss diff  M12.6              # after passing: your code against the reference
```

---

## 1. Why now

`M12.5` turns 25 ms of audio into 201 power values per frame, evenly spaced from 0 to 8 kHz. Whisper's encoder does not take those 201 numbers: it takes 80 (or 128 for large-v3) mel bands, each a weighted sum of bins, then a clipped logarithm. The weights are librosa's mel filters, shipped inside Whisper as `mel_filters.npz`; if your filters differ in the fifth decimal, every feature `L14.1` hands the encoder is wrong by a little, and parity with HF's `WhisperFeatureExtractor` (within $10^{-4}$) fails. The audio near-duplicate detector in `data.11` hashes the same log-mel picture. This module builds the filters and the logs exactly as the reference libraries do.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $f$ | frequency | Hz |
| $m(f)$ | mel value of $f$ | mel |
| $e_0, \dots, e_{n_\text{mels}+1}$ | filter edges, equally spaced in mel | Hz |
| $F_i(f)$ | triangle $i$ at frequency $f$ | float |
| $f_k = k\,\text{sr}/n_\text{fft}$ | center of rfft bin $k$ | Hz |
| $P$ | power spectrogram | `float64[n_fft//2 + 1, T]` |
| $\text{amin}$ | floor before a log | float |
| $\text{top\_db}$ | dynamic range kept below the maximum | dB |

### 2.1 The mel scale

Listeners judge equal pitch steps as roughly equal frequency steps below about 1 kHz and equal frequency ratios above. The Slaney scale (librosa's default, used by Whisper) models this piecewise:
$$m(f) = \begin{cases} f / (200/3) & f < 1000 \\ 15 + \dfrac{\ln(f / 1000)}{\ln(6.4)/27} & f \ge 1000 \end{cases}$$
so 1000 Hz is 15 mel and every factor of 6.4 in frequency adds 27 mel. The HTK scale is one smooth formula, $m(f) = 2595 \log_{10}(1 + f/700)$. The two give different numbers and different filters; Whisper uses Slaney.

### 2.2 Triangular filters and their normalization

Place $n_\text{mels} + 2$ points equally spaced in mel between $m(f_\text{min})$ and $m(f_\text{max})$ (default $\text{sr}/2$) and map them back to Hz: $e_0 < e_1 < \dots < e_{n_\text{mels}+1}$. Filter $i$ rises linearly from 0 at $e_i$ to 1 at $e_{i+1}$ and falls back to 0 at $e_{i+2}$:
$$F_i(f) = \max\left(0,\; \min\left(\frac{f - e_i}{e_{i+1} - e_i},\; \frac{e_{i+2} - f}{e_{i+2} - e_{i+1}}\right)\right),$$
sampled at the bin centers $f_k$. Neighbouring triangles overlap by half, and high triangles are much wider in Hz than low ones. With Slaney normalization each row is multiplied by $2/(e_{i+2} - e_i)$; a triangle of height 1 and base $e_{i+2} - e_i$ has area $(e_{i+2} - e_i)/2$, so every normalized filter has unit area and measures power density rather than total power. The filterbank is the matrix $M \in \mathbb{R}^{n_\text{mels} \times (n_\text{fft}/2+1)}$, and the mel spectrogram is $M P$.

### 2.3 Decibels

Loudness is perceived on a log scale too. A power ratio in decibels is $10 \log_{10}(p / \text{ref})$: a factor of 10 in power is 10 dB. (Amplitudes use $20 \log_{10}$, because power is amplitude squared; using 20 on a power doubles every number.) `power_to_db` floors both $p$ and ref at `amin` so $\log 0$ never happens, and with `top_db` raises everything to at least the maximum minus `top_db`, so silence does not stretch the range to $-\infty$. The base-10 logarithm is `M00.1`'s `log_base(x, 10)`: a change of base, $\ln x / \ln 10$.

### 2.4 Whisper's log-mel

Whisper's features skip decibels and use $\log_{10}$ directly:
$$L = \log_{10}(\max(M P, 10^{-10})), \qquad L \leftarrow \max(L, \max L - 8), \qquad \text{features} = (L + 4)/4 .$$
The floor $10^{-10}$ is $-100$ dB; the clip keeps 8 units, 80 dB, below the loudest bin of the clip; the affine map puts typical speech near $[-1, 1]$. Silence floors everywhere at $\log_{10} 10^{-10} = -10$, which maps to $-1.5$. The maximum is taken over the whole clip, so the features of a frame depend on the loudest moment anywhere in the 30 s window.

## 3. Worked example by hand

Mel values on the Slaney scale: $500$ Hz is $500 \cdot 3/200 = 7.5$ mel; $1000$ Hz is 15 mel; $6400$ Hz is $15 + 27 \ln(6.4)/\ln(6.4) = 42$ mel. On the HTK scale $700$ Hz is $2595 \log_{10} 2 = 781.17$ mel (`test_hand_example_mel_scale`).

Decibels of the powers $(1, 10, 10^{-12})$ with ref 1 and $\text{amin} = 10^{-10}$: $10 \log_{10} 1 = 0$, $10 \log_{10} 10 = 10$, and $10^{-12}$ is floored to $10^{-10}$, giving $-100$. With $\text{top\_db} = 80$ the floor is $10 - 80 = -70$, so the result is $(0, 10, -70)$. With ref 10 everything drops by 10 dB: $(-10, 0, -110)$ when not clipped. This is `test_hand_example_decibels`.

Whisper's scaling of the identity filterbank applied to $P = \begin{pmatrix} 10^{-12} & 100 \\ 1 & 10^4 \end{pmatrix}$: $\log_{10}$ with the floor gives $\begin{pmatrix} -10 & 2 \\ 0 & 4 \end{pmatrix}$; clipping at $4 - 8 = -4$ gives $\begin{pmatrix} -4 & 2 \\ 0 & 4 \end{pmatrix}$; and $(L + 4)/4 = \begin{pmatrix} 0 & 1.5 \\ 1 & 2 \end{pmatrix}$ (in `test_whisper_log_mel_scaling`).

## 4. The interface

```python
def hz_to_mel(f, scale='slaney') -> NDArray: ...
def mel_to_hz(m, scale='slaney') -> NDArray: ...
def mel_filterbank(sr, n_fft, n_mels, fmin=0.0, fmax=None, scale='slaney', norm='slaney') -> NDArray: ...
def power_to_db(p, ref=1.0, amin=1e-10, top_db=80.0) -> NDArray: ...
def log_mel_whisper(power_spec, filters) -> NDArray: ...
```

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_mel_scale` | unit, smoke | section 3's mel values on both scales | the scale itself |
| `test_hand_example_decibels` | unit, smoke | section 3's decibels with and without `top_db` and with a ref | the dB conventions |
| `test_whisper_filterbanks_match_librosa` | golden | 80 and 128 bands at 16 kHz, $n_\text{fft} = 400$, within $10^{-12}$ of float64 librosa and $10^{-7}$ of the float32 filters Whisper ships | `L14.1`'s features |
| `test_other_filterbanks_and_scales_match_librosa` | golden | an HTK bank without normalization, a band-limited bank, both scale conversions, the default fmax | edges and scales |
| `test_power_to_db_matches_librosa` | golden | `power_to_db` with defaults and with ref, amin, no top_db | decibel parity |
| `test_triangles_have_the_declared_edges` | property | every row is the triangle formula on the mel-spaced edges; zero outside | the definition |
| `test_slaney_rows_have_unit_area` | property | on a fine grid each normalized row integrates to 1 | why the normalization exists |
| `test_whisper_log_mel_scaling` | unit | section 3's matrix, silence at $-1.5$, range at most 2 units | Whisper's feature scaling |
| `test_rejects_bad_arguments` | boundary | amin 0, negative top_db, complex input, empty band, unknown norm or scale, mismatched shapes; input untouched | caller bugs fail early |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. the Slaney scale linear everywhere | high bands far too narrow | `test_hand_example_mel_scale`, `test_whisper_filterbanks_match_librosa` (mutant `s01`) |
| 2. skipping the Slaney normalization | high bands dominate every feature | `test_slaney_rows_have_unit_area`, `test_whisper_filterbanks_match_librosa` (mutant `s02`) |
| 3. edges equally spaced in Hz | a linear filterbank, not a mel one | `test_triangles_have_the_declared_edges`, `test_whisper_filterbanks_match_librosa` (mutant `s03`) |
| 4. $20 \log_{10}$ on a power | every dB value doubled | `test_hand_example_decibels`, `test_power_to_db_matches_librosa` (mutant `s04`) |
| 5. clipping `top_db` against 0 dB instead of the maximum | quiet recordings clipped flat | `test_hand_example_decibels`, `test_power_to_db_matches_librosa` (mutant `s05`) |
| 6. the natural log in Whisper's features | features scaled by $\ln 10$ | `test_whisper_log_mel_scaling` (mutant `s06`) |
| 7. Whisper's range as 80 instead of 8 | the clip never happens | `test_whisper_log_mel_scaling` (mutant `s07`) |
| accepting amin = 0 | $\log 0 = -\infty$ in the output | `test_rejects_bad_arguments` (mutant `m01`) |
| accepting fmin = fmax | a filterbank of NaNs | `test_rejects_bad_arguments` (mutant `m02`) |
| accepting a negative top_db | every value clipped above the maximum | `test_rejects_bad_arguments` (mutant `m03`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M00.1` | `power_to_db` and `log_mel_whisper` take base-10 logs with `log_base(x, 10)` |
| Back | `M12.5` | the filters pool its power spectrogram (reading) |
| Forward | `L14.1` | `log_mel_whisper(power(stft(...))[:, :-1], mel_filterbank(16000, 400, 80))` (with B14's L14 group) |
| Forward | `data.11` | mel dHash windows for audio near-duplicates |
| Forward | `S-M12` | mel, logs, decibels, and Whisper's scaling by hand |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `mel_filterbank` | `librosa.filters.mel` | the same filters, cached; `norm` as any $L^p$ norm | `librosa/filters.py` |
| `log_mel_whisper` | HF `WhisperFeatureExtractor` | the full frontend in numpy or torch, batched, with padding to 30 s | `transformers/models/whisper/feature_extraction_whisper.py` |
| `power_to_db` | `torchaudio.transforms.AmplitudeToDB` | the same floor and `top_db` on tensors | `torchaudio/functional/functional.py` |
| filters in Rust | the engine's `tl-engine/src/mm/audio.rs` (`L15.1`) | log-mel from the committed filter file, proven against this module's fixtures | `L15.1` chapter |
