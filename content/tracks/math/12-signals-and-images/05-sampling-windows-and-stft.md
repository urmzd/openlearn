<!-- ol:module M12.5 -->
# Sampling, Nyquist, windows, and the STFT

## Overview

| | |
|---|---|
| **Module** | `M12.5` · build · Python · Pass 12 · 3 h |
| **You build** | `python/tinyllm/sig/stft.py`: `hann` (periodic and symmetric), `frame_count`, `stft` (torch.stft semantics: center reflect padding, explicit window), `istft` (weighted overlap-add), and `power` |
| **Contract** | [`course/contracts/py/tinyllm/sig/stft.pyi`](../../../../course/contracts/py/tinyllm/sig/stft.pyi) |
| **Tests** | `course/tests/M12.5/test_stft.py` (what they check: section 4), golden values from `course/oracle/M12.5/stft_torch.py` (`torch.stft`, `torch.hann_window`) in `course/fixtures/M12.5/stft_torch.npz` |
| **Needs** | `M12.4` `rfft` and `irfft` (or `--ref-deps`) |
| **Used by** | `L14.1` the Whisper frontend (power spectrogram of 400-sample frames at hop 160) |
| **Milestone** | `MS-P12` (the multimodal gate) |
| **Optional depth** | Oppenheim and Schafer, *Discrete-Time Signal Processing*, ch. 4 and 10; Harris, "On the use of windows for harmonic analysis with the discrete Fourier transform" (Proc. IEEE 1978); Smith, *Spectral Audio Signal Processing* (free), the overlap-add chapters |

## Key Takeaways

- Sampling at rate $\text{sr}$ can only hold frequencies below $\text{sr}/2$; a tone above folds to $|f - k\,\text{sr}|$ (`test_tone_above_nyquist_folds`).
- A window tapers each frame so its edges do not smear energy across every bin; the periodic Hann window is three cosines, so a tone on a bin lands in exactly three bins (`test_power_and_periodic_hann_leakage`, `test_hann_matches_torch`).
- `center=True` pads $n_\text{fft}/2$ reflected samples on each side, so there are $1 + \lfloor n / \text{hop} \rfloor$ frames: 3001 for 30 s of Whisper audio, which then drops the last (`test_frame_count_table`, `test_hand_example_stft`).
- When the squared window overlap-adds to a constant (periodic Hann at hop $n_\text{fft}/4$), weighted overlap-add inverts the STFT exactly (`test_istft_reconstructs_under_cola`).

## How to work this chapter

```bash
ol start M12.5              # stubs python/tinyllm/sig/stft.py into your repo
ol tests M12.5              # read the test catalog first: rung R0, you write no tests here
ol check M12.5              # exit code is the verdict
ol check M12.5 --ref-deps   # only if your M12.4 is not passing yet
ol diff  M12.5              # after passing: your code against the reference
```

---

## 1. Why now

`M12.4` transforms one block of samples. Speech is not one block: its spectrum changes every few tens of milliseconds, and Whisper's encoder wants a picture of that change, 100 spectra per second. `L14.1` computes exactly `torch.stft(audio, 400, 160, window=hann_window(400), center=True, pad_mode='reflect')` and squares the magnitudes; Whisper was trained on those numbers. Using a symmetric window, zero padding, or a frame count off by one gives features that look like a spectrogram and shift the transcription. The same module explains why `M12.3` must filter before it downsamples: aliasing is a sampling fact, and here you can see it in one bin.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $\text{sr}$ | sample rate | Hz |
| $f_N = \text{sr}/2$ | Nyquist frequency | Hz |
| $n_\text{fft}$ | frame length (even) | int |
| $\text{hop}$ | samples between frame starts | int |
| $w[j]$ | analysis window | `float64[n_fft]` |
| $x_p$ | the signal after center padding | `float64[n + 2 (n_fft // 2)]` |
| $S[f, t]$ | STFT bin $f$ of frame $t$ | `complex128[n_fft//2 + 1, T]` |
| $f\,\text{sr}/n_\text{fft}$ | frequency of bin $f$ | Hz |

### 2.1 Sampling and aliasing

Sampling $\cos(2\pi f t)$ at $t = j/\text{sr}$ gives $\cos(2\pi f j/\text{sr})$, and since $\cos$ has period $2\pi$ in its argument, the frequencies $f$ and $f + k\,\text{sr}$ give identical samples, as do $f$ and $\text{sr} - f$. So only $[0, \text{sr}/2]$ is distinguishable: the Nyquist limit. A 5 kHz tone sampled at 8 kHz produces exactly the samples of a 3 kHz tone, and no later processing can tell them apart. Anything above $f_N$ must be removed before sampling or resampling, which is why `M12.3` low-passes (antialiasing) and why Whisper's 16 kHz input cannot represent anything above 8 kHz.

### 2.2 Windows and leakage

The DFT of a frame treats the frame as one period of a periodic signal. A sinusoid that does not complete a whole number of periods in the frame has a jump where the period wraps, and the jump spreads energy into every bin: leakage. A window that falls to 0 at the ends removes the jump. The Hann window is
$$w[j] = \tfrac12 - \tfrac12 \cos\frac{2\pi j}{N}, \qquad j = 0, \dots, n - 1,$$
with $N = n$ (periodic) or $N = n - 1$ (symmetric). The periodic version is one period of a cosine of the DFT size, so its own DFT is three nonzero bins, $(-\tfrac14, \tfrac12, -\tfrac14) \cdot n$ at $-1, 0, 1$: multiplying a tone by it convolves the tone's single bin with those three. The symmetric version is the classic filter-design window and leaks a little everywhere. Spectral analysis uses periodic; `torch.hann_window` defaults to it, and so does Whisper.

### 2.3 The STFT and its inverse

Frame $t$ starts at sample $t\,\text{hop}$ of the padded signal; its spectrum is the rfft of the windowed frame:
$$S[f, t] = \sum_{j=0}^{n_\text{fft}-1} w[j]\, x_p[t\,\text{hop} + j]\, e^{-2\pi i f j / n_\text{fft}}, \qquad f = 0, \dots, n_\text{fft}/2 .$$
To invert, take the irfft of each column, multiply by the window again, and add the frames back at their offsets (overlap-add). Sample $m$ then holds $x_p[m] \sum_t w[m - t\,\text{hop}]^2$, so dividing by the overlap-added squared window recovers $x_p$ wherever that sum is nonzero (the NOLA condition). For the periodic Hann window at $\text{hop} = n_\text{fft}/4$ the sum is the constant $1.5$ (constant overlap-add, COLA), so the reconstruction is exact to rounding. Removing the $n_\text{fft}/2$ padding at the start returns the original signal.

### 2.4 Center padding and frame counts

With `center=True` the signal is padded by $n_\text{fft}/2$ samples on each side, so frame $t$ is centered on sample $t\,\text{hop}$ of the original. Reflect padding mirrors without repeating the edge sample, $(x_2, x_1 \mid x_0, x_1, x_2, \dots)$, which keeps the signal continuous at the boundary where zeros would create a click. The frame count is
$$T = 1 + \left\lfloor \frac{n + 2\lfloor n_\text{fft}/2 \rfloor - n_\text{fft}}{\text{hop}} \right\rfloor = 1 + \left\lfloor \frac{n}{\text{hop}} \right\rfloor \text{ for even } n_\text{fft},$$
and without padding $T = 1 + \lfloor (n - n_\text{fft})/\text{hop} \rfloor$. Whisper pads every clip to 30 s, $n = 480000$, gets $1 + 3000 = 3001$ frames, and drops the last one so the encoder sees 3000. Reflect padding needs more than $n_\text{fft}/2$ samples; `torch.stft` raises otherwise, and so does `stft`.

### 2.5 Power

The spectrogram Whisper uses is the power $|S|^2 = \operatorname{Re}(S)^2 + \operatorname{Im}(S)^2$, computed without a square root (no need to take a root you square again). `M12.6` pools it into mel bands and takes a log.

## 3. Worked example by hand

$x = (1, 2, 3, 4, 5)$, $n_\text{fft} = 4$, $\text{hop} = 2$, periodic Hann $w = (0, 0.5, 1, 0.5)$ (the symmetric one would be $(0, 0.75, 0.75, 0)$). Reflect padding by 2 gives $(3, 2, 1, 2, 3, 4, 5, 4, 3)$, and $T = 1 + \lfloor 5/2 \rfloor = 3$ frames:

| $t$ | frame | windowed $v$ | $X_0 = \sum v$ | $X_1 = (v_0 - v_2) + i(v_3 - v_1)$ | $X_2 = v_0 - v_1 + v_2 - v_3$ |
|---|---|---|---|---|---|
| 0 | $(3, 2, 1, 2)$ | $(0, 1, 1, 1)$ | 3 | $-1$ | $-1$ |
| 1 | $(1, 2, 3, 4)$ | $(0, 1, 3, 2)$ | 6 | $-3 + i$ | 0 |
| 2 | $(3, 4, 5, 4)$ | $(0, 2, 5, 2)$ | 9 | $-5$ | 1 |

So $S = \begin{pmatrix} 3 & 6 & 9 \\ -1 & -3+i & -5 \\ -1 & 0 & 1 \end{pmatrix}$, bins down, frames across. This is `test_hand_example_stft`.

## 4. The interface

```python
def hann(n, periodic=True) -> NDArray: ...
def frame_count(n_samples, n_fft, hop, center=True) -> int: ...
def stft(x, n_fft, hop, window, center=True, pad_mode='reflect') -> NDArray: ...   # [n_fft//2 + 1, T]
def istft(S, n_fft, hop, window, length) -> NDArray: ...
def power(S) -> NDArray: ...
```

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_stft` | unit, smoke | section 3's windows and spectrogram | padding, windowing, and orientation |
| `test_frame_count_table` | unit, smoke | Whisper's 3001, centered and uncentered counts, bad arguments | `L14.1`'s 3000 frames |
| `test_hann_matches_torch` | golden | `torch.hann_window` for $n = 1, 2, 8, 400$, both flags | Whisper's window |
| `test_stft_matches_torch` | golden | Whisper-sized, small, zero-padded symmetric, and short-hop cases within $10^{-5}$ | `L14.1` parity |
| `test_istft_reconstructs_under_cola` | property | round trip error below $10^{-6}$ at hop $n_\text{fft}/4$ | nothing is lost in the frames |
| `test_tone_above_nyquist_folds` | property | 5 kHz at 8 kHz peaks at the 3 kHz bin, 96 | aliasing is a sampling fact |
| `test_power_and_periodic_hann_leakage` | property | power is $\lvert S \rvert^2$; a bin-centered tone fills bins 19 to 21 in ratio $1 : 4 : 1$ | why periodic Hann |
| `test_rejects_bad_frames` | boundary | wrong window length, too short to reflect, hop 0, bad pad mode, short uncentered signal, wrong spectrogram height; input untouched | misconfigured frontends fail early |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. the symmetric Hann window | leakage into every bin; features drift from what Whisper was trained on | `test_hann_matches_torch`, `test_power_and_periodic_hann_leakage` (mutant `s01`) |
| 2. zero padding instead of reflect | a click in the first and last frames | `test_hand_example_stft`, `test_stft_matches_torch` (mutant `s02`) |
| 3. the centered frame count without the $+1$ | 3000 frames before Whisper drops one: 2999 | `test_frame_count_table` (mutant `s03`) |
| 4. forgetting to window the frames | leakage everywhere; the round trip breaks | `test_hand_example_stft`, `test_istft_reconstructs_under_cola` (mutant `s04`) |
| 5. normalizing overlap-add by $\sum w$, not $\sum w^2$ | the reconstruction scaled by about 0.75 | `test_istft_reconstructs_under_cola` (mutant `s05`) |
| 6. returning frames by bins | every consumer transposes or breaks | `test_stft_matches_torch`, `test_tone_above_nyquist_folds` (mutant `s06`) |
| 7. power as the magnitude $\lvert S \rvert$ | log-mel features halved | `test_power_and_periodic_hann_leakage` (mutant `s07`) |
| reflect-padding a signal of exactly $n_\text{fft}/2$ samples | numpy repeats the reflection where torch raises | `test_rejects_bad_frames` (mutant `m01`) |
| accepting hop 0 | a division by zero instead of a clear error | `test_rejects_bad_frames` (mutant `m02`) |
| accepting a signal one sample short of a frame | zero frames instead of an error | `test_frame_count_table` (mutant `m03`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M12.4` | `stft` runs `rfft` on every frame; `istft` runs `irfft` |
| Forward | `M12.6` | its filterbank pools this power spectrogram into mel bands (reading) |
| Forward | `L14.1` | the Whisper frontend: `stft(x, 400, 160, hann(400))`, drop the last frame, `power` |
| Forward | `S-M12` | Nyquist, aliasing, windows, COLA, and frame counts by hand |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `stft` | `torch.stft` | batched frames as a strided view, cuFFT on GPU, onesided and normalized flags | `aten/src/ATen/native/SpectralOps.cpp` |
| `istft` | `librosa.istft` | window-sum caching, Griffin-Lim phase recovery on top | `librosa/core/spectrum.py` |
| `hann` | `scipy.signal.get_window` | dozens of windows; `sym=False` for the periodic ones | `scipy/signal/windows/_windows.py` |
| Whisper's frontend in C | optional `L14.6` | STFT power with center reflect padding and drop-last in C | `c/src/kernels/audio.c` |
