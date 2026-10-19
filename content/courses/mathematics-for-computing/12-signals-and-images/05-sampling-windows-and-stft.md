# Sampling, Nyquist, windows, and the STFT

The previous section, [DFT and FFT](04-dft-and-fft.md), transforms one block of samples. Speech is not one block: its spectrum changes every few tens of milliseconds, and Whisper's encoder wants a picture of that change, 100 spectra per second. Whisper's audio frontend computes exactly `torch.stft(audio, 400, 160, window=hann_window(400), center=True, pad_mode='reflect')` and squares the magnitudes, and Whisper was trained on those numbers. Using a symmetric window, zero padding, or a frame count off by one gives features that look like a spectrogram and shift the transcription.

This section derives the Nyquist limit, windowing and spectral leakage, the short-time Fourier transform (STFT) and its overlap-add inverse, and the frame counts that center padding produces, and then works a small spectrogram by hand. Along the way it explains why [Resampling and interpolation](03-resampling-and-interpolation.md) must filter before it downsamples: aliasing is a sampling fact, and here you will see it show up in one bin. The only code this section depends on is `rfft` and `irfft` from the previous section.

The reference implementation is [`python/tinyllm/sig/stft.py`](../../../../course/solutions/python/tinyllm/sig/stft.py), with its interface in [`course/contracts/py/tinyllm/sig/stft.pyi`](../../../../course/contracts/py/tinyllm/sig/stft.pyi). It provides `hann` (periodic and symmetric), `frame_count`, `stft` (`torch.stft` semantics: center reflect padding, explicit window), `istft` (weighted overlap-add), and `power`.

> **Learning objectives**
> - State the Nyquist limit and compute the alias of a tone sampled below twice its frequency.
> - Explain spectral leakage and why the periodic Hann window puts a bin-centered tone into exactly three bins.
> - Compute STFT frame counts with and without center padding, including Whisper's 3001.
> - Invert an STFT by weighted overlap-add and state when the reconstruction is exact.

## Notation

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

## Sampling and aliasing

Sampling $\cos(2\pi f t)$ at $t = j/\text{sr}$ gives $\cos(2\pi f j/\text{sr})$, and since $\cos$ has period $2\pi$ in its argument, the frequencies $f$ and $f + k\,\text{sr}$ give identical samples, as do $f$ and $\text{sr} - f$. So only $[0, \text{sr}/2]$ is distinguishable: the Nyquist limit. A tone above it folds to $|f - k\,\text{sr}|$. A 5 kHz tone sampled at 8 kHz produces exactly the samples of a 3 kHz tone, and no later processing can tell them apart. Anything above $f_N$ must be removed before sampling or resampling, which is why the resampler low-passes (antialiasing) and why Whisper's 16 kHz input cannot represent anything above 8 kHz.

## Windows and leakage

Sampling fixes which frequencies exist. The next problem is measuring them from a short frame. The DFT of a frame treats the frame as one period of a periodic signal. A sinusoid that does not complete a whole number of periods in the frame has a jump where the period wraps, and the jump spreads energy into every bin: leakage. A window that falls to 0 at the ends removes the jump. The Hann window is
$$w[j] = \tfrac12 - \tfrac12 \cos\frac{2\pi j}{N}, \qquad j = 0, \dots, n - 1,$$
with $N = n$ (periodic) or $N = n - 1$ (symmetric).

The periodic version is one period of a cosine of the DFT size, so its own DFT is three nonzero bins, $(-\tfrac14, \tfrac12, -\tfrac14) \cdot n$ at $-1, 0, 1$: multiplying a tone by it convolves the tone's single bin with those three. The symmetric version is the classic filter-design window and leaks a little everywhere. Spectral analysis uses periodic; `torch.hann_window` defaults to it, and so does Whisper.

## The STFT and its inverse

With a window chosen, the STFT is just the DFT applied frame by frame. Frame $t$ starts at sample $t\,\text{hop}$ of the padded signal; its spectrum is the rfft of the windowed frame:
$$S[f, t] = \sum_{j=0}^{n_\text{fft}-1} w[j]\, x_p[t\,\text{hop} + j]\, e^{-2\pi i f j / n_\text{fft}}, \qquad f = 0, \dots, n_\text{fft}/2 .$$
To invert, take the irfft of each column, multiply by the window again, and add the frames back at their offsets (overlap-add). Sample $m$ then holds $x_p[m] \sum_t w[m - t\,\text{hop}]^2$, so dividing by the overlap-added squared window recovers $x_p$ wherever that sum is nonzero (the NOLA condition). For the periodic Hann window at $\text{hop} = n_\text{fft}/4$ the sum is the constant $1.5$ (constant overlap-add, COLA), so the reconstruction is exact to rounding. Removing the $n_\text{fft}/2$ padding at the start returns the original signal.

## Center padding and frame counts

The padded signal $x_p$ deserves a closer look, because it decides how many frames there are. With `center=True` the signal is padded by $n_\text{fft}/2$ samples on each side, so frame $t$ is centered on sample $t\,\text{hop}$ of the original. Reflect padding mirrors without repeating the edge sample, $(x_2, x_1 \mid x_0, x_1, x_2, \dots)$, which keeps the signal continuous at the boundary where zeros would create a click. The frame count is
$$T = 1 + \left\lfloor \frac{n + 2\lfloor n_\text{fft}/2 \rfloor - n_\text{fft}}{\text{hop}} \right\rfloor = 1 + \left\lfloor \frac{n}{\text{hop}} \right\rfloor \text{ for even } n_\text{fft},$$
and without padding $T = 1 + \lfloor (n - n_\text{fft})/\text{hop} \rfloor$. Whisper pads every clip to 30 s, $n = 480000$, gets $1 + 3000 = 3001$ frames, and drops the last one so the encoder sees 3000. Reflect padding needs more than $n_\text{fft}/2$ samples; `torch.stft` raises otherwise, and so does `stft`.

## Power

The spectrogram Whisper uses is the power $|S|^2 = \operatorname{Re}(S)^2 + \operatorname{Im}(S)^2$, computed without a square root (no need to take a root you square again). The next section, [Mel scale and filterbanks](06-mel-scale-and-filterbanks.md), pools it into mel bands and takes a log.

## Working it by hand

Take $x = (1, 2, 3, 4, 5)$, $n_\text{fft} = 4$, $\text{hop} = 2$, and the periodic Hann window $w = (0, 0.5, 1, 0.5)$ (the symmetric one would be $(0, 0.75, 0.75, 0)$). Reflect padding by 2 gives $(3, 2, 1, 2, 3, 4, 5, 4, 3)$, and $T = 1 + \lfloor 5/2 \rfloor = 3$ frames:

| $t$ | frame | windowed $v$ | $X_0 = \sum v$ | $X_1 = (v_0 - v_2) + i(v_3 - v_1)$ | $X_2 = v_0 - v_1 + v_2 - v_3$ |
|---|---|---|---|---|---|
| 0 | $(3, 2, 1, 2)$ | $(0, 1, 1, 1)$ | 3 | $-1$ | $-1$ |
| 1 | $(1, 2, 3, 4)$ | $(0, 1, 3, 2)$ | 6 | $-3 + i$ | 0 |
| 2 | $(3, 4, 5, 4)$ | $(0, 2, 5, 2)$ | 9 | $-5$ | 1 |

So $S = \begin{pmatrix} 3 & 6 & 9 \\ -1 & -3+i & -5 \\ -1 & 0 & 1 \end{pmatrix}$, bins down, frames across.

## Designing the interface

```python
def hann(n, periodic=True) -> NDArray: ...
def frame_count(n_samples, n_fft, hop, center=True) -> int: ...
def stft(x, n_fft, hop, window, center=True, pad_mode='reflect') -> NDArray: ...   # [n_fft//2 + 1, T]
def istft(S, n_fft, hop, window, length) -> NDArray: ...
def power(S) -> NDArray: ...
```

The window is passed explicitly, as in `torch.stft`, so the caller chooses periodic or symmetric once. The defaults (`periodic=True`, `center=True`, `pad_mode='reflect'`) are Whisper's, and the output orientation is torch's, bins down and frames across. `istft` takes the original `length` so it can crop the center padding exactly. A window of the wrong length, a signal too short to reflect, a hop of 0, an unknown pad mode, an uncentered signal shorter than a frame, and a spectrogram of the wrong height are all rejected, and the input is never modified.

## Building it step by step

Seven decisions produce the implementation.

1. **The window is a formula with one choice in it.** Hann is $\tfrac12 - \tfrac12\cos(2\pi j/N)$, and the only decision is $N$: $n$ for the periodic window an FFT frame repeats, $n - 1$ for the symmetric one. Lengths 0 and 1 have no period to speak of; they return ones (as `torch.hann_window` does), which also avoids dividing by $N = 0$.
2. **The frame count follows from the padded length.** Frames start at $0, \text{hop}, 2\,\text{hop}, \dots$ and each needs $n_\text{fft}$ samples, so $T = 1 + \lfloor (L - n_\text{fft})/\text{hop} \rfloor$ for a padded length $L$. With center padding $L = n + 2\lfloor n_\text{fft}/2 \rfloor$; without it $L = n$ and $n < n_\text{fft}$ gives no frame at all, which is an error, not 0 frames.
3. **Padding happens before framing, so framing never needs a boundary case.** Reflect mode is `np.pad(..., mode="reflect")`, which mirrors without repeating the edge sample, the same as torch. NumPy would silently repeat the reflection if the signal is shorter than the pad; torch raises, so the solution checks `a.size <= half` first.
4. **All frames are one fancy-index.** The sample indices of frame $t$ are $t\,\text{hop} + j$; a column of starts plus a row of offsets broadcasts to a `[T, n_fft]` index array, and indexing gives every frame at once. Multiply by the window (broadcast over rows) and take `rfft` along the last axis: `[T, n_fft/2 + 1]`.
5. **Orientation is part of the contract.** torch returns bins down, frames across, `[n_fft/2 + 1, T]`, and so does Whisper's frontend code. One transpose, made contiguous, matches it.
6. **The inverse divides by what overlap-add multiplied.** Each output sample $m$ receives $x_p[m]\,w[m - t\,\text{hop}]^2$ from every frame $t$ covering it (once from analysis, once from the synthesis window). Accumulating $\sum_t w^2$ alongside and dividing undoes exactly that, whether or not the sum is constant; it only has to be nonzero (NOLA). Then crop the $n_\text{fft}/2$ center pad.
7. **Power avoids the root.** $|S|^2 = \operatorname{Re}^2 + \operatorname{Im}^2$; `np.abs(S)**2` would take a square root and square it again, losing a little precision for nothing.

In the listings below, routine argument validation is elided as `...`; the bodies are as in the solution. Steps 1 and 2 come first:

```python
def hann(n: int, periodic: bool = True) -> NDArray:
    if n < 0:
        raise ValueError(f"hann: n must be >= 0, got {n}")
    if n <= 1:
        return np.ones(n)          # torch's convention; also avoids period = 0
    # Periodic: one period of a length-n cosine (the window an FFT frame
    # repeats); symmetric: the length-n window whose last sample is 0 again.
    period = n if periodic else n - 1
    k = np.arange(n)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * k / period)


def frame_count(n_samples, n_fft, hop, center=True) -> int:
    # ... all sizes >= 1 ...
    if center:
        # Padded length n + 2 (n_fft // 2); frames start every hop samples.
        # For even n_fft this is 1 + n // hop: the pad exactly pays for one frame.
        return 1 + (n_samples + 2 * (n_fft // 2) - n_fft) // hop
    if n_samples < n_fft:
        raise ValueError(...)      # not enough samples for even one frame
    return 1 + (n_samples - n_fft) // hop
```

For Whisper, $1 + 480000 // 160 = 3001$.

Steps 3 to 5 are the forward transform:

```python
def stft(x, n_fft, hop, window, center=True, pad_mode="reflect"):
    # ... 1-D signal, window of exactly n_fft samples, hop >= 1 ...
    if center:
        half = n_fft // 2
        if pad_mode == "reflect":
            if a.size <= half:     # torch raises here; numpy would wrap the mirror
                raise ValueError(...)
            # Mirror without repeating the edge sample: [3, 2, | 1, 2, 3].
            a = np.pad(a, half, mode="reflect")
        elif pad_mode == "constant":
            a = np.pad(a, half)    # zeros: allowed, but clicks at the ends
        else:
            raise ValueError(...)
    if a.size < n_fft:
        raise ValueError(...)
    t = 1 + (a.size - n_fft) // hop                       # the frame_count formula
    starts = np.arange(t)[:, None] * hop                  # [T, 1]
    frames = a[starts + np.arange(n_fft)[None, :]] * w  # [frames, n_fft]
    return np.ascontiguousarray(rfft(frames).T)  # [n_fft // 2 + 1, frames]
```

`rfft` transforms the last axis of the whole `[T, n_fft]` batch in one call, and requires $n_\text{fft}$ even with factors 2, 3, 5; 400 qualifies.

Steps 6 and 7 are the inverse and the power:

```python
def istft(S, n_fft, hop, window, length):
    # ... S is [n_fft // 2 + 1, frames], window has n_fft samples ...
    t = spec.shape[1]
    frames = irfft(spec.T, n_fft) * w  # windowed again (synthesis window)
    total = n_fft + hop * (t - 1)                         # length of the padded signal covered
    y = np.zeros(total)
    wsum = np.zeros(total)
    for i in range(t):
        y[i * hop : i * hop + n_fft] += frames[i]         # overlap-add the frames
        wsum[i * hop : i * hop + n_fft] += w * w          # ...and what each sample was scaled by
    # Weighted overlap-add: divide by the summed squared window, which is
    # constant in the interior when (window^2, hop) is COLA.
    half = n_fft // 2
    sl = slice(half, half + length)                       # undo the center padding
    den = wsum[sl]
    if den.size < length or (np.abs(den) < 1e-11).any():
        raise ValueError(
            "istft: the window overlap-add vanishes; the frames cannot be inverted"
        )
    return y[sl] / den


def power(S: NDArray) -> NDArray:
    s = np.asarray(S)
    # |z|^2 = re^2 + im^2, with no square root to undo.
    return s.real * s.real + s.imag * s.imag
```

The check on `den` matters at the edges: the periodic Hann window is 0 at $j = 0$, so the very first padded sample gets no weight from any frame. That sample lies in the cropped padding, so the division only ever happens where the frames cover the signal.

The full solution is [`stft.py`](../../../../course/solutions/python/tinyllm/sig/stft.py).

## Why this design

The cost is $T$ frames of an $O(n_\text{fft} \log n_\text{fft})$ real FFT, so $O((n/\text{hop})\, n_\text{fft} \log n_\text{fft})$. The frame matrix is $T \times n_\text{fft}$, $n_\text{fft}/\text{hop}$ times the signal's size (2.5 for Whisper); `torch.stft` avoids the copy with a strided view, which is a memory optimization, not a different result.

The solution matches `torch.stft` exactly because Whisper's weights were trained on features from that call: periodic Hann, reflect padding, center framing, power. Any other choice of the four produces a plausible spectrogram that the model has never seen.

The inverse divides by $\sum w^2$ instead of relying on COLA. Dividing by a constant 1.5 would work only for this window and hop. Dividing by the accumulated sum inverts any window and hop whose sum is nonzero, and the COLA case is just the one where the sum happens to be flat.

Short signals are rejected rather than repaired. A clip shorter than half a frame, a zero hop, or a wrong window length is a misconfigured frontend; returning something would push the error into the model's output.

## What can go wrong

A correct implementation reproduces the hand-worked windows and spectrogram, which pins down padding, windowing, and orientation. Its frame counts must include Whisper's 3001 and the centered and uncentered cases, and reject bad arguments. `hann` must match `torch.hann_window` for $n = 1, 2, 8, 400$ with both flags. `stft` must match torch within $10^{-5}$ on a Whisper-sized case, a small case, a zero-padded symmetric case, and a short-hop case. The round trip through `istft` at hop $n_\text{fft}/4$ must have error below $10^{-6}$, so nothing is lost in the frames. A 5 kHz tone sampled at 8 kHz must peak at the 3 kHz bin, 96, since aliasing is a sampling fact. `power` must equal $\lvert S \rvert^2$, and a bin-centered tone under the periodic Hann window must fill bins 19 to 21 in power ratio $1 : 4 : 1$. Misconfigured frames must be rejected with the input untouched.

The common mistakes:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| the symmetric Hann window | leakage into every bin; features drift from what Whisper was trained on | `hann` defaults to `periodic=True`, period $n$ |
| zero padding instead of reflect | a click in the first and last frames | `pad_mode` defaults to `"reflect"`, `np.pad(..., mode="reflect")` |
| the centered frame count without the $+1$ | 3000 frames before Whisper drops one: 2999 | `1 + (n + 2*(n_fft//2) - n_fft) // hop` (step 2) |
| forgetting to window the frames | leakage everywhere; the round trip breaks | multiplies the frame matrix by `w` before `rfft`, and again in `istft` |
| normalizing overlap-add by $\sum w$, not $\sum w^2$ | the reconstruction scaled by about 0.75 | accumulates `w * w` into `wsum` (step 6) |
| returning frames by bins | every consumer transposes or breaks | returns `rfft(frames).T`, `[n_fft//2 + 1, T]` |
| power as the magnitude $\lvert S \rvert$ | log-mel features halved | `power` returns $\operatorname{Re}^2 + \operatorname{Im}^2$ |
| reflect-padding a signal of exactly $n_\text{fft}/2$ samples | numpy repeats the reflection where torch raises | raises when `a.size <= half` |
| accepting hop 0 | a division by zero instead of a clear error | `stft` and `frame_count` require `hop >= 1` |
| accepting a signal one sample short of a frame | zero frames instead of an error | raises when fewer than `n_fft` samples remain |

## Where this leads

[Mel scale and filterbanks](06-mel-scale-and-filterbanks.md) pools this power spectrogram into mel bands. In *Build Your Own LLM System*, the [audio frontend](../../build-an-llm/p14-speech/01-audio-frontend.md) computes Whisper's features as `stft(x, 400, 160, hann(400))`, drops the last frame, and applies `power`. The chapter's [problem set](90-problem-set.md) works Nyquist, aliasing, windows, COLA, and frame counts by hand.

Each piece has a production counterpart:

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `stft` | `torch.stft` | batched frames as a strided view, cuFFT on GPU, onesided and normalized flags | `aten/src/ATen/native/SpectralOps.cpp` |
| `istft` | `librosa.istft` | window-sum caching, Griffin-Lim phase recovery on top | `librosa/core/spectrum.py` |
| `hann` | `scipy.signal.get_window` | dozens of windows; `sym=False` for the periodic ones | `scipy/signal/windows/_windows.py` |
| Whisper's frontend in C | an optional C kernel | STFT power with center reflect padding and drop-last in C | `c/src/kernels/audio.c` |

### Further reading

- Oppenheim and Schafer, *Discrete-Time Signal Processing*, ch. 4 and 10.
- Harris, "On the use of windows for harmonic analysis with the discrete Fourier transform" (Proc. IEEE 1978).
- Smith, *Spectral Audio Signal Processing* (free), the overlap-add chapters.

## Key terms

- **Nyquist frequency**: $\text{sr}/2$, the highest frequency a sample rate can represent.
- **Aliasing**: a tone above the Nyquist frequency producing the same samples as a tone at $|f - k\,\text{sr}|$.
- **Spectral leakage**: energy from a tone spreading into many DFT bins because the frame does not hold a whole number of periods.
- **Periodic Hann window**: $\tfrac12 - \tfrac12\cos(2\pi j/n)$, one period of a cosine of the DFT length.
- **STFT**: the DFT of overlapping windowed frames, giving a spectrum per time step.
- **COLA / NOLA**: constant (or nonzero) overlap-add of the squared window, the conditions for exact (or possible) inversion.

## Summary

- Sampling at rate $\text{sr}$ can only hold frequencies below $\text{sr}/2$; a tone above folds to $|f - k\,\text{sr}|$.
- A window tapers each frame so its edges do not smear energy across every bin; the periodic Hann window is three cosines, so a tone on a bin lands in exactly three bins.
- `center=True` pads $n_\text{fft}/2$ reflected samples on each side, so there are $1 + \lfloor n / \text{hop} \rfloor$ frames: 3001 for 30 s of Whisper audio, which then drops the last.
- When the squared window overlap-adds to a constant (periodic Hann at hop $n_\text{fft}/4$), weighted overlap-add inverts the STFT exactly.
- Whisper's features are this STFT's power, computed without a square root.

## Review questions

1. Why does Whisper's frontend use the periodic Hann window and not the symmetric one?
   <details><summary>Answer</summary>
   The periodic window is exactly one period of a cosine of the DFT length, so its spectrum is three bins and a bin-centered tone stays in three bins ($1 : 4 : 1$ in power). The symmetric window's period is $n - 1$, which does not fit the DFT grid, so it leaks a little into every bin. And Whisper was trained with `torch.hann_window`, which defaults to periodic.
   </details>
2. Why is the centered frame count $1 + \lfloor n/\text{hop} \rfloor$, independent of $n_\text{fft}$?
   <details><summary>Answer</summary>
   Padding adds $2 \cdot n_\text{fft}/2 = n_\text{fft}$ samples for even $n_\text{fft}$, which is exactly what one frame consumes, so $(n + n_\text{fft} - n_\text{fft})/\text{hop}$ remains. Frame $t$ is centered on original sample $t\,\text{hop}$, for every $t$ from 0 to $\lfloor n/\text{hop} \rfloor$.
   </details>
3. In `istft`, why is the normalizer $\sum_t w^2$ and not $\sum_t w$?
   <details><summary>Answer</summary>
   Every sample is multiplied by the window twice: once in `stft` (analysis) and once here (synthesis). Overlap-add therefore produces $x \sum_t w^2$. Dividing by $\sum_t w$ would leave a factor $\sum w^2/\sum w$, which is $1.5/2 = 0.75$ for periodic Hann at hop $n_\text{fft}/4$.
   </details>
4. Why reflect padding rather than zeros?
   <details><summary>Answer</summary>
   Zeros create a jump from the signal's first value to 0, and the first and last frames see that jump as a click that spreads energy across all bins. Reflecting continues the signal smoothly through the boundary (and does not repeat the edge sample), so the edge frames look like the interior ones.
   </details>
5. A 5 kHz tone is sampled at 8 kHz and analyzed with $n_\text{fft} = 256$. Which bin peaks, and why can no processing undo it?
   <details><summary>Answer</summary>
   The samples are identical to a $8 - 5 = 3$ kHz tone's, and bin $f$ is at $f \cdot 8000/256 = 31.25 f$ Hz, so the peak is bin $3000/31.25 = 96$. Once sampled, the two tones produce the same numbers, so nothing downstream can tell them apart; the only fix is to remove content above 4 kHz before sampling.
   </details>
