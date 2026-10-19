# Mel scale and filterbanks

The STFT of [Sampling, Nyquist, windows, and the STFT](05-sampling-windows-and-stft.md) turns 25 ms of audio into 201 power values per frame, evenly spaced from 0 to 8 kHz. Whisper's encoder does not take those 201 numbers. It takes 80 (or 128 for large-v3) mel bands, each a weighted sum of bins, then a clipped logarithm. The weights are librosa's mel filters, shipped inside Whisper as `mel_filters.npz`. If your filters differ in the fifth decimal, every feature the audio frontend hands the encoder is wrong by a little, and parity with HF's `WhisperFeatureExtractor` (within $10^{-4}$) fails. An audio near-duplicate detector that hashes the same log-mel picture has the same requirement.

This section derives the mel scale, the triangular filters and their normalization, and the two logarithmic conventions in use (decibels and Whisper's log-mel), works each by hand, and then builds the filters and logs exactly as the reference libraries do. It uses `log_base` from [Exponents, logarithms, and units of information](../01-precalculus/01-exponents-logs-and-units-of-information.md), and the filters it builds pool the power spectrogram from the previous section.

The reference implementation is [`course/solutions/python/tinyllm/sig/mel.py`](../../../../course/solutions/python/tinyllm/sig/mel.py), with its interface in [`course/contracts/py/tinyllm/sig/mel.pyi`](../../../../course/contracts/py/tinyllm/sig/mel.pyi). It provides `hz_to_mel` and `mel_to_hz` (Slaney and HTK), `mel_filterbank` (librosa's triangles with Slaney area normalization), `power_to_db`, and `log_mel_whisper` (log10, floor, 8-unit range, rescale).

> **Learning objectives**
> - Convert between Hz and mel on the Slaney and HTK scales.
> - Build a mel filterbank of $n_\text{mels}$ triangles on mel-spaced edges, sampled at rfft bin centers, with Slaney area normalization.
> - Convert power to decibels with a floor, a reference, and a `top_db` clip relative to the maximum.
> - Compute Whisper's log-mel features and explain each of its constants.

## Notation

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

## The mel scale

Listeners judge equal pitch steps as roughly equal frequency steps below about 1 kHz and equal frequency ratios above. The Slaney scale (librosa's default, used by Whisper) models this piecewise:
$$m(f) = \begin{cases} f / (200/3) & f < 1000 \\ 15 + \dfrac{\ln(f / 1000)}{\ln(6.4)/27} & f \ge 1000 \end{cases}$$
so 1000 Hz is 15 mel and every factor of 6.4 in frequency adds 27 mel. The HTK scale is one smooth formula, $m(f) = 2595 \log_{10}(1 + f/700)$. The two give different numbers and different filters; Whisper uses Slaney.

## Triangular filters and their normalization

A scale tells you where the bands go; the filters say how each band collects power. Place $n_\text{mels} + 2$ points equally spaced in mel between $m(f_\text{min})$ and $m(f_\text{max})$ (default $\text{sr}/2$) and map them back to Hz: $e_0 < e_1 < \dots < e_{n_\text{mels}+1}$. Filter $i$ rises linearly from 0 at $e_i$ to 1 at $e_{i+1}$ and falls back to 0 at $e_{i+2}$:
$$F_i(f) = \max\left(0,\; \min\left(\frac{f - e_i}{e_{i+1} - e_i},\; \frac{e_{i+2} - f}{e_{i+2} - e_{i+1}}\right)\right),$$
sampled at the bin centers $f_k$.

Neighbouring triangles overlap by half, and high triangles are much wider in Hz than low ones. With Slaney normalization each row is multiplied by $2/(e_{i+2} - e_i)$; a triangle of height 1 and base $e_{i+2} - e_i$ has area $(e_{i+2} - e_i)/2$, so every normalized filter has unit area and measures power density rather than total power. The filterbank is the matrix $M \in \mathbb{R}^{n_\text{mels} \times (n_\text{fft}/2+1)}$, and the mel spectrogram is $M P$.

## Decibels

Loudness is perceived on a log scale too. A power ratio in decibels is $10 \log_{10}(p / \text{ref})$: a factor of 10 in power is 10 dB. (Amplitudes use $20 \log_{10}$, because power is amplitude squared; using 20 on a power doubles every number.) `power_to_db` floors both $p$ and ref at `amin` so $\log 0$ never happens, and with `top_db` raises everything to at least the maximum minus `top_db`, so silence does not stretch the range to $-\infty$. The base-10 logarithm is `log_base(x, 10)`: a change of base, $\ln x / \ln 10$.

## Whisper's log-mel

Whisper's features skip decibels and use $\log_{10}$ directly:
$$L = \log_{10}(\max(M P, 10^{-10})), \qquad L \leftarrow \max(L, \max L - 8), \qquad \text{features} = (L + 4)/4 .$$
The floor $10^{-10}$ is $-100$ dB; the clip keeps 8 units, 80 dB, below the loudest bin of the clip; the affine map puts typical speech near $[-1, 1]$. Silence floors everywhere at $\log_{10} 10^{-10} = -10$, which maps to $-1.5$. The maximum is taken over the whole clip, so the features of a frame depend on the loudest moment anywhere in the 30 s window.

## Working it by hand

Start with mel values on the Slaney scale: $500$ Hz is $500 \cdot 3/200 = 7.5$ mel; $1000$ Hz is 15 mel; $6400$ Hz is $15 + 27 \ln(6.4)/\ln(6.4) = 42$ mel. On the HTK scale $700$ Hz is $2595 \log_{10} 2 = 781.17$ mel.

Next, decibels of the powers $(1, 10, 10^{-12})$ with ref 1 and $\text{amin} = 10^{-10}$: $10 \log_{10} 1 = 0$, $10 \log_{10} 10 = 10$, and $10^{-12}$ is floored to $10^{-10}$, giving $-100$. With $\text{top\_db} = 80$ the floor is $10 - 80 = -70$, so the result is $(0, 10, -70)$. With ref 10 everything drops by 10 dB: $(-10, 0, -110)$ when not clipped.

Finally, Whisper's scaling of the identity filterbank applied to $P = \begin{pmatrix} 10^{-12} & 100 \\ 1 & 10^4 \end{pmatrix}$: $\log_{10}$ with the floor gives $\begin{pmatrix} -10 & 2 \\ 0 & 4 \end{pmatrix}$; clipping at $4 - 8 = -4$ gives $\begin{pmatrix} -4 & 2 \\ 0 & 4 \end{pmatrix}$; and $(L + 4)/4 = \begin{pmatrix} 0 & 1.5 \\ 1 & 2 \end{pmatrix}$.

## Designing the interface

```python
def hz_to_mel(f, scale='slaney') -> NDArray: ...
def mel_to_hz(m, scale='slaney') -> NDArray: ...
def mel_filterbank(sr, n_fft, n_mels, fmin=0.0, fmax=None, scale='slaney', norm='slaney') -> NDArray: ...
def power_to_db(p, ref=1.0, amin=1e-10, top_db=80.0) -> NDArray: ...
def log_mel_whisper(power_spec, filters) -> NDArray: ...
```

Defaults are librosa's and Whisper's: the Slaney scale, Slaney normalization, $f_\text{max} = \text{sr}/2$, `amin = 1e-10`, `top_db = 80`. `log_mel_whisper` takes the filterbank as an argument, so the same function works with the 80-band and 128-band banks. Bad arguments (`amin` of 0, a negative `top_db`, complex input, an empty band, an unknown norm or scale, mismatched shapes) raise, and the input is never modified.

## Building it step by step

Eight steps take you from the formulas to code that agrees with librosa to $10^{-12}$.

1. **Two scales, one interface.** The scale is a choice of function $m(f)$ and its inverse. Everything else (edges, triangles, normalization) is written once in terms of `hz_to_mel` and `mel_to_hz`, so switching between Slaney and HTK changes only those two functions.
2. **The Slaney scale is piecewise, so compute both pieces and select.** Below 1000 Hz, $m = f/(200/3)$; above, $m = 15 + \ln(f/1000)/(\ln 6.4/27)$. The two pieces meet at exactly 15 mel, so the scale is continuous. Vectorised code evaluates both on the whole array and picks with `np.where`; the log piece is meaningless (and warns) at $f = 0$, so its warnings are silenced: those values are discarded anyway.
3. **Edges equally spaced in mel, then mapped back to Hz.** $n_\text{mels}$ triangles need $n_\text{mels} + 2$ edges (each triangle uses three consecutive edges, and neighbours share two). `np.linspace` in mel space followed by `mel_to_hz` gives them.
4. **Sample the triangles at the rfft bin centers.** The filterbank is applied to an rfft power spectrum, so the columns are bins $k = 0, \dots, n_\text{fft}/2$ at $f_k = k\,\text{sr}/n_\text{fft}$. Each row is $\max(0, \min(\text{rising}, \text{falling}))$: the rising ramp $(f - e_i)/(e_{i+1} - e_i)$ is below 1 left of the peak, the falling ramp right of it, and the minimum of the two is the triangle.
5. **Normalize to unit area.** A triangle of height 1 and base $e_{i+2} - e_i$ has area $(e_{i+2} - e_i)/2$, so multiplying row $i$ by $2/(e_{i+2} - e_i)$ gives area 1. Without it, a 1000 Hz wide band at the top sums ten times as many bins as a 100 Hz band at the bottom.
6. **Decibels: floor, log, subtract the reference, clip relative to the maximum.** $10\log_{10}$ because the input is power. The floor `amin` goes on before the log; `ref` is floored the same way and subtracted in the log domain; `top_db` clips against the maximum of the result, not against 0 dB, so a quiet recording keeps its own 80 dB of range.
7. **Whisper's log-mel is the same shape with its own constants.** Pool with $MP$, then $\log_{10}$ with a $10^{-10}$ floor, clip to the maximum minus 8, then $(L + 4)/4$. The clip uses the maximum over the whole input array, which is the whole 30 s clip in Whisper's frontend.
8. **Validate before computing.** Every bad argument the contract names (non-positive sizes, an empty band, unknown `norm` or `scale`, `amin <= 0`, negative `top_db`, complex input, mismatched shapes) raises `ValueError` up front, because each would otherwise produce NaNs or $-\infty$ silently.

Steps 1 and 2 are the scale conversion:

```python
_F_SP = 200.0 / 3.0  # Slaney: 200/3 Hz per mel below the break
_MIN_LOG_HZ = 1000.0  # the break between the linear and the log part
_MIN_LOG_MEL = _MIN_LOG_HZ / _F_SP  # = 15 mel
_LOGSTEP = math.log(6.4) / 27.0  # 27 mel per factor 6.4 above the break


def hz_to_mel(f: NDArray, scale: Literal["slaney", "htk"] = "slaney") -> NDArray:
    f = np.asarray(f, dtype=np.float64)
    if scale == "htk":
        return 2595.0 * np.log10(1.0 + f / 700.0)  # one smooth formula
    if scale != "slaney":
        raise ValueError(f"hz_to_mel: scale is 'slaney' or 'htk', got {scale!r}")
    lin = f / _F_SP
    # log(0) at f = 0 would warn; that element takes the linear branch anyway.
    with np.errstate(divide="ignore", invalid="ignore"):
        log = _MIN_LOG_MEL + np.log(f / _MIN_LOG_HZ) / _LOGSTEP
    return np.where(f >= _MIN_LOG_HZ, log, lin)  # both pieces give 15 at 1000 Hz
```

The constants are librosa's, named the way librosa names them, so that the derived edges agree to $10^{-12}$. `_MIN_LOG_MEL` is computed rather than written as `15.0` so that the two pieces meet exactly in floating point. `mel_to_hz` is the same function inverted piece by piece: $f = (200/3)\,m$ below 15 mel and $f = 1000\,e^{\text{\_LOGSTEP}\,(m - 15)}$ above, with overflow warnings silenced for the same reason. Trace the hand example: `hz_to_mel(6400)` is $15 + \ln 6.4 \cdot 27/\ln 6.4 = 42$.

Steps 3 to 5 build the filterbank:

```python
def mel_filterbank(sr, n_fft, n_mels, fmin=0.0, fmax=None, scale="slaney", norm="slaney"):  # annotations elided
    # ... validation: positive sizes, 0 <= fmin < fmax, norm in ("slaney", None)
    # Bin k sits at k sr / n_fft Hz (numpy's rfftfreq, written its way).
    fft_hz = np.arange(n_fft // 2 + 1) * (1.0 / (n_fft * (1.0 / sr)))
    # n_mels + 2 edges equally spaced in mel: band i rises from edge i to its
    # peak at edge i + 1 and falls to zero at edge i + 2.
    edges = mel_to_hz(
        np.linspace(hz_to_mel(fmin, scale), hz_to_mel(fmax, scale), n_mels + 2), scale
    )
    fdiff = np.diff(edges)                        # fdiff[i] = e_{i+1} - e_i
    ramps = edges[:, None] - fft_hz[None, :]      # ramps[j, k] = e_j - f_k
    w = np.empty((n_mels, fft_hz.size))
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]              # (f_k - e_i) / (e_{i+1} - e_i): rising side
        upper = ramps[i + 2] / fdiff[i + 1]       # (e_{i+2} - f_k) / (e_{i+2} - e_{i+1}): falling side
        w[i] = np.maximum(0.0, np.minimum(lower, upper))  # the triangle, zero outside
    if norm == "slaney":
        # Divide each triangle by half its width in Hz: unit area, so a wide
        # high band does not collect more energy just for being wide.
        w *= (2.0 / (edges[2 : n_mels + 2] - edges[:n_mels]))[:, None]
    return w
```

Two details are there for bit-level parity, not for the math. The bin frequencies are written as `k * (1 / (n_fft * (1 / sr)))`, the exact expression `np.fft.rfftfreq` evaluates, instead of the algebraically equal `k * sr / n_fft`; the two round differently in the last bit, and the golden check against librosa is at $10^{-12}$. And the triangle is built from the `ramps` matrix and `fdiff` the way librosa builds it, so the same subtractions happen in the same order. The validation `0 <= fmin < fmax` matters: with `fmin == fmax` every edge is equal, `fdiff` is 0, and every row is NaN. For Whisper the call is `mel_filterbank(16000, 400, 80)`: 201 columns, 82 edges from 0 to 8000 Hz.

Step 6 is decibels:

```python
def power_to_db(p, ref=1.0, amin=1e-10, top_db=80.0):  # annotations elided
    s = np.asarray(p)
    if np.iscomplexobj(s):                        # a complex STFT, not a power: refuse
        raise ValueError("power_to_db: take power(S) of a complex spectrum first")
    # ... amin > 0 and top_db >= 0 checked here
    s = s.astype(np.float64)                      # a copy: the input is untouched
    # Floor at amin before the log: 10 log10(0) would be -inf.
    db = 10.0 * log_base(np.maximum(amin, s), 10.0)
    db = db - 10.0 * log_base(np.maximum(amin, abs(ref)), 10.0)  # 10 log10(p / ref)
    if top_db is not None and db.size:
        db = np.maximum(db, db.max() - top_db)    # clip against the maximum, not 0 dB
    return db
```

Trace the hand example: $(1, 10, 10^{-12})$ floors to $(1, 10, 10^{-10})$, gives $(0, 10, -100)$ dB, and the clip at $\max - 80 = -70$ gives $(0, 10, -70)$. The `db.size` guard keeps `max()` from raising on an empty array.

Step 7 is Whisper's log-mel:

```python
def log_mel_whisper(power_spec: NDArray, filters: NDArray) -> NDArray:
    spec = np.asarray(power_spec, dtype=np.float64)
    fb = np.asarray(filters, dtype=np.float64)    # Whisper ships float32 filters; compute in float64
    if spec.ndim != 2 or fb.ndim != 2 or fb.shape[1] != spec.shape[0]:
        raise ValueError(...)                     # filters [M, F] and power [F, T]
    mel = fb @ spec                               # [M, T]: each band a weighted sum of bins
    log_spec = log_base(np.maximum(mel, 1e-10), 10.0)    # log10, floor at -100 dB
    # Keep 8 units (80 dB) below the loudest value, then map to about [-1, 1].
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    return (log_spec + 4.0) / 4.0
```

On the hand-worked matrix with an identity filterbank: $\log_{10}$ gives $\begin{pmatrix} -10 & 2 \\ 0 & 4 \end{pmatrix}$, the clip at $4 - 8 = -4$ lifts the $-10$, and $(L + 4)/4$ gives $\begin{pmatrix} 0 & 1.5 \\ 1 & 2 \end{pmatrix}$. After the clip the range is at most 8 units, so after dividing by 4 the features span at most 2 units.

The full solution is [`mel.py`](../../../../course/solutions/python/tinyllm/sig/mel.py).

## Why this design

Building the filterbank is $O(n_\text{mels} \cdot n_\text{fft})$ once (80 by 201 for Whisper); it is a constant matrix, computed once and reused. Applying it is one matrix product, $O(n_\text{mels} \cdot n_\text{fft}/2 \cdot T)$, the dominant cost of the frontend after the FFT. The logs and the clip are $O(n_\text{mels} T)$.

The filterbank is a dense matrix, not a sparse one. Each row has only a handful of nonzeros, so a sparse matrix would save multiplies. At 80 by 201 the dense product is already fast, matches what Whisper and HF do, and keeps the code a plain `@`.

The filters are recomputed rather than loaded from `mel_filters.npz`. Loading the file Whisper ships would be simpler, but the point is to know where the numbers come from, and the computed float64 bank lets the golden check confirm them to $10^{-7}$ against the shipped float32 values. The Rust serving engine later commits the filter file, proven against these numbers.

Floors are used instead of adding an epsilon. $\log(p + \epsilon)$ would shift every value slightly; $\log(\max(p, \text{amin}))$ changes only values below the floor, which is what librosa and Whisper do, so parity requires it.

One tradeoff is accepted deliberately: the clip against the clip-wide maximum means a frame's features depend on the loudest moment anywhere in the input. That is Whisper's convention and the encoder was trained on it, so the solution keeps it rather than clipping per frame.

## What can go wrong

A correct implementation reproduces the hand-worked mel values on both scales and the hand-worked decibels with and without `top_db` and with a reference. Its Whisper filterbanks (80 and 128 bands at 16 kHz, $n_\text{fft} = 400$) must agree with float64 librosa to $10^{-12}$ and with the float32 filters Whisper ships to $10^{-7}$. An HTK bank without normalization, a band-limited bank, both scale conversions, and the default `fmax` must match librosa too, as must `power_to_db` with defaults and with a reference, an `amin`, and no `top_db`. Every row must be the triangle formula on the declared mel-spaced edges and zero outside, and on a fine grid each Slaney-normalized row must integrate to 1. Whisper's scaling must reproduce the hand-worked matrix, give exactly $-1.5$ for silence, and span at most 2 units. Bad arguments must raise with the input untouched.

The mistakes these checks catch:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| the Slaney scale linear everywhere | high bands far too narrow | `hz_to_mel` selects the log piece for $f \ge 1000$ Hz (step 2) |
| skipping the Slaney normalization | high bands dominate every feature | rows scaled by $2/(e_{i+2} - e_i)$ when `norm="slaney"` (step 5) |
| edges equally spaced in Hz | a linear filterbank, not a mel one | `np.linspace` in mel, then `mel_to_hz` (step 3) |
| $20 \log_{10}$ on a power | every dB value doubled | `10.0 * log_base(..., 10.0)` (step 6) |
| clipping `top_db` against 0 dB instead of the maximum | quiet recordings clipped flat | `np.maximum(db, db.max() - top_db)` (step 6) |
| the natural log in Whisper's features | features scaled by $\ln 10$ | `log_base(..., 10.0)` in `log_mel_whisper` (step 7) |
| Whisper's range as 80 instead of 8 | the clip never happens | `log_spec.max() - 8.0`: 8 units of $\log_{10}$ are 80 dB (step 7) |
| accepting amin = 0 | $\log 0 = -\infty$ in the output | `amin <= 0` raises (step 8) |
| accepting fmin = fmax | a filterbank of NaNs | `0 <= fmin < fmax` is required (step 8) |
| accepting a negative top_db | every value clipped above the maximum | `top_db < 0` raises (step 8) |

## Where this leads

This section completes the audio front end. In *Build Your Own LLM System*, the [audio frontend](../../build-an-llm/15-speech/01-audio-frontend.md) computes Whisper's features as `log_mel_whisper(power(stft(...))[:, :-1], mel_filterbank(16000, 400, 80))`. The audio deduplication stage of the [corpus pipeline](../../data-engineering/05-corpus-pipeline/README.md) in *Data Engineering* hashes mel dHash windows to find near-duplicate audio. The chapter's [problem set](07-problem-set.md) works mel, logs, decibels, and Whisper's scaling by hand.

Each piece has a production counterpart:

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `mel_filterbank` | `librosa.filters.mel` | the same filters, cached; `norm` as any $L^p$ norm | `librosa/filters.py` |
| `log_mel_whisper` | HF `WhisperFeatureExtractor` | the full frontend in numpy or torch, batched, with padding to 30 s | `transformers/models/whisper/feature_extraction_whisper.py` |
| `power_to_db` | `torchaudio.transforms.AmplitudeToDB` | the same floor and `top_db` on tensors | `torchaudio/functional/functional.py` |
| filters in Rust | the serving engine's `tl-engine/src/mm/audio.rs` | log-mel from the committed filter file, proven against these librosa-matched values | `tl-engine/src/mm/audio.rs` |

### Further reading

- Stevens, Volkmann, Newman, "A scale for the measurement of the psychological magnitude pitch" (1937).
- Slaney, *Auditory Toolbox* (1998), `MakeERBFilters` and `mfcc`.
- Radford et al., "Robust speech recognition via large-scale weak supervision" (2022), section 2.2.

## Key terms

- **Mel scale**: a perceptual pitch scale; Slaney's is linear below 1000 Hz and logarithmic above, HTK's is $2595 \log_{10}(1 + f/700)$.
- **Mel filterbank**: a matrix of triangular filters on mel-spaced edges that pools an rfft power spectrum into bands.
- **Slaney normalization**: scaling each triangle by $2/(e_{i+2} - e_i)$ so it has unit area in Hz.
- **Decibel**: $10\log_{10}$ of a power ratio (or $20\log_{10}$ of an amplitude ratio).
- **top_db**: the dynamic range kept below the maximum; quieter values are raised to the maximum minus `top_db`.
- **Log-mel**: Whisper's features, $\log_{10}$ of the mel spectrogram with a floor, an 8-unit clip, and the map $(L + 4)/4$.

## Summary

- The Slaney mel scale is linear below 1000 Hz (15 mel, $200/3$ Hz per mel) and logarithmic above (27 mel per factor 6.4); HTK's is $2595 \log_{10}(1 + f/700)$.
- A mel filterbank is $n_\text{mels}$ triangles with edges equally spaced in mel, sampled at the rfft bin frequencies.
- Slaney normalization gives every triangle unit area in Hz, so wide high bands do not collect more energy for being wide.
- Power becomes decibels as $10 \log_{10}$ with a floor; Whisper uses $\log_{10}$, clips 8 units (80 dB) below the maximum, and maps by $(x + 4)/4$.
- Bit-level parity with librosa comes from evaluating the same expressions in the same order, not only the same math.

## Review questions

1. Why are there $n_\text{mels} + 2$ edges and not $n_\text{mels}$ or $n_\text{mels} + 1$?
   <details><summary>Answer</summary>
   Triangle $i$ needs three edges: start $e_i$, peak $e_{i+1}$, end $e_{i+2}$. Consecutive triangles share two of them (the peak of one is the start of the next), so $n_\text{mels}$ triangles need $n_\text{mels} + 2$ points: the first triangle starts at $f_\text{min}$ and the last ends at $f_\text{max}$.
   </details>
2. Why is the Slaney normalization factor $2/(e_{i+2} - e_i)$ and not $1/(e_{i+2} - e_i)$?
   <details><summary>Answer</summary>
   The goal is unit area. A triangle of height 1 over a base of $e_{i+2} - e_i$ Hz has area half the base times the height, $(e_{i+2} - e_i)/2$. Dividing by that area means multiplying by $2/(e_{i+2} - e_i)$. Dividing by the full base would give every filter area $1/2$, which is a constant factor off librosa and Whisper.
   </details>
3. `power_to_db` clips at `db.max() - top_db`. What goes wrong if it clips at `-top_db` instead?
   <details><summary>Answer</summary>
   Clipping at $-\text{top\_db}$ assumes the loudest value is 0 dB. A quiet recording whose maximum is, say, $-60$ dB would keep only 20 dB of range below its own maximum, and everything quieter would be flattened to $-80$. Clipping relative to the maximum keeps the same 80 dB of dynamic range whatever the overall level.
   </details>
4. Why does a silent input give Whisper features of exactly $-1.5$?
   <details><summary>Answer</summary>
   Silence means $MP = 0$ everywhere, so the floor gives $\log_{10} 10^{-10} = -10$ in every cell. The maximum is also $-10$, so the clip at $-18$ changes nothing, and $(-10 + 4)/4 = -1.5$.
   </details>
5. Why does the code write the bin frequencies as `k * (1 / (n_fft * (1 / sr)))` instead of `k * sr / n_fft`?
   <details><summary>Answer</summary>
   They are equal in exact arithmetic but round differently in float64. That is the expression `np.fft.rfftfreq` evaluates, which librosa uses, so the solution reproduces librosa's bin frequencies bit for bit, and the filterbank agrees with librosa to $10^{-12}$ instead of only to the last few bits.
   </details>
