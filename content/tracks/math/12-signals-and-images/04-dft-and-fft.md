<!-- ol:module M12.4 -->
# DFT and FFT

## Overview

| | |
|---|---|
| **Module** | `M12.4` · build · Python · Pass 12 · 4 h |
| **You build** | `python/tinyllm/sig/fft.py`: `dft_matrix`, a mixed-radix `fft` (radices 4, 2, 3, 5) and `ifft`, `rfft` and `irfft` from one half-size complex FFT, `fft_convolve` (the convolution theorem), and `dct2_ortho` (the DCT-II via an FFT) |
| **Contract** | [`course/contracts/py/tinyllm/sig/fft.pyi`](../../../../course/contracts/py/tinyllm/sig/fft.pyi) |
| **Tests** | `course/tests/M12.4/test_fft.py` (what they check: section 4), golden values from `course/oracle/M12.4/fft_numpy.py` (numpy.fft, scipy.fft.dct) in `course/fixtures/M12.4/fft_numpy.npz` |
| **Needs** | no module code. Reading: `M00.2` (Euler's formula, roots of unity), `M03.3` (orthogonal and unitary matrices), `M09.3` (error bounds that grow with $\log n$) |
| **Used by** | `M12.5` (`stft` calls `rfft`, `istft` calls `irfft`) · later `data.10` (`dct2_ortho` in pHash, with B14's data group) |
| **Milestone** | `MS-P12` (the multimodal gate) |
| **Optional depth** | Cooley and Tukey, "An algorithm for the machine calculation of complex Fourier series" (1965); Van Loan, *Computational Frameworks for the Fast Fourier Transform*, ch. 1 to 2; Makhoul, "A fast cosine transform in one and two dimensions" (1980) |

## Key Takeaways

- The DFT is multiplication by $W_{kj} = e^{-2\pi i kj/n}$; $W/\sqrt{n}$ is unitary, so the transform keeps energy (Parseval) and its inverse is the conjugate transform divided by $n$ (`test_dft_matrix_is_unitary`, `test_inverse_and_parseval`).
- An FFT splits $x$ into $p$ interleaved subsequences, transforms them, and recombines them with twiddle factors and one size-$p$ DFT; mixed radices cover $n = 400 = 4 \cdot 4 \cdot 5 \cdot 5$ (`test_fft_matches_numpy`, `test_fft_equals_the_dft_matrix`).
- A real signal's spectrum is conjugate symmetric, so `rfft` computes $n/2 + 1$ bins from one complex FFT of size $n/2$ (`test_hand_example_dft`, `test_rfft_and_irfft_match_numpy`).
- The convolution theorem gives circular convolution; zero-padding to $\ge$ `len(a) + len(b) - 1` makes it linear (`test_fft_convolve_is_linear_convolution`).
- The orthonormal DCT-II is a reordered FFT (`test_dct_matches_scipy_and_is_orthogonal`).

## How to work this chapter

```bash
ol start M12.4              # stubs python/tinyllm/sig/fft.py into your repo
ol tests M12.4              # read the test catalog first: rung R0, you write no tests here
ol check M12.4              # exit code is the verdict
ol diff  M12.4              # after passing: your code against the reference
```

---

## 1. Why now

Whisper's frontend (`L14.1`) takes a Fourier transform of every 25 ms frame of audio: 3000 frames of 400 samples for each 30 s clip. Multiplying each frame by a $400 \times 400$ DFT matrix costs 160000 complex multiplies per frame; an FFT does it in about 4000. More importantly, 400 is not a power of two, so the radix-2 FFT of most textbooks cannot transform it at all, and padding to 512 changes the frequencies of the bins and breaks parity with Whisper. This module builds the transform the STFT (`M12.5`) and the perceptual hash in `data.10` both stand on, and checks it against numpy to within a frozen rounding bound.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $n$ | transform length | int |
| $\omega_n = e^{-2\pi i/n}$ | primitive $n$-th root of unity | complex |
| $W$, $W_{kj} = \omega_n^{kj}$ | DFT matrix | `complex128[n, n]` |
| $X = Wx$ | spectrum of $x$ | `complex128[n]` |
| $p$, $m = n/p$ | radix and subsequence length | ints |
| $Y_r$ | DFT of the subsequence $x[r], x[r+p], \dots$ | `complex128[m]` |
| $h = n/2$ | half length for the real FFT | int |
| $\varepsilon$ | float64 machine epsilon, $2^{-52}$ | float |

### 2.1 The DFT and its matrix

The DFT writes $n$ samples as coefficients of $n$ complex sinusoids,
$$X[k] = \sum_{j=0}^{n-1} x[j]\, \omega_n^{jk}, \qquad x[j] = \frac{1}{n} \sum_{k=0}^{n-1} X[k]\, \omega_n^{-jk}.$$
The rows of $W$ are orthogonal: $\sum_j \omega_n^{jk} \overline{\omega_n^{jl}} = \sum_j \omega_n^{j(k-l)}$, a geometric series of roots of unity that is $n$ if $k = l$ and 0 otherwise. So $U = W/\sqrt{n}$ satisfies $U^H U = I$: it is unitary (`M03.3`), it keeps lengths, and that is Parseval's identity $\sum |x|^2 = \frac1n \sum |X|^2$. The inverse is the conjugate transform: $x = \overline{W \bar{X}}/n$, which is how `ifft` reuses `fft`. `dft_matrix` reduces $kj \bmod n$ before computing the phase, so every entry is evaluated at an angle in $[0, 2\pi)$.

### 2.2 The convolution theorem

For circular convolution $(a \circledast b)[j] = \sum_l a[l]\, b[(j - l) \bmod n]$, the DFT turns convolution into a product: $\operatorname{DFT}(a \circledast b) = \operatorname{DFT}(a) \cdot \operatorname{DFT}(b)$. Linear convolution, what `numpy.convolve` computes, has length $L = \text{len}(a) + \text{len}(b) - 1$; padding both inputs with zeros to any $m \ge L$ makes the wrap-around terms vanish, so the circular result equals the linear one. `fft_convolve` picks the smallest $m \ge L$ whose prime factors are 2, 3, 5.

### 2.3 Cooley-Tukey, any radix

Split $x$ into $p$ interleaved subsequences $x_r[t] = x[r + pt]$, $t < m = n/p$, with DFTs $Y_r$ of size $m$. Writing $k = qm + k'$ with $0 \le k' < m$, $0 \le q < p$:
$$X[qm + k'] = \sum_{r=0}^{p-1} \omega_p^{rq} \Big( \omega_n^{rk'}\, Y_r[k'] \Big).$$
So the size-$n$ DFT is $p$ size-$m$ DFTs, a multiply by the twiddle factors $\omega_n^{rk'}$, and a size-$p$ DFT for each $k'$. Recursing until $m = 1$ costs $O(n \sum p_i)$ operations, $O(n \log n)$ for small radices. With radices 4, 2, 3, 5 every $n = 2^a 3^b 5^c$ works: $400 = 4 \cdot 4 \cdot 5 \cdot 5$, Whisper's frame. A size with another prime factor (7, 11, ...) is a `ValueError`, not a silent $O(n^2)$ fallback. Each level adds a few roundings, so the error grows like $\log n$: the tests freeze the bound $8 \log_2(\max(n, 2))\, \varepsilon\, \|x\|_2$ per entry (`M09.3`).

### 2.4 Real input: half the work

If $x$ is real, $X[n - k] = \overline{X[k]}$, so the bins $0 \dots n/2$ determine everything. Pack the even samples as real parts and the odd ones as imaginary parts, $z[t] = x[2t] + i\, x[2t+1]$, and take one complex FFT $Z$ of size $h = n/2$. The DFTs of the even and odd samples untangle as
$$E[k] = \tfrac12\big(Z[k] + \overline{Z[(h-k) \bmod h]}\big), \qquad O[k] = -\tfrac{i}{2}\big(Z[k] - \overline{Z[(h-k) \bmod h]}\big),$$
and $X[k] = E[k] + \omega_n^k O[k]$ for $k < h$, with the Nyquist bin $X[h] = E[0] - O[0]$ because $\omega_n^h = -1$. `irfft` runs the steps backwards; the imaginary parts of $X[0]$ and $X[h]$ must be zero for a real signal, so it drops them first, as numpy does.

### 2.5 The DCT-II via an FFT

pHash (`data.10`) takes the orthonormal DCT-II, $y[k] = s_k \cdot 2 \sum_j x[j] \cos\frac{\pi k (2j+1)}{2n}$ with $s_0 = \sqrt{1/(4n)}$ and $s_k = \sqrt{1/(2n)}$. Makhoul's trick reorders the input into $v = (x_0, x_2, x_4, \dots, x_5, x_3, x_1)$ (even samples forward, odd samples backward); then $y[k] = 2 s_k \operatorname{Re}\big(e^{-i\pi k/(2n)} V[k]\big)$ with $V = \operatorname{FFT}(v)$. The scales make the DCT matrix orthogonal, which the test checks by transforming the identity.

## 3. Worked example by hand

$x = (1, 2, 0, -1)$, $n = 4$, $\omega_4 = -i$:

| $k$ | $\sum_j x[j] (-i)^{jk}$ | $X[k]$ |
|---|---|---|
| 0 | $1 + 2 + 0 - 1$ | $2$ |
| 1 | $1 + 2(-i) + 0 - 1 \cdot (i)$ | $1 - 3i$ |
| 2 | $1 - 2 + 0 - 1 \cdot (-1)$ | $0$ |
| 3 | $1 + 2(i) + 0 - 1 \cdot (-i)$ | $1 + 3i$ |

$X[3] = \overline{X[1]}$, so `rfft` returns $(2, 1 - 3i, 0)$. Parseval: $\sum |x|^2 = 1 + 4 + 0 + 1 = 6$ and $\frac14 (4 + 10 + 0 + 10) = 6$. Through the packing of 2.4: $z = (1 + 2i, 0 - i)$, $Z = (1 + i, 1 + 3i)$; then $E[0] = \frac12((1+i) + (1-i)) = 1$, $O[0] = -\frac i2 (2i) = 1$, so $X[0] = 1 + 1 = 2$ and $X[2] = 1 - 1 = 0$. This is `test_hand_example_dft`.

## 4. The interface

```python
def dft_matrix(n, unitary=False) -> NDArray: ...
def fft(x) -> NDArray: ...                  # last axis, n = 2^a 3^b 5^c
def ifft(X) -> NDArray: ...
def rfft(x) -> NDArray: ...                 # n // 2 + 1 bins, one FFT of size n/2
def irfft(X, n) -> NDArray: ...
def fft_convolve(a, b) -> NDArray: ...      # == numpy.convolve(a, b)
def dct2_ortho(x, axis=-1) -> NDArray: ...  # == scipy.fft.dct(x, 2, norm='ortho')
```

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_dft` | unit, smoke | section 3's spectrum, its rfft half, and Parseval | sign and packing conventions |
| `test_dft_matrix_is_unitary` | property | $U^H U = I$ for several $n$; one entry by formula; $n = 0$ rejected | the transform keeps energy |
| `test_fft_matches_numpy` | golden | 19 sizes up to 512, including 240 and 400, and a batch, within the frozen bound | `M12.5` and `L14.1` parity |
| `test_fft_equals_the_dft_matrix` | differential | mixed-radix sizes against $W x$ | the FFT is only fast multiplication |
| `test_rfft_and_irfft_match_numpy` | golden | rfft and irfft for every even size; irfft drops imaginary DC and Nyquist parts | every STFT frame |
| `test_inverse_and_parseval` | property | `ifft(fft(x)) = x`, Parseval, `irfft(rfft(r)) = r` | `istft` and energy bookkeeping |
| `test_fft_convolve_is_linear_convolution` | differential | equals `numpy.convolve`, including lengths that need padding to 12 | the convolution theorem, done right |
| `test_dct_matches_scipy_and_is_orthogonal` | golden | scipy's orthonormal DCT-II on vectors and a 32 x 32 block; orthogonality | `data.10`'s pHash |
| `test_unsupported_sizes_raise` | boundary | prime factors 7 and 11, empty input, odd rfft, complex rfft, wrong bin count; inputs untouched | misconfigured frames fail loudly |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. twiddle factors with $+2\pi i$ | every size above 4 wrong; spectra mirrored | `test_fft_matches_numpy`, `test_fft_equals_the_dft_matrix` (mutant `s01`) |
| 2. Nyquist bin as $E[0] + O[0]$ | the last rfft bin wrong | `test_hand_example_dft`, `test_rfft_and_irfft_match_numpy` (mutant `s02`) |
| 3. ifft without the $1/n$ | round trips scaled by $n$ | `test_inverse_and_parseval` (mutant `s03`) |
| 4. multiplying unpadded spectra | circular wrap-around; wrong length | `test_fft_convolve_is_linear_convolution` (mutant `s04`) |
| 5. scaling bin 0 like the others in the DCT | the DC coefficient $\sqrt2$ too large; not orthogonal | `test_dct_matches_scipy_and_is_orthogonal` (mutant `s05`) |
| 6. keeping an imaginary DC part in irfft | a spurious alternating component | `test_rfft_and_irfft_match_numpy` (mutant `s06`) |
| 7. splitting into contiguous blocks instead of interleaved subsequences | a different transform that still has the right shape | `test_fft_matches_numpy`, `test_fft_equals_the_dft_matrix` (mutant `s07`) |
| accepting $n = 0$ in `dft_matrix` | an empty matrix instead of an error | `test_dft_matrix_is_unitary` (mutant `m01`) |
| accepting extra bins in irfft | the trailing bins silently ignored | `test_unsupported_sizes_raise` (mutant `m02`) |
| accepting complex input in rfft | the imaginary part discarded | `test_unsupported_sizes_raise` (mutant `m03`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M00.2` | roots of unity from Euler's formula (reading) |
| Back | `M03.3` | unitary matrices keep lengths (reading) |
| Back | `M09.3` | the $\log n$ error bound (reading) |
| Forward | `M12.5` | `stft` runs `rfft` on every frame; `istft` runs `irfft` |
| Forward | `data.10` | pHash's 32 x 32 `dct2_ortho` (with B14's data group) |
| Forward | `S-M12` | DFT properties, the convolution theorem, FFT operation counts |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `fft` | pocketfft (numpy, scipy) | radices up to 11 plus Bluestein for large primes, cache-friendly plans | `numpy/fft/_pocketfft_umath.cpp` |
| `fft` | FFTW | plans tuned by measurement ("wisdom"), codelets generated by `genfft` | Frigo and Johnson, Proc. IEEE 2005 |
| `rfft` in Rust | `rustfft` and `realfft` (the engine, `L15.1`) | SIMD butterflies; the same half-size packing | `realfft` crate docs |
| `fft` in C | optional `L14.6` | mixed-radix real FFT plans for the Whisper frontend | `c/src/kernels/audio.c` |
