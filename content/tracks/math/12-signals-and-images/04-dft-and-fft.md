# DFT and FFT

## Overview

| | |
|---|---|
| **Module** | `M12.4` · build · Python · Pass 12 · 4 h |
| **Solution** | [`python/tinyllm/sig/fft.py`](../../../../course/solutions/python/tinyllm/sig/fft.py): `dft_matrix`, a mixed-radix `fft` (radices 4, 2, 3, 5) and `ifft`, `rfft` and `irfft` from one half-size complex FFT, `fft_convolve` (the convolution theorem), and `dct2_ortho` (the DCT-II via an FFT) |
| **Contract** | [`course/contracts/py/tinyllm/sig/fft.pyi`](../../../../course/contracts/py/tinyllm/sig/fft.pyi) |
| **Needs** | no module code. Reading: `M00.2` (Euler's formula, roots of unity), `M03.3` (orthogonal and unitary matrices), `M09.3` (error bounds that grow with $\log n$) |
| **Used by** | `M12.5` (`stft` calls `rfft`, `istft` calls `irfft`) · later `data.10` (`dct2_ortho` in pHash). Part of Pass 12, which teaches the system to see and hear |
| **Optional depth** | Cooley and Tukey, "An algorithm for the machine calculation of complex Fourier series" (1965); Van Loan, *Computational Frameworks for the Fast Fourier Transform*, ch. 1 to 2; Makhoul, "A fast cosine transform in one and two dimensions" (1980) |

## Key Takeaways

- The DFT is multiplication by $W_{kj} = e^{-2\pi i kj/n}$; $W/\sqrt{n}$ is unitary, so the transform keeps energy (Parseval) and its inverse is the conjugate transform divided by $n$.
- An FFT splits $x$ into $p$ interleaved subsequences, transforms them, and recombines them with twiddle factors and one size-$p$ DFT; mixed radices cover $n = 400 = 4 \cdot 4 \cdot 5 \cdot 5$.
- A real signal's spectrum is conjugate symmetric, so `rfft` computes $n/2 + 1$ bins from one complex FFT of size $n/2$.
- The convolution theorem gives circular convolution; zero-padding to $\ge$ `len(a) + len(b) - 1` makes it linear.
- The orthonormal DCT-II is a reordered FFT.

## How to read this chapter

Sections 1 to 3 derive the DFT, the convolution theorem, the mixed-radix Cooley-Tukey split, the half-size real FFT and the DCT-II, and work a four-point transform by hand. Section 4 fixes the interface and lists what a correct implementation must handle. Section 5 walks through the solution, [`fft.py`](../../../../course/solutions/python/tinyllm/sig/fft.py), and explains why each reshape, twiddle and sign is what it is. Try to predict each step before reading it.

---

## 1. Why now

Whisper's frontend (`L14.1`) takes a Fourier transform of every 25 ms frame of audio: 3000 frames of 400 samples for each 30 s clip. Multiplying each frame by a $400 \times 400$ DFT matrix costs 160000 complex multiplies per frame; an FFT does it in about 4000. More importantly, 400 is not a power of two, so the radix-2 FFT of most textbooks cannot transform it at all, and padding to 512 changes the frequencies of the bins and breaks parity with Whisper. This module builds the transform the STFT (`M12.5`) and the perceptual hash in `data.10` both stand on, and matches numpy to within a rounding bound that grows like $\log n$.

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
So the size-$n$ DFT is $p$ size-$m$ DFTs, a multiply by the twiddle factors $\omega_n^{rk'}$, and a size-$p$ DFT for each $k'$. Recursing until $m = 1$ costs $O(n \sum p_i)$ operations, $O(n \log n)$ for small radices. With radices 4, 2, 3, 5 every $n = 2^a 3^b 5^c$ works: $400 = 4 \cdot 4 \cdot 5 \cdot 5$, Whisper's frame. A size with another prime factor (7, 11, ...) is a `ValueError`, not a silent $O(n^2)$ fallback. Each level adds a few roundings, so the error grows like $\log n$: a correct implementation stays within $8 \log_2(\max(n, 2))\, \varepsilon\, \|x\|_2$ per entry (`M09.3`).

### 2.4 Real input: half the work

If $x$ is real, $X[n - k] = \overline{X[k]}$, so the bins $0 \dots n/2$ determine everything. Pack the even samples as real parts and the odd ones as imaginary parts, $z[t] = x[2t] + i\, x[2t+1]$, and take one complex FFT $Z$ of size $h = n/2$. The DFTs of the even and odd samples untangle as
$$E[k] = \tfrac12\big(Z[k] + \overline{Z[(h-k) \bmod h]}\big), \qquad O[k] = -\tfrac{i}{2}\big(Z[k] - \overline{Z[(h-k) \bmod h]}\big),$$
and $X[k] = E[k] + \omega_n^k O[k]$ for $k < h$, with the Nyquist bin $X[h] = E[0] - O[0]$ because $\omega_n^h = -1$. `irfft` runs the steps backwards; the imaginary parts of $X[0]$ and $X[h]$ must be zero for a real signal, so it drops them first, as numpy does.

### 2.5 The DCT-II via an FFT

pHash (`data.10`) takes the orthonormal DCT-II, $y[k] = s_k \cdot 2 \sum_j x[j] \cos\frac{\pi k (2j+1)}{2n}$ with $s_0 = \sqrt{1/(4n)}$ and $s_k = \sqrt{1/(2n)}$. Makhoul's trick reorders the input into $v = (x_0, x_2, x_4, \dots, x_5, x_3, x_1)$ (even samples forward, odd samples backward); then $y[k] = 2 s_k \operatorname{Re}\big(e^{-i\pi k/(2n)} V[k]\big)$ with $V = \operatorname{FFT}(v)$. The scales make the DCT matrix orthogonal, which transforming the identity confirms.

## 3. Worked example by hand

$x = (1, 2, 0, -1)$, $n = 4$, $\omega_4 = -i$:

| $k$ | $\sum_j x[j] (-i)^{jk}$ | $X[k]$ |
|---|---|---|
| 0 | $1 + 2 + 0 - 1$ | $2$ |
| 1 | $1 + 2(-i) + 0 - 1 \cdot (i)$ | $1 - 3i$ |
| 2 | $1 - 2 + 0 - 1 \cdot (-1)$ | $0$ |
| 3 | $1 + 2(i) + 0 - 1 \cdot (-i)$ | $1 + 3i$ |

$X[3] = \overline{X[1]}$, so `rfft` returns $(2, 1 - 3i, 0)$. Parseval: $\sum |x|^2 = 1 + 4 + 0 + 1 = 6$ and $\frac14 (4 + 10 + 0 + 10) = 6$. Through the packing of 2.4: $z = (1 + 2i, 0 - i)$, $Z = (1 + i, 1 + 3i)$; then $E[0] = \frac12((1+i) + (1-i)) = 1$, $O[0] = -\frac i2 (2i) = 1$, so $X[0] = 1 + 1 = 2$ and $X[2] = 1 - 1 = 0$.

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

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the hand-example DFT | unit | section 3's spectrum, its rfft half, and Parseval | sign and packing conventions |
| the DFT matrix is unitary | property | $U^H U = I$ for several $n$; one entry by formula; $n = 0$ rejected | the transform keeps energy |
| fft matches numpy | golden | 19 sizes up to 512, including 240 and 400, and a batch, within the $\log n$ bound | `M12.5` and `L14.1` parity |
| fft equals the DFT matrix | differential | mixed-radix sizes against $W x$ | the FFT is only fast multiplication |
| rfft and irfft match numpy | golden | rfft and irfft for every even size; irfft drops imaginary DC and Nyquist parts | every STFT frame |
| inverse and Parseval | property | `ifft(fft(x)) = x`, Parseval, `irfft(rfft(r)) = r` | `istft` and energy bookkeeping |
| fft convolution is linear convolution | differential | equals `numpy.convolve`, including lengths that need padding to 12 | the convolution theorem, done right |
| DCT matches scipy and is orthogonal | golden | scipy's orthonormal DCT-II on vectors and a 32 x 32 block; orthogonality | `data.10`'s pHash |
| unsupported sizes raise | boundary | prime factors 7 and 11, empty input, odd rfft, complex rfft, wrong bin count; inputs untouched | misconfigured frames fail loudly |

## 5. The solution, step by step

### 5.1 From first principles

1. **The reference is the matrix.** Everything in the module must equal $Wx$ to rounding, so `dft_matrix` comes first: a direct, obviously correct $O(n^2)$ definition that the fast code is compared against. Reducing $kj \bmod n$ before computing the phase keeps every angle in $[0, 2\pi)$, where `exp` is accurate.
2. **The FFT is the split of section 2.3, applied recursively.** For the smallest usable radix $p$ of $n$: form the $p$ interleaved subsequences, transform each (recursively) at size $m = n/p$, multiply by the twiddles $\omega_n^{rk'}$, and finish with a size-$p$ DFT across $r$ for every $k'$. The base case $n = 1$ is the identity.
3. **Interleaving is a reshape and a swap.** Reading $x$ row-major as an $m \times p$ matrix puts $x[tp + r]$ at row $t$, column $r$; swapping the last two axes gives a $p \times m$ array whose row $r$ is subsequence $r$. No index arithmetic, and the recursion transforms the last axis of a batch of $p$ rows at once.
4. **Only sizes $2^a 3^b 5^c$ are allowed, checked up front.** If any other prime divides $n$, the recursion would reach a size with no radix. Checking before starting gives a clear error at the call site. Radix 4 is tried before 2 so that powers of two use fewer, larger levels.
5. **The inverse reuses the forward transform.** Since $W^{-1} = \overline{W}/n$, $\operatorname{ifft}(X) = \overline{\operatorname{fft}(\overline X)}/n$. One FFT implementation, one set of twiddles.
6. **A real FFT is a half-size complex FFT plus untangling.** Pack even and odd samples into one complex signal, transform at size $h$, and separate $E$ and $O$ using the conjugate symmetry of real signals (section 2.4). `irfft` inverts each step: rebuild $E$ and $O$ from $X$, recombine into $Z$, inverse transform, and unpack real and imaginary parts into even and odd samples.
7. **Linear convolution needs padding.** The DFT computes circular convolution. Padding both inputs to $m \ge \text{len}(a) + \text{len}(b) - 1$ leaves room for the whole result, so nothing wraps; the smallest such $m$ with factors 2, 3, 5 keeps the FFT applicable.
8. **The DCT-II is an FFT of a permuted input.** Makhoul's reordering makes $\sum_j x[j]\cos\frac{\pi k(2j+1)}{2n}$ equal to $\operatorname{Re}(e^{-i\pi k/(2n)} V[k])$, so one FFT plus a phase and a scale computes it.

### 5.2 The code, annotated

```python
def dft_matrix(n: int, unitary: bool = False) -> NDArray:
    if n < 1:
        raise ValueError(f"dft_matrix: n must be >= 1, got {n}")
    j = np.arange(n)
    # Reduce k * j mod n before scaling: the phase stays in [0, 2 pi) and
    # exp is evaluated where it is accurate.
    w = np.exp(-2j * np.pi * (np.outer(j, j) % n) / n)     # minus sign: numpy's forward convention
    return w / np.sqrt(n) if unitary else w
```

The recursive FFT:

```python
_RADICES = (4, 2, 3, 5)

def _radix(n: int) -> int:
    """The first radix the FFT splits n by (4 before 2); ValueError if none."""
    for p in _RADICES:
        if n % p == 0:
            return p
    raise ValueError(f"fft: n = {n} has a prime factor other than 2, 3, 5")


def _fft(x: NDArray) -> NDArray:
    """Decimation in time over the last axis of a complex array."""
    n = x.shape[-1]
    if n == 1:
        return x.copy()                                   # the DFT of one sample is itself
    p = _radix(n)
    m = n // p
    lead = x.shape[:-1]
    # Subsequence r holds x[r], x[r + p], x[r + 2p], ...: shape [..., p, m].
    sub = _fft(np.swapaxes(x.reshape(*lead, m, p), -1, -2))   # p transforms of size m
    r = np.arange(p)[:, None]
    k = np.arange(m)[None, :]
    tw = np.exp(-2j * np.pi * ((r * k) % n) / n)  # W_n^{r k}, [p, m]
    # X[q m + k] = sum_r W_p^{r q} (W_n^{r k} Y_r[k]): a size-p DFT per k.
    out = np.einsum("qr,...rk->...qk", dft_matrix(p), sub * tw)
    return out.reshape(*lead, n)                         # row q, column k -> index q m + k
```

The output's last two axes are $(q, k')$, and the row-major reshape maps them to $qm + k'$, exactly the index in the formula. For $n = 400$ the levels are radix 4 (to 100), 4 (to 25), 5 (to 5), 5 (to 1).

```python
def fft(x: NDArray) -> NDArray:
    a = np.asarray(x, dtype=np.complex128)
    if a.ndim == 0 or a.shape[-1] < 1:
        raise ValueError("fft: need at least one sample on the last axis")
    _check_size(a.shape[-1])          # strip 2, 3, 5 and require 1 to remain
    return _fft(a)


def ifft(X: NDArray) -> NDArray:
    a = np.asarray(X, dtype=np.complex128)
    # The inverse is the conjugate transform, divided by n.
    return np.conj(fft(np.conj(a))) / a.shape[-1]
```

The real FFT, with the reversal trick for $\overline{Z[(h - k) \bmod h]}$:

```python
def rfft(x: NDArray) -> NDArray:
    # ... reject complex input and odd or short lengths ...
    h = n // 2
    # Pack: even samples as the real part, odd samples as the imaginary part.
    z = fft(a[..., 0::2] + 1j * a[..., 1::2])  # [..., h]
    # Reversing gives Z[h-1], ..., Z[0]; rolling by one gives Z[0], Z[h-1], ..., Z[1],
    # which is Z[(h - k) mod h] for k = 0 .. h-1.
    zr = np.conj(np.roll(z[..., ::-1], 1, axis=-1))  # conj(Z[(h - k) mod h])
    even = 0.5 * (z + zr)  # DFT of the even samples
    odd = -0.5j * (z - zr)  # DFT of the odd samples
    k = np.arange(h)
    lower = even + np.exp(-2j * np.pi * k / n) * odd  # bins 0 .. h-1
    nyq = even[..., :1] - odd[..., :1]  # bin h: W_n^h = -1
    return np.concatenate([lower, nyq], axis=-1)
```

The inverse runs the same algebra backwards. Given $X[k]$ and $\overline{X[h - k]}$, $E = \frac12(X[k] + \overline{X[h-k]})$ and $O = \frac12(X[k] - \overline{X[h-k]})\,\omega_n^{-k}$, and $Z = E + iO$:

```python
def irfft(X: NDArray, n: int) -> NDArray:
    # ... require even n >= 2 and exactly h + 1 bins ...
    a = a.copy()                                     # never modify the caller's spectrum
    # The DC and Nyquist bins of a real signal are real: drop any imaginary
    # part, as numpy does, instead of letting it leak into the samples.
    a[..., 0] = a[..., 0].real
    a[..., h] = a[..., h].real
    k = np.arange(h)
    xk = a[..., :h]
    xr = np.conj(a[..., h - k])  # conj(X[h - k]), k = 0 .. h-1
    even = 0.5 * (xk + xr)
    odd = 0.5 * (xk - xr) * np.exp(2j * np.pi * k / n)
    z = ifft(even + 1j * odd)
    out = np.empty(a.shape[:-1] + (n,))
    out[..., 0::2] = z.real                          # unpack: real part -> even samples
    out[..., 1::2] = z.imag                          #         imaginary part -> odd samples
    return out
```

Convolution and the DCT:

```python
def fft_convolve(a: NDArray, b: NDArray) -> NDArray:
    # ... two non-empty 1-D arrays ...
    full = a.size + b.size - 1
    # Zero-pad past the full length so the circular convolution the DFT
    # computes never wraps around: then it equals the linear one.
    m = _fast_size(full)                             # smallest m >= full with factors 2, 3, 5
    fa = fft(np.pad(a.astype(np.complex128), (0, m - a.size)))
    fb = fft(np.pad(b.astype(np.complex128), (0, m - b.size)))
    y = ifft(fa * fb)[:full]
    if not (np.iscomplexobj(a) or np.iscomplexobj(b)):
        return y.real.copy()                         # real inputs: drop rounding-level imaginary parts
    return y


def dct2_ortho(x: NDArray, axis: int = -1) -> NDArray:
    a = np.moveaxis(np.asarray(x, dtype=np.float64), axis, -1)
    # ... n >= 1 ...
    # Makhoul: even samples forward, odd samples backward, then one FFT.
    v = np.concatenate([a[..., 0::2], a[..., 1::2][..., ::-1]], axis=-1)
    vf = fft(v)
    k = np.arange(n)
    # scipy's unnormalized DCT-II: y[k] = 2 Re(exp(-i pi k / 2n) V[k]).
    y = 2.0 * (vf * np.exp(-1j * np.pi * k / (2 * n))).real
    scale = np.full(n, np.sqrt(1.0 / (2 * n)))
    scale[0] = np.sqrt(1.0 / (4 * n))  # norm='ortho' makes the DCT matrix orthogonal
    return np.moveaxis(y * scale, -1, axis)
```

`dct2_ortho` works along any axis, so a 2-D DCT of a 32 x 32 block, as pHash needs, can be computed as `dct2_ortho(dct2_ortho(block, axis=0), axis=1)`.

Full solution: [`fft.py`](../../../../course/solutions/python/tinyllm/sig/fft.py).

### 5.3 Why this solution

- **Cost.** Each recursion level does $O(np)$ work (the size-$p$ DFTs) plus $O(n)$ twiddles, and there are $\log_p n$ levels, so $n = 400$ costs about $400 \cdot (4 + 4 + 5 + 5) = 7200$ complex multiply-adds instead of $160\,000$. `rfft` halves that again by transforming $n/2$ points. Memory is $O(n)$ per level.
- **Why mixed radix and not radix 2 with padding.** Padding a 400-sample frame to 512 changes the bin frequencies from multiples of $16000/400 = 40$ Hz to multiples of $31.25$ Hz, which is a different spectrogram than the one Whisper was trained on. Supporting radices 3 and 5 gives the exact transform for every size the course uses.
- **Why raise on other primes rather than fall back to $O(n^2)$.** A silent quadratic path turns a misconfigured frame length into a slow pipeline with no error. Production libraries handle large primes with Bluestein's algorithm; here the sizes are known, so an error is more useful.
- **Why the recursion uses `dft_matrix(p)` for the small DFT.** For $p \le 5$ a dense $p \times p$ matrix product is the simplest correct butterfly, and `einsum` vectorizes it over every $k'$ and every batch row. Hand-written radix-4 butterflies would be faster but would duplicate the math.
- **Accuracy.** Each level adds a few roundings, so the error grows like $\log n$, about $8 \log_2 n\,\varepsilon\,\|x\|_2$ per entry. The direct matrix product, by contrast, accumulates $n$ terms per entry.

### 5.4 Explain it back

1. Why does `x.reshape(*lead, m, p)` followed by `swapaxes` give the interleaved subsequences rather than contiguous blocks?
   <details><summary>Answer</summary>
   A row-major $m \times p$ view puts $x[tp + r]$ at $(t, r)$: column $r$ collects every $p$-th sample starting at $r$. Swapping makes that column a row. Reshaping to $(p, m)$ directly would give contiguous blocks $x[rm \dots rm + m - 1]$, a different (and wrong) decomposition with the same shape.
   </details>
2. Why is the Nyquist bin $E[0] - O[0]$ and not $E[0] + O[0]$?
   <details><summary>Answer</summary>
   $X[k] = E[k] + \omega_n^k O[k]$ with $E$ and $O$ periodic in $h$. At $k = h$, $E[h] = E[0]$, $O[h] = O[0]$, and $\omega_n^h = e^{-i\pi} = -1$. In the section 3 example this gives $X[2] = 1 - 1 = 0$.
   </details>
3. Why does `irfft` discard the imaginary parts of bins $0$ and $h$?
   <details><summary>Answer</summary>
   For a real signal those bins are real ($X[0] = \sum x$, $X[h] = \sum (-1)^j x[j]$). Any imaginary part there has no real signal behind it, and passing it through the inverse would add a spurious component (an alternating pattern for the Nyquist bin). numpy drops it, so the solution does too.
   </details>
4. Lengths 5 and 4 give a full convolution of length 8. Why not transform at size 5 and multiply?
   <details><summary>Answer</summary>
   The DFT product is circular convolution of length 5: the last 3 terms of the linear result wrap around and add onto the first 3. Padding to $m \ge 8$ leaves room for every term, so nothing wraps and the first 8 entries are the linear convolution.
   </details>
5. Why is `scale[0]` different from the other DCT scales?
   <details><summary>Answer</summary>
   The $k = 0$ basis vector is constant, $\cos 0 = 1$ everywhere, and its squared norm under the unnormalized transform is twice that of the other cosines. Scaling it by $\sqrt{1/(4n)}$ instead of $\sqrt{1/(2n)}$ gives every row unit norm, which is what makes the DCT matrix orthogonal.
   </details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. twiddle factors with $+2\pi i$ | every size above 4 wrong; spectra mirrored | every forward exponent is $-2\pi i$, matching `dft_matrix` |
| 2. Nyquist bin as $E[0] + O[0]$ | the last rfft bin wrong | `nyq = even[..., :1] - odd[..., :1]`, since $\omega_n^h = -1$ |
| 3. ifft without the $1/n$ | round trips scaled by $n$ | `ifft` divides the conjugate transform by `a.shape[-1]` |
| 4. multiplying unpadded spectra | circular wrap-around; wrong length | pads both inputs to `_fast_size(len(a) + len(b) - 1)` and slices to `full`; see step 7 |
| 5. scaling bin 0 like the others in the DCT | the DC coefficient $\sqrt2$ too large; not orthogonal | `scale[0] = sqrt(1 / (4n))` |
| 6. keeping an imaginary DC part in irfft | a spurious alternating component | `irfft` sets bins 0 and $h$ to their real parts first |
| 7. splitting into contiguous blocks instead of interleaved subsequences | a different transform that still has the right shape | reshapes to `(m, p)` and swaps axes; see step 3 |
| 8. accepting $n = 0$ in `dft_matrix` | an empty matrix instead of an error | raises for `n < 1` |
| 9. accepting extra bins in irfft | the trailing bins silently ignored | requires exactly $h + 1$ bins |
| 10. accepting complex input in rfft | the imaginary part discarded | `rfft` rejects complex input with `np.iscomplexobj` |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M00.2` | roots of unity from Euler's formula (reading) |
| Back | `M03.3` | unitary matrices keep lengths (reading) |
| Back | `M09.3` | the $\log n$ error bound (reading) |
| Forward | `M12.5` | `stft` runs `rfft` on every frame; `istft` runs `irfft` |
| Forward | `data.10` | pHash's 32 x 32 `dct2_ortho` |
| Forward | `S-M12` | DFT properties, the convolution theorem, FFT operation counts |

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `fft` | pocketfft (numpy, scipy) | radices up to 11 plus Bluestein for large primes, cache-friendly plans | `numpy/fft/_pocketfft_umath.cpp` |
| `fft` | FFTW | plans tuned by measurement ("wisdom"), codelets generated by `genfft` | Frigo and Johnson, Proc. IEEE 2005 |
| `rfft` in Rust | `rustfft` and `realfft` (the engine, `L15.1`) | SIMD butterflies; the same half-size packing | `realfft` crate docs |
| `fft` in C | optional `L14.6` | mixed-radix real FFT plans for the Whisper frontend | `c/src/kernels/audio.c` |
