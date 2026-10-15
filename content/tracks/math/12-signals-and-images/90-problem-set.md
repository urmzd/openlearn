<!-- ol:module S-M12 -->
# Signals and images problem set: convolution, sampling, Fourier, mel, contrastive losses, and WER

## Overview

| | |
|---|---|
| **Module** | `S-M12` · solve · none · Pass 12 · 6 to 8 h |
| **You build** | answers in `solve/S-M12.toml` (68 checked by SymPy) and 4 proofs in `solve/S-M12/q7.md`, `q21.md`, `q22.md`, and `q24.md` (self-graded against their rubrics) |
| **Contract** | none: a pen and paper set |
| **Tests** | `course/solve/S-M12/key.toml` (hidden): typed answers plus reject canaries; the problems are in `course/solve/S-M12/problems.md` and in section 4 |
| **Needs** | no module. Reading: `M12.1` to `M12.6` and the [Signals and Images topic](README.md) |
| **Used by** | no call site (a solve set). It checks the arithmetic behind `M12.1` to `M12.6`, and holds two solve-only topics the speech and vision parts lean on: contrastive losses (CLIP and SigLIP in `L13`) and word error rate (`L14`) |
| **Milestone** | `MS-P12` (the Pass 12 gate runs `ol check` on every solve part of the pass) |
| **Optional depth** | Dumoulin and Visin, "A guide to convolution arithmetic for deep learning" (free); Oppenheim and Schafer, *Discrete-Time Signal Processing*, ch. 4 and 8; van den Oord, Li, Vinyals, "Representation learning with contrastive predictive coding" (2018); Zhai et al., "Sigmoid loss for language image pre-training" (ICCV 2023); Jurafsky and Martin, *Speech and Language Processing*, ch. 2.5 (edit distance, free) |

## Key Takeaways

- One formula sizes every conv, pool, and patch embedding, and receptive fields grow by $(k - 1)$ times the product of earlier strides (q1 to q3).
- A conv is a Toeplitz matrix; its transpose (col2im) is the backward pass, and the backward of a correlation is a true convolution (q4 to q7).
- Sampling keeps only frequencies below half the rate; everything above folds back, in time and in images (q11, q12).
- The DFT is a unitary change of basis: Parseval, conjugate symmetry for real signals, and the convolution theorem with zero padding (q15, q16); an FFT needs radices that divide $n$ (q17).
- InfoNCE is cross-entropy with the positive as the label, bounded below by $\log N - I$; SigLIP's sigmoid loss decouples the pairs (q20 to q22). WER is an edit distance over the reference length: it can exceed 1 and is not symmetric (q23, q24).

## How to work this chapter

```bash
ol start S-M12              # writes solve/S-M12.toml and one file per proof
ol check S-M12              # SymPy checks the answers, then asks each proof rubric (y/n)
ol check S-M12 --regrade    # ask the rubrics again after you change a proof
```

---

## 1. Why now

Pass 12 hands the system eyes and ears, and most of its bugs are arithmetic: an encoder that expects 1500 positions and gets 1501, a patch embedding with 196 tokens where the prompt reserved 197 slots, a spectrogram whose bins are 80 Hz apart instead of 40, a thumbnail that aliased before it was hashed. `M12.1` to `M12.6` built the functions; this set makes you predict their numbers by hand, so that a wrong shape in `L13` or `L14` looks wrong to you before a test says so. It also covers two topics with no module of their own: the contrastive losses CLIP and SigLIP train with, and the word error rate `L14` reports.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $n, k, s, p, d$ | input length, kernel size, stride, padding, dilation | ints |
| $T$ | Toeplitz matrix of a 1-D correlation | matrix |
| $\text{sr}$, $f_N = \text{sr}/2$ | sample rate and Nyquist frequency | Hz |
| $k(t)$, $a$ | interpolation kernel and Keys' parameter | function, float |
| $X = Wx$ | DFT | vector |
| $n_\text{fft}$, $\text{hop}$ | STFT frame length and hop | ints |
| $m(f)$ | mel value | mel |
| $s_j$, $N$ | contrastive logits and candidate count | floats, int |
| $\sigma(u) = 1/(1 + e^{-u})$ | logistic sigmoid | function |
| $S, D, I$ | substitutions, deletions, insertions | ints |

### 2.1 Convolution arithmetic and the conv as a matrix

$\text{out} = \lfloor (n + 2p - d(k-1) - 1)/s \rfloor + 1$. A stack of layers sees $1 + \sum_\ell (k_\ell - 1) \prod_{m < \ell} s_m$ input samples. The correlation $y = Tx$ has the kernel along each row of $T$, shifted one column per row; $T^\top$ maps an output gradient to an input gradient, which is the full true convolution of the gradient with the kernel. im2col copies each pixel into every window that contains it; col2im adds those copies back (`M12.1`).

### 2.2 Sampling, aliasing, and interpolation

A tone at $f$ sampled at $\text{sr}$ is indistinguishable from $|f - j\,\text{sr}|$ for any integer $j$; it appears at the representative in $[0, f_N]$. Keeping every $q$-th pixel divides the rate by $q$, so frequencies above $1/(2q)$ cycles per input pixel alias. Antialiased resizing stretches the kernel by the scale factor to remove them first (`M12.3`).

### 2.3 Fourier

$X[k] = \sum_j x[j] e^{-2\pi i jk/n}$. $\sum |X|^2 = n \sum |x|^2$. Real input gives $X[n - k] = \overline{X[k]}$; real and even input gives a real spectrum. The product of DFTs is the DFT of the circular convolution; padding to length $\ge n_a + n_b - 1$ makes it linear. An FFT recursion splits $n$ by its factors (`M12.4`). Centered STFT frames number $1 + \lfloor n/\text{hop} \rfloor$, bins are $\text{sr}/n_\text{fft}$ apart, and a periodic Hann at hop $n_\text{fft}/4$ overlap-adds its square to $3/2$ (`M12.5`).

### 2.4 Mel and decibels

Slaney mel is linear to 15 mel at 1000 Hz, then 27 mel per factor 6.4. Power in dB is $10 \log_{10}$. Whisper's feature is $(\max(L, \max L - 8) + 4)/4$ with $L = \log_{10} \max(\text{mel}, 10^{-10})$ (`M12.6`).

### 2.5 Contrastive losses

CLIP scores a batch of $N$ image-text pairs with a similarity matrix and, for each image, classifies its own caption among the $N$ with a softmax: InfoNCE, $L = -\log \frac{e^{s_1}}{\sum_j e^{s_j}}$. Its gradient is softmax minus one-hot, and van den Oord et al. show $I(X; Y) \ge \log N - L$: more negatives can certify more mutual information. SigLIP replaces the softmax by an independent logistic loss per pair, $-\log \sigma(z(ts + b))$ with $z = +1$ on the diagonal and $-1$ elsewhere, so no normalizer couples the pairs.

### 2.6 Edit distance and WER

The word edit distance is the minimum number of substitutions, deletions, and insertions turning one word sequence into another, computed by a DP over prefixes. WER divides it by the number of reference words: insertions can push it above 1, and swapping reference and hypothesis changes the denominator.

## 3. Worked example by hand

**Conv.** $n = 10$, $k = 3$, $s = 2$, $p = 1$: $\lfloor (10 + 2 - 3)/2 \rfloor + 1 = 5$. The window starts sit at $-1, 1, 3, 5, 7$ in input coordinates, and the next one ($9$) would need input sample 11.

**Aliasing.** 9 kHz sampled at 16 kHz: $|9000 - 16000| = 7000$ Hz, below $f_N = 8000$, so it appears at 7 kHz.

**DFT.** $x = (1, -1, 1, -1)$: $X[k] = \sum_j (-1)^j (-i)^{jk}$; only $k = 2$ survives, $X = (0, 0, 4, 0)$, and Parseval gives $16 = 4 \cdot 4$.

**InfoNCE.** Two candidates with equal logits: $L = \log 2$; with the positive ahead by $\ln 3$, $L = \log(1 + 1/3) = \log(4/3)$.

**WER.** Reference "a b c", hypothesis "a x c d": one substitution and one insertion, WER $= 2/3$. Written in `solve/` as `2/3`.

## 4. The problem set

Write each answer in `solve/S-M12.toml`; lettered parts are their own tables:

```toml
[q1.a]
answer = "112"
[q5.a]
answer = "[[1, 0, -1, 0], [0, 1, 0, -1]]"
[q15.c]
answer = "[1, -I, -1, I]"
[q7]
proof = "S-M12/q7.md"
```

Give exact values, not decimals.

<!-- ol:problems S-M12 -->

### Convolution arithmetic

**q1.** Give the output length for (a) $n = 224$, $k = 7$, $s = 2$, $p = 3$ (a ResNet stem); (b) a $3 \times 3$ max pool with $s = 2$, $p = 1$ on that output, along one axis; (c) Whisper's stem on 3000 frames: a conv with $k = 3$, $p = 1$, $s = 1$, then one with $k = 3$, $p = 1$, $s = 2$; (d) $n = 32$, $k = 3$, $d = 4$, $p = 0$, $s = 1$. `[number]`

**q2.** (a) How many input samples does a kernel of size $k$ with dilation $d$ span? Answer in $k$ and $d$. (b) For odd $k$, stride 1, and dilation 1, which padding $p$ keeps the output length equal to the input length? Answer in $k$. `[expr]`

**q3.** The receptive field of an output is the number of input samples (along one axis) it depends on. Give it for (a) three stacked $3 \times 3$ convs with stride 1; (b) a $3 \times 3$ conv with stride 2 followed by a $3 \times 3$ conv with stride 1. `[number]`

### Correlation, convolution, and the conv as a matrix

**q4.** $x = (1, 2, 3, 4)$ and $w = (1, 0, -1)$, no padding. Give (a) the cross-correlation $y[i] = \sum_r w[r]\, x[i + r]$ and (b) the true convolution (kernel flipped). `[vector]`

**q5.** For the cross-correlation of q4 written as $y = T x$: (a) give $T$. `[matrix]` (b) Give the input gradient $T^\top g$ for the upstream gradient $g = (1, 1)$. `[vector]` (c) Is $T^\top g$ equal to the full (zero-padded) true convolution of $g$ with $w$? `[bool]`

**q6.** (a) With $x$ and $g$ as in q4 and q5, give the weight gradient $\partial L / \partial w[r] = \sum_i g[i]\, x[i + r]$. `[vector]` (b) A $1 \times 1 \times 3 \times 3$ image goes through im2col with a $2 \times 2$ kernel, stride 1, no padding. How many times is its center pixel copied into the columns? `[number]`

**q7.** Prove that col2im is the adjoint of im2col: for every input $x$ and every column array $Y$ of the matching shape, $\langle \operatorname{im2col}(x), Y \rangle = \langle x, \operatorname{col2im}(Y) \rangle$, where col2im adds every column entry back to the pixel it was copied from and drops entries copied from the zero padding. `[proof]`

### Parameter and FLOP counts

**q8.** A conv from 64 to 128 channels with a $3 \times 3$ kernel, groups 1, and a bias. Give (a) its parameter count; (b) the parameter count of a depthwise $3 \times 3$ conv on 64 channels with a bias; (c) the multiply-accumulates of the first conv for one image with a $56 \times 56$ output. `[number]`

**q9.** ViT-B/16 on a $3 \times 224 \times 224$ image with width 768. Give (a) the number of patches; (b) the parameters of the patch embedding (a conv with kernel and stride 16, with a bias); (c) the sequence length after prepending a CLS token. `[number]`

### Layouts and strides

**q10.** A float32 tensor with logical shape $(N, C, H, W) = (2, 3, 4, 5)$. (a) Give its byte strides per logical axis when stored NCHW, and (b) when stored NHWC (channels last). `[vector]` (c) Give the byte offset of element $(1, 2, 3, 4)$ in the NHWC storage. `[number]` (d) Give the length of one flattened patch when a $3 \times 224 \times 224$ image is cut into $14 \times 14$ patches. `[number]`

### Sampling and aliasing

**q11.** (a) Give the Nyquist frequency in Hz at a 16 kHz sample rate. At what frequency in Hz does a pure tone appear when (b) 7 kHz is sampled at 10 kHz; (c) 12 kHz is sampled at 16 kHz; (d) 25 kHz is sampled at 16 kHz? `[number]`

**q12.** An image has a stripe pattern with a period of 3 pixels. It is shrunk by 4 on each axis by keeping every fourth pixel, with no filter. (a) Give the period in output pixels of the stripes that appear. `[number]` (b) Can the output grid represent the original pattern without aliasing? `[bool]` (c) Give the highest frequency, in cycles per input pixel, that a shrink by 4 can keep without aliasing. `[number]` (d) A 1-pixel checkerboard is shrunk by 2 by keeping every other pixel on each axis. The result is: (a) a checkerboard, (b) a constant image, (c) stripes. `[choice]`

### Interpolation kernels

**q13.** Keys' cubic kernel is $k(t) = (a+2)|t|^3 - (a+3)|t|^2 + 1$ for $|t| < 1$ and $a|t|^3 - 5a|t|^2 + 8a|t| - 4a$ for $1 \le |t| < 2$. Give (a) $k(1/2)$ and (b) $k(3/2)$ with $a = -1/2$ (Pillow), and (c) $k(3/2)$ with $a = -3/4$ (PyTorch). `[number]`

**q14.** Pillow shrinks with an antialiased kernel stretched by the scale factor. (a) Give the support radius in input pixels of the bilinear kernel (support 1) when shrinking 1000 pixels to 250. (b) Give the number of taps in Pillow's weight table for that pass, $2 \lceil \text{support} \rceil + 1$. (c) Give the support radius of Lanczos3 when shrinking by 2. `[number]`

### The DFT

**q15.** With $X[k] = \sum_j x[j] e^{-2\pi i jk/n}$, give the DFT of (a) $(1, 0, 0, 0)$, (b) $(1, 1, 1, 1)$, (c) $(0, 1, 0, 0)$. `[vector]` (d) A real signal of length 8 has $X[3] = 2 - i$. Give $X[5]$. `[number]`

**q16.** (a) Give $\sum_k |X[k]|^2$ for $x = (1, 2, 0, -1)$. (b) Give the circular convolution of length 4 of $(1, 2, 0, 0)$ and $(1, 1, 0, 0)$. `[vector]` (c) Give the smallest transform length for which an FFT product computes the linear convolution of signals of lengths 300 and 100. `[number]` (d) Is the DFT of a real, even signal ($x[j] = x[(n - j) \bmod n]$) real? `[bool]`

### The FFT

**q17.** (a) Into how many radix stages does an FFT split $n = 400$ when it uses radices 4, 4, 5, 5? (b) How many complex multiplies does the direct DFT of length 400 take ($n^2$)? (c) How many butterflies does a radix-2 FFT of length 1024 take, $(n/2) \log_2 n$? `[number]` (d) Can a radix-2-only FFT compute a 400-point DFT, without changing the bin frequencies? `[bool]`

### Windows and the STFT

**q18.** Whisper: 30 s of audio at 16 kHz, $n_\text{fft} = 400$, hop 160, centered frames. Give (a) the number of STFT frames before Whisper drops the last one; (b) the spacing of the bins in Hz; (c) the number of rfft bins per frame; (d) the constant value of $\sum_t w[m - t\,\text{hop}]^2$ for a periodic Hann window at hop $n_\text{fft}/4$; (e) the frame length in milliseconds. `[number]` (f) Give the periodic Hann window of length 4. `[vector]`

### Mel and decibels

**q19.** On the Slaney scale, give the mel value of (a) 1000 Hz and (b) 6400 Hz. (c) On the HTK scale, $m(f) = 2595 \log_{10}(1 + f / 700)$: give $m(700)$. (d) Give a power ratio of 1000 in decibels. (e) Whisper computes $L = \log_{10}(\max(\text{mel}, 10^{-10}))$, then $\max(L, \max L - 8)$, then $(L + 4)/4$. Give the feature of a band with mel power $10^{-12}$ in a clip whose largest $L$ is 1. (f) Give the feature value of pure silence. `[number]`

### Contrastive losses

**q20.** InfoNCE for one anchor scores $N$ candidates with logits $s_j$ (the positive is $j = 1$) and takes $L = -\log \frac{e^{s_1}}{\sum_j e^{s_j}}$, the cross-entropy with the positive as the label. (a) Give $L$ for $s = (2, 0, 0, 0)$. (b) Give $L$ when all $N$ logits are equal. Answer in $N$. `[expr]` (c) The InfoNCE bound says $I(X; Y) \ge \log N - L$. Give the bound in nats for $N = 1024$ and $L = 2$. `[number]` (d) SigLIP's loss for one pair is $\ell = -\log \sigma\big(z (t s + b)\big)$ with label $z = \pm 1$. Give $\partial \ell / \partial s$ for a positive pair ($z = 1$) with $t = 10$, $b = -10$, $s = 1$. `[number]`

**q21.** Prove that the gradient of the InfoNCE loss of q20 with respect to the logits is $\partial L / \partial s_j = \operatorname{softmax}(s)_j - [j = 1]$, and conclude that $L > 0$ for every finite $s$ and that $L \to 0$ only when $s_1 - s_j \to \infty$ for every $j \ne 1$. `[proof]`

**q22.** Prove that for SigLIP's loss $\ell = \log\big(1 + e^{-z(t s + b)}\big)$, $\partial \ell / \partial s = -t z\, \sigma\big(-z(t s + b)\big)$, and explain why, unlike q21, the gradient of one pair does not depend on the logits of the other pairs in the batch. `[proof]`

### Word error rate

**q23.** The word error rate is $\text{WER} = (S + D + I) / N_\text{ref}$, the minimum number of substitutions, deletions, and insertions turning the reference into the hypothesis, over the number of reference words. Give the WER of the hypothesis "oh hello there" against the reference "hello". `[number]`

**q24.** Prove that WER is not a metric on word sequences, by showing it is not symmetric with an example (compute both directions), and state which metric axiom fails and why. `[proof]`

<!-- /ol:problems -->

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| Dropping the $+1$ in the size formula | 1499 positions where Whisper's encoder has 1500 | q1 c (canary) |
| Ignoring dilation in the kernel span | a dilated layer sized like a dense one | q1 d (canary 30) |
| Flipping the kernel in a network conv | correlation and convolution swapped | q4 a, q5 a (canaries) |
| Counting FLOPs where MACs were asked (or the reverse) | a factor 2 in every cost estimate | q8 c (canary) |
| Forgetting the CLS token | 196 positions where the encoder has 197 | q9 c (canary) |
| Strides in elements, or in memory order instead of per logical axis | a kernel indexing the wrong neighbour | q10 a, q10 b (canaries) |
| Folding once instead of to the nearest multiple of the rate | 9 kHz instead of 7 kHz | q11 d (canary) |
| The bilinear weight where the cubic was asked, or the wrong $a$ | interpolation that does not match Pillow or PyTorch | q13 (canaries) |
| Forgetting the antialias stretch | support 1 where Pillow uses 4 | q14 a (canary) |
| The inverse DFT's sign | a mirrored spectrum | q15 c (canary) |
| Treating 400 as $2^4 \cdot 5^2$ stages | six radix-2 and radix-5 stages instead of four | q17 a (canary 6) |
| Counting frames without the centered padding's extra frame | 3000 where Whisper computes 3001 | q18 a (canary) |
| $\sum w$ instead of $\sum w^2$ for weighted overlap-add | reconstructions scaled by 3/4 | q18 d (canary 2) |
| The symmetric Hann window | $(0, 3/4, 3/4, 0)$ | q18 f (canary) |
| 20 $\log_{10}$ for a power | decibels doubled | q19 d (canary 60) |
| Applying only the floor in Whisper's scaling | $-3/2$ where the 8-unit clip gives $-3/4$ | q19 e (canary) |
| Mixing $\log_2$ into a nats bound | $10 - 2$ instead of $\ln 1024 - 2$ | q20 c (canary) |
| Dividing WER by the hypothesis length | $2/3$ where the answer is 2 | q23 (canary) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M12.1` | conv sizes, Toeplitz form, im2col and col2im (q1 to q9) |
| Back | `M12.2` | strides and patches (q10) |
| Back | `M12.3` | aliasing, interpolation kernels, antialias support (q11 to q14) |
| Back | `M12.4` | DFT properties and FFT counts (q15 to q17) |
| Back | `M12.5`, `M12.6` | frames, windows, mel, and Whisper's scaling (q18, q19) |
| Forward | `L13.1`, `L13.3` | conv layers and the ViT patch embedding (with B14's L13 group) |
| Forward | `L13.5` | CLIP and SigLIP contrastive training (q20 to q22) |
| Forward | `L14.1`, `L14.2`, `L14.5` | the Whisper frontend, stem, and the WER the speech evals report (q18, q19, q23, q24) |
