# Signals and images problem set: convolution, sampling, Fourier, mel, contrastive losses, and WER

Teaching a language model system to see and hear introduces a new class of bugs, and most of them are arithmetic: an encoder that expects 1500 positions and gets 1501, a patch embedding with 196 tokens where the prompt reserved 197 slots, a spectrogram whose bins are 80 Hz apart instead of 40, a thumbnail that aliased before it was hashed. The sections of this chapter built the functions, from [Convolution as a linear map](01-convolution-as-a-linear-map.md) through [Mel scale and filterbanks](06-mel-scale-and-filterbanks.md). This problem set predicts their numbers by hand, so that a wrong shape in the vision and speech parts of *Build Your Own LLM System* looks wrong before anything runs.

The set also covers two topics that have no section of their own but that the multimodal models lean on: the contrastive losses CLIP and SigLIP train with ([Contrastive image-text: CLIP and SigLIP](../../build-an-llm/14-vision/05-contrastive-image-text.md)) and the word error rate that speech evaluations report ([Multimodal eval metrics](../../build-an-llm/15-speech/05-multimodal-eval-metrics.md)). It has 24 problems with 68 exact answers and 4 proofs (Problems 7, 21, 22, and 24), each followed by a collapsed worked answer with the final result, the derivation, and the common wrong answers and why they are wrong. Model proofs are also in [`course/solutions/solve/S-M12/`](../../../../course/solutions/solve/S-M12/). Try each problem before opening its answer. You need nothing beyond the six sections of this chapter and its [opening](README.md).

> **Learning objectives**
> - Size any convolution, pool, or patch embedding and compute receptive fields, parameter counts, and multiply-accumulates.
> - Write a 1-D correlation as a Toeplitz matrix and derive its backward pass, and prove that col2im is the adjoint of im2col.
> - Predict aliasing in audio and images, and evaluate interpolation kernels and antialiased supports.
> - Use DFT properties (Parseval, conjugate symmetry, the convolution theorem) and count FFT work, STFT frames, and mel and decibel values.
> - Derive the gradients of InfoNCE and SigLIP's sigmoid loss, compute WER, and prove that WER is not a metric.

## The ideas you need

The problems use the following symbols.

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

**Convolution arithmetic and the conv as a matrix.** $\text{out} = \lfloor (n + 2p - d(k-1) - 1)/s \rfloor + 1$. A stack of layers sees $1 + \sum_\ell (k_\ell - 1) \prod_{m < \ell} s_m$ input samples. The correlation $y = Tx$ has the kernel along each row of $T$, shifted one column per row; $T^\top$ maps an output gradient to an input gradient, which is the full true convolution of the gradient with the kernel. im2col copies each pixel into every window that contains it; col2im adds those copies back (see [Convolution as a linear map](01-convolution-as-a-linear-map.md)).

**Sampling, aliasing, and interpolation.** A tone at $f$ sampled at $\text{sr}$ is indistinguishable from $|f - j\,\text{sr}|$ for any integer $j$; it appears at the representative in $[0, f_N]$. Keeping every $q$-th pixel divides the rate by $q$, so frequencies above $1/(2q)$ cycles per input pixel alias. Antialiased resizing stretches the kernel by the scale factor to remove them first (see [Resampling and interpolation](03-resampling-and-interpolation.md)).

**Fourier.** $X[k] = \sum_j x[j] e^{-2\pi i jk/n}$. $\sum |X|^2 = n \sum |x|^2$. Real input gives $X[n - k] = \overline{X[k]}$; real and even input gives a real spectrum. The product of DFTs is the DFT of the circular convolution; padding to length $\ge n_a + n_b - 1$ makes it linear. An FFT recursion splits $n$ by its factors (see [DFT and FFT](04-dft-and-fft.md)). Centered STFT frames number $1 + \lfloor n/\text{hop} \rfloor$, bins are $\text{sr}/n_\text{fft}$ apart, and a periodic Hann at hop $n_\text{fft}/4$ overlap-adds its square to $3/2$ (see [Sampling, Nyquist, windows, and the STFT](05-sampling-windows-and-stft.md)).

**Mel and decibels.** Slaney mel is linear to 15 mel at 1000 Hz, then 27 mel per factor 6.4. Power in dB is $10 \log_{10}$. Whisper's feature is $(\max(L, \max L - 8) + 4)/4$ with $L = \log_{10} \max(\text{mel}, 10^{-10})$ (see [Mel scale and filterbanks](06-mel-scale-and-filterbanks.md)).

**Contrastive losses.** CLIP scores a batch of $N$ image-text pairs with a similarity matrix and, for each image, classifies its own caption among the $N$ with a softmax: InfoNCE, $L = -\log \frac{e^{s_1}}{\sum_j e^{s_j}}$. Its gradient is softmax minus one-hot, and van den Oord et al. show $I(X; Y) \ge \log N - L$: more negatives can certify more mutual information. SigLIP replaces the softmax by an independent logistic loss per pair, $-\log \sigma(z(ts + b))$ with $z = +1$ on the diagonal and $-1$ elsewhere, so no normalizer couples the pairs.

**Edit distance and WER.** The word edit distance is the minimum number of substitutions, deletions, and insertions turning one word sequence into another, computed by a DP over prefixes. WER divides it by the number of reference words: insertions can push it above 1, and swapping reference and hypothesis changes the denominator.

## Five warm-up examples

Before the problems, here is one small example of each kind, worked in full.

**Conv.** $n = 10$, $k = 3$, $s = 2$, $p = 1$: $\lfloor (10 + 2 - 3)/2 \rfloor + 1 = 5$. The window starts sit at $-1, 1, 3, 5, 7$ in input coordinates, and the next one ($9$) would need input sample 11.

**Aliasing.** 9 kHz sampled at 16 kHz: $|9000 - 16000| = 7000$ Hz, below $f_N = 8000$, so it appears at 7 kHz.

**DFT.** $x = (1, -1, 1, -1)$: $X[k] = \sum_j (-1)^j (-i)^{jk}$; only $k = 2$ survives, $X = (0, 0, 4, 0)$, and Parseval gives $16 = 4 \cdot 4$.

**InfoNCE.** Two candidates with equal logits: $L = \log 2$; with the positive ahead by $\ln 3$, $L = \log(1 + 1/3) = \log(4/3)$.

**WER.** Reference "a b c", hypothesis "a x c d": one substitution and one insertion, WER $= 2/3$.

## The problems

Answers are exact (for example $9/16$, not 0.5625). The tag after each problem says what kind of answer is expected: a number, an expression, a vector or matrix, a true-or-false, a choice, or a proof. The problems run in the order of the chapter, then the two extra topics.

### Convolution arithmetic

**Problem 1.** Give the output length for (a) $n = 224$, $k = 7$, $s = 2$, $p = 3$ (a ResNet stem); (b) a $3 \times 3$ max pool with $s = 2$, $p = 1$ on that output, along one axis; (c) Whisper's stem on 3000 frames: a conv with $k = 3$, $p = 1$, $s = 1$, then one with $k = 3$, $p = 1$, $s = 2$; (d) $n = 32$, $k = 3$, $d = 4$, $p = 0$, $s = 1$. `[number]`

<details><summary>Answer</summary>

**(a) 112, (b) 56, (c) 1500, (d) 24.** Apply $\text{out} = \lfloor (n + 2p - d(k-1) - 1)/s \rfloor + 1$ each time. (a) $\lfloor (224 + 6 - 6 - 1)/2 \rfloor + 1 = \lfloor 223/2 \rfloor + 1 = 112$: the stem halves the image. (b) On $n = 112$: $\lfloor (112 + 2 - 2 - 1)/2 \rfloor + 1 = 55 + 1 = 56$. (c) The first conv keeps the length, $\lfloor (3000 + 2 - 3)/1 \rfloor + 1 = 3000$; the second gives $\lfloor 2999/2 \rfloor + 1 = 1500$, which is why Whisper's encoder has 1500 positions. (d) The dilated kernel spans $4 \cdot 2 + 1 = 9$ samples, so $32 - 9 + 1 = 24$. Common mistakes: dropping the $+1$ (111, 55, 1499); 1501 from rounding up instead of down; and 30 in (d), which sizes the dilated kernel as if it spanned 3 samples.

</details>

**Problem 2.** (a) How many input samples does a kernel of size $k$ with dilation $d$ span? Answer in $k$ and $d$. (b) For odd $k$, stride 1, and dilation 1, which padding $p$ keeps the output length equal to the input length? Answer in $k$. `[expr]`

<details><summary>Answer</summary>

**(a) $d(k - 1) + 1$, (b) $(k - 1)/2$.** (a) The $k$ taps are $d$ apart, so there are $k - 1$ gaps of $d$ between the first and the last tap, plus the first sample itself. (b) With $s = d = 1$ the formula gives $n + 2p - k + 1$; setting it equal to $n$ gives $2p = k - 1$, an integer for odd $k$ ("same" padding: 1 for $3 \times 3$, 3 for $7 \times 7$). A common mistake is $dk$ in (a), which counts a gap after the last tap, and $k/2$ in (b), which is not even an integer for odd $k$.

</details>

**Problem 3.** The receptive field of an output is the number of input samples (along one axis) it depends on. Give it for (a) three stacked $3 \times 3$ convs with stride 1; (b) a $3 \times 3$ conv with stride 2 followed by a $3 \times 3$ conv with stride 1. `[number]`

<details><summary>Answer</summary>

**(a) 7, (b) 7.** Use $1 + \sum_\ell (k_\ell - 1) \prod_{m < \ell} s_m$. (a) Three layers of $k = 3$ at stride 1: $1 + 2 + 2 + 2 = 7$. (b) The first layer adds $k - 1 = 2$; the second also adds 2, but each of its steps is 2 input samples apart because of the first layer's stride, so it adds $2 \cdot 2 = 4$: $1 + 2 + 4 = 7$. A common mistake in (a) is $3 \cdot 3 = 9$, as if the kernels did not overlap; in (b), 5 forgets that the second layer's steps are scaled by the first layer's stride.

</details>

### Correlation, convolution, and the conv as a matrix

**Problem 4.** $x = (1, 2, 3, 4)$ and $w = (1, 0, -1)$, no padding. Give (a) the cross-correlation $y[i] = \sum_r w[r]\, x[i + r]$ and (b) the true convolution (kernel flipped). `[vector]`

<details><summary>Answer</summary>

**(a) $(-2, -2)$, (b) $(2, 2)$.** There are $4 - 3 + 1 = 2$ outputs. (a) $y[0] = 1 \cdot 1 + 0 \cdot 2 - 1 \cdot 3 = -2$, $y[1] = 2 - 4 = -2$. (b) The true convolution slides the flipped kernel $(-1, 0, 1)$: $-1 + 3 = 2$ and $-2 + 4 = 2$. Neural-network "convs" are cross-correlations; confusing the two swaps the answers.

</details>

**Problem 5.** For the cross-correlation of Problem 4 written as $y = T x$: (a) give $T$. `[matrix]` (b) Give the input gradient $T^\top g$ for the upstream gradient $g = (1, 1)$. `[vector]` (c) Is $T^\top g$ equal to the full (zero-padded) true convolution of $g$ with $w$? `[bool]`

<details><summary>Answer</summary>

**(a) $\begin{pmatrix} 1 & 0 & -1 & 0 \\ 0 & 1 & 0 & -1 \end{pmatrix}$, (b) $(1, 1, -1, -1)$, (c) true.** (a) Row $i$ holds the kernel $w$ starting at column $i$: a Toeplitz matrix. (b) $T^\top g$ is the sum of the rows weighted by $g$: $(1, 0, -1, 0) + (0, 1, 0, -1) = (1, 1, -1, -1)$; it has the input's length 4, not the output's. (c) The full true convolution of $g = (1, 1)$ with $w = (1, 0, -1)$ is $(1, 1 + 0, 0 - 1, -1) = (1, 1, -1, -1)$, the same vector: the backward of a correlation is a true convolution of the gradient with the kernel. A common mistake in (a) is the flipped kernel in each row, which is the matrix of the true convolution.

</details>

**Problem 6.** (a) With $x$ and $g$ as in Problems 4 and 5, give the weight gradient $\partial L / \partial w[r] = \sum_i g[i]\, x[i + r]$. `[vector]` (b) A $1 \times 1 \times 3 \times 3$ image goes through im2col with a $2 \times 2$ kernel, stride 1, no padding. How many times is its center pixel copied into the columns? `[number]`

<details><summary>Answer</summary>

**(a) $(3, 5, 7)$, (b) 4.** (a) $\partial L/\partial w[r] = g[0] x[r] + g[1] x[r + 1]$: $1 + 2 = 3$, $2 + 3 = 5$, $3 + 4 = 7$. It is the correlation of the input with the gradient; reversing it to $(7, 5, 3)$ is a common mistake. (b) A $2 \times 2$ kernel on a $3 \times 3$ image has $2 \times 2 = 4$ windows, and the center pixel lies in all four, so im2col copies it four times (a corner pixel only once).

</details>

**Problem 7.** Prove that col2im is the adjoint of im2col: for every input $x$ and every column array $Y$ of the matching shape, $\langle \operatorname{im2col}(x), Y \rangle = \langle x, \operatorname{col2im}(Y) \rangle$, where col2im adds every column entry back to the pixel it was copied from and drops entries copied from the zero padding. `[proof]`

<details><summary>Answer</summary>

**Claim.** For every input $x$ and every column array $Y$ of the matching shape, $\langle \operatorname{im2col}(x), Y \rangle = \langle x, \operatorname{col2im}(Y) \rangle$.

**Proof.** Index column entries by $\alpha = (n, c, r, s, p, q)$. im2col copies one value into each entry: with padding $(p_h, p_w)$, stride $(s_h, s_w)$, and dilation $(d_h, d_w)$, entry $\alpha$ reads the input position $\pi(\alpha) = (n, c, p s_h + r d_h - p_h, q s_w + s d_w - p_w)$, and is 0 when that position falls in the padding. Let $A$ be the set of entries whose $\pi(\alpha)$ is a real pixel. Then
$$\langle \operatorname{im2col}(x), Y \rangle = \sum_{\alpha \in A} x_{\pi(\alpha)} Y_\alpha .$$
Every $\alpha \in A$ reads exactly one pixel, so grouping the terms by the pixel $u = \pi(\alpha)$ partitions $A$:
$$\sum_{\alpha \in A} x_{\pi(\alpha)} Y_\alpha = \sum_u x_u \Big( \sum_{\alpha \in A,\ \pi(\alpha) = u} Y_\alpha \Big).$$
The inner sum is col2im's definition: every column entry goes back to the pixel it was copied from, entries landing on the same pixel are added, and entries copied from the padding are dropped. So it equals $\operatorname{col2im}(Y)_u$, and the sum is $\langle x, \operatorname{col2im}(Y) \rangle$. Since this holds for all $x$ and $Y$, col2im is the adjoint (the transpose) of the linear map im2col.

A complete proof writes im2col as an index map with the padding case, expands the inner product, regroups by input pixel, and identifies the regrouped coefficient as col2im. A col2im that assigns instead of adding keeps one term of each inner sum and breaks the identity wherever windows overlap. Model proof also in [`q7.md`](../../../../course/solutions/solve/S-M12/q7.md).

</details>

### Parameter and FLOP counts

**Problem 8.** A conv from 64 to 128 channels with a $3 \times 3$ kernel, groups 1, and a bias. Give (a) its parameter count; (b) the parameter count of a depthwise $3 \times 3$ conv on 64 channels with a bias; (c) the multiply-accumulates of the first conv for one image with a $56 \times 56$ output. `[number]`

<details><summary>Answer</summary>

**(a) 73856, (b) 640, (c) 231211008.** (a) Weights $64 \cdot 128 \cdot 3 \cdot 3 = 73728$, plus 128 biases. (b) Depthwise means one $3 \times 3$ filter per channel: $64 \cdot 9 + 64 = 640$. (c) Each of the $128 \cdot 56 \cdot 56 = 401408$ outputs takes $64 \cdot 9 = 576$ multiply-accumulates: $401408 \cdot 576 = 231211008$ (the bias is an add, not a MAC). Common mistakes: 73728 forgets the bias; 36928 in (b) is a full $64 \to 64$ conv; 462422016 in (c) counts FLOPs (two per MAC) where MACs were asked.

</details>

**Problem 9.** ViT-B/16 on a $3 \times 224 \times 224$ image with width 768. Give (a) the number of patches; (b) the parameters of the patch embedding (a conv with kernel and stride 16, with a bias); (c) the sequence length after prepending a CLS token. `[number]`

<details><summary>Answer</summary>

**(a) 196, (b) 590592, (c) 197.** (a) $224/16 = 14$ patches per side, $14^2 = 196$. (b) The patch embedding is a conv from 3 to 768 channels with a $16 \times 16$ kernel: $3 \cdot 16 \cdot 16 \cdot 768 = 589824$ weights plus 768 biases. (c) $196 + 1 = 197$ with the CLS token. Common mistakes: 197 in (a) (the CLS token is not a patch), 589824 in (b) (no bias), and 196 in (c) (CLS forgotten).

</details>

### Layouts and strides

**Problem 10.** A float32 tensor with logical shape $(N, C, H, W) = (2, 3, 4, 5)$. (a) Give its byte strides per logical axis when stored NCHW, and (b) when stored NHWC (channels last). `[vector]` (c) Give the byte offset of element $(1, 2, 3, 4)$ in the NHWC storage. `[number]` (d) Give the length of one flattened patch when a $3 \times 224 \times 224$ image is cut into $14 \times 14$ patches. `[number]`

<details><summary>Answer</summary>

**(a) $(240, 80, 20, 4)$, (b) $(240, 4, 60, 12)$, (c) 476, (d) 588.** float32 is 4 bytes. (a) NCHW is row-major in $(N, C, H, W)$: element strides $(3 \cdot 4 \cdot 5, 4 \cdot 5, 5, 1) = (60, 20, 5, 1)$, times 4 bytes. (b) NHWC stores $(N, H, W, C)$ row-major: element strides $N{:}\ 60$, $H{:}\ 5 \cdot 3 = 15$, $W{:}\ 3$, $C{:}\ 1$; listed per logical axis $(N, C, H, W)$ and times 4 bytes, $(240, 4, 60, 12)$. (c) $1 \cdot 240 + 2 \cdot 4 + 3 \cdot 60 + 4 \cdot 12 = 476$ bytes. (d) A $14 \times 14$ patch over 3 channels flattens to $3 \cdot 14 \cdot 14 = 588$ values. Common mistakes: strides in elements $(60, 20, 5, 1)$; the NHWC strides in memory order $(240, 60, 12, 4)$ instead of per logical axis; and 196 in (d), which forgets the channels.

</details>

### Sampling and aliasing

**Problem 11.** (a) Give the Nyquist frequency in Hz at a 16 kHz sample rate. At what frequency in Hz does a pure tone appear when (b) 7 kHz is sampled at 10 kHz; (c) 12 kHz is sampled at 16 kHz; (d) 25 kHz is sampled at 16 kHz? `[number]`

<details><summary>Answer</summary>

**(a) 8000, (b) 3000, (c) 4000, (d) 7000.** (a) $f_N = 16000/2$. For the others, take the representative of $|f - j\,\text{sr}|$ in $[0, f_N]$. (b) $|7000 - 10000| = 3000 \le 5000$. (c) $|12000 - 16000| = 4000 \le 8000$. (d) $25000 - 16000 = 9000$ is still above 8000; the nearest multiple of the rate is $32000$, and $|25000 - 32000| = 7000$. A common mistake in (d) is 9 kHz, from folding once instead of to the nearest multiple of the rate, and in (b) and (c) reporting the original frequency, which is above Nyquist.

</details>

**Problem 12.** An image has a stripe pattern with a period of 3 pixels. It is shrunk by 4 on each axis by keeping every fourth pixel, with no filter. (a) Give the period in output pixels of the stripes that appear. `[number]` (b) Can the output grid represent the original pattern without aliasing? `[bool]` (c) Give the highest frequency, in cycles per input pixel, that a shrink by 4 can keep without aliasing. `[number]` (d) A 1-pixel checkerboard is shrunk by 2 by keeping every other pixel on each axis. The result is: (a) a checkerboard, (b) a constant image, (c) stripes. `[choice]`

<details><summary>Answer</summary>

**(a) 3, (b) false, (c) 1/8, (d) b.** (a) The stripes have frequency $1/3$ cycle per input pixel. Keeping every fourth pixel samples at 1 per 4 input pixels, i.e. the frequency becomes $4/3$ cycles per output pixel, which aliases to $|4/3 - 1| = 1/3$: stripes with a period of 3 output pixels (12 input pixels), a moiré. (b) The output grid's Nyquist frequency is $1/8$ cycle per input pixel, and $1/3 > 1/8$, so no. (c) That Nyquist limit: $\frac12 \cdot \frac14 = \frac18$. (d) The checkerboard is $(-1)^{i + j}$; keeping even $i$ and even $j$ makes $i + j$ always even, so every kept pixel has the same value: a constant image. Common mistakes: $3/4$ in (a) (the period divided by 4, as if the pattern were simply shrunk), $1/2$ or $1/4$ in (c) (the input's Nyquist, or the scale itself), and "a checkerboard" in (d).

</details>

### Interpolation kernels

**Problem 13.** Keys' cubic kernel is $k(t) = (a+2)|t|^3 - (a+3)|t|^2 + 1$ for $|t| < 1$ and $a|t|^3 - 5a|t|^2 + 8a|t| - 4a$ for $1 \le |t| < 2$. Give (a) $k(1/2)$ and (b) $k(3/2)$ with $a = -1/2$ (Pillow), and (c) $k(3/2)$ with $a = -3/4$ (PyTorch). `[number]`

<details><summary>Answer</summary>

**(a) $9/16$, (b) $-1/16$, (c) $-3/32$.** (a) $a = -1/2$, $|t| = 1/2$: $(3/2)(1/8) - (5/2)(1/4) + 1 = 3/16 - 10/16 + 16/16 = 9/16$. (b) The outer piece at $|t| = 3/2$ is $a\,(27/8 - 5 \cdot 9/4 + 8 \cdot 3/2 - 4) = a\,(27 - 90 + 96 - 32)/8 = a/8$, so $-1/16$. (c) The same with $a = -3/4$: $-3/32$. The negative lobe is what makes cubic sharper than bilinear. Common mistakes: $1/2$ in (a), the bilinear weight; $+1/16$ in (b), a sign slip; and $-1/16$ in (c), Pillow's $a$ where PyTorch's was asked.

</details>

**Problem 14.** Pillow shrinks with an antialiased kernel stretched by the scale factor. (a) Give the support radius in input pixels of the bilinear kernel (support 1) when shrinking 1000 pixels to 250. (b) Give the number of taps in Pillow's weight table for that pass, $2 \lceil \text{support} \rceil + 1$. (c) Give the support radius of Lanczos3 when shrinking by 2. `[number]`

<details><summary>Answer</summary>

**(a) 4, (b) 9, (c) 6.** Shrinking by a factor $\text{scale}$ stretches the kernel by $\text{scale}$ so that it also low-passes below the new Nyquist. (a) $1000/250 = 4$, so bilinear's support 1 becomes 4 input pixels. (b) $2 \lceil 4 \rceil + 1 = 9$ taps. (c) Lanczos3 has support 3; shrinking by 2 gives 6. A common mistake is the unstretched support (1, then 3 taps, and 3), which is resizing without antialiasing.

</details>

### The DFT

**Problem 15.** With $X[k] = \sum_j x[j] e^{-2\pi i jk/n}$, give the DFT of (a) $(1, 0, 0, 0)$, (b) $(1, 1, 1, 1)$, (c) $(0, 1, 0, 0)$. `[vector]` (d) A real signal of length 8 has $X[3] = 2 - i$. Give $X[5]$. `[number]`

<details><summary>Answer</summary>

**(a) $(1, 1, 1, 1)$, (b) $(4, 0, 0, 0)$, (c) $(1, -i, -1, i)$, (d) $2 + i$.** (a) An impulse at 0 has $X[k] = e^0 = 1$ for every $k$. (b) A constant has all its energy at $k = 0$: $X[0] = 4$, and the other sums of roots of unity are 0. With this unnormalized convention, $X[0]$ is the sum, not the mean. (c) An impulse at $j = 1$ gives $X[k] = e^{-2\pi i k/4} = (-i)^k$. (d) For real input $X[n - k] = \overline{X[k]}$, and $8 - 3 = 5$, so $X[5] = \overline{2 - i} = 2 + i$; this symmetry is why rfft keeps only $n/2 + 1$ bins. Common mistakes: $(1, i, -1, -i)$ in (c), the inverse transform's sign, and $2 - i$ in (d).

</details>

**Problem 16.** (a) Give $\sum_k |X[k]|^2$ for $x = (1, 2, 0, -1)$. (b) Give the circular convolution of length 4 of $(1, 2, 0, 0)$ and $(1, 1, 0, 0)$. `[vector]` (c) Give the smallest transform length for which an FFT product computes the linear convolution of signals of lengths 300 and 100. `[number]` (d) Is the DFT of a real, even signal ($x[j] = x[(n - j) \bmod n]$) real? `[bool]`

<details><summary>Answer</summary>

**(a) 24, (b) $(1, 3, 2, 0)$, (c) 399, (d) true.** (a) Parseval: $\sum |X|^2 = n \sum |x|^2 = 4 (1 + 4 + 0 + 1) = 24$. (b) The linear convolution is $(1, 3, 2)$ and has length $3 \le 4$, so nothing wraps around: $(1, 3, 2, 0)$. (c) The linear convolution has length $300 + 100 - 1 = 399$, and the circular one equals it when the transform length is at least that. (d) Yes: an even real signal has $X[k] = \sum_j x[j] \cos(2\pi jk/n)$, because the sine terms of $j$ and $n - j$ cancel. Common mistakes: 6 in (a) (Parseval without the factor $n$ of this convention), a length-5 vector in (b), and 300 in (c).

</details>

### The FFT

**Problem 17.** (a) Into how many radix stages does an FFT split $n = 400$ when it uses radices 4, 4, 5, 5? (b) How many complex multiplies does the direct DFT of length 400 take ($n^2$)? (c) How many butterflies does a radix-2 FFT of length 1024 take, $(n/2) \log_2 n$? `[number]` (d) Can a radix-2-only FFT compute a 400-point DFT, without changing the bin frequencies? `[bool]`

<details><summary>Answer</summary>

**(a) 4, (b) 160000, (c) 5120, (d) false.** (a) $400 = 4 \cdot 4 \cdot 5 \cdot 5$, one stage per radix. (b) $400^2 = 160000$. (c) $(1024/2) \cdot 10 = 5120$. (d) A radix-2 FFT needs a power of 2; padding 400 to 512 computes a different transform, with bins $\text{sr}/512$ apart instead of $\text{sr}/400$, which is why Whisper's 400-point frame needs mixed radices. Common mistakes: 6 in (a) (the prime factorization $2^4 \cdot 5^2$, six stages, is a valid but different plan; the question names radices 4, 4, 5, 5); 400 in (b); 10240 in (c) ($n \log_2 n$ instead of half of it).

</details>

### Windows and the STFT

**Problem 18.** Whisper: 30 s of audio at 16 kHz, $n_\text{fft} = 400$, hop 160, centered frames. Give (a) the number of STFT frames before Whisper drops the last one; (b) the spacing of the bins in Hz; (c) the number of rfft bins per frame; (d) the constant value of $\sum_t w[m - t\,\text{hop}]^2$ for a periodic Hann window at hop $n_\text{fft}/4$; (e) the frame length in milliseconds. `[number]` (f) Give the periodic Hann window of length 4. `[vector]`

<details><summary>Answer</summary>

**(a) 3001, (b) 40, (c) 201, (d) 3/2, (e) 25, (f) $(0, 1/2, 1, 1/2)$.** (a) $30 \cdot 16000 = 480000$ samples; centered framing pads $n_\text{fft}/2$ on each side and gives $1 + \lfloor 480000/160 \rfloor = 3001$ frames, and Whisper drops the last to get 3000. (b) $16000/400 = 40$ Hz. (c) $400/2 + 1 = 201$. (d) The periodic Hann is $w[m] = \sin^2(\pi m/n_\text{fft})$, so $w^2 = \sin^4$, whose mean over a period is $3/8$; overlap-adding at hop $n_\text{fft}/4$ stacks 4 copies, $4 \cdot 3/8 = 3/2$. (e) $400/16000$ s $= 25$ ms. (f) $w[m] = \frac12 - \frac12 \cos(2\pi m/4)$: $(0, 1/2, 1, 1/2)$. Common mistakes: 3000 in (a) (the centered padding's extra frame forgotten), 80 Hz in (b), 200 bins, 2 in (d) (the sum of the window, which is what plain overlap-add needs, not of its square), 10 ms in (e) (the hop), and the symmetric window $(0, 3/4, 3/4, 0)$ in (f).

</details>

### Mel and decibels

**Problem 19.** On the Slaney scale, give the mel value of (a) 1000 Hz and (b) 6400 Hz. (c) On the HTK scale, $m(f) = 2595 \log_{10}(1 + f / 700)$: give $m(700)$. (d) Give a power ratio of 1000 in decibels. (e) Whisper computes $L = \log_{10}(\max(\text{mel}, 10^{-10}))$, then $\max(L, \max L - 8)$, then $(L + 4)/4$. Give the feature of a band with mel power $10^{-12}$ in a clip whose largest $L$ is 1. (f) Give the feature value of pure silence. `[number]`

<details><summary>Answer</summary>

**(a) 15, (b) 42, (c) $2595 \log_{10} 2$, (d) 30, (e) $-3/4$, (f) $-3/2$.** (a) $1000 \cdot 3/200 = 15$, the break point. (b) $6400 = 1000 \cdot 6.4$, one factor of 6.4 above the break: $15 + 27 = 42$. (c) $2595 \log_{10}(1 + 1) \approx 781.17$. (d) $10 \log_{10} 1000 = 30$ dB. (e) $L = \log_{10} \max(10^{-12}, 10^{-10}) = -10$; the clip raises it to $\max(-10, 1 - 8) = -7$; $(-7 + 4)/4 = -3/4$. (f) Silence floors everywhere at $-10$, the maximum is also $-10$, the clip does nothing, and $(-10 + 4)/4 = -3/2$. Common mistakes: 96 in (b) (the linear piece used above 1 kHz), $2595 \ln 2$ in (c), 60 in (d) (the $20\log_{10}$ amplitude rule on a power), $-3/2$ in (e) (the floor applied but not the clip), and $-10$ in (f) (the log without the rescale).

</details>

### Contrastive losses

**Problem 20.** InfoNCE for one anchor scores $N$ candidates with logits $s_j$ (the positive is $j = 1$) and takes $L = -\log \frac{e^{s_1}}{\sum_j e^{s_j}}$, the cross-entropy with the positive as the label. (a) Give $L$ for $s = (2, 0, 0, 0)$. (b) Give $L$ when all $N$ logits are equal. Answer in $N$. `[expr]` (c) The InfoNCE bound says $I(X; Y) \ge \log N - L$. Give the bound in nats for $N = 1024$ and $L = 2$. `[number]` (d) SigLIP's loss for one pair is $\ell = -\log \sigma\big(z (t s + b)\big)$ with label $z = \pm 1$. Give $\partial \ell / \partial s$ for a positive pair ($z = 1$) with $t = 10$, $b = -10$, $s = 1$. `[number]`

<details><summary>Answer</summary>

**(a) $\log(1 + 3e^{-2})$, (b) $\log N$, (c) $\ln 1024 - 2 \approx 4.93$ nats, (d) $-5$.** (a) $L = -\log \frac{e^2}{e^2 + 3} = \log \frac{e^2 + 3}{e^2} = \log(1 + 3e^{-2})$. (b) Equal logits give the positive probability $1/N$, so $L = \log N$: the loss of a model that knows nothing. (c) $\log N$ in nats is $\ln 1024 = 10 \ln 2 \approx 6.93$, minus 2. (d) $u = z(ts + b) = 10 - 10 = 0$, and $\partial \ell/\partial s = -tz\,\sigma(-u) = -10 \cdot \frac12 = -5$ (Problem 22). Common mistakes: $\log(3e^{-2})$ in (a) (the positive left out of the denominator), $0$ or $\log(N - 1)$ in (b), $10 - 2$ in (c) ($\log_2$ mixed into a bound in nats), and $-1/2$ in (d) (the temperature $t$ dropped from the chain rule).

</details>

**Problem 21.** Prove that the gradient of the InfoNCE loss of Problem 20 with respect to the logits is $\partial L / \partial s_j = \operatorname{softmax}(s)_j - [j = 1]$, and conclude that $L > 0$ for every finite $s$ and that $L \to 0$ only when $s_1 - s_j \to \infty$ for every $j \ne 1$. `[proof]`

<details><summary>Answer</summary>

**Claim.** For $L(s) = -\log \frac{e^{s_1}}{\sum_k e^{s_k}}$ with $N \ge 2$ finite logits, $\partial L/\partial s_j = \operatorname{softmax}(s)_j - [j = 1]$; $L > 0$; and $L \to 0$ exactly when $s_1 - s_j \to \infty$ for every $j \ne 1$.

**Proof.** Write $L = -s_1 + \log \sum_k e^{s_k}$. The first term contributes $-[j = 1]$; by the chain rule $\frac{\partial}{\partial s_j} \log \sum_k e^{s_k} = e^{s_j}/\sum_k e^{s_k} = \operatorname{softmax}(s)_j$. So $\partial L/\partial s_j = \operatorname{softmax}(s)_j - [j = 1]$: prediction minus one-hot label, the cross-entropy gradient with the positive as the label.

Positivity: $L = -\log \operatorname{softmax}(s)_1$, and for finite logits every $e^{s_k} > 0$, so with $N \ge 2$, $\operatorname{softmax}(s)_1 < 1$ and $L > 0$.

When the loss vanishes: dividing numerator and denominator by $e^{s_1}$ gives $L = \log\big(1 + \sum_{j \ne 1} e^{s_j - s_1}\big)$. The sum has positive terms, so $L \to 0$ iff every $e^{s_j - s_1} \to 0$ iff $s_1 - s_j \to \infty$ for every $j \ne 1$.

A complete proof writes the loss as log-sum-exp, differentiates both terms, shows strict positivity from $\operatorname{softmax}(s)_1 < 1$, and rewrites $L$ to read off the limit. Model proof also in [`q21.md`](../../../../course/solutions/solve/S-M12/q21.md).

</details>

**Problem 22.** Prove that for SigLIP's loss $\ell = \log\big(1 + e^{-z(t s + b)}\big)$, $\partial \ell / \partial s = -t z\, \sigma\big(-z(t s + b)\big)$, and explain why, unlike Problem 21, the gradient of one pair does not depend on the logits of the other pairs in the batch. `[proof]`

<details><summary>Answer</summary>

**Claim.** For $\ell(s) = \log(1 + e^{-z(ts + b)})$ with $z \in \{1, -1\}$, $t > 0$, $\partial \ell/\partial s = -tz\,\sigma(-z(ts + b))$, and a pair's gradient involves no other pair's logit.

**Proof.** Let $u = z(ts + b)$, so $\ell = \log(1 + e^{-u})$ and $du/ds = zt$. Then $\frac{d\ell}{du} = \frac{-e^{-u}}{1 + e^{-u}} = -\frac{1}{e^u + 1} = -\sigma(-u)$. By the chain rule, $\partial \ell/\partial s = -tz\,\sigma(-z(ts + b))$. (Since $-\log \sigma(u) = \log(1 + e^{-u})$, this is the loss as SigLIP writes it.)

Independence: SigLIP's batch loss is $\sum_{i,j} \ell(s_{ij}, z_{ij})$, and term $(i, j)$ depends only on $s_{ij}$, $z_{ij}$, $t$, and $b$. Differentiating with respect to $s_{ij}$ leaves exactly that one term. In InfoNCE (Problem 21) the softmax normalizer $\sum_k e^{s_k}$ couples every logit of a row, so each gradient depends on all of them; that is why softmax-based contrastive losses need the whole similarity row on one device, and the sigmoid loss shards cheaply.

A complete proof sets up $u$, differentiates the softplus, applies the chain rule, notes the loss is a sum of independent per-pair terms, and contrasts it with the softmax normalizer. Model proof also in [`q22.md`](../../../../course/solutions/solve/S-M12/q22.md).

</details>

### Word error rate

**Problem 23.** The word error rate is $\text{WER} = (S + D + I) / N_\text{ref}$, the minimum number of substitutions, deletions, and insertions turning the reference into the hypothesis, over the number of reference words. Give the WER of the hypothesis "oh hello there" against the reference "hello". `[number]`

<details><summary>Answer</summary>

**2.** The reference has one word; the hypothesis matches "hello" and adds two words, so $S = D = 0$, $I = 2$, and WER $= 2/1 = 2$. WER can exceed 1 because insertions are counted against the reference length. A common mistake is $2/3$, which divides by the hypothesis length.

</details>

**Problem 24.** Prove that WER is not a metric on word sequences, by showing it is not symmetric with an example (compute both directions), and state which metric axiom fails and why. `[proof]`

<details><summary>Answer</summary>

**Claim.** $\text{WER}(r, h) = d(r, h)/|r|$, with $d$ the word edit distance and $|r|$ the reference length, is not a metric, because it is not symmetric.

**Proof.** Take $a = $ "the cat" and $b = $ "the". With $a$ as reference and $b$ as hypothesis, the cheapest alignment matches "the" and deletes "cat": $d = 1$, WER $= 1/2$. With $b$ as reference and $a$ as hypothesis, match "the" and insert "cat": $d = 1$ (no alignment costs 0, since the lengths differ), WER $= 1/1 = 1$. So $\text{WER}(a, b) = 1/2 \ne 1 = \text{WER}(b, a)$, and the symmetry axiom $d(x, y) = d(y, x)$ fails.

The cause is the normalizer: the edit distance itself is symmetric (an insertion one way is a deletion the other, and substitutions are symmetric), so the numerator is the same both ways, but the denominator is the length of whichever sequence plays the reference. The same normalization is why WER can exceed 1 (Problem 23).

A complete proof computes both directions of one example, names symmetry as the failing axiom, and locates the cause in the denominator. Model proof also in [`q24.md`](../../../../course/solutions/solve/S-M12/q24.md).

</details>

## Common mistakes

The wrong answers in this set cluster into a few patterns, each tied to the problem where it shows up.

| Mistake | Symptom | How to avoid it |
|---|---|---|
| Dropping the $+1$ in the size formula | 1499 positions where Whisper's encoder has 1500 | apply the full formula with its $+1$ (Problem 1) |
| Ignoring dilation in the kernel span | a dilated layer sized like a dense one | size the kernel as $d(k - 1) + 1$ (Problem 1(d), Problem 2) |
| Flipping the kernel in a network conv | correlation and convolution swapped | keep the kernel unflipped in each Toeplitz row (Problems 4 and 5) |
| Counting FLOPs where MACs were asked (or the reverse) | a factor 2 in every cost estimate | count one MAC per weight per output (Problem 8(c)) |
| Forgetting the CLS token | 196 positions where the encoder has 197 | add the CLS token after counting patches (Problem 9(c)) |
| Strides in elements, or in memory order instead of per logical axis | a kernel indexing the wrong neighbour | multiply by the itemsize and list strides per logical axis (Problem 10) |
| Folding once instead of to the nearest multiple of the rate | 9 kHz instead of 7 kHz | fold to the representative in $[0, f_N]$ (Problem 11(d)) |
| The bilinear weight where the cubic was asked, or the wrong $a$ | interpolation that does not match Pillow or PyTorch | evaluate Keys' kernel with the library's $a$ (Problem 13) |
| Forgetting the antialias stretch | support 1 where Pillow uses 4 | stretch the support by the scale factor (Problem 14) |
| The inverse DFT's sign | a mirrored spectrum | use $e^{-2\pi i jk/n}$ for the forward transform (Problem 15(c)) |
| Treating 400 as $2^4 \cdot 5^2$ stages | six radix-2 and radix-5 stages instead of four | count one stage per named radix (Problem 17(a)) |
| Counting frames without the centered padding's extra frame | 3000 where Whisper computes 3001 | use $1 + \lfloor n/\text{hop} \rfloor$ for centered frames (Problem 18(a)) |
| $\sum w$ instead of $\sum w^2$ for weighted overlap-add | reconstructions scaled by 3/4 | overlap-add the squared window (Problem 18(d)) |
| The symmetric Hann window | $(0, 3/4, 3/4, 0)$ | divide by $n$, not $n - 1$, in the cosine (Problem 18(f)) |
| 20 $\log_{10}$ for a power | decibels doubled | use $10 \log_{10}$ for power (Problem 19(d)) |
| Applying only the floor in Whisper's scaling | $-3/2$ where the 8-unit clip gives $-3/4$ | floor, then clip at $\max L - 8$, then rescale (Problem 19(e)) |
| Mixing $\log_2$ into a nats bound | $10 - 2$ instead of $\ln 1024 - 2$ | keep every logarithm natural (Problem 20(c)) |
| Dividing WER by the hypothesis length | $2/3$ where the answer is 2 | divide by the reference length (Problem 23) |

## Where this leads

Each group of problems checks one section of this chapter: conv sizes, the Toeplitz form, im2col and col2im (Problems 1 to 9) check [Convolution as a linear map](01-convolution-as-a-linear-map.md); strides and patches (Problem 10) check [Image tensors and layouts](02-image-tensors-and-layouts.md); aliasing, interpolation kernels, and antialias support (Problems 11 to 14) check [Resampling and interpolation](03-resampling-and-interpolation.md); DFT properties and FFT counts (Problems 15 to 17) check [DFT and FFT](04-dft-and-fft.md); and frames, windows, mel, and Whisper's scaling (Problems 18 and 19) check [Sampling, Nyquist, windows, and the STFT](05-sampling-windows-and-stft.md) and [Mel scale and filterbanks](06-mel-scale-and-filterbanks.md).

The same arithmetic carries forward into *Build Your Own LLM System*. [Conv layers and pooling on autograd](../../build-an-llm/14-vision/01-conv-layers-and-pooling.md) and the [Vision transformer encoder](../../build-an-llm/14-vision/03-vision-transformer.md) build conv layers and the ViT patch embedding; [Contrastive image-text: CLIP and SigLIP](../../build-an-llm/14-vision/05-contrastive-image-text.md) trains with the losses of Problems 20 to 22; and the [audio frontend](../../build-an-llm/15-speech/01-audio-frontend.md), the [Whisper encoder](../../build-an-llm/15-speech/02-whisper-encoder.md), and [Multimodal eval metrics](../../build-an-llm/15-speech/05-multimodal-eval-metrics.md) use the frames and features of Problems 18 and 19 and report the WER of Problems 23 and 24.

### Further reading

- Dumoulin and Visin, "A guide to convolution arithmetic for deep learning" (free).
- Oppenheim and Schafer, *Discrete-Time Signal Processing*, ch. 4 and 8.
- van den Oord, Li, Vinyals, "Representation learning with contrastive predictive coding" (2018).
- Zhai et al., "Sigmoid loss for language image pre-training" (ICCV 2023).
- Jurafsky and Martin, *Speech and Language Processing*, ch. 2.5 (edit distance, free).

## Summary

- One formula sizes every conv, pool, and patch embedding, and receptive fields grow by $(k - 1)$ times the product of earlier strides (Problems 1 to 3).
- A conv is a Toeplitz matrix; its transpose (col2im) is the backward pass, and the backward of a correlation is a true convolution (Problems 4 to 7).
- Sampling keeps only frequencies below half the rate; everything above folds back, in time and in images (Problems 11 and 12).
- The DFT is a unitary change of basis: Parseval, conjugate symmetry for real signals, and the convolution theorem with zero padding (Problems 15 and 16); an FFT needs radices that divide $n$ (Problem 17).
- InfoNCE is cross-entropy with the positive as the label, bounded below by $\log N - I$; SigLIP's sigmoid loss decouples the pairs (Problems 20 to 22). WER is an edit distance over the reference length: it can exceed 1 and is not symmetric (Problems 23 and 24).
