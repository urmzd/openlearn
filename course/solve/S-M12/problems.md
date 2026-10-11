# S-M12 problems: convolution, layouts, sampling, interpolation, Fourier, STFT, mel, contrastive losses, WER

Answer every question in `solve/S-M12.toml` (written by `ol start S-M12`).
The tag after each question is its answer type: `[number]` is an exact value
(`3/4`, `log(1024) - 2`, `2 + I`), `[expr]` a formula in the named variables,
`[vector]` a flat list `[1, 2, 3]`, `[matrix]` a list of rows
`[[1, 0], [0, 1]]`, `[bool]` `true` or `false`, `[choice]` the letter of one
option, and `[proof]` a file `solve/S-M12/qN.md` graded against its rubric.
Write the imaginary unit as `I`, the natural log as `log(x)`, and $\log_b x$
as `log(x, b)`. Lettered parts are answered separately (`[q1.a]`, `[q1.b]`).
"Convolution" in a network means cross-correlation (no kernel flip) unless a
question says "true convolution". Sizes use
$\text{out} = \lfloor (n + 2p - d(k-1) - 1)/s \rfloor + 1$.

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
