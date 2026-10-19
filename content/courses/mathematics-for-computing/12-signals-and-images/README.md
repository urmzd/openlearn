# Signals and Images

Convolution, resampling, the Fourier transform, and the mel scale: the numerical front end of every vision and speech model, built as numpy reference code the multimodal passes call.

## Overview

- **Primary textbook**: Oppenheim and Schafer, *Discrete-Time Signal Processing* (3rd ed.); free alternative: Smith, *Mathematics of the Discrete Fourier Transform* and *Spectral Audio Signal Processing* (free online, ccrma.stanford.edu)
- **Supplementary**: Dumoulin and Visin, "A guide to convolution arithmetic for deep learning" (free); Goodfellow, Bengio, Courville, *Deep Learning*, ch. 9 (free); Szeliski, *Computer Vision: Algorithms and Applications*, ch. 3 (free)
- **Prerequisites**: [Precalculus](../00-precalculus/) (logs, Euler's formula), [Linear Algebra](../03-linear-algebra/), [Matrix Calculus & Autodiff](../08-matrix-calculus-and-autodiff/), [Numerical Methods & Floating Point](../09-numerical-methods-and-floating-point/)
- **Estimated time**: 2 weeks at 8-10 hrs/week (the six course modules plus the problem set)

## Key Takeaways

- A convolution layer is a linear map, a Toeplitz matrix; im2col turns it into one matmul and col2im (its adjoint) gives the backward pass
- A layout is a memory order: the same logical NCHW tensor can store its channels first or last, and the strides say which
- Sampling keeps only frequencies below half the rate; resizing and resampling must low-pass first or they alias
- The DFT is a unitary change of basis, and an FFT computes it in $O(n \log n)$ for every size whose factors it supports, including Whisper's 400
- Speech features are a windowed STFT, pooled into mel bands, on a log scale; each step has a convention a checkpoint was trained on

## How to Study

- Work each chapter's worked example by hand before reading its code; the first course test is that example
- Reproduce the reference library's numbers, not just its idea: Pillow, PyTorch, torchaudio, numpy, and librosa each make small choices (pass order, rounding, padding, window symmetry) that matter for parity
- Finish with the problem set: it is where the arithmetic of `L13` and `L14` becomes second nature

---

# Concepts & Techniques

## Core Insight

Images and audio are samples of continuous signals. Every operation in a multimodal front end (convolve, resize, transform, filter, log) is a linear map followed at most by a pointwise nonlinearity, and getting its conventions exactly right is what makes a reimplementation agree with the model's training pipeline.

## 1. Convolution as a Linear Map

**Key ideas**:
- Deep learning's convolution is cross-correlation; true convolution flips the kernel
- Output size $\lfloor (n + 2p - d(k-1) - 1)/s \rfloor + 1$; groups split channels into independent convs
- im2col and col2im: forward as one matmul, backward as the adjoint copy

## 2. Image Tensors and Layouts

**Key ideas**:
- NCHW versus NHWC strides; a transpose is a view, a layout change is a copy
- Rescale and normalize to float32 CHW; Pillow's fixed-point grayscale
- Patchify as a reshape: a ViT patch embedding is a stride-$p$ conv

## 3. Resampling and Interpolation

**Key ideas**:
- Nearest, bilinear, bicubic (Keys, $a = -0.5$ or $-0.75$), Lanczos3
- Antialiasing stretches the kernel by the scale factor
- Separable fixed-point passes reproduce Pillow; windowed-sinc filter banks reproduce torchaudio

## 4. The DFT and the FFT

**Key ideas**:
- Unitarity and Parseval; conjugate symmetry of real signals
- The convolution theorem and zero padding
- Mixed-radix Cooley-Tukey; real FFTs by packing; the DCT-II via an FFT

## 5. Windows and the STFT

**Key ideas**:
- Nyquist and aliasing; windows and spectral leakage
- Center reflect padding and frame counts; COLA and weighted overlap-add

## 6. Mel Scale and Decibels

**Key ideas**:
- Slaney and HTK mel scales; triangular filterbanks with area normalization
- Decibels, `top_db`, and Whisper's log10 with an 8-unit range

---

## Technique Catalog

| Concept | Key Formula | Application |
|---------|-------------|-------------|
| Conv output size | $\lfloor (n + 2p - d(k-1) - 1)/s \rfloor + 1$ | sizing every conv, pool, and patch layer |
| im2col matmul | $Y = W_\text{flat} \cdot \operatorname{im2col}(x)$ | fast conv forward; col2im for backward |
| Antialiased resize | kernel stretched by $\max(\text{scale}, 1)$ | image preprocessing without moire |
| DFT | $X[k] = \sum_j x[j] e^{-2\pi i jk/n}$ | spectra, convolution, DCT |
| STFT frames | $1 + \lfloor n/\text{hop} \rfloor$ (centered) | Whisper's 3001 frames |
| Slaney mel | $15 + 27 \log_{6.4}(f/1000)$ above 1 kHz | speech features |

## Course modules

This topic is new in Pass 12 (the multimodal pass): no earlier topic owns sampling, the DFT, or interpolation. Each build module is numpy reference code with a call site in the vision (`L13`) or speech (`L14`) parts; optional C kernels for the same math are `L13.8`, `L13.9`, and `L14.6`.

| Module | Topic | Kind | Pass |
|---|---|---|---|
| `M12.1` | Convolution as a linear map | build | 12 |
| `M12.2` | Image tensors and layouts | build | 12 |
| `M12.3` | Resampling and interpolation | build | 12 |
| `M12.4` | DFT and FFT | build | 12 |
| `M12.5` | Sampling, Nyquist, windows, and the STFT | build | 12 |
| `M12.6` | Mel scale and filterbanks | build | 12 |
| `S-M12` | Signals and images problem set | solve | 12 |

## Chapters

| # | Module | Chapter | Kind | Pass |
|---|---|---|---|---|
| 1 | `M12.1` | [Convolution as a linear map](01-convolution-as-a-linear-map.md) | build | 12 |
| 2 | `M12.2` | [Image tensors and layouts](02-image-tensors-and-layouts.md) | build | 12 |
| 3 | `M12.3` | [Resampling and interpolation](03-resampling-and-interpolation.md) | build | 12 |
| 4 | `M12.4` | [DFT and FFT](04-dft-and-fft.md) | build | 12 |
| 5 | `M12.5` | [Sampling, Nyquist, windows, and the STFT](05-sampling-windows-and-stft.md) | build | 12 |
| 6 | `M12.6` | [Mel scale and filterbanks](06-mel-scale-and-filterbanks.md) | build | 12 |
| 7 | `S-M12` | [Signals and images problem set: convolution, sampling, Fourier, mel, contrastive losses, and WER](90-problem-set.md) | solve | 12 |

## Connections to Other Tracks

| Signals Concept | Connected Track | Application |
|--------------------|-----------------|-------------|
| Convolution, im2col | [Neural Architectures](../../machine-learning/06-neural-architectures/) | CNN layers and their C kernels |
| FFT, DCT | [Information Theory](../11-information-theory/) | transform coding and rate-distortion |
| Aliasing and resampling | [Numerical Methods & Floating Point](../09-numerical-methods-and-floating-point/) | error bounds and stable evaluation |
| STFT and mel | [Foundation Models](../../machine-learning/05-foundation-models/) | Whisper and audio LLM front ends |

## Company Relevance

| Company | How This Appears | Difficulty |
|---------|-----------------|------------|
| OpenAI | Whisper's log-mel frontend; image preprocessing for vision models | Advanced |
| Google | SigLIP and ViT preprocessing; audio front ends | Advanced |
| Meta | Image and video pipelines at scale; on-device ASR | Advanced |
| NVIDIA | cuDNN convolution algorithms; DALI preprocessing | Advanced |
| Apple | On-device vision and speech with fixed-point kernels | Advanced |
