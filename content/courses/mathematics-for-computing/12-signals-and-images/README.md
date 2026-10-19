# Signals and Images

Images and audio are samples of continuous signals. Every operation in a multimodal front end (convolve, resize, transform, filter, take a log) is a linear map followed at most by a pointwise nonlinearity, and getting its conventions exactly right is what makes a reimplementation agree with the pipeline a model was trained on. This chapter builds that numerical front end of every vision and speech model as numpy reference code: convolution, image layouts, resampling, the Fourier transform, the short-time Fourier transform, and the mel scale. Each section derives the mathematics, works an example by hand, and then walks through a solution that matches the reference library (PyTorch, Pillow, torchaudio, numpy, librosa) value for value.

The arc runs from images to audio. You start by seeing a convolution as a matrix, which turns it into one matmul and gives its backward pass for free. You then fix how images sit in memory, and how they are resized without aliasing, bit for bit as Pillow does it. The second half moves to sound: the DFT as a unitary change of basis and the FFT that computes it for Whisper's 400-sample frames, the windowed STFT that turns a clip into a spectrogram, and the mel filterbank and logarithm that turn the spectrogram into Whisper's features. A problem set closes the chapter and adds two topics the multimodal models need: contrastive losses and word error rate.

Small conventions decide parity throughout: pass order, rounding, padding, window symmetry. Reproduce the reference library's numbers, not just its idea.

> **Learning objectives**
> - Treat a convolution as a linear map: size its output, write it as a Toeplitz matrix and as an im2col matmul, and derive its backward pass with col2im.
> - Explain an image tensor's layout through its strides, and normalize and patchify images the way vision models expect.
> - Resize images and resample audio without aliasing, matching Pillow, `F.interpolate`, and torchaudio.
> - Compute the DFT with a mixed-radix FFT, a real FFT, the convolution theorem, and the DCT-II.
> - Build Whisper's log-mel features from a centered, windowed STFT and a Slaney mel filterbank.

## Before you begin

You need logarithms and Euler's formula from [Precalculus](../00-precalculus/README.md), matrices, transposes, and orthogonality from [Linear Algebra](../03-linear-algebra/README.md), vector-Jacobian products from [Matrix Calculus and Autodiff](../08-matrix-calculus-and-autodiff/README.md), and the floating-point error bounds of [Numerical Methods and Floating Point](../09-numerical-methods-and-floating-point/README.md). Comfort with numpy views, broadcasting, and fancy indexing helps throughout.

## In this chapter

1. [Convolution as a linear map](01-convolution-as-a-linear-map.md). Deep learning's convolution is a cross-correlation and a doubly block Toeplitz matrix; im2col makes it one matmul, and its adjoint col2im gives the backward pass.
2. [Image tensors and layouts](02-image-tensors-and-layouts.md). NCHW versus NHWC strides, why a layout change is a copy, CLIP normalization, Pillow's fixed-point grayscale, and patchify as a reshape.
3. [Resampling and interpolation](03-resampling-and-interpolation.md). Bilinear, bicubic, and Lanczos kernels, antialiasing as a stretched kernel, Pillow's separable fixed-point passes, `F.interpolate`'s convention, and a polyphase sinc resampler for audio.
4. [DFT and FFT](04-dft-and-fft.md). The DFT as a unitary map, the convolution theorem, mixed-radix Cooley-Tukey for sizes like 400, the half-size real FFT, and the DCT-II via an FFT.
5. [Sampling, Nyquist, windows, and the STFT](05-sampling-windows-and-stft.md). Aliasing, spectral leakage and the periodic Hann window, center-padded frame counts, and the overlap-add inverse.
6. [Mel scale and filterbanks](06-mel-scale-and-filterbanks.md). The Slaney and HTK mel scales, area-normalized triangular filters, decibels, and Whisper's log-mel scaling.
7. [Signals and images problem set](90-problem-set.md). Pen-and-paper practice on convolution arithmetic, sampling, Fourier, mel, contrastive losses, and WER, with worked answers.

These ideas connect to several other chapters. Convolution and im2col underlie the CNN layers and C kernels of [Neural Architectures](../../machine-learning/06-neural-architectures/README.md) in *Machine Learning and AI*; the FFT and DCT reappear in transform coding in [Information Theory](11-information-theory-link-placeholder); and the STFT and mel features are the front end of Whisper and audio LLMs in [Foundation Models](../../machine-learning/05-foundation-models/README.md). In *Build Your Own LLM System*, the [vision](../../build-an-llm/p13-vision/README.md) and [speech](../../build-an-llm/p14-speech/README.md) parts call every function built here.

## Further reading

- Oppenheim and Schafer, *Discrete-Time Signal Processing* (3rd ed.), the primary reference; a free alternative is Smith, *Mathematics of the Discrete Fourier Transform* and *Spectral Audio Signal Processing* (ccrma.stanford.edu).
- Dumoulin and Visin, "A guide to convolution arithmetic for deep learning" (free).
- Goodfellow, Bengio, Courville, *Deep Learning*, ch. 9 (free).
- Szeliski, *Computer Vision: Algorithms and Applications*, ch. 3 (free).
