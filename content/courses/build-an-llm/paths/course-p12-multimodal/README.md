# Pass 12: Multimodal

This pass extends the text system to images and audio. A language model reads tokens, so an image or a clip of speech must become a sequence of vectors in the same space as text, and the model sees only the tensor its training processor produced. The pass builds that path from first principles: the signal and image mathematics (convolution as a matrix, layouts, resampling, the FFT, the STFT, and mel filterbanks), the vision models (conv layers, 2D RoPE and M-RoPE, a ViT encoder, dynamic resolution and tiling, image preprocessing to a written specification, and CLIP and SigLIP), and the speech models (Whisper's log-mel frontend, encoder, decoder, and decoding).

Every piece is a Python reference on the course's own autograd and numpy, built on the attention, RoPE, tokenizers, and checkpoints of earlier passes. The pass ends with the metrics that score transcripts and image answers, and with the consent and evidence rules for biometric data: voices and faces are biometric, so consent, not just access, decides what may be measured and published. Serving the multimodal models in the Rust engine and gateway is not part of this pass yet; the text system from Pass 11 keeps working unchanged.

> **Learning objectives**
> - Explain convolution as a linear map, image layouts, resampling kernels, the FFT, the STFT, and mel filterbanks.
> - Explain how a ViT embeds patches, how 2D positions and tiling preserve geometry and detail, and how contrastive training aligns images with text.
> - Explain how Whisper turns audio into log-mel frames, encodes them, and decodes a transcript with special tokens and fallbacks.
> - Compute WER, CER, and the other multimodal metrics with confidence intervals, and explain why each needs consent-backed evidence.

## Before you begin

This pass needs [Pass 11](../course-p11-operate/README.md). It leans most on the encoder-decoder transformer and RoPE from [Pass 5](../course-p05-transformer/README.md).

## In this pass

2. [Convolution as a linear map](../../../mathematics-for-computing/13-signals-and-images/01-convolution-as-a-linear-map.md). Afterwards you can explain why convolution is a linear map and how it is written as a matrix.
3. [Image tensors and layouts](../../../mathematics-for-computing/13-signals-and-images/02-image-tensors-and-layouts.md). Afterwards you can explain image tensor layouts (HWC, CHW) and how to convert between them.
4. [Resampling and interpolation](../../../mathematics-for-computing/13-signals-and-images/03-resampling-and-interpolation.md). Afterwards you can explain how resampling and interpolation (nearest, bilinear, bicubic, antialiasing) compute output pixels.
5. [DFT and FFT](../../../mathematics-for-computing/13-signals-and-images/04-dft-and-fft.md). Afterwards you can explain the DFT as a matrix and how the FFT computes it in O(n log n).
6. [Sampling, Nyquist, windows, and the STFT](../../../mathematics-for-computing/13-signals-and-images/05-sampling-windows-and-stft.md). Afterwards you can explain the Nyquist limit, why windows reduce leakage, and how the STFT frames a signal.
7. [Mel scale and filterbanks](../../../mathematics-for-computing/13-signals-and-images/06-mel-scale-and-filterbanks.md). Afterwards you can explain the mel scale and how a mel filterbank is built from triangular filters.
8. [Signals and images problem set](../../../mathematics-for-computing/13-signals-and-images/07-problem-set.md). Afterwards you can work every problem in the set and explain each answer.
9. [Conv layers and pooling](../../14-vision/01-conv-layers-and-pooling.md). Afterwards you can explain how conv layers run through im2col and how pooling reduces resolution.
10. [2D RoPE and M-RoPE](../../14-vision/02-rope-2d-and-m-rope.md). Afterwards you can explain how 2D RoPE and M-RoPE extend rotary position to images.
11. [Vision transformer encoder](../../14-vision/03-vision-transformer.md). Afterwards you can explain how the ViT encoder embeds patches and runs transformer blocks over them.
12. [Dynamic resolution and tiling](../../14-vision/04-dynamic-resolution-and-tiling.md). Afterwards you can explain how dynamic resolution tiles an image and how pixel shuffle reduces tokens.
13. [Image preprocessing](../../14-vision/06-image-preprocessing.md). Afterwards you can explain each step of image preprocessing to the written spec.
14. [Contrastive image-text](../../14-vision/05-contrastive-image-text.md). Afterwards you can explain how CLIP and SigLIP train image-text alignment with contrastive losses.
15. [Audio frontend](../../15-speech/01-audio-frontend.md). Afterwards you can explain how the audio frontend parses WAV and computes Whisper's log-mel features.
16. [Whisper encoder](../../15-speech/02-whisper-encoder.md). Afterwards you can explain how the Whisper encoder's conv stem and transformer blocks process log-mel frames.
17. [Whisper decoder and weights](../../15-speech/03-whisper-decoder-and-weights.md). Afterwards you can explain how the Whisper decoder cross-attends to the encoder and how its weights are loaded.
18. [Whisper decoding and transcription](../../15-speech/04-whisper-decoding.md). Afterwards you can explain how Whisper decoding uses special tokens, timestamps, beam search, and temperature fallback.
19. [Multimodal eval metrics](../../15-speech/05-multimodal-eval-metrics.md). Afterwards you can explain how WER, CER, and the other multimodal eval metrics are computed.
20. [Consent and biometric evidence](../../../responsible-ai/02-privacy-and-pii/02-consent-and-biometric-evidence.md). Afterwards you can explain why the model card's numbers must match runnable evidence and why every source needs valid consent records.
21. [What the System Does After Pass 12](milestone.md). Afterwards you can explain how the system sees and hears: images and audio flow through preprocessing, encoders, and decoders that match their reference models.

The stages are also in this pass's manifest, [`path.tsv`](path.tsv). After this pass the system [sees and hears](milestone.md), and the text system is unchanged.
