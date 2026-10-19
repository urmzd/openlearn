# What the system does after Pass 12

**Pass result**: signal and image math, vision models, Whisper speech recognition, and multimodal evals in Python, with consent-backed evidence; the text system from Pass 11 is unchanged.

Each capability below is built from the listed modules. The sections that follow say what the finished solution does, which solution files do it, and why each piece is needed. Everything from earlier passes keeps working.

| What the system does | Built from |
|---|---|
| Images and audio become arrays the way the reference processors make them. | `M12.1`, `M12.2`, `M12.3`, `M12.4`, `M12.5`, `M12.6`, `S-M12`, `L13.6`, `L14.1` |
| Vision encoders turn images into embeddings and align them with text. | `L13.1`, `L13.2`, `L13.3`, `L13.4`, `L13.5` |
| Whisper transcribes speech with logits that match Hugging Face. | `L14.1`, `L14.2`, `L14.3`, `L14.4` |
| Transcripts and image answers are scored, and every number is backed by consent records. | `L14.5`, `ethics.08` |

## Details

## Images and audio become arrays the way the reference processors make them

The math modules are the numpy reference semantics for everything after them. [`sig/conv.py`](../../../../../course/solutions/python/tinyllm/sig/conv.py) writes convolution as a cross-correlation and as one matrix (`im2col`, `col2im`, `conv_matrix`); [`image/layout.py`](../../../../../course/solutions/python/tinyllm/image/layout.py) converts uint8 HWC decoder output to float32 CHW and cuts patches; [`sig/resample.py`](../../../../../course/solutions/python/tinyllm/sig/resample.py) resizes with nearest, bilinear, bicubic, and Lanczos3 kernels and resamples audio; [`sig/fft.py`](../../../../../course/solutions/python/tinyllm/sig/fft.py), [`sig/stft.py`](../../../../../course/solutions/python/tinyllm/sig/stft.py), and [`sig/mel.py`](../../../../../course/solutions/python/tinyllm/sig/mel.py) compute the DFT, a mixed-radix FFT, Hann-windowed STFT frames, and mel filterbanks.

On top of them, [`vision/preprocess.py`](../../../../../course/solutions/python/tinyllm/vision/preprocess.py) decodes an image within a pixel budget, applies its EXIF orientation, resizes, crops, and normalizes it to the processor's written config, and hashes the raw bytes (`media_hash`). [`io/wav.py`](../../../../../course/solutions/python/tinyllm/io/wav.py) parses WAV from the RIFF bytes up, and [`speech/frontend.py`](../../../../../course/solutions/python/tinyllm/speech/frontend.py) turns it into Whisper's input: mono 16 kHz, padded or trimmed to 30 s, and a log-mel picture of 80 bands by 3000 frames.

Why it is needed: a model sees only the tensor its training processor produced. A one-pixel shift in resizing or a different mel normalization changes every logit downstream, so preprocessing is checked on its own, before any model.

## Vision encoders turn images into embeddings and align them with text

[`vision/conv.py`](../../../../../course/solutions/python/tinyllm/vision/conv.py) runs `Conv2d`, `Conv1d`, and pooling on the course's autograd through im2col. [`vision/rope2d.py`](../../../../../course/solutions/python/tinyllm/vision/rope2d.py) generalizes Pass 5's RoPE to patch rows and columns and to Qwen2-VL's M-RoPE. [`vision/vit.py`](../../../../../course/solutions/python/tinyllm/vision/vit.py) is the ViT encoder: a stride-p `PatchEmbed`, position tables that interpolate to new grids, and transformer blocks. [`vision/tiling.py`](../../../../../course/solutions/python/tinyllm/vision/tiling.py) chooses a resolution and tiles large images (any-resolution and Idefics3 splits) and trades spatial size for channels with `pixel_shuffle`. [`vision/contrastive.py`](../../../../../course/solutions/python/tinyllm/vision/contrastive.py) trains a dual encoder with the CLIP and SigLIP losses and classifies zero-shot from text prompts; a tiny dual encoder learns zero-shot classification of 24 color-shape classes.

Why it is needed: a language model reads tokens, so an image must become a sequence of vectors in the same space as text. Patches make the sequence, 2D positions keep the geometry, tiling keeps fine detail in large images, and contrastive training is what puts image and text embeddings in one space.

## Whisper transcribes speech with logits that match Hugging Face

[`speech/encoder.py`](../../../../../course/solutions/python/tinyllm/speech/encoder.py) turns the 30 s log-mel picture into 1500 vectors with a convolutional stem and transformer blocks. [`speech/decoder.py`](../../../../../course/solutions/python/tinyllm/speech/decoder.py) is the causal decoder with cached cross-attention (`CrossKV`) and the complete `Whisper` model, and [`io/hf_whisper.py`](../../../../../course/solutions/python/tinyllm/io/hf_whisper.py) maps a published checkpoint into it, so its logits match HF's within `1e-5`. [`speech/decode.py`](../../../../../course/solutions/python/tinyllm/speech/decode.py) and [`speech/tokenizer.py`](../../../../../course/solutions/python/tinyllm/speech/tokenizer.py) build the start-of-transcript sequence, detect the language, apply the timestamp rules, and decode a window greedily or with beam search, falling back to higher temperatures when a segment fails its quality checks.

Why it is needed: the encoder and decoder are Pass 5's encoder-decoder transformer at a new input, and matching a published checkpoint is what proves every layer and every weight mapping is right. Decoding is where Whisper's special tokens and fallbacks turn logits into a usable transcript.

## Transcripts and image answers are scored, and every number is backed by consent records

[`eval/mm.py`](../../../../../course/solutions/python/tinyllm/eval/mm.py) computes WER and CER from edit operations after normalization, VQA accuracy, caption scores, zero-shot accuracy, and bootstrap confidence intervals. The ethics.08 model answer in [`primers/ethics.08`](../../../../../course/solutions/primers/ethics.08/) keeps license and consent as separate records for every source, regenerates the model card's numbers from runnable evidence, and refuses a card whose numbers or groups do not match that evidence.

Why it is needed: a speech or vision model is judged by numbers, and those numbers are only meaningful with an interval and a record of where the media came from. Voices and faces are biometric data, so consent, not just access, decides what may be measured and published.
