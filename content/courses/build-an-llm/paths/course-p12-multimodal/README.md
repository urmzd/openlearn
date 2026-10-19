# Course Pass 12: Multimodal

Pass 12 extends the text system to images and audio: the signal and image math (convolution as a matrix, layouts, resampling, the FFT, the STFT, mel filterbanks), the vision models (conv layers, 2D RoPE and M-RoPE, a ViT encoder, dynamic resolution and tiling, image preprocessing, CLIP and SigLIP), the speech models (the Whisper log-mel frontend, encoder, decoder, and decoding), the multimodal eval metrics, and the consent and evidence rules for biometric data. Every piece is a Python reference on the course's own autograd and numpy, built on the earlier passes' attention, RoPE, tokenizers, and checkpoints.

**Part of**: [the course](../course/). **Needs**: [Pass 11](../course-p11-operate/).

**After this pass**: [what the system does](milestone.md): images and audio become tensors the way the reference processors make them, vision and speech encoders turn them into embeddings, Whisper transcribes speech with HF-matching logits, and the results are scored with WER, CER, VQA, captioning, and zero-shot metrics backed by consent records.

Serving the multimodal models in the Rust engine and gateway is not part of this pass's chapters yet; the text system from Pass 11 keeps working unchanged.

Stages, chapter links, and what "done" means for each are in [`path.tsv`](path.tsv).
