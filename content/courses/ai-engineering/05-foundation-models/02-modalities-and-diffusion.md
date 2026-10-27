# Modalities and diffusion

This part of [Foundation Models and Architectures](README.md) develops the mechanisms below. Work the numerical examples or trace the code before checking the answer.

> **Learning objectives**
> - Explain why modality and generation objective change the serving workload.

## Modalities: text, vision, and audio

Attention operates on token sequences, and the insight that made transformers multimodal is that **everything can be tokens**. One architecture serves every modality; only the tokenizer changes.

**Text**, the original setting, is split into BPE or SentencePiece subwords that index an embedding table. **Vision** works by *patchifying*: the image is cut into a grid of patches, and each patch is linearly embedded as a token, as in the Vision Transformer (ViT). Pixels become patch tokens. **Audio** takes one of two routes. The first converts audio to a mel-spectrogram and treats its frames as tokens, as Whisper's encoder does. The second uses a neural audio codec such as EnCodec or SoundStream, which quantizes audio into discrete codes with residual vector quantization (RVQ), so that audio becomes a token stream a transformer can generate, as in audio language models and text-to-speech.

Text and audio differ sharply in practice. Text is discrete and low-rate; audio is continuous and high-rate, at 16,000 or more samples per second. Audio therefore needs a codec or spectrogram front end to reach a manageable token rate, and latency and streaming matter far more.

Once every modality is tokens, a single decoder can ingest interleaved text, image, and audio tokens, which is how models went **multimodal**. Three fusion patterns are common. *Dual-encoder* contrastive models such as CLIP align image and text embeddings. *Vision-language* models such as LLaVA and PaliGemma project image features into the LLM's token space. *Natively multimodal* models such as Gemma 3 and 4 and Gemini are trained on mixed modalities from the start, including native audio in Gemma 4 E2B and E4B.

## Diffusion models

Everything so far has been autoregressive: predict the next token, append it, repeat. **Diffusion** is the other foundation-model family, and it generates by learning to *denoise* instead. The Hugging Face [Diffusers](https://huggingface.co/docs/diffusers) library implements the methods below.

The **forward process** gradually adds Gaussian noise to data over $T$ steps, with a closed form for any step:

$$q(x_t \mid x_0) = \mathcal{N}\left(\sqrt{\bar\alpha_t}\, x_0,\ (1 - \bar\alpha_t) I\right)$$

The **reverse process** learns to undo it. DDPM trains a network $\epsilon_\theta(x_t, t)$ to predict the added noise, with loss $\mathbb{E}\lVert \epsilon - \epsilon_\theta(x_t, t) \rVert^2$. In the **score view**, $\epsilon_\theta$ is, up to scale, the score $\nabla \log p(x_t)$, and sampling means following the score back to the data distribution; this is the score-based, SDE formulation.

Several refinements made diffusion practical at scale. **Latent diffusion**, as in Stable Diffusion, diffuses in a VAE's compressed latent space rather than in pixels, which is far cheaper, and decodes the final latent to an image. **Classifier-free guidance** (CFG) combines conditional and unconditional predictions, $\hat\epsilon = \epsilon_\text{uncond} + w(\epsilon_\text{cond} - \epsilon_\text{uncond})$, trading diversity for prompt adherence. **Samplers** such as DDIM and DPM++ are ODE solvers that cut sampling from about 1000 steps to about 20.

The architecture then converged with the LLM stack. **DiT** (Diffusion Transformer) replaces the U-Net with a transformer over latent patch tokens, conditioning on the timestep or class through **adaptive LayerNorm (adaLN-zero)**, and it scales the way LLMs do. **Flow matching**, or rectified flow, replaces the noisy DDPM chain: it learns a **velocity field** $v_\theta(x_t, t)$ along the straight path $x_t = (1 - t)x_0 + t x_1$, with loss $\mathbb{E}\lVert v_\theta - (x_1 - x_0) \rVert^2$, giving fewer, straighter sampling steps. Stable Diffusion 3, Flux.1, and Lumina use it. **MMDiT** (multimodal DiT) keeps separate text and image token streams joined by attention, as in SD3 and Flux.

Diffusion dominates image, video, and audio generation, and because its architecture (a transformer trained with flow matching) is converging with the LLM stack, the two families increasingly share infrastructure.

## Summary

Explain why modality and generation objective change the serving workload. State the assumptions before applying the result to a different workload.

## Review questions

1. Why cannot autoregressive tokens per second describe every generative model?

   <details><summary>Answer</summary>

   Diffusion uses iterative denoising, and encoders and audio systems expose different units and latency constraints.

   </details>

[Chapter contents](README.md) · [Next: Model landscape and lifecycle](03-model-landscape-and-lifecycle.md)
