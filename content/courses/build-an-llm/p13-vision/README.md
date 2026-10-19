# Vision foundations

Everything in the book so far reads token ids. This chapter teaches the system to see: by its end, a compressed image file becomes the exact float tensors a published vision tower was trained on, that tower reproduces Hugging Face's hidden states at every layer, and a small dual encoder trained on captions alone can classify images it has never seen labels for.

The arc runs from layers to pixels. You start with convolution and pooling as autograd layers, the patch embedding every vision transformer begins with. Then come positions for a two-dimensional grid: axial 2D RoPE for patches, and Qwen2-VL's M-RoPE, which lets an image occupy a small square of positions inside a language model's sequence. With those pieces, the third section builds one configurable vision transformer that reproduces the ViT, CLIP, SigLIP, and Idefics3 towers. The fourth handles images larger than the tower's training size by tiling them, and makes the resulting token count an exact function of the image size. The fifth gives the tower's vectors meaning through contrastive training on image-text pairs, which also yields zero-shot classifiers. The last section closes the loop at the input end, turning untrusted encoded bytes into tensors exactly to the processor's specification.

One thread runs through every section: match a published reference exactly. A wrong switch, rounding, or pixel order still loads and runs; it just produces features the model was never trained on. So each section pins its behaviour against Hugging Face and against the mathematical properties it derives. A planned vision-language model section will join these features to the language model.

> **Learning objectives**
> - Implement convolution and pooling layers with hand-written backward passes and train LeNet-5 with them.
> - Build axial 2D RoPE and M-RoPE position schemes and explain why each preserves relative position.
> - Configure one vision transformer to reproduce four published towers at every hidden state.
> - Compute tiling and image token counts for Idefics3 and LLaVA-NeXT exactly from the image size.
> - Train a CLIP or SigLIP dual encoder and use it for zero-shot classification.
> - Preprocess encoded images safely and exactly to a model's processor specification.

## Before you begin

The mathematics comes from chapter 12 of *Mathematics for Computing*, [Signals and images](../../mathematics-for-computing/12-signals-and-images/README.md): convolution as a linear map, image layouts, and resampling. You should be comfortable with the autograd engine and layers of [Foundations](../p00-foundations/README.md), attention from [The Transformer (2017)](../p05-transformer-2017/README.md), and the RoPE of [The Modern Decoder Block](../p07-modern-block/README.md). The contrastive section also uses the cross-entropy and softmax-with-temperature ideas covered in those chapters.

## In this chapter

1. [Conv layers and pooling on autograd](01-conv-layers-and-pooling.md). Turn im2col into `Conv2d`, `Conv1d`, and pooling layers with three-formula backward passes, pin the max-pool tie rule, and prove the layers learn by training LeNet-5 on digits.
2. [2D RoPE and M-RoPE](02-rope-2d-and-m-rope.md). Give half the rotary pairs to the row and half to the column so attention scores depend only on patch offsets, and give language-model tokens three position ids so an image costs its side length in positions, not its token count.
3. [Vision transformer encoder](03-vision-transformer.md). Make an image a sequence with a patch embedding, choose among learned, sin-cos, and rotary positions, hide padding on the key side, and reproduce four published towers from one config.
4. [Dynamic resolution and tiling](04-dynamic-resolution-and-tiling.md). Cut large images into tower-sized tiles the way Idefics3 and LLaVA-NeXT do, count image tokens from the size alone, and trade sequence length for width with pixel shuffle.
5. [Contrastive image-text: CLIP and SigLIP](05-contrastive-image-text.md). Derive CLIP's symmetric InfoNCE loss and SigLIP's pairwise sigmoid loss, learn and clamp the temperature, and write a classifier in words.
6. [Image preprocessing to the specification](06-image-preprocessing.md). Decode untrusted bytes within a pixel budget, apply orientation and alpha exactly, follow each processor family's geometry and normalization, and hash media by its original bytes.
