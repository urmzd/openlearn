# The Transformer (2017)

The previous chapter ended with attention bolted onto a recurrent network: the decoder still walks the sequence one step at a time, and attention is a side channel into the encoder's states. This chapter removes the recurrence entirely and builds "Attention Is All You Need" from its parts. Every model in the rest of the book, from GPT and BERT to the modern Llama block and the inference engine's kernels, is a rearrangement of the pieces you build here.

The arc is bottom-up. You start with the one function everything else calls, scaled dot-product attention, $\text{softmax}(QK^{\top}/\sqrt{d_k} + M)V$, and write its backward pass by hand. You then build the masks $M$ that keep a decoder from reading the future or a batch from reading padding, split attention into heads so one position can read several places at once, and give the model a sense of order with sinusoidal and learned positions. The last section assembles everything into the full encoder-decoder transformer, trains it on three-digit addition with the paper's recipe (Noam schedule, label smoothing), and shows the first decision modern models made differently: placing the LayerNorm before each sublayer instead of after, which removes the need for warmup.

> **Learning objectives**
> - Derive and implement scaled dot-product attention and its backward pass as one autograd node, and explain the $1/\sqrt{d}$ scale.
> - Build causal, padding, and sliding-window masks from one convention, and explain why the additive form uses $-\infty$.
> - Implement multi-head attention with correct head splitting, per-head scaling, and masks for every head.
> - Construct sinusoidal and learned positional encodings and prove that a sinusoidal shift is a fixed rotation.
> - Assemble and train an encoder-decoder transformer in both LayerNorm placements, and decode it step by step with beam search.

## Before you begin

You should have the attention mechanisms of [Attention Origins](../p04-attention-origins/README.md) in hand, especially [Luong attention](../p04-attention-origins/03-luong-attention-input-feeding.md) and [beam search](../p04-attention-origins/04-beam-search.md), and the `Tensor`, op library, and layers of the [foundations chapter](../p00-foundations/README.md). From *Mathematics for Computing*, the sections that matter most are the logic in the [discrete math 1 problem set](../../mathematics-for-computing/05-discrete-math-1/90-problem-set.md), the VJP derivations in the [matrix calculus problem set](../../mathematics-for-computing/08-matrix-calculus-and-autodiff/90-problem-set.md), and the rotations and frequency ladders of [Precalculus](../../mathematics-for-computing/00-precalculus/README.md).

## In this chapter

1. [Scaled dot-product attention (forward and backward)](01-scaled-dot-product-attention.md). The soft lookup at the heart of every transformer, the variance argument for its scale, and a hand-derived backward pass registered as a single autograd node, with attention dropout.
2. [Masks: causal, padding, sliding window, additive](02-masks.md). One convention (True = may attend) for every rule, a `q_offset` that makes decode chunks exact rows of the training mask, and why the additive form must be $-\infty$.
3. [Multi-head attention](03-multi-head-attention.md). Splitting the model width into heads that attend independently, the reshape-and-transpose layout, and one module that serves both self- and cross-attention.
4. [Positional encodings: sinusoidal, learned](04-positional-encodings.md). The geometric frequency ladder, the shift-is-a-rotation property that later becomes RoPE, and GPT-2's learned table.
5. [Encoder-decoder Transformer: post-LN and pre-LN, Noam, label smoothing](05-encoder-decoder-transformer.md). The full 2017 architecture, its training recipe worked by hand, step-by-step decoding with beam search, and the LayerNorm placement that decides whether it trains without warmup.

## Further reading

- Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762); Rush et al., [*The Annotated Transformer*](https://nlp.seas.harvard.edu/annotated-transformer/).
- Xiong et al., [*On Layer Normalization in the Transformer Architecture*](https://arxiv.org/abs/2002.04745).
