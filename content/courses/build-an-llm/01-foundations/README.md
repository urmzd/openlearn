# Foundations

Every model in this book, from a bigram to a Llama-family decoder, rests on the same small stack of machinery: a way to represent a model's weights, a way to compute gradients through any expression, a library of differentiable operations and losses, a way to organize parameters into named layers, a loop that trains them, and file formats that save the result so other programs can load it. This chapter builds that stack from scratch in Python and NumPy.

The chapter opens with the smallest real language model: a byte-level bigram fitted by counting, whose logits come from a NumPy row gather and whose weights are written as a safetensors checkpoint that the Rust engine serves. That one model gives every other layer of the system something to carry. The rest of the chapter then builds an autograd engine and retrains the same bigram with it, behind the same checkpoint format. Because the bigram's optimum is known exactly (it is the count model), gradient descent finding it again is the most precise end-to-end test of the autograd there is. The chapter closes with every dtype the book needs, crash-safe checkpoints, and a memory-mapped token stream that resumes bit for bit.

> **Learning objectives**
> - Fit, evaluate, and sample a byte-level bigram language model, and write it as a safetensors file byte-identical to the reference library's.
> - Build reverse-mode autograd on NumPy arrays, including broadcasting, repeated indices, and `no_grad`.
> - Implement and gradient-check a library of differentiable ops and fused losses, with stable numerics and correct padding rules.
> - Organize parameters into modules whose names match PyTorch's, and save and load them in place.
> - Write a reproducible training loop, crash-safe checkpoints, and a token stream that resumes exactly.

## Before you begin

You should be comfortable with [Python and NumPy](../../programming-languages/02-language-and-tool-primers/01-python-and-numpy.md) from *Programming Languages* (arrays, dtypes, broadcasting, views versus copies, uv projects) and with [Vectors, matrices, row-major layout, and matmul](../../mathematics-for-computing/04-linear-algebra/01-vectors-matrices-and-matmul.md) from *Mathematics for Computing*. The autograd sections also draw on [Scalar reverse mode: the Value graph](../../mathematics-for-computing/09-matrix-calculus-and-autodiff/02-scalar-reverse-mode.md) and [Matrix differentials, the trace trick, and closed-form VJPs](../../mathematics-for-computing/09-matrix-calculus-and-autodiff/03-matrix-differentials-and-vjps.md), and the training loop on the optimizers of [Adam and AdamW](../../mathematics-for-computing/11-optimization/03-adam-and-adamw.md). Each section names the exact pieces it uses as it goes.

## In this chapter

1. [Byte bigram: counts, logits, safetensors v0](01-byte-bigram.md). The smallest real language model: a smoothed count table, its negative log-likelihood, temperature sampling by inverse CDF, and a canonical safetensors writer and strict reader.
2. [Tensor, broadcasting backward, and no_grad](02-tensor-and-broadcasting-backward.md). Reverse-mode autograd on arrays: one VJP per op, reverse topological order, why broadcasting and repeated indices mean summed gradients, and a grad mode that nests and survives exceptions.
3. [The op library and gradcheck_all](03-op-library-and-gradcheck.md). Activations, reductions, shape and selection ops, the stable softmax family, and inverted dropout, each with one closed-form VJP, plus a gradient checker strong enough to fail.
4. [Losses with a fused backward](04-losses-with-fused-backward.md). Cross-entropy as one graph node with gradient $p - q$, `ignore_index` and label smoothing done right, and stable MSE and binary cross-entropy, with a test suite that pins them down.
5. [Module system and basic layers](05-module-system-and-layers.md). Parameter registration by assignment, PyTorch-compatible names, all-or-nothing in-place loading, and `Linear`, `Embedding`, `LayerNorm`, `Dropout`, and the containers.
6. [Training loop and the autograd bigram](06-training-loop-and-the-autograd-bigram.md). A seeded data loader, a training step in the right order, honest evaluation, and the bigram retrained by gradient descent to the count model's NLL, sampling exactly as the Rust engine does.
7. [Safetensors for every dtype, atomic checkpoints, and the token stream](07-safetensors-checkpoints-and-token-streams.md). BF16 and F8_E4M3 with round to nearest even, a checkpoint protocol that survives a crash at any instant, and llm.c token shards read through a memory map with an exact cursor.
