# Statistical Language Models

This chapter covers the language models that came before deep learning, and the first neural one. They matter for two reasons. Historically, they are where the central problem of language modelling first shows up: almost every context you meet at test time was never seen in training, so a model must spread probability to things it has no direct evidence for. Practically, they stay useful. The count-based model you build here becomes the perplexity baseline every later model in the book must beat, a draft model for speculative decoding, and the corpus pipeline's perplexity filter in the TinyStories capstone.

The chapter tells one story in three steps. It opens with counts: an n-gram model with interpolated modified Kneser-Ney smoothing, the strongest of the classic smoothers, which discounts every count and redistributes the mass by how many contexts a word continues rather than how often it occurs. It then replaces counts with learned vectors: Bengio's 2003 neural probabilistic language model gives every token an embedding and an MLP over a fixed window, generalizes to unseen n-grams through similar embeddings, and is the first model you train with your own autograd on real text. Finally, it isolates the embeddings themselves: word2vec (skip-gram with negative sampling) learns them without a language model around them, and a PPMI-SVD baseline shows that SGNS implicitly factorizes a shifted PMI matrix, which you can also build and factor explicitly.

> **Learning objectives**
> - Explain why maximum-likelihood counts fail on new text, and implement interpolated modified Kneser-Ney so that every distribution is proper.
> - Build, train, score, and sample from a neural probabilistic language model using the layers and training loop from Part 0, reproducibly from a seed.
> - Derive and implement skip-gram with negative sampling with hand-written sparse gradients.
> - Build PPMI-SVD embeddings and evaluate any embedding table by word similarity and analogies.
> - Compare count-based and neural models on perplexity and bits per byte.

## Before you begin

You should have worked through [Foundations](../01-foundations/README.md), whose tensor, layers, training loop, and safetensors format are used directly here, and [Tokenizers](../02-tokenizers/README.md). From *Mathematics for Computing*, the chapter draws on [SVD, Eckart-Young low rank, and least squares](../../mathematics-for-computing/04-linear-algebra/05-svd-low-rank-and-least-squares.md), [inner products, projections, cosine similarity, and top-k](../../mathematics-for-computing/04-linear-algebra/06-inner-products-projections-cosine-and-top-k.md), and [mutual information, PMI, and PPMI](../../mathematics-for-computing/12-information-theory/03-mutual-information-pmi-ppmi.md), with the problem sets on [linear algebra, part b](../../mathematics-for-computing/04-linear-algebra/08-problem-set-b.md) and [information theory, part b](../../mathematics-for-computing/12-information-theory/05-problem-set-b.md) as practice.

## In this chapter

1. [n-gram language model with interpolated modified Kneser-Ney](01-ngram-kneser-ney.md). Builds Kneser-Ney smoothing from the zero-probability problem one fix at a time: interpolation, continuation counts, the `<s>` rule, and three data-estimated discounts per order, with a scalar and a vectorized query path that agree exactly.
2. [Bengio's neural probabilistic language model](02-bengio-nplm.md). Replaces the count table with an embedding table, a `tanh` hidden layer, and a direct path, trained with the Part 0 loop; it beats the 4-gram's perplexity on the story corpus and is saved as a model directory the zoo loads.
3. [word2vec skip-gram with negative sampling, and the PPMI-SVD baseline](03-word2vec-and-ppmi-svd.md). Derives SGNS and its three-line gradient, the noise and subsampling distributions, and sparse scatter-add SGD, then builds the explicit PPMI-SVD factorization and evaluates both by Spearman's $\rho$ and analogies.

## Further reading

- Jurafsky and Martin, [*Speech and Language Processing*](https://web.stanford.edu/~jurafsky/slp3/), chapter 3; Chen and Goodman, *An Empirical Study of Smoothing Techniques for Language Modeling* (1998); [KenLM](https://kheafield.com/code/kenlm/).
- Bengio et al., *A Neural Probabilistic Language Model* (JMLR 2003); Mikolov et al., [*Distributed Representations of Words and Phrases*](https://arxiv.org/abs/1310.4546); Levy and Goldberg, *Neural Word Embedding as Implicit Matrix Factorization* (NeurIPS 2014).
