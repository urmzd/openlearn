# Tokenizers

Every model in this book reads ids, not text, and the tokenizer decides what those ids are. Until now the tracer has used the simplest possible answer: 256 byte ids, no training, every text round-trips. That costs one token per byte and makes the model learn to spell UTF-8. This chapter goes from bytes to trained subword vocabularies, and it does so with an unusual standard of correctness: wherever a real model's tokenizer exists, your implementation must reproduce its ids exactly.

The chapter's arc follows one idea: a tokenizer is a contract. It opens by fixing that contract, the `Tokenizer` protocol, so that `encode(decode(ids)) == ids` and `decode(encode(text)) == text` hold for every string, including invalid UTF-8 split across a stream, and so that everything downstream is written once. Three trained families then implement it. Byte-level BPE merges the most frequent adjacent pair until the vocabulary is full, and encoding replays the merges by rank; it loads GPT-2's and SmolLM2's Hugging Face files. WordPiece is BERT's normalizer plus greedy longest match. The Unigram language model starts from a large vocabulary, prunes by likelihood loss, and can sample segmentations, which the capstone's BPE-versus-Unigram ablation measures. BPE is then reimplemented independently in Rust as `tl-tok`, with a streaming UTF-8 decoder, and Python and Rust are held to the same frozen fixtures through `tokenizer.json`. The chapter ends with the metrics that decide a vocabulary size for the capstone, of which compression comes first: bytes per token and bits per byte on held-out text.

> **Learning objectives**
> - State the tokenizer protocol and its laws, and validate ids and files against it.
> - Train byte-level BPE and encode by merge rank, reproducing GPT-2's and SmolLM2's ids from their real files.
> - Implement BERT's WordPiece pipeline and the Unigram language model (Viterbi, EM, subword sampling).
> - Build a fast Rust BPE with a lazy merge heap and a streaming UTF-8 decoder that matches the Python id for id.
> - Measure fertility, bytes per token, and fallback rate, and use bytes per token to compare models in bits per byte.

## Before you begin

The mathematics this chapter uses comes from *Mathematics for Computing*: the [GPT-2 byte map](../../mathematics-for-computing/05-discrete-math-1/02-injective-surjective-bijective-gpt2-byte-map.md), [trees and tries with longest-prefix match](../../mathematics-for-computing/06-discrete-math-2/02-trees-and-tries-longest-prefix-match.md), [categorical sampling](../../mathematics-for-computing/07-probability-statistics/01-categorical-sampling.md), [maximum-likelihood estimation](../../mathematics-for-computing/07-probability-statistics/02-mle-laplace-absolute-discounting.md), and [perplexity and bits per byte](../../mathematics-for-computing/11-information-theory/02-perplexity-bits-per-byte-nll-accumulator.md), with practice in the problem sets on [trees and hashing](../../mathematics-for-computing/06-discrete-math-2/91-problem-set-b.md) and [joint distributions and MLE](../../mathematics-for-computing/07-probability-statistics/91-problem-set-b.md). For the Rust section you also need the [Rust primer](../../programming-languages/02-language-and-tool-primers/04-rust.md) in *Programming Languages*, and the [Robin Hood hash map](../../algorithms/16-systems-data-structures/05-robin-hood-hash-map.md) and [binary heap with lazy deletion](../../algorithms/16-systems-data-structures/06-binary-heap-lazy-deletion.md) from *Algorithms*. You should have read [the byte bigram](../p00-foundations/00-byte-bigram.md), whose byte ids this chapter generalizes.

## In this chapter

1. [The tokenizer protocol and the char tokenizer](01-tokenizer-protocol-and-char.md): the one interface every tokenizer implements, shared id validation, the byte tokenizer rewritten against it, and a char tokenizer whose ids do not depend on the order of its training texts.
2. [Byte-level BPE](02-byte-level-bpe.md): a hand-written GPT-2 pre-tokenizer, a lazy-heap BPE trainer that matches Hugging Face's merges, rank-ordered encoding, and loaders that reproduce GPT-2's and SmolLM2's ids.
3. [WordPiece](03-wordpiece.md): BERT's basic tokenizer and greedy longest match, exact enough to reproduce bert-base-uncased's ids, template, and decodes.
4. [The Unigram language model tokenizer](04-unigram-lm.md): Viterbi encoding over a lattice, EM training by forward-backward, pruning by removal loss, and exact sampling of segmentations.
5. [Fast BPE in Rust](05-rust-fast-bpe.md): the `tl-tok` crate, with an $O(n \log n)$ merge heap, Unicode tables generated from Python, parallel batch encoding, and a streaming decoder for the server.
6. [Tokenizer metrics](06-tokenizer-metrics.md): fertility, bytes per token, and fallback rate for any tokenizer, and why bytes per token is the exchange rate to bits per byte.

## Further reading

- Hugging Face [tokenizers](https://github.com/huggingface/tokenizers) (Rust BPE merges and the pre-tokenizer pipeline) and OpenAI [tiktoken](https://github.com/openai/tiktoken).
- Sennrich, Haddow, and Birch, [*Neural Machine Translation of Rare Words with Subword Units*](https://arxiv.org/abs/1508.07909).
- Kudo, [*Subword Regularization*](https://arxiv.org/abs/1804.10959).
