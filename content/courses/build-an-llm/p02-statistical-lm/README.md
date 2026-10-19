# Part 2: Statistical Language Models

The language models before deep learning and the first neural one. An n-gram model with interpolated modified Kneser-Ney smoothing sets the baseline every later model must beat and becomes the corpus pipeline's perplexity filter in the capstone. Bengio's 2003 neural probabilistic LM is the first model trained with your autograd on real text, and word2vec (skip-gram with negative sampling) with a PPMI-SVD baseline shows where embeddings come from.

**Course passes**: L2.1 to L2.3 belong to [Pass 3](../paths/course-p03-tokens-and-data/milestone.md)

**Before you start**: the just-in-time math `M03.5`, `M03.6`, `M11.4` and the solve parts `S-M03b`, `S-M11b`; the tokenizers of [Part 1](../p01-tokenizers/).

## Key ideas

- **Smoothing is the whole game for counts**: Kneser-Ney discounts every count and redistributes the mass by how many contexts a word continues, not how often it occurs.
- **The NPLM** replaces counts with a learned embedding and an MLP, and generalizes to unseen n-grams through similar embeddings.
- **Embeddings factorize co-occurrence**: skip-gram with negative sampling implicitly factorizes a shifted PMI matrix, which PPMI plus SVD does explicitly.

## Modules

| Module | Topic | Kind | Pass |
|---|---|---|---|
| `L2.1` | n-gram LM, interpolated modified Kneser-Ney | build | 3 |
| `L2.2` | Bengio NPLM (2003) | build | 3 |
| `L2.3` | word2vec SGNS + PPMI-SVD baseline | build | 3 |

## Chapters

| # | Module | Chapter | Kind | Pass |
|---|---|---|---|---|
| 1 | `L2.1` | [n-gram language model with interpolated modified Kneser-Ney](01-ngram-kneser-ney.md) | build | 3 |
| 2 | `L2.2` | [Bengio's neural probabilistic language model](02-bengio-nplm.md) | build | 3 |
| 3 | `L2.3` | [word2vec skip-gram with negative sampling, and the PPMI-SVD baseline](03-word2vec-and-ppmi-svd.md) | build | 3 |

## Going further

- Jurafsky and Martin, [*Speech and Language Processing*](https://web.stanford.edu/~jurafsky/slp3/), chapter 3; Chen and Goodman, *An Empirical Study of Smoothing Techniques for Language Modeling* (1998); [KenLM](https://kheafield.com/code/kenlm/).
- Bengio et al., *A Neural Probabilistic Language Model* (JMLR 2003); Mikolov et al., [*Distributed Representations of Words and Phrases*](https://arxiv.org/abs/1310.4546); Levy and Goldberg, *Neural Word Embedding as Implicit Matrix Factorization* (NeurIPS 2014).
