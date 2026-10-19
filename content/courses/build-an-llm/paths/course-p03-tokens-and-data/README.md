# Pass 3: Tokens and Data

A model sees only token ids, and every id comes from a tokenizer and a corpus. If a single id differs from what a published checkpoint expects, the model reads the wrong embedding row; if the corpus is not reproducible, auditable, and free of evaluation text, no later evaluation number can be trusted. This pass builds both sides carefully: tokenizers that match GPT-2 and SmolLM2 exactly in Python and in an independent Rust implementation, and a corpus pipeline that fetches, filters, deduplicates, decontaminates, scrubs, shards, licenses, and tokenizes a corpus deterministically.

With tokens and data in place, the pass ends with the first real language models beyond the bigram: a Kneser-Ney n-gram model, the strongest count-based baseline, and Bengio's neural probabilistic language model, the first model whose embeddings and hidden layer are learned. The engine keeps serving the bigram throughout.

> **Learning objectives**
> - Explain how byte-level BPE, WordPiece, and Unigram tokenizers train and encode, and why the GPT-2 byte map is a bijection.
> - Explain how the Rust tokenizer reproduces the Python one exactly and why batching speeds it up.
> - Explain each stage of the corpus pipeline, from fetch to packed token files, and why its output hash is independent of the worker count.
> - Explain how licensing, PII, and decontamination are recorded and enforced.
> - Explain how the n-gram and neural language models are trained and scored in bits per byte.

## Before you begin

This pass needs [Pass 2](../course-p02-foundations/README.md), whose autograd trains the neural language model and whose token stream reads the files this pass writes.

## In this pass

The pass opens with the discrete mathematics and probability the tokenizers and models need, then builds the tokenizers, the data structures and tests behind them, the corpus pipeline, and finally the language models:

1. [Injective, surjective, bijective: the GPT-2 byte map](../../../mathematics-for-computing/06-discrete-math-1/02-injective-surjective-bijective-gpt2-byte-map.md). Afterwards you can explain why the GPT-2 byte map is a bijection and how it maps bytes to printable characters.
2. [Trees and tries, longest-prefix match](../../../mathematics-for-computing/07-discrete-math-2/02-trees-and-tries-longest-prefix-match.md). Afterwards you can explain how a trie finds the longest-prefix match in one pass.
3. [Discrete math 2 problem set, part b: trees, birthday bounds, MinHash and LSH, Bloom filters, recurrences](../../../mathematics-for-computing/07-discrete-math-2/05-problem-set-b.md). Afterwards you can work every problem in the set and explain each answer.
4. [Perplexity, bits per byte, NLL accumulator](../../../mathematics-for-computing/12-information-theory/02-perplexity-bits-per-byte-nll-accumulator.md). Afterwards you can explain how perplexity and bits per byte follow from the summed NLL and why the accumulator counts bytes, not tokens.
5. [Categorical sampling: inverse CDF, Gumbel-max, alias method](../../../mathematics-for-computing/08-probability-statistics/02-categorical-sampling.md). Afterwards you can explain how inverse-CDF, Gumbel-max, and the alias method each draw from a categorical distribution and what each costs.
6. [MLE, Laplace, absolute discounting](../../../mathematics-for-computing/08-probability-statistics/03-mle-laplace-absolute-discounting.md). Afterwards you can explain how MLE, Laplace smoothing, and absolute discounting assign probability to unseen events.
7. [Probability problem set, part b: joint distributions, covariance, MLE](../../../mathematics-for-computing/08-probability-statistics/10-problem-set-b.md). Afterwards you can work every problem in the set and explain each answer.
8. [The tokenizer protocol and the char tokenizer](../../02-tokenizers/01-tokenizer-protocol-and-char.md). Afterwards you can explain the tokenizer protocol and how the char tokenizer round-trips text.
9. [Byte-level BPE](../../02-tokenizers/02-byte-level-bpe.md). Afterwards you can trace how byte-level BPE pre-tokenizes, trains merges, and loads a Hugging Face tokenizer to match GPT-2.
10. [WordPiece](../../02-tokenizers/03-wordpiece.md). Afterwards you can explain how WordPiece's basic tokenizer splits text and how greedy longest match picks subwords.
11. [The Unigram language model tokenizer](../../02-tokenizers/04-unigram-lm.md). Afterwards you can explain how the Unigram LM trains with EM, segments with Viterbi, and samples subwords.
12. [Tokenizer metrics](../../02-tokenizers/06-tokenizer-metrics.md). Afterwards you can explain what each tokenizer metric measures and how it is computed.
13. [A Robin Hood hash map](../../../algorithms/13-systems-data-structures/05-robin-hood-hash-map.md). Afterwards you can explain how Robin Hood hashing bounds probe lengths and why backward-shift deletion needs no tombstones.
14. [A binary heap with lazy deletion](../../../algorithms/13-systems-data-structures/06-binary-heap-lazy-deletion.md). Afterwards you can explain how a binary heap with lazy deletion skips stale entries.
15. [Fast BPE in Rust](../../02-tokenizers/05-rust-fast-bpe.md). Afterwards you can explain how the Rust fast BPE applies merges, how the byte tokenizer works, and how the streaming decoder holds back incomplete UTF-8.
16. [Property-based tests with Hypothesis, proptest, rapid, and ol_prop.h](../../../software-craftsmanship/03-testing-mentality/02-property-based-tests.md). Afterwards you can explain which properties the tokenizers and data structures satisfy and how a property-based test shrinks a failing case.
17. [Data licensing and the ledger](../../../responsible-ai/01-data-licensing/01-data-licensing-and-the-ledger.md). Afterwards you can explain what the licensing ledger records for each source and why.
18. [Privacy and PII policy](../../../responsible-ai/02-privacy-and-pii/01-privacy-and-pii-policy.md). Afterwards you can explain the privacy and PII policy and which data it lets the pipeline keep.
19. [A Bloom filter](../../../algorithms/13-systems-data-structures/08-bloom-filter.md). Afterwards you can derive a Bloom filter's false-positive rate and explain how the number of hashes is chosen.
20. [Python asyncio: event loop, tasks, cancellation, bounded concurrency](../../../programming-languages/02-language-and-tool-primers/08-python-asyncio.md). Afterwards you can explain how asyncio schedules tasks, how cancellation propagates, and how a semaphore bounds concurrency.
21. [Async fetch with resume, checksums, and license capture](../../../data-engineering/05-corpus-pipeline/01-async-fetch-resume-checksums.md). Afterwards you can explain how the async fetcher resumes downloads, verifies checksums, and captures licenses.
22. [Extract, normalize, and quality filters](../../../data-engineering/05-corpus-pipeline/02-extract-normalize-and-quality-filters.md). Afterwards you can trace how the generator stages extract, normalize, and quality-filter documents.
23. [Exact dedup: paragraph hashes, a Bloom screen, and a sort-merge confirm](../../../data-engineering/05-corpus-pipeline/03-exact-dedup-bloom-and-sort-merge.md). Afterwards you can explain how paragraph hashes, a Bloom screen, and a sort-merge confirm find exact duplicates.
24. [Near-duplicate dedup: MinHash, LSH, union-find, and decontamination](../../../data-engineering/05-corpus-pipeline/04-near-dedup-minhash-lsh.md). Afterwards you can explain how MinHash, LSH, and union-find group near-duplicates and how decontamination removes eval overlap.
25. [PII scrub with typed placeholders and audit spans](../../../data-engineering/05-corpus-pipeline/05-pii-scrub.md). Afterwards you can explain how the PII scrub substitutes typed placeholders and records audit spans.
26. [Parquet shards, the manifest, and the document-hash split](../../../data-engineering/05-corpus-pipeline/06-parquet-shards-and-manifest.md). Afterwards you can explain how Parquet shards and the manifest are written and why the split is decided by document hash.
27. [Tokenize and pack to llm.c .bin](../../../data-engineering/05-corpus-pipeline/07-tokenize-and-pack.md). Afterwards you can explain how documents are tokenized and packed into llm.c .bin shards.
28. [Licensing ledger verification and the datasheet](../../../data-engineering/05-corpus-pipeline/08-ledger-and-datasheet.md). Afterwards you can explain how the ledger is verified against the corpus and what the datasheet reports.
29. [SVD, Eckart-Young low rank, and least squares](../../../mathematics-for-computing/04-linear-algebra/05-svd-low-rank-and-least-squares.md). Afterwards you can explain why truncated SVD is the best low-rank approximation (Eckart-Young) and how it solves least squares.
30. [Inner products, projections, cosine similarity, and top-k](../../../mathematics-for-computing/04-linear-algebra/06-inner-products-projections-cosine-and-top-k.md). Afterwards you can explain how projections and cosine similarity relate to inner products and how top-k is selected.
31. [Linear algebra problem set, part b: inner products, SVD and low rank, matmul accounting and the roofline](../../../mathematics-for-computing/04-linear-algebra/08-problem-set-b.md). Afterwards you can work every problem in the set and explain each answer.
32. [Mutual information, PMI, and PPMI](../../../mathematics-for-computing/12-information-theory/03-mutual-information-pmi-ppmi.md). Afterwards you can explain how mutual information, PMI, and PPMI are computed from co-occurrence counts.
33. [Information theory problem set, part b: coding and Huffman, mutual information, maximum entropy, rate-distortion](../../../mathematics-for-computing/12-information-theory/05-problem-set-b.md). Afterwards you can work every problem in the set and explain each answer.
34. [n-gram language model with interpolated modified Kneser-Ney](../../03-statistical-lm/01-ngram-kneser-ney.md). Afterwards you can explain how interpolated modified Kneser-Ney uses continuation counts and three discounts.
35. [Bengio's neural probabilistic language model](../../03-statistical-lm/02-bengio-nplm.md). Afterwards you can explain how Bengio's NPLM turns a context window of embeddings into a next-token distribution.
36. [word2vec skip-gram with negative sampling, and the PPMI-SVD baseline](../../03-statistical-lm/03-word2vec-and-ppmi-svd.md). Afterwards you can explain how skip-gram with negative sampling trains embeddings and how the PPMI-SVD baseline relates to it.
37. [What the System Does After Pass 3](milestone.md). Afterwards you can explain how the corpus pipeline builds a licensed, deduplicated corpus and how the tokenizers and n-gram and neural language models run on it through the CLI.

The stages are also in this pass's manifest, [`path.tsv`](path.tsv). [Pass 4](../course-p04-sequence-models/README.md) adds models that carry state across a whole sequence.
