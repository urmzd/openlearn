# Information Theory

Information theory answers two fundamental questions. What is the minimum number of bits needed to represent data? That is source coding. And what is the maximum rate at which data can be sent reliably over a noisy channel? That is channel coding. Both answers are given by entropy. Claude Shannon's 1948 paper "A Mathematical Theory of Communication" posed and answered both, and in doing so connected probability theory to practical engineering: every compression and error-correction scheme in use today is bounded by his theorems.

For a book on mathematics for computing, the reason to study the subject now is more specific. Entropy, cross-entropy, and KL divergence are the language of machine learning. The training loss of a language model is a cross-entropy, its evaluation metric is a perplexity or a number of bits per byte, its sampler's temperature is a maximum-entropy knob, and its post-training keeps a policy close to a reference with a KL penalty. This chapter builds those quantities from the idea of surprise and turns them into code that the training and evaluation loops in *Build Your Own LLM System* call.

The chapter moves from definitions to measurement to association. It opens with entropy, cross-entropy, KL, Jensen-Shannon, and the k3 estimator, with careful attention to zero conventions and numerical stability. It then turns per-token losses into perplexity and bits per byte, correctly, over batched and sharded evaluation. Next it builds mutual information and PMI, the count-based route to word vectors. Two problem sets close the chapter: the first fixes the core definitions and identities on paper, and the second covers coding and Huffman, mutual information, maximum entropy and temperature, and rate-distortion, the last two of which appear only there.

> **Learning objectives**
> - Define entropy, cross-entropy, KL divergence, and mutual information, and compute each by hand.
> - Prove the chain rule for entropy, Gibbs' inequality, and Kraft's inequality.
> - Implement entropy, KL, and the k3 estimator so they stay exact at zero probabilities and stable on log-probabilities.
> - Report perplexity and bits per byte from batched evaluation that does not depend on batch size or order.
> - Build PMI and PPMI matrices with context distribution smoothing.
> - Explain softmax temperature as maximum entropy, and rate-distortion as the bits-for-error tradeoff behind quantization.

## Before you begin

You need discrete probability: distributions, expectation, conditioning, and Bayes' rule, as covered in [Probability & Statistics](../07-probability-statistics/README.md). The implementation sections use the stable `log_softmax` and compensated sums of [Stable numerics: logsumexp, softmax, compensated sums](../09-numerical-methods-and-floating-point/02-stable-numerics.md), the units of information from [Exponents, logs, change of base, units of information](../00-precalculus/01-exponents-logs-and-units-of-information.md), and, for PPMI word vectors, the SVD of [Linear Algebra](../03-linear-algebra/README.md).

## In this chapter

1. [Entropy, cross-entropy, KL, JS, and the k3 estimator](01-entropy-cross-entropy-and-kl.md) builds the core quantities from surprise, proves that cross-entropy is the data's entropy plus KL, and implements them with exact zero conventions and a sample-based KL estimator for post-training.
2. [Perplexity, bits per byte, NLL accumulator](02-perplexity-bits-per-byte-nll-accumulator.md) turns per-token losses into perplexity and tokenizer-independent bits per byte, with an accumulator whose answer does not depend on how the corpus is batched or sharded.
3. [Mutual information, PMI, and PPMI](04-mutual-information-pmi-ppmi.md) defines mutual information as a KL divergence and builds the smoothed PPMI matrix that is the count-based baseline for word2vec.
4. [Information theory problem set, part a](90-problem-set-a.md) works entropy, the chain rule, cross-entropy, KL, and k3 by hand, with two proofs.
5. [Information theory problem set, part b](91-problem-set-b.md) covers Huffman coding and Kraft's inequality, mutual information, softmax as a maximum-entropy distribution, and rate-distortion.

## The wider landscape

The sections above cover the parts of information theory the rest of the system calls. The subject is broader, and it helps to see the whole map once.

**Entropy and information content.** The information content of an outcome is $I(x) = -\log_2 P(x)$ bits: rare events carry more information. Shannon entropy $H(X) = -\sum P(x) \log_2 P(x)$ is the average information per symbol; joint entropy $H(X,Y) = -\sum P(x,y) \log_2 P(x,y)$ extends it to pairs; and conditional entropy $H(Y \mid X) = H(X,Y) - H(X)$ is the remaining uncertainty about $Y$ given $X$. Three theorems organize these. Maximum entropy: $H(X) \le \log_2 |X|$, with equality if and only if $X$ is uniform, because a uniform distribution is the one you cannot predict at all. The chain rule: $H(X_1, \dots, X_n) = \sum_i H(X_i \mid X_1, \dots, X_{i-1})$, so joint entropy decomposes into conditional pieces. And conditioning reduces entropy: $H(Y \mid X) \le H(Y)$, since knowing something cannot make you more uncertain on average. As an example, a source emitting $\{A, B, C, D\}$ with probabilities $\{1/2, 1/4, 1/8, 1/8\}$ has $H(X) = 1/2 + 1/2 + 3/8 + 3/8 = 1.75$ bits: on average each symbol carries 1.75 bits, and no lossless code can use fewer than 1.75 bits per symbol on average. Entropy shows up across machine learning as the cross-entropy loss, as information gain in decision-tree splitting, and in maximum-entropy models.

**Source coding.** The source coding theorem says the average length $L$ of an optimal uniquely decodable code satisfies $H(X) \le L < H(X) + 1$, and Kraft's inequality, $\sum 2^{-l_i} \le 1$, constrains the lengths $l_i$ of any such code. Huffman coding is the optimal prefix-free code for known symbol probabilities, built greedily; it is the classic example in [Greedy algorithms](../../algorithms/08-greedy/README.md) in *Algorithms*. Arithmetic coding approaches the entropy more closely on long sequences by encoding the entire message as a single number in $[0, 1)$, and a trained language model can drive an arithmetic coder, which is why bits per byte is a compression rate. Lempel-Ziv (LZ77, LZ78) is dictionary-based and universal, adapting to the source's statistics, and underlies gzip and PNG; zstd and JPEG rest on the same principles. Kolmogorov complexity connects these ideas to algorithmic information theory. Good exercises are to implement Huffman coding from scratch and compare compression ratios on English text against random bytes.

**Mutual information and KL divergence.** Mutual information $I(X;Y) = H(X) - H(X \mid Y) = H(Y) - H(Y \mid X)$ is the information shared between $X$ and $Y$. KL divergence $D_{KL}(P \Vert Q) = \sum P(x) \log(P(x)/Q(x))$ is a "distance" from $Q$ to $P$ that is not symmetric, and cross-entropy $H(P,Q) = H(P) + D_{KL}(P \Vert Q)$ is the expected number of bits when you use a code built for $Q$ on data from $P$. Gibbs' inequality, $D_{KL}(P \Vert Q) \ge 0$ with equality if and only if $P = Q$, says that using the wrong distribution always costs extra bits. The data processing inequality, $I(X;Y) \ge I(X;g(Y))$ for any function $g$, says that processing data cannot create information. In machine learning, cross-entropy is the standard classification loss, KL divergence is the objective in variational inference (VAEs and the ELBO), and mutual information drives feature selection, contrastive learning (InfoNCE), and representation learning; [Deep Learning](../../machine-learning/02-deep-learning/README.md) and [Statistical Learning](../../machine-learning/01-statistical-learning/README.md) in *Machine Learning and AI* use these constantly.

**Channel coding.** A binary symmetric channel flips each bit with probability $p$ and has capacity $C = 1 - H(p)$. In general, channel capacity is $C = \max_{P(X)} I(X;Y)$, the maximum reliable transmission rate. Shannon's noisy channel coding theorem, the central result of the field, says that for any rate $R < C$ there exist codes with arbitrarily low error probability: reliable communication is possible over a noisy channel, up to a calculable limit. Hamming codes are the simplest concrete example; the $[7,4]$ code adds 3 parity bits to 4 data bits and corrects any single error, and in general a code with minimum Hamming distance $d$ corrects $\lfloor (d-1)/2 \rfloor$ errors. Their generator and parity-check matrices are linear algebra over GF(2). Reed-Solomon codes, built from polynomial evaluation over finite fields, protect CDs, QR codes, and deep-space links, and LDPC and turbo codes approach capacity with iterative decoding. These ideas appear in TCP checksums, ECC memory, RAID, and 5G NR, and the hash-based reasoning of [Probabilistic structures](../../algorithms/15-probabilistic-structures/README.md) such as Bloom filters is a close cousin. Good exercises are a $[7,4]$ Hamming encoder and decoder and a simulation of bit error rate over a binary symmetric channel.

**Rate-distortion.** When perfect reconstruction is not needed, the rate-distortion function $R(D) = \min_{P(\hat x \mid x):\, E[d(x,\hat x)] \le D} I(X; \hat X)$ gives the fewest bits per symbol at average distortion $D$. It governs JPEG quality settings, video codec bitrate selection, and audio compression (MP3, AAC), and in machine learning it frames quantization and model compression (pruning, distillation). Problem set B works the binary and Gaussian cases.

**Information-theoretic security.** Perfect secrecy means $H(M \mid C) = H(M)$: the ciphertext reveals nothing about the message. The one-time pad achieves it, and Shannon showed that any perfectly secret scheme needs a key at least as long as the message. Entropy also measures password strength and underlies randomness extraction and quantum key distribution.

For reference, the main formulas collected in one place:

| Concept | Key formula | Application |
|---|---|---|
| Entropy | $H(X) = -\sum P(x) \log P(x)$ | measuring uncertainty, compression limits |
| KL divergence | $D_{KL}(P \Vert Q) = \sum P \log(P/Q)$ | loss functions, variational inference |
| Mutual information | $I(X;Y) = H(X) + H(Y) - H(X,Y)$ | feature selection, contrastive learning |
| Channel capacity | $C = \max I(X;Y)$ | communication system design limits |
| Huffman coding | greedy prefix-free code | optimal symbol-by-symbol compression |
| Hamming distance | minimum codeword distance $d$ | corrects $\lfloor (d-1)/2 \rfloor$ errors |

## Further reading

- *A Student's Guide to Coding and Information Theory*: an accessible treatment with worked examples, a good first companion to this chapter.
- David MacKay, *Information Theory, Inference, and Learning Algorithms*, [free online](http://www.inference.org.uk/mackay/itila/): deeper theory and close connections to machine learning.
- Thomas Cover and Joy Thomas, *Elements of Information Theory*: the standard graduate reference.
- Claude Shannon, "A Mathematical Theory of Communication" (1948): the original paper, still readable.
