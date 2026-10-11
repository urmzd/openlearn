# Information Theory

The mathematical foundations of communication, compression, and coding -- connecting probability theory to practical engineering.

## Overview

- **Primary textbook**: *A Student's Guide to Coding and Information Theory* -- owned (~/Documents/books/)
- **Advanced reference**: *Information Theory, Inference, and Learning Algorithms* by David MacKay -- [free online](http://www.inference.org.uk/mackay/itila/)
- **Supplementary**: *Elements of Information Theory* by Cover & Thomas (standard graduate reference); Claude Shannon's original 1948 paper "A Mathematical Theory of Communication"
- **Prerequisites**: [Probability & Statistics](../07-probability-statistics/), [Linear Algebra](../03-linear-algebra/)
- **Estimated time**: 3-4 weeks at 8-10 hrs/week

## Key Takeaways

- Information theory quantifies uncertainty and the limits of communication -- every compression and error-correction scheme is bounded by Shannon's theorems
- Entropy is the central concept: it measures surprise, determines compression limits, and connects to statistical mechanics and ML
- The noisy channel coding theorem is one of the most profound results in engineering: reliable communication is possible even over noisy channels, up to a calculable limit
- These ideas underpin modern ML (cross-entropy loss, KL divergence, variational inference, diffusion models)

## How to Study

- Start with the Student's Guide for accessible treatment with worked examples
- Move to MacKay for deeper theory and connections to machine learning
- Work through the coding exercises -- implement Huffman coding, LZ77, and a simple error-correcting code
- Connect each concept to its CS/ML application

---

# Concepts & Techniques

## Core Insight

Information theory answers two fundamental questions: (1) what is the minimum number of bits needed to represent data? (source coding), and (2) what is the maximum rate at which data can be reliably sent over a noisy channel? (channel coding). Both answers are given by entropy.

## 1. Entropy & Information Content

**Key definitions**:
- **Information content**: I(x) = -log_2 P(x) bits. Rare events carry more information.
- **Shannon entropy**: H(X) = -sum P(x) log_2 P(x). Average information per symbol.
- **Joint entropy**: H(X,Y) = -sum P(x,y) log_2 P(x,y)
- **Conditional entropy**: H(Y|X) = H(X,Y) - H(X). Remaining uncertainty about Y given X.

**Key theorems**:
- **Maximum entropy**: H(X) <= log_2 |X|, with equality iff X is uniform. *Intuition*: uniform distributions are the most "surprising" -- you can't predict anything.
- **Chain rule**: H(X_1, ..., X_n) = sum H(X_i | X_1, ..., X_{i-1}). Joint entropy decomposes into conditional pieces.
- **Conditioning reduces entropy**: H(Y|X) <= H(Y). *Intuition*: knowing something can't make you more uncertain.

**Worked example**:
> A source emits symbols {A, B, C, D} with probabilities {1/2, 1/4, 1/8, 1/8}.
> H(X) = -(1/2 log 1/2 + 1/4 log 1/4 + 1/8 log 1/8 + 1/8 log 1/8) = 1/2 + 1/2 + 3/8 + 3/8 = 1.75 bits.
> This means on average, each symbol carries 1.75 bits of information. No lossless code can use fewer than 1.75 bits/symbol on average.

**Connections to CS**: entropy appears everywhere in ML -- cross-entropy loss in classification, decision tree splitting (information gain), maximum entropy models.

## 2. Source Coding (Data Compression)

**Key ideas**:
- **Source coding theorem**: the average code length L satisfies H(X) <= L < H(X) + 1 for a uniquely decodable code
- **Huffman coding**: optimal prefix-free code for known symbol probabilities; greedy construction
- **Arithmetic coding**: approaches entropy more closely for long sequences; encodes entire message as a single number in [0,1)
- **Lempel-Ziv (LZ77/LZ78)**: dictionary-based compression; universal (adapts to source statistics); basis of gzip, PNG
- **Kraft's inequality**: sum 2^(-l_i) <= 1 for any uniquely decodable code with lengths l_i

**Essential exercises**: implement Huffman coding from scratch; compare compression ratios on English text vs random bytes

**Connections to CS**: gzip, zstd, PNG, JPEG all use these principles. Kolmogorov complexity connects to algorithmic information theory. [Greedy algorithms](../../algorithms/08-greedy/) (Huffman is the classic greedy example).

## 3. Mutual Information & KL Divergence

**Key definitions**:
- **Mutual information**: I(X;Y) = H(X) - H(Y|X) = H(Y) - H(X|Y). Information shared between X and Y.
- **KL divergence**: D_KL(P||Q) = sum P(x) log(P(x)/Q(x)). "Distance" from Q to P (not symmetric).
- **Cross-entropy**: H(P,Q) = H(P) + D_KL(P||Q). Expected bits when using code Q for distribution P.

**Key theorems**:
- **Gibbs' inequality**: D_KL(P||Q) >= 0, with equality iff P = Q. *Intuition*: using the wrong distribution always costs extra bits.
- **Data processing inequality**: I(X;Y) >= I(X;g(Y)) for any function g. *Intuition*: processing data can't create information.

**Connections to ML**: cross-entropy is THE standard loss function for classification. KL divergence is the objective in variational inference (VAEs, ELBO). Mutual information is used in feature selection, contrastive learning (InfoNCE), and representation learning. [Deep Learning](../../ml/02-deep-learning/) uses these constantly.

## 4. Channel Coding (Error Correction)

**Key ideas**:
- **Binary symmetric channel (BSC)**: each bit flipped with probability p; capacity C = 1 - H(p)
- **Channel capacity**: C = max_{P(X)} I(X;Y). The maximum reliable transmission rate.
- **Shannon's noisy channel coding theorem**: for any rate R < C, there exists a code achieving arbitrarily low error probability. *The most important theorem in information theory.*
- **Hamming codes**: [7,4] code -- 4 data bits + 3 parity bits, corrects 1 error. Hamming distance determines error correction capability.
- **Reed-Solomon codes**: polynomial evaluation over finite fields; used in CDs, QR codes, deep space communication
- **LDPC & Turbo codes**: modern capacity-approaching codes; iterative decoding

**Essential exercises**: implement a [7,4] Hamming encoder/decoder; simulate BER over a BSC

**Connections to CS**: TCP checksums, ECC memory, RAID, QR codes, 5G NR. [Probabilistic structures](../../algorithms/15-probabilistic-structures/) (Bloom filters use similar hash-based ideas).

## 5. Rate-Distortion Theory

**Key ideas**:
- **Lossy compression**: when perfect reconstruction isn't needed, how much can we compress?
- **Rate-distortion function**: R(D) = min_{P(x_hat|x): E[d(x,x_hat)] <= D} I(X; X_hat)
- **Applications**: JPEG quality settings, video codec bitrate selection, audio compression (MP3/AAC)
- **Connections**: quantization in neural networks, model compression (pruning, distillation)

## 6. Information-Theoretic Security

**Key ideas**:
- **Perfect secrecy**: H(M|C) = H(M) -- ciphertext reveals nothing about the message
- **One-time pad**: achieves perfect secrecy; key must be as long as message (Shannon's impossibility result)
- **Connections**: entropy-based password strength, randomness extraction, quantum key distribution

---

## Technique Catalog

| Concept | Key Formula | Application |
|---------|-------------|-------------|
| Entropy | H(X) = -sum P(x) log P(x) | Measuring uncertainty, compression limits |
| KL divergence | D_KL(P\|\|Q) = sum P log(P/Q) | Loss functions (ML), variational inference |
| Mutual information | I(X;Y) = H(X) + H(Y) - H(X,Y) | Feature selection, contrastive learning |
| Channel capacity | C = max I(X;Y) | Communication system design limits |
| Huffman coding | Greedy prefix-free code | Optimal symbol-by-symbol compression |
| Hamming distance | Minimum codeword distance d | Error correction: corrects floor((d-1)/2) errors |

## Course modules

This topic moved here from the former top-level Information Theory track: it is math with call sites in the course. The build modules turn entropy, KL, and perplexity into code the training and evaluation loops call; the rest are solve-only or optional.

| Module | Topic | Kind | Pass |
|---|---|---|---|
| `S-M11a` | Entropy, cross-entropy, and KL problem set | solve | 2 |
| `S-M11b` | Mutual information and PMI problem set | solve | 3 |
| `M11.1` | Entropy, cross-entropy, KL, JS, k3 KL estimator | build | 2 |
| `M11.2` | Perplexity, bits per byte, NLL accumulator | build | 3 |
| `M11.3` | Source coding: Huffman, arithmetic coding, LM as compressor | build | optional |
| `M11.4` | Mutual information, PMI, PPMI | build | 3 |
| `M11.5` | Max entropy, softmax as a Gibbs distribution, temperature | solve | solve set |
| `M11.6` | Rate-distortion and quantization | solve | optional |

Huffman coding (`M11.3`, optional) has its interview-pattern version in [Greedy](../../algorithms/08-greedy/); `sq.lm-compressor` turns a trained model into an arithmetic coder.

## Chapters

<!-- ol:chapters -->
| # | Module | Chapter | Kind | Pass |
|---|---|---|---|---|
| 1 | `M11.1` | [Entropy, cross-entropy, KL, JS, and the k3 estimator](01-entropy-cross-entropy-and-kl.md) | build | 2 |
| 2 | `M11.2` | [Perplexity, bits per byte, NLL accumulator](02-perplexity-bits-per-byte-nll-accumulator.md) | build | 3 |
| 3 | `M11.4` | [Mutual information, PMI, and PPMI](04-mutual-information-pmi-ppmi.md) | build | 3 |
| 4 | `S-M11a` | [Information theory problem set, part a: entropy, chain rules, KL and cross-entropy](90-problem-set-a.md) | solve | 2 |
| 5 | `S-M11b` | [Information theory problem set, part b: coding and Huffman, mutual information, maximum entropy, rate-distortion](91-problem-set-b.md) | solve | 3 |
<!-- /ol:chapters -->

## Connections to Other Tracks

| Info Theory Concept | Connected Track | Application |
|--------------------|-----------------|-------------|
| Entropy, KL divergence | [Deep Learning](../../ml/02-deep-learning/) | Cross-entropy loss, VAE objective (ELBO) |
| Mutual information | [Statistical Learning](../../ml/01-statistical-learning/) | Feature selection, information gain |
| Huffman coding | [Greedy Algorithms](../../algorithms/08-greedy/) | Classic greedy algorithm |
| Error-correcting codes | [Linear Algebra](../03-linear-algebra/) | Generator/parity-check matrices over GF(2) |
| Channel capacity | [Probability](../07-probability-statistics/) | Optimization over probability distributions |
| Rate-distortion | [Deep Learning](../../ml/02-deep-learning/) | Model compression, quantization |

## Company Relevance

| Company | How This Appears | Difficulty |
|---------|-----------------|------------|
| Anthropic | KL divergence in RLHF, information-theoretic alignment | Advanced |
| Google | Compression in distributed systems, ranking entropy | Advanced |
| DeepMind | Information-theoretic bounds in learning, exploration | PhD-level |
| Netflix | Video compression (rate-distortion optimization) | Advanced |
| Jane Street | Entropy in market microstructure, information asymmetry | Advanced |
| NVIDIA | Compression for inference, quantization theory | Advanced |
