# Deep Learning

## Overview

- **Primary textbook**: *Deep Learning* by Goodfellow, Bengio, Courville -- [free online](https://www.deeplearningbook.org/)
- **Supplementary**: [fast.ai](https://course.fast.ai/) (free), [Stanford CS231n](http://cs231n.stanford.edu/) (free), Andrej Karpathy's [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) (free)
- **Prerequisites**: [Statistical Learning](../01-statistical-learning/), [Linear Algebra](../../math/03-linear-algebra/), [Calculus 3](../../math/04-calculus-3/), [Probability](../../math/07-probability-statistics/)
- **Estimated time**: 6-8 weeks at 10-12 hrs/week

## Key Takeaways

- Deep learning is representation learning -- the network discovers useful features, not just decision boundaries
- Backpropagation is just the chain rule applied systematically; everything else is engineering
- Transformers have replaced RNNs/LSTMs for most sequence tasks; attention is the dominant paradigm
- Scaling laws suggest performance improves predictably with data, compute, and parameters

## How to Study

- Read Goodfellow for theory; use fast.ai or Karpathy for hands-on implementation
- Implement a simple neural network from scratch (forward pass + backprop) before using frameworks
- For each architecture, understand: what inductive bias it encodes, what problem it was designed for
- Read the original papers for transformers, ResNet, and batch normalization

---

# Concepts & Techniques

## Core Insight

Deep learning succeeds because deep networks can learn hierarchical representations: early layers learn simple features (edges, phonemes), later layers compose them into complex concepts (faces, sentences). The key innovation isn't any single algorithm but the combination of (1) differentiable computation graphs, (2) backpropagation, and (3) massive scale.

## 1. ML Basics Review

**Goodfellow Ch 5**

**Key ideas**: learning algorithms as optimization; capacity, overfitting, underfitting; MLE as minimizing KL divergence between data distribution and model; Bayesian estimation; curse of dimensionality; manifold hypothesis (real-world data lies on lower-dimensional manifolds).

## 2. Deep Feedforward Networks

**Goodfellow Ch 6**

**Key ideas**:
- **Universal approximation theorem**: a feedforward network with a single hidden layer of sufficient width can approximate any continuous function on a compact set. *Caveat*: says nothing about learnability or efficiency
- **Backpropagation**: compute gradients via reverse-mode automatic differentiation; cost is O(1) forward passes regardless of number of parameters
- **Activation functions**: ReLU (default), GELU, SiLU/Swish for modern architectures; sigmoid/tanh mainly for gates
- **Architecture design**: depth vs width tradeoff; skip connections enable training very deep networks

**Key result**: deeper networks can represent functions exponentially more efficiently than shallow ones (depth separation theorems).

## 3. Regularization

**Goodfellow Ch 7**

**Key ideas**:
- **Parameter norms**: L2 (weight decay) shrinks weights; L1 promotes sparsity
- **Dropout**: randomly zero activations during training; approximately averages over ensemble of subnetworks
- **Batch normalization**: normalize activations per mini-batch; stabilizes training, acts as regularizer
- **Data augmentation**: domain-specific transformations (crops, flips, color jitter for images)
- **Early stopping**: use validation loss to determine when to stop; simple and effective

## 4. Optimization

**Goodfellow Ch 8**

**Key ideas**:
- **SGD**: stochastic gradient descent; mini-batch gradient is unbiased estimate of full gradient
- **Momentum**: accumulate past gradients; helps escape saddle points and navigate ravines
- **Adam**: adaptive learning rates per parameter; momentum + RMSprop; default choice for most tasks
- **Learning rate schedules**: warmup + cosine decay is standard for transformers; step decay for CNNs
- **Batch size**: larger batches → sharper minima (worse generalization); smaller batches → implicit regularization

**Key insight**: the loss landscape of deep networks has many saddle points but few bad local minima. Optimization is easier than theory suggests.

## 5. Convolutional Networks

**Goodfellow Ch 9**

**Key ideas**:
- **Convolution**: parameter sharing + local connectivity; translation equivariance
- **Pooling**: provides (approximate) translation invariance; reduces spatial dimensions
- **Architectures**: LeNet → AlexNet → VGG → GoogLeNet/Inception → ResNet → EfficientNet
- **ResNet insight**: skip connections solve the degradation problem; enable training of 100+ layer networks

**Key architecture**: ResNet block: y = F(x) + x. Learning residuals F(x) = H(x) - x is easier than learning H(x) directly.

## 6. Sequence Models

**Goodfellow Ch 10**

**Key ideas**:
- **RNN**: h_t = f(W_hh * h_{t-1} + W_xh * x_t); sequential processing; vanishing/exploding gradients
- **LSTM**: cell state + input/forget/output gates; solves vanishing gradient for medium sequences
- **GRU**: simplified LSTM with 2 gates; comparable performance, fewer parameters
- **Bidirectional**: process sequence in both directions; captures future context
- **Encoder-decoder**: compress input sequence → decode to output sequence; basis for machine translation

**Historical note**: RNNs/LSTMs dominated 2014-2017; transformers have largely replaced them since 2018.

## 7. Practical Methodology

**Goodfellow Ch 11**

**Key ideas**:
- **Baseline first**: always start with a simple model; establish what "good enough" looks like
- **Debugging strategy**: overfit a single batch first, then scale up; if it can't overfit, there's a bug
- **Hyperparameter tuning**: random search > grid search; Bayesian optimization for expensive models
- **When to collect more data**: if train and test error are close (both high), more data helps; if gap is large, regularize first

## 8. Autoencoders

**Goodfellow Ch 14**

**Key ideas**:
- **Undercomplete autoencoder**: bottleneck forces compression; learns compact representation
- **Variational Autoencoder (VAE)**: encode to distribution parameters (mu, sigma); sample z ~ N(mu, sigma); decode. Objective: reconstruction loss + KL divergence to prior
- **ELBO**: Evidence Lower Bound = E[log p(x|z)] - D_KL(q(z|x) || p(z)); maximizing ELBO approximates maximizing log p(x)

**Connection to information theory**: the KL term in VAE connects directly to [Information Theory](../../math/11-information-theory/) -- it measures how far the encoder distribution is from the prior.

## 9. Generative Models

**Goodfellow Ch 20 + modern extensions**

**Key ideas**:
- **GANs**: generator vs discriminator adversarial game; mode collapse is the main failure mode
- **Diffusion models**: gradually add noise, then learn to denoise; DDPM, score-based models; dominate image generation since 2021
- **Flow models**: invertible transformations with tractable likelihood; normalizing flows

## 10. Representation Learning

**Goodfellow Ch 15**

**Key ideas**:
- **Transfer learning**: pretrain on large dataset, fine-tune on small target dataset; foundation of modern DL
- **Self-supervised learning**: create labels from data itself (masked language modeling, contrastive learning) -- the paradigm that powers LLM pretraining ([Foundation Models §7](../05-foundation-models/))
- **Domain adaptation**: train on source domain, deploy on target domain; distribution shift is the challenge
- **Siamese / joint-embedding networks**: two weight-sharing encoders map two *views* of a datum to embeddings; loss pulls matching pairs together, pushes others apart. Backbone of contrastive (SimCLR, MoCo, CLIP) and metric learning (verification, retrieval)
- **Contrastive learning**: SimCLR, CLIP -- learn representations by pulling similar pairs close, pushing different pairs apart
- **Avoiding representation collapse** (the core difficulty of joint-embedding methods): negative pairs (SimCLR), momentum target encoder (BYOL, MoCo), stop-gradient + predictor (SimSiam), redundancy reduction (Barlow Twins, VICReg)
- **JEPA (Joint-Embedding Predictive Architecture)**: LeCun's *non-generative* self-supervision -- predict the *representation* of a masked target from context in embedding space, not pixels/tokens. I-JEPA (images), V-JEPA / V-JEPA 2 (video, toward world models); asymmetric context/target encoder + EMA target avoid collapse

## 11. Transformers & Modern Architectures

**Beyond the textbook -- key papers**

**Key ideas**:
- **Self-attention**: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V; O(n^2) in sequence length
- **Multi-head attention**: parallel attention heads capture different relationship types
- **Positional encoding**: sinusoidal or learned; necessary because attention is permutation-equivariant
- **Transformer architectures**: encoder-only (BERT), decoder-only (GPT), encoder-decoder (T5)
- **Scaling laws**: loss scales as power law with compute, data, and parameters (Kaplan et al., Hoffmann et al.)
- **Emergent abilities**: capabilities that appear suddenly at certain scales (few-shot learning, chain-of-thought)

**Key papers**: "Attention Is All You Need" (Vaswani 2017), "BERT" (Devlin 2018), "Language Models are Few-Shot Learners" (GPT-3, Brown 2020), "Scaling Laws for Neural Language Models" (Kaplan 2020), "Training Compute-Optimal LLMs" (Chinchilla, Hoffmann 2022)

## 12. RLHF & Alignment

**Beyond the textbook -- frontier research**

**Key ideas**:
- **RLHF**: train reward model from human preferences, then optimize policy with PPO against reward model
- **DPO**: Direct Preference Optimization -- skip the reward model; directly optimize policy from preference pairs
- **Constitutional AI**: self-critique and revision using principles; reduces need for human feedback
- **Scaling supervision**: as models become more capable, how do we evaluate them? Weak-to-strong generalization

**Key papers**: "Training language models to follow instructions with human feedback" (InstructGPT, Ouyang 2022), "Direct Preference Optimization" (Rafailov 2023), "Constitutional AI" (Bai 2022)

---

## Technique Catalog

| Architecture | Inductive Bias | Best For | Key Innovation |
|-------------|---------------|----------|----------------|
| MLP | None | Tabular data | Universal approximation |
| CNN | Translation equivariance | Images, spatial data | Parameter sharing via convolution |
| RNN/LSTM | Sequential processing | (Legacy) time series | Gated memory |
| Transformer | No structural bias | Text, multimodal, general | Self-attention mechanism |
| VAE | Smooth latent space | Generation, compression | Variational inference |
| GAN | Adversarial training | Image generation | Min-max game |
| Diffusion | Gradual denoising | Image/video generation | Score matching |

## Connections to Other Tracks

| Concept | Connected Track | Application |
|---------|-----------------|-------------|
| Gradient, Jacobian, chain rule | [Calculus 3](../../math/04-calculus-3/) | Backpropagation IS the chain rule |
| Eigenvalues, SVD | [Linear Algebra](../../math/03-linear-algebra/) | PCA, weight initialization, spectral normalization |
| KL divergence, entropy | [Information Theory](../../math/11-information-theory/) | VAE objective, cross-entropy loss |
| Statistical learning theory | [Statistical Learning](../01-statistical-learning/) | Bias-variance extends to double descent |
| Reinforcement learning | [RL](../03-reinforcement-learning/) | RLHF for language model alignment |

## Company Relevance

| Company | How This Appears | Difficulty |
|---------|-----------------|------------|
| Anthropic | Transformer internals, RLHF/DPO, alignment, interpretability | PhD-level |
| OpenAI | GPT architecture, scaling laws, RLHF, inference optimization | PhD-level |
| Google/DeepMind | Large-scale training, TPU optimization, multimodal, Gemini | PhD-level |
| NVIDIA | Inference optimization, TensorRT, quantization, hardware-aware design | Expert |
| Meta | Recommendation at scale, LLaMA, self-supervised learning | Expert |
| Netflix | Recommendation, personalization, embeddings | Advanced |
