# Deep Learning

Deep learning is representation learning. A classical model is given features and learns a decision boundary over them; a deep network learns the features too. Early layers discover simple patterns such as edges or phonemes, and later layers compose them into complex concepts such as faces or sentences. That hierarchical representation is why deep networks succeed where hand-engineered features stall, and it is why this chapter is the hinge of *Machine Learning and AI*: reinforcement learning, foundation models, and LLM systems in the chapters that follow all build on it.

No single algorithm is responsible. The success comes from a combination of three things: differentiable computation graphs, backpropagation to compute their gradients, and massive scale. Backpropagation in particular is less mysterious than it looks; it is the chain rule applied systematically, and most of what remains is engineering.

The chapter follows the arc of Goodfellow, Bengio, and Courville's *Deep Learning* and then goes beyond it. It begins with a review of machine learning basics, then builds feedforward networks, the regularization and optimization methods that make them train and generalize, and the two classic architecture families, convolutional networks for spatial data and recurrent networks for sequences. After a section on practical methodology, it turns to generative modeling (autoencoders, GANs, diffusion, flows) and representation learning (transfer, self-supervision, contrastive and joint-embedding methods). It ends with the material no textbook yet covers fully: transformers, which have replaced RNNs and LSTMs for most sequence tasks and made attention the dominant paradigm; scaling laws, which suggest performance improves predictably with data, compute, and parameters; and alignment through human feedback.

> **Learning objectives**
> - Explain why depth helps, what the universal approximation theorem does and does not promise, and how backpropagation computes gradients efficiently.
> - Choose and justify regularization and optimization methods (weight decay, dropout, batch normalization, SGD with momentum, Adam, learning-rate schedules).
> - Describe the inductive bias of MLPs, CNNs, RNNs and LSTMs, and transformers, and the problem each was designed to solve.
> - Derive the VAE objective (the ELBO) and contrast VAEs, GANs, diffusion models, and flows.
> - Explain self-attention, scaling laws, RLHF, and DPO well enough to discuss them in an interview.

## Before you begin

This chapter assumes [Statistical Learning](../01-statistical-learning/README.md), the previous chapter, for the bias-variance vocabulary. From *Mathematics for Computing* you need [Linear Algebra](../../mathematics-for-computing/03-linear-algebra/README.md), multivariable calculus through [Calculus 3](../../mathematics-for-computing/04-calculus-3/README.md) (gradients, Jacobians, and the chain rule), and [Probability and Statistics](../../mathematics-for-computing/07-probability-statistics/README.md).

The primary text is [*Deep Learning*](https://www.deeplearningbook.org/) by Goodfellow, Bengio, and Courville, free online; read it for theory. For hands-on implementation, use [fast.ai](https://course.fast.ai/), [Stanford CS231n](http://cs231n.stanford.edu/), or Andrej Karpathy's [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html), all free. Implement a simple neural network from scratch, forward pass and backpropagation, before you rely on a framework. For each architecture, make sure you can say what inductive bias it encodes and what problem it was designed for. And read the original papers for transformers, ResNet, and batch normalization.

## In this chapter

- [Machine learning basics revisited](#machine-learning-basics-revisited): learning as optimization, capacity, MLE, and the manifold hypothesis.
- [Deep feedforward networks](#deep-feedforward-networks): universal approximation, backpropagation, activations, and why depth matters.
- [Regularization for deep networks](#regularization-for-deep-networks): weight decay, dropout, batch normalization, augmentation, and early stopping.
- [Optimization](#optimization): SGD, momentum, Adam, schedules, batch size, and the shape of the loss landscape.
- [Convolutional networks](#convolutional-networks): convolution, pooling, and the path from LeNet to ResNet.
- [Sequence models](#sequence-models): RNNs, LSTMs, GRUs, and encoder-decoders.
- [Practical methodology](#practical-methodology): baselines, debugging, tuning, and when more data helps.
- [Autoencoders](#autoencoders) and [Generative models](#generative-models): VAEs and the ELBO, GANs, diffusion, and flows.
- [Representation learning](#representation-learning): transfer, self-supervision, contrastive learning, and JEPA.
- [Transformers and modern architectures](#transformers-and-modern-architectures) and [RLHF and alignment](#rlhf-and-alignment): the frontier beyond the textbook.

## Machine learning basics revisited

*Reading: Goodfellow Chapter 5.*

The textbook starts by recasting the previous chapter in the language deep learning uses. A learning algorithm is an optimization procedure; capacity, overfitting, and underfitting describe how flexibility relates to generalization. Maximum likelihood estimation turns out to be the same as minimizing the KL divergence between the data distribution and the model's distribution, which is why cross-entropy is the default loss. Bayesian estimation offers the alternative of keeping a distribution over parameters. Two ideas explain why deep learning is needed at all: the curse of dimensionality, which makes naive generalization in high-dimensional spaces hopeless, and the manifold hypothesis, the observation that real-world data lie near lower-dimensional manifolds that a model can learn.

## Deep feedforward networks

*Reading: Goodfellow Chapter 6.*

The basic deep model is a feedforward network: layers of affine transformations followed by non-linear activations. Its expressive power is guaranteed by the **universal approximation theorem**: a feedforward network with a single hidden layer of sufficient width can approximate any continuous function on a compact set. Read the caveat carefully, though. The theorem says nothing about whether gradient descent can find that network, or how wide it must be.

Depth is the better answer. Depth-separation results show that deeper networks can represent some functions exponentially more efficiently than shallow ones, which is the formal version of the hierarchical-features intuition. Choosing an architecture is a tradeoff between depth and width, and skip connections are what make very deep networks trainable in practice.

To train such a network you need gradients of the loss with respect to every parameter. **Backpropagation** computes them with reverse-mode automatic differentiation: one forward pass to compute and store activations, then one backward pass applying the chain rule layer by layer. The total cost is a small constant multiple of a forward pass regardless of how many parameters there are, which is what makes training networks with billions of parameters feasible.

The choice of activation matters too. ReLU is the default; GELU and SiLU (Swish) are common in modern architectures; sigmoid and tanh survive mainly as gates.

## Regularization for deep networks

*Reading: Goodfellow Chapter 7.*

Deep networks have enormous capacity, so they need regularization to generalize. Parameter-norm penalties carry over from the previous chapter: $L_2$ (weight decay) shrinks weights and $L_1$ promotes sparsity. **Dropout** randomly zeroes activations during training, which approximately averages over an ensemble of subnetworks that share weights. **Batch normalization** normalizes activations per mini-batch; it stabilizes training and also acts as a regularizer. **Data augmentation** applies domain-specific transformations that should not change the label, such as crops, flips, and color jitter for images. **Early stopping** watches validation loss and stops training when it stops improving, which is simple and effective.

## Optimization

*Reading: Goodfellow Chapter 8.*

Regularization decides what you want to find; optimization decides whether you find it. **Stochastic gradient descent** (SGD) follows the gradient of a mini-batch, which is an unbiased estimate of the full gradient at a fraction of the cost. **Momentum** accumulates past gradients, which helps it move through saddle points and along narrow ravines in the loss surface. **Adam** combines momentum with RMSprop's adaptive per-parameter learning rates, and it is the default choice for most tasks.

The learning rate is rarely constant. Warmup followed by cosine decay is standard for transformers, and step decay is common for CNNs. Batch size interacts with generalization as well: a widely reported observation is that larger batches tend to find sharper minima that generalize worse, while the noise of smaller batches acts as an implicit regularizer.

The reassuring insight of this section is that the loss landscape of deep networks has many saddle points but few bad local minima, so optimization in practice is easier than worst-case theory suggests.

## Convolutional networks

*Reading: Goodfellow Chapter 9.*

Convolutional networks encode an assumption about images: a pattern means the same thing wherever it appears. A **convolution** applies the same small filter at every position, combining parameter sharing with local connectivity, and the result is translation equivariant: shift the input and the feature map shifts with it. **Pooling** summarizes neighborhoods, which reduces spatial dimensions and provides approximate translation invariance.

The architecture history is a sequence of ideas: LeNet, AlexNet, VGG, GoogLeNet/Inception, ResNet, and EfficientNet. The most important is ResNet's. Plain networks get *worse* as they get much deeper, even on training data (the degradation problem). A residual block instead computes

$$y = F(x) + x,$$

so the layers learn the residual $F(x) = H(x) - x$ rather than the full mapping $H(x)$. Learning a residual is easier, since doing nothing is just $F = 0$, and skip connections let gradients flow directly, enabling networks of 100+ layers.

## Sequence models

*Reading: Goodfellow Chapter 10.*

For sequences, the classical approach processes one element at a time while carrying a hidden state. A **recurrent neural network** updates

$$h_t = f(W_{hh} h_{t-1} + W_{xh} x_t),$$

and because the same weights are applied at every step, gradients over long sequences are multiplied repeatedly by the same matrix and either vanish or explode. The **LSTM** adds a cell state controlled by input, forget, and output gates, which solves the vanishing-gradient problem for medium-length sequences. The **GRU** simplifies the LSTM to two gates with comparable performance and fewer parameters. **Bidirectional** variants process the sequence in both directions to capture future context, and **encoder-decoder** models compress an input sequence and decode it into an output sequence, the basis of neural machine translation.

RNNs and LSTMs dominated sequence modeling from 2014 to 2017; since 2018, transformers have largely replaced them, for reasons covered at the end of this chapter.

## Practical methodology

*Reading: Goodfellow Chapter 11.*

Knowing the architectures is not enough; you also need a working method. Always start with a baseline, a simple model that establishes what "good enough" looks like. When debugging, first make the model overfit a single batch, then scale up; if it cannot overfit one batch, there is a bug. For hyperparameter tuning, random search beats grid search, and Bayesian optimization is worth it for expensive models. Finally, diagnose before collecting data: if training and test error are close and both high, more data (or more capacity) helps; if the gap between them is large, regularize first.

## Autoencoders

*Reading: Goodfellow Chapter 14.*

So far the networks have predicted labels. Autoencoders learn representations by reconstructing their input. An **undercomplete autoencoder** forces data through a bottleneck, so it must learn a compact representation.

The **variational autoencoder** (VAE) makes this probabilistic. The encoder outputs distribution parameters $(\mu, \sigma)$, a latent code is sampled as $z \sim \mathcal{N}(\mu, \sigma^2)$, and the decoder reconstructs $x$ from $z$. The objective combines a reconstruction loss with a KL-divergence penalty toward the prior, and together they form the **evidence lower bound**:

$$\text{ELBO} = E_{q(z \mid x)}[\log p(x \mid z)] - D_{\mathrm{KL}}\big(q(z \mid x) \,\Vert\, p(z)\big).$$

Because the ELBO is a lower bound on $\log p(x)$, maximizing it approximately maximizes the likelihood of the data. The KL term connects directly to [Information Theory](../../mathematics-for-computing/11-information-theory/README.md) in *Mathematics for Computing*: it measures how far the encoder's distribution is from the prior.

## Generative models

*Reading: Goodfellow Chapter 20, plus modern extensions.*

VAEs are one family of generative models; three others matter. **GANs** pit a generator against a discriminator in an adversarial game; their characteristic failure is mode collapse, where the generator produces only a narrow range of outputs. **Diffusion models** gradually add noise to data and learn to reverse the process by denoising; DDPM and score-based models have dominated image generation since 2021. **Flow models** (normalizing flows) build the generator from invertible transformations, which gives a tractable exact likelihood.

## Representation learning

*Reading: Goodfellow Chapter 15.*

The theme of the chapter returns explicitly here: the representations a network learns are valuable in themselves. **Transfer learning**, pretraining on a large dataset and fine-tuning on a small target one, is the foundation of modern deep learning. **Self-supervised learning** creates labels from the data itself, as in masked language modeling or contrastive learning, and it is the paradigm that powers LLM pretraining; [Foundation Models](../05-foundation-models/README.md) takes this up in its discussion of the LLM development lifecycle. **Domain adaptation** handles training on a source domain and deploying on a different target domain, where distribution shift is the challenge.

Many modern methods use **Siamese** or **joint-embedding networks**: two weight-sharing encoders map two *views* of the same datum to embeddings, and the loss pulls matching pairs together and pushes others apart. This is the backbone of contrastive learning (SimCLR, MoCo, CLIP) and of metric learning for verification and retrieval.

The core difficulty of joint-embedding methods is **representation collapse**: the trivial solution of mapping everything to the same embedding makes all pairs match. Each family avoids it differently. SimCLR uses negative pairs; BYOL and MoCo use a momentum target encoder; SimSiam uses a stop-gradient with a predictor; Barlow Twins and VICReg reduce redundancy between embedding dimensions.

**JEPA** (Joint-Embedding Predictive Architecture) is LeCun's *non-generative* approach to self-supervision. Instead of predicting pixels or tokens, it predicts the *representation* of a masked target from its context, in embedding space. I-JEPA applies this to images and V-JEPA and V-JEPA 2 to video, as a step toward world models; an asymmetric context and target encoder with an EMA target avoids collapse.

## Transformers and modern architectures

*Beyond the textbook: key papers.*

The transformer is built on **self-attention**:

$$\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V,$$

in which every position attends to every other, at a cost of $O(n^2)$ in sequence length. Unlike an RNN, it has no sequential bottleneck, so it trains in parallel and connects distant positions directly. **Multi-head attention** runs several attention heads in parallel so that they can capture different kinds of relationships. Because attention is permutation-equivariant, word order must be supplied by **positional encodings**, sinusoidal or learned.

Three transformer families cover most uses: encoder-only (BERT), decoder-only (GPT), and encoder-decoder (T5). Scale then changes the picture. **Scaling laws** show that loss falls as a power law in compute, data, and parameters (Kaplan et al.; Hoffmann et al.), and some capabilities, such as few-shot learning and chain-of-thought reasoning, have been reported to appear suddenly at particular scales, which are called **emergent abilities**.

The papers to read are "Attention Is All You Need" (Vaswani et al., 2017), "BERT" (Devlin et al., 2018), "Language Models are Few-Shot Learners" (GPT-3; Brown et al., 2020), "Scaling Laws for Neural Language Models" (Kaplan et al., 2020), and "Training Compute-Optimal Large Language Models" (Chinchilla; Hoffmann et al., 2022).

## RLHF and alignment

*Beyond the textbook: frontier research.*

A pretrained language model predicts text; making it helpful and safe takes another stage. **RLHF** trains a reward model from human preference comparisons and then optimizes the policy against it with PPO. **DPO** (Direct Preference Optimization) skips the reward model and optimizes the policy directly from preference pairs. **Constitutional AI** has the model critique and revise its own outputs according to written principles, reducing the need for human feedback. Behind all of them is the problem of **scaling supervision**: as models become more capable than their evaluators, how do we evaluate them? Weak-to-strong generalization is one line of research on that question.

The papers are "Training language models to follow instructions with human feedback" (InstructGPT; Ouyang et al., 2022), "Direct Preference Optimization" (Rafailov et al., 2023), and "Constitutional AI" (Bai et al., 2022).

## Comparing the architectures

The architectures in this chapter differ most clearly in the inductive bias they build in. The catalog below summarizes them:

| Architecture | Inductive Bias | Best For | Key Innovation |
|-------------|---------------|----------|----------------|
| MLP | None | Tabular data | Universal approximation |
| CNN | Translation equivariance | Images, spatial data | Parameter sharing via convolution |
| RNN/LSTM | Sequential processing | (Legacy) time series | Gated memory |
| Transformer | No structural bias | Text, multimodal, general | Self-attention mechanism |
| VAE | Smooth latent space | Generation, compression | Variational inference |
| GAN | Adversarial training | Image generation | Min-max game |
| Diffusion | Gradual denoising | Image/video generation | Score matching |

## Where this leads

The mathematics behind this chapter runs through the rest of the book. Backpropagation *is* the chain rule from [Calculus 3](../../mathematics-for-computing/04-calculus-3/README.md); eigenvalues and SVD from [Linear Algebra](../../mathematics-for-computing/03-linear-algebra/README.md) appear in PCA, weight initialization, and spectral normalization; and KL divergence and entropy from [Information Theory](../../mathematics-for-computing/11-information-theory/README.md) define the VAE objective and the cross-entropy loss. The bias-variance tradeoff of [Statistical Learning](../01-statistical-learning/README.md) extends into the double-descent behavior of large networks. The next chapter, [Reinforcement Learning](../03-reinforcement-learning/README.md), supplies the algorithms behind RLHF, and [Neural Network Architectures & the History of Deep Learning](../06-neural-architectures/README.md) derives backpropagation, CNNs, and LSTMs from the math and codes them from scratch.

These topics are also central to interviews at AI-focused companies, as the table below shows:

| Company | How This Appears | Difficulty |
|---------|-----------------|------------|
| Anthropic | Transformer internals, RLHF/DPO, alignment, interpretability | PhD-level |
| OpenAI | GPT architecture, scaling laws, RLHF, inference optimization | PhD-level |
| Google/DeepMind | Large-scale training, TPU optimization, multimodal, Gemini | PhD-level |
| NVIDIA | Inference optimization, TensorRT, quantization, hardware-aware design | Expert |
| Meta | Recommendation at scale, LLaMA, self-supervised learning | Expert |
| Netflix | Recommendation, personalization, embeddings | Advanced |

### Further reading

- Goodfellow, Bengio, and Courville, [*Deep Learning*](https://www.deeplearningbook.org/) (free online).
- [fast.ai Practical Deep Learning](https://course.fast.ai/) and [Stanford CS231n](http://cs231n.stanford.edu/).
- Andrej Karpathy, [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html).
- The original papers listed in the transformer and alignment sections above.

## Key terms

- **Representation learning**: learning the features themselves, not just a decision rule over given features.
- **Backpropagation**: reverse-mode automatic differentiation that computes all parameter gradients in one backward pass.
- **Dropout**: randomly zeroing activations during training to approximate an ensemble of subnetworks.
- **Residual connection**: computing $y = F(x) + x$ so that layers learn a residual and gradients flow directly.
- **ELBO**: the evidence lower bound, reconstruction likelihood minus KL to the prior, maximized to train a VAE.
- **Representation collapse**: the trivial solution in joint-embedding training where all inputs map to the same embedding.
- **Self-attention**: a layer in which each position computes a weighted average of all positions' values, weighted by query-key similarity.
- **Scaling laws**: empirical power laws relating loss to compute, data, and parameter count.

## Summary

- Deep learning is representation learning: depth lets networks compose simple features into complex ones.
- Backpropagation is the chain rule applied systematically, and it costs about as much as a forward pass regardless of parameter count.
- Regularization (weight decay, dropout, batch normalization, augmentation, early stopping) and optimization (SGD, momentum, Adam, schedules) are what make deep networks train and generalize.
- Each architecture encodes an inductive bias: convolution for translation, recurrence for sequence, attention for flexible all-to-all interaction.
- Transformers have replaced RNNs for most sequence tasks, scaling laws make performance predictable with scale, and RLHF and DPO align models with human preferences.

## Review questions

1. What does the universal approximation theorem guarantee, and what does it not?
   <details><summary>Answer</summary>It guarantees that a single-hidden-layer network of sufficient width can approximate any continuous function on a compact set. It does not say how wide the network must be, whether gradient descent can find the right weights, or how well the result will generalize. Depth separation results are why deeper networks are preferred in practice.</details>
2. Why does a residual block make very deep networks trainable?
   <details><summary>Answer</summary>With $y = F(x) + x$, the layers only need to learn the residual $F(x) = H(x) - x$; the identity is the easy default ($F = 0$), so adding layers cannot easily make the network worse. The skip connection also passes gradients directly backward, avoiding the degradation problem of plain deep networks.</details>
3. Your model cannot overfit a single batch. What does that tell you?
   <details><summary>Answer</summary>That there is a bug, such as a wrong loss, labels misaligned with inputs, a broken gradient path, or a learning rate far off. A model with any reasonable capacity should be able to memorize one batch, so fix this before scaling up.</details>
4. Write the ELBO and explain each term.
   <details><summary>Answer</summary>$\text{ELBO} = E_{q(z\mid x)}[\log p(x \mid z)] - D_{\mathrm{KL}}(q(z \mid x) \Vert p(z))$. The first term rewards accurate reconstruction of $x$ from sampled latents; the second penalizes the encoder's distribution for straying from the prior, keeping the latent space smooth. The ELBO lower-bounds $\log p(x)$, so maximizing it approximately maximizes likelihood.</details>
5. Why do transformers need positional encodings, and why is attention $O(n^2)$?
   <details><summary>Answer</summary>Self-attention is permutation-equivariant: shuffling the inputs just shuffles the outputs, so without positional information the model cannot tell word order. It is $O(n^2)$ because every one of $n$ positions computes a score against every other position.</details>
6. How does DPO differ from RLHF with PPO?
   <details><summary>Answer</summary>RLHF first trains a separate reward model on human preferences and then optimizes the policy against it with PPO. DPO skips the reward model and the RL loop, optimizing the policy directly on preference pairs with a classification-style loss.</details>
7. Name three ways joint-embedding methods avoid representation collapse.
   <details><summary>Answer</summary>Negative pairs (SimCLR), a momentum target encoder (BYOL, MoCo), a stop-gradient with a predictor (SimSiam), or redundancy reduction across embedding dimensions (Barlow Twins, VICReg). JEPA uses an asymmetric context/target encoder with an EMA target.</details>
