# Google DeepMind Software Engineer Interview Guide

Comprehensive preparation for DeepMind SWE and Research Engineer roles. DeepMind is Google's premier AI research lab, responsible for AlphaFold, AlphaGo, Gemini, and foundational AI research. The bar is **research-grade** even for engineering roles.

## Interview Process Overview

Timeline: **4-8 weeks**, **5-6 rounds**

| Round | Format | Duration | Focus |
|-------|--------|----------|-------|
| Recruiter Screen | Phone | 30 min | Background, research interests |
| Phone Screen 1 | Coding | 60 min | DS&A + ML implementation |
| Phone Screen 2 | Technical | 60 min | ML theory or systems |
| Onsite 1 | Coding | 60 min | Algorithmic problem-solving |
| Onsite 2 | ML / Research | 60 min | ML concepts, paper discussion |
| Onsite 3 | System Design | 60 min | ML infrastructure at scale |
| Onsite 4 | Behavioral | 45 min | Research culture, collaboration |

DeepMind is now merged with Google Brain under "Google DeepMind". The interview process blends Google's structured approach with DeepMind's research depth.

## Compensation (Senior SWE / Research Engineer, London/US)

- **Base**: $180-250K (US), GBP 100-150K (London)
- **Equity**: Google RSUs, ~$200-500K/yr
- **Bonus**: 15-25%
- **Total Comp**: ~$400-750K+ (Senior), higher for Research Scientists
- London roles pay less in absolute terms but have strong benefits

## Key Themes

1. **Research depth expected from engineers** -- Even SWE roles require understanding ML theory. You should be able to discuss papers, understand loss functions, and reason about model architectures.
2. **Implementation of research** -- Turning papers into working code at scale. JAX/TensorFlow proficiency valued.
3. **Scale + correctness** -- Training runs on thousands of TPUs. Bugs can waste millions of dollars in compute.
4. **Safety and alignment** -- DeepMind has a dedicated safety team. AI safety is part of the culture.
5. **Publication culture** -- Engineers often co-author papers. Research literacy is expected.

## Coding Rounds

### What to Expect

- Google-style DS&A problems plus ML implementation problems
- May be asked to implement an ML algorithm from scratch
- NumPy / linear algebra fluency expected
- Python and C++ are the primary languages

### Standard Algorithmic Problems

Same difficulty as Google (medium-hard). See [Google guide](../google/README.md) for patterns.

### ML Implementation Problems

#### Implement Softmax + Cross-Entropy Loss

```python
import numpy as np

def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)

def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray) -> float:
    """Cross-entropy loss from logits.
    logits: (batch, num_classes)
    targets: (batch,) integer class labels
    """
    probs = softmax(logits)
    batch_size = logits.shape[0]
    # Select probability of correct class
    correct_probs = probs[np.arange(batch_size), targets]
    return -np.mean(np.log(correct_probs + 1e-10))

def softmax_grad(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Gradient of cross-entropy loss w.r.t. logits."""
    probs = softmax(logits)
    batch_size = logits.shape[0]
    grad = probs.copy()
    grad[np.arange(batch_size), targets] -= 1
    return grad / batch_size
```

#### Implement Attention Mechanism

```python
import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    """Scaled dot-product attention.
    Q: (batch, seq_q, d_k)
    K: (batch, seq_k, d_k)
    V: (batch, seq_k, d_v)
    mask: (batch, seq_q, seq_k) or None
    """
    d_k = Q.shape[-1]
    # Compute attention scores
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)  # (batch, seq_q, seq_k)

    if mask is not None:
        scores = np.where(mask, scores, -1e9)

    # Softmax over key dimension
    weights = softmax(scores)  # (batch, seq_q, seq_k)

    # Weighted sum of values
    output = np.matmul(weights, V)  # (batch, seq_q, d_v)
    return output, weights


def multi_head_attention(Q, K, V, W_q, W_k, W_v, W_o, num_heads):
    """Multi-head attention.
    Q, K, V: (batch, seq, d_model)
    W_q, W_k, W_v: (d_model, d_model) -- projection weights
    W_o: (d_model, d_model) -- output projection
    """
    batch, seq_q, d_model = Q.shape
    d_k = d_model // num_heads

    # Project and reshape to (batch, num_heads, seq, d_k)
    q = (Q @ W_q).reshape(batch, seq_q, num_heads, d_k).transpose(0, 2, 1, 3)
    k = (K @ W_k).reshape(batch, -1, num_heads, d_k).transpose(0, 2, 1, 3)
    v = (V @ W_v).reshape(batch, -1, num_heads, d_k).transpose(0, 2, 1, 3)

    # Attention per head
    attn_output, _ = scaled_dot_product_attention(q, k, v)

    # Concatenate heads and project
    concat = attn_output.transpose(0, 2, 1, 3).reshape(batch, seq_q, d_model)
    return concat @ W_o
```

#### Implement K-Means from Scratch

```python
import numpy as np

def kmeans(X: np.ndarray, k: int, max_iters: int = 100, tol: float = 1e-4):
    """K-means clustering.
    X: (n_samples, n_features)
    Returns: centroids, assignments
    """
    n = X.shape[0]
    # Initialize centroids using k-means++
    centroids = [X[np.random.randint(n)]]
    for _ in range(1, k):
        distances = np.min([np.sum((X - c) ** 2, axis=1) for c in centroids], axis=0)
        probs = distances / distances.sum()
        centroids.append(X[np.random.choice(n, p=probs)])
    centroids = np.array(centroids)

    for iteration in range(max_iters):
        # Assign points to nearest centroid
        distances = np.array([np.sum((X - c) ** 2, axis=1) for c in centroids])
        assignments = np.argmin(distances, axis=0)

        # Update centroids
        new_centroids = np.array([
            X[assignments == i].mean(axis=0) if np.any(assignments == i) else centroids[i]
            for i in range(k)
        ])

        # Check convergence
        if np.all(np.abs(new_centroids - centroids) < tol):
            break
        centroids = new_centroids

    return centroids, assignments
```

## ML / Research Round

### What to Expect

This round tests your understanding of ML concepts. You may be asked to:

- Discuss a recent paper you've read
- Explain a model architecture (Transformer, VAE, GAN, diffusion)
- Derive a loss function or gradient
- Discuss trade-offs between approaches
- Analyze an experiment design

### Topics to Know

| Topic | Depth Required |
|-------|---------------|
| Transformers | Deep -- architecture, attention, positional encoding, KV cache |
| Training dynamics | Optimizer choice, learning rate schedules, gradient clipping |
| Regularization | Dropout, weight decay, data augmentation, early stopping |
| Evaluation | Metrics, overfitting detection, cross-validation, statistical significance |
| Reinforcement Learning | MDP, policy gradient, Q-learning, RLHF (for LLMs) |
| Generative models | VAE, GAN, diffusion models (conceptual understanding) |
| Scaling laws | Chinchilla scaling, compute-optimal training |

### Paper Discussion Framework

When asked to discuss a paper:

1. **Problem**: What problem does the paper solve?
2. **Key insight**: What's the novel contribution?
3. **Method**: How does it work? (high level)
4. **Results**: What did they show?
5. **Limitations**: What doesn't work? What could be improved?
6. **Impact**: How has this influenced the field?

### Papers You Should Know

- **Attention Is All You Need** (Vaswani et al., 2017) -- The Transformer
- **Scaling Laws for Neural Language Models** (Kaplan et al., 2020) -- Scaling behavior
- **Training Compute-Optimal LLMs** (Hoffmann et al., 2022) -- Chinchilla
- **AlphaFold 2** -- Protein structure prediction
- **Constitutional AI** (Bai et al., 2022) -- AI alignment approach
- **Diffusion Models** (Ho et al., 2020) -- Denoising diffusion

## System Design Round

### DeepMind-Specific Focus

System design at DeepMind is about ML infrastructure at massive scale on Google's TPU/GPU infrastructure.

### Common Topics

#### Design a Large-Scale Model Training System

```
[Data Pipeline] --> [Preprocessing] --> [Data Loader]
                                              |
                                        [Training Loop]
                                    (distributed across TPUs/GPUs)
                                              |
                                  +--------+--------+
                                  |        |        |
                            [Gradient Sync] [Checkpoint] [Logging]
                                  |                        |
                            [Optimizer Step]          [Experiment Tracker]
                                  |
                            [Evaluation Pipeline]
                                  |
                            [Model Registry]
```

Key considerations:
- **TPU pods**: Training on 1000s of TPU chips. Data parallelism + model parallelism.
- **Checkpointing**: Frequent async checkpoints (training runs cost millions).
- **Deterministic training**: Same seed + data order = same result (reproducibility).
- **Mixed precision**: BF16 for forward pass, FP32 for accumulation.
- **Data pipeline**: Avoid GPU/TPU starvation. Prefetch aggressively.

#### Design an RL Training Infrastructure

```
[Environment Pool] --> [Actor Workers] --> [Experience Buffer]
                                                |
                                          [Replay Sampling]
                                                |
                                          [Learner (GPU/TPU)]
                                                |
                                          [Policy Update]
                                                |
                                          [Actor Sync]
```

- **Distributed actors**: Thousands of environments running in parallel
- **Experience replay**: Efficient sampling from large replay buffers
- **Prioritized replay**: Sample more from high-error experiences
- **Off-policy correction**: Handle staleness of experience data
- **Evaluation**: Separate evaluation workers with different seeds

#### Design a Model Serving System for Gemini

Similar to Anthropic's inference serving (see [Anthropic System Design](../anthropic/03-system-design.md)):
- Multi-modal (text, image, audio, video)
- KV cache management
- Continuous batching
- Safety filtering
- Streaming responses

## Behavioral Round

### DeepMind Culture

| Value | What They Assess |
|-------|-----------------|
| Scientific rigor | Evidence-based reasoning, reproducibility |
| Collaboration | Cross-team research, pair programming |
| Ambition | Think big, tackle hard problems |
| Safety | Responsible AI development |
| Openness | Publishing research, sharing knowledge |

### Common Questions

- "What's the most interesting ML paper you've read recently? Why?"
- "Tell me about a time you had to debug a complex ML system."
- "How do you approach a research problem with no clear solution?"
- "Describe a time you had to balance research exploration with engineering deadlines."
- "What's your view on AI safety and alignment?"

## AI Safety at DeepMind

DeepMind has dedicated safety research. Know the basics:

- **Alignment**: Making AI systems that do what we intend
- **Interpretability**: Understanding what models learn and why they make decisions
- **Robustness**: Models that work reliably, even on unusual inputs
- **Evaluation**: How to measure whether a model is safe/aligned
- **Governance**: Technical tools for safe deployment

## Preparation Tips

1. **Read papers** -- Read 10-15 landmark papers. Be able to discuss 3-5 in depth.
2. **Implement from scratch** -- Code up attention, backprop, basic optimizers in NumPy. Understand the math.
3. **JAX or TensorFlow** -- DeepMind uses JAX extensively. Familiarity is a strong signal.
4. **Google-style coding** -- The DS&A rounds are similar to Google. See [Google guide](../google/README.md).
5. **Mathematical maturity** -- Linear algebra, calculus, probability, optimization. Be able to derive gradients.
6. **Know DeepMind's work** -- AlphaFold, AlphaGo, Gemini, Gato, RT-2. Understand at a conceptual level.
7. **Research literacy** -- Be comfortable reading and discussing academic papers.
8. **Safety awareness** -- Have a thoughtful perspective on AI safety. It's central to DeepMind's mission.

## Sources

- [DeepMind Research Publications](https://deepmind.google/research/publications/)
- [DeepMind Blog](https://deepmind.google/discover/blog/)
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [Training Compute-Optimal LLMs (Hoffmann et al., 2022)](https://arxiv.org/abs/2203.15556)
- [Glassdoor - DeepMind Interview Questions](https://www.glassdoor.com/Interview/Google-DeepMind-Interview-Questions-E11788021.htm)
- [levels.fyi - Google DeepMind Compensation Data](https://www.levels.fyi/companies/google/salaries/research-scientist)
- r/MachineLearning, r/cscareerquestions, Blind (community reports)
