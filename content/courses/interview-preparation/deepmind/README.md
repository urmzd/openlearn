# Google DeepMind

Google DeepMind is Google's premier AI research lab, the group behind AlphaFold, AlphaGo, Gemini, and a long run of foundational AI research. It hires software engineers and research engineers, and the defining fact about its interviews is that the bar is research-grade even for engineering roles. You are expected to write Google-quality code and also to reason about loss functions, model architectures, and papers as a researcher would.

DeepMind is now merged with Google Brain under the name "Google DeepMind", and its interview process blends Google's structured approach with DeepMind's research depth. This chapter walks through that process round by round: the coding rounds, which add ML implementation problems to standard algorithms; the ML and research round, where you discuss papers and derive gradients; the system design round, which is about ML infrastructure on Google's TPU and GPU fleet; and the behavioral round, where research culture and safety come up. It closes with preparation advice.

> **Learning objectives**
> - Describe the structure of the DeepMind interview loop and what each round assesses.
> - Implement numerically stable softmax with cross-entropy, scaled dot-product and multi-head attention, and k-means from scratch in NumPy.
> - Discuss a research paper using a six-part framework, from problem to impact.
> - Sketch a large-scale training system and a distributed RL training system and name their key design considerations.
> - Speak thoughtfully about AI safety, interpretability, and robustness.

## Before you begin

The algorithmic rounds are at Google's level, so read the [Google](../google/README.md) chapter first for the problem patterns. The ML material assumes you can follow the attention and training math; the cross-company chapter on [ML infrastructure](../shared-concepts/ml-infrastructure.md) covers the systems side. For serving design, the [Anthropic system design](../anthropic/03-system-design.md) section is the closest companion.

## The interview process

Expect the process to take four to eight weeks across five or six rounds. The loop looks like this:

| Round | Format | Duration | Focus |
|-------|--------|----------|-------|
| Recruiter Screen | Phone | 30 min | Background, research interests |
| Phone Screen 1 | Coding | 60 min | DS&A + ML implementation |
| Phone Screen 2 | Technical | 60 min | ML theory or systems |
| Onsite 1 | Coding | 60 min | Algorithmic problem-solving |
| Onsite 2 | ML / Research | 60 min | ML concepts, paper discussion |
| Onsite 3 | System Design | 60 min | ML infrastructure at scale |
| Onsite 4 | Behavioral | 45 min | Research culture, collaboration |

Compensation for a senior SWE or research engineer, in London or the US, breaks down roughly as follows. Base salary is about $180-250K in the US and GBP 100-150K in London. Equity comes as Google RSUs, around $200-500K per year, with a 15-25% bonus on top. Total compensation is around $400-750K or more at the senior level, and higher for research scientists. London roles pay less in absolute terms but carry strong benefits.

## What DeepMind looks for

Five themes run through every round. The first is research depth expected from engineers: even SWE roles require understanding ML theory, so you should be able to discuss papers, understand loss functions, and reason about model architectures. The second is implementation of research, turning papers into working code at scale, where JAX or TensorFlow proficiency is valued. The third is scale combined with correctness. Training runs on thousands of TPUs, and a bug can waste millions of dollars of compute. The fourth is safety and alignment; DeepMind has a dedicated safety team and AI safety is part of the culture. The fifth is a publication culture in which engineers often co-author papers, so research literacy is expected.

## Coding rounds

The coding rounds mix Google-style data structures and algorithms problems with ML implementation problems. You may be asked to implement an ML algorithm from scratch, NumPy and linear algebra fluency are expected, and Python and C++ are the primary languages. The standard algorithmic problems are at the same medium-hard difficulty as Google's; see the [Google](../google/README.md) chapter for the patterns. The ML implementation problems are what make these rounds distinctive, and three come up often.

### Softmax and cross-entropy loss

The first is softmax with cross-entropy loss and its gradient. The detail interviewers look for is numerical stability: subtracting the row maximum before exponentiating leaves the result unchanged but prevents overflow. The gradient of cross-entropy with respect to the logits is the famously simple "probabilities minus one-hot targets", averaged over the batch.

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

### The attention mechanism

The second is attention. Scaled dot-product attention divides the query-key scores by $\sqrt{d_k}$ so that their variance does not grow with the head dimension, applies an optional mask by replacing masked scores with a large negative number, takes the softmax over the key dimension, and uses the weights to average the values. Multi-head attention projects the inputs, reshapes them into `num_heads` heads of size `d_k = d_model // num_heads`, runs attention per head, and concatenates and projects the results. Keep track of the shapes in the comments; interviewers watch for that.

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

> **Warning:** `scaled_dot_product_attention` as written calls `K.transpose(0, 2, 1)`, which is right for three-dimensional inputs. When `multi_head_attention` passes it four-dimensional per-head tensors, that call raises an error; transposing the last two axes (`K.swapaxes(-1, -2)`) works for both. Spotting and explaining that kind of shape bug is exactly what the round rewards.

### K-means from scratch

The third is k-means clustering. The listing initialises centroids with k-means++, which picks each new centroid with probability proportional to its squared distance from the nearest existing one, then alternates assignment and update steps until the centroids move less than a tolerance. An empty cluster keeps its old centroid rather than producing a NaN mean.

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

## The ML and research round

Where the coding rounds test whether you can build ML components, this round tests whether you understand them. You may be asked to discuss a recent paper you have read, explain a model architecture (Transformer, VAE, GAN, diffusion), derive a loss function or gradient, discuss trade-offs between approaches, or analyse an experiment design. The depth expected varies by topic:

| Topic | Depth Required |
|-------|---------------|
| Transformers | Deep: architecture, attention, positional encoding, KV cache |
| Training dynamics | Optimizer choice, learning rate schedules, gradient clipping |
| Regularization | Dropout, weight decay, data augmentation, early stopping |
| Evaluation | Metrics, overfitting detection, cross-validation, statistical significance |
| Reinforcement Learning | MDP, policy gradient, Q-learning, RLHF (for LLMs) |
| Generative models | VAE, GAN, diffusion models (conceptual understanding) |
| Scaling laws | Chinchilla scaling, compute-optimal training |

When you are asked to discuss a paper, a fixed structure keeps you from rambling. Start with the problem the paper solves, then its key insight or novel contribution, then the method at a high level, then the results it showed. Finish with its limitations (what does not work, what could be improved) and its impact on the field. Interviewers weigh the last two most heavily, because they show you read critically rather than just summarising.

You should be ready to discuss these papers:

- **Attention Is All You Need** (Vaswani et al., 2017): the Transformer.
- **Scaling Laws for Neural Language Models** (Kaplan et al., 2020): scaling behaviour.
- **Training Compute-Optimal LLMs** (Hoffmann et al., 2022): Chinchilla.
- **AlphaFold 2**: protein structure prediction.
- **Constitutional AI** (Bai et al., 2022): an AI alignment approach.
- **Denoising Diffusion Probabilistic Models** (Ho et al., 2020): denoising diffusion.

## The system design round

System design at DeepMind is about ML infrastructure at massive scale on Google's TPU and GPU infrastructure. Three problems are common.

### A large-scale model training system

The first is designing a system for training a large model. The data flows from a pipeline through preprocessing into a data loader that feeds a training loop distributed across accelerators, with gradient synchronisation, checkpointing, and logging branching off the loop:

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

The considerations to raise are these. Training happens on TPU pods of thousands of chips, combining data parallelism with model parallelism. Checkpoints should be frequent and asynchronous, because a training run costs millions and a crash must not lose much of it. Training should be deterministic, so that the same seed and data order give the same result and runs are reproducible. Mixed precision uses BF16 for the forward pass and FP32 for accumulation. And the data pipeline must never starve the accelerators, which means prefetching aggressively.

### A distributed RL training system

The second is reinforcement learning infrastructure, where the shape is different: many actors generate experience, and a central learner consumes it.

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

Thousands of environments run in parallel on distributed actors. The learner samples efficiently from large replay buffers, and prioritised replay samples more often from high-error experiences. Because actors run slightly stale policies, the learner needs off-policy correction. Evaluation runs on separate workers with different seeds so that it does not contaminate training.

### Serving Gemini

The third is a model serving system for Gemini. It is similar to the inference serving problem in the [Anthropic system design](../anthropic/03-system-design.md) section, with the addition of multiple modalities (text, image, audio, video). Cover KV cache management, continuous batching, safety filtering, and streaming responses.

## The behavioral round

The behavioral round assesses fit with DeepMind's research culture, which rests on five values:

| Value | What They Assess |
|-------|-----------------|
| Scientific rigor | Evidence-based reasoning, reproducibility |
| Collaboration | Cross-team research, pair programming |
| Ambition | Think big, tackle hard problems |
| Safety | Responsible AI development |
| Openness | Publishing research, sharing knowledge |

Typical questions include: "What's the most interesting ML paper you've read recently? Why?", "Tell me about a time you had to debug a complex ML system.", "How do you approach a research problem with no clear solution?", "Describe a time you had to balance research exploration with engineering deadlines.", and "What's your view on AI safety and alignment?"

That last question deserves preparation of its own, because DeepMind has dedicated safety research and safety is central to its mission. Know the basic areas: alignment (making AI systems that do what we intend), interpretability (understanding what models learn and why they make decisions), robustness (models that work reliably, even on unusual inputs), evaluation (how to measure whether a model is safe and aligned), and governance (technical tools for safe deployment).

## Preparing

Preparation splits into reading, implementing, and practising. Read ten to fifteen landmark papers and be able to discuss three to five in depth; be comfortable reading and discussing academic papers in general. Implement attention, backpropagation, and basic optimizers from scratch in NumPy until you understand the math, and build the mathematical maturity behind it: linear algebra, calculus, probability, and optimization, to the point where you can derive gradients. DeepMind uses JAX extensively, so familiarity with it (or TensorFlow) is a strong signal. Practise Google-style coding for the algorithmic rounds, as covered in the [Google](../google/README.md) chapter. Know DeepMind's own work (AlphaFold, AlphaGo, Gemini, Gato, RT-2) at a conceptual level. And arrive with a thoughtful perspective on AI safety.

## Summary

- The DeepMind loop is Google's structure with a research-grade bar: five or six rounds over four to eight weeks.
- Coding rounds add from-scratch ML implementations (softmax and cross-entropy, attention, k-means) to Google-level algorithms.
- The research round rewards structured, critical paper discussion and the ability to derive losses and gradients.
- System design centres on training and RL infrastructure at TPU-pod scale, with checkpointing, determinism, and data throughput as key concerns.
- Safety is part of the culture; prepare a considered view.

## Further reading

- [DeepMind Research Publications](https://deepmind.google/research/publications/)
- [DeepMind Blog](https://deepmind.google/discover/blog/)
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [Training Compute-Optimal LLMs (Hoffmann et al., 2022)](https://arxiv.org/abs/2203.15556)
- [Glassdoor: DeepMind Interview Questions](https://www.glassdoor.com/Interview/Google-DeepMind-Interview-Questions-E11788021.htm)
- [levels.fyi: Google DeepMind Compensation Data](https://www.levels.fyi/companies/google/salaries/research-scientist)
- Community reports on r/MachineLearning, r/cscareerquestions, and Blind.
