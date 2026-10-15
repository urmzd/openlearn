# Moonshot AI Software Engineer Interview Guide

Comprehensive preparation for Moonshot AI engineering roles. Moonshot AI is a Chinese AI startup founded by Yang Zhilin (co-creator of Transformer-XL and XLNet), valued at $3B+, and building frontier LLMs including the Kimi assistant.

## Overview

Moonshot AI is one of China's leading AI startups, competing with Baidu, Alibaba, and ByteDance in the LLM space. Founded in March 2023, it raised over $1.2B and is known for long-context language models (Kimi supports 2M+ token context windows).

## Interview Process Overview

Timeline: **2-3 weeks** (fast-moving startup), **4-5 rounds**

| Round | Format | Duration | Focus |
|-------|--------|----------|-------|
| Recruiter Screen | Phone | 30 min | Background, motivation |
| Take-Home Assessment | Technical | 2-3 hours | Python, NLP, ML, AI |
| Technical Interview | Video/Onsite | 60-90 min | Coding + ML depth |
| Behavioral + Team Fit | Video/Onsite | 45-60 min | Collaboration, AI ethics |
| Final Round | Panel (CEO/HR) | 45-60 min | Vision alignment, use cases, ethics |

Offers typically come within 2-3 days of the final round.

## Compensation (Estimated)

- Moonshot AI compensation is competitive with top Chinese tech companies
- Significant equity component (early-stage startup with high growth)
- US-based roles (if available) would benchmark against Bay Area AI startups
- Equity upside is the primary draw given the $3B+ valuation and growth trajectory

## Key Themes

1. **Long-context expertise** -- Moonshot's technical edge is long-context LLMs (2M+ tokens). Understand attention optimization, efficient KV cache, and memory management at extreme sequence lengths.
2. **NLP/ML depth** -- Yang Zhilin's research background means the bar for ML understanding is very high.
3. **Chinese AI ecosystem** -- Understanding the competitive landscape (Baidu ERNIE, Alibaba Qwen, ByteDance, Zhipu, Minimax).
4. **Startup velocity** -- Small team, fast shipping, broad responsibility.
5. **AI ethics and safety** -- The final round includes discussions about ethical AI development. Have a thoughtful perspective.

## Technical Assessment

### What to Expect

The take-home covers Python, NLP, ML, and AI fundamentals. It's designed to test both implementation skill and conceptual understanding.

### Reported Topics

#### NLP / Transformer Implementation

```python
import numpy as np

class PositionalEncoding:
    """Sinusoidal positional encoding for transformer models."""

    def __init__(self, d_model: int, max_len: int = 8192):
        self.encoding = np.zeros((max_len, d_model))
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        self.encoding[:, 0::2] = np.sin(position * div_term)
        self.encoding[:, 1::2] = np.cos(position * div_term)

    def __call__(self, seq_len: int) -> np.ndarray:
        return self.encoding[:seq_len]


class RoPE:
    """Rotary Position Embeddings (used in modern LLMs like LLaMA, Kimi)."""

    def __init__(self, d_model: int, base: int = 10000):
        self.d_model = d_model
        self.base = base

    def _compute_freqs(self, seq_len: int) -> np.ndarray:
        freqs = 1.0 / (self.base ** (np.arange(0, self.d_model, 2) / self.d_model))
        positions = np.arange(seq_len)
        angles = np.outer(positions, freqs)  # (seq_len, d_model/2)
        return angles

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Apply rotary embeddings to input tensor.
        x: (batch, seq_len, d_model)
        """
        seq_len = x.shape[1]
        angles = self._compute_freqs(seq_len)

        cos = np.cos(angles)  # (seq_len, d_model/2)
        sin = np.sin(angles)

        x1 = x[..., 0::2]  # Even dimensions
        x2 = x[..., 1::2]  # Odd dimensions

        # Rotate pairs
        rotated = np.zeros_like(x)
        rotated[..., 0::2] = x1 * cos - x2 * sin
        rotated[..., 1::2] = x1 * sin + x2 * cos
        return rotated
```

#### Long-Context Attention Optimization

```python
def sliding_window_attention(Q, K, V, window_size: int):
    """Sliding window attention for efficient long-context processing.

    Instead of full O(n^2) attention, each token attends to only
    the nearest `window_size` tokens.

    Q, K, V: (batch, seq_len, d_model)
    """
    batch, seq_len, d_model = Q.shape
    output = np.zeros_like(Q)

    for i in range(seq_len):
        start = max(0, i - window_size)
        end = min(seq_len, i + window_size + 1)

        q = Q[:, i:i+1, :]          # (batch, 1, d_model)
        k = K[:, start:end, :]       # (batch, window, d_model)
        v = V[:, start:end, :]       # (batch, window, d_model)

        scores = np.matmul(q, k.transpose(0, 2, 1)) / np.sqrt(d_model)
        weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        weights = weights / np.sum(weights, axis=-1, keepdims=True)

        output[:, i, :] = np.matmul(weights, v).squeeze(1)

    return output


def ring_attention(Q_chunks, K_chunks, V_chunks):
    """Ring attention for distributed long-context processing.

    Each device holds a chunk of Q and rotates K,V around a ring.
    This enables processing sequences longer than single-device memory.
    """
    num_devices = len(Q_chunks)
    outputs = [np.zeros_like(q) for q in Q_chunks]
    running_max = [np.full(q.shape[:2], -np.inf) for q in Q_chunks]
    running_sum = [np.zeros(q.shape[:2]) for q in Q_chunks]

    for step in range(num_devices):
        for device in range(num_devices):
            # Each device computes attention with current K,V chunk
            k_idx = (device + step) % num_devices
            q = Q_chunks[device]
            k = K_chunks[k_idx]
            v = V_chunks[k_idx]

            d_k = q.shape[-1]
            scores = np.matmul(q, k.transpose(0, 2, 1)) / np.sqrt(d_k)
            block_max = np.max(scores, axis=-1)  # (batch, seq_q)

            # Online softmax update
            new_max = np.maximum(running_max[device], block_max)
            exp_scores = np.exp(scores - new_max[..., np.newaxis])
            exp_sum = np.sum(exp_scores, axis=-1)

            # Rescale previous accumulated output
            scale = np.exp(running_max[device] - new_max)
            outputs[device] = outputs[device] * scale[..., np.newaxis] + np.matmul(exp_scores, v)
            running_sum[device] = running_sum[device] * scale + exp_sum
            running_max[device] = new_max

    # Normalize
    for device in range(num_devices):
        outputs[device] = outputs[device] / running_sum[device][..., np.newaxis]

    return outputs
```

#### Tokenization

```python
class BPETokenizer:
    """Byte Pair Encoding tokenizer (simplified)."""

    def __init__(self):
        self.merges: list[tuple[str, str]] = []
        self.vocab: dict[str, int] = {}

    def train(self, text: str, vocab_size: int):
        """Train BPE on text corpus."""
        # Initialize with character-level tokens
        words = [list(word) + ["</w>"] for word in text.split()]
        self.vocab = {ch: i for i, ch in enumerate(sorted(set(text + "</w>")))}

        while len(self.vocab) < vocab_size:
            # Count adjacent pairs
            pair_counts = {}
            for word in words:
                for i in range(len(word) - 1):
                    pair = (word[i], word[i + 1])
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1

            if not pair_counts:
                break

            # Find most frequent pair
            best_pair = max(pair_counts, key=pair_counts.get)
            self.merges.append(best_pair)

            # Merge the pair in all words
            merged = best_pair[0] + best_pair[1]
            self.vocab[merged] = len(self.vocab)

            new_words = []
            for word in words:
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i+1]) == best_pair:
                        new_word.append(merged)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_words.append(new_word)
            words = new_words

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        tokens = list(text) + ["</w>"]
        for a, b in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i+1] == b:
                    new_tokens.append(a + b)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return [self.vocab.get(t, 0) for t in tokens]
```

## System Design Topics

### Design a Long-Context LLM Serving System

The core challenge Moonshot faces -- serving models with 2M+ token contexts:

```
[Request (potentially 2M tokens)]
       |
  [Context Manager]
       |
  [Hierarchical KV Cache]
       |
  +----+----+
  |         |
[GPU Memory] [CPU Memory (overflow)]
  |         |
  [Attention: Sliding Window + Global Tokens]
       |
  [Token Generation]
       |
  [Streaming Response]
```

Key challenges:
- **KV cache at 2M tokens**: Doesn't fit in a single GPU's memory
  - Solution: Offload older KV cache to CPU memory, fetch on demand
  - Solution: Hierarchical attention (local window + sparse global)
- **Prefill latency**: Processing 2M input tokens takes significant time
  - Solution: Chunked prefill, pipeline across GPUs
- **Attention optimization**: O(n^2) attention is prohibitive at 2M tokens
  - Solution: Flash Attention, Ring Attention, sparse attention patterns

### Design a Multi-Modal AI Assistant (Kimi)

```
[User Input] --> [Input Router]
                      |
           +----------+----------+
           |          |          |
      [Text LLM]  [Vision]  [Document Parser]
           |          |          |
           +----------+----------+
                      |
              [Context Assembly]
                      |
              [Long-Context LLM]
                      |
              [Safety Filter]
                      |
              [Response Stream]
```

- **Document processing**: Parse PDFs, images, code files into text
- **Vision**: Multimodal understanding (image + text)
- **Long-context**: Fit entire documents into context window
- **Tool use**: Web search, code execution, file analysis

## Final Round: Ethics and Vision

### What to Expect

The final round with senior leadership (potentially CEO) discusses:

- Your vision for AI's future
- Ethical considerations in AI development
- How you'd handle specific ethical dilemmas
- Use cases for Moonshot's technology

### Sample Discussion Topics

- "How should AI companies handle model capabilities that could be misused?"
- "What's your approach to transparency in AI systems?"
- "How do you think about the balance between open-source and proprietary AI?"
- "What role should Chinese AI companies play in global AI safety?"

### How to Approach

- Be genuine and thoughtful
- Show you've thought about these issues before, not for the first time in the interview
- Acknowledge complexity and trade-offs
- Demonstrate awareness of both Chinese and global AI policy landscapes

## Preparation Tips

1. **Understand long-context techniques** -- RoPE, ALiBi, Flash Attention, Ring Attention, sliding window attention. This is Moonshot's technical edge.
2. **NLP fundamentals** -- Tokenization (BPE, SentencePiece), positional encoding, attention mechanisms. Implement from scratch.
3. **Read Yang Zhilin's papers** -- Transformer-XL, XLNet, and related work. Understanding the founder's research shows genuine interest.
4. **Know the Chinese AI landscape** -- Kimi vs. Doubao (ByteDance) vs. Tongyi Qianwen (Alibaba) vs. ERNIE (Baidu). What differentiates Moonshot?
5. **Build something with Kimi** -- If accessible, use Moonshot's APIs. Demonstrate familiarity with their products.
6. **Prepare ethics perspectives** -- The final round emphasizes AI ethics. Have well-reasoned positions.
7. **Startup mindset** -- Show you can move fast, wear multiple hats, and handle ambiguity.

## Sources

- [AI 2026: China's Moonshot AI Contends For Technical Leadership - Bismarck Analysis](https://brief.bismarckanalysis.com/p/ai-2026-chinas-moonshot-ai-contends)
- [Moonshot AI Interview Questions - Glassdoor](https://www.glassdoor.com/Interview/Moonshot-AI-Interview-Questions-E9695889.htm)
- [Moonshot AI Jobs - Ashby](https://jobs.ashbyhq.com/moonshot-ai)
- [Moonshot AI Careers - Wellfound](https://wellfound.com/company/moonshot-ai-1)
- [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context (Dai & Yang et al., 2019)](https://arxiv.org/abs/1901.02860)
- [XLNet: Generalized Autoregressive Pretraining (Yang et al., 2019)](https://arxiv.org/abs/1906.08237)
