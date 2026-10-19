# Moonshot AI

Moonshot AI is a Chinese AI startup building frontier large language models, including the Kimi assistant. It was founded in March 2023 by Yang Zhilin, co-creator of Transformer-XL and XLNet, has raised over $1.2B at a valuation above $3B, and competes with Baidu, Alibaba, and ByteDance in China's LLM market. Its technical signature is long context: Kimi supports context windows of 2M tokens and more.

Two consequences shape the interview. The founder's research background sets a very high bar for ML understanding, and the long-context focus means you should expect to reason about attention, positional encodings, and KV cache memory at extreme sequence lengths. This chapter covers the fast, startup-style process, the take-home assessment with implementations of positional encodings, RoPE, sliding-window and ring attention, and BPE tokenization, two system design problems at the heart of Moonshot's product, and the final round on ethics and vision.

> **Learning objectives**
> - Describe Moonshot's interview loop and what distinguishes it from Big Tech processes.
> - Implement sinusoidal positional encoding and rotary position embeddings, and explain how RoPE encodes relative position.
> - Explain sliding-window attention and ring attention, including the online softmax that ring attention relies on.
> - Train and apply a simple BPE tokenizer.
> - Design a serving system for 2M-token contexts and a multimodal assistant, naming the memory and latency bottlenecks.

## Before you begin

You should be comfortable with scaled dot-product attention and softmax; the [Google DeepMind](../deepmind/README.md) chapter implements both from scratch. The serving material builds on [ML infrastructure](../shared-concepts/ml-infrastructure.md) and on the inference serving problem in the [Anthropic system design](../anthropic/03-system-design.md) section.

## The interview process

Moonshot moves at startup speed: two to three weeks and four or five rounds.

| Round | Format | Duration | Focus |
|-------|--------|----------|-------|
| Recruiter Screen | Phone | 30 min | Background, motivation |
| Take-Home Assessment | Technical | 2-3 hours | Python, NLP, ML, AI |
| Technical Interview | Video/Onsite | 60-90 min | Coding + ML depth |
| Behavioral + Team Fit | Video/Onsite | 45-60 min | Collaboration, AI ethics |
| Final Round | Panel (CEO/HR) | 45-60 min | Vision alignment, use cases, ethics |

Offers typically come within two or three days of the final round. Compensation figures are estimates: pay is competitive with top Chinese tech companies, with a significant equity component, as you would expect of an early-stage, high-growth startup. US-based roles, where they exist, would benchmark against Bay Area AI startups. Given the valuation and growth trajectory, equity upside is the primary draw.

## What Moonshot looks for

Long-context expertise comes first, because it is Moonshot's technical edge: understand attention optimisation, efficient KV caching, and memory management at extreme sequence lengths. NLP and ML depth comes next; with Yang Zhilin's research background, the bar is very high. You should understand the Chinese AI ecosystem and its competitive landscape (Baidu ERNIE, Alibaba Qwen, ByteDance, Zhipu, Minimax). Startup velocity matters: a small team, fast shipping, and broad responsibility. And AI ethics and safety are discussed explicitly in the final round, so have a thoughtful perspective.

## The take-home assessment

The take-home covers Python, NLP, ML, and AI fundamentals, and is designed to test both implementation skill and conceptual understanding. Three areas are reported.

### Positional encodings

A Transformer has no built-in notion of order, so position must be injected. The original sinusoidal encoding adds a fixed vector to each token embedding, with even dimensions using sine and odd dimensions cosine, at frequencies that fall geometrically from $1$ to $1/10000$ across the dimensions. Rotary position embeddings (RoPE), used in modern LLMs such as LLaMA and Kimi, instead rotate each pair of dimensions of the query and key vectors by an angle proportional to the token's position. Because the dot product of two rotated vectors depends only on the difference of their angles, attention scores end up depending on relative position, which is part of why RoPE extends better to long contexts.

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

The two lines under `# Rotate pairs` are a 2-D rotation matrix applied to each (even, odd) pair: $\begin{pmatrix}\cos\theta & -\sin\theta\\ \sin\theta & \cos\theta\end{pmatrix}$, with $\theta$ the position times that pair's frequency.

### Long-context attention

Full attention costs $O(n^2)$ in the sequence length, which is prohibitive at millions of tokens. Two techniques attack the problem from different directions.

Sliding-window attention restricts each token to its neighbours. In this implementation token $i$ attends to tokens from $i - w$ to $i + w$, a window of up to $2w + 1$ positions in both directions, so the cost becomes $O(n \cdot w)$. (A causal language model would use only the left half of the window.) The softmax is computed with the usual max subtraction for stability.

Ring attention keeps full attention but distributes it. Each device holds one chunk of the queries, and the key and value chunks rotate around a ring of devices, so after one full rotation every query chunk has seen every key chunk without any device holding the whole sequence. This enables sequences longer than a single device's memory. The trick that makes it work is the online softmax: you cannot normalise until you have seen all keys, so each device keeps a running maximum and a running sum of exponentials. When a new block raises the maximum, the previously accumulated output and sum are rescaled by $e^{m_\text{old} - m_\text{new}}$ before the new block's contribution is added, and the final division by the running sum happens only at the end. This is the same mechanism Flash Attention uses within one device.

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

The listing simulates the ring in a single process: the inner loop over devices stands for work that happens in parallel, and `k_idx = (device + step) % num_devices` is which key-value chunk has arrived at each device on each step. On the first step `running_max` is $-\infty$, so `scale` is $e^{-\infty} = 0$ and the zero-initialised output is simply replaced.

### Tokenization

Byte pair encoding (BPE) builds a vocabulary bottom-up. Training starts from characters, with an end-of-word marker `</w>` appended to every word, then repeatedly counts adjacent pairs across the corpus, merges the most frequent pair into a new token, and records the merge, until the vocabulary reaches the target size. Encoding replays the recorded merges in the order they were learned.

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

This is deliberately simplified, and an interviewer may ask you to spot where. The initial vocabulary is built from `set(text + "</w>")`, which adds the characters `<`, `/`, `w`, `>` individually rather than the single `</w>` token, so the marker itself is missing from the vocabulary. `encode` treats its whole input as a single word, rather than splitting on whitespace as `train` does. And unknown tokens map to id `0`, which collides with a real token. Production tokenizers work on bytes, so that nothing is ever unknown.

## System design

Moonshot's design questions come straight from its product.

### Serving 2M-token contexts

The core challenge Moonshot faces is serving models with contexts of 2M tokens and more:

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

Three problems dominate, each with known solutions. First, the KV cache for 2M tokens does not fit in a single GPU's memory. You can offload older KV entries to CPU memory and fetch them on demand, and you can use hierarchical attention, a local window plus sparse global tokens, so that most of the cache is touched rarely. Second, prefill latency: processing 2M input tokens takes significant time, which chunked prefill and pipelining across GPUs address. Third, attention itself: $O(n^2)$ is prohibitive at this length, so the system relies on Flash Attention, Ring Attention, and sparse attention patterns.

### A multimodal assistant (Kimi)

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

An input router sends each part of the request to the right handler. Document processing parses PDFs, images, and code files into text; the vision path provides multimodal understanding of images with text. The long context is what lets whole documents fit into the window, so context assembly can often include a document entire rather than retrieving fragments. Tool use (web search, code execution, file analysis) rounds out the assistant, and a safety filter sits before the response stream.

## The final round: ethics and vision

The final round, with senior leadership and potentially the CEO, discusses your vision for AI's future, ethical considerations in AI development, how you would handle specific ethical dilemmas, and use cases for Moonshot's technology. Sample topics include "How should AI companies handle model capabilities that could be misused?", "What's your approach to transparency in AI systems?", "How do you think about the balance between open-source and proprietary AI?", and "What role should Chinese AI companies play in global AI safety?"

Approach these genuinely and thoughtfully. Show that you have thought about the issues before, not for the first time in the interview. Acknowledge complexity and trade-offs rather than offering slogans, and demonstrate awareness of both the Chinese and the global AI policy landscapes.

## Preparing

Understand the long-context techniques, since they are Moonshot's technical edge: RoPE, ALiBi, Flash Attention, Ring Attention, and sliding-window attention. Implement the NLP fundamentals from scratch: tokenization (BPE, SentencePiece), positional encoding, and attention. Read Yang Zhilin's papers, Transformer-XL, XLNet, and related work; understanding the founder's research shows genuine interest. Know the Chinese AI landscape well enough to say what differentiates Kimi from Doubao (ByteDance), Tongyi Qianwen (Alibaba), and ERNIE (Baidu). If you can, build something with Moonshot's APIs to show familiarity with the products. Prepare well-reasoned positions on AI ethics for the final round. And show a startup mindset: you can move fast, wear multiple hats, and handle ambiguity.

## Summary

- Moonshot's process is fast, about two to three weeks, with a take-home and a leadership final round.
- Long context is the company's edge; know RoPE, sliding-window and ring attention, and the online softmax.
- Be able to implement and critique BPE and positional encodings from scratch.
- Serving 2M-token contexts is bounded by KV cache memory, prefill latency, and quadratic attention.
- The final round expects considered, informed views on AI ethics and policy.

## Further reading

- [AI 2026: China's Moonshot AI Contends For Technical Leadership (Bismarck Analysis)](https://brief.bismarckanalysis.com/p/ai-2026-chinas-moonshot-ai-contends)
- [Moonshot AI Interview Questions (Glassdoor)](https://www.glassdoor.com/Interview/Moonshot-AI-Interview-Questions-E9695889.htm)
- [Moonshot AI Jobs (Ashby)](https://jobs.ashbyhq.com/moonshot-ai)
- [Moonshot AI Careers (Wellfound)](https://wellfound.com/company/moonshot-ai-1)
- [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context (Dai & Yang et al., 2019)](https://arxiv.org/abs/1901.02860)
- [XLNet: Generalized Autoregressive Pretraining (Yang et al., 2019)](https://arxiv.org/abs/1906.08237)
