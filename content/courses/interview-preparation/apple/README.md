# Apple Software Engineer Interview Guide

Comprehensive preparation for Apple engineering roles, with focus on AI/ML infrastructure and platform engineering.

## Interview Process Overview

Timeline: **3-6 weeks** (can be slow), **4-6 rounds**

| Round | Format | Duration | Focus |
|-------|--------|----------|-------|
| Recruiter Screen | Phone | 30 min | Background, team matching |
| Phone Screen | Technical | 45-60 min | Coding + domain knowledge |
| Onsite 1 | Coding | 60 min | DS&A, clean implementation |
| Onsite 2 | Coding / Domain | 60 min | Domain-specific (ML, systems, etc.) |
| Onsite 3 | System Design | 60 min | Architecture, scalability |
| Onsite 4 | Behavioral / Manager | 45-60 min | Culture fit, collaboration |

Apple's process is **highly team-dependent**. Different teams have very different interview styles. The hiring manager has significant influence on the process.

## Compensation (ICT4 Senior, US)

- **Base**: $175-220K
- **Equity**: RSUs, ~$200-400K/yr (vest over 4 years, more even than Amazon)
- **Bonus**: 10-20% of base
- **Total Comp**: ~$400-650K (ICT4), ~$600K-1M+ (ICT5 Staff)
- Apple's equity has been a strong performer historically

## Key Themes

1. **Secrecy and compartmentalization** -- Apple is famously secretive. You may not know the exact project until after you're hired. Interview questions tend to be more generic.
2. **Product excellence** -- Apple cares deeply about polish, user experience, and getting details right. This philosophy extends to infrastructure and internal tools.
3. **Team-dependent** -- Each team has its own culture and technical bar. The experience varies significantly.
4. **Hardware-software integration** -- Apple's unique advantage is vertical integration. Even backend roles may touch hardware considerations.
5. **AI/ML is a priority** -- Apple Intelligence, Siri, on-device ML, Core ML, and cloud-based ML services are major investment areas.
6. **Privacy-first** -- Apple's privacy commitment affects every technical decision. Expect questions about privacy-preserving ML, on-device processing, and differential privacy.

## Coding Rounds (2 rounds)

### What to Expect

- Standard DS&A problems, similar difficulty to Google (medium to hard)
- Clean code emphasis -- Apple engineers care about code quality
- May include language-specific questions (Swift, Objective-C for platform roles; Python/C++ for ML)
- Whiteboard or shared editor

### Problem Types by Domain

#### General SWE
- Standard Leetcode medium/hard
- String manipulation, trees, graphs, DP
- System programming (file I/O, networking) for infrastructure roles

#### ML/AI Infrastructure
- Data pipeline design and implementation
- Feature extraction from structured data
- Model evaluation metrics implementation
- Efficient matrix operations

#### Platform / Systems
- Memory-efficient data structures
- Lock-free algorithms
- Custom allocators
- File system operations

### Reported Problems

#### Efficient Data Structure for Autocomplete

```python
class AutocompleteSystem:
    """Autocomplete with frequency-weighted results."""

    def __init__(self, sentences: list[str], frequencies: list[int]):
        self.freq: dict[str, int] = {}
        for s, f in zip(sentences, frequencies):
            self.freq[s] = f
        self.current = ""

    def input(self, c: str) -> list[str]:
        if c == '#':
            self.freq[self.current] = self.freq.get(self.current, 0) + 1
            self.current = ""
            return []

        self.current += c
        # Find all matching sentences
        matches = [
            (-freq, sentence)
            for sentence, freq in self.freq.items()
            if sentence.startswith(self.current)
        ]
        matches.sort()
        return [s for _, s in matches[:3]]
```

#### Thread-Safe LRU Cache with TTL

```python
import time
import threading
from collections import OrderedDict

class TTLLRUCache:
    def __init__(self, capacity: int, default_ttl: float = 300.0):
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: OrderedDict[str, tuple[any, float]] = OrderedDict()
        self.lock = threading.RLock()

    def get(self, key: str) -> any:
        with self.lock:
            if key not in self.cache:
                return None
            value, expiry = self.cache[key]
            if time.monotonic() > expiry:
                del self.cache[key]
                return None
            self.cache.move_to_end(key)
            return value

    def put(self, key: str, value: any, ttl: float = None) -> None:
        with self.lock:
            ttl = ttl or self.default_ttl
            expiry = time.monotonic() + ttl
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = (value, expiry)
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)

    def cleanup_expired(self) -> int:
        """Remove all expired entries. Returns count removed."""
        with self.lock:
            now = time.monotonic()
            expired = [k for k, (_, exp) in self.cache.items() if now > exp]
            for k in expired:
                del self.cache[k]
            return len(expired)
```

#### Privacy-Preserving Aggregation

```python
import random
import math

class DifferentialPrivacyCounter:
    """Count with differential privacy using Laplace mechanism."""

    def __init__(self, epsilon: float = 1.0):
        self.epsilon = epsilon
        self.true_counts: dict[str, int] = {}

    def increment(self, key: str) -> None:
        self.true_counts[key] = self.true_counts.get(key, 0) + 1

    def get_noisy_count(self, key: str) -> float:
        """Return count with Laplace noise for differential privacy."""
        true_count = self.true_counts.get(key, 0)
        # Sensitivity is 1 (one user changes count by at most 1)
        scale = 1.0 / self.epsilon
        noise = self._laplace(scale)
        return max(0, true_count + noise)

    def _laplace(self, scale: float) -> float:
        u = random.uniform(-0.5, 0.5)
        return -scale * math.copysign(1, u) * math.log(1 - 2 * abs(u))

    def get_noisy_top_k(self, k: int) -> list[tuple[str, float]]:
        """Return top-k keys by noisy count."""
        noisy = [(key, self.get_noisy_count(key)) for key in self.true_counts]
        noisy.sort(key=lambda x: -x[1])
        return noisy[:k]
```

## System Design Round

### Apple-Specific Considerations

- **Privacy by design**: Every system design should address data minimization and user privacy
- **On-device vs. cloud**: Apple strongly prefers on-device processing when possible
- **Vertical integration**: Consider how hardware capabilities affect your design
- **Global scale**: 2B+ active devices worldwide

### Common Topics

#### Design Apple Intelligence Backend

```
[On-Device Model] <---> [Private Cloud Compute]
       |                        |
  [Local Processing]    [Secure Enclave Processing]
       |                        |
  [On-Device Results]   [Results (no data retained)]
```

Key principles:
- **On-device first**: Run smaller models on Apple Neural Engine
- **Private Cloud Compute**: For tasks requiring larger models, use Apple's custom cloud with:
  - No persistent data storage
  - Auditable security (researchers can inspect)
  - Cryptographic verification of server code
- **Differential privacy**: Aggregate learning without seeing individual data
- **Federated learning**: Train on-device, share only gradients/updates

#### Design iCloud Photo Search

- **On-device indexing**: ML models run on-device to classify/tag photos
- **Encrypted sync**: Photos encrypted before upload, keys on device
- **Search**: Semantic search using on-device embeddings, never sending raw queries to server
- **Shared albums**: Key sharing between devices for collaborative features

#### Design Siri Backend

```
[Voice Input] --> [On-Device ASR] --> [Intent Classification]
                                              |
                                    [On-Device (simple)] or [Server (complex)]
                                              |
                                    [Action Execution]
                                              |
                                    [TTS Response]
```

- **Latency budget**: < 2 seconds end-to-end for voice queries
- **Intent routing**: Local intents (timer, weather) vs. server intents (complex queries)
- **Personalization**: On-device user model, never uploaded
- **Multi-modal**: Voice, text, visual (point at screen and ask)

#### Design the App Store Recommendation System

- **Privacy constraint**: Cannot use individual user data for recommendations
- **Differential privacy**: Aggregate download/usage patterns
- **Editorial curation**: Balance algorithmic and human-curated recommendations
- **Fraud detection**: Identify fake reviews, manipulated rankings

### On-Device ML Considerations

| Factor | On-Device | Cloud |
|--------|-----------|-------|
| Privacy | Best (data never leaves device) | Requires trust/verification |
| Latency | Lower (no network) | Higher (network round trip) |
| Model size | Limited (few GB) | Unlimited |
| Power | Constrained (battery) | Unlimited |
| Updates | Requires app/OS update | Instant |
| Availability | Works offline | Requires connectivity |

Apple's preference hierarchy: On-device > Private Cloud Compute > Standard cloud (avoid)

## Behavioral Round

### Apple Culture

| Trait | What They Assess |
|-------|-----------------|
| Attention to detail | Do you care about the small things? |
| Cross-functional collaboration | Can you work with design, HW, and QA? |
| User focus | Do you think about the end-user experience? |
| Simplicity | Can you find the simple solution? |
| Confidentiality | Can you be trusted with secrets? |

### Common Questions

- "Tell me about a product you love and why." (They want taste and design thinking)
- "Describe a time you had to simplify a complex system."
- "Tell me about a time you paid attention to a detail that others missed."
- "How do you handle working on something you can't talk about publicly?"
- "Describe a time you had to balance quality with a deadline."
- "Tell me about a cross-functional project and how you navigated it."

### What Makes Apple Different

- They ask about **products you use and admire** -- have thoughtful opinions
- **Design matters** -- even for backend roles, show you care about APIs being elegant
- **Less metric-obsessed** than Amazon -- they care about quality and craft
- **Secrecy is real** -- show you're comfortable not sharing details about your work

## AI/ML-Specific Topics

### Apple's AI Stack
- **Core ML**: On-device ML framework (model inference on Neural Engine, GPU, CPU)
- **Create ML**: Model training framework
- **Apple Neural Engine (ANE)**: Custom silicon for ML inference
- **Apple Intelligence**: On-device + Private Cloud Compute AI features
- **Private Cloud Compute**: Custom cloud infrastructure for AI with cryptographic privacy

### ML Infrastructure Design Questions
- How would you optimize a model to run on Apple Neural Engine?
- Design a federated learning system for improving keyboard predictions
- How would you build an ML pipeline that preserves differential privacy?
- Design a model serving system for on-device inference across iPhone, iPad, Mac

### Key Concepts
- **Model quantization**: FP32 -> INT8/INT4 for on-device efficiency
- **Model distillation**: Train smaller student model from larger teacher
- **Neural Architecture Search**: Optimize model architecture for specific hardware
- **Federated learning**: Train across devices without centralizing data

## Preparation Tips

1. **Use Apple products** -- Have genuine opinions about what works and what doesn't. They'll ask.
2. **Privacy-first thinking** -- In every system design, address privacy proactively. This is Apple's DNA.
3. **Know Core ML / Apple Neural Engine** -- For ML roles, understand on-device inference constraints.
4. **Study Apple's engineering blog** -- machinelearning.apple.com has excellent technical content.
5. **Practice clean code** -- Apple values elegance. Well-named functions, clean interfaces.
6. **Prepare for ambiguity** -- You may not know the exact team. Show adaptability.
7. **Don't badmouth competitors** -- Apple people respect good engineering everywhere. Be collegial.
8. **Understand vertical integration** -- Think about how HW + SW + services work together.

## Sources

- [Apple Machine Learning Research](https://machinelearning.apple.com/)
- [Apple Private Cloud Compute](https://security.apple.com/blog/private-cloud-compute/)
- [Glassdoor - Apple SWE Interview Questions](https://www.glassdoor.com/Interview/Apple-Software-Engineer-Interview-Questions-EI_IE1138.0,5_KO6,23.htm)
- [levels.fyi - Apple Compensation Data](https://www.levels.fyi/companies/apple/salaries/software-engineer)
- r/cscareerquestions, r/ExperiencedDevs, Blind (community reports)
