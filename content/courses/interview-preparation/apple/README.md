# Interviewing at Apple

Apple's engineering interviews are shaped by three things that set the company apart: secrecy, a fixation on product quality, and a privacy commitment that reaches into every technical decision. You may not learn exactly what you would work on until after you are hired, so the questions tend to be more generic than at other companies. But the bar for clean code is high, the design questions expect you to keep data on the device whenever you can, and the behavioral round wants evidence that you care about details and about the person using the product. This chapter prepares you for that combination, with particular attention to AI and ML infrastructure roles.

The chapter starts with the process, which varies more by team than at most companies, and with compensation. It then sets out the themes that run through the loop, works through the coding rounds with three reported problems (frequency-weighted autocomplete, a thread-safe LRU cache with TTLs, and a differentially private counter), and moves on to privacy-first system design, from Apple Intelligence and Private Cloud Compute to Siri and the App Store. It closes with the behavioral round, Apple's AI stack, and a preparation plan.

> **Learning objectives**
> - Describe Apple's team-dependent interview process and the themes that run through it.
> - Implement frequency-weighted autocomplete, a thread-safe LRU cache with TTL expiry, and a Laplace-mechanism differential privacy counter, and explain their limitations.
> - Design systems that prefer on-device processing, then Private Cloud Compute, then standard cloud, and justify the tradeoffs.
> - Explain differential privacy, federated learning, quantization, and distillation as tools for privacy-preserving, on-device ML.
> - Prepare behavioral answers that show attention to detail, taste, simplicity, and discretion.

## Before you begin

The coding rounds assume solid data structures and algorithms at roughly Google's difficulty. The thread-safe cache builds on the locking ideas in [Concurrency patterns](../anthropic/concepts/concurrency-patterns.md), and the ML design questions build on [ML infrastructure](../shared-concepts/ml-infrastructure.md) and [Full-stack AI engineering](../shared-concepts/full-stack-ai.md) in this book.

## The interview process

The process takes three to six weeks and can be slow. It has four to six rounds:

| Round | Format | Duration | Focus |
|-------|--------|----------|-------|
| Recruiter Screen | Phone | 30 min | Background, team matching |
| Phone Screen | Technical | 45-60 min | Coding + domain knowledge |
| Onsite 1 | Coding | 60 min | DS&A, clean implementation |
| Onsite 2 | Coding / Domain | 60 min | Domain-specific (ML, systems, etc.) |
| Onsite 3 | System Design | 60 min | Architecture, scalability |
| Onsite 4 | Behavioral / Manager | 45-60 min | Culture fit, collaboration |

Apple's process is highly team-dependent. Different teams have very different interview styles, and the hiring manager has significant influence over the process.

## Compensation

For a senior engineer (ICT4) in the US, base salary is $175-220K. RSUs add roughly $200-400K a year, vesting over four years on a more even schedule than Amazon's, and the bonus is 10-20% of base. Total compensation is about $400-650K at ICT4 and about $600K-1M or more at ICT5 (Staff). Apple's equity has historically performed well.

## What Apple is looking for

Six themes run through the loop. **Secrecy and compartmentalization** come first: Apple is famously secretive, you may not know the exact project until after you are hired, and interview questions tend to be generic as a result. **Product excellence** is the second: Apple cares deeply about polish, user experience, and getting details right, and that philosophy extends to infrastructure and internal tools. Third, the experience is **team-dependent**, because each team has its own culture and technical bar. Fourth, **hardware-software integration** is Apple's distinctive advantage, so even backend roles may touch hardware considerations. Fifth, **AI and ML are a priority**, with Apple Intelligence, Siri, on-device ML, Core ML, and cloud ML services all major investment areas. And sixth, Apple is **privacy-first**: its privacy commitment affects every technical decision, so expect questions about privacy-preserving ML, on-device processing, and differential privacy.

## The coding rounds

There are two coding rounds. The problems are standard data structures and algorithms, at medium-to-hard difficulty similar to Google's, on a whiteboard or in a shared editor. Code quality matters to Apple engineers, so write clean code. You may also get language-specific questions: Swift or Objective-C for platform roles, Python or C++ for ML.

The problem mix depends on the domain. General software engineering roles get standard LeetCode medium and hard problems in strings, trees, graphs, and dynamic programming, plus systems programming (file I/O, networking) for infrastructure roles. ML and AI infrastructure roles get data pipeline design and implementation, feature extraction from structured data, implementations of model evaluation metrics, and efficient matrix operations. Platform and systems roles get memory-efficient data structures, lock-free algorithms, custom allocators, and file system operations. Three reported problems show the flavor.

### Frequency-weighted autocomplete

The first asks for an autocomplete system. It is initialized with historical sentences and their frequencies. As the user types, each character returns the top three historical sentences that start with the current input, ranked by frequency with ties broken alphabetically. The character `#` ends the sentence, records it in the history, and resets the input.

The straightforward solution stores frequencies in a dictionary and, on each keystroke, collects every sentence matching the prefix. Sorting `(-freq, sentence)` tuples gives descending frequency with alphabetical tie-breaking in one sort.

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

Each keystroke scans every stored sentence, which costs $O(n \cdot L)$ for $n$ sentences of length up to $L$, plus the sort. That is fine as a first answer, and it sets up the follow-up you should offer: a trie keyed by character, where each node can cache its top three sentences so a keystroke costs $O(1)$ after following one edge, at the price of updating cached lists along the path whenever a sentence is recorded.

### A thread-safe LRU cache with TTL

The second combines least-recently-used eviction with per-entry expiry. Each entry stores its value and an absolute expiry time computed from `time.monotonic()`, which, unlike wall-clock time, never jumps backward when the system clock is adjusted. `get` treats an expired entry as a miss and deletes it; `put` refreshes recency and evicts the least recently used entry past capacity; and `cleanup_expired` sweeps all expired entries for callers that want to reclaim memory eagerly.

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

The lock is an `RLock`, which costs a little more than a plain `Lock` but means a method can safely call another locked method, for example if `put` later grows to call `cleanup_expired`. Two details are worth raising yourself. `ttl or self.default_ttl` treats an explicit TTL of `0` as "use the default"; test `ttl is None` if zero should mean "expire immediately." And eviction by capacity removes the least recently used entry even when an *expired* entry is sitting elsewhere in the cache; sweeping expired entries before evicting would keep more live data.

### Privacy-preserving aggregation

The third problem is distinctly Apple: count events with **differential privacy**, so that the published counts reveal almost nothing about any individual. The standard tool is the **Laplace mechanism**. If one user can change a count by at most 1 (the *sensitivity*), then adding noise drawn from a Laplace distribution with scale $b = 1/\varepsilon$ makes the released count $\varepsilon$-differentially private. Smaller $\varepsilon$ means more noise and stronger privacy.

The noise is sampled by inverting the Laplace CDF. Draw $u$ uniformly from $(-\tfrac{1}{2}, \tfrac{1}{2})$; then $-b \cdot \operatorname{sgn}(u) \cdot \ln(1 - 2|u|)$ is Laplace-distributed with scale $b$.

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

Clamping at zero with `max(0, ...)` is post-processing of a private value, so it does not weaken the guarantee. `get_noisy_top_k` ranks keys by noisy rather than true counts, so the ranking itself leaks nothing beyond what the noisy counts reveal.

> **Warning:** Privacy budgets compose. Each call to `get_noisy_count` for the same key spends another $\varepsilon$, and an attacker who queries repeatedly can average the noise away. A production system tracks the total budget spent, or caches and re-serves the same noisy answer. Also note that `random` is not a cryptographically secure source; real deployments use a secure generator, since predictable noise can be subtracted.

## The system design round

Apple's design round adds four considerations to the usual concerns of scale and reliability. Practice **privacy by design**, addressing data minimization and user privacy in every system. Prefer **on-device processing** to the cloud whenever possible. Consider **vertical integration**: how hardware capabilities change your design. And remember the **global scale** of more than two billion active devices.

The tradeoff between on-device and cloud processing underlies every Apple design question:

| Factor | On-Device | Cloud |
|--------|-----------|-------|
| Privacy | Best (data never leaves device) | Requires trust/verification |
| Latency | Lower (no network) | Higher (network round trip) |
| Model size | Limited (few GB) | Unlimited |
| Power | Constrained (battery) | Unlimited |
| Updates | Requires app/OS update | Instant |
| Availability | Works offline | Requires connectivity |

Apple's preference order follows from the first row: on-device first, then Private Cloud Compute, and standard cloud processing avoided where possible.

### The Apple Intelligence backend

For "Design the Apple Intelligence backend," the architecture is a split between the device and Apple's private cloud:

```text
[On-Device Model] <---> [Private Cloud Compute]
       |                        |
  [Local Processing]    [Secure Enclave Processing]
       |                        |
  [On-Device Results]   [Results (no data retained)]
```

Work **on-device first**, running smaller models on the Apple Neural Engine. For tasks that need larger models, use **Private Cloud Compute**, Apple's custom cloud, which has three properties: no persistent data storage, auditable security that researchers can inspect, and cryptographic verification of the server code the device is talking to. Use **differential privacy** for aggregate learning without seeing individual data, and **federated learning** to train on devices and share only gradients or model updates.

### iCloud photo search

For iCloud photo search, the principle is the same: the server should never see what the user sees. ML models run **on the device** to classify and tag photos. Photos are **encrypted before upload**, with the keys kept on the device. **Search** is semantic search over on-device embeddings, so raw queries are never sent to a server. And **shared albums** work by sharing keys between devices.

### The Siri backend

For Siri, the pipeline routes each request to the device or the server depending on its complexity:

```text
[Voice Input] --> [On-Device ASR] --> [Intent Classification]
                                              |
                                    [On-Device (simple)] or [Server (complex)]
                                              |
                                    [Action Execution]
                                              |
                                    [TTS Response]
```

The **latency budget** is under two seconds end to end for a voice query. **Intent routing** separates local intents, such as timers and weather, from server intents for complex queries. **Personalization** uses an on-device user model that is never uploaded. And the system is **multi-modal**, combining voice, text, and visual input, such as pointing at the screen and asking about it.

### App Store recommendations

The App Store recommendation problem inverts the usual starting point: you **cannot use individual user data** for recommendations. Instead, aggregate download and usage patterns with **differential privacy**; balance algorithmic recommendations with **editorial curation**; and invest in **fraud detection** to identify fake reviews and manipulated rankings.

## The behavioral round

Apple's behavioral round assesses five traits:

| Trait | What They Assess |
|-------|-----------------|
| Attention to detail | Do you care about the small things? |
| Cross-functional collaboration | Can you work with design, HW, and QA? |
| User focus | Do you think about the end-user experience? |
| Simplicity | Can you find the simple solution? |
| Confidentiality | Can you be trusted with secrets? |

Expect questions such as "Tell me about a product you love and why," which is looking for taste and design thinking; "Describe a time you had to simplify a complex system"; "Tell me about a time you paid attention to a detail that others missed"; "How do you handle working on something you can't talk about publicly?"; "Describe a time you had to balance quality with a deadline"; and "Tell me about a cross-functional project and how you navigated it."

Four things make Apple's version different. Interviewers ask about **products you use and admire**, so have thoughtful opinions. **Design matters** even for backend roles, so show that you care about elegant APIs. Apple is **less metric-obsessed** than Amazon and cares more about quality and craft. And **secrecy is real**, so show that you are comfortable not sharing details about your work.

## AI and ML topics

For ML roles, know Apple's AI stack. **Core ML** is the on-device ML framework, running inference on the Neural Engine, GPU, or CPU. **Create ML** is the model training framework. The **Apple Neural Engine (ANE)** is custom silicon for ML inference. **Apple Intelligence** is the family of AI features built on on-device models and Private Cloud Compute. And **Private Cloud Compute** is the custom cloud infrastructure that provides those features with cryptographic privacy guarantees.

Reported ML infrastructure questions include optimizing a model to run on the Apple Neural Engine, designing a federated learning system to improve keyboard predictions, building an ML pipeline that preserves differential privacy, and designing a model serving system for on-device inference across iPhone, iPad, and Mac.

Four techniques come up in answering them. **Quantization** converts weights from FP32 to INT8 or INT4 for on-device efficiency. **Distillation** trains a smaller student model to imitate a larger teacher. **Neural architecture search** optimizes a model's architecture for specific hardware. And **federated learning** trains across devices without centralizing the data.

## Preparing well

Use Apple products and have genuine opinions about what works and what does not, because you will be asked. Think privacy first, addressing privacy proactively in every design. For ML roles, understand Core ML and the constraints of on-device inference on the Neural Engine. Study Apple's machine learning research site, which has excellent technical content. Practice clean code, with well-named functions and clean interfaces. Prepare for ambiguity, since you may not know the exact team, and show adaptability. Do not badmouth competitors; Apple engineers respect good engineering everywhere. And think about vertical integration: how hardware, software, and services work together.

## Key terms

- **Private Cloud Compute**: Apple's cloud for larger AI models, with no persistent storage, auditable security, and cryptographically verified server code.
- **Differential privacy**: a guarantee that a released statistic changes very little whether or not any one individual's data is included, parameterized by $\varepsilon$.
- **Laplace mechanism**: achieving $\varepsilon$-differential privacy by adding Laplace noise with scale sensitivity$/\varepsilon$.
- **Privacy budget**: the total $\varepsilon$ spent across queries; repeated queries compose and weaken privacy.
- **Federated learning**: training a shared model on user devices and sending only model updates to the server.
- **Quantization**: reducing numerical precision (for example, FP32 to INT8) to shrink and speed up a model.

## Summary

- Apple's loop is team-dependent and secretive; expect generic questions, a high bar for clean code, and a strong product and privacy sensibility.
- Autocomplete by scanning works as a first answer; a trie with cached top results is the scalable follow-up.
- A TTL LRU cache needs monotonic time, a lock around every operation, and care with zero TTLs and expired entries.
- The Laplace mechanism adds noise with scale $1/\varepsilon$ for sensitivity 1, and budgets compose across repeated queries.
- Design on-device first, then Private Cloud Compute, avoiding standard cloud; use differential privacy and federated learning for aggregate learning.

## Review questions

1. Why does the TTL cache compute expiry with `time.monotonic()` rather than `time.time()`?
   <details><summary>Answer</summary>Monotonic time never goes backward. If the wall clock is adjusted (for example by NTP), <code>time.time()</code> can jump, making entries expire early or live far too long.</details>
2. What is the per-keystroke cost of the scanning autocomplete, and how would you improve it?
   <details><summary>Answer</summary>Each keystroke scans every sentence, costing $O(n \cdot L)$ plus a sort of the matches. A trie lets you follow one edge per keystroke; caching the top three sentences at each node makes the lookup $O(1)$, at the cost of updating those caches when a sentence is recorded.</details>
3. For a count with sensitivity 1, what Laplace scale gives $\varepsilon = 0.5$, and what does lowering $\varepsilon$ do?
   <details><summary>Answer</summary>The scale is $1/\varepsilon = 2$. Lowering $\varepsilon$ increases the noise, strengthening privacy at the cost of accuracy.</details>
4. An analyst calls `get_noisy_count("app_x")` 1,000 times and averages the results. What has gone wrong?
   <details><summary>Answer</summary>Each call spends another $\varepsilon$ of privacy budget, and averaging independent noise samples converges on the true count. The system must track and cap the budget, or return a cached noisy answer for repeated queries.</details>
5. In a Siri design, which intents would you handle on the device, and why?
   <details><summary>Answer</summary>Simple, well-defined intents such as timers and weather, because on-device handling is private, fast (no network round trip), and works offline. Complex queries that need larger models are routed to the server, ideally to Private Cloud Compute.</details>
6. How can the App Store recommend apps without using individual user data?
   <details><summary>Answer</summary>Aggregate download and usage patterns with differential privacy, combine them with editorial curation, and use fraud detection to keep rankings honest. Individual behavior is never used directly.</details>

## Further reading

- [Apple Machine Learning Research](https://machinelearning.apple.com/)
- [Apple Private Cloud Compute](https://security.apple.com/blog/private-cloud-compute/)
- [Glassdoor: Apple software engineer interview questions](https://www.glassdoor.com/Interview/Apple-Software-Engineer-Interview-Questions-EI_IE1138.0,5_KO6,23.htm)
- [levels.fyi: Apple compensation data](https://www.levels.fyi/companies/apple/salaries/software-engineer)
- Community reports on r/cscareerquestions, r/ExperiencedDevs, and Blind.
