# Netflix

Netflix is a distributed systems company that happens to stream video. Its interview reflects that: it is lighter on LeetCode than Google or Meta and much heavier on system design and culture. Netflix generally hires experienced engineers only, calibrating at L5 and above, because it wants people who can operate independently from day one, and its culture of "freedom and responsibility" means the culture round carries real weight rather than being a formality.

This chapter prepares you for Netflix engineering roles with a focus on distributed systems, streaming infrastructure, and AI and ML. It covers the fast process and Netflix's unusual compensation structure, the culture values and how to answer culture questions with candour, the system design round with four Netflix-flavoured prompts and the infrastructure Netflix has open-sourced, and the practical coding round, with a lazy data pipeline, a circuit breaker, and consistent hashing.

> **Learning objectives**
> - Describe the Netflix interview loop and its cash-first compensation model.
> - Answer culture questions in the candid, specific style Netflix values, mapped to its published values.
> - Design video delivery, a recommender, a chaos engineering platform, and an A/B testing platform, with operational detail.
> - Implement a composable lazy pipeline, a three-state circuit breaker, and a consistent hash ring with virtual nodes.

## Before you begin

Netflix assumes distributed systems fluency: read [Distributed systems](../../01-shared-concepts/01-distributed-systems.md) for CAP, consistency, and consensus. The recommendation design builds on [Full-stack AI](../../01-shared-concepts/02-full-stack-ai.md). Above all, read the [Netflix culture memo](https://jobs.netflix.com/culture) before you interview; it is non-negotiable.

## The interview process

Netflix moves fast: two to four weeks and four or five rounds.

| Round | Format | Duration | Focus |
|-------|--------|----------|-------|
| Recruiter Screen | Phone | 30 min | Background, culture fit, comp expectations |
| Hiring Manager Screen | Video | 45-60 min | Technical depth, team fit |
| Onsite 1 | System Design | 60 min | Large-scale distributed systems |
| Onsite 2 | Coding | 60 min | Practical problem-solving |
| Onsite 3 | Culture / Values | 45-60 min | Netflix culture alignment |

Netflix sometimes adds a phone technical screen before the onsite.

Compensation is top-of-market cash with a structure unlike most of the industry. There is no equity by default; compensation is primarily salary, about $350-500K for a senior SWE and $500-700K or more for a staff SWE. Employees choose what percentage of their compensation to take as stock options, post-tax. There are no bonuses, since everything is in the salary, and instead of raises Netflix makes an annual market adjustment, re-benchmarking each salary to the market every year.

## What Netflix looks for

The culture is the starting point. Netflix's "freedom and responsibility" ethos means it hires what it calls "stunning colleagues" and gives them extreme autonomy. The companion principle is "context, not control": leaders provide context, the goals and constraints, and let engineers decide how, so expect questions about how you operate with high autonomy. On the technical side, system design is king. Netflix is a distributed systems company, and its design rounds are thorough and deep. Coding is practical rather than puzzle-based, closer to "design a cache" than "find the kth element". Hiring is senior-only. And AI and ML matter for personalisation: recommendation systems, A/B testing, and content optimisation are core, and ML infrastructure is a growing area.

## Culture and values

Netflix evaluates candidates against its published values, so read the culture memo before interviewing. The values, and what each one assesses, are:

| Value | What They Assess |
|-------|-----------------|
| Judgment | Make wise decisions despite ambiguity |
| Communication | Candid, direct, and concise |
| Curiosity | Learn rapidly and eagerly |
| Courage | Say what you think, even if controversial |
| Passion | Inspire others with your drive |
| Selflessness | Seek what's best for Netflix, not yourself |
| Innovation | Challenge the status quo |
| Inclusion | Collaborate effectively with diverse people |
| Integrity | Be honest and transparent |
| Impact | Accomplish amazing amounts of important work |

Culture questions probe these directly: "Tell me about a time you disagreed with your manager and what you did.", "Describe a situation where you made an unpopular decision.", "How do you handle receiving critical feedback?", "Tell me about a time you simplified something that was over-engineered.", "Describe a time you took a risk that didn't pay off.", and "What would you do if you realized a project you championed was heading in the wrong direction?"

The key to all of them is candour. Do not give safe, polished answers; give honest, specific ones. Acknowledge mistakes directly, and show that you can disagree respectfully and change your mind when presented with better information.

## The system design round

Netflix system design differs from the generic version in four ways. The questions are often derived from real Netflix challenges. The interviewer prefers depth over breadth and will pick one area and drill into it. You are expected to make data-informed decisions, using numbers and metrics. And operational maturity counts: how you would monitor the system, what you would alert on, and how you would debug it. Four prompts are common.

### Video delivery

```
[Content Ingestion] --> [Encoding Pipeline] --> [CDN (Open Connect)]
                                                       |
[Client App] <-- [Adaptive Bitrate Selection] <-- [Edge Server]
                        |
                 [Playback Telemetry] --> [Quality of Experience (QoE) Service]
```

Each title is encoded into hundreds of variants across resolution, bitrate, and codec. Per-title encoding optimises the bitrate ladder for each title, since animation, for example, needs less bitrate than live action. Delivery runs over Open Connect, Netflix's own CDN of hardware appliances installed in ISP data centres. On the client, adaptive bitrate (ABR) selection picks stream quality from a bandwidth estimate, and buffer management prefetches segments to ride out bandwidth fluctuations. Playback telemetry feeds a quality-of-experience service that tracks rebuffer rate, startup time, and a video quality score.

### Recommendation

```
[User Events] --> [Event Stream (Kafka)] --> [Feature Computation]
                                                     |
                                              [Feature Store]
                                                     |
                                              [Candidate Generation]
                                                     |
                                              [Ranking Model]
                                                     |
                                              [Diversity / Business Rules]
                                                     |
                                              [Personalized Row Assembly]
```

Retrieval is two-stage: fast, broad candidate generation followed by slower, precise ranking. The signals combine collaborative filtering (users who watched X also watched Y), content-based features of the video (genre, cast, mood, visual style), and context (time of day, device, recent viewing history). The UI is row-based, with each row a personalised ranked list such as "Because you watched...". And everything is A/B tested, with multiple recommendation models running simultaneously.

### Chaos engineering

Netflix invented Chaos Monkey, so designing a chaos engineering platform is a natural prompt:

```
[Experiment Definition] --> [Scheduler] --> [Fault Injector]
                                                  |
                                           [Service Mesh / Infrastructure]
                                                  |
                                           [Observability Layer]
                                                  |
                                           [Automated Rollback]
```

The platform injects faults such as instance termination, network partitions, latency, and CPU stress. Blast radius is controlled by starting small, with one instance, and expanding gradually. For safety, the experiment halts automatically if the error rate exceeds a threshold. And every experiment begins with a steady-state hypothesis, a definition of what "normal" looks like, so that you can tell whether the fault changed anything.

### A real-time A/B testing platform

Users are assigned to experiments with consistent hashing, so that a user's assignment is stable. Events are ingested at high throughput through Kafka. Statistical analysis uses sequential testing rather than only fixed-horizon tests, with false discovery rate control. Each experiment defines primary and guardrail metrics, and the platform detects interactions when experiments interfere with each other.

### Netflix infrastructure

Netflix built and open-sourced much of the infrastructure behind these designs, and naming the right component shows you know the ecosystem:

| Technology | Purpose |
|-----------|---------|
| **Zuul** | API gateway, routing, load balancing |
| **Eureka** | Service discovery |
| **Hystrix** (legacy) / **Resilience4j** | Circuit breaking, fault tolerance |
| **Conductor** | Workflow orchestration |
| **Mantis** | Real-time stream processing |
| **Atlas** | Telemetry and monitoring |
| **EVCache** | Distributed caching (memcached-based) |
| **Cassandra** | Distributed NoSQL storage |
| **Kafka** | Event streaming |

## The coding round

The coding round uses practical, real-world problems judged on production quality: error handling, testing, and clean design. You may be asked to extend existing code. There is less algorithmic trickery and more engineering judgment. Three problem types are reported.

### A data processing pipeline

The first is a composable pipeline with lazy evaluation. `map` and `filter` only record stages; nothing runs until a terminal operation such as `take` or `group_by` calls `_execute`, which chains generator expressions so that each item flows through all stages before the next is read. That is why `take(n)` on an infinite source terminates: it stops pulling after `n` items.

```python
from typing import Iterator, Callable, TypeVar
from collections import defaultdict

T = TypeVar('T')
R = TypeVar('R')

class Pipeline:
    """Composable data processing pipeline with lazy evaluation."""

    def __init__(self, source: Iterator):
        self._source = source
        self._stages = []

    def map(self, fn: Callable) -> 'Pipeline':
        self._stages.append(('map', fn))
        return self

    def filter(self, fn: Callable) -> 'Pipeline':
        self._stages.append(('filter', fn))
        return self

    def group_by(self, key_fn: Callable) -> dict:
        groups = defaultdict(list)
        for item in self._execute():
            groups[key_fn(item)].append(item)
        return dict(groups)

    def take(self, n: int) -> list:
        results = []
        for item in self._execute():
            results.append(item)
            if len(results) >= n:
                break
        return results

    def _execute(self) -> Iterator:
        stream = self._source
        for stage_type, fn in self._stages:
            if stage_type == 'map':
                stream = (fn(item) for item in stream)
            elif stage_type == 'filter':
                stream = (item for item in stream if fn(item))
        yield from stream
```

Two design points are worth raising. The builder methods mutate the pipeline and return `self`, so two pipelines branched from the same object share stages; returning a new `Pipeline` each time would make them independent. And the source is an iterator, so a pipeline can be executed only once.

### A circuit breaker

The second is a circuit breaker, the pattern Hystrix made famous. It has three states. When closed, calls pass through and failures are counted; reaching `failure_threshold` consecutive failures opens the circuit. When open, calls are rejected immediately with `CircuitBreakerOpenError`, protecting a struggling dependency, until `recovery_timeout` has passed since the last failure. Then the breaker moves to half-open and lets calls through as a test: `success_threshold` successes close it again, and a single failure reopens it.

```python
import time
import threading
from enum import Enum

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0,
                 success_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.lock = threading.Lock()

    def call(self, fn, *args, **kwargs):
        with self.lock:
            if self.state == CircuitState.OPEN:
                if time.monotonic() - self.last_failure_time > self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                else:
                    raise CircuitBreakerOpenError("Circuit is open")

        try:
            result = fn(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        with self.lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
            else:
                self.failure_count = 0

    def _on_failure(self):
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.monotonic()
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
            elif self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN

class CircuitBreakerOpenError(Exception):
    pass
```

Note the locking discipline: the lock guards state transitions but is released while `fn` runs, so a slow dependency does not serialise every caller. It uses `time.monotonic()` so that wall-clock adjustments cannot open or close the circuit. A common follow-up is that half-open lets every concurrent caller through; a stricter breaker admits only a limited number of trial calls.

### Consistent hashing

The third is consistent hashing, which maps keys to nodes so that adding or removing a node moves only about $1/N$ of the keys rather than nearly all of them. Each node is placed on a ring at many points, its virtual nodes (150 here), which evens out the share of the ring each node owns. A key belongs to the first virtual node clockwise from its hash, found by binary search with wrap-around at the end of the ring.

```python
import hashlib
import bisect

class ConsistentHash:
    def __init__(self, num_virtual_nodes: int = 150):
        self.num_virtual_nodes = num_virtual_nodes
        self.ring: list[int] = []
        self.node_map: dict[int, str] = {}

    def _hash(self, key: str) -> int:
        return int(hashlib.sha256(key.encode()).hexdigest(), 16)

    def add_node(self, node: str):
        for i in range(self.num_virtual_nodes):
            h = self._hash(f"{node}:{i}")
            bisect.insort(self.ring, h)
            self.node_map[h] = node

    def remove_node(self, node: str):
        for i in range(self.num_virtual_nodes):
            h = self._hash(f"{node}:{i}")
            self.ring.remove(h)
            del self.node_map[h]

    def get_node(self, key: str) -> str:
        if not self.ring:
            raise ValueError("No nodes in ring")
        h = self._hash(key)
        idx = bisect.bisect_right(self.ring, h)
        if idx == len(self.ring):
            idx = 0
        return self.node_map[self.ring[idx]]
```

Lookup is $O(\log V)$ for $V$ virtual nodes. Insertion and removal use a sorted Python list, so each costs $O(V)$ per virtual node; that is fine for a ring that changes rarely, and it is the kind of trade-off a Netflix interviewer will expect you to state.

## Preparing

Read the Netflix culture memo, publicly available at jobs.netflix.com/culture; this is non-negotiable. Study the Netflix tech blog at netflixtechblog.com, focusing on recent infrastructure posts. Practise system design depth, since Netflix expects deeper answers than most companies; practise going three levels deep on any component. Prepare five or six strong culture stories aligned to the values and practise delivering them concisely. Know distributed systems (the CAP theorem, eventual consistency, consensus), which Netflix lives and breathes. Do not grind LeetCode: focus on medium-difficulty practical problems, where production quality matters more than algorithm tricks. And understand streaming technology: adaptive bitrate, CDN architecture, and video encoding basics.

## Summary

- Netflix hires senior engineers only and pays top-of-market cash, with no default equity or bonus.
- The culture round is decisive; answer with candour, specifics, and acknowledged mistakes.
- System design goes deep on one area and expects numbers and operational maturity.
- Coding is practical: lazy pipelines, circuit breakers, and consistent hashing, judged on production quality.

## Further reading

- [Netflix Culture Memo](https://jobs.netflix.com/culture)
- [Netflix Tech Blog](https://netflixtechblog.com/)
- [The Best and Worst Tech Giants to Interview For (Resume.io)](https://resume.io/blog/the-best-and-worst-tech-giants-to-interview-for-in-2024)
- [Glassdoor: Netflix SWE Interview Questions](https://www.glassdoor.com/Interview/Netflix-Software-Engineer-Interview-Questions-EI_IE11891.0,7_KO8,25.htm)
- [levels.fyi: Netflix Compensation Data](https://www.levels.fyi/companies/netflix/salaries/software-engineer)
- Community reports on r/cscareerquestions, r/ExperiencedDevs, and Blind.
