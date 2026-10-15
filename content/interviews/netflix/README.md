# Netflix Software Engineer Interview Guide

Comprehensive preparation for Netflix engineering roles, with focus on distributed systems, streaming infrastructure, and AI/ML.

## Interview Process Overview

Timeline: **2-4 weeks** (Netflix moves fast), **4-5 rounds**

| Round | Format | Duration | Focus |
|-------|--------|----------|-------|
| Recruiter Screen | Phone | 30 min | Background, culture fit, comp expectations |
| Hiring Manager Screen | Video | 45-60 min | Technical depth, team fit |
| Onsite 1 | System Design | 60 min | Large-scale distributed systems |
| Onsite 2 | Coding | 60 min | Practical problem-solving |
| Onsite 3 | Culture / Values | 45-60 min | Netflix culture alignment |

Netflix sometimes adds a **phone technical screen** before onsite. The process is lighter on leetcode than Google/Meta and heavier on system design and culture.

## Compensation

Netflix pays **top-of-market cash** with a unique comp structure:

- **No equity by default** -- Comp is primarily cash salary
- **Senior SWE**: $350-500K base salary
- **Staff SWE**: $500-700K+ base salary
- Employees choose what percentage of comp to take as stock options (post-tax)
- No bonuses -- it's all in the salary
- Annual market adjustment (not a raise -- they re-benchmark to market each year)

## Key Themes

1. **Freedom & Responsibility** -- Netflix's culture deck is required reading. They hire "stunning colleagues" and give extreme autonomy.
2. **Context, not control** -- Leaders provide context (goals, constraints) and let engineers decide how. Expect questions about how you operate with high autonomy.
3. **System design is king** -- Netflix is a distributed systems company. SD rounds are thorough and deep.
4. **No leetcode grinding** -- Problems tend to be practical/applied. Think "design a cache" not "find the kth element."
5. **Senior-only hiring** -- Netflix generally hires experienced engineers. Expect L5+ calibration. They want people who can operate independently from day one.
6. **AI/ML for personalization** -- Recommendation systems, A/B testing, and content optimization are core. ML infrastructure is a growing area.

## Culture Values (Critical)

Netflix evaluates against their published values. **Read the culture memo before interviewing.**

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

### Culture Interview Questions

- "Tell me about a time you disagreed with your manager and what you did."
- "Describe a situation where you made an unpopular decision."
- "How do you handle receiving critical feedback?"
- "Tell me about a time you simplified something that was over-engineered."
- "Describe a time you took a risk that didn't pay off."
- "What would you do if you realized a project you championed was heading in the wrong direction?"

**Key**: Netflix values **candor**. Don't give safe, polished answers. Give honest, specific ones. Acknowledge mistakes directly. Show you can disagree respectfully and change your mind when presented with better information.

## System Design Round

### What Makes Netflix SD Different

- **Real problems**: Questions are often derived from actual Netflix challenges
- **Depth over breadth**: They'll pick one area and drill deep
- **Data-informed decisions**: They want to see you use numbers and metrics
- **Operational maturity**: How would you monitor it? What alerts? How do you debug?

### Common Topics

#### Design Netflix Streaming (Video Delivery)

```
[Content Ingestion] --> [Encoding Pipeline] --> [CDN (Open Connect)]
                                                       |
[Client App] <-- [Adaptive Bitrate Selection] <-- [Edge Server]
                        |
                 [Playback Telemetry] --> [Quality of Experience (QoE) Service]
```

Key components:
- **Encoding**: Encode each title into hundreds of variants (resolution, bitrate, codec)
- **Per-title encoding**: Optimize bitrate ladder per title (animation needs less bitrate than live action)
- **Open Connect CDN**: Netflix's own CDN, hardware appliances in ISP data centers
- **Adaptive bitrate (ABR)**: Client selects stream quality based on bandwidth estimation
- **Buffer management**: Prefetch segments, handle bandwidth fluctuations
- **QoE metrics**: Rebuffer rate, startup time, video quality score

#### Design Netflix Recommendation System

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

- **Two-stage retrieval**: Candidate generation (fast, broad) -> Ranking (slow, precise)
- **Collaborative filtering**: Users who watched X also watched Y
- **Content-based**: Video features (genre, cast, mood, visual style)
- **Contextual**: Time of day, device, recent viewing history
- **Row-based UI**: Each row is a personalized ranked list (e.g., "Because you watched...")
- **A/B testing**: Everything is tested. Multiple recommendation models run simultaneously.

#### Design a Chaos Engineering Platform (Chaos Monkey)

```
[Experiment Definition] --> [Scheduler] --> [Fault Injector]
                                                  |
                                           [Service Mesh / Infrastructure]
                                                  |
                                           [Observability Layer]
                                                  |
                                           [Automated Rollback]
```

- **Fault types**: Instance termination, network partition, latency injection, CPU stress
- **Blast radius control**: Start small (one instance), expand gradually
- **Safety**: Automated halt if error rate exceeds threshold
- **Steady state hypothesis**: Define what "normal" looks like before the experiment

#### Design a Real-Time A/B Testing Platform

- **Assignment**: Consistent hashing for user-to-experiment assignment
- **Event collection**: High-throughput event ingestion (Kafka)
- **Statistical analysis**: Sequential testing (not just fixed-horizon), false discovery rate control
- **Metrics**: Define primary and guardrail metrics per experiment
- **Interaction detection**: Detect when experiments interfere with each other

### Netflix Infrastructure Concepts

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

## Coding Round

### What to Expect

- Practical, real-world problems
- Production quality: error handling, testing, clean design
- May involve extending existing code
- Less algorithmic trick, more engineering judgment

### Reported Problem Types

#### Data Processing Pipeline

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

#### Circuit Breaker Implementation

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

#### Consistent Hashing

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

## Preparation Tips

1. **Read the Netflix culture memo** -- This is non-negotiable. It's public: jobs.netflix.com/culture
2. **Study Netflix tech blog** -- netflixtechblog.com. Focus on recent posts about infrastructure.
3. **System design depth** -- Netflix expects deeper SD answers than most companies. Practice going 3 levels deep on any component.
4. **Prepare culture stories** -- Have 5-6 strong examples aligned to Netflix values. Practice delivering them concisely.
5. **Know distributed systems** -- CAP theorem, eventual consistency, consensus -- Netflix lives and breathes this.
6. **Don't grind leetcode** -- Focus on medium-difficulty practical problems. Production quality > algorithm tricks.
7. **Understand streaming tech** -- Adaptive bitrate, CDN architecture, video encoding basics.

## Sources

- [Netflix Culture Memo](https://jobs.netflix.com/culture)
- [Netflix Tech Blog](https://netflixtechblog.com/)
- [The Best and Worst Tech Giants to Interview For - Resume.io](https://resume.io/blog/the-best-and-worst-tech-giants-to-interview-for-in-2024)
- [Glassdoor - Netflix SWE Interview Questions](https://www.glassdoor.com/Interview/Netflix-Software-Engineer-Interview-Questions-EI_IE11891.0,7_KO8,25.htm)
- [levels.fyi - Netflix Compensation Data](https://www.levels.fyi/companies/netflix/salaries/software-engineer)
- r/cscareerquestions, r/ExperiencedDevs, Blind (community reports)
