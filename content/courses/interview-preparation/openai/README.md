# OpenAI Software Engineer Interview Guide

Comprehensive preparation for OpenAI engineering roles, with focus on AI infrastructure and full-stack development.

> **Preparation resources**: [OpenAI Interview Guide](https://openai.com/interview-guide/) (official), [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/) (OpenAI's own RL curriculum), [Deep Learning track](../../machine-learning/02-deep-learning/), [Reinforcement Learning track](../../machine-learning/03-reinforcement-learning/)

## Interview Process Overview

Timeline: **3-5 weeks**, **5-6 rounds**

| Round | Format | Duration | Focus |
|-------|--------|----------|-------|
| Recruiter Screen | Phone | 30 min | Background, motivation, role fit |
| Technical Phone Screen | Coding (CoderPad) | 60 min | Algorithms + systems thinking |
| Onsite 1 | Coding | 60 min | Production-quality implementation |
| Onsite 2 | System Design | 60 min | ML infrastructure, API design |
| Onsite 3 | Coding / ML | 60 min | Applied ML or infra coding |
| Onsite 4 | Hiring Manager / Culture | 45-60 min | Mission alignment, collaboration |

## Compensation (Reported Ranges)

- **Senior SWE (L4)**: ~$450-600K total comp
- **Staff SWE (L5)**: ~$600-900K total comp
- **Equity**: Profit Participation Units (PPUs) -- unique to OpenAI's capped-profit structure
- OpenAI's comp is extremely competitive, especially equity upside

## Key Themes

1. **Mission-driven hiring** -- OpenAI cares deeply that you believe in their mission of safe AGI. Genuine passion for AI safety and alignment matters.
2. **Startup speed, massive scale** -- They ship fast but at enormous scale. Expect questions about pragmatic trade-offs.
3. **API-first thinking** -- ChatGPT and the API are core products. API design, rate limiting, and developer experience are central.
4. **Full-stack AI** -- Many roles span from model training infrastructure to user-facing products. Be ready to discuss both.
5. **Python-heavy** -- The codebase is heavily Python. Deep Python knowledge (asyncio, typing, packaging) is expected.

## Coding Rounds (2-3 rounds)

### What to Expect

- Similar to Anthropic in spirit: production-quality code over leetcode tricks
- Problems tend to be more applied/systems-oriented than pure algorithms
- Concurrency, API design, and data pipeline problems are common
- You may be asked to extend an existing codebase rather than start from scratch

### Reported Problem Types

#### 1. API Rate Limiter

Design and implement a rate limiter supporting multiple strategies.

```python
import time
import threading
from collections import deque

class SlidingWindowRateLimiter:
    """Rate limiter using sliding window log."""

    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: dict[str, deque] = {}
        self.lock = threading.Lock()

    def allow(self, client_id: str) -> bool:
        now = time.monotonic()
        with self.lock:
            if client_id not in self.requests:
                self.requests[client_id] = deque()

            window = self.requests[client_id]

            # Evict expired entries
            while window and window[0] <= now - self.window_seconds:
                window.popleft()

            if len(window) < self.max_requests:
                window.append(now)
                return True
            return False


class TokenBucketRateLimiter:
    """Token bucket -- smoother rate limiting for API usage."""

    def __init__(self, tokens_per_second: float, bucket_size: int):
        self.rate = tokens_per_second
        self.bucket_size = bucket_size
        self.buckets: dict[str, tuple[float, float]] = {}  # client -> (tokens, last_refill)
        self.lock = threading.Lock()

    def allow(self, client_id: str, tokens: int = 1) -> bool:
        now = time.monotonic()
        with self.lock:
            if client_id not in self.buckets:
                self.buckets[client_id] = (self.bucket_size, now)

            current_tokens, last_refill = self.buckets[client_id]
            elapsed = now - last_refill
            current_tokens = min(self.bucket_size, current_tokens + elapsed * self.rate)

            if current_tokens >= tokens:
                self.buckets[client_id] = (current_tokens - tokens, now)
                return True
            self.buckets[client_id] = (current_tokens, now)
            return False
```

#### 2. Async Task Queue with Retries

```python
import asyncio
import random
from dataclasses import dataclass, field
from enum import Enum

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"

@dataclass
class Task:
    id: str
    fn: callable
    args: tuple = ()
    max_retries: int = 3
    retry_count: int = 0
    status: TaskStatus = TaskStatus.PENDING
    result: any = None
    error: Exception = None

class AsyncTaskQueue:
    def __init__(self, max_concurrent: int = 10):
        self.queue: asyncio.Queue[Task] = asyncio.Queue()
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.tasks: dict[str, Task] = {}
        self.results: dict[str, any] = {}

    async def submit(self, task: Task) -> str:
        self.tasks[task.id] = task
        await self.queue.put(task)
        return task.id

    async def worker(self):
        while True:
            task = await self.queue.get()
            async with self.semaphore:
                await self._execute(task)
            self.queue.task_done()

    async def _execute(self, task: Task):
        task.status = TaskStatus.RUNNING
        try:
            if asyncio.iscoroutinefunction(task.fn):
                task.result = await task.fn(*task.args)
            else:
                task.result = task.fn(*task.args)
            task.status = TaskStatus.COMPLETED
            self.results[task.id] = task.result
        except Exception as e:
            task.error = e
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.RETRYING
                # Exponential backoff with jitter
                delay = (2 ** task.retry_count) + random.uniform(0, 1)
                await asyncio.sleep(delay)
                await self.queue.put(task)
            else:
                task.status = TaskStatus.FAILED

    async def run(self, num_workers: int = 5):
        workers = [asyncio.create_task(self.worker()) for _ in range(num_workers)]
        await self.queue.join()
        for w in workers:
            w.cancel()
```

#### 3. JSON Schema Validator

```python
def validate(instance, schema: dict) -> list[str]:
    """Validate a JSON instance against a JSON schema. Return list of errors."""
    errors = []
    _validate(instance, schema, errors, path="$")
    return errors

def _validate(instance, schema: dict, errors: list, path: str):
    schema_type = schema.get("type")

    if schema_type == "object":
        if not isinstance(instance, dict):
            errors.append(f"{path}: expected object, got {type(instance).__name__}")
            return
        # Required fields
        for req in schema.get("required", []):
            if req not in instance:
                errors.append(f"{path}: missing required field '{req}'")
        # Property validation
        properties = schema.get("properties", {})
        for key, value in instance.items():
            if key in properties:
                _validate(value, properties[key], errors, f"{path}.{key}")
            elif not schema.get("additionalProperties", True):
                errors.append(f"{path}: unexpected field '{key}'")

    elif schema_type == "array":
        if not isinstance(instance, list):
            errors.append(f"{path}: expected array, got {type(instance).__name__}")
            return
        items_schema = schema.get("items", {})
        for i, item in enumerate(instance):
            _validate(item, items_schema, errors, f"{path}[{i}]")
        if "minItems" in schema and len(instance) < schema["minItems"]:
            errors.append(f"{path}: array has {len(instance)} items, minimum is {schema['minItems']}")

    elif schema_type == "string":
        if not isinstance(instance, str):
            errors.append(f"{path}: expected string, got {type(instance).__name__}")
        elif "minLength" in schema and len(instance) < schema["minLength"]:
            errors.append(f"{path}: string too short (min {schema['minLength']})")

    elif schema_type == "number" or schema_type == "integer":
        if not isinstance(instance, (int, float)):
            errors.append(f"{path}: expected {schema_type}, got {type(instance).__name__}")
        elif "minimum" in schema and instance < schema["minimum"]:
            errors.append(f"{path}: {instance} is less than minimum {schema['minimum']}")
```

#### 4. Streaming Token Counter

Track token usage across concurrent API requests with real-time aggregation.

```python
import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass

@dataclass
class UsageRecord:
    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0

class TokenUsageTracker:
    def __init__(self, window_seconds: int = 60):
        self.window = window_seconds
        self.lock = asyncio.Lock()
        self.records: dict[str, list[tuple[float, UsageRecord]]] = defaultdict(list)

    async def record(self, api_key: str, input_tokens: int, output_tokens: int):
        async with self.lock:
            now = time.monotonic()
            self.records[api_key].append((now, UsageRecord(input_tokens, output_tokens, 1)))

    async def get_usage(self, api_key: str) -> UsageRecord:
        async with self.lock:
            now = time.monotonic()
            cutoff = now - self.window
            records = self.records[api_key]
            # Prune old records
            self.records[api_key] = [(t, r) for t, r in records if t > cutoff]

            total = UsageRecord()
            for _, record in self.records[api_key]:
                total.input_tokens += record.input_tokens
                total.output_tokens += record.output_tokens
                total.requests += record.requests
            return total
```

## System Design Round

### Common Topics

#### Design the ChatGPT Backend

```
[Client (Web/Mobile)] --> [API Gateway] --> [Conversation Manager]
                                                    |
                                             [Message Store]
                                                    |
                                             [Inference Router]
                                                    |
                                        [Model Serving Cluster (GPUs)]
                                                    |
                                        [Streaming Response Handler]
                                                    |
                                        [Safety / Moderation]
                                                    |
                                        [SSE Stream to Client]
```

Key considerations:
- **Conversation state**: Store message history, manage context windows
- **Streaming**: SSE for real-time token delivery (same patterns as Anthropic)
- **Multi-turn**: Efficient context management, truncation strategies
- **Plugins/Function calling**: Route tool calls, manage external API interactions
- **Safety**: Content moderation pipeline, PII detection

#### Design the OpenAI API Platform

- **Multi-tenancy**: Isolate customers, enforce rate limits per API key
- **Model routing**: Route requests to appropriate model (GPT-4, GPT-3.5, embeddings)
- **Usage metering**: Real-time token counting for billing
- **Batch API**: Queue large offline jobs, different pricing tier
- **Fine-tuning pipeline**: Upload data, train, deploy custom models

#### Design a Training Data Pipeline

```
[Raw Data Sources] --> [Ingestion] --> [Dedup & Filtering]
                                              |
                                       [Quality Scoring]
                                              |
                                       [Tokenization]
                                              |
                                       [Shuffling & Packing]
                                              |
                                       [Training Data Loader]
```

- **Scale**: Trillions of tokens, petabytes of data
- **Quality**: Automated filtering (perplexity, toxicity scores, dedup)
- **Reproducibility**: Deterministic data ordering for experiment comparison
- **Streaming**: Data must be fed to GPUs without bottlenecking training

#### Design a Vector Database / Embedding Search

- **Indexing**: HNSW, IVF, or product quantization for ANN search
- **Sharding**: Partition embedding space across nodes
- **Real-time updates**: Handle new documents without full re-indexing
- **Hybrid search**: Combine vector similarity with keyword/metadata filters
- **Serving latency**: Sub-50ms for top-K retrieval

## Hiring Manager / Culture Round

### Mission Alignment

OpenAI's mission is to ensure AGI benefits all of humanity. They'll probe:

- "Why OpenAI over other AI companies?"
- "What does safe AI development mean to you?"
- "How do you think about the trade-off between moving fast and being cautious?"
- "What's your view on open-sourcing AI models?"

Be genuine. Having a nuanced view (acknowledging trade-offs) is better than repeating their mission statement.

### Technical Leadership

- "Tell me about a system you designed that had to evolve significantly."
- "Describe a time you had to make a technical decision with incomplete information."
- "How do you approach technical debt in a fast-moving environment?"
- "Tell me about a time you had to push back on a product requirement."

### Collaboration at Speed

OpenAI moves fast. They want to know:
- Can you make decisions quickly with 70% information?
- Do you over-engineer or find the pragmatic solution?
- Can you work across teams (research, product, infra)?

## Preparation Tips

1. **Study the OpenAI API** -- Use it. Build something. Understand rate limits, streaming, function calling, and error codes intimately.
2. **Read OpenAI research** -- GPT-4 technical report, DALL-E, Whisper, RLHF papers. You don't need to understand every equation, but know the key ideas.
3. **Practice API design** -- Design RESTful APIs for ML services. Think about versioning, backwards compatibility, error handling.
4. **Concurrency in Python** -- asyncio is foundational. Practice writing async services.
5. **Know the competitive landscape** -- Anthropic, Google DeepMind, Meta AI, Mistral. What differentiates OpenAI?
6. **Build something with the API** -- Having a project that uses GPT-4 / function calling / embeddings shows genuine engagement.

## Sources

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [OpenAI Research](https://openai.com/research)
- [Glassdoor - OpenAI Interview Questions](https://www.glassdoor.com/Interview/OpenAI-Interview-Questions-E1880775.htm)
- [levels.fyi - OpenAI Compensation Data](https://www.levels.fyi/companies/openai/salaries/software-engineer)
- [The Reality of Tech Interviews in 2025 - Pragmatic Engineer](https://newsletter.pragmaticengineer.com/p/the-reality-of-tech-interviews)
- r/cscareerquestions, r/ExperiencedDevs, Blind (community reports)
