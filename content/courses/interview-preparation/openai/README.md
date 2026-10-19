# OpenAI

OpenAI ships products, ChatGPT and the API above all, at startup speed and enormous scale, and its interviews reflect both halves of that. Like Anthropic, it prefers production-quality code to LeetCode tricks, and its problems tend to be applied and systems-oriented: rate limiters, async task queues, validators, and usage meters, the components an API platform is built from. Alongside the engineering, OpenAI cares deeply that you believe in its mission of safe AGI.

This chapter prepares you for OpenAI engineering roles with a focus on AI infrastructure and full-stack development. It covers the process and OpenAI's unusual equity, the themes the company hires for, four reported coding problems worked in full, four system design prompts from ChatGPT's backend to embedding search, and the hiring manager round on mission, leadership, and speed.

> **Learning objectives**
> - Describe the OpenAI interview loop and what each round assesses.
> - Implement sliding-window-log and token-bucket rate limiters and explain how they differ.
> - Build an asyncio task queue with bounded concurrency and retries with exponential backoff and jitter.
> - Write a recursive JSON Schema validator and a windowed token usage tracker.
> - Design the ChatGPT backend, the API platform, a training data pipeline, and a vector search service.

## Before you begin

OpenAI publishes an official [interview guide](https://openai.com/interview-guide/), and its [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/) is OpenAI's own reinforcement learning curriculum. For ML background, see the [Deep Learning](../../machine-learning/02-deep-learning/README.md) and [Reinforcement Learning](../../machine-learning/03-reinforcement-learning/README.md) chapters of *Machine Learning and AI*. Within this book, the [Anthropic](../anthropic/README.md) chapter covers very similar coding and serving problems, and [Full-stack AI](../shared-concepts/full-stack-ai.md) covers the ML systems background.

## The interview process

Expect three to five weeks and five or six rounds:

| Round | Format | Duration | Focus |
|-------|--------|----------|-------|
| Recruiter Screen | Phone | 30 min | Background, motivation, role fit |
| Technical Phone Screen | Coding (CoderPad) | 60 min | Algorithms + systems thinking |
| Onsite 1 | Coding | 60 min | Production-quality implementation |
| Onsite 2 | System Design | 60 min | ML infrastructure, API design |
| Onsite 3 | Coding / ML | 60 min | Applied ML or infra coding |
| Onsite 4 | Hiring Manager / Culture | 45-60 min | Mission alignment, collaboration |

Reported total compensation is about $450-600K for a senior SWE (L4) and $600-900K for a staff SWE (L5). Equity comes as Profit Participation Units (PPUs), unique to OpenAI's capped-profit structure. Compensation is extremely competitive, especially the equity upside.

## What OpenAI looks for

Five themes stand out. Hiring is mission-driven: OpenAI cares that you believe in safe AGI, and genuine passion for AI safety and alignment matters. The company combines startup speed with massive scale, so expect questions about pragmatic trade-offs. It thinks API-first, since ChatGPT and the API are the core products, which puts API design, rate limiting, and developer experience at the centre. Many roles are full-stack AI, spanning model training infrastructure to user-facing products, so be ready to discuss both. And the codebase is heavily Python, so deep Python knowledge (asyncio, typing, packaging) is expected.

## Coding rounds

There are two or three coding rounds. In spirit they resemble Anthropic's: production-quality code over LeetCode tricks, with problems that are more applied and systems-oriented than pure algorithms. Concurrency, API design, and data pipeline problems are common, and you may be asked to extend an existing codebase rather than start from scratch. Four problems are reported.

### An API rate limiter

The first asks you to design and implement a rate limiter supporting multiple strategies. Two are worth having ready.

A sliding-window log keeps, for each client, a deque of the timestamps of its accepted requests. On each call it evicts timestamps older than the window, then allows the request if fewer than `max_requests` remain. It is exact, never letting more than the limit through in any window, but its memory grows with the limit.

A token bucket keeps, for each client, a token count and the time of the last refill. Each call first adds `elapsed * rate` tokens, capped at `bucket_size`, then spends the request's cost if enough tokens are available. It uses constant memory per client, permits short bursts up to the bucket size, and supports requests of different costs, which is useful when the cost is a number of model tokens.

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

Both use `time.monotonic()` so that clock adjustments cannot grant or deny requests, and both hold a lock across the read-modify-write so that concurrent requests cannot spend the same capacity twice.

### An async task queue with retries

The second is an asyncio task queue. Workers pull tasks from an `asyncio.Queue`; a semaphore caps how many execute at once; synchronous and coroutine functions are both supported. A failed task is retried up to `max_retries` times with exponential backoff plus random jitter (about 2, 4, 8 seconds, plus up to one more), the jitter preventing many failed tasks from retrying in lockstep. `run` starts the workers, waits on `queue.join()` until every task is marked done, and then cancels them.

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

The ordering in the retry path matters: the task is put back on the queue before the worker calls `task_done()` for the failed attempt, so the queue's unfinished count never drops to zero while a retry is pending and `join()` cannot return early. Two improvements make good discussion points. The backoff sleep happens while the worker holds a semaphore slot, so a burst of failures stalls healthy work; scheduling the re-enqueue with `asyncio.create_task` or a delayed queue frees the slot. And a synchronous `fn` runs on the event loop thread and blocks every other task; `asyncio.to_thread` avoids that.

### A JSON Schema validator

The third, very relevant to function calling and structured outputs, is validating a JSON value against a subset of JSON Schema. The validator recurses on the schema's `type`, accumulating error messages with a JSONPath-style location (`$.user.tags[2]`) rather than stopping at the first error. Objects check `required` fields, recurse into known `properties`, and reject extra keys when `additionalProperties` is false. Arrays recurse into `items` and check `minItems`; strings check `minLength`; numbers check `minimum`.

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

> **Warning:** Two Python subtleties slip through this validator. `bool` is a subclass of `int`, so `True` passes as a number; and `"integer"` accepts `3.5` because it shares the number branch. A careful answer excludes `bool` explicitly and checks `float(instance).is_integer()` for integers. Schemas without a `type`, and keywords such as `enum` or `boolean`, are silently accepted.

### A streaming token counter

The fourth tracks token usage across concurrent API requests with real-time aggregation over a sliding window, the basis of usage-based limits and billing. Each request appends a timestamped record; a usage query prunes records older than the window and sums the rest. An `asyncio.Lock` guards the shared dictionary.

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

Each query costs time proportional to the records in the window, and records are pruned only when someone reads them. At API scale you would bucket usage into fixed intervals (one counter per second, say), which makes both recording and querying constant-time per bucket.

## The system design round

The design round focuses on ML infrastructure and API design. Four prompts are common.

### The ChatGPT backend

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

The conversation manager stores message history and manages the context window. Responses stream to the client over server-sent events for real-time token delivery, the same pattern described in the [Anthropic system design](../anthropic/03-system-design.md) section. Multi-turn conversations need efficient context management and a truncation strategy for when history outgrows the window. Plugins and function calling require routing tool calls and managing interactions with external APIs. And the safety layer runs a content moderation pipeline and PII detection.

### The OpenAI API platform

The API platform is multi-tenant: it isolates customers and enforces rate limits per API key. A model router sends each request to the right model (GPT-4, GPT-3.5, embeddings). Usage metering counts tokens in real time for billing. A batch API queues large offline jobs at a different pricing tier, and a fine-tuning pipeline lets customers upload data, train, and deploy custom models.

### A training data pipeline

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

The scale is trillions of tokens and petabytes of data. Quality is maintained by automated filtering on perplexity, toxicity scores, and deduplication. Data ordering must be deterministic so that experiments can be compared reproducibly. And the pipeline must stream data to the GPUs fast enough never to bottleneck training.

### A vector database for embedding search

Approximate nearest-neighbour search is built on an index such as HNSW, IVF, or product quantization. The embedding space is partitioned across nodes. New documents must be indexed in real time without a full rebuild. Hybrid search combines vector similarity with keyword and metadata filters. And serving latency should be under 50 ms for top-K retrieval.

## The hiring manager and culture round

OpenAI's mission is to ensure AGI benefits all of humanity, and the round probes your alignment with it: "Why OpenAI over other AI companies?", "What does safe AI development mean to you?", "How do you think about the trade-off between moving fast and being cautious?", and "What's your view on open-sourcing AI models?" Be genuine. A nuanced view that acknowledges trade-offs is better than repeating the mission statement.

The round also covers technical leadership: "Tell me about a system you designed that had to evolve significantly.", "Describe a time you had to make a technical decision with incomplete information.", "How do you approach technical debt in a fast-moving environment?", and "Tell me about a time you had to push back on a product requirement."

Finally, because OpenAI moves fast, interviewers want to know whether you collaborate well at speed: whether you can make decisions quickly with 70% of the information, whether you over-engineer or find the pragmatic solution, and whether you can work across research, product, and infrastructure teams.

## Preparing

Study the OpenAI API by using it and building something: understand rate limits, streaming, function calling, and error codes intimately. A project that uses GPT-4, function calling, or embeddings shows genuine engagement. Read OpenAI's research (the GPT-4 technical report, DALL-E, Whisper, the RLHF papers); you do not need every equation, but know the key ideas. Practise designing RESTful APIs for ML services, with attention to versioning, backwards compatibility, and error handling. Make asyncio second nature by writing async services. And know the competitive landscape (Anthropic, Google DeepMind, Meta AI, Mistral) well enough to say what differentiates OpenAI.

## Summary

- OpenAI's coding rounds are applied: rate limiters, async queues, validators, and usage meters, judged on production quality.
- Know the trade-off between a sliding-window log (exact, more memory) and a token bucket (constant memory, bursts, weighted costs).
- Async code must handle retries, backoff with jitter, and the interaction between semaphores, sleeps, and `queue.join()`.
- System design centres on ChatGPT and the API platform: streaming, multi-tenancy, metering, and data pipelines.
- Mission alignment is assessed directly; bring a genuine, nuanced view of AI safety.

## Further reading

- [OpenAI Interview Guide](https://openai.com/interview-guide/)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [OpenAI Research](https://openai.com/research)
- [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/)
- [Glassdoor: OpenAI Interview Questions](https://www.glassdoor.com/Interview/OpenAI-Interview-Questions-E1880775.htm)
- [levels.fyi: OpenAI Compensation Data](https://www.levels.fyi/companies/openai/salaries/software-engineer)
- [The Reality of Tech Interviews in 2025 (Pragmatic Engineer)](https://newsletter.pragmaticengineer.com/p/the-reality-of-tech-interviews)
- Community reports on r/cscareerquestions, r/ExperiencedDevs, and Blind.
