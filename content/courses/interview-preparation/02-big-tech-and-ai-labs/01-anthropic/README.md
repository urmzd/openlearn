# Anthropic

Anthropic's infrastructure team builds the systems that serve Claude: GPU inference at scale, the batching and KV cache management behind it, token streaming to clients, and the safety machinery that runs alongside generation. Its interview loop is built to find engineers who can write production-quality concurrent code and design that kind of system under realistic constraints. This chapter walks through the whole loop, one round per section, and then gives three concept sections that supply the technical background the rounds draw on.

The process typically spans about three weeks and five rounds after a recruiter screen. Four themes run through all of them. **Concurrency appears in every round**, from thread-safe data structures in the online assessment to distributed system design, so expect asyncio, threading, locks, and race conditions throughout. **Inference serving is the core domain**: GPU memory management, batching strategies, the KV cache, streaming over SSE, and autoscaling are the bread and butter of the team. The **culture is safety-first**: Constitutional AI, safety filtering, and responsible deployment are not talking points, and you should expect questions about tradeoffs between safety and latency. And the bar is **production quality over leetcode tricks**: error handling, thread safety, comments, and clean APIs count for more than optimal asymptotic complexity.

> **Learning objectives**
> - Describe each round of the Anthropic infrastructure loop, its format, and what it evaluates.
> - Write production-quality, thread-safe solutions to the reported coding problems, and explain the design of each.
> - Design an LLM inference API covering GPU memory, batching, KV cache, autoscaling, streaming, and safety.
> - Prepare a project deep dive and honest, informed answers about AI safety for the hiring manager round.

## Before you begin

You should be fluent in Python, including asyncio and the `threading` module, and comfortable with basic distributed-systems vocabulary. The shared foundation for all company guides in this book, especially [Distributed systems](../../01-shared-concepts/01-distributed-systems.md) and [ML infrastructure](../../01-shared-concepts/03-ml-infrastructure.md), is worth reading first. For deeper background on the systems themselves, *Systems Engineering* (start at its [front matter](../../../systems/README.md)) and *AI Engineering* (start at its [front matter](../../../ai-engineering/README.md)) cover messaging, search, and inference serving from first principles.

## The loop at a glance

The rounds, in order, are:

| Round | Format | Duration | Focus |
|-------|--------|----------|-------|
| Recruiter screen | Phone/Video | 30 min | Background, motivation, role fit |
| [Online assessment](01-online-assessment.md) | CodeSignal | 90 min | Production-quality coding |
| [Coding round 1](02-coding-round-1.md) | Live coding | 60 min | Concurrent systems, async patterns |
| [System design](03-system-design.md) | Whiteboard/Virtual | 60 min | LLM inference infrastructure |
| [Coding round 2](04-coding-round-2.md) | Live coding | 60 min | Data structures, concurrency |
| [Hiring manager](05-hiring-manager.md) | Behavioral | 45 min | Leadership, project depth, culture |

Reported compensation is roughly $550K total for a senior software engineer and roughly $671K for a lead, combining base, equity, and bonus. Equity is a significant portion, as RSUs on standard vesting schedules.

### The recruiter screen

The process opens with a 30-minute call that covers your background and why you are interested in Anthropic specifically; why infrastructure, and why AI safety; your high-level technical experience with distributed systems, ML infrastructure, and cloud; your timeline and compensation expectations; and an overview of the rest of the process. Be genuine about your interest in AI safety, because Anthropic's mission is central to its hiring decisions. Read about Constitutional AI and RLHF before the call.

## In this chapter

- [The online assessment](01-online-assessment.md): what the 90-minute CodeSignal screen grades, and full solutions to the multi-level LRU cache, the task management system, and the in-memory database.
- [The first coding round: concurrent systems](02-coding-round-1.md): an asyncio web crawler built and defended in full, plus word segmentation and the concurrency background the round assumes.
- [The system design round](03-system-design.md): how to run the hour, and an end-to-end design of an LLM inference API, with search and training infrastructure as alternative prompts.
- [The second coding round: data structures and concurrency primitives](04-coding-round-2.md): stack samples to trace events, a blocking min-heap, top-K over a stream, a readers-writer trie, and an inverted index.
- [The hiring manager interview](05-hiring-manager.md): the project deep dive, judgment and incident scenarios, behavioral questions, and safety and ethics.
- [Concurrency patterns](06-concurrency-patterns.md): the GIL, asyncio and threading primitives, lock-free structures, and distributed consistency.
- [GPU inference serving](07-gpu-inference.md): batching, KV cache sizing and eviction, PagedAttention, warm pools, and scaling signals.
- [Streaming and Server-Sent Events](08-streaming-sse.md): the SSE protocol, typed events, backpressure, detokenization, reconnection, and latency SLOs.

## Sources

This guide is compiled from first-person interview experiences, Reddit (r/cscareerquestions, r/ExperiencedDevs), Glassdoor reviews, engineering blogs, and published interview guides. Formats change, so treat the specifics as a well-informed snapshot rather than a guarantee.
