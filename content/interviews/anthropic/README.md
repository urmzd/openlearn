# Anthropic Infrastructure SWE Interview Guide

Comprehensive preparation guide for the Anthropic Infrastructure Software Engineer role, covering the full interview pipeline and key technical domains.

## Interview Process Overview

The process typically spans **~3 weeks** across **5 rounds**:

| Round | Format | Duration | Focus |
|-------|--------|----------|-------|
| [Recruiter Screen](#recruiter-screen) | Phone/Video | 30 min | Background, motivation, role fit |
| [Online Assessment](01-online-assessment.md) | CodeSignal | 90 min | Production-quality coding |
| [Coding Round 1](02-coding-round-1.md) | Live coding | 60 min | Concurrent systems, async patterns |
| [System Design](03-system-design.md) | Whiteboard/Virtual | 60 min | LLM inference infrastructure |
| [Coding Round 2](04-coding-round-2.md) | Live coding | 60 min | Data structures, concurrency |
| [Hiring Manager](05-hiring-manager.md) | Behavioral | 45 min | Leadership, project depth, culture |

## Compensation (Reported Ranges)

- **Senior SWE**: ~$550K total comp (base + equity + bonus)
- **Lead SWE**: ~$671K total comp
- Equity is a significant portion; RSUs with standard vesting schedules

## Key Themes Across All Rounds

1. **Concurrency appears in every round** -- from thread-safe data structures in the OA to distributed system design. Expect asyncio, threading, locks, and race conditions throughout.
2. **Inference serving is the core domain** -- GPU memory management, batching strategies, KV cache, streaming (SSE), and autoscaling are the bread and butter of the infra team.
3. **Safety-first culture** -- Constitutional AI, safety filtering, and responsible deployment are not just talking points. Expect questions about how you'd handle safety vs. latency trade-offs.
4. **Production quality over leetcode tricks** -- They care about error handling, thread safety, comments, and clean APIs more than optimal asymptotic complexity.

## Recruiter Screen

The initial 30-minute call covers:
- Your background and interest in Anthropic specifically
- Why infrastructure / why AI safety
- High-level technical experience (distributed systems, ML infra, cloud)
- Timeline and compensation expectations
- Overview of the remaining process

**Tip**: Be genuine about your interest in AI safety. Anthropic's mission is central to their hiring decisions. Research Constitutional AI and RLHF before this call.

## Concept Deep Dives

Supplementary reference material for key technical domains:

- [GPU Inference Serving](concepts/gpu-inference.md) -- Batching, KV cache, warm pools, model serving architecture
- [Concurrency Patterns](concepts/concurrency-patterns.md) -- GIL, asyncio, locks, distributed consistency
- [Streaming & SSE](concepts/streaming-sse.md) -- Server-Sent Events, backpressure, token streaming

## Sources

Compiled from first-person interview experiences, Reddit (r/cscareerquestions, r/ExperiencedDevs), Glassdoor reviews, engineering blogs, and published interview guides.
