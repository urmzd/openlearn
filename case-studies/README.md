# Case Studies

Real builds, generalised. Each case study takes a system that was actually
designed and shipped under a deadline, strips it to the concepts that transfer,
and records the sequence it was built in.

> **Prerequisites**: varies by study. Each one names its own, and every study
> links back to the track that teaches the underlying theory.

## Overview

- **What these are**: worked builds. Problem, concepts, the decisions and their
  trade-offs, and the order the thing was actually assembled in.
- **What these are not**: project archives. No original source is preserved.
  Every implementation here was rewritten to be self-contained, dependency-free,
  and runnable, so it teaches the idea rather than documenting a codebase.
- **Estimated time**: 1-2 days each, or an afternoon if you only read.

## Key Takeaways

- **The build order is the lesson.** Knowing that a matching engine needs
  price-time priority is cheap; knowing to write the O(n) version first and earn
  the heap is what actually separates outcomes under time pressure.
- **Every study has one invariant that makes testing possible.** A crossed book,
  a double-fired event, an unbounded query, a rising inertia, a record that
  does not replay. Find that property
  and validation stops being guesswork.
- **The interesting decisions are refusals.** Not adding durability, not letting
  the model be the safety boundary, not shipping exactly-once. Each study states
  what it deliberately did not build.

## How to Study

- Read the README, then run the implementation. Every one is standard library
  only and executes its own tests: `python <file>.py`.
- Then delete a safeguard and watch a test fail. Remove the row cap, remove the
  primary key, skip the zero-quantity removal. The tests exist to catch exactly
  those, and breaking them on purpose is the fastest way to see why they matter.
- Read the build log last, and compare it to how you would have sequenced it.

## Studies

| # | Study | Domain | Runnable |
|---|-------|--------|----------|
| 01 | [Order Book Matching](01-order-book-matching/) | Data structures under a latency budget | [`matching_engine.py`](01-order-book-matching/matching_engine.py) |
| 02 | [Grounded SQL Agent](02-grounded-sql-agent/) | LLM tool loops, grounding, and safety | [`safety_gate.py`](02-grounded-sql-agent/safety_gate.py) |
| 03 | [Exactly-Once Event API](03-exactly-once-event-api/) | API design and processing guarantees | [`event_api.py`](03-exactly-once-event-api/event_api.py) |
| 04 | [K-Means Optimization](04-kmeans-optimization/) | Optimising an algorithm in tiers | [`kmeans_ladder.py`](04-kmeans-optimization/kmeans_ladder.py) |
| 05 | [Agent Evaluation Harness](05-agent-eval-harness/) | Grading an agent that changes state | [`eval_harness.py`](05-agent-eval-harness/eval_harness.py) |

## The Shared Shape

Four of the five studies are the same move applied to different domains: build
the obvious correct version, name its bottleneck precisely, then earn each
improvement.

| Study | Baseline | Earned improvement | What the jump costs |
|-------|----------|--------------------|---------------------|
| Order book | Flat list, linear scan | Heap of price levels, then a tick-indexed ladder | Memory, and an assumption about price range |
| K-means | Random init, fixed iterations | k-means++, then incremental updates and distance bounds | Nothing algorithmic; only code complexity |
| SQL agent | One tool, no grounding | Schema discovery, bounded retry, deterministic gate | Tokens per query, and latency |
| Event API | Check-then-act dedup | Constraint dedup, then atomic claim | Nothing; the correct version is also simpler |
| Eval harness | Compare final rows | Replay verification, then oracles and mutants | Storage per call, and a suite to maintain |

The last row is worth sitting with. Dedup done properly is *less* code than
dedup done by checking first, and it is correct under concurrency. Not every
improvement is a trade-off; some are just the right answer written down.

## Connections to Other Tracks

| Study | Theory lives in |
|-------|-----------------|
| Order Book Matching | [Algorithms](../algorithms/) (heaps, ordering, amortised analysis) |
| Grounded SQL Agent | [LLM Evaluation](../ai-platform-engineering/09-llm-evaluation/), [Retrieval & RAG](../ai-platform-engineering/07-retrieval-and-rag/), [Authorization](../ai-platform-engineering/08-authorization-and-access-control/) |
| Exactly-Once Event API | [Distributed Workers](../ai-platform-engineering/05-durable-orchestration-and-workers/), [Durable Orchestration](../ai-platform-engineering/05-durable-orchestration-and-workers/) |
| K-Means Optimization | [Statistical Learning](../ml/01-statistical-learning/) |

For the failure modes these builds were consciously avoiding, see
[Lessons from Practice](../software-craftsmanship/04-lessons-from-practice/).
