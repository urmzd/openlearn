# Case Studies

This book is five real builds, generalized: a matching engine, a text-to-SQL agent, an idempotent event API, a k-means implementation, and an agent evaluation harness. Each was shipped under a deadline, then rewritten as standard-library Python that runs its own tests, and each records the order it was actually built in, because the build order is the lesson. The book is for engineers who already know the underlying theory and want to see it applied under pressure, and for anyone preparing to defend design decisions in a live interview or a design review.

## What you will be able to do

- Sequence a build so a correct version exists early and each later layer is earned by a named bottleneck.
- Find the one invariant that makes a system testable.
- Justify each optimization with evidence, and say what it costs and where it does not help.
- State plainly what you chose not to build, and why.
- Break an implementation on purpose and predict which test catches it.

## Ways through this book

Each study takes about half a day of focused work, so the whole book takes two to three days.

- **The whole book, in order** (about 2.5 days): [Chapter 1](01-order-book-matching/README.md) through [Chapter 5](05-agent-eval-harness/README.md).
- **Preparing for a live-coding or data-structures interview** (about 1 day): [Chapter 1: Order Book Matching](01-order-book-matching/README.md), then [Chapter 4: K-Means Optimization](04-kmeans-optimization/README.md).
- **Building and grading LLM agents** (about 1 day): [Chapter 2: Grounded SQL Agent](02-grounded-sql-agent/README.md), then [Chapter 5: Agent Evaluation Harness](05-agent-eval-harness/README.md), the same ideas for an agent that changes state.
- **Preparing for a backend design review** (about half a day): [Chapter 3: Exactly-Once Event API](03-exactly-once-event-api/README.md), then the build logs of [Chapter 1](01-order-book-matching/README.md#the-build-log) and [Chapter 5](05-agent-eval-harness/README.md#the-build-log) to compare how each sequenced its work.

In every study, read the concepts, run the file with `python <file>.py` from the study's directory, then break it deliberately and watch the tests catch you.

## Before you start

Read these after the books that carry their theory; each study names its own prerequisites. Chapter 1 rests on [*Algorithms*](../algorithms/README.md) for heaps and amortized analysis. Chapters 2 and 5 rest on [LLM evaluation](../ai-platform-engineering/09-llm-evaluation/README.md), [Retrieval and RAG](../ai-platform-engineering/07-retrieval-and-rag/README.md), and [Authorization and access control](../ai-platform-engineering/08-authorization-and-access-control/README.md) in *AI Platform Engineering*, and Chapter 3 rests on [Durable orchestration and workers](../ai-platform-engineering/05-durable-orchestration-and-workers/README.md) in the same book. Chapter 4 rests on [Statistical learning](../machine-learning/01-statistical-learning/README.md) in *Machine Learning and AI*. Keep [Lessons from practice](../software-craftsmanship/04-lessons-from-practice/README.md) from *Software Craftsmanship* nearby, which catalogs the failures these builds avoid. New to these books? [How to read these books](../../references/how-to-read.md) explains how sections, review questions, and solutions work.

## The shared shape

Most studies make the same move in different domains: build the obvious correct version, name its bottleneck, then earn each improvement.

| Study | Baseline | Earned improvement | What the jump costs |
|-------|----------|--------------------|---------------------|
| Order book | Flat list, linear scan | Heap of price levels, then a tick-indexed ladder | Memory, and an assumption about price range |
| SQL agent | One tool, no grounding | Schema discovery, bounded retry, deterministic gate | Tokens per query, and latency |
| Event API | Check-then-act dedup | Constraint dedup, then atomic claim | Nothing; the correct version is also simpler |
| K-means | Random init, fixed iterations | k-means++, then incremental updates and distance bounds | Nothing algorithmic; only code complexity |
| Eval harness | Compare final rows | Replay verification, then oracles and mutants | Storage per call, and a suite to maintain |

Not every improvement is a trade-off: dedup by unique constraint is less code than checking first, and it is correct under concurrency.
