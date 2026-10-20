# Worked Projects Across the Library

This index connects five worked builds in their subject books: a matching engine, a text-to-SQL agent, an idempotent event API, a k-means implementation, and an agent evaluation harness. Each was shipped under a deadline, then rewritten as standard-library Python that runs its own tests, and each records the order it was actually built in, because the build order is the lesson. The studies are for engineers who already know the underlying theory and want to see it applied under pressure, and for anyone preparing to defend design decisions in a live interview or a design review.

## What you will be able to do

- Sequence a build so a correct version exists early and each later layer is earned by a named bottleneck.
- Find the one invariant that makes a system testable.
- Justify each optimization with evidence, and say what it costs and where it does not help.
- State plainly what you chose not to build, and why.
- Break an implementation on purpose and predict which test catches it.

## Routes through the projects

Each study takes about half a day of focused work, so all five take two to three days.

- **All five:** read order book, SQL agent, event API, k-means, and the agent evaluation harness. Each project belongs to the book that teaches its main subject.
- **Preparing for a live-coding or data-structures interview** (about 1 day): [Chapter 27: Order Book Matching](../courses/programming-and-algorithms/27-case-study-order-book/README.md), then [Chapter 31: K-Means Optimization](../courses/ai-engineering/31-case-study-kmeans/README.md).
- **Building and grading LLM agents** (about 1 day): [Chapter 30: Grounded SQL Agent](../courses/ai-engineering/30-case-study-sql-agent/README.md), then [Chapter 32: Agent Evaluation Harness](../courses/ai-engineering/32-case-study-agent-evals/README.md), the same ideas for an agent that changes state.
- **Preparing for a backend design review** (about half a day): [Chapter 13: Exactly-Once Event API](../courses/systems/13-case-study-event-api/README.md), then the build logs of [Chapter 27](../courses/programming-and-algorithms/27-case-study-order-book/README.md#the-build-log) and [Chapter 32](../courses/ai-engineering/32-case-study-agent-evals/README.md#the-build-log) to compare how each sequenced its work.

In every study, read the concepts, run the file with `python <file>.py` from the study's directory, then break it deliberately and watch the tests catch you.

## Before you start

Each study names its own prerequisites. The order book draws on [Programming and Algorithms](../courses/programming-and-algorithms/README.md). SQL agents, k-means, and agent evaluation draw on [AI Engineering](../courses/ai-engineering/README.md). The event API draws on [Systems Engineering](../courses/systems/README.md). [Software Engineering](../courses/software-engineering/README.md) supplies testing, authorization, and [lessons from practice](../courses/software-engineering/05-lessons-from-practice/README.md). [How to read these books](how-to-read.md) explains the shared method.

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
