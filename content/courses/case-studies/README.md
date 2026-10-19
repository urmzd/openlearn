# Case Studies

This book is a collection of real builds, generalized. Each case study takes a system that was actually designed and shipped under a deadline, strips it down to the concepts that transfer, and records the order in which it was built. They are worked builds: the problem, the concepts, the decisions and their trade-offs, and the sequence the thing was actually assembled in. They are not project archives. No original source is preserved; every implementation here was rewritten to be self-contained, dependency-free, and runnable, so it teaches the idea rather than documenting a codebase.

The book is for engineers who already know the underlying theory and want to see it applied under pressure, and for anyone preparing to defend design decisions in a live interview or a design review. By the end you will be able to sequence a build so that a defensible system exists early, find the one invariant that makes a system testable, earn each optimization with evidence, and say plainly what you chose not to build.

## Three ideas that run through every study

The build order is the lesson. Knowing that a matching engine needs price-time priority is cheap; knowing to write the O(n) version first and earn the heap is what actually separates outcomes under time pressure.

Every study has one invariant that makes testing possible: a book that is never crossed, an event that never fires twice, a query that is always bounded, an inertia that never rises, a record that always replays. Find that property and validation stops being guesswork.

And the interesting decisions are refusals: not adding durability, not letting the model be the safety boundary, not claiming exactly-once. Each study states what it deliberately did not build.

## How the book is organized

There are five studies, and they can be read in any order, though the agent evaluation harness builds on the grounded SQL agent.

[Order book matching](01-order-book-matching/README.md) builds a limit order book and matching engine three times behind one interface: a flat list, a heap of price levels, and a tick-indexed ladder. It is a study in data structures under a latency budget and in justifying each optimization with a bottleneck analysis. Its runnable file is [`matching_engine.py`](01-order-book-matching/matching_engine.py).

[Grounded SQL agent](02-grounded-sql-agent/README.md) turns natural-language questions into SQL against an unseen database, executes it safely behind a deterministic gate, and proves the answers right by result-set equivalence. Its central evidence is an ablation showing that grounding, not model size, closed the accuracy gap. Its runnable file is [`safety_gate.py`](02-grounded-sql-agent/safety_gate.py).

[Exactly-once event API](03-exactly-once-event-api/README.md) designs an API in which submitting twice never processes twice, using a unique constraint as the dedup and an atomic claim for workers, and is honest that the result is at-most-once. Its runnable file is [`event_api.py`](03-exactly-once-event-api/event_api.py).

[K-means optimization](04-kmeans-optimization/README.md) implements k-means in three tiers behind one contract and measures where each improvement helps and where it does not, showing that initialization outweighs every constant-factor change. Its runnable file is [`kmeans_ladder.py`](04-kmeans-optimization/kmeans_ladder.py).

[Agent evaluation harness](05-agent-eval-harness/README.md) grades an agent that changes state, with isolated sessions, replay verification, a set-based checker, and oracles and mutants that test the checker itself. Its runnable file is [`eval_harness.py`](05-agent-eval-harness/eval_harness.py).

## The shared shape

Four of the five studies are the same move applied to different domains: build the obvious correct version, name its bottleneck precisely, then earn each improvement. The table lines them up:

| Study | Baseline | Earned improvement | What the jump costs |
|-------|----------|--------------------|---------------------|
| Order book | Flat list, linear scan | Heap of price levels, then a tick-indexed ladder | Memory, and an assumption about price range |
| K-means | Random init, fixed iterations | k-means++, then incremental updates and distance bounds | Nothing algorithmic; only code complexity |
| SQL agent | One tool, no grounding | Schema discovery, bounded retry, deterministic gate | Tokens per query, and latency |
| Event API | Check-then-act dedup | Constraint dedup, then atomic claim | Nothing; the correct version is also simpler |
| Eval harness | Compare final rows | Replay verification, then oracles and mutants | Storage per call, and a suite to maintain |

The event API row is worth sitting with. Dedup done properly is *less* code than dedup done by checking first, and it is correct under concurrency. Not every improvement is a trade-off; some are just the right answer written down.

## Prerequisites and related books

Prerequisites vary by study, and each one names its own in its "Before you begin" section. Every study links back to the book that teaches its theory. The order book rests on [Algorithms](../algorithms/README.md), for heaps, ordering, and amortized analysis. The SQL agent and the evaluation harness rest on [LLM evaluation](../ai-platform-engineering/09-llm-evaluation/README.md), [Retrieval and RAG](../ai-platform-engineering/07-retrieval-and-rag/README.md), and [Authorization and access control](../ai-platform-engineering/08-authorization-and-access-control/README.md) in *AI Platform Engineering*. The event API rests on [Durable orchestration and workers](../ai-platform-engineering/05-durable-orchestration-and-workers/README.md) in the same book. K-means rests on [Statistical learning](../machine-learning/01-statistical-learning/README.md) in *Machine Learning*. For the failure modes these builds were consciously avoiding, see [Lessons from practice](../software-craftsmanship/04-lessons-from-practice/README.md) in *Software Craftsmanship*.

## Conventions

Each study opens with the problem and a short list of learning objectives, develops the concepts in the order they matter, and then gives the build log: the order in which the system was assembled, which is the most transferable part. A table of techniques to take away, a closing section linking to related books, key terms, a summary, and review questions with answers in collapsible blocks end each study.

Every implementation is standard-library Python and runs its own tests with `python <file>.py`, from the study's directory. Code listings in the text are excerpts from those files. Call-outs marked **Try it** suggest a deliberate break, such as removing a safeguard and watching a named test fail; breaking a system on purpose is the fastest way to see why each safeguard matters. Call-outs marked **Warning** flag a trap that looks correct. Read each study's concepts first, run the implementation, then read the build log and compare it with how you would have sequenced the work.
