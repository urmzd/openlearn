# Interview Preparation

This book prepares you for software engineering interviews at frontier AI labs, the largest technology platforms, selective data and infrastructure companies, quantitative trading firms, and aerospace. It is for engineers who can already write good code and are targeting full-stack roles with AI and ML infrastructure depth. Each company section explains how the loop runs, works through the problems candidates actually report, and covers the system design and behavioral preparation that separate an offer from a near miss.

## What you will be able to do
- Explain how each of your target companies structures its loop and what each round evaluates.
- Solve and defend the reported coding problems for your targets, including their edge cases and concurrency pitfalls.
- Carry a system design conversation from requirements to operations, with the numbers to back it.
- Prepare behavioral stories that fit each company's values.
- Defend a system you have built using evidence such as load reports and postmortems rather than rehearsal.

## Ways through this book
Reading every chapter takes about ten to twelve weeks at eight to ten hours a week, but most readers need only Chapter 1 plus two or three companies. Implement each sample yourself and time your practice at 45 to 60 minutes, the length of most rounds.
- **Targeting an AI lab** (about five weeks): [Chapter 1: Shared Concepts](01-shared-concepts/README.md), then [2.1 Anthropic](02-big-tech-and-ai-labs/01-anthropic/README.md) in full, including its concept sections on [concurrency](02-big-tech-and-ai-labs/01-anthropic/06-concurrency-patterns.md), [GPU inference](02-big-tech-and-ai-labs/01-anthropic/07-gpu-inference.md), and [streaming](02-big-tech-and-ai-labs/01-anthropic/08-streaming-sse.md), then [2.4 OpenAI](02-big-tech-and-ai-labs/04-openai/README.md) or [2.3 Google DeepMind](02-big-tech-and-ai-labs/03-deepmind/README.md), and finish with [Chapter 6: Defend Your System](06-defend-your-system/README.md).
- **Targeting general big tech** (about four weeks): [1.1 Distributed systems](01-shared-concepts/01-distributed-systems.md), then two or three of [2.2 Google](02-big-tech-and-ai-labs/02-google/README.md), [2.5 Meta](02-big-tech-and-ai-labs/05-meta/README.md), [2.6 Apple](02-big-tech-and-ai-labs/06-apple/README.md), and [3.2 Amazon](03-infrastructure-and-data-platforms/02-amazon/README.md), then [Chapter 6](06-defend-your-system/README.md).
- **Targeting a trading firm** (about four weeks): [1.1 Distributed systems](01-shared-concepts/01-distributed-systems.md) and [2.1.6 Concurrency patterns](02-big-tech-and-ai-labs/01-anthropic/06-concurrency-patterns.md), then two firms from [Chapter 4: Quantitative Trading Firms](04-quantitative-trading-firms/README.md), starting with [4.1 Jane Street](04-quantitative-trading-firms/01-jane-street/README.md) or [4.3 Citadel](04-quantitative-trading-firms/03-citadel/README.md).
- **The whole book, in order**: Chapters 1 through 6. Every company section builds on Chapter 1; the rest can be read in any order.

## Before you start
You should be comfortable with data structures and algorithms (the [Programming and Algorithms](../programming-and-algorithms/README.md) book covers them) and with writing production code in at least one language: Python is the most widely useful here, with C++ for the trading firms. For deeper treatment of the systems these interviews ask about, see [*Systems Engineering*](../systems/README.md), [*AI Engineering*](../ai-engineering/README.md), and [*AI Engineering*](../ai-engineering/README.md). Interview formats and pay figures come from first-person reports, forums, Glassdoor, levels.fyi, and company materials; treat them as a snapshot, since processes change. New to these books? [How to read these books](../../references/how-to-read.md) explains how sections, review questions, and solutions work.

## Choosing your targets

Reported senior compensation is a rough range, not a quote.

| Company | Focus | Senior comp |
|---------|-------|-------------|
| [2.1 Anthropic](02-big-tech-and-ai-labs/01-anthropic/README.md) | LLM inference, GPU serving, concurrency | ~$550K |
| [2.2 Google](02-big-tech-and-ai-labs/02-google/README.md) | DS&A at scale, system design, TPU/ML | ~$400-650K |
| [2.3 Google DeepMind](02-big-tech-and-ai-labs/03-deepmind/README.md) | Research-grade engineering, JAX | ~$400-750K |
| [2.4 OpenAI](02-big-tech-and-ai-labs/04-openai/README.md) | API infrastructure, ML systems | ~$450-600K |
| [2.5 Meta](02-big-tech-and-ai-labs/05-meta/README.md) | Planetary scale, AI-enabled coding round | ~$450-700K |
| [2.6 Apple](02-big-tech-and-ai-labs/06-apple/README.md) | Privacy-first and on-device ML | ~$400-650K |
| [2.7 NVIDIA](02-big-tech-and-ai-labs/07-nvidia/README.md) | GPU/CUDA, parallel computing | ~$400-750K |
| [2.8 Moonshot AI](02-big-tech-and-ai-labs/08-moonshot/README.md) | Long-context LLMs, NLP research | Competitive + equity |
| [3.1 Netflix](03-infrastructure-and-data-platforms/01-netflix/README.md) | Distributed systems, streaming, culture | ~$350-500K (cash) |
| [3.2 Amazon](03-infrastructure-and-data-platforms/02-amazon/README.md) | Leadership Principles, services | ~$350-450K (Y1) |
| [3.3 Databricks](03-infrastructure-and-data-platforms/03-databricks/README.md) | Runnable-code rounds, Spark, Delta Lake | ~$400-700K |
| [3.4 Stripe](03-infrastructure-and-data-platforms/04-stripe/README.md) | Practical coding, bug bash, payments | ~$400-650K |
| [3.5 Palantir](03-infrastructure-and-data-platforms/05-palantir/README.md) | Decomposition, data platforms | ~$350-500K |
| [4.1 Jane Street](04-quantitative-trading-firms/01-jane-street/README.md) | OCaml, probability, market making | ~$600K-1.5M |
| [4.2 Two Sigma](04-quantitative-trading-firms/02-two-sigma/README.md) | DP-heavy coding, quant reasoning | ~$500K-1M+ |
| [4.3 Citadel](04-quantitative-trading-firms/03-citadel/README.md) | Hard algorithm rounds, C++ | ~$500K-1.5M+ |
| [4.4 HRT](04-quantitative-trading-firms/04-hrt/README.md) | Nanosecond latency, FPGAs | ~$700K-2M+ |
| [4.5 Renaissance](04-quantitative-trading-firms/05-renaissance-technologies/README.md) | Invitation-only, PhD bar | ~$1M-5M+ |
| [5.1 SpaceX](05-aerospace-and-hardware/01-spacex/README.md) | 7-9 rounds, mission-critical software | ~$300-500K + equity |
