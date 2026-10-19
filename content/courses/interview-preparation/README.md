# Interview Preparation

This book prepares you for software engineering interviews at the companies that are hardest to get into: frontier AI labs, the largest technology platforms, the most selective data and infrastructure companies, quantitative trading firms, and aerospace. It is aimed at engineers targeting full-stack roles with deep AI and ML infrastructure expertise, and it assumes you can already write good code. What it adds is a precise picture of how each company interviews, worked solutions to the problems candidates actually report, and the system design and behavioral preparation that separate an offer from a near miss.

## What you will be able to do

By the end of the book you will be able to explain how each target company's loop is structured and what each round evaluates; solve and defend the reported coding problems for your targets, including their edge cases and concurrency pitfalls; carry a system design conversation from requirements to operations, with the numbers to back it; prepare behavioral stories that fit each company's values; and defend a system you have built using evidence rather than rehearsal.

## How this book is organized

The book has three kinds of chapters. The foundation chapters come first because every company chapter builds on them. The company chapters can be read in any order; pick two or three targets at a time. And one chapter closes the book by turning your own work into interview answers.

### Foundations

[Shared concepts](shared-concepts/README.md) collects the material that every system design round returns to, whatever the company. Its sections cover [distributed systems fundamentals](shared-concepts/distributed-systems.md) (consistency, replication, partitioning, consensus, and failure), [full-stack AI engineering](shared-concepts/full-stack-ai.md) (every layer of an AI system from data pipelines and feature stores to serving and monitoring), and [ML infrastructure patterns](shared-concepts/ml-infrastructure.md) (registries, promotion, distributed training, drift detection, and retraining). Start here.

### Big tech and AI labs

[Anthropic](anthropic/README.md) is the most detailed chapter in the book, with one section per round of the infrastructure loop and three concept sections on concurrency, GPU inference serving, and streaming over Server-Sent Events. [Google](google/README.md) covers data structures and algorithms at scale, system design, Googleyness, and TPU and ML topics. [Google DeepMind](deepmind/README.md) covers its research-grade engineering bar, JAX, and the systems behind AlphaFold and Gemini. [OpenAI](openai/README.md) covers API infrastructure, ML systems, and mission alignment. [Meta](meta/README.md) covers planetary-scale systems, its AI-enabled coding round, and LLaMA. [Apple](apple/README.md) covers privacy-first ML, on-device inference, and product excellence. [NVIDIA](nvidia/README.md) covers GPU and CUDA programming, parallel computing, and AI infrastructure. And [Moonshot AI](moonshot/README.md) covers long-context LLMs and NLP research at the Chinese AI frontier.

### Infrastructure and data platforms

[Netflix](netflix/README.md) covers distributed systems, streaming, recommendations, and its distinctive culture. [Amazon](amazon/README.md) covers the Leadership Principles that are tested in every round, services architecture, and AWS ML. [Databricks](databricks/README.md) covers the most selective loop outside the largest platforms, with runnable-code rounds and Lakehouse, Spark, and Delta Lake design. [Stripe](stripe/README.md) covers its no-LeetCode loop: practical coding, the bug bash, payment infrastructure, and financial correctness. And [Palantir](palantir/README.md) covers problem decomposition, data platforms, entity modeling, and AIP.

### Quantitative trading firms

[Jane Street](jane-street/README.md) covers OCaml, probability, low latency, and market making. [Two Sigma](two-sigma/README.md) covers DP-heavy coding, data infrastructure, and quantitative reasoning. [Citadel and Citadel Securities](citadel/README.md) covers some of the hardest algorithm rounds anywhere, C++, and low-latency matching engines. [Hudson River Trading](hrt/README.md) covers nanosecond latency, C++, FPGAs, and kernel bypass. And [Renaissance Technologies](renaissance-technologies/README.md) covers the most exclusive firm of all: invitation-only, with a PhD-level bar.

### Aerospace and hardware

[SpaceX](spacex/README.md) covers a seven-to-nine-round process, mission-critical software, and Starlink.

### Defending your own system

[Defend your system](defend-your-system/README.md) closes the book. It is a bank of design questions answered against a system you have actually built, with the architecture decision records, load reports, and postmortems that back each answer. Its premise is that the strongest interview answer is about a system you know where it breaks.

## Choosing your targets

The companies differ in focus and pay. Reported senior compensation and focus areas are summarized below; treat the figures as rough, reported ranges.

| Company | Focus Areas | Comp Range (Senior) |
|---------|-------------|-------------------|
| [Anthropic](anthropic/README.md) | LLM inference, GPU serving, concurrency, safety | ~$550K |
| [Google](google/README.md) | DS&A at scale, system design, Googleyness, TPU/ML | ~$400-650K |
| [Google DeepMind](deepmind/README.md) | ML research + engineering, JAX, AlphaFold, Gemini | ~$400-750K |
| [OpenAI](openai/README.md) | API infrastructure, ML systems, mission alignment | ~$450-600K |
| [Meta](meta/README.md) | Planetary scale, AI-enabled coding round, LLaMA | ~$450-700K |
| [Apple](apple/README.md) | Privacy-first ML, on-device inference, product excellence | ~$400-650K |
| [NVIDIA](nvidia/README.md) | GPU/CUDA programming, parallel computing, AI infra | ~$400-750K |
| [Moonshot AI](moonshot/README.md) | Long-context LLMs, NLP research, Chinese AI frontier | Competitive + equity |
| [Netflix](netflix/README.md) | Distributed systems, streaming, culture, recommendations | ~$350-500K (cash) |
| [Amazon](amazon/README.md) | Leadership Principles, services architecture, AWS/ML | ~$350-450K (Y1) |
| [Databricks](databricks/README.md) | Most selective non-FAANG, Lakehouse, Spark, Delta Lake | ~$400-700K |
| [Stripe](stripe/README.md) | No leetcode, bug bash, payments infra, financial correctness | ~$400-650K |
| [Palantir](palantir/README.md) | Decomposition, data platforms, entity modeling, AIP | ~$350-500K |
| [Jane Street](jane-street/README.md) | OCaml, probability, low-latency, market making | ~$600K-1.5M |
| [Two Sigma](two-sigma/README.md) | DP-heavy coding, data infra, quant reasoning | ~$500K-1M+ |
| [Citadel / Citadel Securities](citadel/README.md) | Hardest algo rounds, C++, low-latency matching | ~$500K-1.5M+ |
| [HRT](hrt/README.md) | Nanosecond latency, C++, FPGA, kernel bypass | ~$700K-2M+ |
| [Renaissance Technologies](renaissance-technologies/README.md) | Most exclusive firm on Earth, invite-only, PhD bar | ~$1M-5M+ |
| [SpaceX](spacex/README.md) | 7-9 rounds, mission-critical software, Starlink | ~$300-500K + equity |

Difficulty varies as much as pay. The tiers below reflect selectivity and the technical bar:

| Tier | Companies | What Makes It Hard |
|------|-----------|-------------------|
| Near-impossible | Renaissance Technologies | Invite-only, ~300 employees, PhD required |
| Extreme | HRT, SpaceX, Citadel, Jane Street | Tiny headcount, world-class algorithmic bar |
| Very Hard | DeepMind, Anthropic, Databricks, Two Sigma | Research-grade bar, highly selective |
| Hard | Google, Meta, OpenAI, Stripe, NVIDIA, Moonshot | Structured but demanding, massive applicant pools |
| Hard | Netflix, Amazon, Apple, Palantir | Strong bar with unique cultural/domain requirements |

The loops also differ in shape. Counting rounds by type for ten of the companies shows where each puts its weight:

| Round Type | Google | Meta | OpenAI | Netflix | Amazon | Stripe | NVIDIA | Jane Street | Citadel | SpaceX |
|------------|--------|------|--------|---------|--------|--------|--------|-------------|---------|--------|
| Coding | 2-3 | 2 (1 AI-enabled) | 2 | 1 | 2 | 1 (practical) | 1-2 | 1-2 | 2 | 3-4 |
| System Design | 1 | 1 | 1 | 1-2 | 1 | 1 (payments) | 1 | 1 | 1 | 1 |
| Behavioral | 1 | 1 | 1 | 1-2 (culture) | 4 (all rounds) | 1 | 1 | 1 | 1 | 1 |
| Unique Round | -- | AI coding | -- | Culture deep dive | Bar Raiser | Bug Bash + Integration | GPU/CUDA | Probability/Math | -- | Take-home (4hr) |
| Total Rounds | 5-7 | 5-6 | 5-6 | 4-5 | 5-6 | 5 | 4-6 | 5-7 | 4-5 | 7-9 |

## How to use this book

Start with the shared concepts, which build the foundation that applies everywhere. Then pick your target companies, two or three at a time, and study each one's patterns, since every company has its own interview style. Do not just read the code: implement every sample yourself, and look for the edge cases and limitations the chapters point out, because interviewers ask about exactly those. Time yourself, since most rounds are 45 to 60 minutes with 5 to 10 minutes reserved for your questions. And cross-reference as you go: many concepts recur, so concurrency, for example, appears at Anthropic, Google, NVIDIA, and every quantitative firm.

## Prerequisites and related books

You should be comfortable with data structures and algorithms and with writing production code in at least one language (Python is the most widely useful here, with C++ for the trading firms). Several other books in this collection go deeper on the systems these interviews ask about. *Infrastructure* (start at its [front matter](../infrastructure/README.md)) covers containers, messaging, search, and the Apache data stack. *AI Platform Engineering* (start at its [front matter](../ai-platform-engineering/README.md)) covers durable workflows, distributed data, and retrieval. *Build Your Own LLM System* (start at its [front matter](../build-an-llm/README.md)) builds a language model and its serving stack from scratch. And *Systems and Architecture* (start at its [front matter](../systems/README.md)) covers system design and architecture in depth.

## Conventions

Code appears in fenced listings with a language tag; most solutions are in Python, with C++ where the company works in it. Interview formats, round counts, and compensation figures are compiled from first-person reports, community forums, Glassdoor, levels.fyi, and company materials, and each chapter lists its sources; treat them as a well-informed snapshot, since processes change. Call-outs marked **Warning** flag a bug, an edge case, or a limitation in a reference solution that you should be ready to discuss. Chapters end with key terms, a summary, and review questions whose answers are hidden in expandable blocks; try each one before you open the answer.
