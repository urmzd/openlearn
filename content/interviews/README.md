# Interview Preparation Guides

Comprehensive interview preparation for the **world's hardest companies to get into**, targeting full-stack engineering roles with deep AI/ML infrastructure expertise.

## Companies

### Big Tech & AI Labs

| Company | Focus Areas | Comp Range (Senior) |
|---------|-------------|-------------------|
| [Anthropic](anthropic/) | LLM inference, GPU serving, concurrency, safety | ~$550K |
| [Google](google/) | DS&A at scale, system design, Googleyness, TPU/ML | ~$400-650K |
| [Google DeepMind](deepmind/) | ML research + engineering, JAX, AlphaFold, Gemini | ~$400-750K |
| [OpenAI](openai/) | API infrastructure, ML systems, mission alignment | ~$450-600K |
| [Meta](meta/) | Planetary scale, AI-enabled coding round, LLaMA | ~$450-700K |
| [Apple](apple/) | Privacy-first ML, on-device inference, product excellence | ~$400-650K |
| [NVIDIA](nvidia/) | GPU/CUDA programming, parallel computing, AI infra | ~$400-750K |
| [Moonshot AI](moonshot/) | Long-context LLMs, NLP research, Chinese AI frontier | Competitive + equity |

### Infrastructure & Data Platforms

| Company | Focus Areas | Comp Range (Senior) |
|---------|-------------|-------------------|
| [Netflix](netflix/) | Distributed systems, streaming, culture, recommendations | ~$350-500K (cash) |
| [Amazon](amazon/) | Leadership Principles, services architecture, AWS/ML | ~$350-450K (Y1) |
| [Databricks](databricks/) | Most selective non-FAANG, Lakehouse, Spark, Delta Lake | ~$400-700K |
| [Stripe](stripe/) | No leetcode, bug bash, payments infra, financial correctness | ~$400-650K |
| [Palantir](palantir/) | Decomposition, data platforms, entity modeling, AIP | ~$350-500K |

### Quantitative Trading Firms

| Company | Focus Areas | Comp Range (Senior) |
|---------|-------------|-------------------|
| [Jane Street](jane-street/) | OCaml, probability, low-latency, market making | ~$600K-1.5M |
| [Two Sigma](two-sigma/) | DP-heavy coding, data infra, quant reasoning | ~$500K-1M+ |
| [Citadel / Citadel Securities](citadel/) | Hardest algo rounds, C++, low-latency matching | ~$500K-1.5M+ |
| [HRT (Hudson River Trading)](hrt/) | Nanosecond latency, C++, FPGA, kernel bypass | ~$700K-2M+ |
| [Renaissance Technologies](renaissance-technologies/) | Most exclusive firm on Earth, invite-only, PhD bar | ~$1M-5M+ |

### Aerospace & Hardware

| Company | Focus Areas | Comp Range (Senior) |
|---------|-------------|-------------------|
| [SpaceX](spacex/) | 7-9 rounds, mission-critical software, Starlink | ~$300-500K + equity |

## Difficulty Ranking

| Tier | Companies | What Makes It Hard |
|------|-----------|-------------------|
| Near-impossible | Renaissance Technologies | Invite-only, ~300 employees, PhD required |
| Extreme | HRT, SpaceX, Citadel, Jane Street | Tiny headcount, world-class algorithmic bar |
| Very Hard | DeepMind, Anthropic, Databricks, Two Sigma | Research-grade bar, highly selective |
| Hard | Google, Meta, OpenAI, Stripe, NVIDIA, Moonshot | Structured but demanding, massive applicant pools |
| Hard | Netflix, Amazon, Apple, Palantir | Strong bar with unique cultural/domain requirements |

## Cross-Company Concepts

| Topic | Relevance |
|-------|-----------|
| [Full-Stack AI Engineering](shared-concepts/full-stack-ai.md) | Data pipelines, feature stores, model serving, monitoring |
| [Distributed Systems](shared-concepts/distributed-systems.md) | Consensus, replication, caching, load balancing |
| [ML Infrastructure](shared-concepts/ml-infrastructure.md) | Training infra, experiment tracking, drift detection |

## Interview Round Comparison

| Round Type | Google | Meta | OpenAI | Netflix | Amazon | Stripe | NVIDIA | Jane Street | Citadel | SpaceX |
|------------|--------|------|--------|---------|--------|--------|--------|-------------|---------|--------|
| Coding | 2-3 | 2 (1 AI-enabled) | 2 | 1 | 2 | 1 (practical) | 1-2 | 1-2 | 2 | 3-4 |
| System Design | 1 | 1 | 1 | 1-2 | 1 | 1 (payments) | 1 | 1 | 1 | 1 |
| Behavioral | 1 | 1 | 1 | 1-2 (culture) | 4 (all rounds) | 1 | 1 | 1 | 1 | 1 |
| Unique Round | -- | AI coding | -- | Culture deep dive | Bar Raiser | Bug Bash + Integration | GPU/CUDA | Probability/Math | -- | Take-home (4hr) |
| Total Rounds | 5-7 | 5-6 | 5-6 | 4-5 | 5-6 | 5 | 4-6 | 5-7 | 4-5 | 7-9 |

## Defend Your System

[Defend Your System](defend-your-system/) is the course's interview module (`iv.01`, Pass 11): a question bank answered against the system you built, with the ADRs, load reports, and postmortems that back each answer.

## How to Use This Guide

1. **Start with shared concepts** -- Build the foundation that applies everywhere
2. **Pick your target companies** -- Focus on 2-3 at a time
3. **Study company-specific patterns** -- Each company has unique interview styles
4. **Practice with code** -- Don't just read; implement every code sample
5. **Time yourself** -- Most rounds are 45-60 minutes with 5-10 min for questions
6. **Cross-reference** -- Many concepts overlap (concurrency appears at Anthropic, Google, NVIDIA, and all quant firms)
