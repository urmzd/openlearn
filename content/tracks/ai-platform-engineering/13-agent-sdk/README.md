# Agent SDK

## Overview

- **Primary references**: Anthropic, [*Building effective agents*](https://www.anthropic.com/engineering/building-effective-agents) (free); OpenAI, [function calling guide](https://platform.openai.com/docs/guides/function-calling) (free); [JSON Schema](https://json-schema.org/) (free)
- **Supplementary**: Yao et al., [*ReAct*](https://arxiv.org/abs/2210.03629) (free); Greshake et al., [*Not what you've signed up for: indirect prompt injection*](https://arxiv.org/abs/2302.12173) (free); OWASP, [Top 10 for LLM Applications](https://genai.owasp.org/llm-top-10/) (free); [case study 02, grounded SQL agent](../../../case-studies/02-grounded-sql-agent/)
- **Prerequisites**: the [Go primer](../../software-craftsmanship/12-language-and-tool-primers/06-go.md), [Streaming & SSE](../03-streaming-sse/), the engine's tool calls (`L10.9`), and the durable engine ([Durable Orchestration & Workers](../05-durable-orchestration-and-workers/))
- **Estimated time**: 2 weeks in course Pass 10

## Key Takeaways

- **An agent is a loop**: call the model with messages and tool definitions, execute the tool calls it returns, append the results, and repeat until it answers or a limit stops it. Everything else is policy around that loop.
- **Providers stream deltas, not messages.** Text and tool-call arguments arrive as fragments that must be reassembled, and a retry is safe only before the first content delta.
- **Tools are an attack surface.** Arguments are validated against a JSON Schema, a gate decides allow, deny, or needs-approval before dispatch, and text that came from a retrieved document or a tool result never authorizes a write.
- **Long agent runs need durability.** Each model call and tool call is a recorded step, so a killed worker resumes without calling the model or the tool twice.

## How to Study

Read *Building effective agents* first, then the ReAct paper. Port the safety gate of [case study 02](../../../case-studies/02-grounded-sql-agent/) to Go before `ag.04`. In the course, the SDK is built in Pass 10 against the learner's own gateway and engine (SmolLM2-135M-Instruct with tool calls), with a frontier provider usable through the same `Provider` interface.

---

# Concepts & Techniques

## Core Insight

The model proposes; the program disposes. A language model can only emit text that names a tool and its arguments; the SDK decides whether that call runs, with what limits, and what happens when it fails halfway. Designing the SDK means designing those decisions as typed, testable code: a provider interface, a tool registry, a gate, a step runner, and a loop with budgets.

## 1. Types and providers

**Key ideas**:
- **Messages, tool calls, tool definitions, deltas** as Go types, and a `Provider` with `ChatStream` returning a channel of deltas (`ag.01`).
- **One provider for every backend**: the learner's gateway and a frontier API speak the same OpenAI-compatible subset.

## 2. Tools and the loop

**Key ideas**:
- **`Tool`** has a `Definition` (name, description, JSON Schema) and `Execute`; invalid arguments return a tool error to the model, never a panic (`ag.02`).
- **The loop** (`ag.03`) bounds iterations, parallel tools (preserving result order), and spend, and stops before dispatch when a budget would break.

## 3. Gates and prompt injection

**Key ideas**:
- **`Gate.Check`** returns `Allow`, `Deny{Reason}`, or `NeedApproval{Marker}` (`ag.04`); the deterministic SQL gate from case study 02 rejects every statement in the BLOCKED corpus.
- **Injection suite**: instructions planted in retrieved chunks and tool results never trigger a gated or write tool without approval.

## 4. Durable runs

**Key ideas**:
- **`AgentRun`** (`ag.05`) runs each step through a `StepRunner` on the learner's durable engine; a write tool with an unknown outcome yields `ErrIndeterminate` and waits for a reconcile signal instead of guessing.

## Course modules

| Module | Topic | Kind | Pass |
|---|---|---|---|
| `ag.01` | Types, OpenAI-compatible provider, retry wrapper | build | 10 |
| `ag.02` | Tools, registry, JSON Schema argument validation | build | 10 |
| `ag.03` | Agent loop | build | 10 |
| `ag.04` | Tool gate and deterministic SQL safety gate (case study 02 ported) | build | 10 |
| `ag.05` | Durable agent runs (`AgentRun`) | build | 10 |

Retrieval (`ag.06` to `ag.08`) lives in [Retrieval & RAG](../07-retrieval-and-rag/) and evaluation (`ag.09` to `ag.12`) in [LLM Evaluation](../09-llm-evaluation/); together they close milestone `MS-agent`.

## Chapters

| # | Module | Chapter | Kind | Pass |
|---|---|---|---|---|
| 1 | `ag.01` | [Types, OpenAI-compatible provider, retry wrapper](01-types-and-provider.md) | build | 10 |
| 2 | `ag.02` | [Tools, registry, JSON Schema argument validation](02-tools-and-schemas.md) | build | 10 |
| 3 | `ag.03` | [Agent loop](03-agent-loop.md) | build | 10 |
| 4 | `ag.04` | [Tool gate, SQL safety gate, prompt-injection suite](04-tool-gate-and-injection.md) | build | 10 |
| 5 | `ag.05` | [Durable agent runs (AgentRun)](05-durable-agent-runs.md) | build | 10 |

## Connections to Other Tracks

| Track | Connection |
|---|---|
| [Retrieval & RAG](../07-retrieval-and-rag/) | `search_docs`, the agent's retrieval tool |
| [LLM Evaluation](../09-llm-evaluation/) | agent suites run the loop as their subject |
| [Gateway](../12-gateway/) | the provider endpoint, keys, limits, and usage policy |
| [Durable Orchestration & Workers](../05-durable-orchestration-and-workers/) | the engine behind `AgentRun` |
| [Usage policy](../../responsible-ai/05-usage-policy/) | what the gateway refuses before a request reaches the agent |

## Company Relevance

| Company | Practice |
|---|---|
| Anthropic, OpenAI | tool use APIs and agent SDKs built on the same loop |
| Temporal | durable agent workflows where every model and tool call is a recorded activity |
