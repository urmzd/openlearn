# Agent SDK

A language model can only emit text that names a tool and its arguments. Everything that turns that text into action, and everything that keeps the action safe, is ordinary program code: a client that decodes the model's stream, a registry that validates arguments, a loop that dispatches calls under limits, a gate that decides which calls may run at all, and a journal that lets a run survive a crash. This chapter builds that code in Go, as a small agent SDK, against the gateway and engine from earlier chapters. A frontier provider works through the same `Provider` interface.

The guiding idea is short: **the model proposes; the program disposes.** The SDK decides whether a call runs, with what limits, and what happens when it fails halfway, and it makes each of those decisions as typed, testable code. The chapter's arc follows a single request from the bottom up. Providers stream deltas, not messages, so the first section reassembles text and tool-call fragments and shows that a retry is safe only before the first content delta. Tools are an attack surface, so the next two sections validate arguments against a JSON Schema and wrap the agent loop (call the model, run the calls, append the results, repeat until it answers or a limit stops it) with iteration caps, bounded parallelism, and budgets. The fourth section adds a deterministic gate that allows, denies, or holds each call for approval, so that text from a retrieved document or a tool result never authorizes a write. The last section makes long runs durable: every model and tool call is a recorded step, so a killed worker resumes without calling the model or the tool twice.

> **Learning objectives**
> - Decode a streamed chat completion into typed deltas, reassemble interleaved tool calls, and retry only where it is safe.
> - Validate model-written tool arguments against a JSON Schema subset and turn every failure into a message the model can read.
> - Build an agent loop with an iteration cap, bounded parallel dispatch in call order, and budgets checked before the spend.
> - Gate tool calls with a default-deny policy, an egress allowlist, taint, a lexer-based SQL classifier, and approval markers, and test it against a prompt-injection suite.
> - Make agent runs durable with a step journal, and explain why an unrecorded write must be reconciled by a human.

## Before you begin

You should be comfortable with Go; the [Go primer](../../programming-languages/02-language-and-tool-primers/06-go.md) in *Programming Languages* covers goroutines, channels, context, and `net/http`. The provider decodes server-sent events, so read [Streaming and SSE](../03-streaming-sse/README.md) and the gateway's [SSE streaming proxy](../12-gateway/04-sse-streaming-proxy.md) first. The model on the other end emits tool calls as described in [Tool calls and constrained JSON decoding in the engine](../../build-an-llm/p10-serving/09-tool-calls-and-constrained-decoding.md) in *Build an LLM*. Durable runs reuse the ideas of [Durable Orchestration and Workers](../05-durable-orchestration-and-workers/README.md). The SQL gate is a Go port of the safety gate in the [Grounded SQL Agent case study](../../case-studies/02-grounded-sql-agent/); porting it yourself before reading the gate section is a good exercise. Reading Anthropic's *Building effective agents* and then the ReAct paper (both listed below) gives useful background for the loop.

## In this chapter

1. [Types, an OpenAI-compatible provider, and a retry wrapper](01-types-and-provider.md): the message, tool-call, and delta types; an SSE reader and chunk decoder that reassemble tool calls by index; and a retry wrapper that retries only before the first content delta.
2. [Tools, a registry, and JSON Schema argument validation](02-tools-and-schemas.md): a tool registry that validates every call against a small, exact JSON Schema subset and returns every failure, including panics, as a readable tool message.
3. [The agent loop](03-agent-loop.md): the loop that connects provider and registry, with an iteration cap, bounded parallel calls returned in order, budgets, the gate and approval seam, named steps, and tracing spans.
4. [The tool gate, the SQL safety gate, and the prompt-injection suite](04-tool-gate-and-injection.md): a deterministic default-deny policy with egress checks, taint, a lexer-based SQL classifier, approval markers, and twelve injection conversations it must withstand.
5. [Durable agent runs](05-durable-agent-runs.md): an fsynced step journal that replays completed steps, re-runs unfinished reads, and stops on unfinished writes until a human reconciles them.

The agent system continues in two other chapters of this book: retrieval (the `search_docs` tool, ingest, and context assembly) in [Retrieval and RAG](../07-retrieval-and-rag/README.md), and agent evaluation in [LLM Evaluation](../09-llm-evaluation/README.md). Together with this chapter they complete the agent milestone described in [the Pass 10 milestone](../../build-an-llm/paths/course-p10-agents/milestone.md) of *Build an LLM*.

## How this chapter connects

The [gateway](../12-gateway/README.md) is the provider endpoint, with its keys, limits, and usage policy, and [Usage policy](../../responsible-ai/05-usage-policy/README.md) in *Responsible AI* covers what the gateway refuses before a request ever reaches the agent. [Retrieval and RAG](../07-retrieval-and-rag/README.md) supplies `search_docs`, the agent's retrieval tool, and [LLM Evaluation](../09-llm-evaluation/README.md) runs the loop as the subject of its agent suites. [Durable Orchestration and Workers](../05-durable-orchestration-and-workers/README.md) is the engine the durable run is designed to move onto. In industry, Anthropic and OpenAI build their tool-use APIs and agent SDKs on the same loop, and Temporal runs durable agent workflows in which every model and tool call is a recorded activity.

## Further reading

- Anthropic, [*Building effective agents*](https://www.anthropic.com/engineering/building-effective-agents) (free).
- OpenAI, [function calling guide](https://platform.openai.com/docs/guides/function-calling) (free).
- [JSON Schema](https://json-schema.org/) (free).
- Yao et al., [*ReAct*](https://arxiv.org/abs/2210.03629) (free).
- Greshake et al., [*Not what you've signed up for: indirect prompt injection*](https://arxiv.org/abs/2302.12173) (free).
- OWASP, [Top 10 for LLM Applications](https://genai.owasp.org/llm-top-10/) (free).
