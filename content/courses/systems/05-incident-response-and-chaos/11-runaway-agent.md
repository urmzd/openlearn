# A runaway agent

An agent loop can repeatedly call tools or consume its entire budget without producing useful work. A model that keeps retrying a failing tool, or that fans out into dozens of parallel reads, has no natural stopping point, and every iteration costs tokens and time. The gateway and the agent SDK therefore need bounded behaviour, and this drill tests that the bounds hold. It exercises the loop of [The agent loop](../../ai-platform-engineering/13-agent-sdk/03-agent-loop.md) and the gates of [The tool gate, the SQL safety gate, and the prompt-injection suite](../../ai-platform-engineering/13-agent-sdk/04-tool-gate-and-injection.md), both in *AI Platform Engineering*.

The problem is a close cousin of [A noisy neighbor](10-noisy-neighbor.md): one consumer spending a shared budget. It also has the pause, fix, resume shape of [A poison task lands in the dead-letter queue](04-poison-task.md): the run stops at a budget, someone decides whether to fix the loop or grant more, and the work resumes deliberately.

The section states the principles and the drill scenario, lists what a correct runbook must cover, and derives the procedure, ending with the model runbook `docs/runbooks/RunawayAgent.md`. Try to write the procedure yourself before you read the solution.

> **Learning objectives**
> - Name the four budgets an agent loop needs and what each one bounds.
> - Recognize a loop or a fan-out from the `agent.run`, `agent.llm_call`, and `agent.tool` spans of a trace.
> - Contain a run with cancellation that reaches in-flight LLM and tool calls, behind approval gates for consequential actions.
> - Require a terminal reason that names the budget, so an operator can choose between a fix and a grant.

## Principles

Set iteration, token, wall-clock, and tool concurrency budgets. Require gates for consequential actions, and make cancellation propagate through active calls. An agent loop must be bounded on every axis: iterations (`[agent].max_iterations`, default 8), tokens, wall-clock time, and tool concurrency. Consequential tool actions need a gate (`needs_approval`). A stopped run must report a useful terminal reason, so the operator can tell a budget stop from a crash, and resume after a grant.

## The drill scenario

Run a seeded agent that repeats a tool call and another that fans out parallel reads. Confirm the loop stops at the configured budget and reports a useful terminal reason.

## What a correct runbook must cover

A runbook for a runaway agent is complete when it answers five questions:

| Case | What the runbook must contain | Why it matters |
|---|---|---|
| Owner and detection | the signal: `agent.run` spans with high `tl.agent.iterations`, repeated `agent.tool` spans, token usage | where a runaway shows |
| Budgets | iteration, token, wall-clock, and tool-concurrency limits and where they are configured | every axis bounded |
| Containment | cancel the run; cancellation reaches active calls; the gate holds consequential actions | stops spend and side effects |
| Terminal reason | the run's final status names the budget that stopped it | operators can act on it |
| Recovery | fix the loop (prompt, tool, or policy) or grant more budget, then resume | work continues deliberately |

## Deriving the procedure

### Reasoning from first principles

Why do agents run away? A model that repeats a failing tool call, or fans out reads, has no natural stopping point; each iteration costs tokens and time. The only guarantee is an external budget.

Bound every axis. Iterations alone do not bound a single huge call; tokens alone do not bound a hung tool; wall-clock alone lets a fast loop burn tokens. Each needs its own limit, and tool concurrency bounds fan-out.

Find it in the trace. `agent.run` carries `tl.agent.iterations`; its children `agent.llm_call` carry token usage and `agent.tool <name>` carry `tl.tool.outcome`. A run with the same tool repeated many times is a loop.

Stop it cleanly. Cancellation must reach active calls (the LLM request and tool subprocesses), or a cancelled run keeps spending; consequential tools need `needs_approval` gates so a loop cannot act on the world.

Say why it stopped. A terminal reason naming the budget lets someone decide between fixing the loop and granting more budget, then resuming.

### The model runbook

The procedure written out as `docs/runbooks/RunawayAgent.md`:

````markdown
# Runbook: runaway agent

Owner: the agent platform on-call (`<system>-agent`).

## Symptoms

Token usage or request rate from one agent rises without finishing; traces show
an `agent.run` with `tl.agent.iterations` at its limit, or many repeated
`agent.tool <name>` spans with the same arguments.

## Diagnosis

Open the run's trace: count iterations, repeated tools and their
`tl.tool.outcome` (`error`, `denied`, `needs_approval`), and tokens per
`agent.llm_call`. Classify: a repeating tool call (loop) or a parallel fan-out.

## Containment

Cancel the run; confirm active LLM and tool calls end (no new spans).
Check that no consequential action ran without approval.

## Recovery

Fix the cause (the tool's error handling, the prompt, or the tool policy), or
grant a larger budget explicitly, then resume. Confirm the run's terminal
reason names the budget that stopped it (iterations, tokens, wall-clock).

## Verification

A seeded looping agent stops at the configured budget with that reason; a
fan-out agent never exceeds the tool-concurrency limit.
````

### Why this solution

Independent budgets close each way a loop can spend: count, size, time, and width. Propagated cancellation makes the stop real, not cosmetic. A single timeout might seem simpler, but a fast loop can burn its whole token budget well inside any reasonable timeout, and a timeout gives no reason beyond time.

## What can go wrong

Each of these leaves a gap the loop can spend through:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| Only an iteration limit | a single run burns its token budget in a few huge calls | token, wall-clock, and concurrency budgets too |
| Cancellation that stops the loop but not active calls | spend and side effects continue after the stop | cancellation reaches LLM and tool calls; containment checks for new spans |
| A generic failure status | nobody knows whether to fix or grant | the terminal reason names the budget |

> **Warning:** A run marked "cancelled" whose LLM request and tool subprocesses are still running is still spending, and its side effects can still land. Confirm that no new spans appear.

## Where this leads

The loop and its gates come from [The agent loop](../../ai-platform-engineering/13-agent-sdk/03-agent-loop.md) and [The tool gate, the SQL safety gate, and the prompt-injection suite](../../ai-platform-engineering/13-agent-sdk/04-tool-gate-and-injection.md), and the drill follows [A noisy neighbor](10-noisy-neighbor.md). The next drill, [The event log disk fills up](12-eventlog-disk-full.md), returns to the durable control plane and a quota of a different kind. To compare this inject with the adjacent drills and reuse the same containment and verification fields, see the [chapter opening](README.md).

## Key terms

- **Runaway agent**: an agent run that keeps calling the model or tools without converging, spending budget without useful work.
- **Budget axes**: iterations, tokens, wall-clock time, and tool concurrency, each bounded independently.
- **Approval gate**: `needs_approval` on a consequential tool, which holds the action until a human allows it.
- **Propagated cancellation**: cancellation that ends in-flight LLM requests and tool subprocesses, not just the loop.
- **Terminal reason**: the final status that names which budget stopped the run.

## Summary

- An agent loop must be bounded on every axis: iterations (`[agent].max_iterations`, default 8), tokens, wall-clock time, and tool concurrency.
- A runaway shows in the trace as `agent.run` with high `tl.agent.iterations` and repeated `agent.tool` spans.
- Consequential tools need approval gates, and cancellation must propagate through in-flight LLM and tool calls.
- A stopped run must name the budget that stopped it, so the operator can fix the loop or grant more budget and resume.

## Review questions

1. Why is an iteration limit alone not enough?
   <details><summary>Answer</summary>One iteration can make a huge LLM call or hang on a tool; token and wall-clock budgets bound those, and tool concurrency bounds fan-out.</details>
2. What breaks if cancellation does not propagate to active calls?
   <details><summary>Answer</summary>The run is marked stopped while its LLM request and tool processes keep running and spending, and side effects can still land.</details>
3. Why must the terminal reason name the budget?
   <details><summary>Answer</summary>So the operator can decide: a loop needs a fix, a legitimate long task needs a grant; a generic failure gives neither.</details>
