# Drill: runaway agent

## Overview

| | |
|---|---|
| **Module** | `ops.10` · drill · ops, docs · Pass 11 · 1 to 2 h |
| **Solution** | the model runbook `docs/runbooks/RunawayAgent.md`, written out with its reasoning in section 5 |
| **Needs** | none |
| **Used by** | no code call site: a drill, one of the Pass 11 operations drills |

## Key Takeaways

- An agent loop must be bounded on every axis: iterations (`[agent].max_iterations`, default 8), tokens, wall-clock time, and tool concurrency.
- Consequential tool actions need a gate (`needs_approval`), and cancellation must propagate through in-flight LLM and tool calls.
- A stopped run must report a useful terminal reason, so the operator can tell a budget stop from a crash, and resume after a grant.

## How to read this chapter

This chapter is a worked drill. Sections 1 to 3 state the risk, the principles, and the drill scenario. Section 4 lists what a correct runbook must cover. Section 5 derives the procedure from first principles, gives the model runbook, and explains why each step is the right one. Try to write the procedure yourself from sections 2 and 3 before reading section 5.

---

## 1. Why now

An agent loop can repeatedly call tools or consume its entire budget without producing useful work. The gateway and SDK need bounded behavior.

## 2. Principles

Set iteration, token, wall-clock, and tool concurrency budgets. Require gates for consequential actions, and make cancellation propagate through active calls.

## 3. Worked example

Run a seeded agent that repeats a tool call and another that fans out parallel reads. Confirm the loop stops at the configured budget and reports a useful terminal reason.

## 4. What a correct runbook must cover

| Case | What the runbook must contain | Why it matters |
|---|---|---|
| Owner and detection | the signal: `agent.run` spans with high `tl.agent.iterations`, repeated `agent.tool` spans, token usage | where a runaway shows |
| Budgets | iteration, token, wall-clock, and tool-concurrency limits and where they are configured | every axis bounded |
| Containment | cancel the run; cancellation reaches active calls; the gate holds consequential actions | stops spend and side effects |
| Terminal reason | the run's final status names the budget that stopped it | operators can act on it |
| Recovery | fix the loop (prompt, tool, or policy) or grant more budget, then resume | work continues deliberately |

## 5. The solution, step by step

### 5.1 From first principles

1. **Why do agents run away?** A model that repeats a failing tool call, or fans out reads, has no natural stopping point; each iteration costs tokens and time. The only guarantee is an external budget.
2. **Bound every axis.** Iterations alone do not bound a single huge call; tokens alone do not bound a hung tool; wall-clock alone lets a fast loop burn tokens. Each needs its own limit, and tool concurrency bounds fan-out.
3. **Find it in the trace.** `agent.run` carries `tl.agent.iterations`; its children `agent.llm_call` carry token usage and `agent.tool <name>` carry `tl.tool.outcome`. A run with the same tool repeated many times is a loop.
4. **Stop it cleanly.** Cancellation must reach active calls (the LLM request and tool subprocesses), or a cancelled run keeps spending; consequential tools need `needs_approval` gates so a loop cannot act on the world.
5. **Say why it stopped.** A terminal reason naming the budget lets someone decide between fixing the loop and granting more budget, then resuming.

### 5.2 The model runbook

`docs/runbooks/RunawayAgent.md`:

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

### 5.3 Why this solution

- **Independent budgets** close each way a loop can spend: count, size, time, and width.
- **Propagated cancellation** makes the stop real, not cosmetic.
- **Why not just a timeout?** A fast loop can burn its whole token budget well inside any reasonable timeout; a timeout also gives no reason beyond time.

### 5.4 Explain it back

1. Why is an iteration limit alone not enough?
   <details><summary>Answer</summary>One iteration can make a huge LLM call or hang on a tool; token and wall-clock budgets bound those, and tool concurrency bounds fan-out.</details>
2. What breaks if cancellation does not propagate to active calls?
   <details><summary>Answer</summary>The run is marked stopped while its LLM request and tool processes keep running and spending, and side effects can still land.</details>
3. Why must the terminal reason name the budget?
   <details><summary>Answer</summary>So the operator can decide: a loop needs a fix, a legitimate long task needs a grant; a generic failure gives neither.</details>

## 6. Pitfalls

| # | Pitfall | Symptom | How the solution avoids it |
|---|---|---|---|
| 1 | Only an iteration limit | a single run burns its token budget in a few huge calls | token, wall-clock, and concurrency budgets too |
| 2 | Cancellation that stops the loop but not active calls | spend and side effects continue after the stop | cancellation reaches LLM and tool calls; containment checks for new spans |
| 3 | A generic failure status | nobody knows whether to fix or grant | the terminal reason names the budget |

## 7. Where it's used next

| Direction | Module | Connection |
|---|---|---|
| Back | `ops.09` | A prerequisite: the contract or evidence this runbook relies on. |
| Back | `ag.03` | A prerequisite: the contract or evidence this runbook relies on. |
| Back | `ag.04` | A prerequisite: the contract or evidence this runbook relies on. |

## Going further

| Resource | What to inspect |
|---|---|
| [Incident drill catalog](README.md) | Compare this inject with the adjacent failure drills and use the same containment and verification fields. |
