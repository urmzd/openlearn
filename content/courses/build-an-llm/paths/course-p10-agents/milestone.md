# What the system does after Pass 10

**Pass result**: Go agent SDK on the engine's tool calls, RAG over the system's docs, eval runner with judge and A/B, usage policy at the gateway, agent chart; optional post-training.

A durable agent now answers questions about the system itself: it retrieves from the system's own docs, calls tools through the engine's tool-call API, cites what it used, runs as a durable workflow, and sits behind the gateway's usage policy. Everything from earlier passes keeps working.

| What the system does | Built from |
|---|---|
| A durable agent answers with citations and passes evals and policy checks. | `ag.01`, `ag.02`, `ag.03`, `ag.04`, `ag.05`, `ag.06`, `ag.07`, `ag.08`, `ag.09`, `ag.10`, `ag.11`, `ag.12`, `ethics.05`, `gw.08`, `dep.07`, `craft.23`, `S-M10b` |
| Optional depth: post-training improves the TinyStories model on held-out preference tasks. | `C2`, `L12.1`, `L12.2`, `L12.3`, `L12.4`, `S-M10b` |

## Details

## A durable agent answers with citations and passes evals and policy checks

The services involved are the engine, the gateway, the durable server and worker of Pass 8, and the agent. With the local SmolLM2-135M-Instruct model the agent plans for itself; a deterministic fake tool provider ([`agent.faketool.json`](../../../../../course/solutions/entry/config/agent.faketool.json)) runs the same prompts and tools reproducibly. The `ctl` verbs are in [`go/cmd/ctl`](../../../../../course/solutions/entry/go/cmd/ctl/agent.go).

| Behaviour | Command | What it shows | Why it is needed |
|---|---|---|---|
| The agent answers with citations | `{ctl} agent ask --query "Where are the serving runbooks?" --suite course-docs` | at least one citation, and every tool call succeeds | retrieval-augmented answers are only trustworthy when the sources are named |
| Evals and an A/B experiment | `{ctl} eval experiment --base tinystories-10m --exp smol-135m-instruct --suite agent-smoke` | at least 5 samples per arm and a confidence interval on the difference | comparing two models needs an interval, not two point estimates |
| The usage policy at the gateway | `POST /v1/chat/completions` with a request the policy forbids | HTTP 451 with error code `usage_policy` | policy is enforced once, at the edge, for every model and client |
| Prompt-injection cases | `{ctl} eval run --suite <prompt-injection cases>` | zero unsafe tool calls over at least 10 cases | text retrieved from documents must never be able to direct the agent's tools |

The agent runs as a durable workflow so a long multi-step task survives a worker restart, and its chart (dep.07) deploys it next to the rest of the system.

## Optional depth: post-training improves the TinyStories model on held-out preference tasks

An optional local post-training capstone (C2, with L12.1 to L12.4). The v1.0.0 core course ends at the C1 capstone of Pass 9.
