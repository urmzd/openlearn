# What the System Does After Pass 10

After Pass 10 a durable agent answers questions about the system itself. It retrieves from the system's own documentation, calls tools through the engine's tool-call API, cites what it used, runs as a durable workflow, and sits behind the gateway's usage policy. It passes its evals and policy checks. As optional depth, post-training improves the TinyStories model on held-out preference tasks. Everything from earlier passes keeps working.

The agent is built from the agent-SDK chapter of *AI Platform Engineering*, from [types and the provider](../../../ai-platform-engineering/13-agent-sdk/01-types-and-provider.md), [tools and schemas](../../../ai-platform-engineering/13-agent-sdk/02-tools-and-schemas.md), [the agent loop](../../../ai-platform-engineering/13-agent-sdk/03-agent-loop.md), and [the tool gate](../../../ai-platform-engineering/13-agent-sdk/04-tool-gate-and-injection.md) to [durable agent runs](../../../ai-platform-engineering/13-agent-sdk/05-durable-agent-runs.md); from retrieval, with [RAG ingest](../../../ai-platform-engineering/07-retrieval-and-rag/06-rag-ingest.md), [retrieval](../../../ai-platform-engineering/07-retrieval-and-rag/07-retrieval.md), and [context assembly and citations](../../../ai-platform-engineering/07-retrieval-and-rag/08-context-assembly.md); and from evaluation, with [the eval runner](../../../ai-platform-engineering/09-llm-evaluation/09-eval-runner.md), [scorers](../../../ai-platform-engineering/09-llm-evaluation/10-scorers.md), [the LLM judge](../../../ai-platform-engineering/09-llm-evaluation/11-llm-judge.md), and [paired A/B experiments](../../../ai-platform-engineering/09-llm-evaluation/12-ab-experiments.md). The policy is [Usage policy at the gateway](../../../responsible-ai/05-usage-policy/01-usage-policy-at-the-gateway.md) and its [enforcement](../../../ai-platform-engineering/12-gateway/08-usage-policy-enforcement.md); the deployment is [the agent image and chart](../../../infrastructure/01-containers-kubernetes/07-agent-image-and-chart.md); and the tests are [agent evals as tests](../../../software-craftsmanship/03-testing-mentality/23-agent-evals-as-tests.md).

## A durable agent answers with citations and passes evals and policy checks

The services involved are the engine, the gateway, the durable server and worker of Pass 8, and the agent. With the local SmolLM2-135M-Instruct model, the agent plans for itself. A deterministic fake tool provider, [`agent.faketool.json`](../../../../../course/solutions/entry/config/agent.faketool.json), runs the same prompts and tools reproducibly. The `ctl` verbs are in [`go/cmd/ctl`](../../../../../course/solutions/entry/go/cmd/ctl/agent.go).

Four behaviours define the pass. The first is an answer with citations:

```text
{ctl} agent ask --query "Where are the serving runbooks?" --suite course-docs
```

The answer carries at least one citation, and every tool call succeeds. Retrieval-augmented answers are only trustworthy when their sources are named.

The second is an evaluation with an A/B experiment:

```text
{ctl} eval experiment --base tinystories-10m --exp smol-135m-instruct --suite agent-smoke
```

It reports at least 5 samples per arm and a confidence interval on the difference, because comparing two models needs an interval, not two point estimates.

The third is the usage policy at the gateway. A `POST /v1/chat/completions` with a request the policy forbids is answered with HTTP 451 and the error code `usage_policy`. Policy is enforced once, at the edge, for every model and client.

The fourth is the prompt-injection suite, run with `{ctl} eval run --suite <prompt-injection cases>`. Over at least 10 cases there are zero unsafe tool calls: text retrieved from documents must never be able to direct the agent's tools.

The agent runs as a durable workflow, so a long multi-step task survives a worker restart, and its chart deploys it next to the rest of the system.

## Optional depth: post-training

An optional local post-training capstone, [Capstone 2: post-trained TinyStories model](../../capstones/02-post-trained.md), improves the TinyStories model on held-out preference tasks. It uses [supervised fine-tuning](../../p12-post-training/01-supervised-fine-tuning.md), [DPO and IPO](../../p12-post-training/02-direct-preference-optimization.md), [GRPO with verifiable rewards](../../p12-post-training/03-grpo-with-verifiable-rewards.md), and [distillation](../../p12-post-training/04-distillation.md), with the derivations checked in [the optimization problem set on constrained optimization, duality, and DPO](../../../mathematics-for-computing/10-optimization/91-problem-set-b.md). The v1.0.0 core course ends at the capstone of Pass 9; [Pass 11](../course-p11-operate/README.md) releases and operates it.
