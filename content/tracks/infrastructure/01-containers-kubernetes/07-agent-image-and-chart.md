<!-- ol:module dep.07 -->
# Agent image and chart

## Overview

| | |
|---|---|
| **Module** | `dep.07` · practice · Docker and Helm · Pass 10 · 2 to 3 h |
| **You build** | `deploy/docker/agent.Dockerfile` and `deploy/helm/<system>-agent/` |
| **Contract** | [`agent.values.schema.json`](../../../../course/contracts/helm/agent.values.schema.json) |
| **Tests** | `course/tests/dep.07/` (why: image hardening, schema, queue configuration, and secret references) |
| **Needs** | `dep.06` worker chart, `ag.05` durable agent runs |
| **Used by** | `MS-agent` deploys the worker on the `agent` queue |
| **Milestone** | [MS-agent](../../../../course/milestones/MS-agent.toml) |
| **Optional depth** | Separate agent and tool sandbox workloads |

## Key takeaways

- The image runs the learner-built agent worker as a non-root user.
- The chart consumes the durable address and the `agent` queue.
- Provider keys are injected only through Kubernetes Secret references.
- Resource limits and probes make failure visible to the scheduler.

## How to work this chapter

```bash
ol start dep.07
ol tests dep.07
ol check dep.07
```

## 1. Why now

`ag.05` can resume durable agent runs, but a worker process must be packaged and scheduled for the agent queue. A dedicated chart lets it scale and restart independently from gateway and engine services.

## 2. Principles

Use a pinned multi-stage base image, copy only the built worker and runtime files, set a non-root numeric uid, and provide CPU and memory requests and limits. Readiness and liveness probes use the worker health endpoint. Configuration identifies the durable service, queue, provider URL, model, artifacts mount, and optional OTel endpoint. Any API key comes from a Secret key reference.

| Setting | Purpose |
|---|---|
| `durableAddress` | gRPC address for durable workflow service |
| `provider.baseUrl` | learner gateway or explicitly configured frontier endpoint |
| `provider.apiKey` | Secret reference, never a literal |
| `replicaCount` | bounded worker count for this local chart |

## 3. Worked example

The chart points `durableAddress` at `tinyllm-durable:7233`, mounts `/artifacts` read-write for run state, and reads `TL_API_KEY` from Secret `agent-provider`, key `api-key`. A readiness failure removes the pod from service while its durable workflow lease can expire and be reclaimed.

## 4. The artifact and its check

Build the image, inspect its pinned base and numeric runtime user, then render Helm templates using the contract schema. The reference chart at `deploy/helm/forge-agent/` uses `forge-durable:7233`, selects the `agent` queue, and reads `PROVIDER_API_KEY` with `secretKeyRef`; the values file contains only the Secret name and key. `test_agent_image_policy` rejects root execution, mutable base tags, and broad source copies. `test_agent_chart_schema` compares the chart's schema with the published contract. `test_agent_queue_and_secret` checks the queue, Secret reference, probes, resource bounds, and PodDisruptionBudget. A chart must not carry provider credentials in values files or image layers.

## 5. Pitfalls

| Pitfall | Caught by |
|---|---|
| Copying a developer `.env` into the image | `test_agent_image_policy`; mutant `s01` |
| Starting the default queue instead of `agent` | `test_agent_queue_and_secret`; mutant `s02` |
| Omitting resource limits | `test_agent_chart_schema`; mutant `s03` |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `dep.06` | Reuses the worker chart conventions for probes, resources, and secrets. |
| Back | `ag.05` | Packages the durable agent worker and its queue configuration. |
| Forward | `MS-agent` | Runs the agent worker with the durable service and learner gateway. |
| Forward | `ops.10` | Injects a runaway agent and checks its budget and resume behavior. |

## Going further

Production systems isolate untrusted tool execution, use workload identity for secrets, and persist artifacts in a multi-node store. Those changes need explicit threat models and storage recovery tests.
