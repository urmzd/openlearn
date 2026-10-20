# Field Engineering

This book is about how customer-facing engineers at an inference and fine-tuning cloud turn technical depth into a customer outcome across a whole engagement, from the first discovery call to escalation. It is written for sales engineers (SE), solutions architects (SA), and forward deployed engineers (FDE) at companies shaped like Fireworks AI, Together AI, or Baseten, and for engineers who want to move into those roles. Its position is firm: every call ends in a written artifact someone else can act on, and every chapter's exercise adds one document to the engagement file for Lexa, a mock customer you carry through the whole book.

## What you will be able to do

- Run a discovery conversation and write notes that produce a numeric deployment hypothesis.
- Size a workload from first principles and choose between serverless, dedicated, and batch capacity.
- Design and report a benchmark on the customer's own traffic, against their SLO.
- Write a POC plan whose criteria can fail, and a staged migration with a tested rollback.
- Answer commercial and security questions from evidence, and escalate a problem so the receiving team can act without a follow-up call.

## Ways through this book

Reading every chapter and doing each exercise takes roughly six weeks at six to eight hours a week.

- **The whole engagement, in order** (about six weeks): [Chapter 1: Discovery](01-discovery/README.md) through [Chapter 8: Mock Engagement](08-mock-engagement/README.md), finishing with the [8.8 capstone](08-mock-engagement/08-capstone.md).
- **Starting in a customer-facing role** (about two weeks): [the role landscape](#the-role-landscape) below, then [Chapter 1: Discovery](01-discovery/README.md) and [Chapter 2: Qualification and Sizing](02-qualification-and-sizing/README.md), then [8.1](08-mock-engagement/01-discovery.md) and [8.2](08-mock-engagement/02-qualification-and-sizing.md).
- **Taking a customer from benchmark to production** (about two weeks): [Chapter 3: Performance Testing Engagements](03-performance-testing-engagements/README.md), [Chapter 4: POC and Evaluation](04-poc-and-evaluation/README.md), and [Chapter 5: Migration and Cutover](05-migration-and-cutover/README.md), then [8.3](08-mock-engagement/03-performance-engagement.md), [8.4](08-mock-engagement/04-poc-and-evaluation.md), and [8.5](08-mock-engagement/05-migration-plan.md), reading Chapter 2 first if you have not sized a workload before.
- **Procurement joined, or something is broken** (about one week): [Chapter 6: Commercials and Security](06-commercials-and-security/README.md) and [Chapter 7: Escalation and Handoff](07-escalation-and-handoff/README.md), then [8.6](08-mock-engagement/06-commercials-and-security.md) and [8.7](08-mock-engagement/07-escalation-and-handoff.md).

## Before you start

You should be comfortable reading a latency percentile and a cost table. The technical depth (prefill and decode, the KV cache, load-testing math) lives in [LLM Systems & Inference](../ai-engineering/20-inference-engines/README.md) and its [Serving, Capacity & Load Testing](../ai-engineering/20-inference-engines/04-serving-and-load/README.md) section in *AI Engineering*; when a chapter links there, redo the math yourself, because you will have to defend every number in front of the customer's MLE. Chapter 8 runs against a platform you build in [*AI Engineering*](../ai-engineering/README.md) and draws on [*Software Engineering*](../software-engineering/README.md) and [*AI Engineering*](../ai-engineering/README.md). Model answers for the mock engagement live in [`course/solutions/docs/engagement/`](../../../course/solutions/docs/engagement/discovery.md).

New to these books? [How to read these books](../../references/how-to-read.md) explains how sections, review questions, and solutions work.

## The role landscape

What these companies sell and who owns which part of an engagement. Facts were checked against primary sources on October 6, 2026; product surfaces change monthly, so recheck before a call.

### What the companies sell

All three sell GPU time wrapped in software. A field engineer has to know which wrapper fits a workload and what each costs the customer in control and money.

| Surface | Fireworks AI | Together AI | Baseten |
|---|---|---|---|
| **Serverless per-token API** | Serverless models ([docs](https://docs.fireworks.ai/getting-started/quickstart)) | OpenAI-compatible serverless ([docs](https://docs.together.ai/docs/quickstart)) | Model APIs ([docs](https://docs.baseten.co/inference/model-apis/overview)) |
| **Dedicated deployment** | On-demand GPUs with autoscaling ([docs](https://docs.fireworks.ai/getting-started/ondemand-quickstart)) | Dedicated endpoints ([docs](https://docs.together.ai/docs/dedicated-endpoints)) | Dedicated deployments ([docs](https://docs.baseten.co/development/model/build-your-first-model)) |
| **Fine-tuning** | SFT, DPO, LoRA, full-parameter, RL via SDK ([docs](https://docs.fireworks.ai/fine-tuning/finetuning-intro)) | SFT, LoRA or full, DPO ([docs](https://docs.together.ai/docs/fine-tuning-overview)) | Training Jobs and Loops; methods not confirmed ([docs](https://docs.baseten.co/training/overview)) |
| **Batch / async** | Batch inference ([docs](https://docs.fireworks.ai/guides/batch-inference)) | Batch API ([docs](https://docs.together.ai/docs/batch-inference)) | Async with webhooks, queue up to 72 h ([docs](https://docs.baseten.co/inference/async)) |
| **BYOC / VPC** | Bring Your Own Cluster: private preview, enterprise, inference only ([docs](https://docs.fireworks.ai/ecosystem/integrations/byoc/overview)) | Enterprise Platform in customer VPC or on-prem ([docs](https://docs.together.ai/docs/together-deployments)) | Single-tenant VPC, self-hosted, hybrid ([docs](https://docs.baseten.co/hosting-options/overview)) |
| **Engine** | FireAttention kernels ([V4](https://fireworks.ai/blog/fireattention-v4-fp4-b200)) | Together Kernel Collection, ATLAS speculator ([ATLAS](https://www.together.ai/blog/adaptive-learning-speculator-system-atlas)) | Baseten Inference Stack on TensorRT-LLM ([guide](https://www.baseten.co/resources/guide/the-baseten-inference-stack/)) |

Vendor speedup numbers are the vendor's own benchmarks on the vendor's chosen workload; treat them as hypotheses to reproduce on the customer's traffic ([Chapter 3](03-performance-testing-engagements/README.md)). Compliance claims are contractual: get the current report from the trust center, not from a blog ([Chapter 6](06-commercials-and-security/README.md)).

### Who owns what

| Role | Owns | Does not own |
|---|---|---|
| **Sales engineer (SE)** | Pre-sales technical fit for one deal: demos, discovery, first sizing, questionnaire answers | Post-sale delivery, engine internals |
| **Solutions architect (SA)** | Reference architectures and integration patterns across accounts | Deep per-customer build |
| **Forward deployed engineer (FDE)** | The customer's technical outcome: hypothesis, POC, migration, first-line diagnosis, internal tickets | Engine internals, model research, pricing authority |
| **MLE** | Fine-tuning runs, eval pipelines, model packaging, quality debugging | Contract, account relationship |
| **Performance engineer** | Engine, kernels, scheduler, quantization, per-model tuning | Customer relationship, POC scope |
| **Research scientist** | New methods: speculators, quantization schemes, post-training recipes | Production SLOs |
| **Account executive (AE)** | Commercial relationship, pricing, contract, renewal | Technical claims |

Titles overlap between companies: at some the FDE, SE, and SA are one person. Ask in the interview which rows the role covers.
