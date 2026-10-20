# Pass 9: Capstone Training

Every model so far has been small enough to train in minutes. The capstone is a Llama-family model of about ten million parameters trained on TinyStories, the first model in the course worth serving. Training it on a laptop takes the techniques that make training cheaper without changing what it learns: mixed precision, gradient accumulation, and activation checkpointing, which together match fp32 within 2% in less memory.

The capstone is then owned end to end: trained as a durable TrainRun, evaluated with seeded quality and safety evals, documented with a datasheet and a model card, and shipped through the ModelRelease workflow with gates and a canary before the gateway serves it.

> **Learning objectives**
> - Explain how Hessian-vector products and the recompute-versus-memory schedule underlie activation checkpointing.
> - Explain why mixed precision, accumulation, and recompute together train like fp32.
> - Explain what a datasheet, a model card, and bias and safety evals must contain.
> - Explain how ModelRelease gates and canaries a new model before it replaces a working one.

## Before you begin

This pass needs [Pass 8](../course-p08-durable/README.md), whose TrainRun workflow trains the capstone and whose durable services ModelRelease runs on.

## In this pass

1. [Hessian-vector products and the recompute-versus-memory schedule](../../../mathematics/14-matrix-calculus-and-autodiff/04-hessian-vector-products-and-recompute.md). Afterwards you can explain how a Hessian-vector product is computed without the Hessian and how the recompute-versus-memory schedule is chosen.
2. [Mixed precision, loss scaling, gradient accumulation, and activation checkpointing](../../17-training-at-scale/01-mixed-precision-accumulation-and-checkpointing.md). Afterwards you can explain how mixed precision with loss scaling, gradient accumulation, and activation checkpointing train like fp32 in less memory.
3. [Datasheet and model card](../../35-model-and-data-cards/01-datasheet-and-model-card.md). Afterwards you can explain what the datasheet and model card report and why each section is there.
4. [Bias and safety evals](../../36-bias-and-safety-evals/01-bias-and-safety-evals.md). Afterwards you can explain what the bias and safety evals measure and how their results are reported.
5. [ModelRelease: gates, canary, burn check, and rollback](../../../systems/07-durable-orchestration/12-model-release.md). Afterwards you can explain how ModelRelease gates a model, runs a canary, checks the PromQL burn rate, and rolls back.
6. [Model evals as tests](../../../software-engineering/03-testing/08-model-evals-as-tests.md). Afterwards you can explain how model evals are written as tests with thresholds and confidence intervals.
7. [Capstone: TinyStories, owned end to end](../../capstones/01-tinystories.md). Afterwards you can explain how the TinyStories model is trained, evaluated, released, and served end to end.
8. [What the System Does After Pass 9](milestone.md). Afterwards you can explain how the TinyStories model is trained with mixed precision, evaluated, released through ModelRelease, and served.

## Going deeper

Two optional sections explain what changes when training moves from one machine to many. The capstone does not need them:

9. [Collectives over processes: ring all-reduce](../../17-training-at-scale/02-collectives-over-processes.md). Afterwards you can explain how ring all-reduce over processes sums gradients with bandwidth-optimal traffic.
10. [Data parallelism and ZeRO stages 1 to 3](../../17-training-at-scale/03-ddp-and-zero.md). Afterwards you can explain how DDP and ZeRO stages 1 to 3 partition optimizer state, gradients, and parameters.

The stages are also in this pass's manifest, [`path.tsv`](path.tsv). [Pass 10](../course-p10-agents/README.md) puts an agent on top of the served system.
