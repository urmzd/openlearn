# Capstones

Every earlier chapter of this book builds one piece of the system and tests it on its own. The capstones are where the pieces have to work together, for hours, on one goal. The first capstone trains a roughly 10M-parameter Llama-family model on TinyStories end to end: your corpus pipeline cleans the data as a durable workflow, your tokenizer is trained and ablated (BPE against Unigram), your trainer runs with mixed precision and resumes bitwise, the model is evaluated against the model zoo, released through your release workflow with a model card and data ledger, and served by your engine behind your gateway. The second capstone, which is optional, post-trains that model into an instruction follower.

Both capstones share one idea: the report is the evidence. Nothing is taken on trust. Parameter counts follow from configs, verdicts follow from confidence intervals, decisions are recorded with the measurements that justify them, and a model ships only when the system's own gates let it through.

> **Learning objectives**
> - Run the whole system on one training goal and size that run from a compute budget.
> - Design ablations as paired experiments at an equal budget and read their verdicts from bootstrap confidence intervals.
> - Release a model through a gated workflow with a model card, data ledger, and canary burn check.
> - Evaluate a post-trained model against its base on a fixed held-out set, with claims that match the evidence.

## Three ideas that run through both

The first idea is **one config, two sizes**. The full run is about 10.4M parameters and 100M tokens, an estimated 4 to 12 hours on an M-series laptop. A `short` config of about 2.5M parameters trains in under an hour and is accepted at a looser threshold, and a smoke tier proves the pipeline in minutes.

The second is that **ablations are experiments**. The tokenizer (BPE against Unigram), the attention (MLA against GQA at equal KV bytes), and the MLP (MoE against dense at equal active parameters) are each compared on paired held-out documents, reported with a paired confidence interval, and recorded in an ADR.

The third is that **release is gated**. The model card, the data ledger, and the eval suite must pass before the engine serves the model.

## Before you begin

The capstones assume every chapter before them. The first exercises the [corpus pipeline](../../data-engineering/05-corpus-pipeline/09-corpusbuild-as-a-durable-workflow.md) from *Data Engineering*, the [tokenizers](../07-tokenizers/README.md), the [modern decoder block](../13-decoder-block/README.md), the optimizers from *Mathematics*, [mixed precision, accumulation, and checkpointing](../17-training-at-scale/01-mixed-precision-accumulation-and-checkpointing.md), the durable workflows of *AI Engineering* (in particular [deterministic replay](../../systems/07-durable-orchestration/06-deterministic-replay.md), [subprocess activities](../../systems/07-durable-orchestration/09-subprocess-activities.md), [`TrainRun` and `EvalSuite`](../../systems/07-durable-orchestration/11-trainrun-and-evalsuite.md), and [`ModelRelease`](../../systems/07-durable-orchestration/12-model-release.md)), the [inference](../21-inference/README.md) stack, and the [serving](../23-model-serving/README.md) platform. The second needs the [post-training](../18-post-training/README.md) part. On the course path, the first capstone closes the [capstone training stage](../paths/course-p09-capstone-training/milestone.md), and the optional second belongs to the [agents stage](../paths/course-p10-agents/milestone.md).

## In this chapter

1. [Capstone: TinyStories, owned end to end](01-tinystories.md). Runs the whole system on one goal: sizing the run, three paired ablations, a scaling fit, the zoo table, ADRs, and a gated release, with every number in the reference report recomputed by hand.
2. [Capstone 2: post-trained TinyStories model](02-post-trained.md). Optional. Post-trains the released model with SFT and then DPO or GRPO, and shows how a paired bootstrap and permutation test decide whether the held-out gain is real.

## Further reading

- Eldan and Li, [*TinyStories*](https://arxiv.org/abs/2305.07759).
- Hoffmann et al., [*Training Compute-Optimal Large Language Models*](https://arxiv.org/abs/2203.15556), the source of the scaling-law fit.
