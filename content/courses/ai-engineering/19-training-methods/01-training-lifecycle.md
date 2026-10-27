# Training lifecycle and learning signals

This part of [Choosing Training and Adaptation Methods](README.md) develops the mechanisms below. Work the numerical examples or trace the code before checking the answer.

> **Learning objectives**
> - Identify the data and objective for each training stage.

## The pipeline and who does what

A modern model is made in a fixed sequence of stages, as written down in the [Llama 3 report](https://arxiv.org/abs/2407.21783) and taught in [CS336](https://stanford-cs336.github.io/):

```text
data → pretraining → mid-training → SFT → preference / RL → evals → release
```

The table below summarizes what each stage does, how big it is, and what learning signal it uses.

| Stage | What happens | Typical scale | Signal |
|-------|-------------|---------------|--------|
| **Data** | Crawl (Common Crawl), text extraction, language ID, **dedup** (exact hash + MinHash LSH near-dup), quality filtering (heuristics + classifier, e.g. FineWeb-Edu), PII and toxicity filters, **decontamination** against evals, tokenizer training (BPE), **mixture** weights across web/code/math/books/multilingual | 10s of trillions of raw tokens down to ~15T kept | none (curation) |
| **Pretraining** | Next-token cross entropy on the mixture, cosine or WSD learning-rate schedule | $10^{23}$ to $10^{25+}$ FLOPs | self-supervised |
| **Mid-training** | Anneal on high-quality data, up-weight code/math, **long-context extension** (raise RoPE base $\theta$, train on long documents, 8K to 128K+) | 1-10% of pretraining tokens | self-supervised |
| **SFT** | Instruction→response demonstrations in a chat template | $10^4$ to $10^6$ examples | supervised |
| **Preference / RL** | Reward model + PPO, or DPO on pairs, or GRPO on verifiable tasks; often several rounds | $10^4$ to $10^6$ prompts | preference / reward |
| **Evals** | Capability (MMLU-Pro, GPQA, SWE-bench, AIME), safety, regression vs previous checkpoint | every checkpoint | measurement |
| **Release** | Quantize, convert, write model card, ship weights and serving config | | |

Two of these rows deserve emphasis. First, **data is the source code**: most of the quality differences between open models trace to filtering and mixtures, not to architecture ([FineWeb](https://arxiv.org/abs/2406.17557), free). Second, **decontamination** is a correctness requirement, not a nicety. A benchmark that leaked into pretraining makes every later evaluation on it meaningless.

Different people own different stages, and it helps to know what each of them will ask you. The research scientist owns objectives, data recipes, ablations, and the post-training algorithm, and asks questions like "What is your KL coefficient and how did you pick it?" The ML engineer owns training and evaluation pipelines, data tooling, checkpoints, and reproducibility: "How do you version data and resume a failed run?" The member of technical staff on performance owns MFU, kernels (FlashAttention, fused optimizers), the parallelism layout, communication overlap, and FP8: "What MFU do you get on H100 for a 70B with TP=8?" And the forward deployed engineer owns a customer's adaptation, choosing among prompting, RAG, LoRA, and full fine-tuning, building their dataset and evaluation, running the job, and getting it served. Their question is the one the last section of this chapter answers: "Will fine-tuning fix this, how much data, what will it cost, and how do we know it worked?"

At a fine-tuning cloud, customers almost always start from an open base or instruct model, so the deployed engineer's pipeline is the last three rows of the table. But you cannot reason about those rows without the arithmetic of the first ones, so that is where we start.

## Summary

Identify the data and objective for each training stage. State the assumptions before applying the result to a different workload.

## Review questions

1. Why must the evaluation split be fixed before choosing a fine-tune?

   <details><summary>Answer</summary>

   Choosing data or hyperparameters using the test split makes that split part of training; reserve it for the final comparison.

   </details>

[Chapter contents](README.md) · [Next: Pretraining compute and scaling](02-pretraining-compute-and-scaling.md)
