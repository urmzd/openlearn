# Evaluation and fine-tune scope

This part of [Choosing Training and Adaptation Methods](README.md) develops the mechanisms below. Work the numerical examples or trace the code before checking the answer.

> **Learning objectives**
> - Choose an adaptation method from a measured failure and available data.

## Scoping a fine-tune

Everything so far equips you for the practitioner's core loop: decide, measure, train, measure again. Scoping starts by asking whether the problem needs fine-tuning at all.

### Does this need fine-tuning?

Most requests for fine-tuning are really requests for something else. The table maps symptoms to the first fix to try and to the case where fine-tuning does win:

| Symptom | First fix | Fine-tune when |
|---------|-----------|----------------|
| Model lacks facts or private, changing knowledge | **RAG** ([Retrieval and RAG](../27-retrieval-and-rag/README.md)) | never for fast-changing facts; fine-tuning is poor at reliable knowledge injection |
| Wrong format, style, tone, schema | prompting + few-shot + structured output | prompt is long, brittle, or eats latency/cost: **SFT/LoRA** bakes it in |
| Narrow task, high volume, latency or cost bound | prompt a large model to get a baseline | **SFT or distill** into a small model; often the biggest ROI |
| "Better" is a preference, not a label | prompt iteration | **DPO** on chosen/rejected pairs |
| Correctness is checkable (code runs, answer matches, schema validates) | prompting + verification loop | **RFT / GRPO** with the checker as reward |
| Domain language the base barely knows (new language, proprietary DSL) | none | **continued pretraining** (billions of tokens), then SFT |

### Matching the method to the data

Each method consumes a different shape of data, in different amounts:

| Method | Data shape | Practical minimum | Typical |
|--------|-----------|-------------------|---------|
| SFT (LoRA) | prompt → ideal response | a few hundred high-quality | 1K-50K |
| DPO | prompt, chosen, rejected | ~1K pairs | 5K-100K |
| RFT / GRPO | prompts + grader (no answers needed) | 100s of prompts the model sometimes solves | 1K-20K |
| Distillation | prompts (teacher writes responses) | 1K | 10K-1M |

> **Warning:** GRPO needs a base pass rate strictly between 0 and 1 on the training prompts. If every sample in a group gets the same reward, then $\hat{A}_i = 0$ for all of them and nothing is learned. Prompts the model always solves or never solves are wasted.

### Evaluate first

If you cannot measure base versus tuned on a held-out set the customer agrees with, you are not ready to train. The procedure has five steps. First, build the held-out evaluation set **before training**, with the customer, from real traffic; freeze it and deduplicate it against the training set. Second, define the metric (exact match, pass@k, or a rubric scored by an LLM judge calibrated against human labels) and the threshold for shipping. Third, measure the base model, the base model with the best prompt you can write, and the tuned model on the same set; the prompt baseline is the honest comparison. Fourth, also run a general-capability regression, such as a slice of MMLU-Pro, IFEval, or the customer's other use cases, to catch **catastrophic forgetting**. Fifth, watch train and eval loss per epoch, because overfitting on small SFT sets shows up within 2-3 epochs.

The mitigations for catastrophic forgetting are LoRA instead of full fine-tuning, fewer epochs and a lower learning rate, **replay** (mixing in 5-20% general instruction data), a KL penalty to the base (built into DPO and GRPO), and evaluating the general set at every checkpoint. Evaluation design is covered in depth in [LLM Evaluation](../29-evaluation/README.md).

### Estimating the cost

The cost estimate reuses the pretraining arithmetic with a different constant:

```text
D_train   = examples × avg_tokens_per_example × epochs
FLOPs     = k · N · D_train            k = 6 (full FT), ≈ 4 (LoRA / QLoRA)
GPU-hours = FLOPs / (peak_FLOP/s × MFU) / 3600
$         = GPU-hours × $/GPU-hour  +  data creation  +  eval runs  +  (RL: rollout GPU-hours, often > training)
```

Work an example. Five thousand examples of 800 tokens for 3 epochs is 12M tokens. On an 8B with LoRA that is $4 \times 8\times10^9 \times 1.2\times10^7 = 3.8\times10^{17}$ FLOPs, which at 35% MFU on one H100 is about **0.3 GPU-hours**. The compute is a rounding error; the bill is in the 5,000 examples, the evaluation set, and the iterations. Say this early in every scoping call. Running `python code/memory_calc.py --tokens 12e6` prints the full table.

### A decision table

Putting the pieces together, the common customer situations map to recommendations like this:

| Customer situation | Recommendation |
|--------------------|----------------|
| < 100 examples, no eval set | prompt engineering + build the eval set; no training yet |
| 500-5K labeled examples, format/task adherence | LoRA SFT on an 8B-class instruct model, rank 16-64, all linear layers |
| Wants frontier quality at small-model cost | distill: frontier model generates, filter by grader, SFT a small model |
| Has thumbs up/down or A/B preference logs | SFT first, then DPO (or KTO for unpaired feedback) |
| Has a reliable automatic grader | SFT warm start, then GRPO/RFT |
| Needs > 1 task or per-tenant variants | one base, many LoRA adapters, multi-LoRA serving |
| Regulated, needs weights on-prem | LoRA or full FT on open weights; export merged weights |
| Model must learn a large new corpus | continued pretraining (full FT, multi-node), then SFT; budget weeks |

And the methods themselves, side by side, compare on data, memory, and sampling:

| Method | Data | Models in memory | Online sampling | Typical use |
|--------|------|------------------|-----------------|-------------|
| SFT | prompt → response | 1 | no | format, task, style |
| Reward model | preference pairs | 1 | no | scoring for PPO / rejection sampling |
| PPO (RLHF) | prompts + RM | 4 (policy, ref, RM, critic) | yes | classic RLHF, mostly superseded |
| DPO / KTO / SimPO | pairs / thumbs | 2 (1 for SimPO) | no (online variants: yes) | preference alignment |
| GRPO / RLOO / DAPO | prompts + verifier | 2 (policy, ref) + rollout engine | yes | reasoning, code, agents, RFT |
| Distillation | prompts + teacher | student (+ teacher logits) | optional | cheap small models |
| LoRA / QLoRA / DoRA | any of the above | base frozen + adapter | n/a | memory-bound fine-tuning |

## Summary

Choose an adaptation method from a measured failure and available data. State the assumptions before applying the result to a different workload.

## Review questions

1. What evidence would justify rejecting a successful training run?

   <details><summary>Answer</summary>

   Held-out quality or the serving cost and latency contract can regress even while training loss improves.

   </details>

[Chapter contents](README.md) · [Next: The training step and validation](07-training-step-and-validation.md)
