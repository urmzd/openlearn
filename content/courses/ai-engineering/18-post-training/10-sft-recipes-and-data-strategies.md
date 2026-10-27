# SFT recipes, data strategies, and parameter updates

Supervised fine-tuning learns from target outputs. Its variants differ in which examples supply those targets, which tokens receive loss, and which parameters can change. Keep those three decisions separate before comparing framework recipes. The [SFT implementation](01-supervised-fine-tuning.md) supplies chat rendering, assistant masks, and document-aware packing.

> **Learning objectives**
> - Separate objective, data, parameterization, and execution choices in SFT.
> - Compare completion-only, assistant-only, multi-turn, and weighted training.
> - Validate synthetic and filtered demonstrations without leaking evaluation data.

## What changes between recipes?

| Recipe | Targets and loss | Useful when | Main check |
|---|---|---|---|
| Completion-only SFT | Score only the target completion after its prompt | A task has a clear input/output boundary | The first response token is included after the causal shift |
| Assistant-only chat SFT | Score assistant turns; keep other roles as context | Teach multi-turn assistant behavior | Template-specific assistant spans, end markers, and tool-call tokens |
| Final-turn-only SFT | Score only a selected final assistant turn | Earlier turns provide context rather than demonstrations | Earlier assistant messages are masked intentionally |
| All-token causal training | Score eligible input and output tokens | Reproduce a complete conversation/document distribution | User, system, and tool text now receive training weight too |
| Multi-task or domain-mixture SFT | Mix task demonstrations with chosen sampling weights | Adapt several behaviors in one model | Actual valid-token mixture, task interference, and retained capabilities |
| Weighted SFT | Assign unequal nonnegative weights to selected targets/examples | Weight known quality levels or task priorities | Weight provenance, normalization, and concentration |
| Dynamic fine-tuning (DFT) | Modify token update weighting using the model's token probabilities | Test an objective-derived alternative to ordinary SFT | Exact gradient/stop-gradient convention, weighting, and held-out outcomes |
| Synthetic-demonstration SFT | Use teacher-generated text as targets | Human demonstrations are scarce | Independent correctness, diversity, provenance, teacher mistakes |
| Rejection-sampling SFT | Generate candidates and retain those passing a criterion | A checker can identify useful demonstrations | Sampling cost, selection bias, and rejected/retained counts |

Instruction tuning is SFT on instruction/response demonstrations; it is not a separate optimizer. Domain-adaptive continued pretraining usually retains a language-modeling objective on domain documents, while domain SFT uses task targets. A recipe can combine them, but the data and objective should remain identifiable.

[Dynamic fine-tuning](https://arxiv.org/abs/2508.05629) proposes probability-dependent rescaling of token updates. The versioned TRL SFT reference exposes it through `loss_type="dft"`. This changes the learning objective/gradient convention; it is not an execution flag that promises equivalent SFT behavior or universally better generalization.

SFT can also imitate tool use and reasoning demonstrations. Decide which tool-call arguments, tool observations, reasoning spans, and final answers should be targets. If an inference system hides or reformats a span, verify the actual serialized training/serving contract. A model cannot learn a reliable tool interface from a mask that accidentally removes every tool-call token.

## Masks and weights define the objective

Let $x_j$ be token $j$, $m_j$ select target tokens, and $a_j\ge0$ be their weights. For one sequence, a token-weighted causal loss is

$$
L=-\frac{\sum_{j=1}^{T-1}m_j a_j\log\pi_\theta(x_j\mid x_{<j})}{\sum_{j=1}^{T-1}m_j a_j}.
$$

The logits at input position $j-1$ score target position $j$. Require a positive denominator or skip/report the example according to a declared policy. A numerical denominator clamp does not turn an example with no targets into useful training evidence. Padding and packed-document boundaries require their own masks.

For two target losses 1 and 3 with weights 1 and 3, the weighted mean is $(1+9)/4=2.5$, not the unweighted mean 2. Weighting a demonstration more heavily does not change whether it is correct. Across unequal micro-batches or data-parallel ranks, aggregate loss sums and denominators with the reducer's averaging convention accounted for.

## Which parameters change?

| Parameter strategy | Trainable state | Relationship to the objective |
|---|---|---|
| Full fine-tuning | All selected base-model parameters | Can use any of the SFT masks/data recipes above |
| LoRA | Low-rank updates on selected matrices | Changes the parameterization, not the learning signal |
| QLoRA | Adapters with a quantized frozen base | Changes base storage and execution as well as adapter training |
| Other selective updates | Heads, embeddings, biases, selected layers, or other adapters | Capacity, memory, and forgetting must be measured |

See [parameter-efficient fine-tuning](../19-training-methods/05-parameter-efficient-fine-tuning.md) for LoRA/QLoRA/DoRA mechanics. A preference or RL objective can also train adapters. “SFT versus LoRA” therefore compares different axes of a recipe. Combining several adapters or merging one into base weights requires the same post-export checks as a full fine-tune.

## Build and validate a dataset

Split prompts by source, task, customer, or time when those boundaries matter. Deduplicate related conversations before splitting. If a teacher generates or a judge filters training targets, keep the held-out answers out of both generation instructions and filtering decisions. Log how many examples each stage drops and why.

Before a long run, render a few examples and inspect token IDs, target masks, valid-token counts, and document IDs. Include a one-token answer, an empty answer, Unicode, several assistant turns, and a tool interaction. Check that an intentionally selected target changes loss and gradient while a masked target does not. Packing or changing padding should preserve the declared example boundaries and weighting.

Track training loss, held-out task outcomes, length, stop behavior, format/tool success, and regressions on retained tasks. Compare a new SFT recipe with the same base model and evaluation workload. Lower imitation loss can mean better copying without better real task performance.

## Key terms

- **Completion-only loss**: loss restricted to a designated target completion after context.
- **Final-turn-only loss**: loss restricted to a selected final response within a conversation.
- **Rejection-sampling SFT**: supervised training on generated examples retained by an explicit selection rule.

## Summary

Specify target source, token weighting, trainable parameters, and execution separately. Then validate the complete recipe after export to the serving runtime.

## Review questions

1. Does assistant-only SFT necessarily train every assistant turn?

   <details><summary>Answer</summary>No. A recipe may select all assistant turns or only particular turns; its mask must make that choice explicit.</details>
2. Can rejection sampling replace independent evaluation?

   <details><summary>Answer</summary>No. It selects data using a criterion that can be incomplete or exploitable. Evaluate on held-out outcomes outside that selection rule.</details>

## Sources and checks

[TRL 0.23.1 SFT documentation](https://huggingface.co/docs/trl/v0.23.1/sft_trainer) describes completion/assistant masks and packing controls for one implementation. [InstructGPT](https://arxiv.org/abs/2203.02155) documents supervised demonstrations within an RLHF pipeline. [Distillation data](09-distillation-data-and-deployment.md) and [reward design](07-rewards-and-preference-models.md) explain synthetic targets and selection. The weighted-loss example is checked in [learning checks](../19-training-methods/code/learning_checks.py).
