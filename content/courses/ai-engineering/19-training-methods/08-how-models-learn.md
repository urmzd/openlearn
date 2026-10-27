# How models learn: signals, objectives, and the training loop

Training changes a model's parameters using evidence about what it should predict or do. Inference applies those parameters to new inputs. The same architecture can learn in several ways; a learning method describes the source of evidence and the objective, while an optimizer describes how parameters move. Start here before comparing training libraries or engine flags.

> **Learning objectives**
> - Trace data through a forward pass, loss, backward pass, and optimizer update.
> - Distinguish learning signals, adaptation methods, and inference-time computation.
> - Explain what changes between pretraining, supervised fine-tuning, preferences, reinforcement learning, and distillation.

## Different ways to learn

| Learning method | Evidence and objective | Language-model example | Main limitation to measure |
|---|---|---|---|
| Supervised | Labeled input and target; predict the target | Classify a support ticket; imitate a demonstrated assistant answer | Label errors, coverage, and distribution shift |
| Self-supervised | Construct targets from the data itself | Predict the next token or a masked token | Text likelihood does not directly measure usefulness or truth |
| Unsupervised | Discover structure without supplied target labels | Cluster embeddings or learn a lower-dimensional representation | A useful representation needs an external evaluation |
| Preference learning | Comparisons between candidate outputs | DPO on chosen/rejected answers, or a learned reward model | Annotator disagreement and shortcut preferences |
| Reinforcement learning | Outcomes from actions and their consequences | PPO or GRPO using a checker or learned reward | Exploration, credit assignment, reward exploitation, policy drift |
| Distillation | A teacher's answers, probabilities, or intermediate representations | Train a smaller student on synthetic answers or token distributions | Teacher errors, student capacity, and prefix-distribution mismatch |

These overlap. Next-token prediction is a supervised prediction loss with targets obtained by self-supervision. Distillation can use a supervised cross-entropy loss. A preference model can supply rewards to RL. Contrastive learning constructs positive/negative pairs and learns representations by distinguishing them; it can use labels or self-supervised augmentations. The important questions are who supplies the target, what distribution supplies the examples, and what the objective rewards.

**Continual pretraining** continues the self-supervised objective on a new mixture, often to learn domain language. **SFT** imitates desired responses. **Full fine-tuning**, **LoRA**, and **QLoRA** specify which parameters are trainable and how they are represented, and can be used with several objectives. **Curriculum learning** changes the order or difficulty of examples. **Active learning** chooses examples for labeling. Neither is a new optimizer.

Retrieval, prompting, tool use, best-of-N selection, and longer test-time reasoning can improve a system without changing its weights. They consume serving resources and need end-to-end evaluation, but they are not parameter training by themselves.

## One training step, without a framework

1. Sample examples from a versioned training distribution. Tokenize them, form tensors, and construct attention and loss masks.
2. Run the **forward pass**: weights and inputs produce logits, values, embeddings, or another prediction.
3. Compute a scalar **loss** from predictions and learning signals. Choose whether its unit is examples, valid tokens, pairs, or trajectories.
4. Run the **backward pass**: automatic differentiation computes derivatives of that loss with respect to trainable parameters.
5. Accumulate and aggregate gradients over the intended global batch. Unscale mixed-precision gradients when required, check finiteness, and apply any gradient clipping.
6. Apply an **optimizer update**, such as SGD or AdamW, then advance the learning-rate schedule once per update.
7. Periodically evaluate held-out data and save model, optimizer, schedule, random-generator, and data-position state.

For a binary target whose predicted probability is $p=0.25$, cross entropy is $-\log p\approx1.3863$ nats. If an update raises that probability to $0.5$, this example's loss falls to $0.6931$. That is evidence of fitting this target, not evidence of generalization. Evaluation asks whether the improvement transfers to data and outcomes the update did not use.

For language modeling, logits at position $t$ predict token $t+1$. An SFT example can keep prompt tokens in the attention context while setting their loss weight to zero. Padding must affect neither attention nor the loss denominator. A misplaced shift or mask can yield a decreasing loss while teaching the wrong task.

## Batch size, steps, and epochs

With equal micro-batches, global examples per optimizer update are

$$B_{\mathrm{global}}=B_{\mathrm{micro}}\times A\times D,$$

where $A$ is accumulation steps and $D$ is the number of independent data-parallel replicas. Tensor-parallel ranks cooperate on the same examples and do not multiply this batch. One **epoch** visits the finite dataset once; an online RL stream does not have the same fixed-dataset interpretation.

Variable-length examples require token accounting. Suppose two micro-batches contain 2 and 6 valid target tokens, with mean losses 1 and 3. The global token mean is $(2\times1+6\times3)/8=2.5$, whereas averaging the two means gives 2. Each micro-batch should contribute its loss sum divided by the global valid-token count, with the distributed reducer's averaging convention accounted for. Equal accumulation counts alone do not guarantee the same objective.

Changing batch size can change gradient noise and the number of optimizer steps per token. Reusing the same learning rate and reporting only epochs can hide a large change in training. Compare tokens seen, updates, effective batch, schedule, and evaluation. Even mathematically equivalent accumulation can differ numerically because reduction order, dropout, and kernels differ. Inference [batch invariance](../20-inference-engines/13-batch-invariance-and-reproducibility.md) is a separate contract.

## From learning to a service

The lifecycle is a branching set of choices, not a requirement to run every algorithm:

```text
data and evaluation design → pretrain or choose a base model
  → domain adaptation and/or SFT when justified
  → preferences, reward optimization, and/or distillation when justified
  → held-out evaluation → export and numerical checks
  → inference optimization → load test → deployment and monitoring
```

A useful model can need only SFT. RL can use a learned reward or verifiable outcomes. Distillation can compress a model at several stages. Compare the simplest sufficient method against the baseline under the same task and serving budget. See [fine-tune scoping](06-evaluation-and-fine-tune-scope.md) for the decision process and [on-policy and off-policy data](../18-post-training/05-on-policy-and-off-policy.md) for the next distinction.

## Key terms

- **Parameter**: a learned number used in the model's computation.
- **Objective**: the mathematical quantity learning tries to minimize or maximize.
- **Optimizer**: the rule that updates trainable parameters using gradients and state.
- **Generalization**: performance on relevant examples outside those used to fit or select the model.

## Summary

Identify the signal, example distribution, trainable parameters, loss reduction, and update rule separately. Then evaluate the resulting behavior and serving cost on held-out workloads.

## Review questions

1. Is LoRA an alternative to reinforcement learning?

   <details><summary>Answer</summary>LoRA selects a parameterization for updates; RL selects a learning signal and objective. They can be combined.</details>
2. Why can two runs with the same effective example batch optimize different losses?

   <details><summary>Answer</summary>Different lengths, token masks, per-example versus per-token normalization, or distributed averaging can weight the examples differently.</details>

## Sources and checks

The supervised/self-supervised distinction and optimization loop are developed in [Goodfellow, Bengio, and Courville, *Deep Learning*, chapters 5–8](https://www.deeplearningbook.org/). The arithmetic examples are independently executable in [learning checks](code/learning_checks.py). These are teaching examples, not measured training results.
