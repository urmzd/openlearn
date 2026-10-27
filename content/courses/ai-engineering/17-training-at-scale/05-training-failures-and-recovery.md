# Training failures, profiling, and recovery

A training run is a stateful computation over a data stream. Diagnose failures by locating which state or assumption changed: data, arithmetic, optimization, distribution, or infrastructure. Record the first abnormal step rather than only the final failed checkpoint.

> **Learning objectives**
> - Connect training symptoms to discriminating checks.
> - Save enough state to resume the intended optimization.
> - Separate useful throughput from apparent utilization.

## Diagnose the mechanism

| Symptom | Candidate mechanism | First discriminating check |
|---|---|---|
| Nonfinite loss/gradients | FP16 overflow, invalid log/division, bad data, extreme update | Inspect inputs and activations at the first nonfinite operation; compare higher precision on the same batch |
| Loss spikes at regular intervals | Data boundary, schedule reset, resumed optimizer mismatch | Align spikes with shard IDs, LR, gradient norm, and checkpoint events |
| Training improves, validation worsens | Overfitting, leakage in validation setup, shift, forgetting | Evaluate frozen task slices and the previous checkpoint on identical examples |
| No learning | Masked-out targets, frozen intended parameters, zero LR, detached graph | Check valid-token count, nonzero gradients, and parameter deltas on one batch |
| OOM in backward | Saved activations, long outlier, logits, gathered parameters | Profile memory by phase and sequence length; test recomputation or smaller token micro-batches |
| OOM at optimizer step | Moment/state allocation or optimizer temporaries | Observe the first optimizer step separately from forward/backward |
| Low device utilization | Input stalls, tiny kernels, synchronization, rank imbalance | Trace data wait, host launches, collectives, and per-rank step time |
| Distributed hang | Inconsistent collective order, rank failure, uneven data iteration | Compare the last completed operation and input count on every rank |
| Reward rises, task success falls | Reward exploitation or changing denominator | Re-run independent checks and inspect excluded/truncated samples |

Batch-specific reproduction is valuable: preserve example IDs, token IDs, masks, length, precision, and the preceding optimizer state. Replaying only model weights may miss optimizer-driven instability. Avoid treating every spike as a hardware failure or every NaN as a reason to lower learning rate.

## Optimization controls and their tradeoffs

Learning rate, warmup, decay schedule, weight decay, gradient clipping, optimizer moments, effective batch, precision, loss scale, recomputation, and data mixture are separate controls. Change one mechanism at a time on a short reproducible run. BF16's exponent range reduces one overflow risk but does not guarantee stable losses. FP16 loss scaling protects small gradients; gradients should be unscaled before norm clipping and overflowed updates should follow the recipe's skip/schedule convention.

**Catastrophic forgetting** is a regression on previously learned tasks after domain adaptation. Compare a domain suite and a retained general suite. Mixing replay data, reducing update magnitude, and parameter-efficient adaptation are candidate interventions; none guarantees preservation. Distribution shift is broader: even a stable training run can fail when production inputs or tool behavior change.

For data issues, check train/test contamination, near duplicates, bad special tokens, repeated documents, mislabeled answers, and unintended mixture weights. Packed examples need correct boundaries and a declared attention policy. Logs should distinguish observed tokens, valid loss tokens, and examples dropped for length or quality.

## What a checkpoint must contain

| State | Why weights alone are insufficient |
|---|---|
| Model and any adapters | Defines the predictor and trainable parameterization |
| Optimizer moments and master parameters | Determines the next update, especially for Adam and mixed precision |
| LR schedule, update/token counters, loss scaler | Prevents accidental schedule restart or scale reset |
| Data sampler/cursor and mixture state | Prevents silent replay or omission of examples |
| Random-generator state per relevant process/device | Reproduces sampling, dropout, and data ordering within the supported environment |
| Configuration, tokenizer, template, dataset and code revisions | Establishes what the saved state means |
| Distributed shard metadata | Allows the loader to reconstruct state on a supported layout |

Publish a checkpoint as complete only after every required shard is durable and metadata identifies the completed set. A partially written latest directory is not a valid recovery point. Keep a previous known-complete checkpoint until the new one is verified. Asynchronous save requires a consistent snapshot of the intended step, not tensors that continue mutating during serialization.

[PyTorch distributed checkpoint documentation](https://docs.pytorch.org/docs/2.9/distributed.checkpoint.html) describes one sharded-save implementation and its constraints. Resharding support does not automatically guarantee bitwise continuation after changing world size, data order, kernels, or batch grouping.

## Work a recovery experiment

Run 20 updates with a fixed environment and record per-step losses and final state. Repeat from the same initial state, save after update 10, stop, restore, and run updates 11–20. Compare consumed example IDs, schedule values, optimizer state, losses, and parameters under the stated equality criterion. Repeat with a save interrupted before completion: recovery should select the previous complete checkpoint rather than accepting missing shards.

Changing only the topology is a separate experiment. Recompute global batch and token weighting, confirm the loader's supported resharding, and define a numerical rather than bitwise criterion if reduction order changes. A successful inference from the checkpoint proves neither optimizer restoration nor training continuation.

## Profile useful work

Include data loading, evaluation, checkpointing, compilation, and restarts in end-to-end wall time. Report cold-start cost separately from steady state. If a kernel is faster but the input pipeline dominates, end-to-end throughput may barely change. At scale, the slowest rank determines a synchronized step, so collect per-rank distributions rather than one average utilization value.

## Key terms

- **Recovery point**: a complete, durable state from which the declared training process can resume.
- **Gradient norm**: a measure of update magnitude before optimizer-specific scaling.
- **Forgetting**: loss of previously measured capabilities during later adaptation.

## Summary

Reproduce the first failure, inspect the relevant state, and verify continuation with a deliberate restart. Quality, stability, and useful wall-clock throughput are separate measurements.

## Review questions

1. Why might restored weights give the same first loss but different later losses?

   <details><summary>Answer</summary>The optimizer, LR schedule, data cursor, loss scale, or RNG state may not have been restored. Those affect subsequent updates.</details>
2. Is a lower peak allocated-memory measurement enough to promise a larger production batch?

   <details><summary>Answer</summary>No. Include reserved/temporary memory, the longest intended sequences, evaluation and checkpoint phases, and the actual distributed layout.</details>
