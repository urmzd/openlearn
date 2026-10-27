# Parallelism, topology, and the global training batch

Distributed training splits different parts of the same computation. Choose the axis from the resource that does not fit, then place its communication on an appropriate interconnect. GPU count alone does not determine either memory savings or effective batch size.

> **Learning objectives**
> - Distinguish data, tensor, pipeline, context, sequence, and expert parallelism.
> - Count independent examples separately from cooperating model ranks.
> - Explain communication/computation overlap and its limits.

## What gets split

| Mechanism | Split and replicated state | Communication and typical constraint |
|---|---|---|
| Data parallelism (DP) | Independent batches, replicated model | Aggregate gradients; batch and network limit scaling |
| ZeRO/FSDP | Optimizer, gradient, and/or parameter state across a DP group | Gather needed parameters and reduce/scatter gradients; peak live buffers still matter |
| Tensor parallelism (TP) | Matrix operations across cooperating ranks | Frequent collectives on the layer's critical path |
| Pipeline parallelism (PP) | Layers across stages | Send activations and gradients between stages; bubbles and imbalance |
| Context parallelism (CP) | One long context across ranks | Exchange attention state or partial results while preserving global attention semantics |
| Sequence parallelism with TP | Some activation regions along the sequence dimension | Collective/layout changes around operations such as normalization |
| Expert parallelism (EP) | Experts across ranks | Dispatch tokens and combine outputs; all-to-all bandwidth and expert skew |

The names can differ across frameworks. Inspect the process groups and tensor layouts. EP and DP can use overlapping rank arrangements, so multiplying every advertised degree is not always a valid world-size equation. [Megatron's distributed training paper](https://arxiv.org/abs/2104.04473) develops combined tensor/pipeline/data layouts; [ZeRO](https://arxiv.org/abs/1910.02054) develops model-state partitioning.

## Work a layout

Suppose 32 GPUs use TP=4, PP=2, and four independent DP replicas, with no additional axes. The product is $4\times2\times4=32$. A micro-batch of two sequences per DP replica and eight accumulation micro-batches gives

$$B_{\mathrm{global}}=2\times8\times4=64\text{ sequences per update}.$$

Multiplying by all 32 ranks would give 512 and count the same examples repeatedly. Sequence length and valid-token masks determine the token batch. If some examples contain only 100 target tokens and others contain 1,000, equal sequence counts do not imply equal token weighting.

Put latency-sensitive collectives inside fast links when possible, but profile the actual topology. PCIe peer links, NVLink, host memory, NIC placement, and cross-node bandwidth determine achievable overlap. TP does not guarantee faster training when communication exceeds the work saved.

## Pipeline bubbles and load balance

For a simplified balanced fill-and-drain pipeline with $p$ stages and $m$ micro-batches, the bubble fraction is approximately

$$\frac{p-1}{m+p-1}.$$

At $p=4,m=16$, it is $3/19\approx15.8\%$. This is a schedule model, not a measured utilization prediction. Interleaving, 1F1B schedules, unequal layers, recomputation, and communication change the result. More micro-batches can reduce bubbles but also increase scheduling overhead, change kernel efficiency, and alter the global batch if accumulation is not adjusted.

An MoE job adds a second imbalance: one expert can receive much more work than another. Log per-expert tokens, dropped/rerouted tokens if applicable, and per-rank idle time. Average FLOPs do not reveal the slowest expert or rank.

## Overlap does not erase communication

An all-gather can run while another layer computes only if the dependencies, stream scheduling, network, and memory headroom permit it. Prefetching more parameters can hide latency while increasing peak memory. Smaller gradient buckets can start communication earlier but incur more launches; large buckets delay overlap. A trace should show whether communication completes before its consumer needs the result.

FSDP memory formulas typically describe persistent state, not the complete peak. Include live gathered parameters, backward buffers, activations, optimizer temporaries, and compiler workspaces. The [PyTorch 2.9 `fully_shard` documentation](https://docs.pytorch.org/docs/2.9/distributed.fsdp.fully_shard.html) explains grouping and resharding controls for one concrete implementation.

## Validate before a long run

Use a tiny deterministic model and fixed global batch. Compare one-rank and distributed loss and gradients with explicit tolerances, then several optimizer updates with matched reductions. Check data sharding, last incomplete batches, masked tokens, and duplicated examples. Test saved-state recovery on the intended topology. Only after correctness passes should throughput and scaling efficiency decide the layout.

Measure useful valid tokens per second, optimizer updates per second, peak memory, communication time, and wall time including checkpoint/evaluation overhead. Define whether MFU includes attention terms and which precision's hardware peak is the denominator. A higher MFU obtained by changing model quality or token accounting is not a clean optimization result.

## Key terms

- **Process group**: ranks cooperating in a particular collective or parallel axis.
- **Pipeline bubble**: time a stage cannot perform useful work because the schedule has not filled or drained.
- **Overlap**: concurrent execution of independent communication and computation.

## Summary

Count independent replicas, shard the resource that limits the run, and verify the resulting computation before interpreting speed. Topology and live temporary state are part of the design.

## Review questions

1. Why is TP excluded from the global-batch multiplier?

   <details><summary>Answer</summary>TP ranks process parts of the same examples rather than independent batches.</details>
2. Can more parameter prefetching both improve throughput and cause an OOM?

   <details><summary>Answer</summary>Yes. It can overlap transfers with computation while increasing the number of live gathered parameters.</details>
