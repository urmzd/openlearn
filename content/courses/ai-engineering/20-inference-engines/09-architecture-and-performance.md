# Architecture choices and performance limits

Model architecture determines which work can be skipped, cached, shared, or distributed. The serving engine then chooses how to execute that work. Treat architecture and runtime optimization as separate decisions: enabling a flag cannot turn an arbitrary multi-head checkpoint into a grouped-query or state-space model without a compatible model transformation and quality validation.

> **Learning objectives**
> - Estimate work and state for dense, sparse-expert, attention, and recurrent models.
> - Explain when prefill and decode change bottlenecks.
> - Separate an architecture's theoretical saving from the engine's realized speed.

## Read an architecture as a resource contract

| Architecture choice | Persistent inference state/work | Performance implication |
|---|---|---|
| Dense autoregressive decoder | Most weights participate in each forward; KV grows with retained context | Weight reuse through batching and efficient attention matter |
| Encoder-only | Process input into representations; no autoregressive output loop for embedding tasks | Input batching, padding, and pooling dominate the contract |
| Encoder-decoder | Encoder pass plus decoder self-attention and cross-attention | Account for encoder outputs and cross-attention cache as well as decoder KV |
| MHA | Separate KV heads per query head | Large KV footprint and read traffic |
| MQA/GQA | One or several shared KV heads | Less KV state; quality and kernel compatibility depend on the checkpoint |
| Latent attention | Store a compressed representation instead of conventional full KV | Model-specific cache formula and projection/kernel behavior |
| Sliding-window/local attention | Retain a bounded local history where supported | Bounded state for those layers; full-attention layers still grow |
| Mixture of experts | Route tokens to a subset of expert feed-forward weights | Active compute is smaller than total parameter count; resident weights, routing, and communication still matter |
| Recurrent/state-space | Update a fixed-size per-layer state with new tokens | State can be independent of history length; long-prefix processing and retrieval behavior differ |
| Hybrid | Mix attention and recurrent or local/global layers | Sum the state and compute of each layer type |

[GQA](https://arxiv.org/abs/2305.13245) and [DeepSeek-V2's latent attention and MoE](https://arxiv.org/abs/2405.04434) illustrate distinct ways to reduce serving costs. [Mamba](https://arxiv.org/abs/2312.00752) illustrates recurrent state-space execution. Their reported quality and speed depend on their experiments; reproduce the comparison for the intended model and runtime.

## Work the conventional KV budget

For standard attention with $L$ layers, $H_{KV}$ KV heads, head dimension $d_h$, $s$ stored tokens per sequence, $B$ sequences, and $b$ bytes per element,

$$M_{KV}=2L H_{KV}d_h s B b.$$

The 2 counts keys and values. For $L=32$, $H_{KV}=8$, $d_h=128$, and BF16 ($b=2$), state is 131,072 bytes, or 128 KiB, per token per sequence. At 8,192 tokens that is 1 GiB per sequence. With 32 KV heads it becomes 4 GiB. Padding, paging tails, scale metadata, replication, and workspaces are additional costs.

This formula does not apply unchanged to latent attention, recurrent state, cross-attention, or a hybrid. For local attention use the actually retained window for those layers. Under tensor parallelism, KV heads may be sharded or replicated; dividing by every GPU can undercount memory if there are fewer KV heads than shards.

Paged allocation avoids reserving every sequence's maximum contiguous length, but it still has block tails and metadata. Prefix reuse reduces state and prefill work only for exactly compatible prefixes; it does not reduce the weight storage. See [PagedAttention](https://arxiv.org/abs/2309.06180) and the [cache implementation](../21-inference/03-paged-kv-cache.md).

## Batching changes the roofline

A dense matrix multiply with $N$ weights and $B$ token positions costs approximately $2NB$ FLOPs. Reading its weights once costs approximately $b_wN$ bytes, giving weight-only arithmetic intensity $2B/b_w$. At BF16 this is about $B$ FLOPs/byte. A single decode stream has little weight reuse; many concurrent streams reuse the weights and move closer to a compute limit.

For an illustrative device with 1 PFLOP/s compute and 4 TB/s memory bandwidth, the ridge is 250 FLOPs/byte. Under this simplified model, BF16 linear layers cross it near 250 tokens per step. KV reads, activation traffic, imperfect bandwidth, communication, and kernel overhead can move the actual crossover. Long-context attention can dominate even while the linear layers remain efficient. These are analytical bounds, not benchmark predictions.

Prefill usually offers more token-level parallelism and larger matrix multiplies. Small prefill jobs can still be overhead-bound; large decode batches can be compute-bound; long-context decode can be KV-bandwidth-bound. Profile the phase and shape rather than labeling all inference as bandwidth-bound.

## Sparse parameters require two counts

For MoE, use **total resident parameters** to estimate weight storage and **active parameters per token** to approximate routed compute. An illustrative 100B-total, 10B-active model can require roughly 200 GB of BF16 weight storage while using a much smaller subset for a token. Across a batch, different tokens can activate many experts, so bytes fetched per step need not equal active-per-token weights times one stream. Expert imbalance, small expert matrix multiplies, and all-to-all traffic can dominate.

During training, the optimizer may maintain state for all trainable experts even when a given token activates only a few. The router and auxiliary load-balancing objective also affect learning. During inference, capacity limits and dropping or rerouting policies can affect [batch invariance](13-batch-invariance-and-reproducibility.md). Inspect the implementation rather than assuming all MoE execution has identical semantics.

## From architecture to an optimization experiment

Determine whether weights, KV, activation workspace, communication, or launches dominate at the intended lengths and concurrency. Then choose a supported mechanism: weight/KV compression, prefix reuse, batching, fused kernels, graph replay, parallel placement, or speculative verification. A smaller or distilled architecture is another lever, but it changes the model and must pass the same task evaluation.

Architecture selection also sets training costs and data needs. Compare quality at a total budget, not merely parameter count. A model with lower active FLOPs can still be expensive to train, load, and serve if its state and communication dominate.

## Key terms

- **Active parameters**: parameters participating in the computation for a particular token.
- **Resident parameters**: parameters stored on the deployment's devices or backing memory.
- **Arithmetic intensity**: operations per byte transferred for a defined computation and memory level.

## Summary

Count the actual layer types, retained state, active work, and communication. Then measure how the supported engine executes them at the target batch and context distribution.

## Review questions

1. Does “10B active” establish that a BF16 MoE model fits in 20 GB?

   <details><summary>Answer</summary>No. Weight storage depends on total resident parameters, plus KV, metadata, and workspaces.</details>
2. If GQA divides KV heads by four, what remains unchanged?

   <details><summary>Answer</summary>Other model weights, most feed-forward computation, tokenization, and much of the activation work do not receive the same fourfold reduction.</details>

## Checks

[Learning checks](../19-training-methods/code/learning_checks.py) recomputes the cache sizes and simplified ridge point. Actual speed, memory peaks, and quality require workload-specific measurements.
