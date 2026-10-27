# Batch invariance and numerical reproducibility

An inference service often changes batch composition from one step to the next. A request that produces one answer alone may produce another beside a longer request, even with greedy decoding and the same weights. Understanding this behavior is necessary for reproducible evaluation, caching, regression tests, and RL rollouts.

> **Learning objectives**
> - Distinguish mathematical equivalence, numerical agreement, deterministic execution, and batch invariance.
> - Explain how reduction order, routing, and random-number assignment affect results.
> - Design an invariance test with an explicit acceptance contract.

## Four different contracts

| Contract | Question | Example check |
|---|---|---|
| Mathematical equivalence | Do algorithms represent the same function in exact arithmetic? | Cached attention equals full causal attention with matching positions and masks |
| Numerical agreement | Are finite-precision results within a stated tolerance? | Maximum absolute logit error and per-token log-probability error |
| Determinism | Does repeating the same supported execution produce the same result? | Repeated run with the same hardware, shapes, inputs, and runtime |
| Batch invariance | Does a request's result stay the same when batch size, neighbors, or slot changes? | The same tokenized request alone, permuted, and amid mixed-length requests |

A deterministic kernel for each shape can choose different reduction trees for different batch shapes. It can therefore be deterministic without being batch-invariant. A result can be numerically close but select a different greedy token. Cross-device and cross-release bitwise equality requires a much stronger contract than repeatability on one fixed stack. [PyTorch's reproducibility documentation](https://docs.pytorch.org/docs/2.9/notes/randomness.html) explicitly scopes its guarantees.

## Why a seed and temperature zero are insufficient

Floating-point addition is not associative. In binary64 arithmetic,

```python
a, b, c = 1e16, -1e16, 1.0
assert (a + b) + c == 1.0
assert a + (b + c) == 0.0
```

Tiling, split-K matrix multiplication, attention split counts, and collective reduction trees can regroup sums. Different batch or sequence lengths can select a different kernel. Atomics can add an execution-order effect as well. Casting to a wider format in one stage does not restore precision already lost in another.

For greedy decoding, suppose two leading logits are $1.0000$ and $0.9999$. Perturbations bounded by $10^{-4}$ can reverse their order. If the gap between the best and second-best logits is larger than $2\delta$, a uniform absolute perturbation bound of $\delta$ preserves that step's winner. If the gap is smaller, a flip is possible, not inevitable. Once a token changes, later prefixes differ and small initial numerical changes can become entirely different completions.

The kernel-level analysis in [Thinking Machines Lab's batch-invariance study](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) separates fixed execution from fixed per-request computation across batch shapes. Its implementation and measurements illustrate one supported stack; they are not a universal performance guarantee.

## Audit every source of dependence

| Source | Why neighbors or scheduling may matter | Check or control |
|---|---|---|
| Matrix multiplication and normalization | Shape-dependent tiling or reduction order | Supported kernels with fixed per-request reduction semantics |
| Attention and chunked prefill | Different partitions, padding, positions, or cache paths | Full versus chunked versus cached logit comparison |
| Tensor/expert parallel collectives | Changed reduction order or rank layout | Fix topology; test all supported layouts separately |
| MoE routing | Capacity limits or batch-level balancing can change selected work | Inspect the actual router, overflow/drop policy, and expert outputs |
| Quantization | Dynamic activation scales may depend on a tensor or batch | Record scale granularity and compare alone versus mixed batches |
| Random numbers | A global RNG stream assigns draws by execution order | Request-local streams or counter-based indexing by request, step, and draw |
| Cache reuse | Wrong namespace, stale weights, or position handling | Include model, adapter, tokenizer, position, and relevant request semantics in cache identity |
| Model mode | Dropout or other training-mode behavior remains enabled | Evaluation mode and explicit handling of stateful layers |

Ordinary dense language-model inference should not mathematically attend across requests. If padding or packing lets it do so, that is a correctness bug, not a tolerance decision. Some model components can intentionally depend on batch statistics or capacity; their intended semantics must be documented before asserting invariance.

## Define the test before measuring

Freeze the artifact revision, tokenizer and rendered token IDs, device topology, runtime and kernel versions, precision, sampler, and acceptance criterion. Then run the same probe in these conditions:

1. Alone, repeatedly, with cold and warm caches.
2. At several slots in batches of equal-length and mixed-length neighbors.
3. With different arrival orders and continuously changing neighbors.
4. With chunked prefill on/off and several token budgets.
5. With prefix caching, speculative decoding, quantization, and supported parallel layouts changed one at a time.

First compare teacher-forced logits or token log probabilities on an identical fixed prefix. Then compare greedy token IDs and complete outputs. Record the first differing position, maximum and mean numerical errors, top-two logit margins, and context length. A forced-prefix comparison locates numerical drift; a free-generation comparison measures its behavioral consequence.

For sampling, use request-local random streams when testing exact replay. Equal seeds do not promise equal draws if engines consume RNG differently. An exact speculative algorithm targets the same distribution but need not emit the same sampled sequence under the same seed. If the contract is distributional, compare categorical frequencies on a small enumerable problem with uncertainty bounds and separately evaluate task quality. A finite test cannot prove equality for all outputs.

Report pass/fail against the chosen criterion. “Same text” is insufficient when log probabilities feed an RL learner. “Close logits” is insufficient when the application requires exact greedy replay. Do not call a tolerance pass bitwise invariance.

## Controls are implementation-specific

A runtime may expose deterministic-algorithm selection, batch-invariant kernels, fixed attention splits, fixed graph shapes, deterministic collectives, or disabled autotuning. Each has a support matrix and a performance cost to measure. Some combinations of model, quantization, device, and parallelism remain unsupported.

As a concrete mapping, the [vLLM batch-invariance documentation](https://docs.vllm.ai/en/stable/features/batch_invariance/) describes `VLLM_BATCH_INVARIANT=1`. This is a version-dependent runtime feature, not a portable flag or a guarantee across all models and hardware. Record the installed release and read its support restrictions before testing. The generic requirement remains the same regardless of provider: a request's supported computation must be independent of irrelevant batch context.

SGLang documents `--enable-deterministic-inference` in its [deterministic-inference guide](https://docs.sglang.io/docs/advanced_features/deterministic_inference). Apply the same model/backend/topology qualification and measurements. These rolling-documentation examples are separate from the archived vLLM 0.11.0 option table in the control guide.

## Why training and RL care

In ordinary training, changing the global batch intentionally changes the gradient estimate. Accumulation equivalence requires the same examples, masks, token weighting, optimizer schedule, and stochastic operations; bitwise equality may still fail through reductions. Batch normalization, dropout, or batch-level objectives add further dependencies. That is different from an inference request unintentionally changing because another customer arrived.

In PPO/GRPO, even identical weights can produce different probabilities in the rollout and learner paths if kernels, precision, or batching differ. Record actual behavior log probabilities and compare both paths on fixed tokens. Quantify the mismatch separately from policy-update KL. Batch-invariant inference can remove one source of mismatch; it does not remove stale weights, filtered sampling, or a tokenizer mismatch.

## Key terms

- **Batch invariance**: independence of a request's result from irrelevant co-batched requests and slot placement under a specified execution contract.
- **Reduction tree**: the grouping and order used to combine partial sums.
- **Replay contract**: the inputs, environment, and equality criterion promised for reproducing an execution.

## Summary

Specify the desired equality, isolate numerical differences on fixed prefixes, then test their effect on generation and learning. Treat reproducibility settings as measured controls with scoped guarantees.

## Review questions

1. Every fixed batch repeats exactly, but the probe changes when a neighbor is added. Which contract failed?

   <details><summary>Answer</summary>Batch invariance. Determinism of each fixed execution can still hold.</details>
2. Can a top-two gap of $0.01$ be reversed if every logit changes by at most $0.001$?

   <details><summary>Answer</summary>No at that step: the worst gap reduction is $0.002$, smaller than $0.01$. This does not establish a bound for later steps unless their prefixes and error bounds also match.</details>

## Checks

The float-association and margin examples run in [learning checks](../19-training-methods/code/learning_checks.py). They demonstrate a mechanism; the engine matrix above must be run on the actual serving stack before making an invariance claim.
