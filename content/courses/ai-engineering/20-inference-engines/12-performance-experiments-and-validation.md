# Performance experiments and correctness gates

An optimization claim needs a baseline, a fixed workload, a resource budget, and a behavior contract. A high tokens-per-second number without those conditions cannot establish that a model is cheaper or better for its users.

> **Learning objectives**
> - Separate artifact correctness, numerical behavior, task quality, and system performance.
> - Design controlled experiments for inference controls.
> - Report goodput, errors, and total cost without hiding workload changes.

## Validate four layers

| Layer | Evidence | A passing result does not prove |
|---|---|---|
| Artifact/API | Load success, model identity, templates, response schema, streaming/stop/cancellation tests | Correct logits or useful answers |
| Numerical semantics | Fixed-prefix logits, masks/positions, cache/full equivalence, batch-invariance matrix | Held-out task quality |
| Task behavior | Frozen representative tasks plus regression slices and uncertainty | Latency or stable capacity under load |
| System performance | Open-loop arrivals, successful goodput, tails, resource/cost measurements | Quality outside the tested tasks or reproducibility on another stack |

For precision or quantization changes, set numerical tolerances and task-quality budgets before the comparison. For exact algorithmic transformations, test stronger invariants where feasible: cached versus full attention, grammar validity, rejected-draft rollback, and KV block reclamation. A smoke-test sentence cannot stand in for these checks.

## Freeze a workload manifest

```text
model / adapter / tokenizer / template revisions
device topology, runtime/container and kernel versions, resolved flags
prompt dataset hash, language/task slices, exact input token lengths
requested and actual output length, stop rules, sampler and seeds
shared-prefix distribution, cold/warm cache policy
arrival process and offered rate, concurrency cap, timeout/retry policy
quality suite, numerical criterion, latency SLOs, measurement window
```

If a request produces a different number of output tokens after a change, report that difference. Faster completion can mean earlier EOS rather than a faster decode kernel. Include failed and rejected requests in the offered-load accounting. Record retries separately instead of presenting only successful attempts.

## First isolate, then combine

Measure baseline, one mechanism changed, and combined settings. For prefix reuse, test cold cache, shared exact prefixes, and unique prefixes. For speculative decoding, include the draft's memory, time, and accepted/rejected tokens. For graphs and compilation, show startup/capture time as well as steady state. For parallelism, keep the total hardware budget visible.

Warm up the intended shapes, then use repeated measurement windows. Interleave baseline and candidate runs when temperature, other tenants, or background load may drift. Select the metric before looking at the result and report uncertainty. Avoid selecting the best of many noisy trials and presenting it as a typical result.

## Throughput versus goodput

**Offered rate** is submitted requests per second. **Completed throughput** counts completions. **Goodput** counts successful work satisfying the declared SLO and quality conditions; state whether its unit is requests or tokens. TTFT includes queueing and prompt processing, while ITL reveals stalls during generation. Average TPOT can hide a long pause between two tokens, so keep latency distributions and timeout/rejection rates.

In an illustrative 60-second window, 600 requests arrive, 540 succeed, and 480 of those meet both latency SLOs. Offered rate is 10 requests/s, successful throughput 9/s, and SLO goodput 8/s. If the fleet costs USD 12/hour, cost per SLO-compliant request in that window is $(12/60)/480\approx0.000417$ USD. This deliberately simple calculation excludes other service costs, and its denominator is explicitly compliant requests.

Use **open-loop** arrivals to see overload: the generator submits according to a schedule rather than waiting for each response before sending the next. A closed-loop test is useful for a fixed-user scenario but self-throttles as latency rises. A slow load generator can also hide overload; record its scheduling delay and host utilization.

## Run the dependency-free companion

From the repository root:

```bash
python3 content/courses/ai-engineering/19-training-methods/code/learning_checks.py
python3 content/courses/ai-engineering/20-inference-engines/04-serving-and-load/code/capacity.py --preset llama-3.1-8b
python3 content/courses/ai-engineering/20-inference-engines/04-serving-and-load/code/loadgen.py \
  --mock --rate 8 --num-requests 20 --output-len 8
```

The mock tests workload generation and measurement plumbing; it models neither real transformer kernels nor task quality. The capacity calculator is a model with stated assumptions, not an allocator measurement. Continue with the [local engine lab](14-local-serving-and-validation.md) to run the same process against actual runtimes.

## Decide from the evidence

Reject a configuration if it exceeds a correctness or quality budget even when faster. If two configurations meet the contract, compare goodput per total cost and the operational constraints. Keep the raw per-request records so a later workload change can be distinguished from an engine regression. The [statistical regression gate](../23-model-serving/12-run-comparison-and-regression-gate.md) shows one comparison method; it is not a replacement for a representative workload.

## Key terms

- **Goodput**: successful work meeting declared service criteria per unit time.
- **Coordinated omission**: failure to observe delays that would have occurred had load continued arriving during stalls.
- **Quality budget**: the maximum allowed regression on a defined evaluation metric and slice.

## Summary

Validate behavior first, hold the workload and hardware budget fixed, and report successful compliant work with uncertainty and failure counts.

## Review questions

1. Why can a slower GPU configuration have higher service goodput?

   <details><summary>Answer</summary>Better scheduling or less interference can allow more requests to meet the SLO despite lower peak kernel throughput.</details>
2. Why are mock-server results insufficient for a quantization claim?

   <details><summary>Answer</summary>The mock executes no quantized model kernels and evaluates no model answers. It verifies the measurement path only.</details>
