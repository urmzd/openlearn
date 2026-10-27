# Performance Testing Engagements

A benchmark run **for a customer** is not the same thing as a benchmark run for yourself. It is evidence in a decision, and the customer will quote it to their own leadership. That makes method more important than the number. A result measured on the wrong length mix, from the wrong region, with prefix-cache hits from repeated prompts, will be wrong in production and will be remembered as the vendor's promise. The engineering is the same as any load test; the field skill is agreeing the method with the customer before running it, and writing down what the number does not cover.

A customer benchmark answers one question: **at their traffic shape, how much load meets their SLO, and at what cost?** Anything else is a demo. This chapter shows how to answer that question honestly. You will agree the question in writing, build the workload from the customer's logs, pick the right load model, keep the runs clean, measure from where the customer's users are, compare fairly against the incumbent provider, and write a report that leads with curves and goodput under the SLO rather than a single headline number.

The load-testing math and tools are developed in [Serving & Load section 8](../../ai-engineering/20-inference-engines/04-serving-and-load/06-load-tests-and-deployment-runbook.md#load-testing-methodology) in *AI Engineering*; this chapter is about running that method with and for a customer. In practice an FDE usually leads, an SE runs pre-sales smoke tests, and the performance team reviews anything that will be quoted in a contract.

> **Learning objectives**
> - Write the benchmark question as one signed-off sentence that names the SLO, the comparison, and what is out of scope.
> - Build a workload spec from the customer's logs and choose a dataset that preserves their length mix and prefix structure.
> - Use closed-loop runs to find saturation and open-loop runs to prove the SLO, and control the common sources of bad data.
> - Measure TTFT from the customer's region and separate network time from engine time.
> - Report latency-throughput curves, goodput, and cost at the operating point, and avoid promises the data does not support.

## Before you begin

This chapter tests the hypothesis you wrote in [Qualification and Sizing](../02-qualification-and-sizing/README.md), so have it to hand, along with sections 3 and 8 of [Serving & Load](../../ai-engineering/20-inference-engines/04-serving-and-load/). The primary references are the [`vllm bench serve` docs](https://docs.vllm.ai/en/stable/cli/bench/serve.html) and the [vLLM benchmark CLI guide](https://docs.vllm.ai/en/latest/benchmarking/cli/) (both free), which cover request rate, burstiness, maximum concurrency, goodput, warmups, and custom datasets. Useful supplements are [AIPerf](https://github.com/ai-dynamo/aiperf), NVIDIA's successor to GenAI-Perf (see the [migration notes](https://docs.nvidia.com/aiperf/getting-started/migrating-from-gen-ai-perf)); [GuideLLM](https://github.com/vllm-project/guidellm); the in-repository [`loadgen.py`](../../ai-engineering/20-inference-engines/04-serving-and-load/code/loadgen.py), which uses only the standard library and is readable in one sitting; and [*Service Level Objectives*](https://sre.google/sre-book/service-level-objectives/) from the SRE book. All are free.

## Agree the question before the run

The first deliverable of a benchmark engagement is a sentence, not a number. Write the question as one sentence the customer signs off on, for example: "At the request mix in last Tuesday's logs, what arrival rate does one replica sustain with TTFT p95 < 500 ms and TPOT p95 < 40 ms, measured from eu-west?"

That sentence names the **SLO** completely: the metric, the percentile, the threshold, and where it is measured. Name the **comparison** up front too, whether it is the incumbent, a previous configuration, or none. Agree what is out of scope (for example quality, cold start, or multi-region failover) so nobody assumes it is covered. And agree who runs the client. If the customer runs it, give them the exact command; if you run it, give them the raw output.

## The workload shape from their logs

The workload comes from **their logs**. Ask for one representative day, or the peak hour, of sanitized request logs with token counts, and build a **workload spec** from it. Synthetic fixed lengths are a last resort, and if you use them the report says so. Each dimension of the spec, what to extract, and what to do when logs are missing are listed here:

| Dimension | Extract from logs | Why it matters | If logs are missing |
|---|---|---|---|
| **Input length** | p50, p95, p99, max of prompt tokens | Prefill cost, TTFT, KV memory | Lognormal around their stated mean, labelled as assumed |
| **Output length** | p50, p95, p99, max of completion tokens | Decode time, concurrency, cost | Same; never fixed at one value without saying so |
| **Arrival pattern** | Requests per second by minute; peak-to-mean ratio; burstiness | Open-loop rate and burstiness setting | Poisson at stated peak, plus a burstier run |
| **Prefix sharing** | Length of the common prefix; fraction of requests that share it; is it byte-identical | Prefix cache hit rate, TTFT, prefill compute | Ask for the system prompt and one full request |
| **Session shape** | Turns per conversation; agent loops with tool calls | Context growth, KV reuse across turns | Single-turn only, stated as a limitation |
| **Features** | Streaming, tool calls, JSON schema, LoRA adapter per request | Parser and constrained-decoding overhead | Test each feature at least once |

With the spec in hand, choose a dataset. Replaying real prompts is best, and the other options trade fidelity for privacy or control:

| Option | When | Tool support |
|---|---|---|
| **Replay real prompts** (sanitized) | Best: true length mix and prefix structure | `vllm bench serve --dataset-name custom --dataset-path sample.jsonl` (JSONL with a `"prompt"` field per line) |
| **Replay a timed trace** | Burstiness matters (morning ramp, batch spikes) | vLLM trace datasets; AIPerf trace benchmarking (Mooncake format and others) |
| **Synthetic from fitted distributions** | Logs cannot leave the customer | `loadgen.py --len-dist lognormal`, `vllm bench serve --dataset-name random` |
| **Shared-prefix synthetic** | Measuring the prefix cache on purpose | `vllm bench serve --dataset-name prefix_repetition`, `loadgen.py --shared-prefix-len` |

To practice, take any public request trace, or invent a 200-row CSV of input and output lengths, and write a workload spec from it.

## The load model: closed loop, open loop, replay

The workload says *what* to send; the load model says *when*. The choice changes what the measurement means.

A **closed loop** keeps $N$ requests in flight and sends a new one whenever one finishes. It measures maximum throughput but hides queueing, because the load backs off when the server slows down. Use it to find **saturation throughput per replica**, with `--max-concurrency` sweeps or AIPerf's `--concurrency`.

An **open loop** sends requests on a schedule (Poisson at a target rate, or burstier) regardless of completions. It shows what users see when traffic does not wait for the server. Use it to **prove the SLO**, with `--request-rate` and `--burstiness`, or `loadgen.py --rate` or `--sweep`. The short version: use closed loop to find saturation and open loop to prove the SLO, because closed-loop latency hides queueing.

**Replay** of a real traffic hour adds realistic burstiness and length mix. Run it last, as confirmation.

Put together, a standard engagement runs four experiments:

| Run | Mode | Load | Purpose |
|---|---|---|---|
| A | Closed loop | Concurrency sweep 1, 8, 32, 128, 256 | Saturation throughput per replica |
| B | Open loop | Poisson at average, peak, and 30% over peak | SLO attainment and goodput |
| C | Open loop | Replay of a real traffic hour | Realistic burstiness and length mix |
| D | Scale event | 1 replica to N under ramping load | Time to absorb the morning ramp (cold start) |

## Run hygiene

Even with the right workload and load model, a handful of mistakes can make the data worthless. Each has a known control:

| Pitfall | What goes wrong | Control |
|---|---|---|
| **No warmup** | First requests pay CUDA graph capture, JIT, allocator growth; inflates p99 | Send warmup requests and exclude them (`--num-warmups`, `loadgen.py --warmup`) |
| **Prefix-cache contamination** | Reusing identical prompts across runs measures cache hits, not prefill | Unique nonce at the start of every prompt, or caching disabled, unless the cache is under test; then test it deliberately and say so |
| **Uncontrolled output length** | One config stops early, another runs long; throughput not comparable | Force length with `ignore_eos` on engines that support it (vLLM, SGLang); otherwise cap `max_tokens` and report the actual output distribution |
| **Client bottleneck** | A single-process Python client saturates before the server | Watch client CPU; use a multiprocess client such as AIPerf at high concurrency |
| **Too short** | Seconds of data, p99 from a handful of samples | Steady state of at least several minutes per point; enough requests for the percentile you report |
| **Single run** | Noise read as signal | Repeat key points; report spread |
| **Undisclosed config** | Nobody can reproduce the number | Record engine, version, flags, model revision, quantization, GPU type and count |

You can see the first two effects for yourself. Run [`loadgen.py`](../../ai-engineering/20-inference-engines/04-serving-and-load/code/loadgen.py) with `--mock --sweep 2,8,32` and find the knee, then rerun it with `--shared-prefix-len 800` and explain the change in TTFT.

## Measuring from the customer's region

TTFT at the client is not TTFT at the server. The number the user experiences is a sum of several pieces, only some of which the engine controls:

```text
client TTFT = connection setup (DNS, TCP, TLS; first request only)
            + network RTT to the endpoint
            + gateway and auth
            + queue wait
            + prefill
            + network time for the first streamed chunk
```

That decomposition drives five practices. Run the client in the **same cloud region** as the customer's application, or from their network if they insist; a benchmark from your laptop on another continent measures the ocean. Measure **round-trip time** separately, with a trivial authenticated request or the time to first byte of a one-token completion, and report it next to TTFT. Reuse connections (keep-alive) as production does, and report whether the first request's handshake is included. Report **client-side and server-side** TTFT (from engine metrics) side by side; the difference is network plus gateway, and it tells the customer whether to fix region placement or the deployment. Finally, check that every proxy on the path streams, because a buffering proxy turns good TTFT into end-to-end latency.

## Apples to apples against the incumbent

Customers usually want to know whether you are "faster than what we have." A comparison is only fair if every control below matches; otherwise it is not a comparison.

| Control | Why | How |
|---|---|---|
| Same prompts | Length mix drives latency | Identical sanitized sample to both endpoints |
| Same output-length control | Closed APIs do not honor `ignore_eos` | Same `max_tokens`; compare per-token latency and report actual output lengths for both |
| Tokenizer difference | The same text is a different number of tokens per model | Compare TTFT and E2E per request, and output speed in characters per second as well as tokens per second |
| Same region and network path | RTT dominates small TTFTs | Clients co-located, same connection reuse |
| Same time window | Shared serverless endpoints vary by hour and day | Interleave runs or run concurrently; repeat at their peak hour |
| Same features | Tool calls and JSON schema add overhead | Enable the same features on both |
| Same load model and rate | Different modes give different curves | Open loop at the same rates |
| Rate limits | Incumbent tier limits can cap the run, not capacity | Record 429s separately from errors; state the account tier |

When a control cannot be matched (the incumbent's region, for example), write it in the report as a limitation next to the result it affects.

## Reporting curves, goodput, and cost

The report is where method becomes evidence. Its centerpiece is **one chart**: achieved throughput (req/s or output tok/s) on the x-axis, TTFT p95/p99 and TPOT p95/p99 on the y-axis, one point per offered rate, the SLO lines drawn, and the operating point marked. The knee is the finding.

Next to the chart goes **goodput**: the requests per second that meet every SLO. Throughput past the knee is not capacity the customer can use. Both `vllm bench serve --goodput ttft:500 tpot:40` and `loadgen.py --slo-ttft-ms 500 --slo-tpot-ms 40` report it.

Then **cost per 1M tokens at the operating point**, not at saturation:

$$
\text{cost per 1M tokens} = \frac{\text{GPU \$/hr} \times \text{GPUs}}{\text{tok/s} \times 3600} \times 10^6
$$

The full cost model is in [Commercials and Security](../06-commercials-and-security/README.md). Finally, keep **the bound and the measurement** apart: quote the measured number, and explain the roofline bound only as a ceiling ([Serving & Load section 8](../../ai-engineering/20-inference-engines/04-serving-and-load/06-load-tests-and-deployment-runbook.md#load-testing-methodology)).

### Tools

Three tools cover the run matrix: the readable standard-library client for open-loop sweeps, `vllm bench serve` for replaying sanitized prompts with goodput, and AIPerf for closed-loop saturation with a multiprocess client.

```bash
# Open-loop sweep with the customer's SLO, readable stdlib client (in repo)
python content/courses/ai-engineering/20-inference-engines/04-serving-and-load/code/loadgen.py \
    --base-url https://<endpoint> --api-key "$KEY" --model <model> --endpoint chat \
    --sweep 2,4,8,12 --num-requests 600 --warmup 10 \
    --input-len 2000 --output-len 300 --len-dist lognormal \
    --slo-ttft-ms 500 --slo-tpot-ms 40 --json-out run.jsonl

# Replay their sanitized prompts at a target rate, with goodput
vllm bench serve --backend openai-chat --endpoint /v1/chat/completions \
    --base-url https://<endpoint> --model <model> \
    --dataset-name custom --dataset-path sample.jsonl \
    --request-rate 8 --burstiness 1.0 --num-warmups 10 \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
    --goodput ttft:500 tpot:40 --save-result

# Closed-loop saturation with a multiprocess client
aiperf profile --model <model> --url https://<endpoint> \
    --endpoint-type chat --streaming --concurrency 128 --request-count 2000
```

> **Warning:** Flags change between releases. Check `--help` for the installed version before quoting a command in a customer document.

### What never to promise

The report is also where engagements go wrong, because a number in a customer document becomes a commitment. Never promise a p99 without load data at their traffic shape, and never quote a vendor benchmark as the customer's number. The full list, with what to say instead:

| Do not promise | Why | Say instead |
|---|---|---|
| A p99 latency guarantee without load data at their traffic shape | Tails come from queueing and length outliers the benchmark may not have seen | "At the measured mix, p99 TTFT was X at Y req/s; we will confirm on replayed traffic" |
| Vendor benchmark numbers (blog speedups) as their result | Vendor-chosen model, lengths, hardware, and load | "Our published result was on workload Z; on yours we measured W" |
| Throughput at saturation as capacity | Past the knee, latency breaks the SLO | Goodput at the SLO |
| Results from serverless as dedicated performance (or the reverse) | Different batching, isolation, and neighbors | Benchmark the tier you will sell |
| Cost from the roofline bound | Assumes full utilization and no SLO | Cost at the measured operating point |
| Numbers on a config you will not ship | Quantization or flags may change quality | Benchmark the config that passed the eval ([POC and Evaluation](../04-poc-and-evaluation/README.md)) |

## The benchmark report template

Everything above lands in one document, and its first paragraph is the only one some readers will see. Write it as if the customer's VP will read only the verdict.

```markdown
# Benchmark report: <customer> / <workload> / <YYYY-MM-DD>

## Verdict (read this first)
<one paragraph: at <mix>, one <config> replica sustains <goodput> req/s within
TTFT p95 < <x> ms and TPOT p95 < <y> ms, measured from <region>. Cost at that
point: $<z> per 1M tokens. Compared with <incumbent>: <result, with caveats>.>

## Question agreed
<the one-sentence question the customer signed off, with date and name>

## Workload
- Source: <logs, date range, sample size, sanitization>
- Input tokens p50/p95/p99/max: <...>; output tokens p50/p95/p99/max: <...>
- Arrival: <Poisson / burstiness / replay>, rates tested: <...>
- Prefix sharing: <length, byte-identical yes/no, cache state during runs>
- Features: <streaming, tools, JSON schema, LoRA>

## Configuration under test
- Endpoint and tier: <serverless / dedicated>, region: <...>
- Model, revision, quantization: <...>
- Engine and version, key flags: <...>
- GPU type, count, parallelism, replicas: <...>

## Method
- Client: <tool and version>, location: <region>, RTT to endpoint: <ms>
- Load model per run: <closed / open / replay>; warmup: <n>; duration per point: <...>
- Output-length control: <ignore_eos / max_tokens>, repeats: <n>

## Results
<chart: throughput vs TTFT and TPOT percentiles, SLO lines, operating point>
| Offered rate | Achieved req/s | Goodput | TTFT p50/p95/p99 | TPOT p50/p95/p99 | Errors | 429s |
- Client-side versus server-side TTFT at the operating point: <...>
- Incumbent comparison (same table, same controls): <...>

## Cost
- At operating point: <formula with inputs> = $<...> per 1M tokens

## Limitations
- <what this does not cover: quality, cold start, other regions, unmatched controls>

## Reproduce
- Commands, dataset hash, raw outputs: <links>
```

## Where this leads

The method behind every run, open versus closed loop, goodput, and warmup, comes from [Serving & Load](../../ai-engineering/20-inference-engines/04-serving-and-load/). Poisson arrivals and tail percentiles are the subject of [Probability & Statistics](../../mathematics/13-probability-statistics/) in *Mathematics*. Why client TTFT differs from server TTFT, including buffering proxies, is explained in [Streaming & SSE](../../systems/04-streaming/) in *Systems Engineering*, and the server-side metrics you report next to client results come from [Observability](../../systems/11-observability/) in *Systems Engineering*.

This work shows up across the industry at an advanced level. Fireworks AI, Together AI, and Baseten run head-to-head benchmarks against a customer's incumbent during evaluation. NVIDIA maintains AIPerf, GenAI-Perf, and the MLPerf Inference methodology, and the vLLM and SGLang maintainers design `bench serve` and `bench_serving`. Any enterprise buying inference will eventually ask for a benchmark report it can audit.

The next chapter, [POC and Evaluation](../04-poc-and-evaluation/README.md), wraps the benchmark into a decision.

### Further reading

- [`vllm bench serve`](https://docs.vllm.ai/en/stable/cli/bench/serve.html) and the [vLLM benchmark CLI guide](https://docs.vllm.ai/en/latest/benchmarking/cli/) (free).
- [AIPerf](https://github.com/ai-dynamo/aiperf) and [GuideLLM](https://github.com/vllm-project/guidellm) (free).
- [*Service Level Objectives*](https://sre.google/sre-book/service-level-objectives/), *Site Reliability Engineering* ch. 4 (free).

## Exercise

The deliverable is a **benchmark plan and mock report for Lexa** (see the [brief](../01-discovery/README.md#exercise)), using your hypothesis from the [Qualification and Sizing exercise](../02-qualification-and-sizing/README.md#exercise):

1. The one-sentence question, with the SLO you would propose given that they "measure about 8 seconds per section" (state which latency you chose and why).
2. The workload spec, filling unknown rows with stated assumptions and the data request that would replace each.
3. The run matrix (A to D) adapted to Lexa's online path, plus how you would test the nightly job (throughput and completion time, not TTFT).
4. The apples-to-apples controls you can and cannot match against their closed frontier API.
5. A filled report template using results from `loadgen.py --mock` as stand-in data, clearly labelled as mock.

### Done when

- The 3,000-token instruction block appears as a deliberate prefix-cache test, separate from runs with caching controlled.
- The client region is in the EU and the RTT is reported.
- No result is stated without its percentile, rate, and region.
- The "what never to promise" table has been checked against your verdict paragraph line by line.

## Key terms

- **Workload spec**: the input and output length distributions, arrival pattern, prefix sharing, session shape, and features extracted from a customer's logs.
- **Closed loop**: a load model that keeps a fixed number of requests in flight; it finds saturation but hides queueing.
- **Open loop**: a load model that sends requests on a schedule regardless of completions; it proves the SLO.
- **Goodput**: requests per second that meet every SLO.
- **Knee**: the point on the latency-throughput curve where latency starts rising steeply; capacity past it is unusable.
- **Operating point**: the load at which the deployment is meant to run, below the knee, where cost is quoted.

## Summary

- A customer benchmark answers one question: at their traffic shape, how much load meets their SLO, and at what cost.
- The workload comes from their logs; synthetic fixed lengths are a last resort, and the report says so.
- Closed loop finds saturation; open loop proves the SLO; replay confirms.
- Measure TTFT from the customer's region and report client-side and server-side numbers separately.
- Comparisons with the incumbent match prompts, output-length control, region, and time window, or they are not comparisons.
- Report the latency-throughput curve and goodput at the SLO; never promise a p99 without load data or quote a vendor benchmark as the customer's number.

## Review questions

1. Why does closed-loop latency hide queueing?
   <details><summary>Answer</summary>A closed loop sends a new request only when one finishes, so when the server slows down the offered load drops with it. Requests never pile up the way real traffic does, so latency looks better than users would see. Open loop sends on a schedule regardless and exposes the queue.</details>
2. Client-side TTFT is 420 ms and server-side TTFT is 150 ms at the operating point. What do you tell the customer?
   <details><summary>Answer</summary>About 270 ms is network and gateway, not the engine. The fix is region placement, connection reuse, or the gateway path (check for a buffering proxy), not the deployment configuration.</details>
3. Why use a unique nonce at the start of every prompt?
   <details><summary>Answer</summary>Reusing identical prompts across runs produces prefix-cache hits, so the run measures cache lookups instead of prefill. The nonce defeats the cache, unless the cache is deliberately under test, in which case the report says so.</details>
4. Why is cost quoted at the operating point rather than at saturation or from the roofline?
   <details><summary>Answer</summary>Past the knee the SLO breaks, so saturation throughput is not usable capacity; the roofline assumes full utilization and no SLO. Only the measured operating point reflects what the customer will actually pay per token.</details>
