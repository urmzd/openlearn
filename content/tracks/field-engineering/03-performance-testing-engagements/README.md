# Performance Testing Engagements

How to run a benchmark **for a customer**: agree the workload shape from their logs, pick the right load model, measure from where their users are, compare fairly against their incumbent provider, and write a results document that reports curves and goodput under their SLO instead of a single headline number.

> Parent track: [Field Engineering](../). The load-testing math and tools are in [Serving & Load section 8](../../ml/04-llm-systems/serving-and-load/#8-load-testing-methodology); this topic is about running that method with and for a customer. Next: [POC and Evaluation](../04-poc-and-evaluation/) wraps the benchmark into a decision.

## Overview

- **Primary reference**: [`vllm bench serve` docs](https://docs.vllm.ai/en/stable/cli/bench/serve.html) (free) and [vLLM benchmark CLI guide](https://docs.vllm.ai/en/latest/benchmarking/cli/) (free): request rate, burstiness, max concurrency, goodput, warmups, custom datasets
- **Supplementary**: [AIPerf](https://github.com/ai-dynamo/aiperf) (free, NVIDIA's successor to GenAI-Perf, [migration notes](https://docs.nvidia.com/aiperf/getting-started/migrating-from-gen-ai-perf)); [GuideLLM](https://github.com/vllm-project/guidellm) (free); the in-repo [`loadgen.py`](../../ml/04-llm-systems/serving-and-load/code/loadgen.py) (stdlib only, readable in one sitting); [*Service Level Objectives*](https://sre.google/sre-book/service-level-objectives/) (free)
- **Prerequisites**: [Qualification and Sizing](../02-qualification-and-sizing/) (the hypothesis you are testing); [Serving & Load](../../ml/04-llm-systems/serving-and-load/) sections 3 and 8
- **Estimated time**: 4-5 days at 6-8 hrs/week
- **Usually led by**: FDE; SE for pre-sales smoke tests; performance team reviews anything that will be quoted in a contract

## Key Takeaways

- A customer benchmark answers one question: **at their traffic shape, how much load meets their SLO, and at what cost?** Anything else is a demo.
- The workload comes from **their logs**: input and output length distributions, arrival pattern, prefix sharing. Synthetic fixed lengths are a last resort, and the report says so.
- Use **closed loop** to find saturation and **open loop** to prove the SLO. Closed-loop latency hides queueing.
- Measure TTFT **from the customer's region**, and report client-side and server-side numbers separately so network round trips are visible, not blamed on the engine.
- Comparisons with the incumbent use the same prompts, the same output-length control, the same region, and the same time window, or they are not comparisons.
- Report the **latency-throughput curve** and **goodput** at the SLO. Never promise a p99 without load data, and never quote a vendor benchmark as the customer's number.

## How to Study

- Run [`loadgen.py`](../../ml/04-llm-systems/serving-and-load/code/loadgen.py) `--mock --sweep 2,8,32` and find the knee. Then rerun with `--shared-prefix-len 800` and explain the change in TTFT.
- Take any public request trace (or invent a 200-row CSV of input and output lengths) and write the workload spec in section 2 from it.
- Write the [report template](#8-benchmark-report-template) for the mock run as if a customer's VP will read only the first paragraph.

---

# Concepts & Techniques

## Core Insight

A benchmark run for a customer is evidence in a decision, and the customer will quote it to their own leadership. That makes method more important than the number: a result measured on the wrong length mix, from the wrong region, with prefix-cache hits from repeated prompts, will be wrong in production and will be remembered as the vendor's promise. The engineering is the same as any load test; the field skill is agreeing the method with the customer before running it, and writing down what the number does not cover.

## 1. Agree the Question Before the Run

**Key ideas**:
- Write the question as one sentence the customer signs off on: "At the request mix in last Tuesday's logs, what arrival rate does one replica sustain with TTFT p95 < 500 ms and TPOT p95 < 40 ms, measured from eu-west?"
- Name the **SLO** (metric, percentile, threshold, where measured) and the **comparison** (incumbent, previous config, or none) up front.
- Agree what is out of scope (for example: quality, cold start, multi-region failover) so it is not assumed to be covered.
- Agree who runs the client. If the customer runs it, give them the exact command; if you run it, give them the raw output.

## 2. Workload Shape from Their Logs

Ask for one representative day (or the peak hour) of request logs with token counts, sanitized. From it, build a **workload spec**:

| Dimension | Extract from logs | Why it matters | If logs are missing |
|---|---|---|---|
| **Input length** | p50, p95, p99, max of prompt tokens | Prefill cost, TTFT, KV memory | Lognormal around their stated mean, labelled as assumed |
| **Output length** | p50, p95, p99, max of completion tokens | Decode time, concurrency, cost | Same; never fixed at one value without saying so |
| **Arrival pattern** | Requests per second by minute; peak-to-mean ratio; burstiness | Open-loop rate and burstiness setting | Poisson at stated peak, plus a burstier run |
| **Prefix sharing** | Length of the common prefix; fraction of requests that share it; is it byte-identical | Prefix cache hit rate, TTFT, prefill compute | Ask for the system prompt and one full request |
| **Session shape** | Turns per conversation; agent loops with tool calls | Context growth, KV reuse across turns | Single-turn only, stated as a limitation |
| **Features** | Streaming, tool calls, JSON schema, LoRA adapter per request | Parser and constrained-decoding overhead | Test each feature at least once |

Then choose the dataset:

| Option | When | Tool support |
|---|---|---|
| **Replay real prompts** (sanitized) | Best: true length mix and prefix structure | `vllm bench serve --dataset-name custom --dataset-path sample.jsonl` (JSONL with a `"prompt"` field per line) |
| **Replay a timed trace** | Burstiness matters (morning ramp, batch spikes) | vLLM trace datasets; AIPerf trace benchmarking (Mooncake format and others) |
| **Synthetic from fitted distributions** | Logs cannot leave the customer | `loadgen.py --len-dist lognormal`, `vllm bench serve --dataset-name random` |
| **Shared-prefix synthetic** | Measuring the prefix cache on purpose | `vllm bench serve --dataset-name prefix_repetition`, `loadgen.py --shared-prefix-len` |

## 3. Load Model: Closed Loop, Open Loop, Replay

**Key ideas**:
- **Closed loop** keeps N requests in flight and sends a new one when one finishes. It measures maximum throughput but hides queueing, because the load backs off when the server slows. Use it to find **saturation throughput per replica** (`--max-concurrency` sweeps, AIPerf `--concurrency`).
- **Open loop** sends requests on a schedule (Poisson at a target rate, or burstier) regardless of completions. It shows what users see when traffic does not wait. Use it to **prove the SLO** (`--request-rate`, `--burstiness`, `loadgen.py --rate` or `--sweep`).
- **Replay** of a real traffic hour adds realistic burstiness and length mix; run it last, as confirmation.

A standard run matrix:

| Run | Mode | Load | Purpose |
|---|---|---|---|
| A | Closed loop | Concurrency sweep 1, 8, 32, 128, 256 | Saturation throughput per replica |
| B | Open loop | Poisson at average, peak, and 30% over peak | SLO attainment and goodput |
| C | Open loop | Replay of a real traffic hour | Realistic burstiness and length mix |
| D | Scale event | 1 replica to N under ramping load | Time to absorb the morning ramp (cold start) |

## 4. Run Hygiene

| Pitfall | What goes wrong | Control |
|---|---|---|
| **No warmup** | First requests pay CUDA graph capture, JIT, allocator growth; inflates p99 | Send warmup requests and exclude them (`--num-warmups`, `loadgen.py --warmup`) |
| **Prefix-cache contamination** | Reusing identical prompts across runs measures cache hits, not prefill | Unique nonce at the start of every prompt, or caching disabled, unless the cache is under test; then test it deliberately and say so |
| **Uncontrolled output length** | One config stops early, another runs long; throughput not comparable | Force length with `ignore_eos` on engines that support it (vLLM, SGLang); otherwise cap `max_tokens` and report the actual output distribution |
| **Client bottleneck** | A single-process Python client saturates before the server | Watch client CPU; use a multiprocess client such as AIPerf at high concurrency |
| **Too short** | Seconds of data, p99 from a handful of samples | Steady state of at least several minutes per point; enough requests for the percentile you report |
| **Single run** | Noise read as signal | Repeat key points; report spread |
| **Undisclosed config** | Nobody can reproduce the number | Record engine, version, flags, model revision, quantization, GPU type and count |

## 5. Measuring from the Customer's Region

TTFT at the client is not TTFT at the server:

```text
client TTFT = connection setup (DNS, TCP, TLS; first request only)
            + network RTT to the endpoint
            + gateway and auth
            + queue wait
            + prefill
            + network time for the first streamed chunk
```

**Key ideas**:
- Run the client in the **same cloud region** as the customer's application, or from their network if they insist. A benchmark from your laptop in another continent measures the ocean.
- Measure **RTT** separately (a trivial authenticated request, or the time to first byte of a 1-token completion) and report it next to TTFT.
- Reuse connections (keep-alive) as production does; report whether the first request's handshake is included.
- Report **client-side and server-side** TTFT (engine metrics) side by side. The difference is network plus gateway; it tells the customer whether to fix region placement or the deployment.
- Check that proxies on the path stream; a buffering proxy turns good TTFT into E2E latency.

## 6. Apples-to-Apples Against the Incumbent

Customers usually want "faster than what we have". A comparison is only fair if every row below matches:

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

When a control cannot be matched (for example, the incumbent's region), write it in the report as a limitation next to the result it affects.

## 7. Reporting: Curves, Goodput, Cost

**Key ideas**:
- **One chart**: x = achieved throughput (req/s or output tok/s), y = TTFT p95/p99 and TPOT p95/p99, one point per offered rate, SLO lines drawn, operating point marked. The knee is the finding.
- **Goodput**: requests per second that meet every SLO. Throughput past the knee is not capacity the customer can use. `vllm bench serve --goodput ttft:500 tpot:40` and `loadgen.py --slo-ttft-ms 500 --slo-tpot-ms 40` both report it.
- **Cost per 1M tokens at the operating point**: `(GPU $/hr x GPUs) / (tok/s x 3600) x 1e6` ([06](../06-commercials-and-security/)), not at saturation.
- **The bound versus the measurement**: quote the measured number; explain the roofline bound as a ceiling ([Serving & Load section 8](../../ml/04-llm-systems/serving-and-load/#8-load-testing-methodology)).

### Tools

```bash
# Open-loop sweep with the customer's SLO, readable stdlib client (in repo)
python content/tracks/ml/04-llm-systems/serving-and-load/code/loadgen.py \
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

Flags change between releases; check `--help` for the installed version before quoting a command in a customer document.

### What never to promise

| Do not promise | Why | Say instead |
|---|---|---|
| A p99 latency guarantee without load data at their traffic shape | Tails come from queueing and length outliers the benchmark may not have seen | "At the measured mix, p99 TTFT was X at Y req/s; we will confirm on replayed traffic" |
| Vendor benchmark numbers (blog speedups) as their result | Vendor-chosen model, lengths, hardware, and load | "Our published result was on workload Z; on yours we measured W" |
| Throughput at saturation as capacity | Past the knee, latency breaks the SLO | Goodput at the SLO |
| Results from serverless as dedicated performance (or the reverse) | Different batching, isolation, and neighbors | Benchmark the tier you will sell |
| Cost from the roofline bound | Assumes full utilization and no SLO | Cost at the measured operating point |
| Numbers on a config you will not ship | Quantization or flags may change quality | Benchmark the config that passed the eval ([04](../04-poc-and-evaluation/)) |

## 8. Benchmark Report Template

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

## Connections to Other Tracks

| Concept | Connected Track | Application |
|---------|-----------------|-------------|
| Open versus closed loop, goodput, warmup | [Serving & Load](../../ml/04-llm-systems/serving-and-load/) | The method behind every run |
| Poisson arrivals, percentiles | [Probability & Statistics](../../math/07-probability-statistics/) | Arrival models and tail estimation |
| SSE streaming, buffering proxies | [Streaming & SSE](../../ai-platform-engineering/03-streaming-sse/) | Why client TTFT differs from server TTFT |
| Latency SLIs, dashboards | [Observability](../../systems/04-observability/) | Server-side metrics next to client results |

## Company Relevance

| Company | How This Appears | Difficulty |
|---------|-----------------|------------|
| Fireworks AI / Together AI / Baseten | Head-to-head benchmarks against a customer's incumbent during evaluation | Advanced |
| NVIDIA | AIPerf, GenAI-Perf, and MLPerf Inference methodology | Advanced |
| vLLM / SGLang maintainers | `bench serve` and `bench_serving` design | Advanced |
| Any enterprise buying inference | Procurement asks for a benchmark report it can audit | Intermediate |

## Exercise

**Deliverable**: a **benchmark plan and mock report for Lexa** ([brief](../01-discovery/#exercise)), using your hypothesis from [02](../02-qualification-and-sizing/#exercise):

1. The one-sentence question, with the SLO you would propose given they "measure about 8 seconds per section" (state which latency you chose and why).
2. The workload spec from section 2, filling unknown rows with stated assumptions and the data request that would replace each.
3. The run matrix (A to D) adapted to Lexa's online path, plus how you would test the nightly job (throughput and completion time, not TTFT).
4. The apples-to-apples controls you can and cannot match against their closed frontier API.
5. A filled report template using results from `loadgen.py --mock` as stand-in data, clearly labelled as mock.

### Done when

- The 3,000-token instruction block appears as a deliberate prefix-cache test, separate from runs with caching controlled.
- The client region is in the EU and the RTT is reported.
- No result is stated without its percentile, rate, and region.
- The "what never to promise" table has been checked against your verdict paragraph line by line.
