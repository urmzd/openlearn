# Capstone: One Customer, End to End

Every part of this path, applied to one customer. The customer is **Lexa**, the legal-tech company from the [Field Engineering](../) exercises: contract review on a closed frontier API, 6,000 to 14,000-token sections plus a 3,000-token instruction block, 400 to 1,200-token JSON outputs, 3 req/s average and 8 req/s peak during EU business hours, a nightly re-review of 40,000 sections, $60k/month and rising, EU residency and zero retention required, 150 expert-labelled sections, and a CISO who has not been involved yet.

Produce each artifact below. Each one has a check you can apply without an instructor. The worked outlines after the table show how a strong answer reasons; attempt each artifact before opening its outline.

| # | Artifact | Draws on | Passes when |
|---|----------|----------|-------------|
| 1 | **Discovery notes** | Field Engineering 1 | Every unknown in the brief is a question, each question names the decision it changes, and the missing CISO is flagged as a risk with an owner. |
| 2 | **Model shortlist** | LLM Foundations 3 | Three open models, each with total and active parameters, attention variant, and context length from its `config.json`, and one sentence on why it can or cannot handle 17k-token inputs. |
| 3 | **Checkpoint report** | Frameworks and Models 2 | `config_explain.py` and `safetensors_inspect.py --url` output for each shortlisted model, KV bytes per token recorded, and the chat template checked for a JSON-output instruction. |
| 4 | **Sizing** | Inference Performance 4, Field Engineering 2 | `capacity.py` run for the chosen model and GPU; GPU count for 8 req/s at the stated lengths; separate sizing for the nightly batch; a serverless vs dedicated vs batch recommendation with the breakeven shown. |
| 5 | **Benchmark plan and dry run** | Field Engineering 3, Inference Performance 4 | A plan using Lexa's length distribution and arrival pattern, and a `loadgen.py --mock --sweep` run reported as a curve with goodput under the SLO. If you have a GPU, a real vLLM or SGLang run replaces the mock. |
| 6 | **Eval plan** | AI Full Stack 3 | Metrics for clause extraction (per-clause precision and recall, JSON validity rate) on the 150 labelled sections, a parity threshold against the incumbent, and the sample-size caveat stated. |
| 7 | **Fine-tune decision** | Training 2 | A yes or no on LoRA with 150 examples, the reasoning, and if yes, a `memory_calc.py` estimate of GPU-hours and cost. |
| 8 | **POC plan** | Field Engineering 4 | Success criteria tied to artifacts 4 to 6, owners, a timeline, and exit criteria for both success and failure. |
| 9 | **Migration plan** | Field Engineering 5 | Template and sampling parity checks, shadow traffic, canary, cutover, and a rollback trigger. |
| 10 | **TCO and security brief** | Field Engineering 6 | 12-month cost for current spend vs proposed, with the growth rate applied, EU residency and zero retention addressed, and the CISO meeting on the plan. |
| 11 | **Escalation drill** | Field Engineering 7, Inference Performance 1 | A report for this symptom that the performance team could act on without a follow-up: "p99 TTFT jumps from 1.5 s to 9 s at 8 req/s when 14k-token sections arrive together." Include your hypothesis (chunked prefill, batched-token budget) and the evidence that would confirm it. |

**Done when**: all eleven pass their checks, and you can present artifacts 1, 4, 5, 8, and 10 as a 20-minute customer readout without reading from them.

## Model answers, in outline

The full engagement artifacts for a different mock customer, Northstar Commerce, are in [`course/solutions/docs/engagement/`](../../../../course/solutions/docs/engagement/discovery.md) and are walked through in chapters [01](01-discovery.md) to [07](07-escalation-and-handoff.md). The outlines below apply the same reasoning to Lexa. Numbers that come from running a tool (`capacity.py`, `loadgen.py`, `memory_calc.py`) are not given here: the outline shows what the run must establish and the arithmetic that frames it.

<details><summary>1. Discovery notes</summary>

- Turn each unknown into a question tied to a decision: the exact length distribution (sizing), whether the 3,000-token instruction block is identical across requests (prefix caching), the latency the users actually notice (SLO choice), what "cut cost by half" is measured against (TCO baseline), who signs off on EU residency and zero retention (security gate).
- Separate the stated "latency is fine, around 8 seconds per section" from a measurement: ask which latency (end to end?) and at which percentile.
- Flag the missing CISO as a risk with an owner and a date: residency and retention are adoption blockers that nobody on the call can approve.
</details>

<details><summary>2. Model shortlist</summary>

- The context requirement fixes the first filter: the largest request is $14{,}000 + 3{,}000 = 17{,}000$ input tokens plus up to 1,200 output tokens, about 18,200 tokens in context. A model with an 8k context window is out, whatever its quality.
- For each candidate, read total and active parameters, attention variant (MHA, GQA, MLA), and `max_position_embeddings` from its `config.json`, and state in one sentence whether 18k tokens fits natively.
- Prefer GQA or MLA models: KV memory per token scales with the number of KV heads, and at 18k tokens per sequence that dominates memory (artifact 3).
</details>

<details><summary>3. Checkpoint report</summary>

- KV bytes per token is $2 \times \text{layers} \times \text{KV heads} \times \text{head dim} \times \text{bytes per value}$. For an 8B-class GQA config with 32 layers, 8 KV heads, and head dimension 128 in bf16 (the shape of Llama 3.1 8B), that is $2 \times 32 \times 8 \times 128 \times 2 = 131{,}072$ bytes, 128 KiB per token, so about 2.39 GB for one 18,200-token sequence.
- Check the chat template for how a system or JSON-output instruction is rendered: the contract-review prompt depends on it, and parity (artifact 9) depends on rendering it the same way.
</details>

<details><summary>4. Sizing</summary>

- Online path: by Little's law, concurrency is arrival rate times time in system. At the 8 req/s peak and today's 8 s per section, that is 64 requests in flight; at 2.39 GB of KV each (artifact 3's example config), about 153 GB of KV before weights and headroom. This is why the GPU count is driven by KV memory and why prefix caching of the shared 3,000-token block matters.
- Nightly batch: 40,000 sections at 9,000 to 17,000 input tokens is 360M to 680M input tokens and 16M to 48M output tokens per run, sized for throughput and completion time, not TTFT.
- Recommend per path: dedicated capacity for the business-hours online path if the breakeven against serverless pricing is below current volume; batch or spot capacity for the nightly job, which is near zero at other hours. Show the breakeven volume.
</details>

<details><summary>5. Benchmark plan and dry run</summary>

- Replay Lexa's length distribution (not an average length) with open-loop arrivals at the measured pattern, ramping to 8 req/s, and include bursts of 14k-token sections arriving together.
- Report a curve of offered load against TTFT and end-to-end percentiles, and goodput (requests completed within the SLO per second), labelled as mock when run with `loadgen.py --mock`.
</details>

<details><summary>6. Eval plan</summary>

- Metrics: per-clause precision and recall against the expert labels, and the JSON validity rate. Parity threshold: the candidate's recall on the 150 sections is not lower than the incumbent's by more than a stated margin, with JSON validity at or above the incumbent's.
- Caveat: 150 sections is small. A difference of a few points in recall is within sampling noise, and rare failure modes may not appear at all; say what the set cannot establish.
</details>

<details><summary>7. Fine-tune decision</summary>

- No LoRA as the first step. 150 labelled sections are the only evaluation set; training on them leaves nothing held out to measure parity, and 150 examples are few for a 9k to 17k-token task. Try prompt changes and a different open model first; revisit fine-tuning once more labels exist, and size it then with `memory_calc.py`.
</details>

<details><summary>8. POC plan</summary>

- Success criteria taken directly from artifacts 4 to 6 (latency at 8 req/s within the agreed SLO, parity on the 150 sections, JSON validity, nightly completion before business hours), plus EU residency and zero retention verified. Name an owner for each, a timeline, and both exits: what a pass permits and what a fail returns to.
</details>

<details><summary>9. Migration plan</summary>

- Parity before traffic: the same rendered prompt (template and instruction block), the same sampling parameters (temperature, `max_tokens`, JSON mode). Then shadow traffic, a canary cohort, staged cutover, and a rollback trigger (for example JSON validity or recall on sampled reviews falling below the parity threshold) that routes back to the incumbent API.
</details>

<details><summary>10. TCO and security brief</summary>

- Current spend compounds: at 15% monthly growth from 60,000 dollars a month, month 12 costs $60{,}000 \times 1.15^{11} \approx 279{,}000$ dollars, and the 12-month total is $60{,}000 \times (1.15^{12} - 1)/0.15 \approx 1.74$ million dollars. Compare the proposed deployment over the same 12 months with the same growth applied to volume; "cut by half" is measured against this curve, not against today's monthly spend.
- Address EU residency (where the deployment and logs run) and zero retention (what is logged, for how long) with evidence, and put the CISO meeting on the plan with the documents to bring.
</details>

<details><summary>11. Escalation drill</summary>

- Hypothesis: when several 14k-token sections arrive together, their prefills (each about 17k tokens) fill the scheduler's per-step batched-token budget, so requests behind them wait whole prefills before their first token. Chunked prefill, or a smaller per-step token budget, interleaves prefill chunks with decode and bounds that wait.
- Evidence that would confirm it: TTFT rising with the number of concurrent long prefills rather than with request rate; scheduler logs or metrics showing steps saturated by prefill tokens during the spikes; and the spike shrinking when chunked prefill is enabled in a replay of the same burst. Report the symptom, the reproduction, the configuration, and the data, so the performance team can act without a follow-up.
</details>
