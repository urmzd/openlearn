# Capstone: One Customer, End to End

Every part of this path, applied to one customer. The customer is **Lexa**, the legal-tech company from the [Field Engineering](../) exercises: contract review on a closed frontier API, 6,000 to 14,000-token sections plus a 3,000-token instruction block, 400 to 1,200-token JSON outputs, 3 req/s average and 8 req/s peak during EU business hours, a nightly re-review of 40,000 sections, $60k/month and rising, EU residency and zero retention required, 150 expert-labelled sections, and a CISO who has not been involved yet.

Produce each artifact below. Each one has a check you can apply without an instructor. Keep them in `.scratchpad/capstone/` (gitignored) or wherever you keep work; nothing here is committed.

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
