# Performance engagement plan: Northstar Commerce

**Status:** test protocol agreed, run pending customer thresholds  
**Customer owner:** Maya Chen  
**Test owner:** Alex Rivera  
**Environment owner:** Northstar platform operations

## Question and scope

The test will determine whether the candidate serving configuration can handle Northstar's weekday peak request mix within customer-defined latency and error limits. It will not establish production readiness, model quality for all users, or performance on unrepresented prompts.

The candidate and baseline must use the same model weights, tokenizer, hardware class, prompt sample, request arrival schedule, and warm-up procedure. The input sample is redacted and approved by Priya Shah. Keep the sample hash, config, software revision, and load-generator version with the report. Use an open-loop arrival schedule to avoid coordinated omission.

## Measurement protocol

1. Capture a 30-minute baseline and a 30-minute candidate run after an agreed warm-up.
2. Replay the approved peak pattern with a 10-minute ramp, 20-minute steady window, and 5-minute drain.
3. Record offered and completed requests/s, input and output tokens/s, p50/p95/p99 TTFT and TPOT, end-to-end latency, timeout/error rate, queue depth, GPU/CPU utilization, and memory high-water mark.
4. Repeat each condition three times. Report each run and the median; include spread and any invalidated runs with the reason.
5. Stop if error rate exceeds 2% for two consecutive minutes, memory reaches 90% of the limit, or customer operations requests a stop.

The 2% stop limit is a proposed safety guard for the test, not an agreed success threshold. Maya and operations must approve the stop rule and set p95 TTFT/TPOT targets before the run. A report that lacks those approvals is exploratory only.

## Analysis and response to regression

Compare like-for-like runs and identify saturation before recommending changes. If the candidate regresses by more than the agreed 5% performance budget, preserve the run metadata and use the performance bisect procedure from `craft.16` to locate the change. Do not tune during a measured run. Keep customer traffic isolated from this exercise.

**Next action:** Maya confirms target percentiles and the test window. Alex records the baseline and candidate revision IDs, then publishes a report with raw summary tables, environment, limitations, and recommendation. **Recovery:** stop load, return the test environment to baseline, and notify Northstar operations if the stop condition fires.
