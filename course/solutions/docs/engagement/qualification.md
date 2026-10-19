# Qualification and capacity estimate: Northstar Commerce

**Status:** qualified for a bounded test, subject to data approval  
**Customer sponsor:** Maya Chen, platform lead  
**Technical owner:** Alex Rivera, solutions engineer  
**Decision date:** 2026-10-23

## Qualification

| Question | Finding | Owner / evidence |
|---|---|---|
| Is the problem urgent? | Peak-hour delay causes manual fallback, but lost staff time is not yet measured | Maya to quantify from one week of support records |
| Is there a sponsor? | Yes, Maya owns the technical evaluation | Discovery notes |
| Is there a decision maker? | Budget owner remains unknown | Maya to identify before POC approval |
| Can data be used? | Only a redacted sample after security approval | Priya Shah; approval is a gate, not assumed |
| Is there a decision date? | Yes, go/no-go on Oct 23 | Customer calendar |

## First-pass sizing, explicitly an estimate

The sample provided for this exercise describes a peak of **12 requests/s**, **1,200 input tokens/request**, and **300 generated tokens/request**. Treat these as customer supplied but not yet validated. At peak, the offered load is approximately 14,400 input tokens/s and 3,600 output tokens/s. The capacity test must replay the observed arrival pattern rather than assume a constant rate.

For a transparent memory estimate, assume a 1.3B parameter model with 24 layers, 16 KV heads, head dimension 64, and bf16 KV values. One token of KV cache is approximately `2 × layers × KV heads × head dimension × 2 bytes = 98,304 bytes` per sequence position. At 1,500 total tokens and 12 concurrent requests, that is about 1.77 GiB of KV data before allocator/block overhead, weights, runtime state, and safety margin. This estimate is not a deployment commitment. Validate the actual config and concurrency against load and memory reports.

## Decision and gates

Proceed with a one-day offline capacity test if Priya approves the redacted sample and Maya identifies the budget owner. Success means replaying the agreed peak mix without exceeding the customer-approved p95 TTFT and TPOT limits, with errors below the agreed threshold and no increase in data exposure. Those numeric limits are still open; the customer must set them before the run.

**Next action:** Maya supplies the budget owner and proposed thresholds by Oct 16. Alex records the exact model config, workload file hash, and test environment before running the test. If data approval is late, use synthetic prompts and label the result as a harness check only.
