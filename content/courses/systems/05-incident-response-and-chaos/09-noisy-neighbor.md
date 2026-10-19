# Drill: noisy neighbor

## Overview

| | |
|---|---|
| **Module** | `ops.09` · drill · ops, docs · Pass 11 · 1 to 2 h |
| **Solution** | the model runbook `docs/runbooks/NoisyNeighbor.md`, written out with its reasoning in section 5 |
| **Needs** | none |
| **Used by** | no code call site: a drill, one of the Pass 11 operations drills |

## Key Takeaways

- On shared engines one tenant's flood raises every tenant's queueing and TTFT; only **per-tenant** series (`tl_gateway_requests_total` by `tenant`) attribute it.
- Recovery is **bounded admission for the offender** (rate and token limits, 429s), not global throttling that penalises everyone.
- Reserved capacity per tenant keeps a control workload within its SLO even while the limiter is being tuned.

## How to read this chapter

This chapter is a worked drill. Sections 1 to 3 state the risk, the principles, and the drill scenario. Section 4 lists what a correct runbook must cover. Section 5 derives the procedure from first principles, gives the model runbook, and explains why each step is the right one. Try to write the procedure yourself from sections 2 and 3 before reading section 5.

---

## 1. Why now

Shared compute lets one tenant degrade another tenant’s latency. This drill tests detection, attribution, and fair recovery.

## 2. Principles

Measure per-tenant queueing and resource use. Protect reserved capacity, apply bounded admission, and avoid penalizing unrelated tenants.

## 3. Worked example

Generate a high-rate workload from one tenant while a steady control workload runs. Compare queue depth, TTFT, 429s, and utilization by tenant.

## 4. What a correct runbook must cover

| Case | What the runbook must contain | Why it matters |
|---|---|---|
| Owner and detection | the SLO alert that pages, and the per-tenant panels of `obs.04` | a symptom and where to look |
| Attribution | requests per second and 5xx by tenant, queue depth, TTFT, and 429s by tenant | names the offender with evidence |
| Containment | the offender's limits (requests per minute, tokens per minute) lowered; unrelated tenants untouched | fair recovery |
| Verification | the control tenant's TTFT back within SLO; the offender sees 429s, not timeouts | done means measured |
| Follow-up | reserved capacity or per-tenant queues; limits reviewed with the tenant | prevents recurrence |

## 5. The solution, step by step

### 5.1 From first principles

1. **Why does one tenant hurt others?** Engines share a queue. When one tenant's arrival rate pushes utilisation past 1, every queued request waits, whoever sent it (the same queueing as `ops.01`, section 2.1).
2. **Attribute before acting.** The tenant row of the serving dashboard groups `tl_gateway_requests_total` by `tenant`; one line rising while the others are flat, with queue depth rising, names the offender.
3. **Contain the offender only.** The gateway's limiter answers 429 with a split code (`requests_limit_exceeded`, `tokens_limit_exceeded`) and counts `tl_gateway_ratelimit_rejections_total{limit, tenant}`. Lowering that tenant's limits removes the excess load; a global limit would penalise unrelated tenants.
4. **429 is better than a timeout.** A fast, explicit rejection lets the client back off; letting the request queue gives it a slow failure and hurts everyone else meanwhile.
5. **Prevent with reserved capacity.** Per-tenant admission bounds or reserved shares make the next flood a 429 for its sender, not a page.

### 5.2 The model runbook

`docs/runbooks/NoisyNeighbor.md`:

````markdown
# Runbook: noisy neighbor

Owner: the gateway on-call (`<system>-gateway`).

## Symptoms

`TTFTBudgetBurnFast` pages; queue depth rises on engine pods; the tenant row of
the serving dashboard shows one tenant's requests per second rising while
others are flat.

## Diagnosis

```promql
sum by (tenant) (rate(tl_gateway_requests_total[$__rate_interval]))
sum by (tenant) (rate(tl_gateway_ratelimit_rejections_total[$__rate_interval]))
sum by (pod) (tl_engine_queue_depth)
```

Compare TTFT for the control tenant before and during the flood.

## Mitigation

Lower the offending tenant's requests-per-minute and tokens-per-minute limits
in the gateway policy and roll it out. Do not lower global limits. Confirm the
offender now receives 429 (`requests_limit_exceeded` or
`tokens_limit_exceeded`) and the queue drains.

## Verification

The control tenant's TTFT p95 is back under the SLO threshold and the burn
rate under 1; record the tenant, the limits, and the follow-up (reserved
capacity or a limit review with the tenant).
````

### 5.3 Why this solution

- **Per-tenant attribution** turns a vague latency page into a named cause in one query.
- **Targeted limits** restore fairness: only the tenant over its share is slowed, and it gets an explicit signal to back off.
- **Why not scale up?** Adding engines raises capacity for everyone, but a flood can grow to fill it; scaling is a cost decision, limits are the fairness mechanism.

### 5.4 Explain it back

1. Why does a flood from one tenant raise TTFT for all tenants?
   <details><summary>Answer</summary>All requests share the engines' queue; when utilisation exceeds 1 the queue grows and every request's TTFT includes its wait, regardless of tenant.</details>
2. Why lower the offender's limit instead of a global one?
   <details><summary>Answer</summary>A global limit rejects well-behaved tenants too; the per-tenant limit removes exactly the excess load.</details>
3. Why prefer a 429 over letting requests queue?
   <details><summary>Answer</summary>A 429 is immediate and explicit, so the client can back off; queueing turns the excess into slow failures for everyone.</details>

## 6. Pitfalls

| # | Pitfall | Symptom | How the solution avoids it |
|---|---|---|---|
| 1 | Usage graphed as a total | the flood looks like organic growth | diagnosis groups by `tenant` |
| 2 | Throttling every tenant | unrelated tenants are penalised for one tenant's load | only the offender's limits change |
| 3 | No verification on the control workload | the incident is declared over while other tenants still suffer | verification uses the control tenant's TTFT and the burn rate |

## 7. Where it's used next

| Direction | Module | Connection |
|---|---|---|
| Back | `ops.08` | A prerequisite: the contract or evidence this runbook relies on. |
| Back | `load.01` | A prerequisite: the contract or evidence this runbook relies on. |

## Going further

| Resource | What to inspect |
|---|---|
| [Incident drill catalog](README.md) | Compare this inject with the adjacent failure drills and use the same containment and verification fields. |
