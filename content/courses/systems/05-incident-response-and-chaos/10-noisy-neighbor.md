# A noisy neighbor

Shared compute lets one tenant degrade another tenant's latency. On shared engines, one tenant's flood raises every tenant's queueing and TTFT, and the page that arrives, a TTFT burn, looks exactly like the lost replica of [A decode worker dies mid-stream](02-kill-decode.md). The difference is in attribution: only per-tenant series (`tl_gateway_requests_total` by `tenant`) tell you whose load it is. This drill tests detection, attribution, and fair recovery.

The steady control workload and the flood both come from the load generator of [An open-loop load generator](../../build-an-llm/11-serving/11-load-generator.md) in *Build Your Own LLM System*, and the per-tenant panels come from [Dashboards as code for the serving path](../04-observability/05-dashboards-as-code.md). The drill follows [A data incident](09-data-incident.md) in the sequence of operations drills.

The section states the principles and the drill scenario, lists what a correct runbook must cover, and derives the procedure, ending with the model runbook `docs/runbooks/NoisyNeighbor.md`. Try to write the procedure yourself before you read the solution.

> **Learning objectives**
> - Explain, in queueing terms, why one tenant's flood raises every tenant's TTFT.
> - Attribute the load to a tenant with per-tenant queries before acting.
> - Contain only the offender with bounded admission, and explain why a 429 beats a timeout.
> - Verify recovery on the control tenant, and prevent recurrence with reserved capacity.

## Principles

Measure per-tenant queueing and resource use. Protect reserved capacity, apply bounded admission, and avoid penalizing unrelated tenants. Recovery is bounded admission for the offender (rate and token limits, 429s), not global throttling that penalises everyone. Reserved capacity per tenant keeps a control workload within its SLO even while the limiter is being tuned.

## The drill scenario

Generate a high-rate workload from one tenant while a steady control workload runs. Compare queue depth, TTFT, 429s, and utilization by tenant.

## What a correct runbook must cover

A runbook for this incident is complete when it answers five questions:

| Case | What the runbook must contain | Why it matters |
|---|---|---|
| Owner and detection | the SLO alert that pages, and the per-tenant panels of the serving dashboard | a symptom and where to look |
| Attribution | requests per second and 5xx by tenant, queue depth, TTFT, and 429s by tenant | names the offender with evidence |
| Containment | the offender's limits (requests per minute, tokens per minute) lowered; unrelated tenants untouched | fair recovery |
| Verification | the control tenant's TTFT back within SLO; the offender sees 429s, not timeouts | done means measured |
| Follow-up | reserved capacity or per-tenant queues; limits reviewed with the tenant | prevents recurrence |

## Deriving the procedure

### Reasoning from first principles

Why does one tenant hurt others? Engines share a queue. When one tenant's arrival rate pushes utilisation past 1, every queued request waits, whoever sent it. This is the same queueing as in [the lost-replica drill](02-kill-decode.md#what-a-lost-replica-does), with demand rising instead of capacity falling.

Attribute before acting. The tenant row of the serving dashboard groups `tl_gateway_requests_total` by `tenant`; one line rising while the others are flat, with queue depth rising, names the offender.

Contain the offender only. The gateway's limiter answers 429 with a split code (`requests_limit_exceeded`, `tokens_limit_exceeded`) and counts `tl_gateway_ratelimit_rejections_total{limit, tenant}`. Lowering that tenant's limits removes the excess load; a global limit would penalise unrelated tenants.

A 429 is better than a timeout. A fast, explicit rejection lets the client back off; letting the request queue gives it a slow failure and hurts everyone else meanwhile.

Prevent with reserved capacity. Per-tenant admission bounds or reserved shares make the next flood a 429 for its sender, not a page.

### The model runbook

The procedure written out as `docs/runbooks/NoisyNeighbor.md`:

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

### Why this solution

Per-tenant attribution turns a vague latency page into a named cause in one query. Targeted limits restore fairness: only the tenant over its share is slowed, and it gets an explicit signal to back off. Scaling up is the tempting alternative, but adding engines raises capacity for everyone, and a flood can grow to fill it. Scaling is a cost decision; limits are the fairness mechanism.

## What can go wrong

Three mistakes either hide the offender or punish the innocent:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| Usage graphed as a total | the flood looks like organic growth | diagnosis groups by `tenant` |
| Throttling every tenant | unrelated tenants are penalised for one tenant's load | only the offender's limits change |
| No verification on the control workload | the incident is declared over while other tenants still suffer | verification uses the control tenant's TTFT and the burn rate |

## Where this leads

The workloads in this drill come from [An open-loop load generator](../../build-an-llm/11-serving/11-load-generator.md), and it follows [A data incident](09-data-incident.md). [A runaway agent](11-runaway-agent.md) is the same problem one level up: a single agent run consuming shared budget, contained by a limit on that run alone. To compare this inject with the adjacent drills and reuse the same containment and verification fields, see the [chapter opening](README.md).

## Key terms

- **Noisy neighbor**: a tenant whose load degrades other tenants' latency on shared resources.
- **Per-tenant series**: metrics labelled by `tenant`, which attribute load and rejections to their source.
- **Bounded admission**: per-tenant request and token limits that reject excess load with a 429.
- **Control workload**: a steady, well-behaved tenant whose TTFT measures whether recovery is fair.
- **Reserved capacity**: a share of the engines guaranteed to a tenant regardless of others' load.

## Summary

- On shared engines one tenant's flood raises every tenant's queueing and TTFT; only per-tenant series attribute it.
- Attribute before acting: one tenant's rate rising while others are flat, with queue depth rising, names the offender.
- Recovery is bounded admission for the offender, with explicit 429s, not global throttling.
- Verify on the control tenant's TTFT and the burn rate, and prevent recurrence with reserved capacity or per-tenant queues.

## Review questions

1. Why does a flood from one tenant raise TTFT for all tenants?
   <details><summary>Answer</summary>All requests share the engines' queue; when utilisation exceeds 1 the queue grows and every request's TTFT includes its wait, regardless of tenant.</details>
2. Why lower the offender's limit instead of a global one?
   <details><summary>Answer</summary>A global limit rejects well-behaved tenants too; the per-tenant limit removes exactly the excess load.</details>
3. Why prefer a 429 over letting requests queue?
   <details><summary>Answer</summary>A 429 is immediate and explicit, so the client can back off; queueing turns the excess into slow failures for everyone.</details>
4. The page is `TTFTBudgetBurnFast`, the same alert as for a lost decode replica. What single observation tells the two apart?
   <details><summary>Answer</summary>Whether demand or capacity changed. A lost replica shows a pod that went away with per-tenant rates flat; a noisy neighbor shows all replicas present and one tenant's request rate rising.</details>
