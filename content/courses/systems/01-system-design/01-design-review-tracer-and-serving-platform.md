# Design review: tracer and serving platform

## Overview

| | |
|---|---|
| **Module** | `review.01` · proof · docs · Pass 7 · 4 to 6 h |
| **Solution** | worked answers to the six capacity and budget questions (q1 to q6, section 5.2); a model design document of the system after Pass 7, [`q7.md`](../../../../course/solutions/solve/review.01/q7.md), and a model review of it, [`q8.md`](../../../../course/solutions/solve/review.01/q8.md), both walked through in section 5 |
| **Contract** | none of its own: the design names the contracts of every boundary it describes (section 2.2) |
| **Needs** | nothing to run. Reading: the system as built, `L10.0`, `gw.00`, `dep.00`, `obs.00` (the tracer) and `L10.2`, `L10.4`, `L10.5`, `obs.01` with the rest of Pass 7 (the serving platform); [`craft.02`](../../diagramming/02-documentation-writing/01-architecture-decision-records.md) for how one decision is recorded; [`craft.08`](../../software-craftsmanship/05-code-review-and-ci/02-code-review.md) for review as a discipline; the [System Design topic](README.md) |
| **Used by** | no call site (a proof): it is the review that precedes running the platform in production on kind; `review.02` and `review.03` review the control plane and the whole system with the same rubric, and `iv.01` defends this document in an interview |
| **Optional depth** | Kleppmann, *Designing Data-Intensive Applications*, ch. 1; Beyer et al., *The Site Reliability Workbook*, ch. 2 "Implementing SLOs" and ch. 5 "Alerting on SLOs" (sre.google/workbook, free); Harchol-Balter, *Performance Modeling and Design of Computer Systems*, ch. 6 (Little's law) |

## Key Takeaways

- A design document exists to be **reviewed before it is expensive to change**: problem and constraints first, options compared on the same criteria, and the cost of the choice written down.
- **Capacity is arithmetic, not adjectives.** KV bytes per token, pool size, concurrency by Little's law, and the error budget are four lines of multiplication, and a wrong factor (keys without values, query heads instead of KV heads, 32 gaps instead of 31) changes the design (q1 to q3).
- An **error budget** turns an availability target into minutes, and a **burn rate** into how fast those minutes go: 99.5% over 30 days is 216 minutes, and a 14.4x burn spends 1/50 of it per hour (q4 to q6).
- A **failure-modes table** is only as good as its detection column: a dead gateway records no 5xx, so an alert computed from the gateway's own metrics cannot see it.
- A **review** answers every rubric line, recomputes at least one number, challenges at least one failure mode, and ends in a decision with what closes each comment (q8).

## How to read this chapter

Section 2 defines the rubric and the arithmetic a serving design needs, and section 3 reviews a small sibling design by hand. Section 4 states the eight questions. Section 5 answers them: q1 to q6 worked from first principles, then the model design document (q7) and the model review of it (q8), with the reasoning behind each decision. Try each number and each rubric line yourself before opening its answer. The model review was written the way a real one is: the design first, put away for a day, then read as a stranger, with every "no" fixed before it became a "yes".

---

## 1. Why now

Pass 7 turned the tracer, one process per language joined by a C ABI and HTTP, into a serving platform: a batching engine split into prefill and decode, a gateway with keys, limits, routing, and a ledger, a load generator, charts, and a telemetry pipeline. Each module was built and checked alone, against its contract. Nothing yet checks that the pieces add up: that the KV pool holds the concurrency the load implies, that the latency budget leaves headroom for the hop just added, that an alert exists for each way the system can fail, or that a release can be rolled back. Production on kind is next. A design review is the cheapest place to find those gaps: on paper, before a drill finds them.

## 2. Principles

### 2.1 Design document and design review

A **design document** describes a system or a change before (or while) it is built: the problem, the options, the choice, and what follows from it. A decision that stands alone is an ADR (`craft.02`); a design document holds many decisions and the arithmetic that ties them together. A **design review** is a reader who did not write the document checking it against a fixed list of questions, the **rubric**, and returning comments and a decision: approve, approve with changes, or rework. The rubric for this module has eight lines:

| Rubric line | What a yes looks like | Why it matters |
|---|---|---|
| problem, users, constraints first | who calls the system, what it must do, and the limits (one 6-CPU node, the SLOs) before any box or arrow | a solution read before its problem is judged on taste |
| two options, same criteria, numbers | a table: options as columns, criteria as rows, a number in each cell where one exists | one option is a decision already made |
| what the choice gives up | a sentence that starts "we accept..." | every option costs something; an unnamed cost surprises on call |
| every interface names its contract | a table of boundaries and their files (`tinyllm.h`, `openai-subset.v1.yaml`, `tl.kv.v1`, `slo.schema.json`) | contracts are how parts change independently (P3) |
| failure modes, detection, degradation | a table: failure, the signal that shows it, what users see meanwhile | an undetected failure mode is found by users |
| capacity and latency budgets, arithmetic | the multiplication, not just the result | a number without its factors cannot be checked or updated |
| rollout and rollback, migrations | how a version reaches production, how it leaves, and what happens to data and contracts in between | most outages are changes |
| open questions with an owner | a table: question, owner, by when | an unowned question is a decision by default |

### 2.2 The arithmetic a serving design needs

| Symbol | Meaning | Type / shape |
|---|---|---|
| $L$ | transformer layers | integer |
| $H_{kv}$ | key/value heads (grouped-query attention stores these, not the query heads) | integer |
| $d_h$ | head dimension, hidden size over query heads | integer |
| $b$ | bytes per stored value (2 for float16, KV format v1) | integer |
| $B$, $s$ | KV pool blocks, tokens per block (`kv_blocks`, `block_size`) | integers |
| $\lambda$ | arrival rate | requests per second |
| $W$ | time a request spends in the system (end to end) | seconds |
| $N$ | requests in the system, averaged over time | requests |
| $o$ | SLO objective (fraction of good events, 0.995) | fraction |
| $T$ | SLO period (30 days) | hours |
| $r$ | burn rate: observed error ratio over $1 - o$ | dimensionless |

**KV bytes per token** $= 2 \cdot L \cdot H_{kv} \cdot d_h \cdot b$: one key and one value vector per layer and KV head. **Pool bytes** $= B \cdot s \cdot$ (bytes per token), and a sequence of $n$ tokens needs $\lceil n / s \rceil$ blocks.

**Little's law**: in any system where requests arrive and leave, averaged over a long time, $N = \lambda W$. It needs no assumption about the distribution of arrivals or service times. For a streamed completion of $n$ output tokens, $W = \text{TTFT} + (n - 1)\,\text{TPOT}$: the first token arrives at TTFT, and $n - 1$ gaps follow.

**Error budget**: an objective $o$ over a period $T$ allows a fraction $1 - o$ of bad events, which as a full outage is $(1 - o)\,T$. A **burn rate** $r$ spends budget $r$ times faster than the period allows, so a window of length $w$ at rate $r$ spends $r\,w / T$ of the budget, and the whole budget lasts $T / r$. Multi-window alerts (obs.03) fire when $r$ exceeds a threshold over both a long and a short window: 14.4 over 1 h and 5 min pages, 6 over 6 h and 30 min opens a ticket.

## 3. Worked example by hand

A sibling design, not the one of q7: the gateway response cache of `gw.06`, as a one-page draft, then its review.

> **Response cache.** *Problem:* identical temperature-0 requests (retries, dashboards polling) recompute the same completion. *Users:* every client; the engine (load). *Constraints:* one gateway replica, 128 MiB memory limit. *Options:* (A) no cache; (B) in-process LRU with TTL, keyed by tenant, config revision, and the canonical request. *Choice:* B with 4,096 entries and a 60 s TTL. *Interfaces:* `X-TL-Cache: hit|miss` (openai-subset.v1). *Failure modes:* a stale answer after a model rollout. *Capacity:* small. *Rollout:* behind a flag.

The review, line by line:

| Rubric line | Answer | Comment |
|---|---|---|
| problem first | yes | |
| two options with numbers | no | A and B have no numbers: what hit rate makes B worth it? |
| what it gives up | no | memory, and staleness up to the TTL, are not named |
| contracts | yes | |
| failure modes | no | the rollout case is listed, but with no detection and no mitigation |
| capacity, arithmetic | no | "small" is not arithmetic |
| rollout and rollback | yes | |
| open questions | no | none listed |

The comment that matters most recomputes the capacity. A cached stream must be replayable chunk by chunk, so an entry holds the JSON response (about 2 KB for 32 tokens) plus 32 SSE chunks of about 200 B: about 8.4 KB. Then 4,096 entries x 8.4 KB = 34 MB, about a quarter of the 128 MiB limit: it fits, and now the document can say so. The failure-mode comment proposes a fix: the key already includes the config revision, so a rollout that bumps the revision makes old entries unreachable, and the detection is the `tl.gateway.cache.hits` ratio dropping to zero at the rollout. Decision: **approve with changes**, closed when the options carry a hit-rate estimate from the load report and the four "no" lines have their sections.

## 4. The problem set

Exact questions want an integer or a fraction (`1/50`, not `0.02`); q3 accepts a decimal. q7 is judged against the eight lines of the design-review rubric (section 2.1). q8 is judged against these five lines:

- Every design-review rubric line is answered, and each no is a written comment naming the section it concerns.
- At least one comment recomputes a number of the design (a capacity, latency, or budget figure) and says whether it holds.
- Every comment proposes a concrete change, or asks a question and names who answers it.
- At least one comment challenges a failure mode: a way the system fails that the design does not list, or a detection that would not fire.
- The record ends with a decision (approve, approve with changes, or rework) and what closes each open comment.

### Capacity and budgets

Use SmolLM2-135M: 30 layers, 9 query heads, 3 key/value heads, head dimension 64.
The KV cache is format v1, float16 (2 bytes per value), and stores one key and
one value vector per layer, key/value head, and token.

**q1.** How many bytes of KV cache does one token of one sequence occupy? `[number, exact]`

**q2.** The engine's `runtime.toml` sets `kv_blocks = 2048` and `block_size = 16` (tokens per block). (a) How many bytes does the whole KV pool hold? (b) How many sequences of 512 tokens each (prompt plus output) fit in the pool at once? `[number, exact]`

**q3.** Requests arrive at 4 per second. Each streams 32 output tokens, with time to first token 0.5 s and 0.06 s between consecutive output tokens. By Little's law, how many requests are in flight on average? `[number]`

**q4.** The availability SLO is 99.5% of requests without a 5xx over 30 days. Expressed as a full outage, how many minutes of error budget is that? `[number, exact]`

**q5.** A fast-burn alert fires at a burn rate of 14.4 sustained over its 1 hour long window. What fraction of the 30-day error budget has that hour spent? `[number, exact]`

**q6.** At a constant burn rate of 14.4, how many hours until the whole 30-day budget is gone? `[number, exact]`

### The design and its review

**q7.** Write the design document of the system as it stands at the end of Pass 7: the tracer of Pass 1 and the serving platform built on it (engine, gateway, load generator, deployment, observability). Judged against the design-review rubric: the problem, users, and constraints first; at least two options compared on the same criteria with numbers; what the chosen option gives up; the contract of every interface; failure modes with detection and degradation; capacity and latency budgets with the arithmetic (q1 to q6 are part of it); rollout and rollback, including contract and data migrations; open questions with an owner. `[proof]`

**q8.** Review q7 as a reviewer who did not write it (a peer, or the author a day later): answer each line of the design-review rubric, record every "no" and every number recomputed as a comment, and end with a decision. Judged against the five lines above. `[proof]`

## 5. The solution, step by step

### 5.1 From first principles

1. **Numbers before boxes.** A design is reviewable only if its claims can be recomputed. So the capacity questions come first: each is a product of factors that a reader can check one at a time (section 2.2), and the design document quotes them with their factors.
2. **Memory is per token, then per block, then per pool.** One token stores a key and a value per layer and per KV head; the pool is blocks times tokens per block; a sequence needs whole blocks. Every capacity claim about concurrency follows from those three lines.
3. **Concurrency comes from time in system, not from rate.** Little's law, $N = \lambda W$, turns the arrival rate and the end-to-end time into requests in flight, which is what the KV pool and `max_seqs` must hold.
4. **Reliability is a budget of time.** An objective over a period is a number of bad minutes; a burn rate is a speed of spending them. Alert thresholds are chosen so a fast burn pages long before the budget is gone.
5. **A design compares options on the same criteria and names its cost.** The model design compares three engine topologies on TTFT, TPOT, KV moved, blast radius, and footprint, and says what the chosen one gives up.
6. **A review is adversarial and constructive.** It answers every rubric line, recomputes numbers, challenges detections, and every comment ends in a change or a named owner.

### 5.2 Worked answers: capacity and budgets

<details><summary>q1. Answer: 23040 bytes</summary>

KV bytes per token $= 2 \cdot L \cdot H_{kv} \cdot d_h \cdot b = 2 \cdot 30 \cdot 3 \cdot 64 \cdot 2 = 23{,}040$.

The factor 2 is one key and one value vector. Grouped-query attention stores the 3 KV heads; the 9 query heads are recomputed every step and never cached. Common mistakes: 11,520 (keys only, the values forgotten); 69,120 (9 query heads instead of 3 KV heads); 46,080 (float32, 4 bytes, instead of float16); 768 (one layer only).
</details>

<details><summary>q2. Answer: (a) 754974720 bytes, (b) 64 sequences</summary>

(a) Pool bytes $= B \cdot s \cdot 23{,}040 = 2048 \cdot 16 \cdot 23{,}040 = 754{,}974{,}720$ B, exactly 720 MiB.

(b) A 512-token sequence needs $\lceil 512 / 16 \rceil = 32$ blocks, so $2048 / 32 = 64$ sequences fit at once.

Common mistakes: 47,185,920 in (a), which forgets `block_size` and treats 2048 blocks as 2048 tokens; 720, which is MiB, not bytes; 4 in (b), which divides 2048 by 512 as if blocks were tokens; 32 in (b), which is the blocks one sequence needs, not the number of sequences.
</details>

<details><summary>q3. Answer: 9.44 requests (236/25)</summary>

End-to-end time $W = \text{TTFT} + (n - 1)\,\text{TPOT} = 0.5 + 31 \cdot 0.06 = 2.36$ s: the first token arrives at TTFT, then 31 gaps follow for the remaining 31 tokens. Little's law: $N = \lambda W = 4 \cdot 2.36 = 9.44$.

Common mistakes: 9.68, from 32 gaps instead of 31; 2.36, which is $W$ alone and not $\lambda W$; 2, from $\lambda \cdot \text{TTFT}$, which counts only requests still waiting for their first token.
</details>

<details><summary>q4. Answer: 216 minutes</summary>

$(1 - o)\,T = 0.005 \cdot 30 \cdot 24 \cdot 60 = 0.005 \cdot 43{,}200 = 216$ minutes.

Common mistakes: 3.6, which is the same budget in hours; 7.2, which is 0.5% of one day (1,440 minutes) rather than of 30 days.
</details>

<details><summary>q5. Answer: 1/50 of the budget</summary>

A window of length $w$ at burn rate $r$ spends $r\,w / T$ of the budget. With $w = 1$ h and $T = 30 \cdot 24 = 720$ h: $14.4 \cdot 1 / 720 = 1/50$.

Common mistakes: 12/25, from $14.4 / 30$, which puts days where hours belong; 18/125, from $14.4 / 100$.
</details>

<details><summary>q6. Answer: 50 hours</summary>

The whole budget lasts $T / r = 720 / 14.4 = 50$ hours. Consistent with q5: 50 hours at 1/50 per hour.

Common mistakes: 25/12, from $30 / 14.4$, which measures the period in days and gives about 2 days, not hours; 720, which is the period at burn rate 1.
</details>

### 5.3 The model design document (q7)

Full model answer: [`q7.md`](../../../../course/solutions/solve/review.01/q7.md). It follows the rubric's order, and each section exists because one rubric line asks for it.

**Problem, users, constraints (rubric line 1).** Serve completions from the course's own models and from SmolLM2-135M-Instruct behind an OpenAI-compatible API with keys, limits, usage accounting, and latency and availability objectives, observable end to end, on one laptop running a single-node kind cluster. Users: API clients (scripts, the `openai` SDK, the Pass 10 agent, the load generator) and the operator on call. Constraints: one kind node with 6 CPUs and 12 GiB, CPU only; TTFT p95 < 500 ms and TPOT p95 < 60 ms at 4 rps on SmolLM2-135M int4; availability 99.5% over 30 days. Non-goals are written down too: more than one node, GPUs, scaling the gateway out.

**Interfaces (line 4).** One row per boundary, each naming its contract file: safetensors and `tokenizer.json` from training to serving; `openai-subset.v1.yaml` for client to gateway and gateway to engine; `admin.v1.yaml`; `tl.engine.v1` `EngineControl`; `tl.kv.v1` carrying `kv-block.md` v1; `tl.control.v1` heartbeats; `runtime.schema.json` and the Helm values schemas; the telemetry contracts (`semconv.md`, `metrics.yaml`, `slo.schema.json`, and the `release: observability` selection label); `usage.v1.sql`; the load-report schema.

**Options (lines 2 and 3).** Three engine topologies on the same inputs (4 rps, 256 prompt and 32 output tokens, prefill about 1,000 tokens/s per thread, a decode step of about 30 ms):

| Criterion | A: one unified engine, 4 threads | B: two unified engines, 2 threads each | C: 1 prefill (2 threads) + 2 decode (1 thread each) |
|---|---|---|---|
| prefill time of one prompt | 256 / 4,000 = 64 ms | 256 / 2,000 = 128 ms | 128 ms |
| TPOT while a prompt prefills | with 128-token chunks, about 15 + 32 = 47 ms | about 20 + 64 = 84 ms: misses 60 ms | about 30 ms: decode never shares a step with prefill |
| KV moved per request | none | none | 256 x 23,040 B = 5.9 MB, about 6 ms on loopback |
| one pod lost | everything | half the capacity | one decode: about half the in-flight streams; prefill: all new requests |

The decision is to bring up on A and run C in production: C keeps TPOT flat under prefill bursts and turns a pod loss into a partial failure, and A stays one values change away as the rollback topology. **What C gives up** is stated as a cost: a third engine's CPU and memory, a KV transfer path with its own failure modes, TTFT about 70 ms worse than A, and a single prefill pod. It is accepted because the TTFT budget keeps 1.4x headroom and the TPOT gain is the difference between meeting and missing the SLO.

**Failure modes (line 5).** Eight rows, each with a detection that would actually fire and what users see meanwhile. The rows that carry the lesson:

| Failure | Detection | Degradation |
|---|---|---|
| a decode pod dies mid-stream | TTFT burn as the queue moves to the surviving decode; `tl.engine.active_sequences` drops | streams on that pod end with an SSE `error` event, never silently truncated; requests before their first byte retry on the other decode |
| the gateway pod dies | clients see connection errors; **no 5xx is recorded**, because the process that counts them is gone | restart in about 10 s; the detection gap becomes open question 2 |
| KV pool exhausted | free blocks near 0, queue depth rising | preempt by recompute; bounded admission answers 429 with `Retry-After` |
| Prometheus down | nothing fires | none yet: open question 3 |

**Capacity and latency (line 6).** The q1 to q6 numbers with their factors: 23,040 B per token; a 720 MiB pool, 64 sequences of 512 tokens, which is `max_seqs = 64`; 9.44 in flight, each holding 288 tokens = 18 blocks, so about 170 of 2,048 blocks (8%) in steady state; a memory total of about 4.8 GiB of 12 GiB. The TTFT budget is itemised (gateway 2 ms, prefill queueing as an M/D/1 queue at 51% utilization with a p95 allowance of 200 ms, prefill 128 ms, KV push 6 ms, first decode step 30 ms: about 366 ms of 500), and the error budget is 216 minutes, 1/50 per fast-burn hour, gone in 50 hours.

**Rollout and rollback (line 7).** Images by digest; engines roll with `maxUnavailable: 0, maxSurge: 1` and a PDB; the gateway uses `Recreate` because two gateways would contend for one SQLite ledger, with that outage counted against the budget; the A to C change moves traffic 10% then 100% through the admin routes API, and rolls back with the same call; KV v2 arrives readers first, then writers; ledger migrations are forward-only with a backup taken before upgrade.

**Open questions (line 8).** Four rows with owners and dates: the single prefill pod, a synthetic probe so a dead gateway is visible, a dead man's switch for Prometheus, and shared state if the gateway ever scales out.

### 5.4 The model review (q8)

Full model answer: [`q8.md`](../../../../course/solutions/solve/review.01/q8.md). It reviews the first draft of q7, and the current q7 is the result of applying it.

| Rubric line | First read | Comment |
|---|---|---|
| problem, users, constraints first | yes | |
| two options, same criteria, numbers | no | C1 |
| what the choice gives up | yes | |
| every interface names its contract | no | C2 |
| failure modes, detection, degradation | no | C3, C4 |
| capacity and latency budgets, arithmetic | yes, one number wrong | C5 |
| rollout and rollback, migrations | no | C6 |
| open questions with an owner | yes | |

How the five q8 lines are met:

- **Every line answered, every no a comment naming its section.** The table above, with C1 to C6 each headed by the section it concerns.
- **A recomputed number.** C5 recomputes every capacity figure. 23,040 B holds, 9.44 holds, 216 minutes and 1/50 hold, but the draft's pool was 2048 x 23,040 = 47 MB, forgetting the 16 tokens per block; the right figure is 720 MiB, which moves the memory total from about 3 GiB to about 4.8 GiB (still within the node). C1 recomputes option A's TPOT: a 512-token chunk on 4 threads is 128 ms inside one step, so about 143 ms, not "fine".
- **Every comment proposes a change or names who answers.** C1: add option B, a number in every cell, state `prefill_chunk`. C2: add the `release: observability` label to the telemetry row. C6: `Recreate` for the gateway, forward-only migrations with a backup, readers first for KV v2.
- **A challenged failure mode.** C3: the draft said "a dead gateway shows up as availability burn". It cannot: the 5xx ratio is computed from metrics the gateway itself records, so a dead gateway produces no 5xx. C4: streams killed mid-way already returned `200 OK`, so they are invisible to a 5xx-based SLI too.
- **A decision and what closes each comment.** Approve with changes; every comment says what closed it, and open questions 1 and 2 must be answered before production.

### 5.5 Why this solution

- **The arithmetic is the cheapest test of the design.** Six lines of multiplication found the review's largest error (a pool 16 times smaller than claimed); no amount of reading prose would have.
- **Three options, not two.** With only A and C, the comparison hides that "more engines" alone (B) is the worst choice for TPOT; adding B makes the reason for C (separating prefill from decode, not adding replicas) visible.
- **The failure table is judged by its detection column.** Every row asks "which signal, computed by whom, fires?". That question is what exposed the dead-gateway gap, and the fix (an external synthetic probe) is the general pattern: a component cannot report its own absence.
- **Accepted tradeoffs, stated:** about 70 ms of TTFT and a third pod for flat TPOT; about 10 s of outage per gateway deploy for a single-writer ledger; a review by the author a day later instead of a peer, which is weaker and is why q8 has to recompute and challenge, not just tick lines.

### 5.6 Explain it back

1. Why does the KV formula use 3 heads and not 9, and what would the design buy if it used 9?
   <details><summary>Answer</summary>
   Under grouped-query attention, 9 query heads share 3 key/value heads; only keys and values are cached, so the cache stores 3 heads per layer. Using 9 overstates KV memory per token 3x, so the design would believe the pool holds a third of the sequences it really does, and buy three times the RAM it needs.
   </details>
2. Why is $W = 0.5 + 31 \cdot 0.06$ and not $0.5 + 32 \cdot 0.06$?
   <details><summary>Answer</summary>
   TTFT is the time to the first of the 32 tokens. After it, 31 more tokens arrive, each one TPOT after the previous: 31 gaps. Counting 32 adds one TPOT that does not exist and overstates in-flight requests (9.68 instead of 9.44).
   </details>
3. Why can the availability alert not see a dead gateway, and what is the fix?
   <details><summary>Answer</summary>
   The SLI is the 5xx ratio computed from the gateway's own request metrics. A dead gateway serves nothing and records nothing, so the ratio has no bad events (and no events at all). Detection has to come from outside the failed component: a synthetic probe that sends requests and alerts on its own failures.
   </details>
4. A fast-burn alert at 14.4x over 1 h spends 1/50 of the month. Why page at that threshold rather than at, say, 2x?
   <details><summary>Answer</summary>
   At 14.4x the whole 30-day budget is gone in 50 hours, about two days, so a human must act now; at 2x it would last 15 days, which is a ticket, not a page. Pairing the 1 h window with a 5 min window also means the page stops soon after the burn stops.
   </details>
5. Why does the model design choose `Recreate` for the gateway when every engine rolls with surge?
   <details><summary>Answer</summary>
   The gateway's ledger is a SQLite file on a hostPath: during a rolling update two gateways would write the same file, and SQLite allows one writer. `Recreate` accepts about 10 s of outage per deploy (under 2 minutes a month at 10 deploys, against a 216-minute budget) to keep one writer.
   </details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| Counting keys but not values in the KV cache | the pool holds half of what the design claims | the factor 2 in q1 (common wrong answer 11,520) |
| Query heads instead of KV heads under grouped-query attention | KV memory overstated 3x, so the design buys RAM it does not need | $H_{kv} = 3$ in q1 (common wrong answer 69,120) |
| Reading `kv_blocks` as tokens | concurrency underestimated 16x | q2 multiplies by `block_size` and divides a sequence into 32 blocks (wrong answers 47,185,920 and 4) |
| $n$ gaps instead of $n - 1$ in end-to-end time | a small but systematic overestimate of in-flight requests | q3 uses 31 gaps (wrong answer 9.68) |
| $W$ quoted as the concurrency | the scheduler's `max_seqs` sized for 2, not 9 | q3 multiplies by $\lambda$ (wrong answer 2.36) |
| Hours and days mixed in the budget | a fast burn read as half the month's budget | q5 and q6 put $T$ in hours, 720 (wrong answers 12/25 and 25/12) |
| A detection column that reads the failed component's own metrics | the availability alert never fires when the gateway itself is down | review comment C3 and open question 2: an external probe |
| Answering yes to a rubric line because the section exists | a failure table with no detection passes the review | the model review marks the failure line "no" until every row has a detection that would fire |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L10.0` | the tracer engine the design starts from |
| Back | `gw.00` | the tracer gateway |
| Back | `dep.00` | the tracer deployment on kind |
| Back | `obs.00` | the first trace, gateway to engine |
| Back | `L10.2` | continuous batching: what TPOT and `max_seqs` depend on |
| Back | `L10.4` | the block manager whose pool q2 sizes |
| Back | `L10.5` | `tl-serve`: bounded admission, 429 with `Retry-After` |
| Back | `obs.01` | the span tree the failure-modes table relies on |
| Back | `craft.02` | one decision as an ADR; a design document holds many |
| Back | `craft.08` | code review: the same habit one level down |
| Forward | `review.02` | the control plane (durable engine, workers) reviewed with the same rubric |
| Forward | `review.03` | the whole system, after the migrations of Pass 11 |
| Forward | `iv.01` | defending this document's choices in an interview |
| Forward | `field.02` | qualification and sizing for a customer reuses q1 to q3's arithmetic with their numbers |
| Forward | `ops.01` | the decode-kill drill tests the failure-modes table's first row |

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| a design review by the author a day later | design review boards, RFC processes (Rust RFCs, Kubernetes KEPs) | named approvers, a public comment period, an implementation-tracking issue | github.com/kubernetes/enhancements (KEP template) |
| hand capacity arithmetic | capacity planning from load tests and queueing models | measured service-time distributions, M/G/k models, headroom policies | Harchol-Balter, ch. 13 to 15 |
| the error-budget lines | an error-budget policy | what the team stops doing (feature work) when the budget is spent | *Site Reliability Workbook*, appendix B |
