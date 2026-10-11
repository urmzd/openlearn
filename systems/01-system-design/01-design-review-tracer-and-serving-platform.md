<!-- ol:module review.01 -->
# Design review: tracer and serving platform

## Overview

| | |
|---|---|
| **Module** | `review.01` · proof · docs · Pass 7 · 4 to 6 h |
| **You build** | `solve/review.01.toml`: six capacity and budget answers checked by SymPy (q1 to q6), the design document of your system as it stands after Pass 7 in `solve/review.01/q7.md`, and a review of that document in `solve/review.01/q8.md`, both self-graded against rubrics |
| **Contract** | none of its own: the design names the contracts of every boundary it describes (section 2.2) |
| **Tests** | `course/solve/review.01/key.toml` (hidden): typed answers with reject canaries; q7 uses [`course/rubrics/design-review.md`](../../course/rubrics/design-review.md), q8 the rubric in section 4 |
| **Needs** | nothing to run. Reading: what you built, `L10.0`, `gw.00`, `dep.00`, `obs.00` (the tracer) and `L10.2`, `L10.4`, `L10.5`, `obs.01` with the rest of Pass 7 (the serving platform); [`craft.02`](../../software-craftsmanship/06-documentation-writing/01-architecture-decision-records.md) for how one decision is recorded; [`craft.08`](../../software-craftsmanship/08-code-review-and-ci/02-code-review.md) for review as a discipline; the [System Design topic](README.md) |
| **Used by** | no call site (a proof): `review.02` and `review.03` review the control plane and the whole system with the same rubric, and `iv.01` defends this document in an interview |
| **Milestone** | `MS-prod` (the production milestone requires review.01) |
| **Optional depth** | Kleppmann, *Designing Data-Intensive Applications*, ch. 1; Beyer et al., *The Site Reliability Workbook*, ch. 2 "Implementing SLOs" and ch. 5 "Alerting on SLOs" (sre.google/workbook, free); Harchol-Balter, *Performance Modeling and Design of Computer Systems*, ch. 6 (Little's law) |

## Key Takeaways

- A design document exists to be **reviewed before it is expensive to change**: problem and constraints first, options compared on the same criteria, and the cost of the choice written down.
- **Capacity is arithmetic, not adjectives.** KV bytes per token, pool size, concurrency by Little's law, and the error budget are four lines of multiplication, and a wrong factor (keys without values, query heads instead of KV heads, 32 gaps instead of 31) changes the design (q1 to q3).
- An **error budget** turns an availability target into minutes, and a **burn rate** into how fast those minutes go: 99.5% over 30 days is 216 minutes, and a 14.4x burn spends 1/50 of it per hour (q4 to q6).
- A **failure-modes table** is only as good as its detection column: a dead gateway records no 5xx, so an alert computed from the gateway's own metrics cannot see it.
- A **review** answers every rubric line, recomputes at least one number, challenges at least one failure mode, and ends in a decision with what closes each comment (q8).

## How to work this chapter

```bash
ol start review.01               # writes solve/review.01.toml and the two documents to fill
ol check review.01               # SymPy checks q1 to q6, then asks each rubric line of q7 and q8 (y/n)
ol check review.01 --regrade     # ask the rubrics again after you change a document
```

Write q7 first, put it away for a day (or hand it to a peer), then write q8 as its reviewer and apply what q8 asks for. The self-grade is honest only if you answer no where q8 found a gap and fix it before you answer yes.

---

## 1. Why now

Pass 7 turned the tracer, one process per language joined by a C ABI and HTTP, into a serving platform: a batching engine split into prefill and decode, a gateway with keys, limits, routing, and a ledger, a load generator, charts, and a telemetry pipeline. Each module was graded alone, against its contract. Nothing yet checks that the pieces add up: that the KV pool holds the concurrency your load implies, that the latency budget leaves headroom for the hop you added, that an alert exists for each way the system can fail, or that you can roll back. Production on kind (`MS-prod`) is next. A design review is the cheapest place to find those gaps: on paper, before a drill finds them for you.

## 2. Principles

### 2.1 Design document and design review

A **design document** describes a system or a change before (or while) it is built: the problem, the options, the choice, and what follows from it. A decision that stands alone is an ADR (`craft.02`); a design document holds many decisions and the arithmetic that ties them together. A **design review** is a reader who did not write the document checking it against a fixed list of questions, the **rubric**, and returning comments and a decision: approve, approve with changes, or rework. The rubric for this module is [`course/rubrics/design-review.md`](../../course/rubrics/design-review.md):

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

A sibling design, not the one you write: the gateway response cache of `gw.06`, as a one-page draft, then its review.

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

Write each answer in `solve/review.01.toml`; lettered parts are their own tables, the documents are files:

```toml
[q1]
answer = "2*30*3*64*2"
[q2.a]
answer = "754974720"
[q3]
answer = "9.44"
[q7]
proof = "review.01/q7.md"
[q8]
proof = "review.01/q8.md"
```

Exact questions take integers or fractions (`1/50`); `0.02` fails a question marked exact. q3 accepts a decimal. q7 is graded against the eight lines of the design-review rubric. q8 is graded against these five lines, which `ol check` asks one by one:

- Every design-review rubric line is answered, and each no is a written comment naming the section it concerns.
- At least one comment recomputes a number of the design (a capacity, latency, or budget figure) and says whether it holds.
- Every comment proposes a concrete change, or asks a question and names who answers it.
- At least one comment challenges a failure mode: a way the system fails that the design does not list, or a detection that would not fire.
- The record ends with a decision (approve, approve with changes, or rework) and what closes each open comment.

<!-- ol:problems review.01 -->

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

**q7.** Write the design document of your system as it stands at the end of Pass 7: the tracer of Pass 1 and the serving platform built on it (engine, gateway, load generator, deployment, observability). Graded against `course/rubrics/design-review.md`: the problem, users, and constraints first; at least two options compared on the same criteria with numbers; what the chosen option gives up; the contract of every interface; failure modes with detection and degradation; capacity and latency budgets with the arithmetic (q1 to q6 are part of it); rollout and rollback, including contract and data migrations; open questions with an owner. `[proof]`

**q8.** Review q7 as a reviewer who did not write it (a peer, or you a day later): answer each line of the design-review rubric, record every "no" and every number you recomputed as a comment, and end with a decision. Graded against its own rubric (chapter section 4). `[proof]`

<!-- /ol:problems -->

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| Counting keys but not values in the KV cache | the pool holds half of what the design claims | q1 (canary 11520) |
| Query heads instead of KV heads under grouped-query attention | KV memory overstated 3x, so the design buys RAM it does not need | q1 (canary 69120) |
| Reading `kv_blocks` as tokens | concurrency underestimated 16x | q2 (canaries 47185920, 4) |
| $n$ gaps instead of $n - 1$ in end-to-end time | a small but systematic overestimate of in-flight requests | q3 (canary 9.68) |
| $W$ quoted as the concurrency | the scheduler's `max_seqs` sized for 2, not 9 | q3 (canary 2.36) |
| Hours and days mixed in the budget | a fast burn read as half the month's budget | q5 (canary 12/25), q6 (canary 25/12) |
| A detection column that reads the failed component's own metrics | the availability alert never fires when the gateway itself is down | q8 rubric, line 4 |
| Answering yes to a rubric line because the section exists | a failure table with no detection passes the self-grade | q7 rubric, line 5 |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L10.0` | the tracer engine your design starts from |
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

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| a self-graded design review | design review boards, RFC processes (Rust RFCs, Kubernetes KEPs) | named approvers, a public comment period, an implementation-tracking issue | github.com/kubernetes/enhancements (KEP template) |
| hand capacity arithmetic | capacity planning from load tests and queueing models | measured service-time distributions, M/G/k models, headroom policies | Harchol-Balter, ch. 13 to 15 |
| the error-budget lines | an error-budget policy | what the team stops doing (feature work) when the budget is spent | *Site Reliability Workbook*, appendix B |
