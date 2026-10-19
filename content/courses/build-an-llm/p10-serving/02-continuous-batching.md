# Continuous batching scheduler with priority and preemption

## Overview

| | |
|---|---|
| **Module** | `L10.2` · build · Rust · Pass 7 · 10 to 14 h |
| **Solution** | [`rust/crates/tl-engine/src/sched.rs`](../../../../course/solutions/rust/crates/tl-engine/src/sched.rs): the `BlockSpace` seam and its simplest implementation `FreeListBlocks`, the `PrefillPolicy` seam with `WholePrompt`, and `Scheduler`: admission by free blocks, a waiting queue ordered by priority with aging, a running batch re-formed every step, preemption by recompute, abort |
| **Contract** | no Rust trait contract file yet: section 4 pins the API; `X-TL-Priority` in [`openai-subset.v1.yaml`](../../../../course/contracts/openapi/openai-subset.v1.yaml) and `PrefillRequest.priority` in [`engine.proto`](../../../../course/contracts/proto/tl/engine/v1/engine.proto) feed `Request.priority` |
| **Needs** | `L10.1` the runner, `SchedPolicy`, and the KV pool ([chapter](01-model-runner-and-sampler.md)) · `ds.06` the lazy-deletion heap ([chapter](../../../tracks/algorithms/16-systems-data-structures/06-binary-heap-lazy-deletion.md)) · reading: `lang.04` Rust, `L8.3` paged KV in Python, `M05.1` KV memory accounting |
| **Used by** | `L10.3` (chunked prefill is a `PrefillPolicy`), `L10.4` (the block manager is a `BlockSpace`), `L10.5` (the engine loop calls `schedule` and `on_step` every step) · with it the engine can complete 64 concurrent requests and return the KV pool to baseline |
| **Optional depth** | [Yu et al. 2022, Orca: iteration-level scheduling](https://www.usenix.org/conference/osdi22/presentation/yu) (free); [Kwon et al. 2023, vLLM and PagedAttention](https://arxiv.org/abs/2309.06180) (free), section 4.5 on preemption; [vLLM scheduler](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py) (free) |

## Key Takeaways

- The batch is re-formed at **every step**: finished requests leave and waiting ones join, so no sequence waits for the longest one in its batch.
- Admission is by **free blocks**: a request joins only when the pool can hold its whole sequence so far; the first request that does not fit stops admission, so big requests are not starved by small ones behind them.
- Priority with aging needs no re-sorting: ranking by `priority + age / A` orders requests exactly like the fixed key `priority * A - arrival`.
- When a running request needs a block and none is free, the lowest-ranked running request is **preempted**: its blocks are freed and it is recomputed later from its tokens, with the same output.
- Batched greedy output equals single-request output, because the kernels are batch-invariant.

## How to read this chapter

Sections 1 to 3 derive continuous batching, admission by free blocks, aging as a fixed key, and preemption by recompute, then run three steps of a small schedule by hand. Section 4 fixes the Rust interface and lists what a correct scheduler must handle. Section 5 walks through the solution, [`sched.rs`](../../../../course/solutions/rust/crates/tl-engine/src/sched.rs), phase by phase. Try to predict each step of the solution before reading it.

The waiting queue is the `tl-ds` `LazyHeap` from `ds.06`, a path dependency of `tl-engine`.

---

## 1. Why now

`L10.1` runs one step over any batch it is given. Something has to decide what goes into each step: which requests run, which wait, and what happens when memory runs out. Serving requests one after another wastes the hardware: a decode step for one sequence reads every weight to produce a single token, and reading the weights is the expensive part, so 16 sequences in one step cost little more than 1. **Static** batching (collect 16 requests, run them to the end together) wastes the other way: short requests finish and their slots sit empty until the longest one ends, and a new request waits for the whole batch. **Continuous** batching decides again at every step (Orca, 2022). This module is that decision.

## 2. Principles

### 2.1 A step, its budget, and its cap

| Symbol | Meaning | Type |
|---|---|---|
| $\beta$ | token budget per step (`max_batch_tokens`) | integer |
| $S_{\max}$ | running sequences at most (`max_seqs`) | integer |
| $B$ | token positions per KV block (`block_size`) | integer |
| $n_r$ | tokens of request $r$ so far: prompt plus generated | integer |
| $c_r$ | positions of $r$ whose K and V are computed | integer |
| $P_r$, $a_r$ | priority and arrival step of $r$ | integers |
| $A$ | steps per level of aging (`aging_steps`) | integer |
| $t$ | the current step | integer |

A request is **decoding** when $n_r - c_r = 1$ (its newest token has not been fed yet) and **prefilling** when $n_r - c_r > 1$. A decode costs one token of the budget; a prefill chunk costs its length. Each step processes at most $\beta$ tokens and runs at most $S_{\max}$ sequences.

### 2.2 The three phases of `schedule`

1. **Decodes**, highest rank first. Each needs room for one more position: `append_slot(r, n_r)` grows its block table to $\lceil n_r / B \rceil$ blocks. When the pool has no block, the lowest-ranked running request not yet scheduled is preempted, and the slot is tried again; when nothing ranks lower, the request preempts itself.
2. **Prefills in progress** (only with chunked prefill, `L10.3`): each gets the chunk its `PrefillPolicy` grants from the remaining budget.
3. **Admissions**, from the waiting queue in rank order, while fewer than $S_{\max}$ run and budget is left. A request is admitted only when `allocate` finds blocks for all $n_r$ positions (all or nothing); the first one that does not fit **stops** admission (no skipping ahead), so a large request at the head is not starved by a stream of small ones.

Decodes go first because a running request's next token is what its user is waiting for; the time per output token (TPOT) stays steady while new requests queue.

### 2.3 How many blocks a request needs

A sequence of $n$ positions occupies $\lceil n / B \rceil$ blocks. One block holds $B$ positions of K and V for every layer and KV head, $2 \cdot L \cdot H_{kv} \cdot B \cdot D \cdot 2$ bytes in f16 (`M05.1`): for SmolLM2-135M ($L = 30$, $H_{kv} = 3$, $D = 64$) and $B = 16$, that is 368,640 bytes, so a pool of 2048 blocks is 755 MB and holds 32,768 positions. The scheduler never touches bytes: a `BlockSpace` hands out block ids (`FreeListBlocks` here, the block manager of `L10.4` later), and the runner writes into them.

`add` refuses at once what could never run: an empty prompt, a prompt that fills the context, and a request whose longest possible sequence, $\min(\text{prompt} + \text{max\_tokens}, \text{context})$, needs more blocks than the pool holds (it would be preempted forever). A full waiting queue is `QueueFull`, which the server answers with 429.

### 2.4 Priority with aging, as a fixed key

Strict priority starves: a steady stream of priority-10 requests keeps a priority-0 request waiting forever. **Aging** raises a waiting request's effective priority with its age: $\pi_r(t) = P_r + (t - a_r) / A$. Re-sorting the queue every step would cost $O(W \log W)$; instead notice that for two waiting requests

$$\pi_r(t) > \pi_s(t) \iff P_r + \tfrac{t - a_r}{A} > P_s + \tfrac{t - a_s}{A} \iff P_r A - a_r > P_s A - a_s,$$

because $t$ cancels. So the waiting queue is a heap on the **fixed** key $k_r = P_r A - a_r$ (higher first; equal keys by arrival), built once at push. It is the `ds.06` `LazyHeap`, whose handles also make `abort` of a waiting request O(1). With `Fcfs` every key is 0 and arrival decides. A preempted request goes back with its original arrival, so it keeps its age.

### 2.5 Preemption by recompute

When memory runs out mid-decode, something must give. vLLM's two options: **swap** the victim's blocks to host memory and back, or **recompute**: free its blocks and later prefill its whole sequence again (prompt plus every token generated so far). This engine recomputes: no second memory tier, and the recomputed K and V are bit-identical to the originals because the runner is chunk-invariant (`L10.1`), so the victim's output does not change. Its sampler state is untouched (its tokens and its generator stay with the request), so even seeded sampling continues where it stopped.

The victim is the **lowest-ranked** running request, at equal rank the newest: it has had the least service, so evicting it wastes the least work.

### 2.6 Two seams

`BlockSpace` is the scheduler's only view of memory (block size, totals, `allocate`, `append_slot`, `release`, `block_table`); `PrefillPolicy` is its only view of how prompts are cut (`chunk_len`). Both are traits so this module can be finished and tested before `L10.3` and `L10.4` exist, and so the scheduler can be driven with a fake model and a plain free list.

## 3. Worked example by hand

Blocks of $B = 4$ (six of them, ids 0 to 5), budget $\beta = 8$, $S_{\max} = 2$, `Fcfs`. Requests: A has a 5-token prompt and `max_tokens` 3; B a 3-token prompt and 2; C a 2-token prompt and 1.

**Step 1.** Nothing runs, so phase 3 admits. A needs $\lceil 5/4 \rceil = 2$ blocks: ids 0 and 1; its whole prompt (5 tokens) fits the budget, which drops to 3. B needs 1 block: id 2; 3 tokens, budget 0. Two sequences run, so C waits. The plan: prefill A 0..5 and B 0..3, both last chunks (sampled). Each samples one token: A has 6 tokens, B has 4.

**Step 2.** Phase 1: A decodes (it needs room for 6 positions, which 2 blocks already give), then B (4 positions, 1 block). Budget 6 left, but $S_{\max} = 2$ blocks C again. B's second token reaches its `max_tokens`: B finishes with `Length`, and block 2 goes back.

**Step 3.** A decodes (7 positions); C is admitted into block 2 and prefilled (2 tokens). A's third token and C's first end both: everything is free again.

**Aging.** With $A = 4$: a priority-0 request that arrived at step 0 has key $0 \cdot 4 - 0 = 0$; a priority-10 request arriving at step $t$ has key $40 - t$. From step $t = 41$ on, new priority-10 arrivals have keys below 0, so the old request is admitted first: aging bounded its wait at about $P \cdot A = 40$ steps.

## 4. The interface

```rust
// rust/crates/tl-engine/src/sched.rs
pub type RequestId = u64;
pub struct NoCapacity;
pub struct Allocation { pub cached_tokens: usize }          // 0 here; prefix hits in L10.4
pub trait BlockSpace {
    fn block_tokens(&self) -> usize;  fn total_blocks(&self) -> usize;  fn free_blocks(&self) -> usize;
    fn allocate(&mut self, id: RequestId, tokens: &[u32]) -> Result<Allocation, NoCapacity>;   // all or nothing
    fn append_slot(&mut self, id: RequestId, total: usize) -> Result<(), NoCapacity>;
    fn release(&mut self, id: RequestId, computed: &[u32]);
    fn block_table(&self, id: RequestId) -> &[u32];
}
pub struct FreeListBlocks { /* ... */ }  impl FreeListBlocks { pub fn new(block_tokens: usize, ids: Vec<u32>) -> Self; }
pub trait PrefillPolicy { fn chunk_len(&self, remaining: usize, budget: usize, alone: bool) -> usize; }
pub struct WholePrompt;                  // remaining if it fits the budget or the step is empty, else 0

pub struct SchedulerConfig { pub max_seqs: usize, pub max_batch_tokens: usize, pub max_waiting: usize,
                             pub policy: SchedPolicy, pub aging_steps: u64, pub max_model_len: usize }
pub struct Request { pub prompt: Vec<u32>, pub max_tokens: usize, pub priority: i32 }
pub enum AdmitError { QueueFull, Empty, TooLong { tokens: usize, limit: usize }, NeverFits { blocks: usize, total: usize } }
pub struct Chunk { pub id: RequestId, pub start: usize, pub len: usize, pub last: bool }
pub struct ScheduleOutput { pub prefill: Vec<Chunk>, pub decode: Vec<RequestId>, pub preempted: Vec<RequestId> }
pub struct StepOutput { pub id: RequestId, pub token: u32, pub stop: bool }
pub enum FinishReason { Stop, Length }
pub enum RequestEvent { Token { id: RequestId, token: u32 }, Finished { id: RequestId, reason: FinishReason } }
impl<B: BlockSpace> Scheduler<B> {
    pub fn new(cfg: SchedulerConfig, blocks: B) -> Self;
    pub fn with_prefill_policy(self, p: Box<dyn PrefillPolicy + Send>) -> Self;
    pub fn add(&mut self, req: Request) -> Result<RequestId, AdmitError>;
    pub fn abort(&mut self, id: RequestId);
    pub fn schedule(&mut self) -> ScheduleOutput;
    pub fn on_step(&mut self, outputs: &[StepOutput]) -> Vec<RequestEvent>;
    pub fn tokens(&self, id: RequestId) -> Option<&[u32]>;  pub fn prompt_len(&self, id: RequestId) -> Option<usize>;
    pub fn unfinished(&self) -> Vec<RequestId>;  pub fn has_unfinished(&self) -> bool;  pub fn stats(&self) -> SchedStats;
    pub fn blocks(&self) -> &B;  pub fn blocks_mut(&mut self) -> &mut B;
}
```

The caller's loop is: `schedule`, build a `ForwardBatch` (one sequence per decode at position $n_r - 1$, one per chunk), `ModelRunner::forward`, sample every decode row and every chunk with `last`, then `on_step` with one `StepOutput` per sampled row. `on_step` advances $c_r$, appends each token, and finishes requests that stopped (`stop`, set by the engine for EOS and stop strings) or reached `max_tokens` or the context.

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the hand-worked schedule | unit | section 3, step by step: chunks, decodes, block ids, the finishes | the worked example |
| an all-or-nothing free list | unit | lowest ids first; no partial allocation | admission never half-allocates |
| every request finishes, no block leaks | property | random arrivals with forced preemptions: exact outputs, caps respected, every block free at the end | the engine runs for weeks |
| priority runs first | unit | `Priority` admits by priority, `Fcfs` by arrival | `X-TL-Priority` from the gateway's tenant tiers |
| aging bounds the wait | property | with aging a low-priority request waits at most about $P A$ steps; without, it starves | fairness under load |
| preemption keeps outputs | fault, differential | a 14-block pool forces preemption; greedy outputs equal the roomy run | memory pressure never changes answers |
| batched greedy equals single | differential | four requests batched give each one's alone tokens | batching is invisible to users |
| abort frees blocks | unit | abort of running and waiting requests returns every block | client disconnects (L10.5) |
| decodes count against the budget | boundary | three decodes leave room for 5 tokens: a 6-token prompt waits | the budget bounds every step |
| a stop finishes with `Stop` | unit | an end condition finishes with `Stop` at once | EOS and stop strings |
| admission refuses what can never run | boundary | `Empty`, `TooLong`, `NeverFits`, `QueueFull` | 400 and 429 answers instead of a stuck queue |

These cases are easiest to check with a fake model whose next token is a known function of the sequence, so expected outputs are known in advance and the scheduler is exercised without real forwards.

## 5. The solution, step by step

### 5.1 From first principles

1. **The state of a request is two numbers on top of its tokens:** $n_r$ (tokens so far) and $c_r$ (positions whose KV is computed). Everything the scheduler decides follows from $n_r - c_r$: 1 means decode, more means prefill, and after a preemption $c_r = 0$ means "recompute everything".
2. **The plan for one step is a list of chunks and decodes under two caps**, $\beta$ tokens and $S_{\max}$ sequences (section 2.1). The scheduler only plans; the engine runs the forward and reports sampled tokens back through `on_step`, which is the only place $c_r$ and $n_r$ advance.
3. **Running requests have priority over new ones** (phase 1 before phase 3), because their users are mid-stream. Within running requests, the order is rank, and the lowest-ranked is the one to sacrifice.
4. **Memory is checked at the moment it is needed:** `append_slot` before each decode, `allocate` for the whole sequence at admission. Both are all-or-nothing, so a failure leaves the pool unchanged and the scheduler can preempt and retry.
5. **A fixed heap key gives aging for free** (section 2.4): $P A - a$ never changes for a waiting request, so the heap is built once per push and never re-sorted. Preemption re-enqueues with the original arrival, so a victim keeps its age.
6. **Preemption by recompute needs no extra state:** free the blocks, set $c_r = 0$, keep the tokens. Because the runner is chunk-invariant, the recomputed KV equals the original, and the next token is the same.
7. **Requests that can never complete must be refused at `add`**, or they would cycle between admission and preemption forever. That is the `NeverFits` check on the longest possible sequence.

### 5.2 The code, annotated

**The waiting key and the rank.** `wait_order` sorts higher rank first, then earlier arrival. `rank` implements the three cases: FCFS (all 0, so arrival decides), priority without aging, and priority with aging as the fixed key.

```rust
fn wait_order(a: &WaitKey, b: &WaitKey) -> Ordering {
    b.rank.cmp(&a.rank).then(a.arrival.cmp(&b.arrival))      // higher rank first, then FIFO
}

    fn rank(&self, priority: i32, arrival: u64) -> i64 {
        match self.cfg.policy {
            SchedPolicy::Fcfs => 0,                                         // arrival alone decides
            SchedPolicy::Priority if self.cfg.aging_steps == 0 => priority as i64,
            SchedPolicy::Priority => priority as i64 * self.cfg.aging_steps as i64 - arrival as i64, // P*A - a
        }
    }
```

In the aging example, the old priority-0 request has key 0 and a priority-10 request arriving at step 41 has key $40 - 41 = -1$, so the old one sorts first.

**`add` refuses early.** The `NeverFits` check uses the longest sequence the request could reach, capped by the context, against the *whole* pool, not the free part: a request that fits the pool will eventually run; one that does not never will.

```rust
    pub fn add(&mut self, req: Request) -> Result<RequestId, AdmitError> {
        if req.prompt.is_empty() || req.max_tokens == 0 { return Err(AdmitError::Empty); }
        if req.prompt.len() + 1 > self.cfg.max_model_len {          // no room for even one token
            return Err(AdmitError::TooLong { tokens: req.prompt.len(), limit: self.cfg.max_model_len });
        }
        let longest = (req.prompt.len() + req.max_tokens).min(self.cfg.max_model_len);
        let need = longest.div_ceil(self.blocks.block_tokens());
        if need > self.blocks.total_blocks() {                       // would be preempted forever
            return Err(AdmitError::NeverFits { blocks: need, total: self.blocks.total_blocks() });
        }
        if self.waiting.len() >= self.cfg.max_waiting { return Err(AdmitError::QueueFull); } // -> 429
        // ... new Seq { tokens: prompt, computed: 0, arrival: self.step, ... }; enqueue ...
    }
```

**`schedule`, phase 1: decodes with preemption.** Running requests are sorted by rank so the last element is the first victim. For each decoding request, `append_slot` is retried after each preemption; when the request itself is the lowest-ranked left, it preempts itself.

```rust
        let mut order: Vec<RequestId> = self.running.clone();
        order.sort_by(/* wait_order on (rank, arrival) */);          // highest rank first
        let mut i = 0;
        while i < order.len() {
            let id = order[i];
            let s = &self.seqs[&id];
            if s.tokens.len() - s.computed != 1 { i += 1; continue; } // prefilling: phase 2
            if budget == 0 { break; }
            let total = s.tokens.len();
            loop {
                if self.blocks.append_slot(id, total).is_ok() {       // room for position n_r - 1
                    out.decode.push(id);
                    budget -= 1;                                      // a decode costs one token
                    break;
                }
                match order.len() - 1 > i {
                    true => {                                         // someone ranks lower: evict them
                        let victim = order.pop().expect("a lower-ranked request");
                        self.preempt(victim, &mut out);
                    }
                    false => {                                        // nobody lower: give way ourselves
                        self.preempt(id, &mut out);
                        order.remove(i);
                        break;
                    }
                }
            }
            if out.decode.last() == Some(&id) { i += 1; }
        }
```

**Preemption** frees the blocks, zeroes `computed`, and re-enqueues with the same arrival. The tokens, including those generated so far, stay, so the later prefill recomputes the whole sequence.

```rust
    fn preempt(&mut self, id: RequestId, out: &mut ScheduleOutput) {
        self.running.retain(|&r| r != id);
        let s = self.seqs.get_mut(&id).expect("running id");
        let computed = std::mem::take(&mut s.computed);              // c_r = 0: recompute from scratch
        let toks = s.tokens[..computed].to_vec();
        self.blocks.release(id, &toks);                              // every block goes back
        self.enqueue(id);                                            // same arrival: keeps its age
        out.preempted.push(id);
        self.stats.preemptions += 1;
    }
```

**Phase 2 and phase 3.** Prefills in progress ask the policy for a chunk. Admission peeks the head of the heap, tries to allocate the whole sequence, and **stops** at the first failure instead of skipping to a smaller request.

```rust
        // 3. admissions
        while self.running.len() < self.cfg.max_seqs && budget > 0 {
            let Some(&key) = self.waiting.peek() else { break };
            let id = key.id;
            let tokens = self.seqs[&id].tokens.clone();             // prompt plus any generated tokens
            let Ok(a) = self.blocks.allocate(id, &tokens) else { break }; // head-of-line: no skipping
            self.waiting.pop();
            let s = self.seqs.get_mut(&id).expect("waiting id");
            s.handle = None;
            s.computed = a.cached_tokens.min(tokens.len() - 1);     // always feed >= 1 token for logits
            self.running.push(id);
            let remaining = tokens.len() - s.computed;
            let n = self.policy.chunk_len(remaining, budget, out.is_empty());
            if n == 0 { break; }                                    // admitted, prefills next step
            out.prefill.push(Chunk { id, start: s.computed, len: n, last: n == remaining });
            budget -= n.min(budget);
        }
```

`WholePrompt::chunk_len` returns `remaining` if it fits the budget, or if the step is empty (`alone`), so a prompt longer than the budget still runs, in a step of its own.

**`on_step`** is where state advances. First every chunk and decode of the last plan marks its positions computed; then each sampled token is appended only for requests whose sequence is now fully computed (a mid-prefill chunk or a preempted request is ignored); then the finish conditions are checked.

```rust
    pub fn on_step(&mut self, outputs: &[StepOutput]) -> Vec<RequestEvent> {
        let last = std::mem::take(&mut self.last);
        for c in &last.prefill {
            if let Some(s) = self.seqs.get_mut(&c.id) { s.computed = (c.start + c.len).max(s.computed); }
        }
        for id in &last.decode {
            if let Some(s) = self.seqs.get_mut(id) { s.computed = s.tokens.len(); }
        }
        let mut events = Vec::new();
        for o in outputs {
            let Some(s) = self.seqs.get_mut(&o.id) else { continue };
            if s.handle.is_some() || s.computed != s.tokens.len() { continue; } // not sampled this step
            s.tokens.push(o.token);
            events.push(RequestEvent::Token { id: o.id, token: o.token });
            let generated = s.tokens.len() - s.prompt_len;               // generated only, never the prompt
            let reason = if o.stop { Some(FinishReason::Stop) }
                else if generated >= s.max_tokens || s.tokens.len() >= self.cfg.max_model_len { Some(FinishReason::Length) }
                else { None };
            if let Some(reason) = reason {
                let s = self.seqs.remove(&o.id).expect("known id");
                self.running.retain(|&r| r != o.id);
                self.blocks.release(o.id, &s.tokens[..s.computed]);       // blocks back at once
                self.stats.finished += 1;
                events.push(RequestEvent::Finished { id: o.id, reason });
            }
        }
        events
    }
```

**`abort`** removes the request wherever it is: from the heap by its handle (O(1) lazy deletion) if waiting, or from the running list with its blocks released if running.

**`FreeListBlocks`** keeps its free ids reversed so `pop` hands out the lowest first (deterministic block tables), and `grow` checks that enough blocks are free before draining any.

Full solution: [`sched.rs`](../../../../course/solutions/rust/crates/tl-engine/src/sched.rs).

### 5.3 Why this solution

- **Cost.** `schedule` sorts the running set, $O(R \log R)$ with $R \le S_{\max}$, and pops at most $S_{\max}$ heap entries, $O(\log W)$ each; `add` and `abort` are $O(\log W)$ and $O(1)$. The scheduler is negligible next to a forward pass.
- **Why recompute rather than swap?** Swap needs a second memory tier, a copy engine, and bookkeeping for where each block lives. Recompute costs a prefill of the victim's sequence, which is cheap relative to decode on a CPU, and with a chunk-invariant runner it cannot change the output.
- **Why head-of-line admission rather than best fit?** Best fit raises utilization but lets a stream of small requests starve a large one indefinitely. Stopping at the head makes the wait of every request bounded by the requests ahead of it.
- **Why a fixed key instead of recomputing priorities each step?** Recomputing would be $O(W \log W)$ per step and would need floating-point effective priorities; the fixed integer key gives exactly the same order (the $t$ terms cancel).
- **Tradeoff accepted:** the victim is chosen by rank, not by how much memory it would free. Evicting the largest request would free more blocks at once, but would punish long requests and waste more already-done work.

### 5.4 Explain it back

1. Why does aging not require re-sorting the waiting queue?
   <details><summary>Answer</summary>Every waiting request ages at the same rate, so comparing $P_r + (t - a_r)/A$ with $P_s + (t - a_s)/A$ is the same as comparing $P_r A - a_r$ with $P_s A - a_s$: $t$ cancels. That key is fixed at push time.</details>
2. Why does a preempted request produce the same output after it is recomputed?
   <details><summary>Answer</summary>Its tokens are kept, its sampler state lives with the request, and the runner is chunk-invariant, so prefilling the whole sequence again produces the same K and V and the same logits for the next token.</details>
3. Why does `add` compare the longest possible sequence with the whole pool rather than with the free blocks?
   <details><summary>Answer</summary>Free blocks change as other requests finish; a request that fits the whole pool will eventually be admitted. One that needs more than the pool holds can never finish: it would be admitted, grow, be preempted, and repeat forever.</details>
4. Why does admission set `computed` to at most `tokens.len() - 1`?
   <details><summary>Answer</summary>The next token is sampled from the logits of the last fed position. If every position were already cached (a full prefix hit), the step would feed nothing and produce no logits, so at least the last token is always recomputed.</details>
5. Why does `on_step` skip tokens for requests with `computed != tokens.len()`?
   <details><summary>Answer</summary>Those requests were not sampled this step: they were mid-prefill (a chunk that did not reach the end) or were preempted while planning. Appending a token for them would add a token that the engine did not actually sample from their sequence.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| Ordering the waiting heap lowest rank first | low priority runs first | `wait_order` compares `b.rank` with `a.rank` |
| Ranking by priority alone under `Priority` | a steady high-priority stream starves the rest | rank $= P A - a$ when `aging_steps > 0` |
| Preempting without releasing the victim's blocks | the pool drains under pressure; requests never finish | `preempt` calls `blocks.release` |
| Recomputing from the prompt only | a preempted request repeats or loses its answer | `tokens` keeps generated tokens; admission allocates for all of them |
| Admitting past `max_seqs` | steps grow without bound; memory per step explodes | admission loop checks `running.len() < max_seqs` |
| Aborting a running request without releasing its blocks | every disconnect leaks KV | `abort` releases when the request has no heap handle |
| Queueing a request larger than the pool | it is preempted forever and blocks the queue | `NeverFits` at `add` |
| Not counting decodes against the budget | steps exceed `max_batch_tokens`; latency spikes | `budget -= 1` per decode |
| Counting generated tokens off by one | every answer has one token too many | `generated = tokens.len() - prompt_len` after the push |
| A free list that hands out arbitrary ids | block tables differ from run to run; harder to debug | ids reversed once so `pop` gives the lowest |
| Skipping ahead in the queue when the head does not fit | big requests starve behind small ones | `let Ok(a) = allocate(...) else { break }` |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L10.1` | the runner executes each plan; `SchedPolicy` comes from `EngineConfig` |
| Back | `ds.06` | `LazyHeap` is the waiting queue; its handles make abort O(1) |
| Back | `L8.3` | the paged KV semantics in Python, block by block |
| Forward | `L10.3` | `Chunked` is a `PrefillPolicy`; `plan` turns a `ScheduleOutput` into a mixed batch |
| Forward | `L10.4` | `BlockManager` is a `BlockSpace` with a prefix cache |
| Forward | `L10.5` | the engine loop: `schedule`, forward, sample, `on_step`; `AdmitError::QueueFull` becomes 429 |

## Going further

| This solution | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| recompute only | vLLM swap preemption | victims' blocks moved to host memory and back | `vllm` v0 `scheduler.py`, `_preempt_by_swap` |
| one queue with aging | multi-level feedback queues, SLO-aware schedulers | separate queues per tenant tier, deadlines | [Sarathi-Serve](https://arxiv.org/abs/2403.02310), [Llumnix](https://arxiv.org/abs/2406.03243) |
| head-of-line admission | best-fit admission | smaller requests fill gaps when the head waits | the vLLM `max_num_batched_tokens` discussion |
| a fixed budget | adaptive budgets from measured step time | targets a TPOT instead of a token count | `L10.7` metrics, `load.01` |
