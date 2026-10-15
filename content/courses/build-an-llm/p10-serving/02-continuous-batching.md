<!-- ol:module L10.2 -->
# Continuous batching scheduler with priority and preemption

## Overview

| | |
|---|---|
| **Module** | `L10.2` · build · Rust · Pass 7 · 10 to 14 h |
| **You build** | `rust/crates/tl-engine/src/sched.rs`: the `BlockSpace` seam and its simplest implementation `FreeListBlocks`, the `PrefillPolicy` seam with `WholePrompt`, and `Scheduler`: admission by free blocks, a waiting queue ordered by priority with aging, a running batch re-formed every step, preemption by recompute, abort |
| **Contract** | no Rust trait contract file yet: section 4 pins the API; `X-TL-Priority` in [`openai-subset.v1.yaml`](../../../../course/contracts/openapi/openai-subset.v1.yaml) and `PrefillRequest.priority` in [`engine.proto`](../../../../course/contracts/proto/tl/engine/v1/engine.proto) feed `Request.priority` |
| **Tests** | `course/tests/rust/l10_2.rs`, 11 tests (what they check: section 4), most with a fake model whose outputs are known in advance |
| **Needs** | `L10.1` the runner, `SchedPolicy`, and the KV pool ([chapter](01-model-runner-and-sampler.md)) · `ds.06` the lazy-deletion heap ([chapter](../../../tracks/algorithms/16-systems-data-structures/06-binary-heap-lazy-deletion.md)) · reading: `lang.04` Rust, `L8.3` paged KV in Python, `M05.1` KV memory accounting · or `--ref-deps` |
| **Used by** | `L10.3` (chunked prefill is a `PrefillPolicy`), `L10.4` (the block manager is a `BlockSpace`), `L10.5` (the engine loop calls `schedule` and `on_step` every step) |
| **Milestone** | `MS-L10` (64 concurrent requests complete, KV back at baseline) |
| **Optional depth** | [Yu et al. 2022, Orca: iteration-level scheduling](https://www.usenix.org/conference/osdi22/presentation/yu) (free); [Kwon et al. 2023, vLLM and PagedAttention](https://arxiv.org/abs/2309.06180) (free), section 4.5 on preemption; [vLLM scheduler](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py) (free) |

## Key Takeaways

- The batch is re-formed at **every step**: finished requests leave and waiting ones join, so no sequence waits for the longest one in its batch (`hand_example_schedule`).
- Admission is by **free blocks**: a request joins only when the pool can hold its whole sequence so far; the first request that does not fit stops admission, so big requests are not starved by small ones behind them (`admission_refuses_what_can_never_run`).
- Priority with aging needs no re-sorting: ranking by `priority + age / A` orders requests exactly like the fixed key `priority * A - arrival` (`aging_bounds_the_wait`).
- When a running request needs a block and none is free, the lowest-ranked running request is **preempted**: its blocks are freed and it is recomputed later from its tokens, with the same output (`preemption_recomputes_and_keeps_outputs`).
- Batched greedy output equals single-request output, because the kernels are batch-invariant (`batched_greedy_equals_single_request`).

## How to work this chapter

```bash
ol start L10.2               # stubs tl-engine/src/sched.rs
ol tests L10.2               # rung R0: read the catalog; the fake model is at the top of the test file
ol check L10.2
ol check L10.2 --ref-deps    # if L10.1 or ds.06 is not passing yet
```

Add `tl-ds` (a path dependency) to your `tl-engine/Cargo.toml` if your `L8.4` did not already: the waiting queue is your `LazyHeap`.

---

## 1. Why now

`L10.1` runs one step over any batch you give it. Something has to decide what goes into each step: which requests run, which wait, and what happens when memory runs out. Serving requests one after another wastes the hardware: a decode step for one sequence reads every weight to produce a single token, and reading the weights is the expensive part, so 16 sequences in one step cost little more than 1. **Static** batching (collect 16 requests, run them to the end together) wastes the other way: short requests finish and their slots sit empty until the longest one ends, and a new request waits for the whole batch. **Continuous** batching decides again at every step (Orca, 2022). This module is that decision.

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

because $t$ cancels. So the waiting queue is a heap on the **fixed** key $k_r = P_r A - a_r$ (higher first; equal keys by arrival), built once at push. It is your `ds.06` `LazyHeap`, whose handles also make `abort` of a waiting request O(1). With `Fcfs` every key is 0 and arrival decides. A preempted request goes back with its original arrival, so it keeps its age.

### 2.5 Preemption by recompute

When memory runs out mid-decode, something must give. vLLM's two options: **swap** the victim's blocks to host memory and back, or **recompute**: free its blocks and later prefill its whole sequence again (prompt plus every token generated so far). This engine recomputes: no second memory tier, and the recomputed K and V are bit-identical to the originals because the runner is chunk-invariant (`L10.1`), so the victim's output does not change. Its sampler state is untouched (its tokens and its generator stay with the request), so even seeded sampling continues where it stopped.

The victim is the **lowest-ranked** running request, at equal rank the newest: it has had the least service, so evicting it wastes the least work.

### 2.6 Two seams

`BlockSpace` is the scheduler's only view of memory (block size, totals, `allocate`, `append_slot`, `release`, `block_table`); `PrefillPolicy` is its only view of how prompts are cut (`chunk_len`). Both are traits so this module can be finished and tested before `L10.3` and `L10.4` exist, and so tests can drive the scheduler with a fake model and a plain free list.

## 3. Worked example by hand

Blocks of $B = 4$ (six of them, ids 0 to 5), budget $\beta = 8$, $S_{\max} = 2$, `Fcfs`. Requests: A has a 5-token prompt and `max_tokens` 3; B a 3-token prompt and 2; C a 2-token prompt and 1.

**Step 1.** Nothing runs, so phase 3 admits. A needs $\lceil 5/4 \rceil = 2$ blocks: ids 0 and 1; its whole prompt (5 tokens) fits the budget, which drops to 3. B needs 1 block: id 2; 3 tokens, budget 0. Two sequences run, so C waits. The plan: prefill A 0..5 and B 0..3, both last chunks (sampled). Each samples one token: A has 6 tokens, B has 4.

**Step 2.** Phase 1: A decodes (it needs room for 6 positions, which 2 blocks already give), then B (4 positions, 1 block). Budget 6 left, but $S_{\max} = 2$ blocks C again. B's second token reaches its `max_tokens`: B finishes with `Length`, and block 2 goes back.

**Step 3.** A decodes (7 positions); C is admitted into block 2 and prefilled (2 tokens). A's third token and C's first end both: everything is free again. This is `hand_example_schedule`.

**Aging.** With $A = 4$: a priority-0 request that arrived at step 0 has key $0 \cdot 4 - 0 = 0$; a priority-10 request arriving at step $t$ has key $40 - t$. From step $t = 41$ on, new priority-10 arrivals have keys below 0, so the old request is admitted first: aging bounded its wait at about $P \cdot A = 40$ steps (`aging_bounds_the_wait`).

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

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `hand_example_schedule` | unit | section 3, step by step: chunks, decodes, block ids, the finishes | the worked example |
| `free_list_is_all_or_nothing` | unit | lowest ids first; no partial allocation | admission never half-allocates |
| `every_request_finishes_and_no_block_leaks` | property | random arrivals with forced preemptions: exact outputs, caps respected, every block free at the end | the engine runs for weeks |
| `priority_runs_first` | unit | `Priority` admits by priority, `Fcfs` by arrival | `X-TL-Priority` from the gateway's tenant tiers |
| `aging_bounds_the_wait` | property | with aging a low-priority request waits at most about $P A$ steps; without, it starves | fairness under load |
| `preemption_recomputes_and_keeps_outputs` | fault, differential | a 14-block pool forces preemption; greedy outputs equal the roomy run | memory pressure never changes answers |
| `batched_greedy_equals_single_request` | differential | four requests batched give each one's alone tokens | batching is invisible to users |
| `abort_frees_blocks_and_drops_waiting` | unit | abort of running and waiting requests returns every block | client disconnects (L10.5) |
| `decodes_count_against_the_budget` | boundary | three decodes leave room for 5 tokens: a 6-token prompt waits | the budget bounds every step |
| `stop_finishes_with_stop` | unit | an end condition finishes with `Stop` at once | EOS and stop strings |
| `admission_refuses_what_can_never_run` | boundary | `Empty`, `TooLong`, `NeverFits`, `QueueFull` | 400 and 429 answers instead of a stuck queue |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| Ordering the waiting heap lowest rank first | low priority runs first | `priority_runs_first` (mutant `s01`) |
| Ranking by priority alone under `Priority` | a steady high-priority stream starves the rest | `aging_bounds_the_wait` (mutant `s02`) |
| Preempting without releasing the victim's blocks | the pool drains under pressure; requests never finish | `every_request_finishes_and_no_block_leaks` (mutant `s03`) |
| Recomputing from the prompt only | a preempted request repeats or loses its answer | `preemption_recomputes_and_keeps_outputs` (mutant `s04`) |
| Admitting past `max_seqs` | steps grow without bound; memory per step explodes | `hand_example_schedule` (mutant `s05`) |
| Aborting a running request without releasing its blocks | every disconnect leaks KV | `abort_frees_blocks_and_drops_waiting` (mutant `s06`) |
| Queueing a request larger than the pool | it is preempted forever and blocks the queue | `admission_refuses_what_can_never_run` (mutant `s07`) |
| Not counting decodes against the budget | steps exceed `max_batch_tokens`; latency spikes | `decodes_count_against_the_budget` (mutant `s08`) |
| Counting generated tokens off by one | every answer has one token too many | `every_request_finishes_and_no_block_leaks` (mutant `s09`) |
| A free list that hands out arbitrary ids | block tables differ from run to run; harder to debug | `free_list_is_all_or_nothing` (mutant `s10`) |
| Skipping ahead in the queue when the head does not fit | big requests starve behind small ones | `hand_example_schedule` |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L10.1` | the runner executes each plan; `SchedPolicy` comes from `EngineConfig` |
| Back | `ds.06` | `LazyHeap` is the waiting queue; its handles make abort O(1) |
| Back | `L8.3` | the paged KV semantics in Python, block by block |
| Forward | `L10.3` | `Chunked` is a `PrefillPolicy`; `plan` turns a `ScheduleOutput` into a mixed batch |
| Forward | `L10.4` | `BlockManager` is a `BlockSpace` with a prefix cache |
| Forward | `L10.5` | the engine loop: `schedule`, forward, sample, `on_step`; `AdmitError::QueueFull` becomes 429 |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| recompute only | vLLM swap preemption | victims' blocks moved to host memory and back | `vllm` v0 `scheduler.py`, `_preempt_by_swap` |
| one queue with aging | multi-level feedback queues, SLO-aware schedulers | separate queues per tenant tier, deadlines | [Sarathi-Serve](https://arxiv.org/abs/2403.02310), [Llumnix](https://arxiv.org/abs/2406.03243) |
| head-of-line admission | best-fit admission | smaller requests fill gaps when the head waits | the vLLM `max_num_batched_tokens` discussion |
| a fixed budget | adaptive budgets from measured step time | targets a TPOT instead of a token count | `L10.7` metrics, `load.01` |
