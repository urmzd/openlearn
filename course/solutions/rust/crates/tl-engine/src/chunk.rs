//! Chunked prefill (L10.3): long prompts are processed in pieces that share
//! each step with the running decodes.
//!
//! A step's cost grows with the tokens it processes. Prefilling a 2,000
//! token prompt in one step makes that step slow, and every running request
//! waits for it before its next token: time per output token (TPOT) spikes.
//! Splitting the prompt into chunks of at most `chunk` tokens, and filling
//! each step's token budget with the decodes first and prefill chunks after
//! (a **mixed batch**), bounds every step's work, so decodes keep their pace
//! while the prompt still advances each step. The price is a slightly later
//! first token (TTFT) for the long prompt, which now needs several steps.
//!
//! Correctness rests on chunk invariance: the logits after the last chunk
//! agree with a whole prefill within floating-point tolerance, because
//! positions are absolute and each chunk's K and V land in the same blocks.
//!
//! This file holds the policy the scheduler consults ([`Chunked`]) and the
//! assembly of one step's mixed batch from the scheduler's plan ([`plan`]),
//! which the engine loop (L10.5) runs.

use crate::forward::{ForwardBatch, ForwardSeq};
use crate::sched::{BlockSpace, PrefillPolicy, RequestId, ScheduleOutput, Scheduler};

/// Prefill at most `chunk` tokens of a request per step, within the budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Chunked {
    pub chunk: usize,
}

impl PrefillPolicy for Chunked {
    fn chunk_len(&self, remaining: usize, budget: usize, _alone: bool) -> usize {
        remaining.min(self.chunk.max(1)).min(budget)
    }
}

/// One step ready to run: the batch for `ModelRunner::forward`, and which
/// logits rows to sample, as (row, request).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StepPlan<'a> {
    pub batch: ForwardBatch<'a>,
    pub sample: Vec<(usize, RequestId)>,
}

/// The mixed batch of a schedule: one sequence per decode (its newest token
/// at position n - 1) and one per prefill chunk (positions `start..start +
/// len`). Every decode row is sampled; a chunk's row only when the chunk
/// ends the sequence (`last`). Rows follow the order decodes, then chunks.
pub fn plan<'a, B: BlockSpace>(s: &'a Scheduler<B>, out: &ScheduleOutput) -> StepPlan<'a> {
    let mut p = StepPlan::default();
    for &id in &out.decode {
        let Some(toks) = s.tokens(id) else { continue };
        let n = toks.len();
        p.sample.push((p.batch.seqs.len(), id));
        p.batch.seqs.push(ForwardSeq { tokens: &toks[n - 1..], start: n - 1, blocks: s.blocks().block_table(id) });
    }
    for c in &out.prefill {
        let Some(toks) = s.tokens(c.id) else { continue };
        if c.last {
            p.sample.push((p.batch.seqs.len(), c.id));
        }
        p.batch.seqs.push(ForwardSeq { tokens: &toks[c.start..c.start + c.len], start: c.start, blocks: s.blocks().block_table(c.id) });
    }
    p
}

/// The chunk boundaries of a prompt of `n` tokens prefilled `chunk` at a
/// time with nothing else in the batch: (start, len) pairs.
pub fn split(n: usize, chunk: usize) -> Vec<(usize, usize)> {
    let c = chunk.max(1);
    (0..n).step_by(c).map(|s| (s, c.min(n - s))).collect()
}
