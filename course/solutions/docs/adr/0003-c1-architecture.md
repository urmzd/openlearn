# ADR-0003: The capstone architecture keeps GQA attention and a dense MLP

## Status

Accepted (2026-10-09)

## Context

Two architecture choices of the capstone model were measured as ablations
at the smoke tier (docs/capstone/c1/report.json): attention, GQA against
multi-head latent attention (MLA) at equal KV cache bytes per token (1024
each), and the MLP, a dense SwiGLU against a top-1 mixture of four experts
at equal active parameters (791,680 against 793,728). Each arm trained once
for 200 steps with seed 0; the comparison is paired over 32 held-out
documents. The engine (L10.1) and the Rust KV manager (L10.4) serve GQA today; MLA
needs the latent cache of L8.2 and weight absorption, and an MoE needs
expert dispatch.

## Decision

We will keep GQA and a dense MLP. MLA minus GQA is -0.011 bits per byte,
95% CI (-0.042, 0.008); MoE minus dense is 0.013, 95% CI (-0.000, 0.033).
Both intervals include zero: neither alternative is measurably better at
this tier, and both cost engine work.

## Consequences

- Good: the served model uses the attention and MLP our engine already runs
  with Candle tensor operations.
- Bad: if MLA's small advantage is real, we leave it on the table; the full
  tier repeats the ablation with more documents.
- Bad: an MoE with four experts holds four times the MLP weights for the
  same active compute; we would pay that memory for no measured gain.

## Alternatives considered

| Option | Why we did not choose it |
|---|---|
| MLA (rank 112, rope 16) | tie at equal KV bytes; needs the latent cache in the engine |
| top-1 MoE of 4 experts | tie at equal active parameters; four times the MLP memory |
