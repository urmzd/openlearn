# ADR-0002: The capstone tokenizer is byte-level BPE

## Status

Accepted (2026-10-09)

## Context

The capstone model needs one tokenizer for training, serving, and the
engine's request path. Two of our tokenizers can be trained on the capstone
sample: byte-level BPE (L1.2) and Unigram (L1.4). The vocabulary size sets
the embedding table, which is the largest single tensor of a small model,
and the bytes per token, which sets how much text one context window holds.
We compared the two at the same vocabulary size on the same sample with the
same model and seed (the `tokenizer` ablation in
docs/capstone/c1/report.json, smoke tier: vocabulary 512, 32 held-out
documents).

## Decision

We will train byte-level BPE at vocabulary 512 for the smoke tier (4096 for
the full and short tiers). On the paired held-out documents, Unigram's bits
per byte minus BPE's is 0.032 on average, 95% CI (0.014, 0.059): BPE is
better and the interval excludes zero.

## Consequences

- Good: one tokenizer family across Python, the Rust engine (`tl-tok`), and
  the HF `tokenizer.json` export.
- Good: the measured gain is in bits per byte, which no tokenizer can game.
- Bad: the comparison is one seed at the smoke tier; the full tier reruns it.

## Alternatives considered

| Option | Why we did not choose it |
|---|---|
| Unigram at 512 | worse by 0.032 bits per byte, CI excludes zero |
| a byte vocabulary (256) | every character costs a position of context |
