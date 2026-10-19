# Drill: KV v2 migration

## Overview

| | |
|---|---|
| **Module** | `ops.04` · drill · ops, docs · Pass 11 · 1 to 2 h |
| **Solution** | the model runbook `docs/runbooks/KvV2Migration.md`, written out with its reasoning in section 5 |
| **Needs** | craft.13 |
| **Used by** | no code call site: a drill, one of the Pass 11 operations drills |

## Key Takeaways

- KV format v2 stores fp8 (E4M3) payloads with one f32 scale per (layer, K or V, head) slab; v1 stores f16. The **block hash** names token ids only, so it is the same in both formats.
- A v2 reader reads both versions; a v1-only reader refuses version 2 with `FAILED_PRECONDITION`. So **readers upgrade before writers**, and a rollback flips the writer back first.
- The migration-specific signal is `FAILED_PRECONDITION` on `HasBlocks` and `PushKv`, not the generic error rate: a mixed fleet can fail transfers while health checks stay green.

## How to read this chapter

This chapter is a worked drill. Sections 1 to 3 state the risk, the principles, and the drill scenario. Section 4 lists what a correct runbook must cover. Section 5 derives the procedure from first principles, gives the model runbook, and explains why each step is the right one. Try to write the procedure yourself from sections 2 and 3 before reading section 5.

---

## 1. Why now

A format change affects every prefill/decode boundary and stored block. A partial rollout can mix fp8 payloads with f16 readers.

## 2. Principles

Before changing writers, verify v1/v2 negotiation, v2 scale metadata, peer capabilities, and spare capacity. Protect the no-loss invariant and keep a tested downgrade.

## 3. Worked example

Canary one engine pair on a known fixture. Confirm v2 hashes, transfer and reconstruct blocks, then raise v2 traffic while watching negotiation failures and request errors. Roll back the writer flag if any peer lacks support.

## 4. What a correct runbook must cover

| Case | What the runbook must contain | Why it matters |
|---|---|---|
| Owner and preconditions | who owns the change; every decode reader runs a build that reads v2; spare capacity for a canary pair | an unsupported reader turns a writer flip into failed transfers |
| Canary gate | one prefill/decode pair on `kv_format = 2`, a fixed prompt set, and numeric thresholds to advance | a decision rule written before the data arrives |
| Evidence for v1 and v2 | equal block hashes across formats, the envelope's version and dtype (1 and f16, 2 and E4M3), per-slab scale | proves the bytes are what the contract says |
| Rollback | the trigger (any `FAILED_PRECONDITION`, error-rate or quality threshold) and the action (writers back to `kv_format = 1`) | the old format must be restorable before the canary advances |
| Verification | transfers, dedup hit rate, request errors, and output quality equal to v1 on the fixed prompts | done means measured, not assumed |

## 5. The solution, step by step

### 5.1 From first principles

1. **Who must change first?** The contract says a v2 reader reads v1 and v2, while a v1-only reader refuses v2. A block written in v2 and pushed to a v1 reader fails. So every reader must support v2 before any writer emits it; the precondition is fleet-wide reader support.
2. **What stays the same?** The block hash is computed over token ids only, so prefix-cache lookups (`HasBlocks`) and dedup keep working across formats. A canary can prove this directly: the same prompt yields the same hashes under both formats.
3. **What changes in the bytes?** The envelope's `version` (1 to 2) and `dtype` (1 = f16 to 3 = E4M3), and the payload gains `f32 [L][2][Hkv]` scales with `scale = amax / 448` per slab, so no value saturates. These are the evidence fields to record.
4. **What is the failure signal?** A format mismatch fails `HasBlocks`/`PushKv` with `FAILED_PRECONDITION`; a corrupt envelope with `DATA_LOSS`. Generic HTTP error rates may hide these, because a failed transfer can fall back to local prefill, which shows up only as latency.
5. **How to roll back?** Flip writers back to `kv_format = 1`. Readers that support v2 still read the v1 blocks, so rollback is safe at any time, which is exactly why readers go first.

### 5.2 The model runbook

`docs/runbooks/KvV2Migration.md`:

````markdown
# Runbook: KV format v2 migration

Owner: the serving on-call (`<system>-engine`). Change ticket: the craft.13 migration.

## Preconditions

- Every decode engine runs a build whose reader accepts envelope versions 1 and 2
  (`kubectl -n <system> get pods -l app.kubernetes.io/component=engine -o jsonpath='{..image}'`).
- `[engine].kv_format` is 1 everywhere; capacity for one extra prefill/decode pair.
- The SLO burn rates are below 1 and no other change is in flight.

## Canary

1. Deploy one prefill/decode pair with `kv_format = 2`, routed a fixed prompt set.
2. Evidence: for each prompt, the v1 and v2 block hashes are equal; the envelope
   header shows version 2, dtype 3 (E4M3); every slab carries a positive scale
   (`amax / 448`, 1.0 for an all-zero slab).
3. Gate to advance (held 15 minutes): zero `FAILED_PRECONDITION` or `DATA_LOSS`
   on `HasBlocks`/`PushKv`; request error rate and TTFT within the v1 baseline;
   greedy outputs on the fixed prompts within the agreed tolerance of v1.
4. Advance in steps (one pair, 25%, 100% of writers), repeating the gate.

## Rollback

Trigger: any `FAILED_PRECONDITION` (a reader without v2), any `DATA_LOSS`, or a
gate threshold missed. Action: set writers back to `kv_format = 1` with
`helm upgrade`; v2-capable readers keep reading both formats, so in-flight
blocks are safe. Confirm transfers succeed and record the trigger.

## Verification

Transfers and dedup hit rate equal to before; no format errors for 24 hours;
the decision, evidence, and owner recorded in the change ticket.
````

### 5.3 Why this solution

- **Readers first, writers last** makes every intermediate state valid: at no point does a writer produce a block some reader cannot read, so rollback is always a single flag flip.
- **Hash equality as evidence** is cheap and decisive: it proves the prefix cache and dedup are unaffected without comparing tensors.
- **Why not flip all engines at once?** A single reader without v2 support would fail every transfer it receives, and fp8 quality effects would reach all traffic before anyone measured them.

### 5.4 Explain it back

1. Why must readers be upgraded before writers, and not the other way round?
   <details><summary>Answer</summary>A v2 reader reads v1 and v2, but a v1-only reader refuses v2 with `FAILED_PRECONDITION`. Upgrading readers first means every block any writer emits can be read, so each step and its rollback are safe.</details>
2. Why do block hashes not change between formats?
   <details><summary>Answer</summary>The hash is computed over token ids, chained with the parent hash; it never covers the payload bytes. Two formats of the same prefix name the same content.</details>
3. Why is the scale `amax / 448`?
   <details><summary>Answer</summary>448 is the largest finite E4M3 value. Dividing by `amax / 448` maps the slab's largest magnitude to 448, so no element saturates, while using the full range for precision.</details>
4. Why watch `FAILED_PRECONDITION` instead of the request error rate?
   <details><summary>Answer</summary>A failed transfer can degrade to slower serving rather than an HTTP error, so the generic rate can stay green while transfers fail. The gRPC code names the migration's failure exactly.</details>

## 6. Pitfalls

| # | Pitfall | Symptom | How the solution avoids it |
|---|---|---|---|
| 1 | Writers flipped before every reader supports v2 | transfers to old readers fail with `FAILED_PRECONDITION`; TTFT rises as prefill is redone | the precondition requires v2-capable readers fleet-wide before the canary |
| 2 | Watching only request errors | a mixed fleet fails transfers while health checks and error rates stay green | the gate watches `FAILED_PRECONDITION` and `DATA_LOSS` on the KV rpcs |
| 3 | No recorded rollback trigger or owner | the canary advances on a feeling; the next on-call cannot tell whether the change is contained | the runbook names the owner, the trigger, and the action before the canary starts |

## 7. Where it's used next

| Direction | Module | Connection |
|---|---|---|
| Back | `craft.13` | A prerequisite: the contract or evidence this runbook relies on. |
| Back | `ops.03` | A prerequisite: the contract or evidence this runbook relies on. |

## Going further

| Resource | What to inspect |
|---|---|
| [Incident drill catalog](README.md) | Compare this inject with the adjacent failure drills and use the same containment and verification fields. |
