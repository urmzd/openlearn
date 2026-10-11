<!-- ol:module ds.90 -->
# Kahan sum

## Overview

| | |
|---|---|
| **Module** | `ds.90` · build · Rust · Pass 1 · 20 min |
| **You build** | `rust/crates/tl-demo/src/acc.rs`: `kahan_sum` (and the crate root) |
| **Contract** | none (Rust and Go fixtures check through the course tests) |
| **Tests** | `course/tests/rust/ds_90.rs` (what they check: section 4) |
| **Needs** | nothing |
| **Used by** | `ds.91` |
| **Optional depth** | none: this is a harness fixture |

## Key Takeaways

- Kahan summation carries the rounding error $c$ of each addition into the next term: $y = x - c$, $t = s + y$, $c = (t - s) - y$, $s = t$.

## How to work this chapter

```bash
ol start ds.90
ol tests ds.90
ol check ds.90
```

---

## 1. Why now

Small terms added to a large running sum vanish one at a time; the mean in ds.91 needs them kept.

## 2. Principles

Kahan summation carries the rounding error $c$ of each addition into the next term: $y = x - c$, $t = s + y$, $c = (t - s) - y$, $s = t$.

| Symbol | Meaning | Type / shape |
|---|---|---|
| $s$ | running sum | `f64` |
| $c$ | carried error | `f64` |
| $t$ | new sum | `f64` |

## 3. Worked example by hand

1 + 2 + 3: c stays 0 because every partial sum is exact, so the result is 6.

## 4. The interface

```rust
pub fn kahan_sum(xs: &[f64]) -> f64
```

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `hand_example` | unit | the chapter's worked example | ds.91 |
| `compensates_small_terms` | differential | ten terms of 1e-16 vanish when added to 1 | ds.91 |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| dropping the carried error | small terms vanish | `compensates_small_terms` |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Forward | `ds.91` | mean over the compensated sum |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| kahan sum | the real course module this fixture stands in for | everything | `course/DESIGN.md` |
