<!-- ol:module ds.91 -->
# Mean over the compensated sum

## Overview

| | |
|---|---|
| **Module** | `ds.91` · build · Rust · Pass 2 · 20 min |
| **You build** | `rust/crates/tl-demo/src/mean.rs`: `mean` |
| **Contract** | none (Rust and Go fixtures check through the course tests) |
| **Tests** | `course/tests/rust/ds_91.rs` (what they check: section 4) |
| **Needs** | `ds.90` |
| **Used by** | `craft.90` |
| **Optional depth** | none: this is a harness fixture |

## Key Takeaways

- The mean is $\bar{x} = \frac{1}{n}\sum_i x_i$, undefined for $n = 0$.

## How to work this chapter

```bash
ol start ds.91
ol tests ds.91
ol check ds.91
```

---

## 1. Why now

The demo report needs an average that does not lose small contributions.

## 2. Principles

The mean is $\bar{x} = \frac{1}{n}\sum_i x_i$, undefined for $n = 0$.

| Symbol | Meaning | Type / shape |
|---|---|---|
| $n$ | number of values | `usize` |
| $\bar{x}$ | the mean | `Option<f64>` |

## 3. Worked example by hand

[1, 2, 3, 4]: the sum is 10 and n is 4, so the mean is 2.5.

## 4. The interface

```rust
pub fn mean(xs: &[f64]) -> Option<f64>
```

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `hand_example` | unit | the chapter's worked example | craft.90 |
| `empty_is_none` | boundary | the mean of nothing is undefined, not 0 and not NaN | craft.90 |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| returning 0 for an empty slice | a fake average | `empty_is_none` |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `ds.90` | kahan sum |
| Forward | `craft.90` | document the demo system |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| mean over the compensated sum | the real course module this fixture stands in for | everything | `course/DESIGN.md` |
