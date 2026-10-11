<!-- ol:module dur.90 -->
# Sum that skips NaN

## Overview

| | |
|---|---|
| **Module** | `dur.90` · build · Go · Pass 1 · 20 min |
| **You build** | `go/ds/demo/sum.go`: `Sum` |
| **Contract** | none (Rust and Go fixtures check through the course tests) |
| **Tests** | `course/tests/go/dur_90/` (what they check: section 4) |
| **Needs** | nothing |
| **Used by** | `dur.91` |
| **Optional depth** | none: this is a harness fixture |

## Key Takeaways

- NaN compares unequal to everything, including itself, so it must be tested with `math.

## How to work this chapter

```bash
ol start dur.90
ol tests dur.90
ol check dur.90
```

---

## 1. Why now

The Go side of the demo totals samples, and one NaN sample would poison every later total.

## 2. Principles

NaN compares unequal to everything, including itself, so it must be tested with `math.IsNaN`.

## 3. Worked example by hand

Sum([1, NaN, 2]) skips the NaN and returns 3.

## 4. The interface

```go
func Sum(xs []float64) float64
```

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `TestHandExample` | unit | the chapter's worked example | dur.91 |
| `TestSkipsNaN` | boundary | one NaN would otherwise poison every later total | dur.91 |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| comparing x != x by hand and getting it backwards | NaN totals | `TestSkipsNaN` |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Forward | `dur.91` | mean with an empty-input error |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| sum that skips nan | the real course module this fixture stands in for | everything | `course/DESIGN.md` |
