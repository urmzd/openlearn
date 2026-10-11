<!-- ol:module dur.91 -->
# Mean with an empty-input error

## Overview

| | |
|---|---|
| **Module** | `dur.91` · build · Go · Pass 2 · 20 min |
| **You build** | `go/ds/demo/mean.go`: `Mean` |
| **Contract** | none (Rust and Go fixtures check through the course tests) |
| **Tests** | `course/tests/go/dur_91/` (what they check: section 4) |
| **Needs** | `dur.90` |
| **Used by** | `craft.90` |
| **Optional depth** | none: this is a harness fixture |

## Key Takeaways

- An empty input has no mean; returning a sentinel error lets callers branch on it with `errors.

## How to work this chapter

```bash
ol start dur.91
ol tests dur.91
ol check dur.91
```

---

## 1. Why now

Callers of the Go mean need to know when there was nothing to average.

## 2. Principles

An empty input has no mean; returning a sentinel error lets callers branch on it with `errors.Is`.

## 3. Worked example by hand

Mean([1, 2, 3, 4]) = 10 / 4 = 2.5.

## 4. The interface

```go
func Mean(xs []float64) (float64, error)
```

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `TestHandExample` | unit | the chapter's worked example | craft.90 |
| `TestEmptyIsError` | boundary | the mean of nothing is undefined; callers must see that, not a 0 | craft.90 |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| returning 0, nil for empty input | silent wrong answers | `TestEmptyIsError` |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `dur.90` | sum that skips nan |
| Forward | `craft.90` | document the demo system |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| mean with an empty-input error | the real course module this fixture stands in for | everything | `course/DESIGN.md` |
