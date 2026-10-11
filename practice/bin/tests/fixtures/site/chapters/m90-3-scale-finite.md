<!-- ol:module M90.3 -->
# Scale rejects non-finite factors

## Overview

| | |
|---|---|
| **Module** | `M90.3` · build · Python · Pass 2 · 20 min |
| **You build** | `python/tinyllm/demo/scale.py` (taken over from M90.1): `scale` rejects a non-finite factor |
| **Contract** | unchanged contract of M90.1 |
| **Tests** | `course/tests/M90.3/` (what they check: section 4) |
| **Needs** | `M90.1` |
| **Used by** | `M90.2` |
| **Optional depth** | none: this is a harness fixture |

## Key Takeaways

- A finite float is neither inf nor nan; `math.

## How to work this chapter

```bash
ol start M90.3
ol tests M90.3
ol check M90.3
```

---

## 1. Why now

A factor of inf from an upstream bug turns the whole vector into inf and nan, and the normalizer carries it on silently.

## 2. Principles

A finite float is neither inf nor nan; `math.isfinite` tests both. The contract is unchanged: same name, same parameters.

## 3. Worked example by hand

scale([1, 2], inf) raises ValueError instead of returning [inf, inf]; `test_rejects_nonfinite_k` checks exactly this case.

## 4. The interface

```python
def scale(xs: list[float], k: float) -> list[float]: ...  # ValueError when k is not finite
```

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_rejects_nonfinite_k` | boundary | a non-finite factor turns every element into inf or nan, which the sum in M90 | M90.2 |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| checking only for inf | nan slips through | `test_rejects_nonfinite_k` |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M90.1` | scale a vector |
| Forward | `M90.2` | l1 normalize through the c sum |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| scale rejects non-finite factors | the real course module this fixture stands in for | everything | `course/DESIGN.md` |
