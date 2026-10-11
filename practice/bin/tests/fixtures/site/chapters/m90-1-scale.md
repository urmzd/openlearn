<!-- ol:module M90.1 -->
# Scale a vector

## Overview

| | |
|---|---|
| **Module** | `M90.1` · build · Python · Pass 1 · 20 min |
| **You build** | `python/tinyllm/demo/scale.py`: `scale` |
| **Contract** | [`contracts/py/tinyllm/demo/scale.pyi`](../course/contracts/py/tinyllm/demo/scale.pyi) |
| **Tests** | `course/tests/M90.1/` (what they check: section 4) |
| **Needs** | nothing |
| **Used by** | `M90.2` · `M90.3` |
| **Optional depth** | none: this is a harness fixture |

## Key Takeaways

- Scaling multiplies each element by the same factor $k$, so $y_i = k x_i$.

## How to work this chapter

```bash
ol start M90.1
ol tests M90.1
ol check M90.1
```

---

## 1. Why now

The normalizer in M90.2 divides by a norm; it needs a scaling step that never touches its input.

## 2. Principles

Scaling multiplies each element by the same factor $k$, so $y_i = k x_i$.

| Symbol | Meaning | Type / shape |
|---|---|---|
| $x_i$ | input element | `float` |
| $k$ | scale factor | `float` |
| $y_i$ | output element | `float` |

## 3. Worked example by hand

k = 2.5 and x = [1, -2, 0.5]: y = [2.5, -5, 1.25].

## 4. The interface

```python
def scale(xs: list[float], k: float) -> list[float]: ...
```

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example` | unit | the chapter's worked example | M90.2, M90.3 |
| `test_input_not_modified` | property | callers reuse their vector after scaling it; a function that scales in place silently corrupts the caller's data | M90.2, M90.3 |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| scaling in place | the caller's vector changes | `test_input_not_modified` |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Forward | `M90.2` | l1 normalize |
| Forward | `M90.3` | scale rejects non-finite factors |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| scale a vector | the real course module this fixture stands in for | everything | `course/DESIGN.md` |
