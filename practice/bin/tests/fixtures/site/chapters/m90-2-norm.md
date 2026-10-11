<!-- ol:module M90.2 -->
# L1 normalize

## Overview

| | |
|---|---|
| **Module** | `M90.2` · build · Python · Pass 2 · 20 min |
| **You build** | `python/tinyllm/demo/norm.py`: `l1_normalize` |
| **Contract** | [`contracts/py/tinyllm/demo/norm.pyi`](../course/contracts/py/tinyllm/demo/norm.pyi) |
| **Tests** | `course/tests/M90.2/` (what they check: section 4) |
| **Needs** | `M90.1` |
| **Used by** | `craft.90` |
| **Optional depth** | none: this is a harness fixture |

## Key Takeaways

- The L1 norm is $\lVert x \rVert_1 = \sum_i |x_i|$, and the normalized vector is $x / \lVert x \rVert_1$, whose absolute values sum to 1.

## How to work this chapter

```bash
ol start M90.2
ol tests M90.2
ol check M90.2
```

---

## 1. Why now

The demo system reports proportions; raw magnitudes from different inputs are not comparable until they are normalized.

## 2. Principles

The L1 norm is $\lVert x \rVert_1 = \sum_i |x_i|$, and the normalized vector is $x / \lVert x \rVert_1$, whose absolute values sum to 1.

| Symbol | Meaning | Type / shape |
|---|---|---|
| $x$ | input vector | `list[float]` |
| $\lVert x \rVert_1$ | sum of absolute values | `float` |

## 3. Worked example by hand

x = [1, -3]: the norm is 1 + 3 = 4, so the result is [0.25, -0.75].

## 4. The interface

```python
def l1_normalize(xs: list[float]) -> list[float]: ...
```

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example` | unit | the chapter's worked example | craft.90 |
| `test_fixture_vectors` | golden | the committed fixture pins a few vectors with known answers, so the test and the chapter agree on the definition across languages | craft.90 |
| `test_abs_sums_to_one` | property | the defining property of L1 normalization, over seeded random inputs | craft.90 |
| `test_all_zero_raises` | boundary | an all-zero vector has no direction; dividing by 0 would produce nan | craft.90 |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| dividing by a zero norm | a list of nan | `test_all_zero_raises` |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M90.1` | scale a vector |
| Forward | `craft.90` | document the demo system |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| l1 normalize through the c sum | the real course module this fixture stands in for | everything | `course/DESIGN.md` |
