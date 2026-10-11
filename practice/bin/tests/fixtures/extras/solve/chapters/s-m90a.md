<!-- ol:module S-M90a -->
# Demo problem set

## Overview

| | |
|---|---|
| **Module** | `S-M90a` · solve · none · Pass 1 · 20 min |
| **You build** | answers in `solve/S-M90a.toml` and one proof |
| **Contract** | none |
| **Tests** | `course/solve/S-M90a/key.toml` (hidden) |
| **Needs** | nothing |
| **Used by** | no call site: a pen and paper set |

## Key Takeaways

- Answers are ASCII math checked by SymPy; proofs are self-graded.

## How to work this chapter

```bash
ol start S-M90a
ol check S-M90a
```

---

## 1. Why now

The demo system needs nothing here; this is a harness fixture for the solve kind.

## 2. Principles

An answer is equal to the key when SymPy proves it or samples agree.

## 3. Worked example by hand

The derivative of sin(x^2) is 2x cos(x^2) by the chain rule.

## 4. The problems

The five problems are in `course/solve/S-M90a/problems.md`.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| forgetting the inner derivative | q1 fails | the key |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Forward | none | a fixture |
