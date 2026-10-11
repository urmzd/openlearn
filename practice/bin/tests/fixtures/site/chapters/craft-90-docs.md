<!-- ol:module craft.90 -->
# Document the demo system

## Overview

| | |
|---|---|
| **Module** | `craft.90` · practice · docs · Pass 3 · 20 min |
| **You build** | `docs/demo.md`: one page that names every demo component |
| **Contract** | none |
| **Tests** | `course/tests/craft.90/check` (what they check: section 4) |
| **Needs** | `M90.2` · `ds.91` · `dur.91` |
| **Used by** | no call site: a practice artifact |
| **Optional depth** | none: this is a harness fixture |

## Key Takeaways

- A system page names each component, its language, and the module that built it.

## How to work this chapter

```bash
ol start craft.90
ol tests craft.90
ol check craft.90
```

---

## 1. Why now

Nobody can operate the demo system without a page that says what each part does.

## 2. Principles

A system page names each component, its language, and the module that built it.

## 3. Worked example by hand

A page titled `# Demo system` with one line per component.

## 4. The artifact and its check

```text
docs/demo.md, starting with a `# ` title
```

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| no title line | the check fails | the artifact check |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M90.2` | l1 normalize |
| Back | `ds.91` | mean over the compensated sum |
| Back | `dur.91` | mean with an empty-input error |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| document the demo system | the real course module this fixture stands in for | everything | `course/DESIGN.md` |
