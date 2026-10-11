# Archive

Historical coursework, kept for reference and history, not for study. Nothing here is part of the [course](../paths/course/) or the tracks: the site and the book exclude `archive/`, `ol verify` does not run it, and its links are checked only so they do not rot.

## What is here

| Path | What it was | Why archived | Still cited by |
|---|---|---|---|
| [`algorithms/12-concurrency-systems/`](algorithms/12-concurrency-systems/) | a C proof-of-work miner and its test harness from a systems course | coursework with no call site; the course teaches threads and pools through `rt.03` | none |
| [`algorithms/13-functional-programming/`](algorithms/13-functional-programming/) | Scheme exercises on recursion, visitors, and iterators | coursework; the type theory now lives in [07 Type Systems](../software-craftsmanship/07-type-systems/) | none |
| [`algorithms/14-ml-statistics/`](algorithms/14-ml-statistics/) | Python, R, Perl, and Prolog labs from ML, statistics, and NLP courses | coursework; the course rebuilds these ideas with tests | `k-means.py` and `tf-idf-vector-search.py` are worked examples for `ag.07` |
| [`practice/cpp-03-hash-map/`](practice/cpp-03-hash-map/) | the C++ hash-map drill | six hash maps across the drills became two course modules, `ds.02` and `ds.05` | none |
| [`justfile`](justfile) | the `run-12`, `run-13`, and `run-14` recipes | they ran the archived coursework | none |

## Running archived code

```bash
just --justfile archive/justfile --working-directory . --list
just --justfile archive/justfile --working-directory . run-12-concurrency-systems
```

Archived code is not maintained: toolchains may have moved on, and CI only lints its Python (ruff); it never runs it.
