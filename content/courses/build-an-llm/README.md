# The Course: Build Your Own LLM System

You build one system, end to end, in four languages: a byte-level language model trained in Python on your own numerics, kernels in C behind a stable ABI, a Rust inference engine that streams completions over HTTP, a Go gateway in front of it, and the containers, charts, traces, and runbooks that keep it running on Kubernetes. Every layer is yours. The course grades it with tests you can read, milestones that run your entry points, and drills that break your deployment on purpose.

**For**: engineers who want to own an LLM system from the matrix multiply to the pager, not just call one.

**How it works**: each chapter teaches one module from first principles (the math, a worked example by hand, the interface, the pitfalls), then you write the code in your own repo and `ol check <ID>` grades it. A pass ends at a milestone that runs your whole system as it stands: after every pass it runs end to end ([the spiral](paths/course/SYSTEM.md#the-spiral)).

## Start

```bash
practice/bin/ol doctor                         # the tools pass 0 and pass 1 need
practice/bin/ol course init --name forge       # your repo (default .scratchpad/course; OL_COURSE_HOME moves it)
practice/bin/ol learn course                   # every stage and your progress
practice/bin/ol learn course next              # read the next unfinished stage
practice/bin/ol start lang.01                  # stub a module into your repo
practice/bin/ol check lang.01                  # grade it
```

`ol learn course --done <stage>` marks a stage done only when its check passes (`[x]`), or with `--force` (`[~]`). The fifth column of [`path.tsv`](paths/course/path.tsv) names each stage's check.

## Passes

| Pass | Path | What your system does after it | Gate |
|---|---|---|---|
| P0 | [course-p00-setup](paths/course-p00-setup) | an empty repo whose CI lints commits, runs native tests, and runs `ol check --all --ci` | MS-P0 |
| P1 | [course-p01-tracer](paths/course-p01-tracer) | a byte bigram trained by your CLI, served by your Rust engine, behind your Go gateway, on kind, with one trace and one runbook | MS-P1 |
| P2-P11 | [pass paths](paths/course-p02-foundations) through [operate](paths/course-p11-operate) | autograd, data, model architecture, inference, serving, durable workflows, agents, and operations | MS-P2 to MS-P11 |
| P12 | [course-p12-multimodal](paths/course-p12-multimodal) | multimodal extension planned in B14 | MS-P12, after Step C |

Passes 2 to 11 take the system from autograd through operations. Pass 12 is the multimodal extension in the B14 design. Each pass has its own staged path and check column in [`path.tsv`](paths/course/path.tsv).

## Reading order

- [SYSTEM.md](paths/course/SYSTEM.md): the components, their languages, the contracts between them, and how the system grows pass by pass. Read it first.
- Each pass path's README says what that pass builds; its `milestone.md` says what the gate runs.
- Contracts live in [`course/contracts/`](../../../course/contracts) and are vendored into your repo's `contracts/` by `ol course init`.
