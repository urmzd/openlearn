# The Course: Build Your Own LLM System

One system, end to end, in four languages: a byte-level language model trained in Python on its own numerics, kernels in C behind a stable ABI, a Rust inference engine that streams completions over HTTP, a Go gateway in front of it, and the containers, charts, traces, and runbooks that keep it running on Kubernetes. Every layer is built from first principles, and every layer's solution is here to read: the course is a solutions manual, not a coding exercise.

**For**: engineers who want to understand an LLM system from the matrix multiply to the pager well enough to explain every design decision in it, not just call one.

**How it works**: each chapter teaches one module from first principles (the math, a worked example by hand, the interface, the pitfalls), then walks through the reference solution in [`course/solutions/`](../../../course/solutions) step by step: the reasoning that forces the design, the annotated code, why it beats the alternatives, and questions to check that you can explain it back. A pass ends with the whole system running end to end as it stands after that pass ([the spiral](paths/course/SYSTEM.md#the-spiral)).

## Start

- Read [SYSTEM.md](paths/course/SYSTEM.md) first: the components, their languages, the contracts between them, and how the system grows pass by pass.
- Then follow the [course path](paths/course/) (or the [study plan](study-plan.md)) pass by pass. Each stage lists the chapters to read and the understanding it should leave you with.
- Each chapter links the solution files it explains and the contract they implement.

## Passes

| Pass | Path | What the system does after it |
|---|---|---|
| P0 | [course-p00-setup](paths/course-p00-setup) | a repository whose CI lints commits and runs the native and Python test suites |
| P1 | [course-p01-tracer](paths/course-p01-tracer) | a byte bigram trained by the system's CLI, served by the Rust engine, behind the Go gateway, on kind, with one trace and one runbook |
| P2-P11 | [pass paths](paths/course-p02-foundations) through [operate](paths/course-p11-operate) | autograd, data, model architecture, inference, serving, durable workflows, agents, and operations |
| P12 | [course-p12-multimodal](paths/course-p12-multimodal) | the multimodal extension planned in B14 |

Passes 2 to 11 take the system from autograd through operations. Pass 12 is the multimodal extension in the B14 design. Each pass has its own staged path in [`path.tsv`](paths/course/path.tsv).

## Reading order

- [SYSTEM.md](paths/course/SYSTEM.md): read it first.
- Each pass path's README says what that pass builds; its `milestone.md` says what the system does end to end after the pass, and why each piece is needed.
- Contracts live in [`course/contracts/`](../../../course/contracts): the interfaces and formats every solution implements.
- Solutions live in [`course/solutions/`](../../../course/solutions): `python/`, `rust/`, `go/`, `c/`, the entry points and deploy files in `entry/`, and the reference documents in `docs/`.
