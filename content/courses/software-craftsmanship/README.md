# Software Craftsmanship

Good software is not about clever code. It is about managing complexity over time: code that survives reality, that a newcomer can understand, and that you can change without breaking. This book is for working programmers who want the practices and tradeoff judgment behind that. The first half covers principles, drawn from *The Pragmatic Programmer*, *Software Engineering at Google*, and postmortems of real projects; the second half applies them to one running system, an LLM serving platform with a Go gateway and a Rust engine.

## What you will be able to do

- Use DRY, orthogonality, reversibility, design by contract, and crashing early to keep a codebase changeable.
- Judge a test suite by the faults it catches, and add property-based, contract, resilience, and evaluation tests where example tests fall short.
- Compute a release from commit history, write a deprecation policy, and migrate an interface in reversible phases.
- Upgrade a dependency safely and find a performance regression with `git bisect run`.
- Threat-model a whole system with STRIDE, inventory its supply chain, and review its secrets and authorization.

## Ways through this book

The four principle chapters take three to four weeks at six to eight hours a week; the four practice chapters add about six more, and are best paced by the system you build alongside them.

- **Principles first** (3 to 4 weeks): [Chapter 1: The Pragmatic Programmer](01-pragmatic-programmer/README.md), then [Chapter 2: Software Engineering at Google](02-swe-at-google/README.md), the opening of [Chapter 3: The Testing Mentality](03-testing-mentality/README.md) with [3.2 Property-based tests](03-testing-mentality/02-property-based-tests.md), then [Chapter 4: Lessons from Practice](04-lessons-from-practice/README.md).
- **Alongside the running system** (paced by the [course path](../build-an-llm/paths/course/README.md)): [5.1 Your repo and CI gate](05-code-review-and-ci/01-your-repo-and-ci-gate.md) right after the setup primers, Chapter 3 sections as their components are built, [5.2 Code review](05-code-review-and-ci/02-code-review.md) once the serving platform exists, then [Chapter 6](06-releases-and-deprecation/README.md), [Chapter 7](07-maintenance/README.md), and [Chapter 8](08-security/README.md) once the whole system runs.
- **Changing a system that already ships** (about 3 weeks): [6.1 Commits, semver, and releases](06-releases-and-deprecation/01-commits-semver-and-releases.md), [6.2 Deprecation policy](06-releases-and-deprecation/02-deprecation-policy.md), and [7.1](07-maintenance/01-kv-format-v1-to-v2.md) through [7.4](07-maintenance/04-performance-regression-bisect.md).
- **Strengthening a test suite** (about 2 weeks): [Chapter 3](03-testing-mentality/README.md) from [3.1 TDD, unit tests, and mutation grading](03-testing-mentality/01-tdd-unit-tests-and-mutation-grading.md) to [3.9 Agent evals as tests](03-testing-mentality/09-agent-evals-as-tests.md).

## Before you start

You need some professional programming experience. Chapters 5 to 8 and the system sections of Chapter 3 are written against the system built in *Build Your Own LLM System*; the [Language and Tool Primers](../programming-languages/02-language-and-tool-primers/README.md) in *Programming Languages* cover the tools it uses. Chapter 8 starts from the container views in [Diagramming with the C4 Model](../diagramming/01-diagramming-c4/README.md) from *Diagramming and Documentation*, which also covers [Documentation Writing](../diagramming/02-documentation-writing/README.md). New to these books? [How to read these books](../../references/how-to-read.md) explains how sections, review questions, and solutions work.
