# Software Engineering at Google

## Overview

- **Primary reference**: *Software Engineering at Google* by Winters, Manshreck, Wright -- [free online](https://abseil.io/resources/swe-book)
- **Supplementary**: [Google Engineering Practices](https://google.github.io/eng-practices/) (free), [Testing on the Toilet](https://testing.googleblog.com/) (free)
- **Prerequisites**: [The Pragmatic Programmer](../01-pragmatic-programmer/) (individual craft; this topic is organizational craft)
- **Estimated time**: 1.5-2 weeks at 6-8 hrs/week

## Key Takeaways

- **Software engineering is programming integrated over time** -- the discipline of code that must survive years, thousands of engineers, and billions of users.
- **Culture is an engineering concern**: humility/respect/trust, knowledge sharing, and readability are load-bearing at scale.
- **Process is leverage**: small CLs, mandatory review, docs-as-code, and a deprecation budget keep a large codebase changeable.
- **Hyrum's Law** governs every API and migration: all observable behavior will eventually be depended on.

## How to Study

- Read selectively -- the book is a reference; start with the chapters nearest your current pain.
- For each process (code review, CI, deprecation), map it onto how your team does it today.
- Internalize the test pyramid and Hyrum's Law before designing any public interface.

---

# Concepts & Techniques

## Core Insight

Software engineering is programming integrated over time. Google's lessons are about what happens when code must survive years, be maintained by thousands of engineers, and serve billions of users.

## 1. Culture

**Key concepts**:
- **What is software engineering?** -- not just writing code; it's writing code that must be maintained. Time, scale, and trade-offs distinguish engineering from programming
- **How to work well on teams** -- humility, respect, trust (HRT). Hide nothing, share early, accept feedback
- **Knowledge sharing** -- readability reviews, go/links, documentation as code, internal tech talks
- **Engineering for equity** -- build for everyone, test with diverse inputs, avoid encoding bias

**How this appears at Staff+ level**: you're responsible for team culture. If knowledge is siloed, that's your problem.

## 2. Processes

**Key concepts**:
- **Style guides and rules** -- consistency beats personal preference. Google's style guides exist to make code *readable by others*, not *writable by you*
- **Code review** -- every change is reviewed. Reviews are about readability and correctness, not gatekeeping. Keep changes small
- **Documentation** -- treat docs like code: version it, review it, test it, deprecate it (see [Documentation & Technical Writing](../06-documentation-writing/))
- **Testing overview** -- small tests (unit), medium tests (integration), large tests (end-to-end). Pyramid shape: many small, few large (deepened in [The Testing Mentality](../03-testing-mentality/))
- **Deprecation** -- removing old systems is as important as building new ones. Budget for it. Make it incremental

**Key insight**: code review at Google serves three purposes: correctness, knowledge transfer, and establishing norms. The social function matters as much as the technical one.

## 3. Testing at Scale

**Key concepts**:
- **Unit testing** -- fast, isolated, deterministic. Test behavior, not implementation. Google's Testing on the Toilet one-pagers
- **Test doubles** -- fakes (preferred at Google) vs mocks vs stubs. Avoid mock-heavy tests -- they're brittle and don't test real behavior
- **Larger testing** -- integration, end-to-end, load testing. Slower but catch interface bugs. Keep the test pyramid shape
- **Hyrum's Law** -- "with a sufficient number of users, all observable behaviors of your system will be depended on." Test for what you promise, not what happens to work

**Hyrum's Law in practice**: if your API returns JSON keys in sorted order (even though the spec doesn't require it), someone will depend on that ordering. When you stop sorting, their code breaks.

## 4. Tools and Infrastructure

**Key concepts**:
- **Version control and branch management** -- monorepo at Google. Trunk-based development. Feature flags over feature branches
- **Build systems** -- Bazel, reproducible builds, hermetic builds. Build once, cache forever
- **Code search** -- critical infrastructure. Every engineer can search all of Google's code instantly
- **Static analysis** -- compiler warnings, linters, type checkers. Make it automatic. Don't require humans to catch what machines can
- **Dependency management** -- the hardest problem in software engineering at scale. Diamond dependency problem. Minimum version selection
- **Continuous integration** -- presubmit checks, post-submit testing, release qualification
- **Large-scale changes (LSCs)** -- tools for making changes across thousands of files/repos atomically

**Staff+ insight**: the build system and CI pipeline are force multipliers. Investing a week to speed up CI by 30% pays for itself in a month across 100 engineers.

## Technique Catalog

| Principle | When to Apply |
|-----------|---------------|
| Hyrum's Law | Designing APIs, planning migrations, deprecating features |
| Small CLs | Every code review (keep changes under ~200 lines) |
| Test pyramid | Designing test strategy for a service |
| Fakes over mocks | Writing test doubles (prefer fakes for real behavior) |
| Deprecation budget | Planning roadmaps (explicitly budget for cleanup) |
| Hermetic builds | Reproducible CI; "build once, cache forever" |
| Large-scale changes (LSCs) | Repo-wide refactors that must land atomically |

## Connections to Other Tracks

| Concept | Connected Track | How |
|---------|-----------------|-----|
| Code review, testing | [Systems & Architecture](../../systems/) | System design interviews test whether you'd build maintainable systems |
| CI/CD, build systems | [Cloud Native](../../systems/03-cloud-native/) | Practical implementation of these processes |
| Deprecation, LSCs | [Observability](../../systems/04-observability/) | Need observability to safely deprecate |
| Docs-as-code | [Documentation & Technical Writing](../06-documentation-writing/) | The hands-on counterpart to "treat docs like code" |
| SDLC vs the LLM development lifecycle | [Foundation Models §7](../../ml/05-foundation-models/) | Why building an LLM (grow weights from data) is a different discipline than the software SDLC |

## Company Relevance

| Company | How This Appears | Focus |
|---------|-----------------|-------|
| Google | Directly -- their engineering culture IS this book. Readability reviews, Hyrum's Law | Culture, process |
| Meta | Move fast with stable infrastructure. Code review culture | Testing, CI |
| Stripe | API design excellence, backwards compatibility, documentation | Contracts, docs |
| Amazon | Operational excellence, ownership | Process, responsibility |
| All Staff+ roles | Every company expects Staff+ engineers to drive engineering excellence | All principles |
