# Maintenance

## Overview

- **Primary references**: *Software Engineering at Google*, ch. 21 [Dependency Management](https://abseil.io/resources/swe-book/html/ch21.html) and ch. 22 [Large-Scale Changes](https://abseil.io/resources/swe-book/html/ch22.html) (free); Feathers, *Working Effectively with Legacy Code*
- **Supplementary**: [`git bisect`](https://git-scm.com/docs/git-bisect) documentation (free); Kleppmann, *Designing Data-Intensive Applications*, ch. 4 (encoding and evolution); [Protocol Buffers: updating a message type](https://protobuf.dev/programming-guides/proto3/#updating) (free)
- **Prerequisites**: [Releases and Deprecation](../06-releases-and-deprecation/), [The Testing Mentality](../03-testing-mentality/), and a running system from course Passes 1 to 10
- **Estimated time**: 2 weeks in course Pass 11 (four modules, each paired with a drill)

## Key Takeaways

- **Most engineering time is spent changing systems that already work.** Maintenance is a skill with techniques: versioned formats, dual-read migrations, dependency hygiene, and bisection.
- **Change an interface in phases**: add the new version, read both, write the new, migrate data, stop reading the old, remove it. Each phase is deployable and reversible.
- **A dependency upgrade is a change you did not write**: pin, read the changelog, run the full suite, and roll out like any other change.
- **Regressions are found by search, not by staring**: a benchmark with a threshold plus `git bisect run` finds the commit in $\log_2 n$ steps.

## How to Study

Read the two SWE at Google chapters and DDIA chapter 4, then practice `git bisect run` on a small repo with a planted regression. In the course, each module is paired with an incident drill on your cluster: the KV v2 and API v2 migrations are build modules graded by tests, the upgrade and the bisect are practices graded by their artifacts.

---

# Concepts & Techniques

## Core Insight

A system that cannot change safely is already failing slowly. Every migration in this topic follows the same shape: make the change additive, prove both sides work with tests, roll out behind a switch, and remove the old path only when telemetry shows nobody uses it.

## 1. Interface migrations

**Key ideas**:
- **KV format v1 to v2** (`craft.13`, build): fp8 E4M3 KV blocks with per-(layer, head) scales, mixed-version negotiation during rollout, drill `ops.04`.
- **API v1 to v2** (`craft.14`, build): the [`openai-subset.v2`](../../../../course/contracts/openapi/openai-subset.v2.yaml) contract, served alongside v1 under the deprecation policy, drill `ops.05`.

## 2. Dependencies and regressions

**Key ideas**:
- **Dependency upgrades** (`craft.15`): lockfiles, changelog review, the allowlist in `allowed-deps.toml`, drill `ops.06`.
- **Perf regression bisect** (`craft.16`): a benchmark gate, `git bisect run`, and a fix with a regression test, drill `ops.07`.

## Course modules

| Module | Topic | Kind | Pass |
|---|---|---|---|
| `craft.13` | Interface migration: KV format v1 to v2 | build | 11 |
| `craft.14` | Interface migration: API v1 to v2 | build | 11 |
| `craft.15` | Dependency upgrades | practice | 11 |
| `craft.16` | Perf regression bisect | practice | 11 |

## Chapters

| # | Module | Chapter | Kind | Pass |
|---|---|---|---|---|
| 1 | `craft.13` | [Interface migration: KV format v1 to v2](01-kv-format-v1-to-v2.md) | build | 11 |
| 2 | `craft.14` | [Interface migration: API v1 to v2](02-api-v1-to-v2.md) | build | 11 |
| 3 | `craft.15` | [Dependency upgrades](03-dependency-upgrades.md) | practice | 11 |
| 4 | `craft.16` | [Perf regression bisect](04-performance-regression-bisect.md) | practice | 11 |

## Connections to Other Tracks

| Track | Connection |
|---|---|
| [Releases and Deprecation](../06-releases-and-deprecation/) | the policy each migration follows |
| [Incident Response and Chaos](../../systems/05-incident-response-and-chaos/) | the drills `ops.04` to `ops.07` |
| [tinyllm Part 8](../../../courses/build-an-llm/p08-inference/) | the KV block pool whose format changes |
| [Numerical Methods and Floating Point](../../mathematics-for-computing/09-numerical-methods-and-floating-point/) | fp8 encoding (`M09.4`) behind KV v2 |
