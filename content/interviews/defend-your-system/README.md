# Defend Your System

## Overview

- **Primary references**: your own design documents from the course: ADRs (`craft.02`), C4 diagrams (`craft.09`), the threat model (`craft.17`), and the design reviews (`review.01` to `review.03`)
- **Supplementary**: Kleppmann, *Designing Data-Intensive Applications*; the [shared interview concepts](../shared-concepts/) and the company folders in this track for the question styles each company uses
- **Prerequisites**: the whole course through Pass 11, or a system of your own that you designed and operate
- **Estimated time**: 1 week in course Pass 11

## Key Takeaways

- **The strongest interview answer is about a system you built**: you know where it breaks, why each choice was made, and what you would change.
- **Answer with evidence**: a load report, a benchmark, a drill postmortem, an ADR with the rejected options.
- **Every choice has a cost**: name the alternative you rejected and what it would have bought.

## How to Study

Answer the question bank out loud against your own design, then write each answer down with the artifact that backs it. `iv.01` is a proof module graded by rubric.

---

# Concepts & Techniques

## Core Insight

System design interviews test judgment under questioning. Defending a system you built end to end turns that from rehearsal into recall: the trade-offs are ones you made, measured, and lived with through drills and migrations.

## 1. The question bank

**Key ideas**:
- **Architecture**: why a C ABI and HTTP between the languages; why the gateway owns keys and limits; why a durable engine instead of cron.
- **Scale and failure**: what breaks first at 10x load; what happens when a decode pod dies mid-stream; how resume stays exactly-once.
- **Models and data**: why this tokenizer and vocabulary size; how you know the eval numbers are not contaminated.

## Course modules

| Module | Topic | Kind | Pass |
|---|---|---|---|
| `iv.01` | Defend your system | proof | 11 |

## Chapters

| # | Module | Chapter | Kind | Pass |
|---|---|---|---|---|
| 1 | `iv.01` | [Defend your system](01-defend-your-system.md) | proof | 11 |

## Connections to Other Tracks

| Track | Connection |
|---|---|
| [System Design](../../tracks/systems/01-system-design/) | the design reviews this module draws on |
| [Documentation Writing](../../tracks/software-craftsmanship/06-documentation-writing/) | ADRs as the record of your choices |
| [Interview Prep](../) | company-specific formats |
