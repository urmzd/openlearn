# Mock Engagement

## Overview

- **Primary references**: the earlier topics of this track, [01 Discovery](../01-discovery/) through [07 Escalation and Handoff](../07-escalation-and-handoff/); your own platform from the [course](../../../courses/build-an-llm/paths/course/)
- **Supplementary**: the role-path [capstone](capstone.md) (one mock customer, Lexa, through every part); Shostack, *Threat Modeling* (for the security questionnaire)
- **Prerequisites**: field-engineering topics 01 to 07; for the course modules, your system at `v1.0.0` (course Passes 1 to 10)
- **Estimated time**: 1 to 2 weeks in course Pass 11

## Key Takeaways

- **A field engineer sells and delivers a system they understand to the bottom.** Running a mock engagement against a platform you built removes the guesswork: every sizing number, benchmark, and security answer can be checked.
- **Each stage has a deliverable with a check**: discovery notes, a sizing, a performance report, a POC plan, a migration plan, a security questionnaire, a handoff.
- **Numbers come from your own tools**: capacity from `M05.1`, load reports from `load.01`, threat model and SBOMs from `craft.17` and `craft.18`.

## How to Study

Work the [capstone](capstone.md) for the role path, then run `field.01` to `field.07` against your own platform in course order, each graded by its rubric.

---

# Concepts & Techniques

## Core Insight

The engagement is the same for any platform; what changes is how much you can prove. With your own system, every claim in a proposal links to a test, a report, or a document you produced, which is the standard a customer's architects will hold you to.

## 1. The engagement against your platform

**Key ideas**:
- **Stages** `field.01` to `field.05`: discovery, qualification and sizing, performance engagement, POC and evaluation, migration plan.
- **Commercials and security** (`field.06`): answer a security questionnaire from your threat model, SBOMs, and usage policy.
- **Escalation and handoff** (`field.07`): hand the platform to a mock customer team using your runbooks.

## Course modules

| Module | Topic | Kind | Pass |
|---|---|---|---|
| `field.01` | Discovery against your platform | practice | 11 |
| `field.02` | Qualification and sizing | practice | 11 |
| `field.03` | Performance engagement | practice | 11 |
| `field.04` | POC and evaluation | practice | 11 |
| `field.05` | Migration plan | practice | 11 |
| `field.06` | Commercials and security review: answer a customer security questionnaire from your threat model and SBOM | practice | 11 |
| `field.07` | Escalation and handoff: hand the platform to a mock customer team using your runbooks | practice | 11 |

## Chapters

<!-- ol:chapters -->
| # | Module | Chapter | Kind | Pass |
|---|---|---|---|---|
| 1 | `field.01` | [Mock engagement: discovery](01-discovery.md) | practice | 11 |
| 2 | `field.02` | [Mock engagement: qualification and sizing](02-qualification-and-sizing.md) | practice | 11 |
| 3 | `field.03` | [Mock engagement: performance engagement](03-performance-engagement.md) | practice | 11 |
| 4 | `field.04` | [Mock engagement: POC and evaluation](04-poc-and-evaluation.md) | practice | 11 |
| 5 | `field.05` | [Mock engagement: migration plan](05-migration-plan.md) | practice | 11 |
| 6 | `field.06` | [Commercials and security review](06-commercials-and-security.md) | practice | 11 |
| 7 | `field.07` | [Escalation and handoff](07-escalation-and-handoff.md) | practice | 11 |
<!-- /ol:chapters -->

## Connections to Other Tracks

| Track | Connection |
|---|---|
| [Field Engineering](../) | the track this topic applies |
| [Security](../../software-craftsmanship/11-security/) | the threat model and SBOMs behind the questionnaire |
| [Incident Response and Chaos](../../systems/05-incident-response-and-chaos/) | the runbooks the handoff uses |
