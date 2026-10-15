# Usage Policy

## Overview

- **Primary references**: NIST, [AI Risk Management Framework 1.0](https://www.nist.gov/itl/ai-risk-management-framework) (free); OWASP, [Top 10 for LLM Applications](https://genai.owasp.org/llm-top-10/) (free)
- **Supplementary**: Inan et al., [*Llama Guard*](https://arxiv.org/abs/2312.06674) (free); published acceptable-use policies of model providers (compare two); the [Gateway](../../ai-platform-engineering/12-gateway/) topic
- **Prerequisites**: [Bias and Safety Evals](../04-bias-and-safety-evals/), the gateway (`gw.01` to `gw.07`), and the fine-tuning heads (`L6.5`)
- **Estimated time**: 1 week in course Pass 10

## Key Takeaways

- **A usage policy is a contract with your users**: what the service is for, what it refuses, and what happens on a violation.
- **Enforce it where every request passes**: the gateway, before the request reaches an engine, with a decision that is logged and appealable.
- **Classification is cheap when you reuse what you built**: a linear head over the engine's embeddings, exported as `linear-head.schema.json` and evaluated as a dot product in Go.
- **Every classifier errs**: publish the measured false-positive and false-negative rates on a labelled set, and choose the threshold on purpose.

## How to Study

Read the NIST AI RMF core functions (Govern, Map, Measure, Manage), the OWASP LLM top 10, and two providers' acceptable-use policies side by side. In the course, `ethics.05` writes your policy (`formats/policy.v1`) and `gw.08` enforces it at your gateway with the linear-head classifier (course decision D33).

---

# Concepts & Techniques

## Core Insight

Policy without enforcement is a wish, and enforcement without policy is arbitrary. The usage policy states categories and actions in a machine-readable file; the gateway evaluates each request against it with a measured classifier; and the ledger records each decision so it can be audited and appealed.

## 1. Writing the policy

**Key ideas**:
- **Categories, actions, and appeals**: allow, refuse with a reason, or flag for review, per category.
- **Machine-readable**: the policy file is a contract (`formats/policy.v1`) the gateway loads, not prose in a wiki.

## 2. Enforcing it at the gateway

**Key ideas**:
- **Classifier**: `SequenceClassifier(pool='mean')` from `L6.5`, fitted with IRLS (`M07.7`) over `/v1/embeddings`, exported as a linear head.
- **Decision path**: the policy middleware sits before routing in the handler chain; refusals and their scores go to the usage ledger with redacted logs.

## Course modules

| Module | Topic | Kind | Pass |
|---|---|---|---|
| `ethics.05` | Usage policy at the gateway | practice | 10 |

Its enforcement is the build module `gw.08` (usage policy enforcement, Pass 10), whose chapter lives in [Gateway](../../ai-platform-engineering/12-gateway/).

## Chapters

<!-- ol:chapters -->
| # | Module | Chapter | Kind | Pass |
|---|---|---|---|---|
| 1 | `ethics.05` | [Usage policy at the gateway](01-usage-policy-at-the-gateway.md) | practice | 10 |
<!-- /ol:chapters -->

## Connections to Other Tracks

| Track | Connection |
|---|---|
| [Responsible AI](../) | the track overview and how the six topics connect |
| [Gateway](../../ai-platform-engineering/12-gateway/) | the middleware chain the policy check joins |
| [Agent SDK](../../ai-platform-engineering/13-agent-sdk/) | tool gates as the agent-side counterpart |
| [Security](../../software-craftsmanship/11-security/) | the threat model that names abuse cases |
