# Usage Policy

Evals tell you how a model fails; a usage policy decides what you do about it in front of users. A usage policy is a contract with your users: what the service is for, what it refuses, and what happens on a violation. Policy without enforcement is a wish, and enforcement without policy is arbitrary, so this chapter does both. The policy states categories and actions in a machine-readable file, not prose in a wiki; the gateway evaluates each request against it, before the request reaches an engine, with a decision that is logged and appealable; and the audit trail records each decision so it can be reviewed.

The chapter's section, [Usage policy at the gateway](01-usage-policy-at-the-gateway.md), writes the policy document and its ordered rules and walks through the Go evaluator that enforces them. Classification is cheap when you reuse what you built: a linear head over the engine's embeddings, exported as `linear-head.schema.json` and evaluated as a dot product in Go, so the policy middleware can sit before routing in the handler chain without adding a service. Every classifier errs, so the policy publishes the measured false-positive and false-negative rates on a labelled set and chooses its threshold on purpose. A compact example of the finished policy document, [USAGE_POLICY.md](02-usage-policy.md), sits alongside the section.

> **Learning objectives**
> - Write a usage policy as ordered, machine-readable rules with allow and deny actions, stable rule ids, and user-safe reasons.
> - Enforce it at the gateway with exact metadata checks first and a linear-head classifier last.
> - Fail closed when the classifier's dependency is down, and keep prompt text out of the audit log.
> - Give the policy an owner, a revision, a review cadence, and an appeal path.

## Before you begin

Read [Bias and Safety Evals](../04-bias-and-safety-evals/README.md) first; this chapter acts on what those evals find. The enforcement runs in the gateway of *AI Platform Engineering*, from [The server skeleton](../../ai-platform-engineering/12-gateway/02-server-skeleton.md) through [The usage ledger and admin API](../../ai-platform-engineering/12-gateway/05-usage-ledger-and-admin-api.md), and the classifier is the linear head from [Fine-tuning heads and the linear-head export](../../build-an-llm/07-objectives/05-fine-tuning-heads-and-the-linear-head-export.md) in *Build Your Own LLM System*, fitted with IRLS over `/v1/embeddings`. Good preparation is the NIST AI RMF core functions (Govern, Map, Measure, Manage), the OWASP LLM top 10, and two providers' acceptable-use policies read side by side.

## In this chapter

- [Usage policy at the gateway](01-usage-policy-at-the-gateway.md): policy as configuration, the linear head and its threshold, first-match rule ordering, fail-closed enforcement, privacy-safe audit events, and an annotated walkthrough of the rules, the evaluator, and the policy document.
- [Usage policy](02-usage-policy.md): a compact example of the published policy document.

The enforcement code itself is built in [Usage policy enforcement](../../ai-platform-engineering/12-gateway/06-usage-policy-enforcement.md), part of the [Gateway](../../ai-platform-engineering/12-gateway/) chapter whose middleware chain the policy check joins. Tool gates in the [Agent SDK](../../ai-platform-engineering/13-agent-sdk/) are the agent-side counterpart, and the threat model that names abuse cases is in [Security](../../software-craftsmanship/08-security/) in *Software Craftsmanship*.

## Further reading

- NIST, [AI Risk Management Framework 1.0](https://www.nist.gov/itl/ai-risk-management-framework) (free).
- OWASP, [Top 10 for LLM Applications](https://genai.owasp.org/llm-top-10/) (free).
- Inan et al., [*Llama Guard*](https://arxiv.org/abs/2312.06674) (free).
- The published acceptable-use policies of model providers; compare two.
