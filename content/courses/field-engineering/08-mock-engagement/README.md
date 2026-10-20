# Mock Engagement

A field engineer sells and delivers a system they understand to the bottom. The earlier chapters of this book taught each stage of an engagement on its own; this chapter runs the whole engagement end to end, against a platform you built yourself, so that the guesswork disappears. Every sizing number, benchmark, and security answer can be checked, because every claim links to a test, a report, or a document you produced. That is the standard a customer's architects will hold you to.

The engagement itself is the same for any platform; what changes is how much you can prove. The chapter follows one mock customer, Northstar Commerce, through seven artifacts in order: discovery notes, a qualification and sizing, a performance test protocol, a POC plan, a migration plan, a security questionnaire, and an operational handoff. Each section walks through a model answer, explains why each part is there, and names the alternatives it rejects. The numbers come from your own tools: capacity arithmetic from [Counting: parameters, FLOPs, KV bytes, and memory plans](../../mathematics/11-discrete-math-1/01-counting-params-flops-kv-bytes.md) in *Mathematics*, load reports from the [open-loop load generator](../../ai-engineering/23-model-serving/11-load-generator.md) in *AI Engineering*, and the threat model and SBOMs from the [STRIDE threat model](../../software-engineering/09-security/01-threat-model-stride.md) and [SBOM and supply chain](../../software-engineering/09-security/02-sbom-and-supply-chain.md) sections of *Software Engineering*. The chapter closes with a capstone that applies the whole sequence to a second customer, Lexa.

> **Learning objectives**
> - Produce each artifact of an engagement, from discovery notes to a handoff, with a check that someone else can apply.
> - Tie every claim in those artifacts to evidence, and mark estimates and open items with owners.
> - Answer a customer security questionnaire from your own threat model, SBOMs, and usage policy.
> - Hand a platform to a customer team using your runbooks, and test the handoff with a rehearsal.

## Before you begin

This chapter applies the seven chapters before it, from [Discovery](../01-discovery/README.md) through [Escalation and Handoff](../07-escalation-and-handoff/README.md); read those first. The model answers refer to a working platform at release `v1.0.0`, the system built in [the course path](../../ai-engineering/paths/course/) of *AI Engineering*. For the security questionnaire, Shostack's *Threat Modeling* is a useful companion.

## In this chapter

1. [Writing the discovery record](01-discovery.md): interview a platform team reporting slow inference, and write a record that separates symptoms from measured impact, names every stakeholder and data gate, and promises nothing.
2. [Qualifying and sizing the engagement](02-qualification-and-sizing.md): check that a decision can happen, then derive offered load and KV memory step by step, and turn the estimate into a bounded, gated test.
3. [Designing the performance test protocol](03-performance-engagement.md): write a like-for-like, open-loop protocol with repetitions, spread, stop conditions, and a regression procedure.
4. [Pre-registering the POC](04-poc-and-evaluation.md): freeze the dataset, rubric, and thresholds before the first run, stratify the set, and make safety gates blocking.
5. [Planning the staged migration](05-migration-plan.md): widen exposure from shadow traffic to default route, one gate at a time, with rollback decided in advance.
6. [Answering the security questionnaire](06-commercials-and-security.md): answer each question from a control or artifact, keep implemented apart from planned, and refuse to price without agreed assumptions.
7. [Handing off operations](07-escalation-and-handoff.md): define the service boundary and escalation sequence, and prove the handoff with a rehearsal.
8. [Capstone: one customer, end to end](08-capstone.md): produce eleven artifacts for Lexa, a legal-tech customer, each with a check you can apply yourself.

## Further reading

- [Security](../../software-engineering/09-security/README.md) in *Software Engineering*, for the threat model and SBOMs behind the questionnaire.
- [Incident Response and Chaos](../../systems/12-incident-response-and-chaos/README.md) in *Systems Engineering*, for the runbooks the handoff uses.
- Adam Shostack, *Threat Modeling: Designing for Security*.
