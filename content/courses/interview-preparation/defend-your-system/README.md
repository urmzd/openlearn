# Defend Your System

System design interviews test judgment under questioning. Most candidates prepare for them by rehearsing designs they have never built, which means that the first hard follow-up ("what breaks first at ten times the load?") pushes them from recall into improvisation. This chapter takes a different route. If you have built and operated a system end to end, the trade-offs in it are ones you made, measured, and lived with through drills and migrations, and defending that system turns the interview from rehearsal into recall.

The strongest interview answer is about a system you built: you know where it breaks, why each choice was made, and what you would change. The second principle is that you answer with evidence, such as a load report, a benchmark, a drill postmortem, or an architecture decision record (ADR) with the rejected options written down. The third is that every choice has a cost, so a good answer names the alternative you rejected and what it would have bought you.

The chapter has a single section. It defines what a defense is, fixes a rubric that every defense answers, and then walks through a model defense of the reference system line by line, with the evidence each answer cites and why that evidence is the right kind.

> **Learning objectives**
> - Answer a design question by tying each claim to a contract file, a config value, or a measured number.
> - Structure a defense in rubric order, from problem and constraints through rollout and open questions.
> - Recognise when an answer is only partly supported and turn the gap into a review note with an owner and a next step.
> - Explain the trade-offs behind the system's major boundaries well enough to survive follow-up questions.

## Before you begin

This chapter assumes you have a system to defend: either the whole reference system built through the other books, or a system of your own that you designed and operate. The defense draws on your own design documents, especially your [Architecture decision records](../../diagramming/02-documentation-writing/01-architecture-decision-records.md) and [C4 diagrams in D2](../../diagramming/01-diagramming-c4/01-c4-diagrams-in-d2.md) from *Diagramming and Documentation*, your [Threat model (STRIDE on your DFD)](../../software-craftsmanship/08-security/01-threat-model-stride.md) from *Software Craftsmanship*, and the three design reviews in *Systems and Architecture*, starting with [Design review: tracer and serving platform](../../systems/01-system-design/01-design-review-tracer-and-serving-platform.md). For question styles at particular companies, see the company chapters in this book and the [shared concepts](../shared-concepts/README.md) chapter.

A good way to use the chapter is to answer the question bank below out loud against your own design, then write each answer down alongside the artifact that backs it. The questions fall into three groups. On architecture: why a C ABI and HTTP between the languages, why the gateway owns keys and limits, and why a durable engine instead of cron. On scale and failure: what breaks first at 10x load, what happens when a decode pod dies mid-stream, and how resume stays exactly-once. On models and data: why this tokenizer and vocabulary size, and how you know the eval numbers are not contaminated.

## In this chapter

- [Defend your system](01-defend-your-system.md): what a technical defense is, the eight-line design-review rubric, and a model defense of the reference system with the evidence behind every answer.

## Further reading

- Martin Kleppmann, *Designing Data-Intensive Applications*, for the vocabulary of replication, consistency, and failure that most defenses lean on.
- [Design review: the whole system](../../systems/01-system-design/03-design-review-whole-system.md) in *Systems and Architecture*, the review this defense is checked against.
