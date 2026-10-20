# Systems and Architecture

This book is about production systems: how to design, build, deploy, and operate software at scale. It is for engineers moving toward Staff+ roles, preparing for system design interviews, stepping into an architect role, or joining an SRE or platform team. Much of it works on the LLM serving platform and control plane built across the companion books, so the reviews and drills are about software you can run and break.

## What you will be able to do

- Reason about a design from its access patterns, scale, and consistency requirements.
- Choose an architecture pattern, name the anti-pattern it decays into, and defend the choice in a decision record.
- Run services on Kubernetes with Helm and GitOps.
- Instrument a system with traces, metrics, and logs, and turn them into SLOs and burn-rate alerts.
- Respond to an incident with a diagnosis method, a runbook, and a blameless postmortem.

## Ways through this book

Reading every chapter takes about fifteen weeks at eight to ten hours a week.

- **Preparing for a system design interview** (about five weeks): [Chapter 1: System Design](01-system-design/README.md); in its design reviews, [1.1](01-system-design/01-design-review-tracer-and-serving-platform.md) to [1.3](01-system-design/03-design-review-whole-system.md), check your estimates against theirs. Then [4.4 SLOs and multi-window burn-rate alerts](04-observability/04-slos-and-burn-rate-alerts.md).
- **Stepping into an architect role** (about eight weeks): [Chapter 2: Software Architecture](02-software-architecture/README.md), then [Chapter 1: System Design](01-system-design/README.md) for breadth, ending with [1.3 Design review: the whole system](01-system-design/03-design-review-whole-system.md).
- **New to production, or joining an SRE or platform team** (about nine weeks): [Chapter 3: Cloud Native](03-cloud-native/README.md) (ideally on a local kind cluster), then [Chapter 4: Observability](04-observability/README.md) from [4.1](04-observability/01-one-trace.md) to [4.6](04-observability/06-control-plane-observability.md), then the four worked incidents of [Chapter 5](05-incident-response-and-chaos/README.md), [5.1](05-incident-response-and-chaos/01-first-drill.md) to [5.4](05-incident-response-and-chaos/04-poison-task.md).
- **The whole book, in order** (about fifteen weeks): Chapters 1 to 5. Each drill in Chapter 5 breaks a component built elsewhere, so read it once you have built or read about that piece, and predict the diagnosis before reading the solution.

## Before you start

You should be comfortable with data structures and complexity analysis at the level of *[Algorithms](../algorithms/README.md)*. The design reviews and drills operate on a system built in other books: the engine of *[Build Your Own LLM System](../build-an-llm/README.md)*, the gateway, durable orchestration, and agent SDK of *[AI Platform Engineering](../ai-platform-engineering/README.md)*, and the images and Helm charts of *[Infrastructure](../infrastructure/README.md)*. The conceptual chapters stand alone. Runbooks and decision records are treated in depth in *[Diagramming and Documentation](../diagramming/README.md)*, and the migrations and upgrades behind several drills in *[Software Craftsmanship](../software-craftsmanship/README.md)*. Model runbooks, postmortems, and alert rules live under [`course/solutions/`](../../../course/solutions/). New to these books? [How to read these books](../../references/how-to-read.md) explains how sections, review questions, and solutions work.
