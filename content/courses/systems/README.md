# Systems and Architecture

This book is about production systems: how to design, build, deploy, and operate software at scale. It is written for engineers moving toward Staff+ roles, for whom the hard questions are no longer how to write a function but which tradeoffs a design makes, what happens when a server dies at 3 a.m., and how to know, from the outside, why a system is slow. It is equally useful if you are preparing for system design interviews, stepping into an architect role, or joining an SRE or platform team.

By the end you will be able to reason about a design from its access patterns, scale, and consistency requirements; choose and defend an architecture pattern; run services on Kubernetes with Helm and GitOps; instrument a system with traces, metrics, and logs, and turn them into SLOs and burn-rate alerts; and respond to incidents with a diagnosis method, a runbook, and a blameless postmortem. Much of the book works on a real system, the LLM serving platform and durable control plane built across the companion books, so the design reviews, dashboards, and drills are about software you can run and break.

## How the book is organized

The book has five chapters. The first two are about deciding what to build, the next two about running it, and the last about what happens when it fails.

[System Design](01-system-design/README.md) surveys the building blocks (scaling, load balancing, caching, databases and replication, consensus and distributed transactions, messaging, API design) and the classic design problems that exercise them. It then puts them to work in three design reviews of the system built across these books: the serving platform, the control plane, and the whole system. The primary references are ByteByteGo and the free [System Design Primer](https://github.com/donnemartin/system-design-primer).

[Software Architecture](02-software-architecture/README.md) covers the decisions that are expensive to change. It walks through layered, event-driven, microkernel, microservices, and space-based patterns, each with when to use it and the anti-pattern it decays into, then the habits that make a choice defensible: architecture decision records and quality-attribute analysis. The references are *Software Architecture Patterns* (O'Reilly) and the free [AOSA](https://aosabook.org/) books.

[Cloud Native](03-cloud-native/README.md) builds the vocabulary of designing for failure: containers and images, the Kubernetes objects that run them and the reconciliation model behind those objects, packaging and deployment with Helm, GitOps, and CI/CD, and the networking and resilience patterns that keep a distributed system standing. The references are *Cloud Native DevOps with Kubernetes* and the free [Kubernetes docs](https://kubernetes.io/docs/).

[Observability](04-observability/README.md) goes from one request to the whole fleet: a single hand-written trace across the gateway and engine, a shared tracing kit, metrics and logs that join to traces by id, SLOs and multi-window burn-rate alerts, dashboards kept as code, and finally the same rules applied to the durable control plane. The references are *Observability Engineering* and the free [Google SRE Book](https://sre.google/sre-book/).

[Incident Response and Chaos](05-incident-response-and-chaos/README.md) teaches incident response by causing incidents on purpose. Its thirteen drills break the system you built (a crashloop, a lost decode replica, a server killed in a loop, a poison task, migrations, regressions, bad data, noisy tenants, a runaway agent, a full disk, a Raft partition) and walk through detection, mitigation, the runbook, and the postmortem. The references are the free [SRE Book](https://sre.google/sre-book/table-of-contents/) and [SRE Workbook](https://sre.google/workbook/table-of-contents/).

You can read the chapters in order, but different goals suggest different entry points. For interview preparation, start with System Design: it is asked at every Staff+ interview. If you are new to production, read Cloud Native first, then Observability. For an architect role, read Software Architecture, then System Design for breadth. For SRE or platform work, read Observability, Cloud Native, and System Design, then work the Incident Response and Chaos drills against your own system. The [study plan](study-plan.md) suggests a paced reading schedule.

## Prerequisites and related books

You should be comfortable with data structures and complexity analysis at the level of *[Algorithms](../algorithms/README.md)*. The design reviews and drills operate on a system built in other books: the engine of *[Build Your Own LLM System](../build-an-llm/README.md)*, the gateway, durable orchestration, and agent SDK of *[AI Platform Engineering](../ai-platform-engineering/README.md)*, and the images and Helm charts of *[Infrastructure](../infrastructure/README.md)*. Runbooks and decision records are treated in depth in *[Diagramming and Documentation](../diagramming/README.md)*, and the interface migrations and upgrades behind several drills in *[Software Craftsmanship](../software-craftsmanship/README.md)*. You can read the conceptual chapters without those books; the drills and design reviews are most rewarding if you have built, or at least read, the pieces they exercise.

## Conventions

Code listings are fenced with their language: shell commands as `bash`, Prometheus queries as `promql`, Kubernetes and alert rules as `yaml`. Placeholders appear in angle brackets, such as `<system>` for the namespace and release prefix, and in braces for the system's command-line tools, such as `{ctl}` and `{corpus}`. Mathematics is set inline as $\rho = \lambda / (n\mu)$ or in display blocks, and every symbol is defined in a table before it is used.

Call-outs mark material to treat with care: **Learning objectives** at the start of each section say what you should be able to do, and **Warning** call-outs flag the actions that turn an incident into data loss. Each section ends with key terms, a summary, and review questions whose answers are folded under an "Answer" disclosure, so you can test yourself before looking.

The full solution code, including model runbooks, postmortems, and alert rules, lives in the repository under [`course/solutions/`](../../../course/solutions/), with the shared interface contracts under [`course/contracts/`](../../../course/contracts/). Sections link to the specific files they discuss.
