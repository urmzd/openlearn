# Software Architecture

Software architecture is the set of decisions that are expensive to change. Good architects minimize the number of irreversible decisions and make the reversible ones easy to change. Every architecture pattern trades one quality attribute for another, so patterns are tools, not religions: each has specific strengths and specific failure modes, and the important skill is choosing the right pattern for the problem rather than applying the fanciest one.

This chapter walks through five classic patterns, each with when to use it, when not to, and the anti-pattern it tends to decay into: layered, event-driven, microkernel, microservices, and space-based. It then covers the two habits that make pattern choice defensible: recording decisions in architecture decision records so future engineers understand why and not just what, and analysing quality attributes, because performance, scalability, and availability drive architecture choices far more than features do. The chapter is self-contained; the [System Design](../01-system-design/README.md) chapter puts these ideas to work in full design reviews.

> **Learning objectives**
> - Describe the layered, event-driven, microkernel, microservices, and space-based patterns and the problem each solves.
> - State when each pattern should not be used and recognize its characteristic anti-pattern.
> - Write an architecture decision record with status, context, decision, and consequences.
> - Analyse a decision as a trade between quality attributes and align it with business priorities.

## Before you begin

You need professional development experience: the patterns make most sense once you have maintained a codebase that outgrew its first structure. No other chapter is required.

## Layered architecture

The layered pattern stacks presentation, business logic, persistence, and the database, and each layer calls only the layer directly below it. Closed layers enforce that separation strictly; open layers allow a request to bypass them. It suits CRUD applications and small-to-medium teams, and it is a sound choice when time-to-market matters more than scalability. It is a poor fit for high-performance systems, where the layers of indirection cost too much, and for microservices, where it reintroduces monolith coupling. Its characteristic anti-pattern is the **sinkhole**: requests that pass through layers adding no value at all.

## Event-driven architecture

Event-driven systems are asynchronous and highly decoupled, and they come in two topologies. In the **mediator topology**, a central mediator orchestrates event processing. In the **broker topology**, there is no central mediator, and events flow through channels from component to component. The pattern suits complex workflows, systems that must react to events in real time, and systems whose components evolve independently. Avoid it for simple CRUD, when you need synchronous request-response, and when event ordering is critical and hard to guarantee.

## Microkernel architecture

A microkernel system has a minimal, stable core and plug-in modules that supply features and extensions. A registry tracks the available plug-ins; the core provides services and the plug-ins extend behavior. This is the shape of product-based applications such as IDEs and browsers, of systems with optional features, and of anything where third-party extensibility matters. It is the wrong tool for a pure microservices problem, which is different in kind, and for plug-ins that need to communicate with each other, where an event-driven design fits better.

## Microservices architecture

Microservices are independently deployable services, each owning its data. Bounded contexts from domain-driven design define the service boundaries. An API gateway provides external access, service discovery (DNS, Consul) lets services find each other, and workflows across services are coordinated either by orchestration, an explicit workflow, or by choreography, where services react to each other's events.

The pattern pays off for large teams, independent deployment cadences, polyglot stacks, and services with genuinely different scaling needs. It does not pay off for small teams (fewer than about 10 engineers), for early-stage products whose domain boundaries are not yet known, or when a modular monolith would suffice.

> **Warning:** Beware the **distributed monolith**. Microservices that must be deployed together are a monolith with network calls: all of the coupling, plus all of the latency and partial failure.

## Space-based architecture

Space-based systems are built from processing units, each holding business logic and an in-memory data grid. Virtualized middleware handles messaging, data replication, and request routing. There is no central database in the request path under load; data is replicated across the processing units instead. The pattern suits variable, spiky load such as ticket sales and flash sales, where horizontal scaling must be nearly instantaneous. It is a poor fit for systems that need strong consistency, and overkill for small-scale applications.

## Architecture decision records

Decisions are forgotten, context is lost, and new engineers do not know why things are the way they are. An **architecture decision record** (ADR) is a lightweight decision log that fixes this, one short document per decision:

```text
# ADR-NNN: [Title]
Status: [Proposed | Accepted | Deprecated | Superseded by ADR-XXX]
Context: [What is the problem?]
Decision: [What did we decide?]
Consequences: [What are the trade-offs?]
```

The consequences line is the one that matters most later, because it records what the decision gave up. [Architecture decision records](../../software-engineering/12-documentation/01-architecture-decision-records.md) in *Software Engineering* treats the practice in depth.

## Quality attributes and tradeoff analysis

The attributes that drive architecture are performance (latency and throughput), scalability (handling growth), availability (uptime), fault tolerance (handling failure), security, maintainability, and testability. Every architecture decision trades one of them for another. Microservices improve deployability but hurt consistency. Caching improves performance but also hurts consistency. The architect's job is to make these tradeoffs explicit and align them with business priorities, which is exactly what the "what the choice gives up" line of a design review asks for.

How large companies have resolved these trades is instructive, because each choice follows from a constraint. The following table summarizes a few.

| Company | Architecture style | Why |
|---|---|---|
| Google | Monorepo + large services | Scale, consistency, shared infrastructure |
| Amazon | Microservices (pioneered it) | Two-pizza teams, independent deployment |
| Netflix | Microservices + event-driven | Resilience, independent scaling |
| Stripe | Modular monolith, then services | Deliberate extraction as boundaries clarify |
| Anthropic | Service-oriented, GPU-aware | Inference serving has unique constraints |

## Summary

- Architecture is the set of decisions that are expensive to change; minimize the irreversible ones.
- Each pattern (layered, event-driven, microkernel, microservices, space-based) has a problem it fits and a failure mode it decays into.
- Microservices that must be deployed together are a distributed monolith.
- ADRs record why a decision was made, including what it gave up.
- Quality attributes, not features, drive architecture; every decision trades one attribute for another.

## Key terms

- **Layered architecture**: a stack of presentation, business logic, persistence, and database layers, each calling only the layer below; it suits CRUD applications and small teams.
- **Sinkhole anti-pattern**: requests passing through architectural layers that add no value, only indirection; the characteristic decay of a layered architecture.
- **Event-driven architecture**: components that communicate asynchronously through events, highly decoupled; it suits complex, real-time, independently evolving systems but not simple CRUD or strictly ordered work.
- **Mediator topology / broker topology**: the two shapes of event-driven architecture: a central mediator orchestrates event processing, or events flow through channels from component to component with no central coordinator.
- **Microkernel architecture**: a minimal, stable core extended by plug-in modules tracked in a registry; the shape of IDEs, browsers, and other products built for extensibility.
- **Microservices architecture**: independently deployable services, each owning its data, with boundaries drawn from bounded contexts; it pays off for large teams and different scaling needs, not for small teams or unclear domains.
- **Bounded context**: a domain-driven design boundary within which a model and its terms have one consistent meaning; it is the natural line for a service boundary.
- **Distributed monolith**: microservices that must be deployed together, carrying all the coupling of a monolith plus the latency and partial failure of network calls.
- **Modular monolith**: one deployable application with strong internal module boundaries, often a better starting point than microservices while domain boundaries are still being discovered.
- **Space-based architecture**: processing units that each hold business logic and an in-memory data grid, replicated by middleware, with no central database in the request path; it suits spiky load such as ticket sales.
- **Quality attribute**: a non-functional property such as performance, scalability, availability, fault tolerance, security, maintainability, or testability; quality attributes, not features, drive architecture, and every decision trades one for another.

## Further reading

- Mark Richards, *Software Architecture Patterns* (O'Reilly).
- [The Architecture of Open Source Applications](https://aosabook.org/) (free).
- Martin Fowler's [blog](https://martinfowler.com/) (free).
