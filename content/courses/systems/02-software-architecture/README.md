# Software Architecture

## Overview

- **Primary reference**: *Software Architecture Patterns* by Mark Richards (O'Reilly, recommended)
- **Supplementary**: [The Architecture of Open Source Applications](https://aosabook.org/) (free), Martin Fowler's [blog](https://martinfowler.com/) (free)
- **Prerequisites**: Professional development experience
- **Estimated time**: 2-3 weeks at 8-10 hrs/week

## Key Takeaways

- Architecture patterns are tools, not religions -- each has specific strengths and failure modes
- The most important skill is choosing the right pattern for the problem, not applying the fanciest one
- Document decisions with ADRs so future engineers understand why, not just what
- Quality attributes (performance, scalability, availability) drive architecture choices, not features

---

# Concepts & Techniques

## Core Insight

Software architecture is the set of decisions that are expensive to change. Good architects minimize the number of irreversible decisions and make the reversible ones easy to change. Every pattern trades one quality attribute for another.

## 1. Layered Architecture

**Key ideas**: Presentation → Business → Persistence → Database. Each layer only calls the layer directly below it. Closed layers enforce separation; open layers allow bypassing.

**When to use**: CRUD applications, small-to-medium teams, when time-to-market matters more than scalability.

**When NOT to use**: High-performance systems (too many layers of indirection), microservices (monolith coupling).

**Anti-pattern**: "Sinkhole" -- requests pass through layers that add no value.

## 2. Event-Driven Architecture

**Key ideas**: **Mediator topology** -- central mediator orchestrates event processing. **Broker topology** -- no central mediator; events flow through channels. Asynchronous, highly decoupled.

**When to use**: Complex workflows, systems that need to react to events in real-time, when components evolve independently.

**When NOT to use**: Simple CRUD, when you need synchronous request-response, when event ordering is critical and hard to guarantee.

## 3. Microkernel Architecture

**Key ideas**: Core system (minimal, stable) + plug-in modules (features, extensions). Registry tracks available plug-ins. Core provides services; plug-ins extend behavior.

**When to use**: Product-based applications (IDEs, browsers), systems with optional features, when third-party extensibility matters.

**When NOT to use**: Pure microservices (different problem), when plug-ins need to communicate with each other (use event-driven instead).

## 4. Microservices Architecture

**Key ideas**: Independently deployable services, each owning its data. Bounded contexts from DDD define service boundaries. API gateway for external access. Service discovery (DNS, Consul). Orchestration (explicit workflow) vs choreography (event-driven).

**When to use**: Large teams, independent deployment cadences, polyglot tech stacks, when services have genuinely different scaling needs.

**When NOT to use**: Small teams (< 10 engineers), early-stage products (you don't know your domain boundaries yet), when a modular monolith would suffice. **The distributed monolith anti-pattern** -- microservices that must be deployed together are a monolith with network calls.

## 5. Space-Based Architecture

**Key ideas**: Processing units contain business logic + in-memory data grid. Virtualized middleware handles messaging, data replication, request routing. No central database under load; data is replicated across processing units.

**When to use**: Variable, spiky load (ticket sales, flash sales); when horizontal scaling needs to be nearly instantaneous.

**When NOT to use**: Systems that need strong consistency, small-scale applications (overkill).

## 6. Architecture Decision Records (ADRs)

**Format**:
```
# ADR-NNN: [Title]
Status: [Proposed | Accepted | Deprecated | Superseded by ADR-XXX]
Context: [What is the problem?]
Decision: [What did we decide?]
Consequences: [What are the trade-offs?]
```

**Why**: decisions are forgotten, context is lost, new engineers don't know why things are the way they are. ADRs are a lightweight decision log.

## 7. Quality Attributes & Trade-off Analysis

**Key attributes**: Performance (latency, throughput), Scalability (handles growth), Availability (uptime), Fault tolerance (handles failure), Security, Maintainability, Testability.

**Trade-off analysis**: every architecture decision trades one attribute for another. Microservices improve deployability but hurt consistency. Caching improves performance but hurts consistency. The architect's job is to make these trade-offs explicit and aligned with business priorities.

---

## Company Relevance

| Company | Architecture Style | Why |
|---------|-------------------|-----|
| Google | Monorepo + large services | Scale, consistency, shared infrastructure |
| Amazon | Microservices (pioneered it) | Two-pizza teams, independent deployment |
| Netflix | Microservices + event-driven | Resilience, independent scaling |
| Stripe | Modular monolith → services | Deliberate extraction as boundaries clarify |
| Anthropic | Service-oriented, GPU-aware | Inference serving has unique constraints |
